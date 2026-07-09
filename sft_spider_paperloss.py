"""Paper-faithful loss variant of the Spider SFT baseline (A/B test harness).

This reuses the ENTIRE pipeline from ``sft_spider_baseline.py`` (data loading,
tokenizer/model setup, training args, generation, execution-match eval) and
swaps out exactly ONE thing: how the training labels are built.

It replicates the reference repo's ``create_masked_labels``
(galilai-group/llm-jepa, finetune.py) so you can confirm that the loss masking
is what explains the gap between your ~63% baseline and the paper's 47.52%.

Two behaviours differ from your robust masking, both reproduced here:

  1. The trailing stop token (``<|eot_id|>`` / EOS) is NOT included in the loss.
     Only the raw assistant-content tokens are unmasked. The model is never
     trained to *stop*, so at eval it tends to over-generate past the SQL.

  2. The assistant span is located by re-encoding the assistant content on its
     own and matching per-token *decoded strings* against the full sequence.
     When that match fails (common: leading-space / merge differences between
     standalone and in-context tokenization), the whole example stays -100 and
     contributes ZERO training signal. The reference keeps such rows (does not
     drop them), so we do the same and just report how many there are.

Everything else -- lr (4e-5), epochs (4), eval, data, seeds -- is inherited
unchanged from sft_spider_baseline.py, so any accuracy difference isolates the
masking variable.

Run exactly like the baseline, but point it at a DISTINCT output dir so you keep
both checkpoints/metrics:

    torchrun --nproc_per_node=4 sft_spider_paperloss.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --seed 82 --learning_rate 4e-5 \
        --output_dir sft_spider_PAPERLOSS_seed82_lr4e5
"""

from typing import Any, Dict, List, Sequence

from datasets import load_dataset

import sft_spider_baseline as base


def create_masked_labels_paper(
    messages: Sequence[Dict[str, str]],
    tokenizer: Any,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
) -> List[int]:
    """Verbatim reproduction of the reference repo's create_masked_labels.

    Unmasks ONLY the exact assistant-content tokens (no EOS), located via a
    per-token decoded-string match. On a failed match the row stays all -100.
    """
    labels = [-100] * len(input_ids)

    # Mask padding tokens (no-op here since we pad dynamically, kept for parity).
    for i, mask in enumerate(attention_mask):
        if mask == 0:
            labels[i] = -100

    for msg in messages:
        if msg["role"] != "assistant":
            continue

        assistant_content = msg["content"]
        assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
        if not assistant_tokens:
            continue

        # NOTE: standalone-encode + per-token decode is the brittle part. The
        # same SQL string can tokenize differently in-context than on its own,
        # so this span search can silently fail to find a match.
        decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
        decoded_input = [tokenizer.decode(item) for item in input_ids]

        span = len(assistant_tokens)
        for i in range(len(input_ids) - span + 1):
            if attention_mask[i] == 1 and decoded_input[i:i + span] == decoded_assistant:
                # Unmask ONLY the content tokens -- crucially NOT the trailing
                # <|eot_id|>/EOS that follows the assistant turn.
                for j in range(i, min(i + span, len(input_ids))):
                    if attention_mask[j] == 1:
                        labels[j] = input_ids[j]
                break  # first match only, like the reference

    return labels


def tokenize_sft_example_paper(
    example: Dict[str, Any],
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> Dict[str, List[int]]:
    """Same full-text tokenization as the baseline, paper-faithful labels."""
    messages = base.example_to_messages(example)
    # full_text == reference's `formatted_chat`
    # (apply_chat_template(get_messages(...), add_generation_prompt=False)).
    full_text, _prompt_text = base.format_full_and_prompt(tokenizer, model_name, messages)

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = create_masked_labels_paper(messages, tokenizer, input_ids, attention_mask)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_train_dataset_paper(
    train_file: str,
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> Any:
    """Like the baseline's prepare_train_dataset, but does NOT drop rows whose
    labels are entirely -100 -- the reference keeps them. We count and report
    them instead, so the brittleness (fragility #2) is visible in the logs.
    """
    dataset = load_dataset("json", data_files=train_file, split="train")
    base.log(f"Loaded {len(dataset)} training examples from {train_file}.")

    tokenized = dataset.map(
        lambda example: tokenize_sft_example_paper(example, tokenizer, model_name, max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT data (paper loss)",
    )

    n_empty = sum(
        1 for ex in tokenized if not any(label != -100 for label in ex["labels"])
    )
    base.log(
        f"[paper-loss] {n_empty}/{len(tokenized)} examples have NO unmasked "
        f"label (span match failed -> zero training signal). Reference keeps "
        f"these rows; not dropping."
    )
    return tokenized


# --- Swap in the paper-faithful masking, then run the unchanged pipeline. ---
# main() looks these names up in base's module globals at call time, so
# reassigning them here is enough to override behaviour.
base.tokenize_sft_example = tokenize_sft_example_paper
base.prepare_train_dataset = prepare_train_dataset_paper


if __name__ == "__main__":
    base.main()
