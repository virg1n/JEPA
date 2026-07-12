"""LLM-JEPA training on NL-RX-SYNTH (paper reproduction).

This is the JEPA counterpart of ``sft_spider_paperloss.py``. It reuses:

  * the JEPA loss mechanism you wrote in ``main.py`` (three views per example:
    a Text view with predictor tokens, a Code view, and a Generation view;
    ``jepa_loss = 1 - cos(p, t)``, ``total = gamma*lm_loss + lbd*jepa_loss``,
    NO stop-gradient on the target -- exactly like the reference repo);

  * the chat formatting helpers from ``sft_spider_baseline.py``. NL-RX-SYNTH
    already contains the three-message ``messages`` records consumed by those
    helpers;

  * the paper-faithful LM masking from ``sft_spider_paperloss.py``
    (``create_masked_labels_paper``: unmasks assistant-content tokens only, NOT
    the trailing EOS), so the LM half of the loss matches the paper exactly.

The three views (following main.py / the reference finetune.py):

  Text view  : the user message with <|predictor_K|>...<|predictor_1|>
               appended inside its content, then rendered by the chat template.
               Predicted embedding = the final predictor token (offset=-2 for
               Llama, immediately before <|eot_id|>).
  Code view  : the assistant message rendered by the chat template.
               Target embedding = the final content token (offset=-2 for
               Llama, immediately before <|eot_id|>).
  Gen view   : full chat (question + answer). Standard next-token LM loss with
               paper-faithful masking.

Predictor tokens are a training-time-only auxiliary: inference/eval is plain
greedy generation, so the eval path is identical to the SFT baseline.

Run (4 GPUs, NL-RX-SYNTH paper hyper-params, seed 82):

    torchrun --standalone --nproc_per_node=4 jepa_spider_paperloss.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --seed 82 --learning_rate 2e-5 --num_epochs 4 \
        --per_device_batch_size 4 --gradient_accumulation_steps 8 \
        --pred_k 1 --lbd 1.0 \
        --output_dir runs/nlrx-jepa-seed82-lambda1-k1

Sweep --lbd (the JEPA weight lambda) and --pred_k as in the paper. With lbd=0
the optimized objective is the paper-faithful NTP baseline, although this
script intentionally still executes the two auxiliary JEPA forwards.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

import sft_spider_baseline as base
# Same masking the paper uses (no EOS in the loss). Importing this module also
# monkeypatches `base`, but we never call base.main(), so that is a no-op for us.
from sft_spider_paperloss import create_masked_labels_paper


# --------------------------------------------------------------------------- #
# Data: build the three JEPA views for one example.
# --------------------------------------------------------------------------- #
def build_views(example, tokenizer, model_name, max_length, pred_ids):
    messages = base.example_to_messages(example)
    full_text, _prompt_text = base.format_full_and_prompt(tokenizer, model_name, messages)

    # Generation view -> LM loss (paper-faithful masking, EOS stays masked).
    full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=True)
    gen_ids = full["input_ids"]
    gen_mask = full["attention_mask"]
    labels = create_masked_labels_paper(messages, tokenizer, gen_ids, gen_mask)

    # Text view -> JEPA predicted embedding. This mirrors the reference repo's
    # get_user_messages(messages) and descending predictor-token append loop.
    user_messages = [dict(msg) for msg in messages if msg["role"] == "user"][:1]
    if not user_messages:
        raise ValueError("Example has no user message.")
    user_messages[0]["content"] += "".join(
        f"<|predictor_{i}|>" for i in range(len(pred_ids), 0, -1)
    )
    text_text = tokenizer.apply_chat_template(
        base.adapt_messages_for_model(model_name, user_messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    text = tokenizer(
        text_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    text_ids = text["input_ids"]

    # Code view -> JEPA target embedding. The reference renders the assistant
    # message by itself, including its role header and trailing <|eot_id|>.
    assistant_messages = [dict(msg) for msg in messages if msg["role"] == "assistant"][:1]
    if not assistant_messages:
        raise ValueError("Example has no assistant message.")
    code_text = tokenizer.apply_chat_template(
        base.adapt_messages_for_model(model_name, assistant_messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    code_ids = tokenizer(
        code_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )["input_ids"]

    return {"gen_ids": gen_ids, "labels": labels, "text_ids": text_ids, "code_ids": code_ids}


def pad_with_mask(batch, pad_value):
    width = max(len(b) for b in batch)
    ids = torch.full((len(batch), width), pad_value, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b)] = torch.tensor(b, dtype=torch.long)
        mask[i, : len(b)] = 1
    return ids, mask


def pad_labels(batch, width):
    out = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        out[i, : len(b)] = torch.tensor(b, dtype=torch.long)
    return out


def make_collate(pad_id):
    def collate(samples):
        gen_ids, gen_mask = pad_with_mask([s["gen_ids"] for s in samples], pad_id)
        labels = pad_labels([s["labels"] for s in samples], gen_ids.size(1))
        text_ids, text_mask = pad_with_mask([s["text_ids"] for s in samples], pad_id)
        code_ids, code_mask = pad_with_mask([s["code_ids"] for s in samples], pad_id)
        return {
            "gen_ids": gen_ids, "gen_mask": gen_mask, "labels": labels,
            "text_ids": text_ids, "text_mask": text_mask,
            "code_ids": code_ids, "code_mask": code_mask,
        }

    return collate


def get_last_token(hidden, mask, offset=-1):
    idx = mask.sum(1) + offset
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def unwrap_model(model):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


# --------------------------------------------------------------------------- #
# NL-RX exact-match evaluation. For paper parity this is called only on rank 0,
# just like run.sh launches evaluate.py on one GPU after distributed training.
# --------------------------------------------------------------------------- #
def evaluate_nlrx_distributed(model, tokenizer, args, rank, world_size, device):
    dataset = load_dataset("json", data_files=args.eval_file, split="train")
    if args.max_eval_examples is not None:
        dataset = dataset.select(range(min(args.max_eval_examples, len(dataset))))

    model.eval()

    # rank i handles indices i, i+world_size, i+2*world_size, ...
    indices = list(range(rank, len(dataset), world_size))
    iterator = base.tqdm(indices, desc=f"Evaluating NL-RX-SYNTH (x{world_size} GPUs)") if rank == 0 else indices

    local_records = []
    for idx in iterator:
        example = dataset[idx]
        messages = base.example_to_messages(example)
        # Reference evaluate.py compares directly with messages[2]["content"];
        # unlike generated text, the target is not stripped or normalized.
        gold = base.assistant_content(messages)
        formatted = base.adapt_messages_for_model(args.model_name, messages)
        prompt_text = tokenizer.apply_chat_template(
            base.prompt_messages(formatted), tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=args.max_length, add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )[0]
        generated = base.generated_sql_from_output(
            tokenizer, inputs["input_ids"].shape[1], output_ids
        )
        is_correct = generated == gold
        local_records.append({
            "idx": idx,
            "correct": is_correct,
            "generated": generated,
            "gold": gold,
        })

    # Gather every rank's records onto rank 0.
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]

    if rank != 0:
        return None

    all_records = [r for part in gathered for r in part]
    all_records.sort(key=lambda r: r["idx"])

    correct = sum(1 for r in all_records if r["correct"])
    total = len(all_records)
    accuracy = correct / total if total else 0.0

    predictions_file = os.path.join(args.output_dir, "nlrx_eval_predictions.jsonl")
    metrics_file = os.path.join(args.output_dir, "nlrx_eval_metrics.json")
    Path(predictions_file).parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = {
        "exact_match_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "eval_file": args.eval_file,
    }
    Path(metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    base.log(f"Success Rate: {args.output_dir}, {accuracy:.4f}")
    base.log(f"Wrote predictions to {predictions_file}")
    base.log(f"Wrote metrics to {metrics_file}")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Train LLM-JEPA on NL-RX-SYNTH (paper-faithful loss).")
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train_file", default="datasets/synth_train.jsonl")
    p.add_argument("--eval_file", default="datasets/synth_test.jsonl")
    p.add_argument("--output_dir", default=None)
    # JEPA knobs
    p.add_argument("--pred_k", type=int, default=1, help="Number of predictor tokens (K).")
    p.add_argument("--lbd", type=float, default=1.0, help="JEPA loss weight (lambda). 0 == pure SFT.")
    p.add_argument("--gamma", type=float, default=1.0, help="LM loss weight.")
    # Optimisation: matches the validated NL-RX lambda=0 SFT invocation.
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--no_bf16", action="store_true")
    # Greedy exact-match evaluation, matching evaluate.py for NL-RX-SYNTH.
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_eval_examples", type=int, default=None)
    return p


def main():
    args = build_arg_parser().parse_args()
    base.seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = (
            f"jepa_nlrx_{base.safe_name(args.model_name)}_seed{args.seed}"
            f"_k{args.pred_k}_lbd{args.lbd}"
        )

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Evaluation file not found: {args.eval_file}")

    # --- distributed setup (same pattern as main.py) ---
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = (not distributed) or dist.get_rank() == 0

    bf16 = torch.cuda.is_available() and not args.no_bf16

    # --- model / tokenizer (reuse baseline helpers) ---
    tokenizer = base.load_tokenizer(args.model_name, trust_remote_code=False)
    model = base.load_model(args.model_name, trust_remote_code=False, bf16=bf16)
    base.add_upstream_special_tokens(tokenizer, model)  # adds <|predictor_1..10|> etc.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    assert 1 <= args.pred_k <= 10, "pred_k must be in [1, 10] (predictor tokens are <|predictor_1..10|>)."
    pred_ids = [tokenizer.convert_tokens_to_ids(f"<|predictor_{i}|>") for i in range(1, args.pred_k + 1)]
    assert all(pid is not None and pid >= 0 for pid in pred_ids), "Missing predictor tokens in vocab."

    # --- data: build the three views ---
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    base.log(f"Loaded {len(dataset)} training examples from {args.train_file}.")
    dataset = dataset.map(
        lambda ex: build_views(ex, tokenizer, args.model_name, args.max_length, pred_ids),
        remove_columns=dataset.column_names,
        desc="Building JEPA views",
    )
    n_empty = sum(1 for ex in dataset if not any(l != -100 for l in ex["labels"]))
    base.log(f"[jepa] {n_empty}/{len(dataset)} examples have NO unmasked LM label (paper masking).")

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )

    # --- training setup ---
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        # Non-reentrant checkpointing is required here: we run 3 forwards per
        # step through the same DDP model, and the reentrant variant marks DDP
        # params "ready" multiple times ("marked as ready twice" error).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()

    if distributed:
        # static_graph=True lets DDP handle several forwards feeding ONE backward
        # (plus activation checkpointing) without erroring; the graph is fixed
        # across iterations here, so it is safe.
        model = DDP(model, device_ids=[local_rank], static_graph=True)

    accum = args.gradient_accumulation_steps
    num_batches = len(loader)
    remainder = num_batches % accum
    total_steps = math.ceil(num_batches / accum) * args.num_epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    base.log(
        f"Starting LLM-JEPA: model={args.model_name}, seed={args.seed}, "
        f"epochs={args.num_epochs}, lr={args.learning_rate}, K={args.pred_k}, "
        f"lambda={args.lbd}, gamma={args.gamma}, batch/GPU={args.per_device_batch_size}, "
        f"grad_accum={accum}"
    )

    for epoch in range(args.num_epochs):
        if distributed:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            is_last = (step + 1) == num_batches
            should_step = ((step + 1) % accum == 0) or is_last

            # NOTE: we intentionally do NOT use model.no_sync() for gradient
            # accumulation. static_graph=True needs DDP's autograd hooks active
            # on the first backward, and no_sync() disables them (the crash was
            # reducer.cpp: expect_autograd_hooks_). Syncing every micro-step is
            # exactly equivalent numerically (all-reduce is linear) -- only a bit
            # more communication.

            # JEPA: predicted embedding (Text view) vs target (Code view).
            p = get_last_token(
                model(input_ids=batch["text_ids"], attention_mask=batch["text_mask"],
                      output_hidden_states=True).hidden_states[-1],
                mask=batch["text_mask"],
                offset=-2,
            )  # last predictor token before <|eot_id|> (Llama last_token=-2)
            t = get_last_token(
                model(input_ids=batch["code_ids"], attention_mask=batch["code_mask"],
                      output_hidden_states=True).hidden_states[-1],
                mask=batch["code_mask"],
                offset=-2,
            )  # last target token before <|eot_id|> (no stop-gradient)
            jepa_loss = (1 - F.cosine_similarity(p, t, dim=-1)).mean()

            # LM loss on the Generation view (paper-faithful masking).
            out = model(input_ids=batch["gen_ids"], attention_mask=batch["gen_mask"])
            logits = out.logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            lm_loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))

            if master_process and step % args.logging_steps == 0:
                print(f"epoch {epoch} step {step}: lm={lm_loss.item():.4f} jepa={jepa_loss.item():.4f}",
                      flush=True)

            divisor = remainder if (remainder != 0 and step >= num_batches - remainder) else accum
            loss = (args.gamma * lm_loss + args.lbd * jepa_loss) / divisor
            loss.backward()

            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

    # --- save (rank 0) ---
    save_model = unwrap_model(model)
    if master_process:
        save_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        base.log(f"Saved model to {args.output_dir}")

    # End DDP before evaluation. Non-master workers exit; rank 0 performs the
    # same single-GPU greedy evaluation used by the reference run.sh.
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if not master_process:
        return

    if not args.no_gradient_checkpointing:
        save_model.gradient_checkpointing_disable()
    base.log("Evaluating NL-RX-SYNTH on rank 0 (paper-compatible single-GPU eval).")
    evaluate_nlrx_distributed(save_model, tokenizer, args, rank=0, world_size=1, device=device)


if __name__ == "__main__":
    main()
