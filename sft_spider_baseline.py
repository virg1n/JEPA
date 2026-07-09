"""SFT baseline runner for LLM-JEPA Spider experiments.

Train with:
    torchrun --nproc_per_node=4 sft_spider_baseline.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --seed 82

The defaults match the upstream baseline layout:
    spider_train.jsonl
    spider_test.jsonl
    spider_data/database
"""

import argparse
import json
import os
import random
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(items: Iterable[Any], **_: Any) -> Iterable[Any]:
        return items


UPSTREAM_SPECIAL_TOKENS = [
    "<|predictor_1|>",
    "<|predictor_2|>",
    "<|predictor_3|>",
    "<|predictor_4|>",
    "<|predictor_5|>",
    "<|predictor_6|>",
    "<|predictor_7|>",
    "<|predictor_8|>",
    "<|predictor_9|>",
    "<|predictor_10|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|perception|>",
]

SPIDER_DB_RE = re.compile(r"For db_id:\[(.+?)\]")


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def ensure_chat_template(tokenizer: Any) -> None:
    if tokenizer.chat_template:
        return

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\\n' + message['content'] + eos_token + '\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\\n' + message['content'] + eos_token + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    )


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            trust_remote_code=trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ensure_chat_template(tokenizer)
    return tokenizer


def add_upstream_special_tokens(tokenizer: Any, model: Any) -> None:
    vocab = tokenizer.get_vocab()
    new_tokens = [token for token in UPSTREAM_SPECIAL_TOKENS if token not in vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        log(f"Added {len(new_tokens)} upstream special tokens.")


def load_model(model_name: str, trust_remote_code: bool, bf16: bool) -> Any:
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "use_cache": False,
    }
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model


def coerce_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)

    messages = []
    for msg in raw_messages:
        messages.append(
            {
                "role": str(msg["role"]),
                "content": str(msg["content"]),
            }
        )
    return messages


def example_to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    if example.get("messages") is not None:
        return coerce_messages(example["messages"])

    db_id = str(example.get("db_id", ""))
    question = str(example.get("question", example.get("text", "")))
    query = str(example.get("query", example.get("sql", example.get("code", ""))))
    if not db_id or not question or not query:
        raise ValueError(
            "Each example must contain either `messages` or Spider fields "
            "`db_id`, `question`, and `query`."
        )

    return [
        {"role": "system", "content": "Convert natural language to SQL."},
        {"role": "user", "content": f"For db_id:[{db_id}]\n\n{question}"},
        {"role": "assistant", "content": query},
    ]


def adapt_messages_for_model(model_name: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Mirror upstream handling for Gemma chat templates."""
    if "google/gemma" not in model_name:
        return messages

    if len(messages) >= 3 and messages[0]["role"] == "system":
        adapted = [dict(messages[1]), dict(messages[2])]
        adapted[0]["content"] = messages[0]["content"] + "\n\n" + adapted[0]["content"]
        return adapted
    return messages


def assistant_content(messages: Sequence[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg["content"]
    raise ValueError("Example has no assistant message.")


def prompt_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(msg) for msg in messages if msg["role"] != "assistant"]


def format_full_and_prompt(
    tokenizer: Any,
    model_name: str,
    messages: List[Dict[str, str]],
) -> Tuple[str, str]:
    formatted_messages = adapt_messages_for_model(model_name, messages)
    full_text = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages(formatted_messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    return full_text, prompt_text


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    for idx in range(n):
        if a[idx] != b[idx]:
            return idx
    return n


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last_start = len(haystack) - len(needle)
    for start in range(last_start + 1):
        if list(haystack[start:start + len(needle)]) == list(needle):
            return start
    return -1


def tokenize_sft_example(
    example: Dict[str, Any],
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> Dict[str, List[int]]:
    messages = example_to_messages(example)
    full_text, prompt_text = format_full_and_prompt(tokenizer, model_name, messages)

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = list(input_ids)

    prefix_len = common_prefix_len(input_ids, prompt["input_ids"])
    if prefix_len < min(len(input_ids), len(prompt["input_ids"])) - 4:
        labels = [-100] * len(input_ids)
        target_ids = tokenizer(
            assistant_content(messages),
            add_special_tokens=False,
        )["input_ids"]
        target_start = find_subsequence(input_ids, target_ids)
        if target_start >= 0:
            target_end = min(target_start + len(target_ids), len(labels))
            labels[target_start:target_end] = input_ids[target_start:target_end]
        else:
            labels[prefix_len:] = input_ids[prefix_len:]
    else:
        labels[:prefix_len] = [-100] * prefix_len

    labels = [label if mask else -100 for label, mask in zip(labels, attention_mask)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class SFTDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [self.label_pad_token_id] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def prepare_train_dataset(
    train_file: str,
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> Any:
    dataset = load_dataset("json", data_files=train_file, split="train")
    log(f"Loaded {len(dataset)} training examples from {train_file}.")

    tokenized = dataset.map(
        lambda example: tokenize_sft_example(example, tokenizer, model_name, max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT data",
    )
    tokenized = tokenized.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc="Dropping fully truncated examples",
    )
    log(f"Using {len(tokenized)} training examples after truncation filtering.")
    return tokenized


def extract_db_id(example: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    if example.get("db_id"):
        return str(example["db_id"])

    for msg in messages:
        match = SPIDER_DB_RE.search(msg["content"])
        if match:
            return match.group(1)
    raise ValueError("Could not find db_id in example.")


def generated_sql_from_output(tokenizer: Any, input_len: int, output_ids: torch.Tensor) -> str:
    generated_tokens = output_ids[input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    text = text.strip()
    if text.endswith("<|end|>"):
        text = text[:-7].strip()
    return text


def python_sqlite_stdout(db_file: str, query: str) -> Tuple[str, str, int]:
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        rows = cur.execute(query).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        return "", str(exc), 1

    lines = []
    for row in rows:
        lines.append("|".join("" if value is None else str(value) for value in row))
    return ("\n".join(lines) + ("\n" if lines else ""), "", 0)


def run_sql(
    db_file: str,
    query: str,
    sqlite_bin: str,
    timeout: int,
) -> Tuple[str, str, int]:
    try:
        result = subprocess.run(
            [sqlite_bin, db_file, query],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        return stdout, stderr, result.returncode
    except FileNotFoundError:
        return python_sqlite_stdout(db_file, query)
    except subprocess.TimeoutExpired:
        return "", "SQL execution timed out", 124


def spider_execution_match(
    generated_sql: str,
    gold_sql: str,
    db_id: str,
    spider_database_dir: str,
    sqlite_bin: str,
    timeout: int,
) -> Tuple[bool, Dict[str, Any]]:
    db_file = os.path.join(spider_database_dir, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_file):
        return False, {
            "db_file": db_file,
            "generated_stdout": "",
            "gold_stdout": "",
            "generated_stderr": f"Missing database: {db_file}",
            "gold_stderr": "",
            "generated_returncode": None,
            "gold_returncode": None,
        }

    generated_stdout, generated_stderr, generated_returncode = run_sql(
        db_file,
        generated_sql,
        sqlite_bin,
        timeout,
    )
    gold_stdout, gold_stderr, gold_returncode = run_sql(
        db_file,
        gold_sql,
        sqlite_bin,
        timeout,
    )

    return generated_stdout == gold_stdout, {
        "db_file": db_file,
        "generated_stdout": generated_stdout,
        "gold_stdout": gold_stdout,
        "generated_stderr": generated_stderr,
        "gold_stderr": gold_stderr,
        "generated_returncode": generated_returncode,
        "gold_returncode": gold_returncode,
    }


def evaluate_spider(
    model: Any,
    tokenizer: Any,
    eval_file: str,
    model_name_for_template: str,
    spider_database_dir: str,
    output_predictions_file: str,
    metrics_file: str,
    max_length: int,
    max_new_tokens: int,
    sqlite_bin: str,
    sql_timeout: int,
    max_eval_examples: Optional[int],
) -> Dict[str, Any]:
    dataset = load_dataset("json", data_files=eval_file, split="train")
    if max_eval_examples is not None:
        dataset = dataset.select(range(min(max_eval_examples, len(dataset))))

    model.eval()
    model_device = next(model.parameters()).device
    log(f"Evaluating Spider on {model_device} ({len(dataset)} examples).")
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    correct = 0
    strict_correct = 0
    stdout_match_generated_error = 0
    generated_error = 0
    gold_error = 0
    total = 0
    prediction_path = Path(output_predictions_file)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("w", encoding="utf-8") as pred_f:
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating Spider")):
            messages = example_to_messages(example)
            db_id = extract_db_id(example, messages)
            gold_sql = assistant_content(messages)
            formatted_messages = adapt_messages_for_model(model_name_for_template, messages)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages(formatted_messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]

            generated_sql = generated_sql_from_output(
                tokenizer,
                inputs["input_ids"].shape[1],
                output_ids,
            )
            is_correct, exec_info = spider_execution_match(
                generated_sql,
                gold_sql,
                db_id,
                spider_database_dir,
                sqlite_bin,
                sql_timeout,
            )
            gen_ok = exec_info["generated_returncode"] == 0
            gold_ok = exec_info["gold_returncode"] == 0
            is_strict_correct = is_correct and gen_ok and gold_ok

            correct += int(is_correct)
            strict_correct += int(is_strict_correct)
            stdout_match_generated_error += int(is_correct and not gen_ok)
            generated_error += int(not gen_ok)
            gold_error += int(not gold_ok)
            total += 1
            record = {
                "idx": idx,
                "db_id": db_id,
                "correct": is_correct,
                "strict_correct": is_strict_correct,
                "generated": generated_sql,
                "gold": gold_sql,
                **exec_info,
            }
            pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_f.flush()

    accuracy = correct / total if total else 0.0
    strict_accuracy = strict_correct / total if total else 0.0
    metrics = {
        "execution_accuracy": accuracy,
        "correct": correct,
        "strict_execution_accuracy": strict_accuracy,
        "strict_correct": strict_correct,
        "stdout_match_generated_error": stdout_match_generated_error,
        "generated_error": generated_error,
        "gold_error": gold_error,
        "total": total,
        "eval_file": eval_file,
        "spider_database_dir": spider_database_dir,
    }
    metrics_path = Path(metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log(f"Success Rate: {correct}/{total} = {accuracy:.4f}")
    log(f"Strict Success Rate: {strict_correct}/{total} = {strict_accuracy:.4f}")
    log(
        "SQL errors: "
        f"generated={generated_error}, gold={gold_error}, "
        f"stdout-matches-with-generated-error={stdout_match_generated_error}"
    )
    log(f"Wrote predictions to {output_predictions_file}")
    log(f"Wrote metrics to {metrics_file}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standard SFT on Spider and evaluate with execution match."
    )
    parser.add_argument("--model_name", required=True, help="HF model name/path to fine-tune.")
    parser.add_argument("--seed", type=int, required=True, help="Fine-tuning seed.")
    parser.add_argument("--train_file", default="spider_train.jsonl")
    parser.add_argument("--eval_file", default="spider_test.jsonl")
    parser.add_argument("--spider_database_dir", default="spider_data/database")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_epochs", type=float, default=4.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save per-epoch model checkpoints. Off by default to avoid huge optimizer checkpoint writes.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--sqlite_bin", default="sqlite3")
    parser.add_argument("--sql_timeout", type=int, default=30)
    parser.add_argument(
        "--eval_device",
        default=None,
        help="Device for eval-only generation. Defaults to cuda:0 when CUDA is available.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = f"sft_spider_{safe_name(args.model_name)}_seed{args.seed}"

    if not args.eval_only and not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Evaluation file not found: {args.eval_file}")

    if args.eval_only and not is_main_process():
        return

    bf16 = torch.cuda.is_available() and not args.no_bf16 and not args.fp16
    tokenizer_source = args.output_dir if args.eval_only else args.model_name
    model_source = args.output_dir if args.eval_only else args.model_name

    tokenizer = load_tokenizer(tokenizer_source, args.trust_remote_code)
    model = load_model(model_source, args.trust_remote_code, bf16=bf16)
    add_upstream_special_tokens(tokenizer, model)
    model.config.pad_token_id = tokenizer.pad_token_id

    if not args.eval_only:
        train_dataset = prepare_train_dataset(
            args.train_file,
            tokenizer,
            args.model_name,
            args.max_length,
        )

        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        training_kwargs = dict(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            logging_steps=args.logging_steps,
            save_strategy="epoch" if args.save_checkpoints else "no",
            save_total_limit=args.save_total_limit,
            bf16=bf16,
            fp16=args.fp16,
            optim="adamw_torch",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            dataloader_num_workers=args.dataloader_num_workers,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        )
        if args.save_checkpoints and "save_only_model" in TrainingArguments.__dataclass_fields__:
            training_kwargs["save_only_model"] = True
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            training_kwargs["ddp_find_unused_parameters"] = False

        trainer = Trainer(
            model=model,
            args=TrainingArguments(**training_kwargs),
            train_dataset=train_dataset,
            data_collator=SFTDataCollator(tokenizer),
            tokenizer=tokenizer,
        )

        log(
            "Starting SFT: "
            f"model={args.model_name}, seed={args.seed}, epochs={args.num_epochs}, "
            f"lr={args.learning_rate}, batch/GPU={args.per_device_batch_size}, "
            f"grad_accum={args.gradient_accumulation_steps}"
        )
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        model = trainer.model
        if hasattr(model, "module"):
            model = model.module

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    if not is_main_process():
        return

    eval_device = args.eval_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if next(model.parameters()).device != torch.device(eval_device):
        model.to(eval_device)
    log(f"Prepared evaluation model on {next(model.parameters()).device}.")

    predictions_file = os.path.join(args.output_dir, "spider_eval_predictions.jsonl")
    metrics_file = os.path.join(args.output_dir, "spider_eval_metrics.json")
    evaluate_spider(
        model=model,
        tokenizer=tokenizer,
        eval_file=args.eval_file,
        model_name_for_template=args.model_name,
        spider_database_dir=args.spider_database_dir,
        output_predictions_file=predictions_file,
        metrics_file=metrics_file,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        sqlite_bin=args.sqlite_bin,
        sql_timeout=args.sql_timeout,
        max_eval_examples=args.max_eval_examples,
    )


if __name__ == "__main__":
    main()
