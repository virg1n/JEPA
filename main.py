
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
import math
import os
from contextlib import nullcontext

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

if distributed:
    backend = "nccl"
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

master_process = (not distributed) or dist.get_rank() == 0
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=torch.bfloat16).to(device)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({"additional_special_tokens": ["[PRED]"]})
model.resize_token_embeddings(len(tokenizer))
PRED_ID = tokenizer.convert_tokens_to_ids("[PRED]")

MAX_LEN = 512
K = 1
JEPA_GAMMA = 0.5
accum_steps = 4
lr = 2e-5
weight_decay = 0.01
num_epochs = 3
use_compile = True


def tokenize(batch):
    q = tokenizer(batch["question"], truncation=True, max_length=MAX_LEN - (K+1), add_special_tokens=False)
    ans = tokenizer(batch["query"], truncation=True, max_length=MAX_LEN - 1, add_special_tokens=False)

    eos_id = tokenizer.eos_token_id

    tokenized = {}
    tokenized["q_ids"] = [input_ids + [eos_id] + [PRED_ID] * K for input_ids in q["input_ids"]]
    tokenized["ans_ids"] =  [input_ids + [eos_id] for input_ids in ans["input_ids"]]
    tokenized["generation_ids"] = [qi + [eos_id] + ai + [eos_id] for qi, ai in zip(q["input_ids"], ans["input_ids"])]
    tokenized["generation_len"] = [len(qi) + 1 for qi in q["input_ids"]]
    return tokenized

dataset = load_dataset("xlangai/spider", split="train")
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def pad(batch, pid):
    m = max(len(b) for b in batch)
    ids = torch.full((len(batch), m), pid, dtype=torch.long)
    mask = torch.zeros((len(batch), m), dtype=torch.long)

    for i, b in enumerate(batch):
        ids[i, :len(b)] = torch.tensor(b)
        mask[i, :len(b)] = 1

    return ids, mask

def collate_fn(samples):
    pid = tokenizer.pad_token_id

    q_ids, q_mask = pad([sample["q_ids"] for sample in samples], pid)
    ans_ids, ans_mask = pad([sample["ans_ids"] for sample in samples], pid)
    gen_ids, gen_mask = pad([sample["generation_ids"] for sample in samples], pid)

    labels = gen_ids.clone()
    labels[gen_mask == 0] = -100


    for i, s in enumerate(samples):
        labels[i, :s["generation_len"]] = -100

    return {"q_ids": q_ids, "q_mask": q_mask,
            "ans_ids": ans_ids, "ans_mask": ans_mask,
            "gen_ids": gen_ids, "gen_mask": gen_mask, "labels": labels}

def get_last_token(hidden, mask, offset = -1):
    idx = mask.sum(1) + offset
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

sampler = DistributedSampler(dataset, shuffle=True) if distributed else None

loader = DataLoader(dataset, batch_size=8, shuffle=(sampler is None),
                    sampler=sampler,
                     collate_fn = collate_fn)

model.config.use_cache = False

# assert len(loader) % accum_steps == 0, "len(loader) is not divisile by accum steps"


total_steps = (math.ceil(len(loader) / accum_steps)) * num_epochs

optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps = int(0.03 * total_steps), num_training_steps = total_steps
)

optimizer.zero_grad(set_to_none=True)

model.gradient_checkpointing_enable()
model.train()


if distributed:
    model = DDP(model, device_ids=[local_rank])
    
if use_compile:
    model = torch.compile(model)

num_batches = len(loader)
remainder = num_batches % accum_steps

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)



for epoch in range(num_epochs):
    if distributed:
        sampler.set_epoch(epoch)
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch["labels"].contiguous()

        is_last_batch = (step + 1) == num_batches
        should_step = ((step + 1) % accum_steps == 0) or is_last_batch

        sync_context = (
            model.no_sync()
            if distributed and not should_step
            else nullcontext()
        )
        
        with sync_context:
            p = get_last_token(model(input_ids=batch["q_ids"], attention_mask=batch["q_mask"], output_hidden_states=True).hidden_states[-1], 
                            mask=batch["q_mask"]) #B, H
            
            t = get_last_token(model(input_ids=batch["ans_ids"], attention_mask=batch["ans_mask"], output_hidden_states=True).hidden_states[-1], 
                            mask=batch["ans_mask"],  #B, H
                            offset=-2) # last SQL token before eos

            jepa_loss = (1 - torch.nn.functional.cosine_similarity(p, t, dim=-1)).mean()
            output = model(input_ids=batch["gen_ids"], attention_mask=batch["gen_mask"])
            
            logits = output.logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            

            loss = loss_fn(
                logits.view(-1, logits.size(-1)), shifted_labels.view(-1)
            )
            
            # loss = output.loss
            # print(loss)
            if master_process and step % 500 == 0:
                print(loss.item())
                print(jepa_loss)

            divisor = remainder if (remainder != 0 and step >= num_batches - remainder) else accum_steps 

            loss = loss + JEPA_GAMMA * jepa_loss
            loss = loss / divisor


            loss.backward()

        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # break



if master_process:
    save_model = model.module if distributed else model
    save_model.save_pretrained("llm-jepa-smollm-spider")
    tokenizer.save_pretrained("llm-jepa-smollm-spider")

if distributed:
    dist.destroy_process_group()