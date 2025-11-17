# 
# # Enhanced train_lora_adapter_cpu.py
"""
Train a micro LoRA adapter on failure shards from logs/patch_trials.jsonl.
CPU-only version for GPUs with compute capability < 7.5.
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# We skip importing bitsandbytes since it requires CUDA sm_70+ GPUs

def build_failure_shard(trials_path="logs/patch_trials.jsonl", out_path="data/lora_failure_shard.jsonl",
                       max_examples=1200, seed=42):
    random.seed(seed)
    examples = []

    if not Path(trials_path).exists():
        raise FileNotFoundError(f"No trials log at {trials_path}")

    with open(trials_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            failure_label = rec.get("failure_after", {}).get("failure_label", "")
            if failure_label in ("NLI_FAILURE", "KG_MISMATCH", "BOTH_FAIL", "NO_EVIDENCE"):
                query = rec["query"]
                retrieved_texts = rec.get("failure_before", {}).get("retrieved_texts", [])
                context = "\n".join(retrieved_texts) if retrieved_texts else rec.get("final_answer", "")
                examples.append({
                    "query": query,
                    "context": context,
                    "gold_answer": rec["gold_answer"]
                })

    # Deduplicate and shuffle
    unique_examples = []
    seen_queries = set()
    for ex in examples:
        if ex["query"] not in seen_queries:
            unique_examples.append(ex)
            seen_queries.add(ex["query"])

    random.shuffle(unique_examples)
    unique_examples = unique_examples[:max_examples]

    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex) + "\n")
    return out_path

def load_shard(shard_path):
    rows = []
    with open(shard_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)

def format_for_causal(batch, tokenizer):
    examples = []
    for ex in batch:
        examples.append(
            f"Context:\n{ex['context']}\nQuestion: {ex['query']}\nInstructions: Cite evidence, avoid speculation, keep entity–relation pairs consistent with KG.\nAnswer: {ex['gold_answer']}"
        )
    encodings = tokenizer(
        examples,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return encodings

def train(args):
    # Build failure shard
    shard_path = build_failure_shard(
        args.trials,
        args.out_shard,
        max_examples=args.max_examples
    )

    # Load dataset
    ds = load_shard(shard_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    # Load base model (CPU-only)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map={"": "cpu"},  # Force CPU
        torch_dtype=torch.float32
    )

    # LoRA configuration
    lcfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lcfg)

    # Training arguments (CPU)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        fp16=False,  # CPU only
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=format_for_causal
    )

    # Train
    try:
        trainer.train()
        model.save_pretrained(args.out_dir)
        print(f"✅ LoRA adapter saved to {args.out_dir}")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only LoRA training")
    parser.add_argument("--trials", default="logs/patch_trials.jsonl")
    parser.add_argument("--out_shard", default="data/lora_failure_shard.jsonl")
    parser.add_argument("--base", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--out_dir", default="adapters/fever_v1")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=1200)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
