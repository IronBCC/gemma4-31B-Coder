#!/usr/bin/env python3
"""Rust LoRA SFT over a FROZEN Gemma-4-31B base (PLAN §6 / gemma4-coder-phaseC-multi-plan).

Frozen base + LoRA adapter = the no-regression guarantee (base weights never change;
Python/C++ paths are untouched). This trains ONLY the adapter on execution-verified
Rust SFT data. Hyperparameters per PLAN §6 QLoRA validation:
  r=32, alpha=32, dropout=0, all proj modules, lr=2e-4 cosine, 3% warmup,
  1-3 epochs, max_seq_len=8192. Unsloth patched loader (KV-shared use_cache bug).

Single GPU1 (GPU0 = prod, OFF-LIMITS): CUDA_VISIBLE_DEVICES=1.
Saves the adapter only (not merged) so serving loads it as a live LoRA over the
frozen base — never merge (would destroy the frozen-base guarantee).
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    ap.add_argument("--data", default="data/rust_sft")
    ap.add_argument("--out", default="adapters/rust-lora-v1")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=-1)  # -1 = full epochs; >0 caps for a quick run
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA 4-bit base (less VRAM)")
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer

    print(f"[load] base={a.base} 4bit={a.load_4bit} max_seq={a.max_seq}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.base,
        max_seq_length=a.max_seq,
        dtype=None,                       # auto (bf16 on Blackwell)
        load_in_4bit=a.load_4bit,
        full_finetuning=False,
    )
    # Gemma-4 chat template (thinking-on, matched train==serve)
    try:
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    except Exception as e:
        print(f"[warn] get_chat_template gemma-4 fallback: {e}", flush=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.rank,
        lora_alpha=a.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",   # Unsloth patched (use_cache=False handled)
        random_state=0,
    )

    ds = load_from_disk(a.data)
    print(f"[data] {len(ds)} sft examples", flush=True)

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}
    ds = ds.map(fmt, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])

    cfg = SFTConfig(
        output_dir=a.out,
        per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=0,
        dataset_num_proc=4,
        max_seq_length=a.max_seq,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)
    print("[train] starting LoRA SFT (frozen base)…", flush=True)
    trainer.train()

    # save ADAPTER ONLY — never merge into base (frozen-base guarantee)
    model.save_pretrained(a.out)
    tokenizer.save_pretrained(a.out)
    print(f"[done] adapter saved -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
