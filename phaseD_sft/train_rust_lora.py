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
import json
import os
import re
import sys
import time


def rendered_assistant_turn_spans(tokenizer, messages: list[dict], text: str) -> tuple[list[tuple[int, int]], bool]:
    """Return Gemma-rendered assistant turn character spans.

    The fast path follows Gemma's native markers. The fallback handles a changed
    template by using prefix diffs, but reports that fallback was used so audits
    can catch template drift.
    """
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    turn_markers = list(re.finditer(r"<\|turn\>(system|user|model|tool)\n", text))
    model_starts = [m.start() for m in turn_markers if m.group(1) == "model"]
    if len(model_starts) == assistant_count:
        marker_starts = [m.start() for m in turn_markers]
        spans = []
        for start in model_starts:
            end = next((pos for pos in marker_starts if pos > start), len(text))
            spans.append((start, end))
        return spans, False

    spans = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix = (
            tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
            if i
            else ""
        )
        prefix_with_assistant = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        start = len(prefix)
        end = len(prefix_with_assistant)
        if end <= start or not text.startswith(prefix_with_assistant):
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            pos = text.find(content, start)
            if pos < 0:
                continue
            start = pos
            end = pos + len(content)
        spans.append((start, end))
    return spans, True


def label_tokens_for_spans(input_ids: list[int], offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> tuple[list[int], int]:
    labels = [-100] * len(input_ids)
    supervised = 0
    span_i = 0
    for j, (start, end) in enumerate(offsets):
        while span_i < len(spans) and spans[span_i][1] <= start:
            span_i += 1
        if span_i < len(spans) and end > spans[span_i][0] and start < spans[span_i][1]:
            labels[j] = input_ids[j]
            supervised += 1
    return labels, supervised


def configure_attention_implementation(model, implementation: str | None) -> int:
    """Apply an explicit Transformers attention backend after wrapper loading.

    Unsloth may load a Gemma4 adapter with SDPA even when an attention backend
    is supplied to its loader.  The model and PEFT wrappers retain the
    Transformers config objects, so set the post-load property on every
    relevant root/nested config before training starts.
    """
    if implementation is None:
        return 0

    seen_models: set[int] = set()
    seen_configs: set[int] = set()
    model_stack = [model]
    config_stack = []

    while model_stack:
        candidate = model_stack.pop()
        if candidate is None or id(candidate) in seen_models:
            continue
        seen_models.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None:
            config_stack.append(config)
        for attr in ("base_model", "model"):
            child = getattr(candidate, attr, None)
            if child is not None and child is not candidate:
                model_stack.append(child)

    configured = 0
    while config_stack:
        config = config_stack.pop()
        if config is None or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        try:
            config._attn_implementation = implementation
        except (AttributeError, TypeError):
            continue
        configured += 1
        for attr in ("text_config", "vision_config", "audio_config"):
            child = getattr(config, attr, None)
            if child is not None:
                config_stack.append(child)
    return configured


def install_hf_flex_routing_debug() -> None:
    """Call the HF Flex interface outside the regional Dynamo trace and log routing once."""
    import torch
    from transformers.integrations import flex_attention as hf_flex_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    seen = 0

    @torch.compiler.disable(recursive=False)
    def hf_flex_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=None,
        softcap=None,
        s_aux=None,
        **kwargs,
    ):
        nonlocal seen
        if seen == 0:
            print(
                "[attention-route] before "
                f"module={type(module).__name__} layer={getattr(module, 'layer_idx', None)} "
                f"mask={type(attention_mask).__module__}.{type(attention_mask).__name__} "
                f"dynamo={hf_flex_attention.is_torchdynamo_compiling()} "
                f"hf_compiled={hf_flex_attention.WrappedFlexAttention._is_flex_compiled}",
                flush=True,
            )
        seen += 1
        result = original(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            softcap=softcap,
            s_aux=s_aux,
            **kwargs,
        )
        if seen == 1:
            print(
                "[attention-route] after "
                f"hf_compiled={hf_flex_attention.WrappedFlexAttention._is_flex_compiled}",
                flush=True,
            )
        return result

    ALL_ATTENTION_FUNCTIONS.register("flex_attention", hf_flex_attention_forward)


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
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1)  # -1 = full epochs; >0 caps for a quick run
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--save-total-limit", type=int, default=10)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA 4-bit base (less VRAM)")
    ap.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flex_attention", "flash_attention_2", "flash_attention_3"),
        default=None,
        help="override the post-load Transformers attention backend (required for long masked Gemma4 sequences)",
    )
    ap.add_argument(
        "--debug-hf-flex-routing",
        action="store_true",
        help="run the HF Flex interface outside regional Dynamo and emit one routing assertion",
    )
    ap.add_argument(
        "--sm120-attn",
        action="store_true",
        help="sm_120 long-context stack (Gate B verified @49,152): xformers sliding + "
             "chunked-global (hd-512) attention + fused cut_cross_entropy loss. "
             "Requires env UNSLOTH_RETURN_HIDDEN_STATES=1.",
    )
    ap.add_argument("--init-adapter", default=None,
                    help="initialize from an existing adapter dir, but start a fresh optimizer/scheduler")
    ap.add_argument("--full-transcript-loss", action="store_true",
                    help="train on all rendered tokens; default is assistant-message tokens only")
    ap.add_argument("--assistant-content-only-loss", action="store_true",
                    help="legacy mode: train only assistant content text, excluding Gemma turn/tool-call tokens")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out (reboot-safe)")
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from datasets import load_from_disk
    from transformers import Trainer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if a.resume and not a.init_adapter:
        checkpoints_dir = a.out
        has_checkpoint = any(
            name.startswith("checkpoint-")
            for name in os.listdir(checkpoints_dir)
        ) if os.path.isdir(checkpoints_dir) else False
        adapter_path = os.path.join(a.out, "adapter_model.safetensors")
        if not has_checkpoint and os.path.isfile(adapter_path):
            a.init_adapter = a.out
            print(f"[resume] no checkpoint found, initializing from final adapter in {a.out}", flush=True)

    model_name = a.init_adapter or a.base
    print(f"[load] model={model_name} base={a.base} 4bit={a.load_4bit} max_seq={a.max_seq}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=a.max_seq,
        dtype=None,                       # auto (bf16 on Blackwell)
        load_in_4bit=a.load_4bit,
        full_finetuning=False,
    )
    # Keep the model's native Gemma chat template. Unsloth's generic gemma-4
    # template drops tool_calls, which masks the real agent action target.
    base_chat_template = tokenizer.chat_template
    print("[template] using base tokenizer Gemma chat template", flush=True)

    if a.init_adapter:
        print(f"[init] continuing adapter weights from {a.init_adapter} with fresh optimizer", flush=True)
        FastLanguageModel.for_training(model)
    else:
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

    attention_configs_changed = configure_attention_implementation(model, a.attn_implementation)
    if a.attn_implementation is not None:
        effective_attention = getattr(getattr(model, "config", None), "_attn_implementation", None)
        print(
            f"[attention] requested={a.attn_implementation} effective={effective_attention} "
            f"configs_changed={attention_configs_changed}",
            flush=True,
        )
    if a.debug_hf_flex_routing:
        if a.attn_implementation != "flex_attention":
            raise ValueError("--debug-hf-flex-routing requires --attn-implementation flex_attention")
        install_hf_flex_routing_debug()
        print("[attention] interface=hf_flex_outside_regional_dynamo", flush=True)
    if a.sm120_attn:
        from phaseD_sft.sm120_attention import (
            SM120_ATTN_IMPL,
            register_sm120_attention,
            require_return_hidden_states,
            swap_to_nonreentrant_checkpoint,
        )
        require_return_hidden_states()
        register_sm120_attention()
        attention_configs_changed = configure_attention_implementation(model, SM120_ATTN_IMPL)
        swap_to_nonreentrant_checkpoint(model)
        print(f"[sm120] attention backend forced to {SM120_ATTN_IMPL}, "
              f"configs_changed={attention_configs_changed}", flush=True)
    print(f"[progress] event=model_ready epoch_seconds={int(time.time())}", flush=True)

    ds = load_from_disk(a.data)
    print(f"[data] {len(ds)} sft examples", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "base": a.base,
                "data": a.data,
                "data_len": len(ds),
                "data_fingerprint": getattr(ds, "_fingerprint", None),
                "out": a.out,
                "init_adapter": a.init_adapter,
                "rank": a.rank,
                "alpha": a.alpha,
                "lr": a.lr,
                "epochs": a.epochs,
                "bsz": a.bsz,
                "grad_accum": a.grad_accum,
                "max_seq": a.max_seq,
                "warmup_steps": a.warmup_steps,
                "max_steps": a.max_steps,
                "logging_steps": a.logging_steps,
                "save_steps": a.save_steps,
                "save_total_limit": a.save_total_limit,
                "load_4bit": a.load_4bit,
                "attn_implementation": a.attn_implementation,
                "attention_configs_changed": attention_configs_changed,
                "sm120_attn": a.sm120_attn,
                "debug_hf_flex_routing": a.debug_hf_flex_routing,
                "full_transcript_loss": a.full_transcript_loss,
                "assistant_content_only_loss": a.assistant_content_only_loss,
                "resume": a.resume,
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}

    def assistant_tokenize(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(
            text=text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=a.max_seq,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        labels = [-100] * len(input_ids)
        spans = []
        if a.assistant_content_only_loss:
            cursor = 0
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                pos = text.find(content, cursor)
                if pos < 0:
                    continue
                end = pos + len(content)
                spans.append((pos, end))
                cursor = end
        else:
            # Label the exact assistant turn as rendered by Gemma's chat template,
            # including <|turn>model, tool-call JSON, and turn/response delimiters.
            # System/user/tool observations remain conditioning only.
            spans, _fallback_used = rendered_assistant_turn_spans(tokenizer, messages, text)

        labels, supervised = label_tokens_for_spans(input_ids, offsets, spans)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "supervised_tokens": supervised,
        }

    if a.full_transcript_loss:
        print("[data] full-transcript loss enabled", flush=True)
        ds = ds.map(fmt, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])
    else:
        if a.assistant_content_only_loss:
            print("[data] assistant content-only loss enabled", flush=True)
        else:
            print("[data] assistant turn loss enabled", flush=True)
        ds = ds.map(
            assistant_tokenize,
            remove_columns=ds.column_names,
            num_proc=4,
            desc="assistant-tokenize",
        )
        before = len(ds)
        ds = ds.filter(lambda ex: ex["supervised_tokens"] > 0, num_proc=4)
        print(f"[data] supervised examples {len(ds)}/{before}", flush=True)
        ds = ds.remove_columns(["supervised_tokens"])

    print(f"[progress] event=data_ready epoch_seconds={int(time.time())} examples={len(ds)}", flush=True)

    cfg = SFTConfig(
        output_dir=a.out,
        per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.grad_accum,
        warmup_steps=a.warmup_steps,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        logging_steps=a.logging_steps,
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=0,
        dataset_num_proc=4,
        max_seq_length=a.max_seq,
        ignore_data_skip=True,
        report_to="none",
    )

    class TrainingProgressCallback(TrainerCallback):
        """Emit flushed microstep/optimizer-step liveness and an ETA to the run log."""

        def __init__(self):
            self.started_at: float | None = None
            self.last_optimizer_step_at: float | None = None
            self.completed_microsteps = 0

        def _report(self, event: str, state) -> None:
            now = time.monotonic()
            epoch_seconds = int(time.time())
            if self.started_at is None:
                self.started_at = now
            elapsed_seconds = now - self.started_at
            total_steps = max(0, int(state.max_steps or 0))
            total_microsteps = total_steps * a.grad_accum
            eta_seconds = -1
            if self.completed_microsteps and total_microsteps > self.completed_microsteps:
                eta_seconds = int(
                    elapsed_seconds / self.completed_microsteps * (total_microsteps - self.completed_microsteps)
                )
            print(
                f"[progress] event={event} optimizer_step={state.global_step}/{total_steps} "
                f"microstep={self.completed_microsteps}/{total_microsteps} "
                f"elapsed_seconds={int(elapsed_seconds)} eta_seconds={eta_seconds} epoch_seconds={epoch_seconds}",
                flush=True,
            )

        def on_train_begin(self, args, state, control, **kwargs):
            self.started_at = time.monotonic()
            self.last_optimizer_step_at = self.started_at
            self._report("train_begin", state)
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            self.completed_microsteps += 1
            self._report("gradient_microstep", state)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.completed_microsteps += 1
            now = time.monotonic()
            step_wall_seconds = 0.0 if self.last_optimizer_step_at is None else now - self.last_optimizer_step_at
            self.last_optimizer_step_at = now
            self._report("optimizer_step", state)
            print(
                f"[progress] event=optimizer_step_timing optimizer_step={state.global_step}/{state.max_steps} "
                f"step_wall_seconds={step_wall_seconds:.2f}",
                flush=True,
            )
            return control

    if a.sm120_attn:
        from phaseD_sft.sm120_attention import install_fused_ce_loss
        install_fused_ce_loss(SFTTrainer if a.full_transcript_loss else Trainer)

    progress_callback = TrainingProgressCallback()
    if a.full_transcript_loss:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            args=cfg,
            callbacks=[progress_callback],
        )
    else:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def collate(features):
            import torch
            max_len = max(len(f["input_ids"]) for f in features)
            input_ids, attention_mask, labels = [], [], []
            for f in features:
                pad = max_len - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [pad_id] * pad)
                attention_mask.append(f["attention_mask"] + [0] * pad)
                labels.append(f["labels"] + [-100] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            args=cfg,
            data_collator=collate,
            callbacks=[progress_callback],
        )
    resume = a.resume
    if resume:
        import glob
        cks = glob.glob(os.path.join(a.out, "checkpoint-*"))
        if not cks:
            print("[resume] no checkpoint found — starting fresh", flush=True)
            resume = False
        else:
            print(f"[resume] resuming from latest of {len(cks)} checkpoints in {a.out}", flush=True)
    print("[train] starting LoRA SFT (frozen base)…", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    # save ADAPTER ONLY — never merge into base (frozen-base guarantee)
    model.save_pretrained(a.out)
    tokenizer.chat_template = base_chat_template
    tokenizer.save_pretrained(a.out)
    print(f"[done] adapter saved -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
