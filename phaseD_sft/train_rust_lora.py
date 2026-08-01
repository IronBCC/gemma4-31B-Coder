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
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def assert_training_dataset_admitted(path: str | os.PathLike[str]) -> None:
    """Fail closed for teacher datasets carrying the schema-v2 admission manifest."""

    dataset_path = Path(path)
    manifest_path = dataset_path / "manifest.json"
    if not dataset_path.is_dir() or not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("training dataset manifest is unreadable") from exc
    if manifest.get("schema_version") != 2:
        return
    train_jsonl = dataset_path / "train.jsonl"
    digest = hashlib.sha256(train_jsonl.read_bytes()).hexdigest()
    if manifest.get("dataset_variant") == "teacher_train_mix_v2p11_fable_reasoned_v1":
        gate = manifest.get("format_loss_gate")
        report = gate.get("report") if isinstance(gate, dict) else None
        report_path = Path(str(report.get("path", ""))) if isinstance(report, dict) else None
        structural = manifest.get("structural_gates")
        if (
            manifest.get("complete") is not True
            or manifest.get("all_training_gates_complete") is not True
            or manifest.get("training_admitted") != manifest.get("rendered")
            or manifest.get("evaluation_overlap") != 0
            or manifest.get("fable_rows") != 92
            or manifest.get("canonical_bash_tool_turns") != 517
            or manifest.get("reasoned_tool_turns") != 468
            or structural != {
                "all_fable_supervised_tool_calls_are_canonical_bash": True,
                "all_messages_have_boolean_loss": True,
                "evaluation_overlap": 0,
                "reasoned_tool_turns": 468,
            }
            or not isinstance(gate, dict)
            or gate.get("status") != "passed_full_dataset_verification"
            or gate.get("failure_count") != 0
            or gate.get("samples") != manifest.get("rendered")
            or report_path is None
            or not report_path.is_file()
            or report.get("sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest()
            or manifest.get("train_jsonl_sha256") != digest
        ):
            raise ValueError("teacher dataset admission or format/loss gate is incomplete")
        return
    gate = manifest.get("standard_native_format_loss_gate")
    if (
        manifest.get("all_training_gates_complete") is not True
        or manifest.get("training_admitted") != manifest.get("rendered")
        or not isinstance(gate, dict)
        or gate.get("status") != "passed"
        or gate.get("failure_count") != 0
        or manifest.get("train_jsonl_sha256") != digest
    ):
        raise ValueError("teacher dataset admission or format/loss gate is incomplete")


def load_training_dataset(path: str | os.PathLike[str]):
    """Load a Hugging Face dataset directory or a JSONL dataset file."""
    from datasets import Dataset, load_from_disk

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")
    if dataset_path.is_dir():
        manifest_path = dataset_path / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("training dataset manifest is unreadable") from exc
            if manifest.get("schema_version") == 2:
                train_jsonl = dataset_path / "train.jsonl"
                if not train_jsonl.is_file():
                    raise ValueError("schema-v2 teacher dataset is missing train.jsonl")
                return Dataset.from_json(str(train_jsonl))
        return load_from_disk(str(dataset_path))
    if dataset_path.is_file():
        return Dataset.from_json(str(dataset_path))
    raise ValueError(
        f"dataset path must be a Hugging Face dataset directory or JSONL file: {dataset_path}"
    )


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


def supervised_assistant_turn_spans(
    tokenizer, messages: list[dict], text: str
) -> tuple[list[tuple[int, int]], bool]:
    """Return assistant spans whose message-level ``loss`` flag is not false.

    Existing datasets do not carry the flag and therefore retain the historical
    all-assistant-turn objective. Patch-decision datasets can mark earlier
    assistant actions ``loss=False`` while preserving them as conditioning.
    """
    spans, fallback_used = rendered_assistant_turn_spans(tokenizer, messages, text)
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if len(spans) != len(assistant_messages):
        raise ValueError(
            f"assistant span/message mismatch: spans={len(spans)} messages={len(assistant_messages)}"
        )
    return [
        span
        for message, span in zip(assistant_messages, spans)
        if message.get("loss", True) is not False
    ], fallback_used


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


def ignore_data_skip_for_run(*, resume: bool) -> bool:
    """Preserve exact dataset position when restoring a Trainer checkpoint."""

    return not resume


def should_force_checkpoint(*, global_step: int, save_steps: int) -> bool:
    """Honor the requested absolute cadence even when checkpoint state restores an older one."""

    return global_step > 0 and save_steps > 0 and global_step % save_steps == 0


def resolve_gradient_checkpointing(mode: str) -> str | bool:
    """Map the CLI policy to the Unsloth PEFT loader contract."""

    if mode in {"unsloth", "bounded_unsloth", "hybrid"}:
        return "unsloth"
    if mode == "standard":
        return True
    raise ValueError(f"unsupported gradient checkpointing mode: {mode}")


def configure_hybrid_gradient_checkpointing(
    model,
    checkpoint_module,
    *,
    standard_stride: int = 10,
) -> dict[str, int]:
    """Use native reentrant checkpoints for a bounded subset of text layers."""

    from functools import partial

    if standard_stride < 2:
        raise ValueError("hybrid checkpoint standard stride must be at least 2")
    pristine_checkpoint = getattr(
        checkpoint_module,
        "_unsloth_pristine_checkpoint",
        None,
    )
    if not callable(pristine_checkpoint):
        raise RuntimeError("pristine PyTorch checkpoint function is unavailable")

    layers = [
        module
        for module in model.modules()
        if type(module).__name__ == "Gemma4TextDecoderLayer"
        and getattr(module, "gradient_checkpointing", False)
        and callable(getattr(module, "_gradient_checkpointing_func", None))
    ]
    if not layers:
        raise RuntimeError("no checkpoint-enabled Gemma4 text layers found")

    standard_layers = 0
    for index, layer in enumerate(layers):
        if index % standard_stride == 0:
            layer._gradient_checkpointing_func = partial(
                pristine_checkpoint,
                use_reentrant=True,
            )
            standard_layers += 1
    return {
        "text_layers": len(layers),
        "offloaded_layers": len(layers) - standard_layers,
        "standard_layers": standard_layers,
        "standard_stride": standard_stride,
    }


def recycle_bounded_unsloth_host_buffers(
    torch_module,
    checkpointing_module,
) -> dict[str, int | bool]:
    """Replace Unsloth's retained pinned activation pool with bounded pageable buffers."""

    expected_count = int(
        getattr(checkpointing_module, "INITIAL_CPU_BUFFER_COUNT", -1)
    )
    initial_elements = int(
        getattr(checkpointing_module, "INITIAL_CPU_BUFFER_SIZE", -1)
    )
    buffers = getattr(checkpointing_module, "CPU_BUFFERS", None)
    if expected_count != 200 or initial_elements != 128 * 1024:
        raise RuntimeError(
            "unexpected Unsloth host-buffer constants: "
            f"count={expected_count} elements={initial_elements}"
        )
    if not isinstance(buffers, list) or len(buffers) != expected_count:
        count = None if buffers is None else len(buffers)
        raise RuntimeError(
            f"unexpected Unsloth CPU buffer pool: count={count} expected={expected_count}"
        )
    if getattr(checkpointing_module, "BACKWARD_PASS", None) is not True:
        raise RuntimeError(
            "refusing to recycle Unsloth host buffers outside a completed backward"
        )
    cpu_index = getattr(checkpointing_module, "CPU_INDEX", None)
    if not isinstance(cpu_index, int) or not 0 <= cpu_index <= expected_count:
        raise RuntimeError(
            f"unexpected Unsloth CPU buffer index: {cpu_index!r}"
        )
    cuda = getattr(torch_module, "cuda", None)
    cuda_synchronize = getattr(cuda, "synchronize", None)
    if (
        not callable(getattr(cuda, "is_available", None))
        or not cuda.is_available()
        or not callable(cuda_synchronize)
    ):
        raise RuntimeError("CUDA synchronization is unavailable")
    cuda_synchronize()
    try:
        dtypes = {buffer.dtype for buffer in buffers}
        retained_bytes = sum(
            int(buffer.untyped_storage().nbytes()) for buffer in buffers
        )
        pinned_buffers = sum(bool(buffer.is_pinned()) for buffer in buffers)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Unsloth CPU buffer pool contains invalid entries") from exc
    if len(dtypes) != 1:
        raise RuntimeError(f"Unsloth CPU buffer dtypes diverged: {dtypes!r}")
    host_empty_cache = getattr(
        getattr(torch_module, "_C", None),
        "_host_emptyCache",
        None,
    )
    if not callable(host_empty_cache):
        raise RuntimeError("PyTorch host allocator cache drain is unavailable")

    dtype = next(iter(dtypes))
    replacement = [
        torch_module.empty(initial_elements, dtype=dtype, device="cpu")
        for _ in range(expected_count)
    ]
    if any(bool(buffer.is_pinned()) for buffer in replacement):
        raise RuntimeError("bounded Unsloth replacement buffer remained pinned")
    checkpointing_module.CPU_BUFFERS = replacement
    del buffers
    gc.collect()
    host_empty_cache()
    return {
        "buffer_count": expected_count,
        "initial_buffer_elements": initial_elements,
        "retained_bytes": retained_bytes,
        "pinned_buffers": pinned_buffers,
        "pageable_buffers": len(replacement),
        "cuda_synchronized": True,
        "host_cache_drained": True,
    }


def release_cuda_cache(torch_module) -> dict[str, int] | None:
    """Release only unoccupied CUDA allocator blocks and report the reservation delta."""

    cuda = torch_module.cuda
    if not cuda.is_available():
        return None
    allocated = int(cuda.memory_allocated())
    reserved_before = int(cuda.memory_reserved())
    cuda.empty_cache()
    reserved_after = int(cuda.memory_reserved())
    return {
        "allocated_bytes": allocated,
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": reserved_after,
    }


def configure_inductor_compile_threads(
    torch_module,
    unsloth_compile_common,
    requested_threads: int | str | None = None,
) -> dict[str, int]:
    """Apply the requested bounded Inductor worker count after Unsloth patches."""

    try:
        requested_threads = int(
            requested_threads
            if requested_threads is not None
            else os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "1")
        )
    except ValueError as exc:
        raise ValueError("TORCHINDUCTOR_COMPILE_THREADS must be an integer") from exc
    if not 1 <= requested_threads <= 32:
        raise ValueError("TORCHINDUCTOR_COMPILE_THREADS must be between 1 and 32")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(requested_threads)
    torch_module._inductor.config.compile_threads = requested_threads

    def configured_compile_threads() -> int:
        return requested_threads

    unsloth_compile_common.determine_compile_threads = configured_compile_threads
    unsloth_compile_common.torch_compile_options["compile_threads"] = requested_threads

    generated_options = unsloth_compile_common.get_torch_compile_options()
    torch_threads = int(torch_module._inductor.config.compile_threads)
    unsloth_threads = int(generated_options.get("compile_threads", -1))
    if torch_threads != requested_threads or unsloth_threads != requested_threads:
        raise RuntimeError(
            "failed to enforce bounded Inductor compilation: "
            f"requested={requested_threads} torch={torch_threads} unsloth={unsloth_threads}"
        )
    return {
        "torch_inductor_compile_threads": torch_threads,
        "unsloth_compile_threads": unsloth_threads,
    }


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
    ap.add_argument(
        "--gradient-checkpointing",
        choices=("unsloth", "bounded_unsloth", "standard", "hybrid"),
        default="unsloth",
        help="activation checkpointing policy; bounded_unsloth recycles pageable host buffers after each backward",
    )
    ap.add_argument("--full-transcript-loss", action="store_true",
                    help="train on all rendered tokens; default is assistant-message tokens only")
    ap.add_argument("--assistant-content-only-loss", action="store_true",
                    help="legacy mode: train only assistant content text, excluding Gemma turn/tool-call tokens")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out (reboot-safe)")
    a = ap.parse_args()
    assert_training_dataset_admitted(a.data)

    requested_compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "1")
    from unsloth import FastLanguageModel
    import torch
    from unsloth_zoo.temporary_patches import common as unsloth_compile_common
    from transformers import Trainer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    unsloth_compile_disabled = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1"
    unsloth_double_buffer_disabled = (
        os.environ.get("UNSLOTH_DISABLE_DOUBLE_BUFFER", "0") == "1"
    )
    torchdynamo_disabled = os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1"
    torch_compile_disabled = os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
    if unsloth_compile_disabled != bool(unsloth_compile_common.UNSLOTH_COMPILE_DISABLE):
        raise RuntimeError(
            "UNSLOTH_COMPILE_DISABLE was not applied before importing Unsloth"
        )
    compile_policy = configure_inductor_compile_threads(
        torch,
        unsloth_compile_common,
        requested_compile_threads,
    )
    print(
        "[progress] event=compile_policy "
        f"torch_inductor_compile_threads={compile_policy['torch_inductor_compile_threads']} "
        f"unsloth_compile_threads={compile_policy['unsloth_compile_threads']} "
        f"unsloth_compile_disabled={int(unsloth_compile_disabled)} "
        f"torchdynamo_disabled={int(torchdynamo_disabled)} "
        f"torch_compile_disabled={int(torch_compile_disabled)}",
        flush=True,
    )

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
    gradient_checkpointing = resolve_gradient_checkpointing(a.gradient_checkpointing)
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
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=0,
        )

    hybrid_checkpoint_policy = None
    bounded_unsloth_host_buffer_policy = None
    unsloth_gradient_checkpointing = None
    if a.gradient_checkpointing == "hybrid":
        import torch.utils.checkpoint as checkpoint_module

        hybrid_checkpoint_policy = configure_hybrid_gradient_checkpointing(
            model,
            checkpoint_module,
            standard_stride=10,
        )
        if hybrid_checkpoint_policy != {
            "text_layers": 60,
            "offloaded_layers": 54,
            "standard_layers": 6,
            "standard_stride": 10,
        }:
            raise RuntimeError(
                f"unexpected Gemma4 hybrid checkpoint policy: {hybrid_checkpoint_policy}"
            )
        print(
            "[progress] event=gradient_checkpoint_policy "
            f"mode=hybrid text_layers={hybrid_checkpoint_policy['text_layers']} "
            f"offloaded_layers={hybrid_checkpoint_policy['offloaded_layers']} "
            f"standard_layers={hybrid_checkpoint_policy['standard_layers']} "
            f"standard_stride={hybrid_checkpoint_policy['standard_stride']}",
            flush=True,
        )
    elif a.gradient_checkpointing == "bounded_unsloth":
        if not unsloth_double_buffer_disabled:
            raise RuntimeError(
                "bounded Unsloth checkpointing requires "
                "UNSLOTH_DISABLE_DOUBLE_BUFFER=1 before importing Unsloth"
            )
        import unsloth_zoo.gradient_checkpointing as unsloth_gradient_checkpointing

        initial_recycle = recycle_bounded_unsloth_host_buffers(
            torch,
            unsloth_gradient_checkpointing,
        )
        bounded_unsloth_host_buffer_policy = {
            "buffer_count": initial_recycle["buffer_count"],
            "initial_buffer_elements": initial_recycle["initial_buffer_elements"],
            "pageable_buffers": initial_recycle["pageable_buffers"],
            "recycle_after_backward": True,
            "cuda_synchronized": initial_recycle["cuda_synchronized"],
            "host_cache_drained": initial_recycle["host_cache_drained"],
        }
        print(
            "[progress] event=unsloth_host_buffer_recycle phase=initial "
            f"retained_bytes={initial_recycle['retained_bytes']} "
            f"pinned_buffers={initial_recycle['pinned_buffers']} "
            f"pageable_buffers={initial_recycle['pageable_buffers']} "
            "cuda_synchronized=1 host_cache_drained=1",
            flush=True,
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

    ds = load_training_dataset(a.data)
    print(f"[data] {len(ds)} sft examples", flush=True)
    selective_assistant_loss = any(
        message.get("role") == "assistant" and "loss" in message
        for example in ds
        for message in example.get("messages", [])
    )
    if selective_assistant_loss:
        print("[data] selective assistant-turn loss flags detected", flush=True)
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
                "gradient_checkpointing": a.gradient_checkpointing,
                "hybrid_checkpoint_policy": hybrid_checkpoint_policy,
                "bounded_unsloth_host_buffer_policy": bounded_unsloth_host_buffer_policy,
                "attn_implementation": a.attn_implementation,
                "attention_configs_changed": attention_configs_changed,
                "sm120_attn": a.sm120_attn,
                "debug_hf_flex_routing": a.debug_hf_flex_routing,
                "full_transcript_loss": a.full_transcript_loss,
                "assistant_content_only_loss": a.assistant_content_only_loss,
                "selective_assistant_loss": selective_assistant_loss,
                "unsloth_compile_disabled": unsloth_compile_disabled,
                "unsloth_double_buffer_disabled": unsloth_double_buffer_disabled,
                "torchdynamo_disabled": torchdynamo_disabled,
                "torch_compile_disabled": torch_compile_disabled,
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
            spans, _fallback_used = supervised_assistant_turn_spans(tokenizer, messages, text)

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
            num_proc=1,
            desc="assistant-tokenize",
        )
        before = len(ds)
        ds = ds.filter(lambda ex: ex["supervised_tokens"] > 0, num_proc=1)
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
        dataset_num_proc=1,
        max_seq_length=a.max_seq,
        ignore_data_skip=ignore_data_skip_for_run(resume=a.resume),
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
            if unsloth_gradient_checkpointing is not None:
                recycle = recycle_bounded_unsloth_host_buffers(
                    torch,
                    unsloth_gradient_checkpointing,
                )
                print(
                    "[progress] event=unsloth_host_buffer_recycle "
                    "phase=gradient_microstep "
                    f"retained_bytes={recycle['retained_bytes']} "
                    f"pinned_buffers={recycle['pinned_buffers']} "
                    f"pageable_buffers={recycle['pageable_buffers']} "
                    "cuda_synchronized=1 host_cache_drained=1",
                    flush=True,
                )
            cache = release_cuda_cache(torch)
            if cache is not None:
                print(
                    "[progress] event=cuda_cache_release "
                    f"allocated_bytes={cache['allocated_bytes']} "
                    f"reserved_before_bytes={cache['reserved_before_bytes']} "
                    f"reserved_after_bytes={cache['reserved_after_bytes']}",
                    flush=True,
                )
            self._report("gradient_microstep", state)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.completed_microsteps += 1
            now = time.monotonic()
            step_wall_seconds = 0.0 if self.last_optimizer_step_at is None else now - self.last_optimizer_step_at
            self.last_optimizer_step_at = now
            if should_force_checkpoint(global_step=int(state.global_step), save_steps=int(a.save_steps)):
                control.should_save = True
            if unsloth_gradient_checkpointing is not None:
                recycle = recycle_bounded_unsloth_host_buffers(
                    torch,
                    unsloth_gradient_checkpointing,
                )
                print(
                    "[progress] event=unsloth_host_buffer_recycle "
                    "phase=optimizer_step "
                    f"retained_bytes={recycle['retained_bytes']} "
                    f"pinned_buffers={recycle['pinned_buffers']} "
                    f"pageable_buffers={recycle['pageable_buffers']} "
                    "cuda_synchronized=1 host_cache_drained=1",
                    flush=True,
                )
            cache = release_cuda_cache(torch)
            if cache is not None:
                print(
                    "[progress] event=cuda_cache_release "
                    f"allocated_bytes={cache['allocated_bytes']} "
                    f"reserved_before_bytes={cache['reserved_before_bytes']} "
                    f"reserved_after_bytes={cache['reserved_after_bytes']}",
                    flush=True,
                )
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
