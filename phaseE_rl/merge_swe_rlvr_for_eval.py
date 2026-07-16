#!/usr/bin/env python3
"""Materialize a frozen-base + SFT + RLVR LoRA model for vLLM evaluation."""

import argparse
import json
from pathlib import Path


def lora_key_to_module_path(key: str) -> str:
    """Map a saved PEFT LoRA key onto the grafted Gemma-4 text model."""
    suffix = ".lora_A.weight"
    if not key.startswith("base_model.model.") or not key.endswith(suffix):
        raise ValueError(f"not a LoRA-A key: {key}")
    path = key.removeprefix("base_model.model.").removesuffix(suffix)
    return path.replace("model.language_model.", "model.", 1)


def lora_delta(b_matrix, a_matrix, scale: float, device):
    """Compute a LoRA update where the target linear layer resides."""
    import torch

    return (b_matrix.to(device=device, dtype=torch.float32) @
            a_matrix.to(device=device, dtype=torch.float32)) * scale


def merge_adapter(model, adapter_dir: Path) -> int:
    """Apply one LoRA adapter into a model in-place and return merged pairs."""
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    state = load_file(adapter_dir / "adapter_model.safetensors")
    a_keys = [key for key in state if key.endswith(".lora_A.weight")]
    if not a_keys:
        raise ValueError(f"no LoRA matrices in {adapter_dir}")

    merged = 0
    with torch.no_grad():
        for a_key in a_keys:
            b_key = a_key.removesuffix(".lora_A.weight") + ".lora_B.weight"
            if b_key not in state:
                raise KeyError(f"missing matching B matrix for {a_key}")
            module = model.get_submodule(lora_key_to_module_path(a_key))
            linear = getattr(module, "linear", module)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"target is not nn.Linear: {a_key} -> {type(linear)}")
            delta = lora_delta(state[b_key], state[a_key], scale, linear.weight.device)
            if delta.shape != linear.weight.shape:
                raise ValueError(
                    f"shape mismatch for {a_key}: delta={tuple(delta.shape)} "
                    f"weight={tuple(linear.weight.shape)}"
                )
            linear.weight.add_(delta.to(dtype=linear.weight.dtype, device=linear.weight.device))
            merged += 1
    return merged


def merge_adapters(model, adapter_dirs: list[Path]) -> list[tuple[Path, int]]:
    """Merge adapters in order, rejecting an incomplete merge at each boundary."""
    results: list[tuple[Path, int]] = []
    for adapter_dir in adapter_dirs:
        count = merge_adapter(model, adapter_dir)
        if count <= 100:
            raise RuntimeError(
                f"only {count} matrices merged from {adapter_dir}; expected >100"
            )
        results.append((adapter_dir, count))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    ap.add_argument(
        "--adapter",
        action="append",
        default=None,
        help="repeatable; merge in CLI order (for example: SFT, GRPO, vGRPO)",
    )
    ap.add_argument(
        "--sft-adapter",
        help="deprecated compatibility alias; maps to the first ordered adapter",
    )
    ap.add_argument(
        "--rl-adapter",
        help="deprecated compatibility alias; maps to the second ordered adapter",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-size", default="20GB")
    return ap


def resolve_adapter_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    """Return one unambiguous ordered adapter list, preserving the old CLI."""
    ordered = args.adapter or []
    legacy = [args.sft_adapter, args.rl_adapter]
    if ordered:
        if any(legacy):
            parser.error("--adapter cannot be combined with deprecated adapter aliases")
        return [Path(adapter) for adapter in ordered]
    if all(legacy):
        print("[merge] DEPRECATED: --sft-adapter/--rl-adapter; use repeatable --adapter", flush=True)
        return [Path(adapter) for adapter in legacy]
    if any(legacy):
        parser.error("legacy mode requires both --sft-adapter and --rl-adapter")
    parser.error("provide at least one --adapter or both legacy adapter aliases")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    adapter_dirs = resolve_adapter_paths(args, parser)

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing merged model: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration

    full_config = AutoConfig.from_pretrained(args.base)
    multimodal = Gemma4ForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    with init_empty_weights():
        model = Gemma4ForCausalLM(full_config.text_config)
    model.model = (
        multimodal.model.language_model
        if hasattr(multimodal.model, "language_model")
        else multimodal.language_model
    )
    model.lm_head = multimodal.lm_head
    model.config = full_config.text_config
    model.generation_config = multimodal.generation_config
    del multimodal
    torch.cuda.empty_cache()

    meta = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if meta:
        raise RuntimeError(f"graft left meta parameters: {meta[:3]}")

    merge_results = merge_adapters(model, adapter_dirs)
    counts = ", ".join(f"{path}={count}" for path, count in merge_results)
    print(f"[merge] adapters: {counts} matrices; saving {out}", flush=True)
    model.save_pretrained(
        out, safe_serialization=True, max_shard_size=args.max_shard_size
    )
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print(f"[merge] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
