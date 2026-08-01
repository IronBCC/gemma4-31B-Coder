#!/usr/bin/env python3
"""Merge a LoRA adapter into a base checkpoint with bounded host memory."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from phaseD_sft.atomic_checkpoint_publish import publish_after_audit


COPY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def base_key_for(adapter_key: str) -> str | None:
    if ".lora_A" not in adapter_key and ".lora_B" not in adapter_key:
        return None
    key = adapter_key
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    for source, target in (
        (".lora_A.default.weight", ".weight"),
        (".lora_B.default.weight", ".weight"),
        (".lora_A.weight", ".weight"),
        (".lora_B.weight", ".weight"),
    ):
        key = key.replace(source, target)
    return key


def load_adapter_pairs(
    adapter_dir: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict, float]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(
        str(adapter_dir / "adapter_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        for key in handle.keys():
            base_key = base_key_for(key)
            if base_key is None:
                continue
            side = "A" if ".lora_A" in key else "B"
            pairs.setdefault(base_key, {})[side] = handle.get_tensor(key)
    for base_key, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise SystemExit(
                f"adapter tensor missing its pair for {base_key}: has {sorted(pair)}"
            )
    return pairs, config, scale


def proc_memory_kib(field: str) -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(f"{field}:"):
                return int(line.split()[1])
    raise RuntimeError(f"/proc/self/status is missing {field}")


def assert_peak_rss(max_rss_bytes: int, phase: str) -> int:
    peak_bytes = proc_memory_kib("VmHWM") * 1024
    if peak_bytes > max_rss_bytes:
        raise MemoryError(
            f"merge peak RSS exceeded at {phase}: "
            f"{peak_bytes / (1 << 30):.2f} GiB > "
            f"{max_rss_bytes / (1 << 30):.2f} GiB"
        )
    return peak_bytes


def release_host_memory() -> None:
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def audit_merged_checkpoint(
    checkpoint: Path,
    *,
    audit_path: Path,
    expected_architecture: str,
    expected_tensors: int,
    expected_vision: int,
    max_rss_bytes: int,
) -> dict[str, object]:
    config = json.loads((checkpoint / "config.json").read_text())
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )
    weight_map = index["weight_map"]
    if config.get("architectures") != [expected_architecture]:
        raise ValueError(
            "merged checkpoint architecture mismatch: "
            f"{config.get('architectures')!r}"
        )
    if len(weight_map) != expected_tensors:
        raise ValueError(
            "merged checkpoint tensor count mismatch: "
            f"{len(weight_map)} != {expected_tensors}"
        )
    vision = sum("vision" in name for name in weight_map)
    if vision != expected_vision:
        raise ValueError(
            "merged checkpoint vision tensor count mismatch: "
            f"{vision} != {expected_vision}"
        )

    actual_tensors: set[str] = set()
    misplaced_tensors: list[str] = []
    nonfinite_tensors: list[str] = []
    for shard_name in sorted(set(weight_map.values())):
        shard_path = checkpoint / shard_name
        if not shard_path.is_file():
            raise ValueError(f"merged checkpoint shard missing: {shard_path}")
        with safe_open(
            str(shard_path),
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                actual_tensors.add(key)
                if weight_map.get(key) != shard_name:
                    misplaced_tensors.append(key)
                tensor_slice = handle.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                if not shape:
                    chunks = (handle.get_tensor(key),)
                else:
                    row_width = (
                        math.prod(shape[1:]) if len(shape) > 1 else 1
                    )
                    rows_per_chunk = max(
                        1,
                        8_000_000 // max(row_width, 1),
                    )
                    chunks = (
                        tensor_slice[
                            start : min(
                                start + rows_per_chunk,
                                shape[0],
                            )
                        ]
                        for start in range(
                            0,
                            shape[0],
                            rows_per_chunk,
                        )
                    )
                for chunk in chunks:
                    if (
                        (
                            torch.is_floating_point(chunk)
                            or torch.is_complex(chunk)
                        )
                        and not bool(torch.isfinite(chunk).all())
                    ):
                        nonfinite_tensors.append(key)
                        break
                assert_peak_rss(max_rss_bytes, f"audit_{len(actual_tensors)}")

    missing_tensors = sorted(set(weight_map) - actual_tensors)
    unexpected_tensors = sorted(actual_tensors - set(weight_map))
    if (
        missing_tensors
        or unexpected_tensors
        or misplaced_tensors
        or nonfinite_tensors
    ):
        raise ValueError(
            "merged checkpoint audit failed: "
            f"missing={missing_tensors[:3]} "
            f"unexpected={unexpected_tensors[:3]} "
            f"misplaced={misplaced_tensors[:3]} "
            f"nonfinite={nonfinite_tensors[:3]}"
        )
    audit: dict[str, object] = {
        "schema_version": 1,
        "architecture": expected_architecture,
        "expected_tensors": expected_tensors,
        "actual_tensors": len(actual_tensors),
        "expected_vision": expected_vision,
        "actual_vision": vision,
        "missing_tensors": missing_tensors,
        "unexpected_tensors": unexpected_tensors,
        "misplaced_tensors": misplaced_tensors,
        "nonfinite_tensors": nonfinite_tensors,
        "complete": True,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-gb", type=float, default=3.0)
    parser.add_argument("--max-rss-gb", type=float, default=12.0)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-architecture")
    parser.add_argument("--audit-tensors", type=int)
    parser.add_argument("--audit-vision", type=int)
    parser.add_argument("--audit-filename")
    args = parser.parse_args()
    if args.group_gb <= 0 or args.max_rss_gb <= 0:
        raise SystemExit("--group-gb and --max-rss-gb must be positive")
    audit_values = (
        args.audit_architecture,
        args.audit_tensors,
        args.audit_vision,
        args.audit_filename,
    )
    if any(value is not None for value in audit_values) and not all(
        value is not None for value in audit_values
    ):
        raise SystemExit("all --audit-* arguments must be provided together")
    if args.audit_tensors is not None and args.audit_tensors <= 0:
        raise SystemExit("--audit-tensors must be positive")
    if args.audit_vision is not None and args.audit_vision < 0:
        raise SystemExit("--audit-vision must be nonnegative")
    if args.audit_filename is not None and (
        not args.audit_filename
        or Path(args.audit_filename).name != args.audit_filename
    ):
        raise SystemExit("--audit-filename must be a basename")

    base = Path(args.base)
    adapter = Path(args.adapter)
    out = Path(args.out)
    max_rss_bytes = int(args.max_rss_gb * (1 << 30))
    adapter_pairs, config, scale = load_adapter_pairs(adapter)
    assert_peak_rss(max_rss_bytes, "adapter_load")
    print(
        f"[merge] adapter r={config['r']} alpha={config['lora_alpha']} "
        f"scale={scale:g} matrices={len(adapter_pairs)}",
        flush=True,
    )

    index_path = base / "model.safetensors.index.json"
    input_shards = (
        sorted(
            set(
                json.loads(index_path.read_text())["weight_map"].values()
            )
        )
        if index_path.exists()
        else ["model.safetensors"]
    )

    def output_name(key: str) -> str | None:
        if not args.text_only:
            return key
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        if key == "lm_head.weight" or key.startswith("lm_head."):
            return key
        return None

    plan: list[tuple[str, str, str]] = []
    sizes: dict[tuple[str, str], int] = {}
    dropped = 0
    for shard in input_shards:
        with safe_open(
            str(base / shard),
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                merged_key = output_name(key)
                if merged_key is None:
                    dropped += 1
                    continue
                tensor_slice = handle.get_slice(key)
                elements = 1
                for dimension in tensor_slice.get_shape():
                    elements *= dimension
                sizes[(shard, key)] = elements * 2
                plan.append((shard, key, merged_key))
    if args.text_only:
        print(
            f"[merge] text-only: dropped {dropped} vision/projector tensors, "
            f"keeping {len(plan)} language tensors",
            flush=True,
        )

    base_keys = {key for _, key, _ in plan}
    missing_adapter_keys = sorted(set(adapter_pairs) - base_keys)
    if missing_adapter_keys:
        raise SystemExit(
            f"[merge] FAIL: {len(missing_adapter_keys)} adapter matrices "
            f"match no base tensor: {missing_adapter_keys[:3]}"
        )

    budget = int(args.group_gb * (1 << 30))
    groups: list[list[tuple[str, str, str]]] = []
    current: list[tuple[str, str, str]] = []
    current_bytes = 0
    for item in plan:
        size = sizes[(item[0], item[1])]
        if current and current_bytes + size > budget:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        groups.append(current)
    total_gib = sum(sizes.values()) / (1 << 30)
    print(
        f"[merge] {len(plan)} tensors, {total_gib:.1f}GB -> "
        f"{len(groups)} output shards of <= {args.group_gb}GB",
        flush=True,
    )

    if args.dry_run:
        print(
            f"[merge] dry-run OK — {len(adapter_pairs)}/{len(adapter_pairs)} "
            "adapter matrices matched, nothing written",
            flush=True,
        )
        return
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    staging = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"staging output already exists: {staging}")

    staging.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    applied = 0
    output_shards = len(groups)
    try:
        for group_index, group in enumerate(groups, 1):
            filename = (
                f"model-{group_index:05d}-of-"
                f"{output_shards:05d}.safetensors"
            )
            tensors: dict[str, torch.Tensor] = {}
            by_shard: dict[str, list[tuple[str, str]]] = {}
            for shard, key, merged_key in group:
                by_shard.setdefault(shard, []).append((key, merged_key))
            for shard, keys in by_shard.items():
                with safe_open(
                    str(base / shard),
                    framework="pt",
                    device="cpu",
                ) as handle:
                    for key, merged_key in keys:
                        tensor = handle.get_tensor(key)
                        pair = adapter_pairs.get(key)
                        if pair is not None:
                            adapter_a = pair["A"].to(torch.float32)
                            adapter_b = pair["B"].to(torch.float32)
                            delta = adapter_b @ adapter_a
                            if delta.shape != tensor.shape:
                                raise SystemExit(
                                    f"shape mismatch {key}: "
                                    f"base {tuple(tensor.shape)} vs "
                                    f"delta {tuple(delta.shape)}"
                                )
                            merged = tensor.to(torch.float32)
                            merged.add_(delta, alpha=scale)
                            tensor = merged.to(tensor.dtype)
                            applied += 1
                            del adapter_a, adapter_b, delta, merged
                        tensors[merged_key] = tensor
                        weight_map[merged_key] = filename
                        assert_peak_rss(
                            max_rss_bytes,
                            f"group_{group_index}_tensor_{len(tensors)}",
                        )
            save_file(
                tensors,
                str(staging / filename),
                metadata={"format": "pt"},
            )
            del tensors
            release_host_memory()
            peak_bytes = assert_peak_rss(
                max_rss_bytes,
                f"group_{group_index}_saved",
            )
            print(
                f"[merge] wrote {filename} ({len(group)} tensors, "
                f"{applied}/{len(adapter_pairs)} deltas, "
                f"peak_rss_gib={peak_bytes / (1 << 30):.2f})",
                flush=True,
            )

        if applied != len(adapter_pairs):
            raise SystemExit(
                f"[merge] FAIL: applied {applied} of "
                f"{len(adapter_pairs)} adapter matrices"
            )
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum(sizes.values())},
                    "weight_map": weight_map,
                }
            ),
            encoding="utf-8",
        )
        for name in COPY_FILES:
            if (base / name).exists():
                shutil.copy2(base / name, staging / name)
        for name in (
            "chat_template.jinja",
            "tokenizer_config.json",
            "tokenizer.json",
        ):
            if (adapter / name).exists():
                shutil.copy2(adapter / name, staging / name)
        if args.audit_filename is None:
            publish_after_audit(staging, out, lambda _candidate: None)
        else:
            publish_after_audit(
                staging,
                out,
                lambda candidate: audit_merged_checkpoint(
                    candidate,
                    audit_path=candidate / args.audit_filename,
                    expected_architecture=args.audit_architecture,
                    expected_tensors=args.audit_tensors,
                    expected_vision=args.audit_vision,
                    max_rss_bytes=max_rss_bytes,
                ),
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[merge] done -> {out} ({applied}/{len(adapter_pairs)} matrices, "
        f"{output_shards} shards, {len(os.listdir(out))} files)",
        flush=True,
    )


if __name__ == "__main__":
    main()
