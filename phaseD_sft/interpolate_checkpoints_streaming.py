#!/usr/bin/env python3
"""Interpolate two compatible sharded checkpoints with bounded host memory."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from phaseD_sft.atomic_checkpoint_publish import publish_after_audit
from phaseD_sft.merge_lora_streaming import COPY_FILES


_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_TORCH_DTYPES = {
    "BOOL": torch.bool,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F64": torch.float64,
}

_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _weight_map(model: Path) -> dict[str, str]:
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    if index.is_file():
        value = _read_object(index).get("weight_map")
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"invalid model weight map: {index}")
        result = {}
        for key, shard in value.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(shard, str)
                or Path(shard).name != shard
            ):
                raise ValueError(f"invalid model weight map entry: {key!r}")
            result[key] = shard
        return result
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise ValueError(f"model weights are missing: {model}")


def _model_contract(
    model: Path,
    copied_files: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    model = model.resolve()
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"model config is missing: {config}")
    weight_map = _weight_map(model)
    shards = sorted({model / shard for shard in weight_map.values()})
    if any(not shard.is_file() for shard in shards):
        raise ValueError(f"model weight shard is missing: {model}")
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    return {
        "model_path": str(model),
        "model_config_sha256": _sha256(config),
        "model_index_sha256": _sha256(index) if index.is_file() else None,
        "model_safetensors_sha256": (
            _sha256(single) if single.is_file() else None
        ),
        "model_artifacts": [_binding(shard) for shard in shards],
        "copied_file_bindings": [
            _binding(model / name) for name in copied_files
        ],
    }


def _peak_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _assert_peak_rss(limit: int, phase: str) -> int:
    actual = _peak_rss_bytes()
    if actual > limit:
        raise MemoryError(
            f"interpolation peak RSS exceeded at {phase}: "
            f"{actual / (1 << 30):.2f} GiB > {limit / (1 << 30):.2f} GiB"
        )
    return actual


def _release_host_memory() -> None:
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def _tensor_specs(
    model: Path,
    weight_map: Mapping[str, str],
) -> dict[str, tuple[tuple[int, ...], str, int]]:
    specs = {}
    actual_keys = set()
    for shard_name in sorted(set(weight_map.values())):
        shard_path = model / shard_name
        if not shard_path.is_file():
            raise ValueError(f"model weight shard is missing: {shard_path}")
        with safe_open(
            str(shard_path), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                actual_keys.add(key)
                if weight_map.get(key) != shard_name:
                    raise ValueError(f"tensor is assigned to the wrong shard: {key}")
                tensor_slice = handle.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                if dtype not in _DTYPE_BYTES:
                    raise ValueError(f"unsupported tensor dtype {dtype}: {key}")
                specs[key] = (
                    shape,
                    dtype,
                    math.prod(shape) * _DTYPE_BYTES[dtype],
                )
    if actual_keys != set(weight_map):
        raise ValueError(
            "weight index and shard keys differ: "
            f"missing={sorted(set(weight_map) - actual_keys)[:3]} "
            f"unexpected={sorted(actual_keys - set(weight_map))[:3]}"
        )
    return specs


def _interpolate_tensor_streaming(
    *,
    anchor_path: Path,
    candidate_path: Path,
    key: str,
    shape: tuple[int, ...],
    dtype: str,
    alpha: float,
    chunk_bytes: int,
    max_rss_bytes: int,
) -> torch.Tensor:
    output = torch.empty(shape, dtype=_TORCH_DTYPES[dtype], device="cpu")
    floating = dtype in _FLOAT_DTYPES
    with safe_open(
        str(anchor_path), framework="pt", device="cpu"
    ) as anchor_handle, safe_open(
        str(candidate_path), framework="pt", device="cpu"
    ) as candidate_handle:
        if not shape:
            anchor_tensor = anchor_handle.get_tensor(key)
            candidate_tensor = candidate_handle.get_tensor(key)
            if floating:
                anchor_float = anchor_tensor.to(torch.float32)
                candidate_float = candidate_tensor.to(torch.float32)
                candidate_float.sub_(anchor_float)
                anchor_float.add_(candidate_float, alpha=alpha)
                output.copy_(anchor_float.to(_TORCH_DTYPES[dtype]))
            else:
                if not torch.equal(anchor_tensor, candidate_tensor):
                    raise ValueError(f"non-floating tensor mismatch: {key}")
                output.copy_(anchor_tensor)
            _assert_peak_rss(max_rss_bytes, f"tensor_{key}_scalar")
            return output

        anchor_slice = anchor_handle.get_slice(key)
        candidate_slice = candidate_handle.get_slice(key)
        row_bytes = max(
            1,
            math.prod(shape[1:]) * _DTYPE_BYTES[dtype],
        )
        rows_per_chunk = max(1, chunk_bytes // row_bytes)
        for start in range(0, shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, shape[0])
            anchor_chunk = anchor_slice[start:stop]
            candidate_chunk = candidate_slice[start:stop]
            if floating:
                anchor_float = anchor_chunk.to(torch.float32)
                candidate_float = candidate_chunk.to(torch.float32)
                candidate_float.sub_(anchor_float)
                anchor_float.add_(candidate_float, alpha=alpha)
                blended = anchor_float.to(_TORCH_DTYPES[dtype])
                output[start:stop].copy_(blended)
                del anchor_float, candidate_float, blended
            else:
                if not torch.equal(anchor_chunk, candidate_chunk):
                    raise ValueError(f"non-floating tensor mismatch: {key}")
                output[start:stop].copy_(anchor_chunk)
            del anchor_chunk, candidate_chunk
            _assert_peak_rss(max_rss_bytes, f"tensor_{key}_{stop}")
    return output


def _compatible_copy_files(anchor: Path, candidate: Path) -> list[str]:
    copied = []
    for name in COPY_FILES:
        left = anchor / name
        right = candidate / name
        if left.is_file() != right.is_file():
            raise ValueError(f"checkpoint copied-file presence mismatch: {name}")
        if left.is_file():
            if left.read_bytes() != right.read_bytes():
                raise ValueError(f"checkpoint copied-file mismatch: {name}")
            copied.append(name)
    if "config.json" not in copied:
        raise ValueError("checkpoint config.json is missing")
    return sorted(copied)


def _normalized_output_contract(
    staging: Path,
    output: Path,
    copied_files: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    contract = _model_contract(staging, copied_files)
    contract["model_path"] = str(output.resolve())
    for artifact in (
        contract["model_artifacts"] + contract["copied_file_bindings"]
    ):
        artifact["path"] = str(output.resolve() / Path(artifact["path"]).name)
    return contract


def _audit_output(
    staging: Path,
    *,
    output: Path,
    expected_specs: Mapping[str, tuple[tuple[int, ...], str, int]],
    expected_manifest: Mapping[str, Any],
    max_rss_bytes: int,
) -> None:
    weight_map = _weight_map(staging)
    if set(weight_map) != set(expected_specs):
        raise ValueError("interpolated output tensor keys changed")
    specs = _tensor_specs(staging, weight_map)
    if specs != dict(expected_specs):
        raise ValueError("interpolated output tensor specs changed")
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(
            str(staging / shard_name), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                tensor_slice = handle.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                if not shape:
                    chunks = (handle.get_tensor(key),)
                else:
                    row_width = math.prod(shape[1:]) if len(shape) > 1 else 1
                    rows = max(1, 8_000_000 // max(row_width, 1))
                    chunks = (
                        tensor_slice[start : min(start + rows, shape[0])]
                        for start in range(0, shape[0], rows)
                    )
                for chunk in chunks:
                    if torch.is_floating_point(chunk) and not bool(
                        torch.isfinite(chunk).all()
                    ):
                        raise ValueError(f"non-finite interpolated tensor: {key}")
                _assert_peak_rss(max_rss_bytes, f"audit_{key}")
    manifest = _read_object(staging / "interpolation_manifest.json")
    if manifest != dict(expected_manifest):
        raise ValueError("interpolation manifest changed before publication")
    if manifest["output"] != _normalized_output_contract(
        staging,
        output,
        manifest["copied_files"],
    ):
        raise ValueError("interpolation output contract changed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--group-gb", type=float, default=3.0)
    parser.add_argument("--max-rss-gb", type=float, default=14.0)
    parser.add_argument(
        "--manifest-name", default="interpolation_manifest.json"
    )
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be strictly between zero and one")
    if args.group_gb <= 0.0 or args.max_rss_gb <= 0.0:
        raise SystemExit("--group-gb and --max-rss-gb must be positive")
    if (
        not args.manifest_name
        or Path(args.manifest_name).name != args.manifest_name
        or args.manifest_name != "interpolation_manifest.json"
    ):
        raise SystemExit(
            "--manifest-name must be interpolation_manifest.json"
        )
    return args


def main() -> None:
    args = _parse_args()
    anchor = args.anchor.resolve()
    candidate = args.candidate.resolve()
    output = args.out.resolve()
    if anchor == candidate:
        raise SystemExit("anchor and candidate checkpoints must differ")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"staging output already exists: {staging}")

    anchor_map = _weight_map(anchor)
    candidate_map = _weight_map(candidate)
    if set(anchor_map) != set(candidate_map):
        raise SystemExit(
            "checkpoint tensor-key mismatch: "
            f"anchor_only={sorted(set(anchor_map) - set(candidate_map))[:3]} "
            f"candidate_only={sorted(set(candidate_map) - set(anchor_map))[:3]}"
        )
    anchor_specs = _tensor_specs(anchor, anchor_map)
    candidate_specs = _tensor_specs(candidate, candidate_map)
    mismatched_specs = [
        key
        for key in sorted(anchor_specs)
        if anchor_specs[key][:2] != candidate_specs[key][:2]
    ]
    if mismatched_specs:
        raise SystemExit(
            f"checkpoint tensor shape/dtype mismatch: {mismatched_specs[:3]}"
        )
    copied_files = _compatible_copy_files(anchor, candidate)
    max_rss_bytes = int(args.max_rss_gb * (1 << 30))
    _assert_peak_rss(max_rss_bytes, "preflight")

    budget = int(args.group_gb * (1 << 30))
    oversized = [
        key for key, spec in sorted(anchor_specs.items()) if spec[2] > budget
    ]
    if oversized:
        raise SystemExit(
            "tensor exceeds --group-gb shard budget: "
            f"{oversized[:3]}"
        )
    chunk_bytes = min(budget, 256 * 1024 * 1024)
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in sorted(anchor_specs):
        size = anchor_specs[key][2]
        if current and current_bytes + size > budget:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    if current:
        groups.append(current)

    staging.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    floating_tensors = 0
    copied_nonfloating = 0
    peak_rss = _peak_rss_bytes()
    try:
        for index, group in enumerate(groups, 1):
            filename = (
                f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            )
            tensors = {}
            for key in group:
                shape, dtype, _ = anchor_specs[key]
                tensor = _interpolate_tensor_streaming(
                    anchor_path=anchor / anchor_map[key],
                    candidate_path=candidate / candidate_map[key],
                    key=key,
                    shape=shape,
                    dtype=dtype,
                    alpha=args.alpha,
                    chunk_bytes=chunk_bytes,
                    max_rss_bytes=max_rss_bytes,
                )
                if dtype in _FLOAT_DTYPES:
                    floating_tensors += 1
                else:
                    copied_nonfloating += 1
                tensors[key] = tensor
                weight_map[key] = filename
                peak_rss = max(
                    peak_rss,
                    _assert_peak_rss(max_rss_bytes, f"tensor_{key}"),
                )
            save_file(
                tensors,
                str(staging / filename),
                metadata={"format": "pt"},
            )
            del tensors
            _release_host_memory()
            peak_rss = max(
                peak_rss,
                _assert_peak_rss(max_rss_bytes, f"shard_{index}"),
            )
        total_size = sum(spec[2] for spec in anchor_specs.values())
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for name in copied_files:
            shutil.copy2(anchor / name, staging / name)
        output_contract = _normalized_output_contract(
            staging,
            output,
            copied_files,
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "checkpoint_weight_interpolation",
            "status": "complete",
            "alpha": args.alpha,
            "equation": "anchor + alpha * (candidate - anchor)",
            "anchor": _model_contract(anchor, copied_files),
            "candidate": _model_contract(candidate, copied_files),
            "output": output_contract,
            "tensor_count": len(anchor_specs),
            "floating_tensors": floating_tensors,
            "copied_nonfloating_tensors": copied_nonfloating,
            "total_tensor_bytes": total_size,
            "output_shards": len(groups),
            "copied_files": copied_files,
            "copied_file_bindings": output_contract[
                "copied_file_bindings"
            ],
            "peak_rss_bytes": peak_rss,
        }
        (staging / args.manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_after_audit(
            staging,
            output,
            lambda path: _audit_output(
                path,
                output=output,
                expected_specs=anchor_specs,
                expected_manifest=manifest,
                max_rss_bytes=max_rss_bytes,
            ),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"interpolation complete: tensors={len(anchor_specs)} "
        f"shards={len(groups)} alpha={args.alpha:g} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
