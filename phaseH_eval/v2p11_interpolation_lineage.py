#!/usr/bin/env python3
"""Validate the exact v2.11r4 interpolation lineage before evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseD_sft.checkpoint_copy_compatibility import compatible_copy_files


EXPECTED_ALPHA = 0.25
EXPECTED_CANDIDATE_NAME = "teacher_sft_v2p11r4_blend25"
EXPECTED_EQUATION = "anchor + alpha * (candidate - anchor)"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _weight_contract(model: Path) -> tuple[dict[str, Any], dict[str, str]]:
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    if index.is_file():
        index_value = _read_object(index)
        raw_map = index_value.get("weight_map")
        if not isinstance(raw_map, Mapping) or not raw_map:
            raise ValueError(f"invalid model weight map: {index}")
        weight_map = {}
        for key, shard in raw_map.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(shard, str)
                or Path(shard).name != shard
            ):
                raise ValueError(f"invalid model weight map entry: {key!r}")
            weight_map[key] = shard
        contract = {
            "model_index_sha256": _sha256(index),
            "model_safetensors_sha256": None,
        }
    elif single.is_file():
        weight_map = {"__single__": single.name}
        contract = {
            "model_index_sha256": None,
            "model_safetensors_sha256": _sha256(single),
        }
    else:
        raise ValueError(f"model weights are missing: {model}")
    return contract, weight_map


def _model_contract(
    model: Path,
    copied_files: Sequence[str],
) -> dict[str, Any]:
    model = model.resolve()
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"model config is missing: {config}")
    weight_contract, weight_map = _weight_contract(model)
    shards = sorted({model / shard for shard in weight_map.values()})
    if any(not shard.is_file() for shard in shards):
        raise ValueError(f"model weight shard is missing: {model}")
    return {
        "model_path": str(model),
        "model_config_sha256": _sha256(config),
        **weight_contract,
        "model_artifacts": [_binding(shard) for shard in shards],
        "copied_file_bindings": [
            _binding(model / name) for name in copied_files
        ],
    }


def _compatible_copy_files(
    anchor: Path,
    source: Path,
    output: Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    try:
        copied, controlled_differences = compatible_copy_files(
            anchor,
            source,
            COPY_FILES,
        )
    except ValueError as exc:
        name = str(exc).rsplit(": ", 1)[-1]
        raise ValueError(f"copied checkpoint files differ: {name}") from exc
    for name in copied:
        output_path = output / name
        if (
            not output_path.is_file()
            or output_path.read_bytes() != (anchor / name).read_bytes()
        ):
            raise ValueError(f"copied checkpoint files differ: {name}")
    return copied, controlled_differences


def validate_interpolation_lineage(
    *,
    manifest_path: Path,
    anchor_model_path: Path,
    source_model_path: Path,
    output_model_path: Path,
    candidate_name: str,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    anchor = Path(anchor_model_path).resolve()
    source = Path(source_model_path).resolve()
    output = Path(output_model_path).resolve()
    if candidate_name != EXPECTED_CANDIDATE_NAME:
        raise ValueError("candidate served name changed")
    if manifest_path != output / "interpolation_manifest.json":
        raise ValueError("interpolation manifest path changed")
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("interpolation manifest schema changed")
    if manifest.get("artifact_type") != "checkpoint_weight_interpolation":
        raise ValueError("interpolation artifact type changed")
    if manifest.get("status") != "complete":
        raise ValueError("interpolation status is incomplete")
    if manifest.get("alpha") != EXPECTED_ALPHA:
        raise ValueError("interpolation alpha changed")
    if manifest.get("equation") != EXPECTED_EQUATION:
        raise ValueError("interpolation equation changed")

    copied_files, controlled_differences = _compatible_copy_files(
        anchor,
        source,
        output,
    )
    if manifest.get("copied_files") != copied_files:
        raise ValueError("interpolation copied-file list changed")
    if (
        manifest.get("controlled_copy_file_differences")
        != controlled_differences
    ):
        raise ValueError("interpolation copied-file differences changed")
    anchor_contract = _model_contract(anchor, copied_files)
    source_contract = _model_contract(source, copied_files)
    output_contract = _model_contract(output, copied_files)
    if manifest.get("anchor") != anchor_contract:
        raise ValueError("anchor model contract changed")
    if manifest.get("candidate") != source_contract:
        raise ValueError("source model contract changed")
    if manifest.get("output") != output_contract:
        raise ValueError("output model contract changed")
    if manifest.get("copied_file_bindings") != output_contract[
        "copied_file_bindings"
    ]:
        raise ValueError("output copied-file bindings changed")

    index = output / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError("interpolated output must be sharded")
    index_value = _read_object(index)
    weight_map = index_value.get("weight_map")
    metadata = index_value.get("metadata")
    if not isinstance(weight_map, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("interpolated output index is incomplete")
    tensor_count = len(weight_map)
    floating = manifest.get("floating_tensors")
    nonfloating = manifest.get("copied_nonfloating_tensors")
    if (
        manifest.get("tensor_count") != tensor_count
        or not isinstance(floating, int)
        or isinstance(floating, bool)
        or not isinstance(nonfloating, int)
        or isinstance(nonfloating, bool)
        or floating < 0
        or nonfloating < 0
        or floating + nonfloating != tensor_count
    ):
        raise ValueError("interpolation tensor counts changed")
    if manifest.get("output_shards") != len(set(weight_map.values())):
        raise ValueError("interpolation shard count changed")
    total_size = metadata.get("total_size")
    if (
        not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or manifest.get("total_tensor_bytes") != total_size
    ):
        raise ValueError("interpolation tensor byte count changed")
    peak_rss = manifest.get("peak_rss_bytes")
    if not isinstance(peak_rss, int) or isinstance(peak_rss, bool) or peak_rss <= 0:
        raise ValueError("interpolation RSS evidence changed")

    return {
        "schema_version": 1,
        "artifact_type": "v2p11_interpolation_lineage",
        "status": "complete",
        "served_name": candidate_name,
        "alpha": EXPECTED_ALPHA,
        "manifest": _binding(manifest_path),
        "anchor": anchor_contract,
        "source": source_contract,
        "output": output_contract,
        "controlled_copy_file_differences": controlled_differences,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-model", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_interpolation_lineage(
        manifest_path=args.manifest,
        anchor_model_path=args.anchor_model,
        source_model_path=args.source_model,
        output_model_path=args.output_model,
        candidate_name=args.candidate_name,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
