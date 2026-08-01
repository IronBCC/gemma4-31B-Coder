#!/usr/bin/env python3
"""Validate the immutable v2.10 training and merge lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from phaseH_eval.empty_retry_composite import (
    _binding,
    _read_object,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract


_TRAINING_FIELDS = {
    "base": "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it",
    "data": "data/teacher_train_mix_v2p10",
    "data_len": 1211,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-5,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "warmup_steps": 8,
    "load_4bit": False,
    "selective_assistant_loss": True,
    "gradient_checkpointing": "bounded_unsloth",
    "init_adapter": None,
}
_MERGE_FIELDS = {
    "schema_version": 1,
    "complete": True,
    "architecture": "Gemma4ForConditionalGeneration",
    "expected_tensors": 1188,
    "actual_tensors": 1188,
    "expected_vision": 356,
    "actual_vision": 356,
    "missing_tensors": [],
    "unexpected_tensors": [],
    "misplaced_tensors": [],
    "nonfinite_tensors": [],
}


def _require_fields(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    changed = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if changed:
        raise ValueError(f"{label} is incomplete: {changed}")


def _require_stage_base_binding(
    stage_manifest: Mapping[str, Any],
    *,
    dataset_manifest_path: Path,
    train_jsonl_path: Path,
) -> None:
    base = stage_manifest.get("base")
    files = base.get("files") if isinstance(base, Mapping) else None
    if not isinstance(files, list):
        raise ValueError("Stage-A base binding is incomplete")
    expected = {
        dataset_manifest_path.resolve(): _binding(dataset_manifest_path),
        train_jsonl_path.resolve(): _binding(train_jsonl_path),
    }
    actual: dict[Path, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("path"), str
        ):
            continue
        actual[Path(item["path"]).resolve()] = item
    if any(
        path not in actual
        or actual[path].get("sha256") != binding["sha256"]
        or actual[path].get("bytes") != binding["bytes"]
        for path, binding in expected.items()
    ):
        raise ValueError("Stage-A base binding differs from v2.10 data")


def build_v2p10_training_lineage(
    *,
    marker_path: Path,
    run_manifest_path: Path,
    dataset_manifest_path: Path,
    train_jsonl_path: Path,
    adapter_path: Path,
    merge_audit_path: Path,
    model_path: Path,
    composite_path: Path,
    stage_data_manifest_path: Path,
) -> dict[str, Any]:
    paths = {
        "marker": Path(marker_path).resolve(),
        "run_manifest": Path(run_manifest_path).resolve(),
        "dataset_manifest": Path(dataset_manifest_path).resolve(),
        "train_jsonl": Path(train_jsonl_path).resolve(),
        "adapter": Path(adapter_path).resolve(),
        "merge_audit": Path(merge_audit_path).resolve(),
        "composite": Path(composite_path).resolve(),
        "stage_data_manifest": Path(stage_data_manifest_path).resolve(),
    }
    marker = _read_object(paths["marker"])
    run_manifest = _read_object(paths["run_manifest"])
    dataset_manifest = _read_object(paths["dataset_manifest"])
    merge_audit = _read_object(paths["merge_audit"])
    composite = _read_object(paths["composite"])
    stage_manifest = _read_object(paths["stage_data_manifest"])

    _require_fields(
        marker,
        {
            "schema_version": 1,
            "complete": True,
            "adapter_sha256": _binding(paths["adapter"])["sha256"],
            "dataset_manifest_sha256": _binding(
                paths["dataset_manifest"]
            )["sha256"],
            "merge_audit_sha256": _binding(paths["merge_audit"])[
                "sha256"
            ],
        },
        label="v2.10 completion marker",
    )
    _require_fields(
        run_manifest,
        _TRAINING_FIELDS,
        label="v2.10 training manifest",
    )
    native_gate = dataset_manifest.get(
        "standard_native_format_loss_gate"
    )
    _require_fields(
        dataset_manifest,
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p10",
            "rendered": 1211,
            "training_admitted": 1211,
            "train_jsonl_sha256": _binding(paths["train_jsonl"])[
                "sha256"
            ],
            "fable_revision_rows": 10,
            "gpt56sol_rows": 59,
            "all_training_gates_complete": True,
        },
        label="v2.10 dataset manifest",
    )
    if (
        not isinstance(native_gate, Mapping)
        or native_gate.get("status") != "passed"
        or native_gate.get("failure_count") != 0
        or native_gate.get("samples") != 1211
    ):
        raise ValueError("v2.10 native format/loss gate is incomplete")
    source_bindings = dataset_manifest.get("artifact_bindings")
    if (
        not isinstance(source_bindings, list)
        or len(source_bindings) != 69
        or sum(
            isinstance(row, Mapping)
            and row.get("delta_kind") == "fable_revision"
            for row in source_bindings
        )
        != 10
        or sum(
            isinstance(row, Mapping)
            and row.get("delta_kind") == "gpt56sol"
            for row in source_bindings
        )
        != 59
    ):
        raise ValueError("v2.10 teacher-source lineage is incomplete")
    _require_fields(
        merge_audit,
        _MERGE_FIELDS,
        label="v2.10 merge audit",
    )
    _require_fields(
        stage_manifest,
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11",
            "base_variant": "teacher_train_mix_v2p10",
            "base_rows": 1211,
            "fable_rows": 36,
        },
        label="v2.11 Stage-A dataset manifest",
    )
    _require_stage_base_binding(
        stage_manifest,
        dataset_manifest_path=paths["dataset_manifest"],
        train_jsonl_path=paths["train_jsonl"],
    )

    model_contract = {
        key: value
        for key, value in _model_contract(Path(model_path)).items()
        if value is not None
    }
    model_contract["served_name"] = "teacher_sft_v2p10"
    _require_fields(
        composite,
        {
            "schema_version": 1,
            "artifact_type": "disjoint_panel_full300_composite",
            "status": "complete",
            "name": "teacher_sft_v2p10",
            "model_contract": model_contract,
        },
        label="v2.10 full300 composite",
    )
    return {
        "schema_version": 1,
        "artifact_type": "v2p10_training_lineage",
        "status": "complete",
        "artifacts": {
            name: _binding(path)
            for name, path in paths.items()
        },
        "model_contract": model_contract,
        "training": {
            "rows": 1211,
            "max_seq": 32768,
            "optimizer_steps": 76,
            "rank": 32,
            "alpha": 32,
            "gradient_accumulation": 16,
        },
        "fable": {
            "historical_verified_revision_rows": 10,
            "recent_v2p11_rows": 0,
        },
        "stage_a_base": {
            "rows": 1211,
            "variant": "teacher_train_mix_v2p10",
        },
    }


def validate_v2p10_training_lineage_contract(
    contract_path: Path,
    *,
    composite_path: Path,
) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    composite_path = Path(composite_path).resolve()
    report = _read_object(contract_path)
    composite = _read_object(composite_path)
    artifacts = report.get("artifacts")
    required_artifacts = {
        "marker",
        "run_manifest",
        "dataset_manifest",
        "train_jsonl",
        "adapter",
        "merge_audit",
        "composite",
        "stage_data_manifest",
    }
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type") != "v2p10_training_lineage"
        or report.get("status") != "complete"
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != required_artifacts
        or report.get("training")
        != {
            "rows": 1211,
            "max_seq": 32768,
            "optimizer_steps": 76,
            "rank": 32,
            "alpha": 32,
            "gradient_accumulation": 16,
        }
        or report.get("fable")
        != {
            "historical_verified_revision_rows": 10,
            "recent_v2p11_rows": 0,
        }
        or report.get("stage_a_base")
        != {
            "rows": 1211,
            "variant": "teacher_train_mix_v2p10",
        }
        or not isinstance(report.get("model_contract"), Mapping)
        or report.get("model_contract") != composite.get("model_contract")
    ):
        raise ValueError("v2.10 training lineage contract is incomplete")
    for name, artifact in artifacts.items():
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("path"), str)
            or dict(artifact) != _binding(Path(artifact["path"]))
        ):
            raise ValueError(
                f"v2.10 training lineage artifact changed: {name}"
            )
    if artifacts["composite"] != _binding(composite_path):
        raise ValueError(
            "v2.10 training lineage binds a different full300 composite"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument(
        "--stage-data-manifest",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    report = build_v2p10_training_lineage(
        marker_path=args.marker,
        run_manifest_path=args.run_manifest,
        dataset_manifest_path=args.dataset_manifest,
        train_jsonl_path=args.train_jsonl,
        adapter_path=args.adapter,
        merge_audit_path=args.merge_audit,
        model_path=args.model,
        composite_path=args.composite,
        stage_data_manifest_path=args.stage_data_manifest,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
