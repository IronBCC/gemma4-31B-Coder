from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.v2p10_training_lineage import (
    build_v2p10_training_lineage,
    validate_v2p10_training_lineage_contract,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data" / "teacher_train_mix_v2p10"
    train = data / "train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text("{}\n" * 1211)
    dataset_manifest = data / "manifest.json"
    _write_json(
        dataset_manifest,
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p10",
            "rendered": 1211,
            "training_admitted": 1211,
            "train_jsonl_sha256": _sha(train),
            "fable_revision_rows": 10,
            "gpt56sol_rows": 59,
            "artifact_bindings": [
                *[
                    {
                        "delta_kind": "fable_revision",
                        "instance_id": f"fable-{index}",
                    }
                    for index in range(10)
                ],
                *[
                    {
                        "delta_kind": "gpt56sol",
                        "instance_id": f"gpt-{index}",
                    }
                    for index in range(59)
                ],
            ],
            "all_training_gates_complete": True,
            "standard_native_format_loss_gate": {
                "status": "passed",
                "failure_count": 0,
                "samples": 1211,
            },
        },
    )
    adapter_dir = tmp_path / "adapters" / "teacher_sft_v2p10_bf16"
    adapter = adapter_dir / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    run_manifest = adapter_dir / "run_manifest.json"
    _write_json(
        run_manifest,
        {
            "base": (
                "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
            ),
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
        },
    )
    model = tmp_path / "models" / "teacher_sft_v2p10_full"
    model.mkdir(parents=True)
    _write_json(model / "config.json", {"architecture": "Gemma4"})
    (model / "model.safetensors").write_bytes(b"merged")
    merge_audit = model / "v2p10_merge_audit.json"
    _write_json(
        merge_audit,
        {
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
        },
    )
    marker = tmp_path / "runs" / "v2p10_train_merge_complete.json"
    _write_json(
        marker,
        {
            "schema_version": 1,
            "complete": True,
            "adapter_sha256": _sha(adapter),
            "dataset_manifest_sha256": _sha(dataset_manifest),
            "merge_audit_sha256": _sha(merge_audit),
        },
    )
    composite = tmp_path / "runs" / "v2p10_full300_composite.json"
    model_contract = {
        key: value
        for key, value in _model_contract(model).items()
        if value is not None
    }
    model_contract["served_name"] = "teacher_sft_v2p10"
    _write_json(
        composite,
        {
            "schema_version": 1,
            "artifact_type": "disjoint_panel_full300_composite",
            "status": "complete",
            "name": "teacher_sft_v2p10",
            "model_contract": model_contract,
        },
    )
    stage_manifest = (
        tmp_path / "data" / "teacher_train_mix_v2p11_frozen1247"
        / "manifest.json"
    )
    _write_json(
        stage_manifest,
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11",
            "base_variant": "teacher_train_mix_v2p10",
            "base_rows": 1211,
            "fable_rows": 36,
            "base": {
                "files": [
                    {
                        "path": str(dataset_manifest.resolve()),
                        "sha256": _sha(dataset_manifest),
                        "bytes": dataset_manifest.stat().st_size,
                    },
                    {
                        "path": str(train.resolve()),
                        "sha256": _sha(train),
                        "bytes": train.stat().st_size,
                    },
                ]
            },
        },
    )
    return {
        "marker": marker,
        "run_manifest": run_manifest,
        "dataset_manifest": dataset_manifest,
        "train": train,
        "adapter": adapter,
        "merge_audit": merge_audit,
        "model": model,
        "composite": composite,
        "stage_manifest": stage_manifest,
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return build_v2p10_training_lineage(
        marker_path=paths["marker"],
        run_manifest_path=paths["run_manifest"],
        dataset_manifest_path=paths["dataset_manifest"],
        train_jsonl_path=paths["train"],
        adapter_path=paths["adapter"],
        merge_audit_path=paths["merge_audit"],
        model_path=paths["model"],
        composite_path=paths["composite"],
        stage_data_manifest_path=paths["stage_manifest"],
    )


def test_binds_control_training_merge_model_and_stage_base(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    report = _build(paths)

    assert report["status"] == "complete"
    assert report["training"]["max_seq"] == 32768
    assert report["training"]["rows"] == 1211
    assert report["fable"] == {
        "historical_verified_revision_rows": 10,
        "recent_v2p11_rows": 0,
    }
    assert report["model_contract"]["served_name"] == "teacher_sft_v2p10"


def test_rejects_short_context_control_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text())
    manifest["max_seq"] = 8192
    _write_json(paths["run_manifest"], manifest)

    with pytest.raises(ValueError, match="training manifest"):
        _build(paths)


def test_rejects_stage_data_not_based_on_exact_control(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    stage = json.loads(paths["stage_manifest"].read_text())
    stage["base"]["files"][0]["sha256"] = "0" * 64
    _write_json(paths["stage_manifest"], stage)

    with pytest.raises(ValueError, match="Stage-A base"):
        _build(paths)


def test_contract_revalidation_rejects_changed_bound_artifact(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    contract = tmp_path / "runs" / "v2p10_training_lineage.json"
    _write_json(contract, _build(paths))
    assert validate_v2p10_training_lineage_contract(
        contract,
        composite_path=paths["composite"],
    )["status"] == "complete"

    paths["run_manifest"].write_text("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_v2p10_training_lineage_contract(
            contract,
            composite_path=paths["composite"],
        )
