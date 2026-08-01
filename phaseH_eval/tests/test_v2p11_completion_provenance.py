from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from phaseH_eval.tests.v2p11_lineage_fixture import (
    create_complete_phase_lineage,
    write_phase_marker,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _write_v2p10_lineage_contract(
    root: Path,
    *,
    composite: Path,
) -> Path:
    artifacts: dict[str, dict[str, object]] = {
        "composite": _binding(composite),
    }
    for name in (
        "marker",
        "run_manifest",
        "dataset_manifest",
        "train_jsonl",
        "adapter",
        "merge_audit",
        "stage_data_manifest",
    ):
        artifact = root / f"v2p10_{name}.json"
        artifact.write_text("{}\n")
        artifacts[name] = _binding(artifact)
    contract = root / "v2p10_training_lineage.json"
    _write_json(
        contract,
        {
            "schema_version": 1,
            "artifact_type": "v2p10_training_lineage",
            "status": "complete",
            "artifacts": artifacts,
            "model_contract": json.loads(
                composite.read_text()
            )["model_contract"],
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
        },
    )
    return contract


def test_publishes_one_immutable_fable_and_posttrain_provenance_report(
    tmp_path: Path,
) -> None:
    provenance_module = importlib.import_module(
        "phaseH_eval.v2p11_completion_provenance"
    )
    full_ids = tmp_path / "full_ids.json"
    v2p10 = tmp_path / "v2p10.json"
    training = tmp_path / "training.json"
    recovery = tmp_path / "recovery.json"
    posttrain = tmp_path / "posttrain.json"
    stage_data = tmp_path / "stage_data.json"
    stage_marker = tmp_path / "v2p11_training_complete.json"
    posttrain_marker = tmp_path / "posttrain_marker.json"
    portability_gate = tmp_path / "portability_gate.json"
    final_model = tmp_path / "final_model"
    _write_json(full_ids, [f"case-{index:03d}" for index in range(300)])
    _write_json(
        v2p10,
        {
            "status": "complete",
            "model_contract": {
                "served_name": "teacher_sft_v2p10",
                "model_path": "/control",
                "model_config_sha256": "c" * 64,
                "model_index_sha256": "d" * 64,
                "model_artifacts": [],
            },
        },
    )
    v2p10_lineage = _write_v2p10_lineage_contract(
        tmp_path,
        composite=v2p10,
    )
    _write_json(
        training,
        {
            "rows": 1247,
            "base_rows": 1211,
            "fable_rows": 36,
            "max_seq": 32768,
            "optimizer_steps": 78,
            "dataset_manifest_sha256": "a" * 64,
            "train_jsonl_sha256": "b" * 64,
            "v2p10_full300_sha256": hashlib.sha256(
                v2p10.read_bytes()
            ).hexdigest(),
        },
    )
    _write_json(
        recovery,
        {
            "rows": 46,
            "evaluation_overlap": 0,
            "combined_exclusion_sha256": "e" * 64,
            "combined_exclusion_ids": 707,
            "excluded_repositories": 12,
            "source_counts": {
                "data/fable5_recent_verified_revision_v1": 10,
                "data/fable5_v2p11_frozen47_prepared_safe36": 36,
            },
        },
    )
    _write_json(
        posttrain,
        {
            "recovery_rows": 138,
            "recovery_optimizer_steps": 105,
            "behavior_rows": 606,
            "behavior_negative_counts": {
                "empty_terminal": 33,
                "repeated_read_loop": 55,
                "wrong_nonempty_replay": 228,
            },
            "full_evaluation_ids": 300,
            "combined_exclusion_ids": 707,
            "excluded_repositories": 12,
            "evaluation_overlap": 0,
            "recovery_manifest_sha256": "c" * 64,
            "behavior_manifest_sha256": "d" * 64,
            "exclusion_sha256": "e" * 64,
        },
    )
    _write_json(
        stage_data,
        {
            "rows": 1247,
            "base_rows": 1211,
            "new_fable_rows": 36,
            "dataset_manifest_sha256": "a" * 64,
            "train_jsonl_sha256": "b" * 64,
            "strict_fable_manifest_sha256": "1" * 64,
            "campaign_manifest_sha256": "2" * 64,
            "fable_lineage": {
                "inherited_pinned_rows": 31,
                "historical_verified_revision_rows": 10,
                "new_strict_rows": 36,
                "total_stage_a_fable_rows": 77,
            },
            "fable_freeze": {
                "attempted": 60,
                "resolved": 47,
                "collection_rejected": 13,
                "strict_admitted": 36,
                "distillation_rejected": 11,
            },
            "evaluation_exclusion": {
                "ids": 707,
                "repositories": 12,
                "overlap": 0,
                "sha256": "e" * 64,
            },
        },
    )
    _write_json(
        stage_marker,
        {
            "schema_version": 1,
            "complete": True,
            "data_manifest_sha256": "a" * 64,
            "train_jsonl_sha256": "b" * 64,
            "adapter_sha256": "f" * 64,
            "optimizer_steps": 78,
        },
    )
    lineage = create_complete_phase_lineage(
        tmp_path / "lineage",
        final_model=final_model,
        stage_marker=stage_marker,
    )
    posttrain_value = json.loads(posttrain.read_text())
    posttrain_value["recovery_manifest_sha256"] = _binding(
        lineage["recovery_manifest"]
    )["sha256"]
    posttrain_value["behavior_manifest_sha256"] = _binding(
        lineage["behavior_manifest"]
    )["sha256"]
    _write_json(posttrain, posttrain_value)
    shard = final_model / "model-00001-of-00001.safetensors"
    _write_json(
        posttrain_marker,
        {
            "schema_version": 1,
            "complete": True,
            "stage_a_adapter_sha256": json.loads(
                stage_marker.read_text()
            )["adapter_sha256"],
            "recovery_manifest_sha256": posttrain_value[
                "recovery_manifest_sha256"
            ],
            "behavior_manifest_sha256": posttrain_value[
                "behavior_manifest_sha256"
            ],
            "recovery_optimizer_steps": 105,
            "kto_optimizer_steps": 25,
            "recovery_input_marker_sha256": _binding(
                lineage["recovery_input"]
            )["sha256"],
            "kto_input_marker_sha256": _binding(
                lineage["kto_input"]
            )["sha256"],
            "kto_training_evidence_sha256": _binding(
                lineage["kto_evidence"]
            )["sha256"],
            "final_merge_audit_sha256": _binding(
                final_model / "v2p11_final_merge_audit.json"
            )["sha256"],
            "final_merge_marker_sha256": _binding(
                lineage["final_merge"]
            )["sha256"],
        },
    )
    model_contract = provenance_module._model_contract(final_model)
    portability_criteria = {
        "candidate_empty_at_most_one": True,
        "candidate_empty_no_regression": True,
        "candidate_no_format_regression": True,
        "candidate_no_loop_regression": True,
        "candidate_resolution_floor": True,
    }
    _write_json(
        portability_gate,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_controller_free_portability_gate",
            "status": "complete",
            "passed": True,
            "control_name": "teacher_sft_v2p10",
            "candidate_name": "teacher_sft_v2p11",
            "population": 10,
            "evaluation_overlap": 0,
            "control": {"model_contract": {"model_path": "/control"}},
            "candidate": {"model_contract": model_contract},
            "criteria": portability_criteria,
        },
    )
    output = tmp_path / "provenance.json"

    report = provenance_module.publish_completion_provenance(
        training_contract_path=training,
        recovery_contract_path=recovery,
        posttrain_contract_path=posttrain,
        stage_data_contract_path=stage_data,
        stage_marker_path=stage_marker,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        posttrain_marker_path=posttrain_marker,
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        v2p10_lineage_contract_path=v2p10_lineage,
        final_model_path=final_model,
        portability_gate_path=portability_gate,
        output_path=output,
    )

    assert report["status"] == "complete"
    assert report["fable"] == {
        "stage_a": {
            "inherited_pinned_rows": 31,
            "historical_verified_revision_rows": 10,
            "new_strict_rows": 36,
            "total_rows": 77,
        },
        "extracted_campaign": {
            "attempted": 60,
            "resolved": 47,
            "collection_rejected": 13,
            "strict_admitted": 36,
            "distillation_rejected": 11,
        },
        "recovery_unique_sources": 46,
        "targeted_recovery_rows": 138,
    }
    assert report["evaluation_exclusion"] == {
        "full_ids": 300,
        "excluded_ids": 707,
        "excluded_repositories": 12,
        "stage_a_overlap": 0,
        "overlap": 0,
        "sha256": "e" * 64,
    }
    assert report["final_model"]["model_artifacts"] == [_binding(shard)]
    assert report["portability"] == {
        "gate": _binding(portability_gate),
        "passed": True,
        "population": 10,
        "criteria": portability_criteria,
    }
    assert provenance_module.publish_completion_provenance(
        training_contract_path=training,
        recovery_contract_path=recovery,
        posttrain_contract_path=posttrain,
        stage_data_contract_path=stage_data,
        stage_marker_path=stage_marker,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        posttrain_marker_path=posttrain_marker,
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        v2p10_lineage_contract_path=v2p10_lineage,
        final_model_path=final_model,
        portability_gate_path=portability_gate,
        output_path=output,
    ) == report


def test_rejects_structurally_valid_but_content_free_phase_marker(
    tmp_path: Path,
) -> None:
    provenance_module = importlib.import_module(
        "phaseH_eval.v2p11_completion_provenance"
    )
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"arbitrary")
    marker = tmp_path / "marker.json"
    _write_json(
        marker,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_poststage_phase",
            "phase": "recovery_sft",
            "status": "complete",
            "artifacts": [_binding(payload)],
        },
    )

    with pytest.raises(ValueError, match="artifact set is incomplete"):
        provenance_module._validate_phase_marker(
            marker,
            expected_phase="recovery_sft",
        )


def test_rejects_recovery_input_with_short_stage_context(
    tmp_path: Path,
) -> None:
    provenance_module = importlib.import_module(
        "phaseH_eval.v2p11_completion_provenance"
    )
    stage_marker = tmp_path / "v2p11_training_complete.json"
    _write_json(
        stage_marker,
        {"complete": True, "optimizer_steps": 78},
    )
    lineage = create_complete_phase_lineage(
        tmp_path / "lineage",
        final_model=tmp_path / "teacher_sft_v2p11_full",
        stage_marker=stage_marker,
    )
    recovery_input = lineage["recovery_input"]
    input_report = json.loads(recovery_input.read_text())
    artifacts = [Path(item["path"]) for item in input_report["artifacts"]]
    run_manifest = next(
        path for path in artifacts if path.name == "run_manifest.json"
    )
    manifest = json.loads(run_manifest.read_text())
    manifest["max_seq"] = 8192
    _write_json(run_manifest, manifest)
    stage_report = json.loads(stage_marker.read_text())
    stage_report["run_manifest_sha256"] = _binding(run_manifest)["sha256"]
    _write_json(stage_marker, stage_report)
    write_phase_marker(
        recovery_input,
        "recovery_sft_inputs",
        artifacts,
    )

    with pytest.raises(ValueError, match="Stage-A run manifest"):
        provenance_module._validate_phase_marker(
            recovery_input,
            expected_phase="recovery_sft_inputs",
        )


def test_rejects_stage_run_with_compile_policy_drift(
    tmp_path: Path,
) -> None:
    provenance_module = importlib.import_module(
        "phaseH_eval.v2p11_completion_provenance"
    )
    stage_marker = tmp_path / "v2p11_training_complete.json"
    _write_json(
        stage_marker,
        {"complete": True, "optimizer_steps": 78},
    )
    lineage = create_complete_phase_lineage(
        tmp_path / "lineage",
        final_model=tmp_path / "teacher_sft_v2p11_full",
        stage_marker=stage_marker,
    )
    recovery_input = lineage["recovery_input"]
    input_report = json.loads(recovery_input.read_text())
    artifacts = [Path(item["path"]) for item in input_report["artifacts"]]
    run_manifest = next(
        path for path in artifacts if path.name == "run_manifest.json"
    )
    manifest = json.loads(run_manifest.read_text())
    manifest["torch_compile_disabled"] = True
    _write_json(run_manifest, manifest)
    stage_report = json.loads(stage_marker.read_text())
    stage_report["run_manifest_sha256"] = _binding(run_manifest)["sha256"]
    _write_json(stage_marker, stage_report)
    write_phase_marker(
        recovery_input,
        "recovery_sft_inputs",
        artifacts,
    )

    with pytest.raises(ValueError, match="Stage-A run manifest"):
        provenance_module._validate_phase_marker(
            recovery_input,
            expected_phase="recovery_sft_inputs",
        )


def test_rejects_kto_marker_with_wrong_deterministic_training_rows(
    tmp_path: Path,
) -> None:
    provenance_module = importlib.import_module(
        "phaseH_eval.v2p11_completion_provenance"
    )
    stage_marker = tmp_path / "v2p11_training_complete.json"
    _write_json(
        stage_marker,
        {"complete": True, "optimizer_steps": 78},
    )
    lineage = create_complete_phase_lineage(
        tmp_path / "lineage",
        final_model=tmp_path / "teacher_sft_v2p11_full",
        stage_marker=stage_marker,
    )
    evidence = json.loads(lineage["kto_evidence"].read_text())
    evidence["selected_sample_uids"] = list(
        reversed(evidence["selected_sample_uids"])
    )
    _write_json(lineage["kto_evidence"], evidence)
    marker = json.loads(lineage["kto"].read_text())
    marker["artifacts"] = [
        (
            _binding(lineage["kto_evidence"])
            if Path(artifact["path"]) == lineage["kto_evidence"].resolve()
            else artifact
        )
        for artifact in marker["artifacts"]
    ]
    _write_json(lineage["kto"], marker)

    with pytest.raises(ValueError, match="KTO training evidence"):
        provenance_module._validate_phase_marker(
            lineage["kto"],
            expected_phase="kto_full",
        )
