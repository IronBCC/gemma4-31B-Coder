#!/usr/bin/env python3
"""Publish the immutable data and training lineage for the v2.11 verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseE_rl.kto_swe_outcome import select_behavior_coverage_rows
from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
)


_FABLE_SOURCE_COUNTS = {
    "data/fable5_recent_verified_revision_v1": 10,
    "data/fable5_v2p11_frozen47_prepared_safe36": 36,
}
_BEHAVIOR_NEGATIVE_COUNTS = {
    "empty_terminal": 33,
    "repeated_read_loop": 55,
    "wrong_nonempty_replay": 228,
}
_PORTABILITY_CRITERIA = {
    "candidate_empty_at_most_one": True,
    "candidate_empty_no_regression": True,
    "candidate_no_format_regression": True,
    "candidate_no_loop_regression": True,
    "candidate_resolution_floor": True,
}


def _model_contract(model_path: Path) -> dict[str, Any]:
    model_path = Path(model_path).resolve()
    config = model_path / "config.json"
    index = model_path / "model.safetensors.index.json"
    single = model_path / "model.safetensors"
    if not config.is_file():
        raise ValueError("final model config is missing")
    if index.is_file():
        index_value = _read_object(index)
        weight_map = index_value.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("final model weight map is invalid")
        shards = sorted({
            model_path / shard
            for shard in weight_map.values()
            if isinstance(shard, str)
        })
        if len(shards) != len(set(weight_map.values())):
            raise ValueError("final model weight map is invalid")
        weight_contract = {
            "model_index_sha256": _binding(index)["sha256"],
            "model_safetensors_sha256": None,
        }
    elif single.is_file():
        shards = [single]
        weight_contract = {
            "model_index_sha256": None,
            "model_safetensors_sha256": _binding(single)["sha256"],
        }
    else:
        raise ValueError("final model weights are missing")
    if not shards or any(not shard.is_file() for shard in shards):
        raise ValueError("final model weight shard is missing")
    return {
        "model_path": str(model_path),
        "model_config_sha256": _binding(config)["sha256"],
        **weight_contract,
        "model_artifacts": [_binding(shard) for shard in shards],
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
        raise ValueError(f"{label} contract is incomplete: {changed}")


def _validate_portability_gate(
    path: Path,
    *,
    final_model: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _read_object(path)
    control = gate.get("control")
    candidate = gate.get("candidate")
    if (
        gate.get("schema_version") != 1
        or gate.get("artifact_type")
        != "v2p11_controller_free_portability_gate"
        or gate.get("status") != "complete"
        or gate.get("passed") is not True
        or gate.get("control_name") != "teacher_sft_v2p10"
        or gate.get("candidate_name") != "teacher_sft_v2p11"
        or gate.get("population") != 10
        or gate.get("evaluation_overlap") != 0
        or gate.get("criteria") != _PORTABILITY_CRITERIA
        or not isinstance(control, Mapping)
        or not isinstance(candidate, Mapping)
        or candidate.get("model_contract") != final_model
    ):
        raise ValueError(
            "controller-free portability gate did not pass for final model"
        )
    return {
        "gate": _binding(path),
        "passed": True,
        "population": 10,
        "criteria": dict(_PORTABILITY_CRITERIA),
    }


def _read_current_phase_marker(
    path: Path,
    *,
    expected_phase: str,
) -> dict[str, Any]:
    report = _read_object(path)
    artifacts = report.get("artifacts")
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type") != "v2p11_poststage_phase"
        or report.get("phase") != expected_phase
        or report.get("status") != "complete"
        or not isinstance(artifacts, list)
        or not artifacts
    ):
        raise ValueError(f"{expected_phase} phase marker is incomplete")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("path"), str)
            or dict(artifact) != _binding(Path(artifact["path"]))
        ):
            raise ValueError(
                f"{expected_phase} phase artifact binding changed"
            )
    return report


def _phase_artifacts(
    report: Mapping[str, Any],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for artifact in report["artifacts"]:
        path = Path(artifact["path"])
        if path.name in result:
            raise ValueError(
                f"{report['phase']} phase has duplicate artifact basenames"
            )
        result[path.name] = path
    return result


def _require_artifact_names(
    artifacts: Mapping[str, Path],
    expected: set[str],
    *,
    phase: str,
) -> None:
    if set(artifacts) != expected:
        raise ValueError(
            f"{phase} phase artifact set is incomplete: "
            f"actual={sorted(artifacts)} expected={sorted(expected)}"
        )


def _validate_model_phase(
    report: Mapping[str, Any],
    *,
    audit_name: str,
    source_markers: Mapping[str, str],
) -> None:
    artifacts = _phase_artifacts(report)
    fixed_names = {
        audit_name,
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        *source_markers,
    }
    if not fixed_names.issubset(artifacts):
        raise ValueError(
            f"{report['phase']} phase model artifacts are incomplete"
        )
    index_path = artifacts["model.safetensors.index.json"]
    model_path = index_path.parent
    index = _read_object(index_path)
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, Mapping)
        or len(weight_map) != 1188
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in weight_map.items()
        )
        or sum("vision" in key for key in weight_map) != 356
    ):
        raise ValueError(f"{report['phase']} model index is invalid")
    shard_names = set(weight_map.values())
    expected = fixed_names | shard_names
    _require_artifact_names(
        artifacts,
        expected,
        phase=str(report["phase"]),
    )
    model_names = (
        shard_names
        | {
            audit_name,
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "chat_template.jinja",
        }
    )
    if any(artifacts[name].parent != model_path for name in model_names):
        raise ValueError(
            f"{report['phase']} model artifacts span multiple directories"
        )
    config = _read_object(artifacts["config.json"])
    if config.get("architectures") != [
        "Gemma4ForConditionalGeneration"
    ]:
        raise ValueError(f"{report['phase']} model architecture is invalid")
    audit = _read_object(artifacts[audit_name])
    if (
        audit.get("complete") is not True
        or audit.get("architecture")
        != "Gemma4ForConditionalGeneration"
        or audit.get("expected_tensors") != 1188
        or audit.get("actual_tensors") != 1188
        or audit.get("expected_vision") != 356
        or audit.get("actual_vision") != 356
        or audit.get("missing_tensors") != []
        or audit.get("unexpected_tensors") != []
        or audit.get("misplaced_tensors") != []
        or audit.get("nonfinite_tensors") != []
    ):
        raise ValueError(f"{report['phase']} merge audit is incomplete")
    for marker_name, phase in source_markers.items():
        _validate_phase_marker(
            artifacts[marker_name],
            expected_phase=phase,
        )


def _validate_kto_evidence(
    evidence_path: Path,
    *,
    input_report: Mapping[str, Any],
) -> None:
    evidence = _read_object(evidence_path)
    inputs = _phase_artifacts(input_report)
    data_path = inputs["v2p11_behavior_kto_v2.jsonl"]
    manifest_path = inputs["v2p11_behavior_kto_v2_manifest.json"]
    rows = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("behavior KTO source row is invalid")
        rows.append(row)
    selected, counts = select_behavior_coverage_rows(rows)
    selected_uids = [str(row["sample_uid"]) for row in selected]
    recovery_marker = inputs["v2p11_recovery_merge_complete.json"]
    recovery_report = _read_current_phase_marker(
        recovery_marker,
        expected_phase="recovery_merge",
    )
    recovery_model_path = _phase_artifacts(
        recovery_report
    )["config.json"].parent.resolve()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        evidence.get("schema_version") != 1
        or evidence.get("artifact_type")
        != "v2p11_kto_training_evidence"
        or evidence.get("status") != "complete"
        or evidence.get("base_model_path") != str(recovery_model_path)
        or evidence.get("source_data_path") != str(data_path.resolve())
        or evidence.get("source_data_sha256") != sha(data_path)
        or evidence.get("source_manifest_path")
        != str(manifest_path.resolve())
        or evidence.get("source_manifest_sha256")
        != sha(manifest_path)
        or evidence.get("source_rows") != 606
        or evidence.get("training_rows") != 50
        or evidence.get("coverage_required") is not True
        or evidence.get("coverage_counts") != counts
        or evidence.get("selected_sample_uids") != selected_uids
        or evidence.get("per_device_train_batch_size") != 2
        or evidence.get("gradient_accumulation_steps") != 1
        or evidence.get("optimizer_steps") != 25
        or evidence.get("seed") != 0
    ):
        raise ValueError("KTO training evidence is incomplete")


def _validate_phase_marker(
    path: Path,
    *,
    expected_phase: str,
) -> dict[str, Any]:
    report = _read_current_phase_marker(
        path,
        expected_phase=expected_phase,
    )
    artifacts = _phase_artifacts(report)
    if expected_phase == "recovery_sft_inputs":
        _require_artifact_names(
            artifacts,
            {
                "v2p11_training_complete.json",
                "adapter_model.safetensors",
                "adapter_config.json",
                "run_manifest.json",
                "train.jsonl",
                "manifest.json",
            },
            phase=expected_phase,
        )
        stage_marker = _read_object(
            artifacts["v2p11_training_complete.json"]
        )
        stage_adapter = artifacts["adapter_model.safetensors"]
        stage_config = _read_object(artifacts["adapter_config.json"])
        stage_run_manifest = _read_object(artifacts["run_manifest.json"])
        expected_stage_run = {
            "base": (
                "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
            ),
            "data": "data/teacher_train_mix_v2p11_frozen1247",
            "data_len": 1247,
            "rank": 32,
            "alpha": 32,
            "lr": 2e-5,
            "epochs": 1.0,
            "bsz": 1,
            "grad_accum": 16,
            "max_seq": 32768,
            "warmup_steps": 8,
            "logging_steps": 20,
            "save_steps": 1,
            "save_total_limit": 6,
            "load_4bit": False,
            "gradient_checkpointing": "bounded_unsloth",
            "hybrid_checkpoint_policy": None,
            "bounded_unsloth_host_buffer_policy": {
                "buffer_count": 200,
                "initial_buffer_elements": 128 * 1024,
                "pageable_buffers": 200,
                "recycle_after_backward": True,
                "cuda_synchronized": True,
                "host_cache_drained": True,
            },
            "selective_assistant_loss": True,
            "unsloth_compile_disabled": False,
            "unsloth_double_buffer_disabled": True,
            "torchdynamo_disabled": False,
            "torch_compile_disabled": False,
        }
        if (
            stage_marker.get("complete") is not True
            or stage_marker.get("optimizer_steps") != 78
            or stage_marker.get("adapter_sha256")
            != _binding(stage_adapter)["sha256"]
            or stage_marker.get("run_manifest_sha256")
            != _binding(artifacts["run_manifest.json"])["sha256"]
            or stage_config.get("r") != 32
            or stage_config.get("lora_alpha") != 32
        ):
            raise ValueError(
                "recovery inputs do not bind the completed Stage-A adapter"
            )
        if any(
            stage_run_manifest.get(key) != value
            for key, value in expected_stage_run.items()
        ):
            raise ValueError("Stage-A run manifest is incomplete")
    elif expected_phase == "recovery_sft":
        _require_artifact_names(
            artifacts,
            {
                "v2p11_recovery_sft_inputs.json",
                "adapter_model.safetensors",
                "adapter_config.json",
                "run_manifest.json",
                "trainer_state.json",
            },
            phase=expected_phase,
        )
        _validate_phase_marker(
            artifacts["v2p11_recovery_sft_inputs.json"],
            expected_phase="recovery_sft_inputs",
        )
        state = _read_object(artifacts["trainer_state.json"])
        manifest = _read_object(artifacts["run_manifest.json"])
        config = _read_object(artifacts["adapter_config.json"])
        if (
            state.get("global_step") != 105
            or state.get("max_steps") != 105
            or manifest.get("data")
            != "data/v2p11_portable_recovery138_targeted"
            or manifest.get("init_adapter")
            != "adapters/teacher_sft_v2p11_bf16"
            or manifest.get("max_seq") != 32768
            or manifest.get("load_4bit") is not False
            or config.get("r") != 32
            or config.get("lora_alpha") != 32
        ):
            raise ValueError("recovery SFT phase evidence is incomplete")
    elif expected_phase == "recovery_merge":
        _validate_model_phase(
            report,
            audit_name="v2p11_recovery_merge_audit.json",
            source_markers={
                "v2p11_recovery_sft_complete.json": "recovery_sft",
            },
        )
    elif expected_phase == "kto_full_inputs":
        _require_artifact_names(
            artifacts,
            {
                "v2p11_recovery_merge_complete.json",
                "v2p11_behavior_kto_v2.jsonl",
                "v2p11_behavior_kto_v2_manifest.json",
            },
            phase=expected_phase,
        )
        _validate_phase_marker(
            artifacts["v2p11_recovery_merge_complete.json"],
            expected_phase="recovery_merge",
        )
    elif expected_phase == "kto_full":
        _require_artifact_names(
            artifacts,
            {
                "v2p11_kto_inputs.json",
                "adapter_model.safetensors",
                "adapter_config.json",
                "training_evidence.json",
                "trainer_state.json",
            },
            phase=expected_phase,
        )
        input_report = _validate_phase_marker(
            artifacts["v2p11_kto_inputs.json"],
            expected_phase="kto_full_inputs",
        )
        _validate_kto_evidence(
            artifacts["training_evidence.json"],
            input_report=input_report,
        )
        state = _read_object(artifacts["trainer_state.json"])
        config = _read_object(artifacts["adapter_config.json"])
        if (
            state.get("global_step") != 25
            or state.get("max_steps") != 25
            or config.get("r") != 32
            or config.get("lora_alpha") != 32
        ):
            raise ValueError("KTO phase evidence is incomplete")
    elif expected_phase == "final_merge":
        _validate_model_phase(
            report,
            audit_name="v2p11_final_merge_audit.json",
            source_markers={
                "v2p11_recovery_merge_complete.json": "recovery_merge",
                "v2p11_kto_complete.json": "kto_full",
            },
        )
    else:
        raise ValueError(f"unsupported v2.11 phase marker: {expected_phase}")
    return report


def _phase_artifact_sha256(
    report: Mapping[str, Any],
    basename: str,
) -> str:
    matches = [
        artifact
        for artifact in report["artifacts"]
        if Path(artifact["path"]).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"phase marker is missing {basename}")
    return str(matches[0]["sha256"])


def _phase_artifact_path(
    report: Mapping[str, Any],
    basename: str,
) -> Path:
    artifacts = _phase_artifacts(report)
    if basename not in artifacts:
        raise ValueError(f"phase marker is missing {basename}")
    return artifacts[basename].resolve()


def _validate_bound_contract_semantics(
    *,
    training_contract_path: Path,
    recovery_contract_path: Path,
    posttrain_contract_path: Path,
    stage_data_contract_path: Path,
    stage_marker_path: Path,
    v2p10_composite_path: Path,
) -> dict[str, dict[str, Any]]:
    training = _read_object(training_contract_path)
    recovery = _read_object(recovery_contract_path)
    posttrain = _read_object(posttrain_contract_path)
    stage_data = _read_object(stage_data_contract_path)
    stage_marker = _read_object(stage_marker_path)
    _require_fields(
        training,
        {
            "rows": 1247,
            "base_rows": 1211,
            "fable_rows": 36,
            "max_seq": 32768,
            "optimizer_steps": 78,
            "v2p10_full300_sha256": _binding(
                v2p10_composite_path
            )["sha256"],
        },
        label="stage training",
    )
    _require_fields(
        recovery,
        {
            "rows": 46,
            "evaluation_overlap": 0,
            "source_counts": _FABLE_SOURCE_COUNTS,
            "combined_exclusion_sha256": posttrain.get(
                "exclusion_sha256"
            ),
            "combined_exclusion_ids": 707,
            "excluded_repositories": 12,
        },
        label="portable recovery",
    )
    _require_fields(
        posttrain,
        {
            "recovery_rows": 138,
            "recovery_optimizer_steps": 105,
            "behavior_rows": 606,
            "behavior_negative_counts": _BEHAVIOR_NEGATIVE_COUNTS,
            "full_evaluation_ids": 300,
            "combined_exclusion_ids": 707,
            "excluded_repositories": 12,
            "evaluation_overlap": 0,
        },
        label="posttrain",
    )
    _require_fields(
        stage_data,
        {
            "rows": 1247,
            "base_rows": 1211,
            "new_fable_rows": 36,
            "dataset_manifest_sha256": training.get(
                "dataset_manifest_sha256"
            ),
            "train_jsonl_sha256": training.get("train_jsonl_sha256"),
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
                "sha256": posttrain.get("exclusion_sha256"),
            },
        },
        label="Stage-A data",
    )
    _require_fields(
        stage_marker,
        {
            "schema_version": 1,
            "complete": True,
            "data_manifest_sha256": training.get(
                "dataset_manifest_sha256"
            ),
            "train_jsonl_sha256": training.get("train_jsonl_sha256"),
            "optimizer_steps": 78,
        },
        label="stage marker",
    )
    required_hashes = {
        "training.dataset_manifest_sha256": training.get(
            "dataset_manifest_sha256"
        ),
        "training.train_jsonl_sha256": training.get(
            "train_jsonl_sha256"
        ),
        "stage.adapter_sha256": stage_marker.get("adapter_sha256"),
        "posttrain.recovery_manifest_sha256": posttrain.get(
            "recovery_manifest_sha256"
        ),
        "posttrain.behavior_manifest_sha256": posttrain.get(
            "behavior_manifest_sha256"
        ),
        "posttrain.exclusion_sha256": posttrain.get("exclusion_sha256"),
        "stage_data.strict_fable_manifest_sha256": stage_data.get(
            "strict_fable_manifest_sha256"
        ),
        "stage_data.campaign_manifest_sha256": stage_data.get(
            "campaign_manifest_sha256"
        ),
    }
    invalid_hashes = sorted(
        name
        for name, value in required_hashes.items()
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        )
    )
    if invalid_hashes:
        raise ValueError(
            f"completion contract hashes are invalid: {invalid_hashes}"
        )
    return {
        "training": training,
        "recovery": recovery,
        "posttrain": posttrain,
        "stage_data": stage_data,
        "stage_marker": stage_marker,
    }


def publish_completion_provenance(
    *,
    training_contract_path: Path,
    recovery_contract_path: Path,
    posttrain_contract_path: Path,
    stage_data_contract_path: Path,
    stage_marker_path: Path,
    recovery_marker_path: Path,
    kto_marker_path: Path,
    final_merge_marker_path: Path,
    posttrain_marker_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_contract_path: Path,
    final_model_path: Path,
    portability_gate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    from phaseH_eval.v2p10_training_lineage import (
        validate_v2p10_training_lineage_contract,
    )

    training_contract_path = Path(training_contract_path).resolve()
    recovery_contract_path = Path(recovery_contract_path).resolve()
    posttrain_contract_path = Path(posttrain_contract_path).resolve()
    stage_data_contract_path = Path(stage_data_contract_path).resolve()
    stage_marker_path = Path(stage_marker_path).resolve()
    recovery_marker_path = Path(recovery_marker_path).resolve()
    kto_marker_path = Path(kto_marker_path).resolve()
    final_merge_marker_path = Path(final_merge_marker_path).resolve()
    posttrain_marker_path = Path(posttrain_marker_path).resolve()
    full_ids_path = Path(full_ids_path).resolve()
    v2p10_composite_path = Path(v2p10_composite_path).resolve()
    v2p10_lineage_contract_path = Path(
        v2p10_lineage_contract_path
    ).resolve()
    portability_gate_path = Path(portability_gate_path).resolve()
    output_path = Path(output_path).resolve()
    contracts = _validate_bound_contract_semantics(
        training_contract_path=training_contract_path,
        recovery_contract_path=recovery_contract_path,
        posttrain_contract_path=posttrain_contract_path,
        stage_data_contract_path=stage_data_contract_path,
        stage_marker_path=stage_marker_path,
        v2p10_composite_path=v2p10_composite_path,
    )
    training = contracts["training"]
    recovery = contracts["recovery"]
    posttrain = contracts["posttrain"]
    stage_data = contracts["stage_data"]
    stage_marker = contracts["stage_marker"]
    v2p10_lineage = validate_v2p10_training_lineage_contract(
        v2p10_lineage_contract_path,
        composite_path=v2p10_composite_path,
    )
    recovery_marker = _validate_phase_marker(
        recovery_marker_path,
        expected_phase="recovery_sft",
    )
    kto_marker = _validate_phase_marker(
        kto_marker_path,
        expected_phase="kto_full",
    )
    final_merge_marker = _validate_phase_marker(
        final_merge_marker_path,
        expected_phase="final_merge",
    )
    recovery_input_path = _phase_artifact_path(
        recovery_marker,
        "v2p11_recovery_sft_inputs.json",
    )
    recovery_input = _read_current_phase_marker(
        recovery_input_path,
        expected_phase="recovery_sft_inputs",
    )
    if (
        _phase_artifact_path(
            recovery_input,
            "v2p11_training_complete.json",
        )
        != stage_marker_path
    ):
        raise ValueError(
            "recovery lineage does not bind the Stage-A marker"
        )
    kto_input_path = _phase_artifact_path(
        kto_marker,
        "v2p11_kto_inputs.json",
    )
    kto_input = _read_current_phase_marker(
        kto_input_path,
        expected_phase="kto_full_inputs",
    )
    recovery_merge_path = _phase_artifact_path(
        kto_input,
        "v2p11_recovery_merge_complete.json",
    )
    recovery_merge = _read_current_phase_marker(
        recovery_merge_path,
        expected_phase="recovery_merge",
    )
    if (
        _phase_artifact_path(
            recovery_merge,
            "v2p11_recovery_sft_complete.json",
        )
        != recovery_marker_path
    ):
        raise ValueError(
            "KTO lineage does not bind the recovery SFT marker"
        )
    if (
        _phase_artifact_path(
            final_merge_marker,
            "v2p11_recovery_merge_complete.json",
        )
        != recovery_merge_path
        or _phase_artifact_path(
            final_merge_marker,
            "v2p11_kto_complete.json",
        )
        != kto_marker_path
    ):
        raise ValueError(
            "final merge lineage does not bind the poststage markers"
        )
    recovery_manifest_path = _phase_artifact_path(
        recovery_input,
        "manifest.json",
    )
    behavior_manifest_path = _phase_artifact_path(
        kto_input,
        "v2p11_behavior_kto_v2_manifest.json",
    )
    posttrain_marker = _read_object(posttrain_marker_path)
    full_ids = _read_ids(full_ids_path)

    _require_fields(
        posttrain_marker,
        {
            "schema_version": 1,
            "complete": True,
            "stage_a_adapter_sha256": stage_marker.get(
                "adapter_sha256"
            ),
            "recovery_manifest_sha256": posttrain.get(
                "recovery_manifest_sha256"
            ),
            "behavior_manifest_sha256": posttrain.get(
                "behavior_manifest_sha256"
            ),
            "recovery_optimizer_steps": 105,
            "kto_optimizer_steps": 25,
            "recovery_input_marker_sha256": _binding(
                recovery_input_path
            )["sha256"],
            "kto_input_marker_sha256": _binding(
                kto_input_path
            )["sha256"],
            "kto_training_evidence_sha256": (
                _phase_artifact_sha256(
                    kto_marker,
                    "training_evidence.json",
                )
            ),
            "final_merge_audit_sha256": (
                _phase_artifact_sha256(
                    final_merge_marker,
                    "v2p11_final_merge_audit.json",
                )
            ),
            "final_merge_marker_sha256": _binding(
                final_merge_marker_path
            )["sha256"],
        },
        label="posttrain marker",
    )
    if len(full_ids) != 300:
        raise ValueError("completion provenance requires 300 evaluation IDs")
    if (
        _binding(recovery_manifest_path)["sha256"]
        != posttrain.get("recovery_manifest_sha256")
        or _binding(behavior_manifest_path)["sha256"]
        != posttrain.get("behavior_manifest_sha256")
    ):
        raise ValueError(
            "posttrain contract does not bind the phase input manifests"
        )
    final_model = _model_contract(final_model_path)
    final_merge_model_path = _phase_artifacts(
        final_merge_marker
    )["config.json"].parent.resolve()
    if final_merge_model_path != Path(final_model["model_path"]):
        raise ValueError(
            "final merge marker does not bind the final model path"
        )
    portability = _validate_portability_gate(
        portability_gate_path,
        final_model=final_model,
    )

    report = {
        "schema_version": 1,
        "artifact_type": "v2p11_completion_provenance",
        "status": "complete",
        "full_ids": _binding(full_ids_path),
        "v2p10_full300": _binding(v2p10_composite_path),
        "contracts": {
            "training": _binding(training_contract_path),
            "portable_recovery": _binding(recovery_contract_path),
            "posttrain": _binding(posttrain_contract_path),
            "stage_data": _binding(stage_data_contract_path),
            "v2p10_training_lineage": _binding(
                v2p10_lineage_contract_path
            ),
        },
        "markers": {
            "stage_a": _binding(stage_marker_path),
            "recovery": _binding(recovery_marker_path),
            "kto": _binding(kto_marker_path),
            "final_merge": _binding(final_merge_marker_path),
            "posttrain": _binding(posttrain_marker_path),
        },
        "portability": portability,
        "training": {
            "rows": 1247,
            "base_rows": 1211,
            "new_fable_rows": 36,
            "max_seq": 32768,
            "optimizer_steps": 78,
            "dataset_manifest_sha256": training[
                "dataset_manifest_sha256"
            ],
            "train_jsonl_sha256": training["train_jsonl_sha256"],
        },
        "v2p10_training": dict(v2p10_lineage["training"]),
        "fable": {
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
        },
        "behavior_kto": {
            "rows": 606,
            "negative_counts": _BEHAVIOR_NEGATIVE_COUNTS,
            "optimizer_steps": 25,
        },
        "evaluation_exclusion": {
            "full_ids": 300,
            "excluded_ids": 707,
            "excluded_repositories": 12,
            "stage_a_overlap": 0,
            "overlap": 0,
            "sha256": posttrain["exclusion_sha256"],
        },
        "final_model": final_model,
    }
    if os.path.lexists(output_path):
        if _read_object(output_path) != report:
            raise ValueError(
                "existing completion provenance differs from current inputs"
            )
    else:
        _publish_json_noreplace(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--posttrain-contract", type=Path, required=True)
    parser.add_argument("--stage-data-contract", type=Path, required=True)
    parser.add_argument("--stage-marker", type=Path, required=True)
    parser.add_argument("--recovery-marker", type=Path, required=True)
    parser.add_argument("--kto-marker", type=Path, required=True)
    parser.add_argument("--final-merge-marker", type=Path, required=True)
    parser.add_argument("--posttrain-marker", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument(
        "--v2p10-lineage-contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_completion_provenance(
        training_contract_path=args.training_contract,
        recovery_contract_path=args.recovery_contract,
        posttrain_contract_path=args.posttrain_contract,
        stage_data_contract_path=args.stage_data_contract,
        stage_marker_path=args.stage_marker,
        recovery_marker_path=args.recovery_marker,
        kto_marker_path=args.kto_marker,
        final_merge_marker_path=args.final_merge_marker,
        posttrain_marker_path=args.posttrain_marker,
        full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10,
        v2p10_lineage_contract_path=args.v2p10_lineage_contract,
        final_model_path=args.final_model,
        portability_gate_path=args.portability_gate,
        output_path=args.out,
    )
    print(json.dumps({
        "status": report["status"],
        "new_fable_traces": report["fable"]["stage_a"][
            "new_strict_rows"
        ],
        "stage_a_fable_rows": report["fable"]["stage_a"]["total_rows"],
        "recovery_unique_fable_sources": report["fable"][
            "recovery_unique_sources"
        ],
        "evaluation_overlap": report["evaluation_exclusion"]["overlap"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
