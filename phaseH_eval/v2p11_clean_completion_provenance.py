#!/usr/bin/env python3
"""Publish and revalidate the fresh raw-base v2.11 Fable-51 lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
)
from phaseH_eval.v2p10_training_lineage import (
    validate_v2p10_training_lineage_contract,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract
from phaseH_eval.v2p11_stage_data_contract import (
    validate_stage_data as _validate_stage_data,
)
from phaseH_eval.v2p11r2_completion_provenance import (
    PORTABILITY_CRITERIA,
    _MERGE_AUDIT,
    _validate_dataset as _validate_outer_dataset,
)


EXPECTED_ROWS = 1_262
EXPECTED_BASE_ROWS = 1_211
EXPECTED_STAGE_ROWS = 36
EXPECTED_LATE_ROWS = 15
EXPECTED_FULL_IDS = 300
EXPECTED_V2P10_RESOLVED = 157
EXPECTED_BASE_MODEL = "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
EXPECTED_TRAIN_UNIT = "v2p11-clean-fable51-train-gpu1-v2.service"
EXPECTED_GPU_IDENTITY_ARTIFACT_TYPE = "v2p11_clean_gpu_training_identity"
EXPECTED_TRAINING_COMPLETION_ARTIFACT_TYPE = "v2p11_clean_training_completion"
EXPECTED_MANIFEST_SHA256 = (
    "40531f44c8d5ab1d47a179418aa1adaf1ca31d0f0265f262b15c5fd424b4758a"
)
EXPECTED_TRAIN_SHA256 = (
    "3d131c531ca060ad2472304b95b423f292cefde94cf4538788ef52ce36b919c1"
)
EXPECTED_BASE_TRAIN_SHA256 = (
    "6fbd3212ddf200428587025e70f080b0b6250e04dc192149517553c5124c3398"
)
SOURCE_CONTEXT_AUDIT = Path(
    "runs/v2p11r2_v2p10init_fable51_dataset_context_audit.json"
)
STAGE_DATA = Path("data/teacher_train_mix_v2p11_frozen1247")
STAGE_TRAINING_CONTRACT = Path("runs/v2p11_completion_training_contract.json")
STAGE_EXCLUSIONS = Path("data/swe_all_eval_exclusions_v2.json")
STAGE_CAMPAIGN = Path("runs/teacher_fable5_v2p11_frozen47_merged")
_SOURCE_COUNTS = {
    "teacher:claude:claude-fable-5": EXPECTED_STAGE_ROWS,
    "fable5_verified_finalpatch": EXPECTED_LATE_ROWS,
}
_RUN_CONTRACT = {
    "data_len": EXPECTED_ROWS,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-5,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "max_steps": 79,
    "warmup_steps": 8,
    "logging_steps": 5,
    "save_steps": 5,
    "save_total_limit": 3,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}


def _require_current_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is incomplete")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} binding path is incomplete")
    current = _binding(Path(raw_path).resolve())
    if (
        value.get("sha256") != current["sha256"]
        or value.get("bytes") != current["bytes"]
    ):
        raise ValueError(f"{label} binding changed")
    return current


def _resolve_run_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("training run path is incomplete")
    path = Path(value)
    return (path if path.is_absolute() else Path.cwd() / path).resolve()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid training JSONL: {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"invalid training rows: {path}")
    return values


def _validate_dataset(
    dataset: Path,
    base_data: Path,
    context_audit_path: Path,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    base_data = Path(base_data).resolve()
    manifest_path = dataset / "manifest.json"
    train_path = dataset / "train.jsonl"
    base_manifest_path = base_data / "manifest.json"
    base_train_path = base_data / "train.jsonl"
    source_context = Path(SOURCE_CONTEXT_AUDIT).resolve()

    outer = _validate_outer_dataset(dataset, source_context)
    rows = _read_rows(train_path)
    base_rows = _read_rows(base_train_path)
    manifest_binding = _binding(manifest_path)
    train_binding = _binding(train_path)
    base_train_binding = _binding(base_train_path)
    if (
        manifest_binding["sha256"] != EXPECTED_MANIFEST_SHA256
        or train_binding["sha256"] != EXPECTED_TRAIN_SHA256
        or base_train_binding["sha256"] != EXPECTED_BASE_TRAIN_SHA256
        or len(rows) != EXPECTED_ROWS
        or len(base_rows) != EXPECTED_BASE_ROWS
    ):
        raise ValueError("clean dataset immutable bindings changed")
    if rows[:EXPECTED_BASE_ROWS] != base_rows:
        raise ValueError("clean dataset is not the exact v2.10 prefix")
    instance_ids = [row.get("instance_id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in instance_ids)
        or len(set(instance_ids)) != len(instance_ids)
    ):
        raise ValueError("clean dataset instance identities are incomplete")
    added = rows[EXPECTED_BASE_ROWS:]
    source_counts = dict(
        sorted(Counter(row.get("source") for row in added).items())
    )
    expected_counts = {
        "teacher:claude:claude-fable-5": EXPECTED_STAGE_ROWS,
        "fable5_verified_finalpatch": EXPECTED_LATE_ROWS,
    }
    if source_counts != expected_counts:
        raise ValueError("clean dataset Fable source split changed")
    outer_source_ids = outer.get("source_ids")
    if (
        not isinstance(outer_source_ids, list)
        or len(outer_source_ids) != EXPECTED_LATE_ROWS
        or len(set(outer_source_ids)) != EXPECTED_LATE_ROWS
    ):
        raise ValueError("clean dataset late-Fable replay evidence is incomplete")

    stage_data = Path(STAGE_DATA).resolve()
    stage_contract = Path(STAGE_TRAINING_CONTRACT).resolve()
    stage_exclusions = Path(STAGE_EXCLUSIONS).resolve()
    stage_campaign = Path(STAGE_CAMPAIGN).resolve()
    stage = _validate_stage_data(
        stage_data=stage_data,
        training_contract_path=stage_contract,
        exclusions_path=stage_exclusions,
        campaign_root=stage_campaign,
        require_production_identity=True,
    )
    if {
        key: stage.get(key) for key in ("rows", "base_rows", "new_fable_rows")
    } != {
        "rows": EXPECTED_BASE_ROWS + EXPECTED_STAGE_ROWS,
        "base_rows": EXPECTED_BASE_ROWS,
        "new_fable_rows": EXPECTED_STAGE_ROWS,
    }:
        raise ValueError("clean dataset Stage-A Fable evidence is incomplete")

    context_path = Path(context_audit_path).resolve()
    context = _read_object(context_path)
    max_rendered = context.get("max_rendered_tokens")
    if (
        context.get("schema_version") != 1
        or context.get("artifact_type")
        != "v2p11_clean_fable51_context_audit"
        or context.get("status") != "complete"
        or context.get("dataset_manifest_sha256")
        != manifest_binding["sha256"]
        or context.get("train_jsonl_sha256") != train_binding["sha256"]
        or context.get("base_train_jsonl_sha256")
        != base_train_binding["sha256"]
        or context.get("rows") != EXPECTED_ROWS
        or context.get("base_rows") != EXPECTED_BASE_ROWS
        or context.get("new_fable_rows")
        != EXPECTED_STAGE_ROWS + EXPECTED_LATE_ROWS
        or context.get("source_counts") != expected_counts
        or context.get("max_seq") != 32768
        or type(max_rendered) is not int
        or not 0 < max_rendered <= 32768
        or context.get("over_limit_rows") != 0
        or context.get("optimizer_steps") != 79
        or context.get("init_adapter") is not None
    ):
        raise ValueError("clean full-context audit is incomplete")

    return {
        "path": str(dataset),
        "manifest": manifest_binding,
        "train": train_binding,
        "base_data": {
            "path": str(base_data),
            "manifest": _binding(base_manifest_path),
            "train": base_train_binding,
        },
        "rows": EXPECTED_ROWS,
        "base_rows": EXPECTED_BASE_ROWS,
        "new_fable_rows": EXPECTED_STAGE_ROWS + EXPECTED_LATE_ROWS,
        "source_counts": source_counts,
        "init_adapter": None,
        "max_rendered_tokens": max_rendered,
        "context_audit": _binding(context_path),
        "source_context_audit": _binding(source_context),
        "late_fable": outer,
        "stage_a": {
            "summary": {
                key: stage[key]
                for key in ("rows", "base_rows", "new_fable_rows")
            },
            "data_manifest": _binding(stage_data / "manifest.json"),
            "training_contract": _binding(stage_contract),
            "exclusions": _binding(stage_exclusions),
            "campaign_manifest": _binding(stage_campaign / "manifest.json"),
        },
    }


def _validate_gpu_identity(
    identity_path: Path,
    *,
    train_pid: int,
    invocation_id: str,
) -> dict[str, Any]:
    identity_path = Path(identity_path).resolve()
    identity = _read_object(identity_path)
    gpu_pids = identity.get("gpu_compute_pids")
    command_sha = identity.get("trainer_cmdline_sha256")
    if (
        identity.get("schema_version") != 1
        or identity.get("artifact_type")
        != EXPECTED_GPU_IDENTITY_ARTIFACT_TYPE
        or identity.get("status") != "complete"
        or identity.get("train_unit") != EXPECTED_TRAIN_UNIT
        or identity.get("train_invocation_id") != invocation_id
        or identity.get("train_pid") != train_pid
        or identity.get("gpu_index") != 1
        or not isinstance(identity.get("gpu_uuid"), str)
        or not identity["gpu_uuid"].startswith("GPU-")
        or identity.get("cuda_visible_devices") != "1"
        or identity.get("train_pid_on_gpu") is not True
        or not isinstance(gpu_pids, list)
        or any(type(pid) is not int or pid <= 0 for pid in gpu_pids)
        or len(set(gpu_pids)) != len(gpu_pids)
        or train_pid not in gpu_pids
        or not isinstance(command_sha, str)
        or len(command_sha) != 64
        or any(character not in "0123456789abcdef" for character in command_sha)
    ):
        raise ValueError("clean training does not prove physical GPU1 execution")
    return {
        "artifact": _binding(identity_path),
        "gpu_index": 1,
        "gpu_uuid": identity["gpu_uuid"],
        "cuda_visible_devices": "1",
        "gpu_compute_pids": list(gpu_pids),
        "train_pid_on_gpu": True,
        "trainer_cmdline_sha256": command_sha,
    }


def _validate_training(
    *,
    dataset: Path,
    adapter: Path,
    completion_path: Path,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    adapter = Path(adapter).resolve()
    run_path = adapter / "run_manifest.json"
    run = _read_object(run_path)
    if (
        run.get("base") != EXPECTED_BASE_MODEL
        or any(run.get(key) != value for key, value in _RUN_CONTRACT.items())
        or _resolve_run_path(run.get("data")) != dataset
        or _resolve_run_path(run.get("out")) != adapter
        or run.get("init_adapter") is not None
    ):
        raise ValueError("clean training run contract is incomplete")

    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    checkpoint_path = adapter / "checkpoint-79" / "trainer_state.json"
    config = _read_object(config_path)
    checkpoint = _read_object(checkpoint_path)
    log_history = checkpoint.get("log_history")
    finite_losses = [
        item.get("loss")
        for item in log_history
        if isinstance(item, Mapping)
        and isinstance(item.get("loss"), (int, float))
        and math.isfinite(float(item["loss"]))
    ] if isinstance(log_history, list) else []
    finite_grad_norms = [
        item.get("grad_norm")
        for item in log_history
        if isinstance(item, Mapping)
        and isinstance(item.get("grad_norm"), (int, float))
        and math.isfinite(float(item["grad_norm"]))
    ] if isinstance(log_history, list) else []
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 32
        or checkpoint.get("global_step") != 79
        or checkpoint.get("max_steps") != 79
        or not finite_losses
        or not finite_grad_norms
    ):
        raise ValueError("clean adapter checkpoint is incomplete")

    completion_path = Path(completion_path).resolve()
    completion = _read_object(completion_path)
    adapter_binding = _require_current_binding(
        completion.get("adapter"), label="clean adapter"
    )
    config_binding = _require_current_binding(
        completion.get("adapter_config"), label="clean adapter config"
    )
    run_binding = _require_current_binding(
        completion.get("run_manifest"), label="clean run manifest"
    )
    checkpoint_binding = _require_current_binding(
        completion.get("checkpoint_state"), label="clean checkpoint state"
    )
    journal_binding = _require_current_binding(
        completion.get("training_journal"), label="clean training journal"
    )
    watchdog_binding = _require_current_binding(
        completion.get("watchdog_log"), label="clean watchdog log"
    )
    unit_journal_binding = _require_current_binding(
        completion.get("unit_journal"), label="clean unit journal"
    )
    train_pid = completion.get("train_pid")
    invocation_id = completion.get("train_invocation_id")
    if (
        completion.get("schema_version") != 1
        or completion.get("artifact_type")
        != EXPECTED_TRAINING_COMPLETION_ARTIFACT_TYPE
        or completion.get("status") != "complete"
        or completion.get("base_model") != EXPECTED_BASE_MODEL
        or completion.get("init_adapter") is not None
        or completion.get("train_unit") != EXPECTED_TRAIN_UNIT
        or not isinstance(invocation_id, str)
        or len(invocation_id) != 32
        or any(character not in "0123456789abcdef" for character in invocation_id)
        or any(
            type(completion.get(key)) is not int or completion[key] <= 0
            for key in ("train_wrapper_pid", "train_pid", "watchdog_pid")
        )
        or completion.get("optimizer_steps") != 79
        or completion.get("max_steps") != 79
        or completion.get("adapter_tensor_count") != 820
        or completion.get("lora_a_tensor_count") != 410
        or completion.get("lora_b_tensor_count") != 410
        or completion.get("nonfinite_tensor_count") != 0
        or type(completion.get("nonzero_lora_b_tensor_count")) is not int
        or completion["nonzero_lora_b_tensor_count"] < 1
        or completion.get("finite_loss_samples") != len(finite_losses)
        or completion.get("finite_grad_norm_samples") != len(finite_grad_norms)
        or Path(adapter_binding["path"]) != weights_path
        or Path(config_binding["path"]) != config_path
        or Path(run_binding["path"]) != run_path
        or Path(checkpoint_binding["path"]) != checkpoint_path
    ):
        raise ValueError("clean training completion is incomplete")
    gpu_identity_binding = _require_current_binding(
        completion.get("gpu_identity"), label="clean GPU identity"
    )
    gpu_identity = _validate_gpu_identity(
        Path(gpu_identity_binding["path"]),
        train_pid=train_pid,
        invocation_id=invocation_id,
    )
    journal = Path(journal_binding["path"]).read_text(encoding="utf-8")
    watchdog = Path(watchdog_binding["path"]).read_text(encoding="utf-8")
    try:
        unit_rows = [
            json.loads(line)
            for line in Path(unit_journal_binding["path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("clean unit journal is invalid") from error
    if (
        "optimizer_step=79/79" not in journal
        or "[done] adapter saved" not in journal
        or f"event=target_exited pid={train_pid}" not in watchdog
        or f"event=below_threshold pid={train_pid}" in watchdog
        or f"event=meminfo_error pid={train_pid}" in watchdog
        or f"event=health_failure pid={train_pid}" in watchdog
        or not any(
            isinstance(row, Mapping)
            and row.get("_SYSTEMD_INVOCATION_ID") == invocation_id
            for row in unit_rows
        )
    ):
        raise ValueError("clean training logs are incomplete")
    return {
        "run_manifest": run_binding,
        "adapter": adapter_binding,
        "adapter_config": config_binding,
        "checkpoint_state": checkpoint_binding,
        "completion": _binding(completion_path),
        "training_journal": journal_binding,
        "watchdog_log": watchdog_binding,
        "unit_journal": unit_journal_binding,
        "gpu_identity": gpu_identity,
        "train_unit": completion["train_unit"],
        "train_invocation_id": invocation_id,
        "train_wrapper_pid": completion["train_wrapper_pid"],
        "train_pid": train_pid,
        "watchdog_pid": completion["watchdog_pid"],
        "finite_loss_samples": len(finite_losses),
        "finite_grad_norm_samples": len(finite_grad_norms),
        "adapter_tensor_count": completion["adapter_tensor_count"],
        "nonzero_lora_b_tensor_count": completion[
            "nonzero_lora_b_tensor_count"
        ],
        "init_adapter": None,
    }


def _validate_v2p10_lineage(
    *,
    lineage_path: Path,
    v2p10_composite_path: Path,
    base_data_path: Path,
) -> dict[str, Any]:
    lineage_path = Path(lineage_path).resolve()
    lineage = validate_v2p10_training_lineage_contract(
        lineage_path,
        composite_path=Path(v2p10_composite_path).resolve(),
    )
    artifacts = lineage.get("artifacts")
    base_data = Path(base_data_path).resolve()
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("dataset_manifest")
        != _binding(base_data / "manifest.json")
        or artifacts.get("train_jsonl") != _binding(base_data / "train.jsonl")
    ):
        raise ValueError("clean dataset does not use canonical v2.10 training data")
    return {
        "contract": _binding(lineage_path),
        "dataset_manifest": dict(artifacts["dataset_manifest"]),
        "train_jsonl": dict(artifacts["train_jsonl"]),
        "training": dict(lineage["training"]),
        "model_contract": dict(lineage["model_contract"]),
    }


def _validate_final(
    *,
    candidate_name: str,
    final_model: Path,
    merge_audit: Path,
    portability_gate: Path,
) -> dict[str, Any]:
    final_model = Path(final_model).resolve()
    merge_audit = Path(merge_audit).resolve()
    portability_gate = Path(portability_gate).resolve()
    gate_contract = _model_contract(final_model)
    model_contract = {**gate_contract, "served_name": candidate_name}
    gate = _read_object(portability_gate)
    candidate = gate.get("candidate")
    if (
        _read_object(merge_audit) != _MERGE_AUDIT
        or gate.get("schema_version") != 1
        or gate.get("artifact_type")
        != "v2p11_controller_free_portability_gate"
        or gate.get("status") != "complete"
        or gate.get("passed") is not True
        or gate.get("control_name") != "teacher_sft_v2p10"
        or gate.get("candidate_name") != candidate_name
        or gate.get("population") != 10
        or gate.get("evaluation_overlap") != 0
        or gate.get("criteria") != PORTABILITY_CRITERIA
        or not isinstance(candidate, Mapping)
        or candidate.get("model_contract") != gate_contract
    ):
        raise ValueError("clean final model or portability gate is incomplete")
    return {
        "model_contract": model_contract,
        "merge_audit": _binding(merge_audit),
        "portability_gate": _binding(portability_gate),
    }


def _build_report(
    *,
    candidate_name: str,
    dataset_path: Path,
    base_data_path: Path,
    dataset_context_audit_path: Path,
    adapter_path: Path,
    training_completion_path: Path,
    final_model_path: Path,
    merge_audit_path: Path,
    portability_gate_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
) -> dict[str, Any]:
    if not candidate_name or candidate_name == "teacher_sft_v2p10":
        raise ValueError("clean v2.11 candidate name is invalid")
    full_ids_path = Path(full_ids_path).resolve()
    v2p10_composite_path = Path(v2p10_composite_path).resolve()
    ids = _read_ids(full_ids_path)
    control = _read_object(v2p10_composite_path)
    if (
        len(ids) != EXPECTED_FULL_IDS
        or len(set(ids)) != EXPECTED_FULL_IDS
        or control.get("status") != "complete"
        or control.get("name") != "teacher_sft_v2p10"
        or control.get("expected") != EXPECTED_FULL_IDS
        or control.get("resolved") != EXPECTED_V2P10_RESOLVED
    ):
        raise ValueError("canonical v2.10 full300 control is incomplete")
    dataset = _validate_dataset(
        dataset_path, base_data_path, dataset_context_audit_path
    )
    training = _validate_training(
        dataset=dataset_path,
        adapter=adapter_path,
        completion_path=training_completion_path,
    )
    lineage = _validate_v2p10_lineage(
        lineage_path=v2p10_lineage_path,
        v2p10_composite_path=v2p10_composite_path,
        base_data_path=base_data_path,
    )
    final = _validate_final(
        candidate_name=candidate_name,
        final_model=final_model_path,
        merge_audit=merge_audit_path,
        portability_gate=portability_gate_path,
    )
    return {
        "schema_version": 1,
        "artifact_type": "v2p11_clean_completion_provenance",
        "status": "complete",
        "candidate_name": candidate_name,
        "lineage": {
            "kind": "fresh_raw_base_lora",
            "base_model": EXPECTED_BASE_MODEL,
            "init_adapter": None,
            "recovery_sft": False,
            "kto": False,
            "interpolation": False,
        },
        "full_ids": _binding(full_ids_path),
        "v2p10_full300": _binding(v2p10_composite_path),
        "v2p10_training_lineage": lineage,
        "dataset": dataset,
        "training": {
            "rows": EXPECTED_ROWS,
            "base_rows": EXPECTED_BASE_ROWS,
            "new_fable_rows": EXPECTED_STAGE_ROWS + EXPECTED_LATE_ROWS,
            "max_seq": 32768,
            "optimizer_steps": 79,
            **training,
        },
        "final_model": final["model_contract"],
        "merge_audit": final["merge_audit"],
        "portability": {
            "gate": final["portability_gate"],
            "passed": True,
            "population": 10,
            "criteria": dict(PORTABILITY_CRITERIA),
        },
    }


def publish_completion_provenance(
    *,
    candidate_name: str,
    dataset_path: Path,
    base_data_path: Path,
    dataset_context_audit_path: Path,
    adapter_path: Path,
    training_completion_path: Path,
    final_model_path: Path,
    merge_audit_path: Path,
    portability_gate_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = _build_report(
        candidate_name=candidate_name,
        dataset_path=dataset_path,
        base_data_path=base_data_path,
        dataset_context_audit_path=dataset_context_audit_path,
        adapter_path=adapter_path,
        training_completion_path=training_completion_path,
        final_model_path=final_model_path,
        merge_audit_path=merge_audit_path,
        portability_gate_path=portability_gate_path,
        full_ids_path=full_ids_path,
        v2p10_composite_path=v2p10_composite_path,
        v2p10_lineage_path=v2p10_lineage_path,
    )
    _publish_json_noreplace(Path(output_path).resolve(), report)
    return report


def _model_contracts_match(
    candidate: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    candidate_value = dict(candidate)
    expected_value = dict(expected)
    for key in ("model_index_sha256", "model_safetensors_sha256"):
        if key not in candidate_value and expected_value.get(key) is None:
            candidate_value[key] = None
        if key not in expected_value and candidate_value.get(key) is None:
            expected_value[key] = None
    return candidate_value == expected_value


def validate_completion_provenance(
    provenance_path: Path,
    *,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
    candidate_model_contract: Mapping[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    report = _read_object(Path(provenance_path).resolve())
    dataset = report.get("dataset")
    training = report.get("training")
    lineage = report.get("v2p10_training_lineage")
    final_model = report.get("final_model")
    portability = report.get("portability")
    if (
        report.get("artifact_type")
        != "v2p11_clean_completion_provenance"
        or report.get("candidate_name") != candidate_name
        or not isinstance(dataset, Mapping)
        or not isinstance(dataset.get("base_data"), Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(lineage, Mapping)
        or not isinstance(lineage.get("contract"), Mapping)
        or not isinstance(final_model, Mapping)
        or not isinstance(portability, Mapping)
        or not isinstance(portability.get("gate"), Mapping)
    ):
        raise ValueError("clean v2.11 completion provenance is incomplete")
    expected = _build_report(
        candidate_name=candidate_name,
        dataset_path=Path(str(dataset.get("path", ""))),
        base_data_path=Path(str(dataset["base_data"].get("path", ""))),
        dataset_context_audit_path=Path(
            str(dataset.get("context_audit", {}).get("path", ""))
        ),
        adapter_path=Path(
            str(training.get("adapter", {}).get("path", ""))
        ).parent,
        training_completion_path=Path(
            str(training.get("completion", {}).get("path", ""))
        ),
        final_model_path=Path(str(final_model.get("model_path", ""))),
        merge_audit_path=Path(str(report.get("merge_audit", {}).get("path", ""))),
        portability_gate_path=Path(str(portability["gate"].get("path", ""))),
        full_ids_path=Path(full_ids_path),
        v2p10_composite_path=Path(v2p10_composite_path),
        v2p10_lineage_path=Path(v2p10_lineage_path).resolve(),
    )
    if report != expected:
        raise ValueError("clean v2.11 completion provenance changed")
    if not _model_contracts_match(
        candidate_model_contract, expected["final_model"]
    ):
        raise ValueError("clean provenance does not bind evaluated model")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--dataset-context-audit", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--v2p10-lineage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_completion_provenance(
        candidate_name=args.candidate_name,
        dataset_path=args.dataset,
        base_data_path=args.base_data,
        dataset_context_audit_path=args.dataset_context_audit,
        adapter_path=args.adapter,
        training_completion_path=args.training_completion,
        final_model_path=args.final_model,
        merge_audit_path=args.merge_audit,
        portability_gate_path=args.portability_gate,
        full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10,
        v2p10_lineage_path=args.v2p10_lineage,
        output_path=args.out,
    )
    print(
        json.dumps(
            {"status": report["status"], "candidate_name": report["candidate_name"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
