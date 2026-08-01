#!/usr/bin/env python3
"""Publish and revalidate the direct-LoRA v2.11r2 training lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract
from teacher_platform.generic_trace_replay import (
    admission_is_exact,
    canonical_json_bytes,
)


EXPECTED_DATASET_ROWS = 1_262
EXPECTED_BASE_ROWS = 1_247
EXPECTED_NEW_FABLE_ROWS = 15
EXPECTED_FULL_IDS = 300
EXPECTED_V2P10_RESOLVED = 157
EXPECTED_BASE_MODEL = (
    "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
)
PORTABILITY_CRITERIA = {
    "candidate_empty_at_most_one": True,
    "candidate_empty_no_regression": True,
    "candidate_no_format_regression": True,
    "candidate_no_loop_regression": True,
    "candidate_resolution_floor": True,
}
_MERGE_AUDIT = {
    "schema_version": 1,
    "architecture": "Gemma4ForConditionalGeneration",
    "expected_tensors": 1188,
    "actual_tensors": 1188,
    "expected_vision": 356,
    "actual_vision": 356,
    "missing_tensors": [],
    "unexpected_tensors": [],
    "misplaced_tensors": [],
    "nonfinite_tensors": [],
    "complete": True,
}
_RUN_CONTRACT = {
    "data_len": EXPECTED_DATASET_ROWS,
    "rank": 32,
    "alpha": 32,
    "lr": 0.000002,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "max_steps": 79,
    "warmup_steps": 4,
    "logging_steps": 5,
    "save_steps": 10,
    "save_total_limit": 3,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}


def _resolve_bound_path(
    value: Mapping[str, Any],
    *,
    base: Path | None,
    label: str,
) -> Path:
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} binding is incomplete")
    path = Path(raw)
    if not path.is_absolute():
        if base is None:
            raise ValueError(f"{label} binding path is not absolute")
        path = base / path
    return path.resolve()


def _require_current_binding(
    value: object,
    *,
    base: Path | None = None,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is incomplete")
    path = _resolve_bound_path(value, base=base, label=label)
    current = _binding(path)
    if (
        value.get("sha256") != current["sha256"]
        or value.get("bytes") != current["bytes"]
    ):
        raise ValueError(f"{label} binding changed")
    return current


def _read_jsonl_line(path: Path, line_number: int) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= line_number <= len(lines):
        raise ValueError("raw admission result line is invalid")
    value = json.loads(lines[line_number - 1])
    if not isinstance(value, dict):
        raise ValueError("raw admission result is invalid")
    return value


def _validate_late_fable_training_target(
    *,
    source_id: str,
    row: Mapping[str, Any],
    raw_patch: Mapping[str, Any],
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source_id} training messages are invalid")
    supervised = [
        message
        for message in messages
        if isinstance(message, Mapping)
        and message.get("role") == "assistant"
        and message.get("loss") is True
    ]
    if len(supervised) != 1:
        raise ValueError(f"{source_id} has no unique supervised final patch")
    calls = supervised[0].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError(f"{source_id} has no unique supervised patch call")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        raise ValueError(f"{source_id} supervised patch call is invalid")
    arguments = function.get("arguments")
    if function.get("name") != "bash" or not isinstance(arguments, str):
        raise ValueError(f"{source_id} supervised patch call is invalid")
    try:
        command_value = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source_id} supervised patch arguments are invalid") from error
    command = command_value.get("command") if isinstance(command_value, Mapping) else None
    match = (
        re.fullmatch(r"git apply - <<'([^']+)'\n(.*)\n\1", command, flags=re.DOTALL)
        if isinstance(command, str)
        else None
    )
    if match is None:
        raise ValueError(f"{source_id} supervised patch does not reproduce raw patch")
    raw_path = _resolve_bound_path(raw_patch, base=None, label=f"{source_id} patch")
    expected = raw_path.read_text(encoding="utf-8")
    candidate = match.group(2) + "\n"
    if candidate != expected:
        raise ValueError(f"{source_id} supervised patch does not reproduce raw patch")
    return {
        "source_instance_id": source_id,
        "raw_patch": _binding(raw_path),
        "supervised_patch_sha256": hashlib.sha256(
            candidate.encode("utf-8")
        ).hexdigest(),
        "exact_patch": True,
    }


def _validate_dataset(
    dataset: Path,
    dataset_context_audit: Path,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    manifest_path = dataset / "manifest.json"
    train_path = dataset / "train.jsonl"
    manifest = _read_object(manifest_path)
    try:
        train_rows = [
            json.loads(line)
            for line in train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("v2.11r2 training JSONL is invalid") from error
    if any(not isinstance(row, Mapping) for row in train_rows):
        raise ValueError("v2.11r2 training rows are invalid")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("dataset_variant")
        != "teacher_train_mix_v2p11_fable_extension"
        or manifest.get("base_rows") != EXPECTED_BASE_ROWS
        or manifest.get("new_fable_rows") != EXPECTED_NEW_FABLE_ROWS
        or manifest.get("rendered") != EXPECTED_DATASET_ROWS
        or manifest.get("training_admitted") != EXPECTED_DATASET_ROWS
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("evaluation_overlap") != 0
        or len(train_rows) != EXPECTED_DATASET_ROWS
        or manifest.get("train_jsonl_sha256")
        != _binding(train_path)["sha256"]
    ):
        raise ValueError("v2.11r2 dataset contract is incomplete")
    gate = manifest.get("standard_native_format_loss_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "passed"
        or gate.get("failure_count") != 0
        or gate.get("samples") != EXPECTED_DATASET_ROWS
    ):
        raise ValueError("v2.11r2 format-loss gate did not pass")
    gate_binding = _require_current_binding(
        gate.get("artifact"),
        base=dataset,
        label="format-loss gate",
    )
    gate_report = _read_object(Path(gate_binding["path"]))
    if (
        gate_report.get("failure_count") != 0
        or gate_report.get("samples") != EXPECTED_DATASET_ROWS
    ):
        raise ValueError("v2.11r2 format-loss report is incomplete")

    base = manifest.get("base")
    if not isinstance(base, Mapping):
        raise ValueError("v2.11r2 frozen base binding is incomplete")
    base_manifest = _require_current_binding(
        base.get("manifest"), label="frozen base manifest"
    )
    base_train = _require_current_binding(
        base.get("train"), label="frozen base train"
    )
    if Path(base_manifest["path"]).parent != Path(base_train["path"]).parent:
        raise ValueError("v2.11r2 frozen base bindings differ")

    recent = manifest.get("new_fable")
    if not isinstance(recent, Mapping):
        raise ValueError("v2.11r2 recent Fable binding is incomplete")
    source_ids = recent.get("source_instance_ids")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != EXPECTED_NEW_FABLE_ROWS
        or len(set(source_ids)) != len(source_ids)
        or any(not isinstance(value, str) or not value for value in source_ids)
    ):
        raise ValueError("v2.11r2 recent Fable IDs are incomplete")
    recent_bindings = {
        key: _require_current_binding(
            recent.get(key),
            label=f"recent Fable {key}",
        )
        for key in ("train", "manifest", "report")
    }

    evidence = manifest.get("new_fable_admission_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(source_ids):
        raise ValueError("v2.11r2 raw admission evidence is incomplete")
    validated_ids = []
    evidence_bindings = []
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("v2.11r2 raw admission evidence is invalid")
        source_id = item.get("source_instance_id")
        line_number = item.get("result_line")
        if (
            not isinstance(source_id, str)
            or type(line_number) is not int
        ):
            raise ValueError("v2.11r2 raw admission identity is invalid")
        results = _require_current_binding(
            item.get("results"), label=f"{source_id} results"
        )
        stream = _require_current_binding(
            item.get("stream"), label=f"{source_id} stream"
        )
        patch = _require_current_binding(
            item.get("patch"), label=f"{source_id} patch"
        )
        result = _read_jsonl_line(Path(results["path"]), line_number)
        result_sha = hashlib.sha256(
            canonical_json_bytes(result)
        ).hexdigest()
        if (
            result.get("instance_id") != source_id
            or not admission_is_exact(result)
            or item.get("result_sha256") != result_sha
            or item.get("task_contract_sha256")
            != result.get("task_contract_sha256")
            or item.get("admission_evidence_sha256")
            != result.get("admission_evidence_sha256")
            or stream["sha256"] != result.get("stream_sha256")
            or patch["sha256"] != result.get("patch_sha256")
        ):
            raise ValueError(
                f"v2.11r2 raw admission changed for {source_id}"
            )
        validated_ids.append(source_id)
        evidence_bindings.append({
            "source_instance_id": source_id,
            "results": results,
            "result_line": line_number,
            "result_sha256": result_sha,
            "stream": stream,
            "patch": patch,
            "task_contract_sha256": result["task_contract_sha256"],
            "admission_evidence_sha256": result[
                "admission_evidence_sha256"
            ],
        })
    if set(validated_ids) != set(source_ids):
        raise ValueError("v2.11r2 raw admissions do not match Fable rows")
    evidence_by_source = {
        str(item["source_instance_id"]): item
        for item in evidence_bindings
    }
    recent_rows = train_rows[EXPECTED_BASE_ROWS:]
    if len(recent_rows) != EXPECTED_NEW_FABLE_ROWS:
        raise ValueError("v2.11r2 late Fable training rows are incomplete")
    training_targets = [
        _validate_late_fable_training_target(
            source_id=source_id,
            row=row,
            raw_patch=evidence_by_source[source_id]["patch"],
        )
        for source_id, row in zip(source_ids, recent_rows, strict=True)
    ]
    context_path = Path(dataset_context_audit).resolve()
    context_audit = _read_object(context_path)
    context_dataset = context_audit.get("dataset")
    max_rendered_tokens = context_audit.get("max_rendered_tokens")
    if (
        context_audit.get("schema_version") != 1
        or context_audit.get("artifact_type")
        != "v2p11r2_dataset_context_audit"
        or context_audit.get("status") != "complete"
        or not isinstance(context_dataset, Mapping)
        or context_dataset.get("manifest") != _binding(manifest_path)
        or context_dataset.get("train") != _binding(train_path)
        or context_audit.get("rows") != EXPECTED_DATASET_ROWS
        or context_audit.get("max_seq") != _RUN_CONTRACT["max_seq"]
        or type(max_rendered_tokens) is not int
        or max_rendered_tokens < 1
        or max_rendered_tokens > _RUN_CONTRACT["max_seq"]
        or context_audit.get("over_limit_rows") != 0
    ):
        raise ValueError("v2.11r2 full-context audit is incomplete")
    return {
        "path": str(dataset),
        "manifest": _binding(manifest_path),
        "train": _binding(train_path),
        "format_loss_gate": gate_binding,
        "base_manifest": base_manifest,
        "base_train": base_train,
        "recent_bindings": recent_bindings,
        "source_ids": list(source_ids),
        "evidence": evidence_bindings,
        "training_targets": training_targets,
        "max_rendered_tokens": max_rendered_tokens,
        "context_audit": _binding(context_path),
    }


def _resolve_run_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("training run path is incomplete")
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _validate_training_completion(
    *,
    completion_path: Path,
    adapter: Path,
    init_adapter: Path,
) -> dict[str, Any]:
    completion_path = Path(completion_path).resolve()
    completion = _read_object(completion_path)
    adapter_binding = _require_current_binding(
        completion.get("adapter"),
        label="completed candidate adapter",
    )
    init_binding = _require_current_binding(
        completion.get("init_adapter"),
        label="completed initial adapter",
    )
    journal_binding = _require_current_binding(
        completion.get("training_journal"),
        label="training journal",
    )
    watchdog_binding = _require_current_binding(
        completion.get("watchdog_log"),
        label="training watchdog log",
    )
    train_pid = completion.get("train_pid")
    invocation_id = completion.get("train_invocation_id")
    if (
        completion.get("schema_version") != 1
        or completion.get("artifact_type")
        != "v2p11r2_training_completion"
        or completion.get("status") != "complete"
        or type(train_pid) is not int
        or train_pid <= 0
        or not isinstance(invocation_id, str)
        or len(invocation_id) != 32
        or any(character not in "0123456789abcdef" for character in invocation_id)
        or completion.get("train_unit")
        != "v2p11r2-fable51-train-gpu0.service"
        or completion.get("watchdog_unit")
        != "v2p11r2-fable51-watchdog-gpu0.service"
        or completion.get("optimizer_steps") != 79
        or completion.get("max_steps") != 79
        or type(completion.get("finite_loss_samples")) is not int
        or completion["finite_loss_samples"] < 1
        or type(completion.get("finite_grad_norm_samples")) is not int
        or completion["finite_grad_norm_samples"] < 1
        or completion.get("adapter_changed_from_init") is not True
        or type(completion.get("changed_tensor_count")) is not int
        or completion["changed_tensor_count"] < 1
        or Path(adapter_binding["path"]) != adapter / "adapter_model.safetensors"
        or Path(init_binding["path"])
        != init_adapter / "adapter_model.safetensors"
    ):
        raise ValueError("v2.11r2 training completion is incomplete")
    journal = Path(journal_binding["path"]).read_text(encoding="utf-8")
    watchdog = Path(watchdog_binding["path"]).read_text(encoding="utf-8")
    if (
        "optimizer_step=79/79" not in journal
        or "[done] adapter saved" not in journal
        or f"event=target_exited pid={train_pid}" not in watchdog
        or f"event=below_threshold pid={train_pid}" in watchdog
        or f"event=meminfo_error pid={train_pid}" in watchdog
        or f"event=health_failure pid={train_pid}" in watchdog
    ):
        raise ValueError("v2.11r2 training completion logs are incomplete")
    return {
        "artifact": _binding(completion_path),
        "train_pid": train_pid,
        "train_unit": completion["train_unit"],
        "train_invocation_id": invocation_id,
        "watchdog_unit": completion["watchdog_unit"],
        "optimizer_steps": completion["optimizer_steps"],
        "max_steps": completion["max_steps"],
        "finite_loss_samples": completion["finite_loss_samples"],
        "finite_grad_norm_samples": completion[
            "finite_grad_norm_samples"
        ],
        "adapter_changed_from_init": True,
        "changed_tensor_count": completion["changed_tensor_count"],
        "training_journal": journal_binding,
        "watchdog_log": watchdog_binding,
    }


def _validate_training(
    *,
    dataset: Path,
    adapter: Path,
    init_adapter: Path,
    training_completion: Path,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    adapter = Path(adapter).resolve()
    init_adapter = Path(init_adapter).resolve()
    run_path = adapter / "run_manifest.json"
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    init_config_path = init_adapter / "adapter_config.json"
    init_weights_path = init_adapter / "adapter_model.safetensors"
    run = _read_object(run_path)
    expected = dict(_RUN_CONTRACT)
    expected["data_len"] = EXPECTED_DATASET_ROWS
    if (
        run.get("base") != EXPECTED_BASE_MODEL
        or any(run.get(key) != value for key, value in expected.items())
        or _resolve_run_path(run.get("data")) != dataset
        or _resolve_run_path(run.get("out")) != adapter
        or _resolve_run_path(run.get("init_adapter")) != init_adapter
    ):
        raise ValueError("v2.11r2 training run contract is incomplete")
    config = _read_object(config_path)
    init_config = _read_object(init_config_path)
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 32
        or init_config.get("r") != 32
    ):
        raise ValueError("v2.11r2 adapter configuration is incomplete")
    completion = _validate_training_completion(
        completion_path=training_completion,
        adapter=adapter,
        init_adapter=init_adapter,
    )
    return {
        "run_manifest": _binding(run_path),
        "adapter": _binding(weights_path),
        "adapter_config": _binding(config_path),
        "init_adapter": _binding(init_weights_path),
        "init_adapter_config": _binding(init_config_path),
        "completion": completion,
    }


def _validate_final_model(
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
    contract = dict(gate_contract)
    contract["served_name"] = candidate_name
    if _read_object(merge_audit) != _MERGE_AUDIT:
        raise ValueError("v2.11r2 merge audit is incomplete")
    gate = _read_object(portability_gate)
    candidate = gate.get("candidate")
    if (
        gate.get("schema_version") != 1
        or gate.get("artifact_type")
        != "v2p11_controller_free_portability_gate"
        or gate.get("status") != "complete"
        or gate.get("passed") is not True
        or gate.get("control_name") != "teacher_sft_v2p10"
        or gate.get("candidate_name") != candidate_name
        or gate.get("population") != 10
        or gate.get("evaluation_overlap") != 0
        or gate.get("criteria") != PORTABILITY_CRITERIA
        or not isinstance(gate.get("control"), Mapping)
        or not isinstance(candidate, Mapping)
        or candidate.get("model_contract") != gate_contract
    ):
        raise ValueError("v2.11r2 portability gate did not pass")
    return {
        "model_contract": contract,
        "merge_audit": _binding(merge_audit),
        "portability_gate": _binding(portability_gate),
    }


def _build_report(
    *,
    candidate_name: str,
    dataset_path: Path,
    dataset_context_audit_path: Path,
    adapter_path: Path,
    init_adapter_path: Path,
    training_completion_path: Path,
    final_model_path: Path,
    merge_audit_path: Path,
    portability_gate_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
) -> dict[str, Any]:
    if not candidate_name or candidate_name == "teacher_sft_v2p10":
        raise ValueError("v2.11r2 candidate name is invalid")
    full_ids_path = Path(full_ids_path).resolve()
    v2p10_composite_path = Path(v2p10_composite_path).resolve()
    ids = _read_ids(full_ids_path)
    control = _read_object(v2p10_composite_path)
    if (
        len(ids) != EXPECTED_FULL_IDS
        or len(set(ids)) != len(ids)
        or control.get("status") != "complete"
        or control.get("name") != "teacher_sft_v2p10"
        or control.get("expected") != EXPECTED_FULL_IDS
        or control.get("resolved") != EXPECTED_V2P10_RESOLVED
    ):
        raise ValueError("canonical v2.10 full300 control is incomplete")
    dataset = _validate_dataset(dataset_path, dataset_context_audit_path)
    training = _validate_training(
        dataset=Path(dataset_path),
        adapter=Path(adapter_path),
        init_adapter=Path(init_adapter_path),
        training_completion=Path(training_completion_path),
    )
    final = _validate_final_model(
        candidate_name=candidate_name,
        final_model=Path(final_model_path),
        merge_audit=Path(merge_audit_path),
        portability_gate=Path(portability_gate_path),
    )
    return {
        "schema_version": 1,
        "artifact_type": "v2p11r2_completion_provenance",
        "status": "complete",
        "candidate_name": candidate_name,
        "full_ids": _binding(full_ids_path),
        "v2p10_full300": _binding(v2p10_composite_path),
        "lineage": {
            "kind": "direct_lora_from_v2p10",
            "recovery_sft": False,
            "kto": False,
        },
        "dataset": {
            "path": dataset["path"],
            "manifest": dataset["manifest"],
            "train": dataset["train"],
            "format_loss_gate": dataset["format_loss_gate"],
            "base_manifest": dataset["base_manifest"],
            "base_train": dataset["base_train"],
            "max_rendered_tokens": dataset["max_rendered_tokens"],
            "context_audit": dataset["context_audit"],
        },
        "training": {
            "rows": EXPECTED_DATASET_ROWS,
            "base_rows": EXPECTED_BASE_ROWS,
            "new_fable_rows": EXPECTED_NEW_FABLE_ROWS,
            "max_seq": 32768,
            "optimizer_steps": 79,
            **training,
        },
        "recent_fable": {
            "strict_rows": EXPECTED_NEW_FABLE_ROWS,
            "source_instance_ids": dataset["source_ids"],
            "inputs": dataset["recent_bindings"],
            "raw_admission_evidence": dataset["evidence"],
            "training_targets": dataset["training_targets"],
        },
        "evaluation_exclusion": {
            "full_ids": EXPECTED_FULL_IDS,
            "overlap": 0,
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
    dataset_context_audit_path: Path,
    adapter_path: Path,
    init_adapter_path: Path,
    training_completion_path: Path,
    final_model_path: Path,
    merge_audit_path: Path,
    portability_gate_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = _build_report(
        candidate_name=candidate_name,
        dataset_path=dataset_path,
        dataset_context_audit_path=dataset_context_audit_path,
        adapter_path=adapter_path,
        init_adapter_path=init_adapter_path,
        training_completion_path=training_completion_path,
        final_model_path=final_model_path,
        merge_audit_path=merge_audit_path,
        portability_gate_path=portability_gate_path,
        full_ids_path=full_ids_path,
        v2p10_composite_path=v2p10_composite_path,
    )
    _publish_json_noreplace(Path(output_path).resolve(), report)
    return report


def validate_completion_provenance(
    provenance_path: Path,
    *,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    candidate_model_contract: Mapping[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    provenance_path = Path(provenance_path).resolve()
    report = _read_object(provenance_path)
    training = report.get("training")
    dataset = report.get("dataset")
    final_model = report.get("final_model")
    merge_audit = report.get("merge_audit")
    portability = report.get("portability")
    if (
        report.get("artifact_type") != "v2p11r2_completion_provenance"
        or report.get("candidate_name") != candidate_name
        or not isinstance(training, Mapping)
        or not isinstance(dataset, Mapping)
        or not isinstance(final_model, Mapping)
        or not isinstance(merge_audit, Mapping)
        or not isinstance(portability, Mapping)
        or not isinstance(portability.get("gate"), Mapping)
    ):
        raise ValueError("v2.11r2 completion provenance is incomplete")
    expected = _build_report(
        candidate_name=candidate_name,
        dataset_path=Path(str(dataset.get("path", ""))),
        dataset_context_audit_path=_resolve_bound_path(
            dataset.get("context_audit"),
            base=None,
            label="dataset context audit",
        ),
        adapter_path=Path(
            _resolve_bound_path(
                training["adapter"],
                base=None,
                label="candidate adapter",
            )
        ).parent,
        init_adapter_path=Path(
            _resolve_bound_path(
                training["init_adapter"],
                base=None,
                label="initial adapter",
            )
        ).parent,
        training_completion_path=_resolve_bound_path(
            training["completion"]["artifact"],
            base=None,
            label="training completion",
        ),
        final_model_path=Path(str(final_model.get("model_path", ""))),
        merge_audit_path=_resolve_bound_path(
            merge_audit,
            base=None,
            label="merge audit",
        ),
        portability_gate_path=_resolve_bound_path(
            portability["gate"],
            base=None,
            label="portability gate",
        ),
        full_ids_path=Path(full_ids_path),
        v2p10_composite_path=Path(v2p10_composite_path),
    )
    if report != expected:
        raise ValueError("v2.11r2 completion provenance changed")
    if dict(candidate_model_contract) != expected["final_model"]:
        raise ValueError(
            "v2.11r2 provenance does not bind evaluated model"
        )
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-context-audit", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument(
        "--training-completion",
        type=Path,
        required=True,
    )
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_completion_provenance(
        candidate_name=args.candidate_name,
        dataset_path=args.dataset,
        dataset_context_audit_path=args.dataset_context_audit,
        adapter_path=args.adapter,
        init_adapter_path=args.init_adapter,
        training_completion_path=args.training_completion,
        final_model_path=args.final_model,
        merge_audit_path=args.merge_audit,
        portability_gate_path=args.portability_gate,
        full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10,
        output_path=args.out,
    )
    print(json.dumps({
        "status": report["status"],
        "candidate_name": report["candidate_name"],
        "rows": report["training"]["rows"],
        "new_fable_rows": report["training"]["new_fable_rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
