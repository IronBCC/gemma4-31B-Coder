#!/usr/bin/env python3
"""Publish and revalidate the corrected direct-LoRA v2.11r3 lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract
from phaseH_eval.v2p11r2_completion_provenance import (
    PORTABILITY_CRITERIA,
    _MERGE_AUDIT,
    _validate_dataset as _validate_fable_source,
)
from phaseH_eval.v2p11_stage_data_contract import (
    validate_stage_data as _validate_stage_data,
)


EXPECTED_ROWS = 1_262
EXPECTED_FULL_IDS = 300
EXPECTED_V2P10_RESOLVED = 157
EXPECTED_BASE = "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
SOURCE_CONTEXT_AUDIT = Path(
    "runs/v2p11r2_v2p10init_fable51_dataset_context_audit.json"
)
STAGE_DATA = Path("data/teacher_train_mix_v2p11_frozen1247")
STAGE_TRAINING_CONTRACT = Path("runs/v2p11_completion_training_contract.json")
STAGE_EXCLUSIONS = Path("data/swe_all_eval_exclusions_v2.json")
STAGE_CAMPAIGN = Path("runs/teacher_fable5_v2p11_frozen47_merged")
_RUN_CONTRACT = {
    "data_len": EXPECTED_ROWS,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-6,
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
_STRUCTURAL_GATES = {
    "all_fable_supervised_tool_calls_are_canonical_bash": True,
    "all_messages_have_boolean_loss": True,
    "evaluation_overlap": 0,
    "reasoned_tool_turns": 468,
}


def _require_current_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is incomplete")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} binding path is incomplete")
    path = Path(raw_path).resolve()
    current = _binding(path)
    if value.get("sha256") != current["sha256"] or value.get("bytes") != current["bytes"]:
        raise ValueError(f"{label} binding changed")
    return current


def _resolve_run_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("training run path is incomplete")
    path = Path(value)
    return (path if path.is_absolute() else Path.cwd() / path).resolve()


def _validate_dataset(dataset: Path, context_audit_path: Path) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    manifest_path = dataset / "manifest.json"
    train_path = dataset / "train.jsonl"
    manifest = _read_object(manifest_path)
    try:
        rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
    except json.JSONDecodeError as error:
        raise ValueError("v2.11r3 training JSONL is invalid") from error
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("dataset_variant") != "teacher_train_mix_v2p11_fable_reasoned_v1"
        or manifest.get("source_variant") != "teacher_train_mix_v2p11_fable_extension"
        or manifest.get("rendered") != EXPECTED_ROWS
        or manifest.get("training_admitted") != EXPECTED_ROWS
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("evaluation_overlap") != 0
        or manifest.get("fable_rows") != 92
        or manifest.get("canonical_bash_tool_turns") != 517
        or manifest.get("reasoned_tool_turns") != 468
        or manifest.get("structural_gates") != _STRUCTURAL_GATES
        or len(rows) != EXPECTED_ROWS
        or manifest.get("train_jsonl_sha256") != _binding(train_path)["sha256"]
    ):
        raise ValueError("v2.11r3 reasoning-contract dataset is incomplete")
    gate = manifest.get("format_loss_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "passed_full_dataset_verification"
        or gate.get("failure_count") != 0
        or gate.get("samples") != EXPECTED_ROWS
    ):
        raise ValueError("v2.11r3 format-loss gate did not pass")
    format_gate = _require_current_binding(gate.get("report"), label="reasoned format-loss gate")
    report = _read_object(Path(format_gate["path"]))
    if report.get("failure_count") != 0 or report.get("samples") != EXPECTED_ROWS:
        raise ValueError("v2.11r3 format-loss report is incomplete")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("v2.11r3 Fable source binding is incomplete")
    source_manifest = _require_current_binding(source.get("manifest"), label="Fable source manifest")
    source_train = _require_current_binding(source.get("train"), label="Fable source train")
    source_path = Path(source_manifest["path"]).parent
    if source_path != Path(source_train["path"]).parent:
        raise ValueError("v2.11r3 Fable source bindings disagree")
    source_context = Path(SOURCE_CONTEXT_AUDIT).resolve()
    if not source_context.is_file():
        raise ValueError("v2.11r3 Fable source context audit is missing")
    source_fable = _validate_fable_source(source_path, source_context)
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
        key: stage.get(key)
        for key in ("rows", "base_rows", "new_fable_rows")
    } != {"rows": 1247, "base_rows": 1211, "new_fable_rows": 36}:
        raise ValueError("v2.11r3 Stage-A Fable evidence is incomplete")

    context_path = Path(context_audit_path).resolve()
    context = _read_object(context_path)
    context_data = context.get("dataset")
    if (
        context.get("schema_version") != 1
        or context.get("artifact_type") != "v2p11r3_dataset_context_audit"
        or context.get("status") != "complete"
        or not isinstance(context_data, Mapping)
        or context_data.get("manifest_sha256") != _binding(manifest_path)["sha256"]
        or context_data.get("train_sha256") != _binding(train_path)["sha256"]
        or context.get("rows") != EXPECTED_ROWS
        or context.get("max_seq") != 32768
        or type(context.get("max_rendered_tokens")) is not int
        or not 0 < context["max_rendered_tokens"] <= 32768
        or context.get("over_limit_rows") != 0
    ):
        raise ValueError("v2.11r3 full-context audit is incomplete")
    return {
        "path": str(dataset),
        "manifest": _binding(manifest_path),
        "train": _binding(train_path),
        "format_loss_gate": format_gate,
        "context_audit": _binding(context_path),
        "max_rendered_tokens": context["max_rendered_tokens"],
        "fable_rows": manifest["fable_rows"],
        "canonical_bash_tool_turns": manifest["canonical_bash_tool_turns"],
        "reasoned_tool_turns": manifest["reasoned_tool_turns"],
        "source_manifest": source_manifest,
        "source_train": source_train,
        "source_context_audit": _binding(source_context),
        "source_fable": source_fable,
        "stage_a": {
            "summary": {
                key: stage[key]
                for key in ("rows", "base_rows", "new_fable_rows")
            },
            "training_contract": _binding(stage_contract),
            "exclusions": _binding(stage_exclusions),
            "campaign_root_manifest": _binding(stage_campaign / "manifest.json"),
        },
    }


def _validate_training(
    *, dataset: Path, adapter: Path, init_adapter: Path, completion_path: Path
) -> dict[str, Any]:
    dataset, adapter, init_adapter = map(lambda item: Path(item).resolve(), (dataset, adapter, init_adapter))
    run_path = adapter / "run_manifest.json"
    run = _read_object(run_path)
    if (
        run.get("base") != EXPECTED_BASE
        or any(run.get(key) != value for key, value in _RUN_CONTRACT.items())
        or _resolve_run_path(run.get("data")) != dataset
        or _resolve_run_path(run.get("out")) != adapter
        or _resolve_run_path(run.get("init_adapter")) != init_adapter
    ):
        raise ValueError("v2.11r3 training run contract is incomplete")
    weights = adapter / "adapter_model.safetensors"
    initial = init_adapter / "adapter_model.safetensors"
    config = _read_object(adapter / "adapter_config.json")
    if config.get("r") != 32 or config.get("lora_alpha") != 32:
        raise ValueError("v2.11r3 adapter configuration is incomplete")
    completion_path = Path(completion_path).resolve()
    completion = _read_object(completion_path)
    candidate = _require_current_binding(completion.get("adapter"), label="candidate adapter")
    baseline = _require_current_binding(completion.get("init_adapter"), label="initial adapter")
    journal = _require_current_binding(completion.get("training_journal"), label="training journal")
    watchdog = _require_current_binding(completion.get("watchdog_log"), label="training watchdog")
    if (
        completion.get("schema_version") != 1
        or completion.get("artifact_type") != "v2p11r3_training_completion"
        or completion.get("status") != "complete"
        or completion.get("optimizer_steps") != 79
        or completion.get("max_steps") != 79
        or completion.get("adapter_changed_from_init") is not True
        or type(completion.get("changed_tensor_count")) is not int
        or completion["changed_tensor_count"] < 1
        or Path(candidate["path"]) != weights
        or Path(baseline["path"]) != initial
    ):
        raise ValueError("v2.11r3 training completion is incomplete")
    journal_text = Path(journal["path"]).read_text(encoding="utf-8")
    watchdog_text = Path(watchdog["path"]).read_text(encoding="utf-8")
    if (
        "optimizer_step=79/79" not in journal_text
        or "[done] adapter saved" not in journal_text
        or "event=target_exited pid=" not in watchdog_text
        or "event=below_threshold" in watchdog_text
        or "event=meminfo_error" in watchdog_text
        or "event=health_failure" in watchdog_text
    ):
        raise ValueError("v2.11r3 training evidence is incomplete")
    return {
        "run_manifest": _binding(run_path),
        "adapter": candidate,
        "adapter_config": _binding(adapter / "adapter_config.json"),
        "init_adapter": baseline,
        "init_adapter_config": _binding(init_adapter / "adapter_config.json"),
        "completion": _binding(completion_path),
        "training_journal": journal,
        "watchdog_log": watchdog,
        "changed_tensor_count": completion["changed_tensor_count"],
    }


def _validate_final(
    *, candidate_name: str, final_model: Path, merge_audit: Path, portability_gate: Path
) -> dict[str, Any]:
    final_model = Path(final_model).resolve()
    audit = _read_object(Path(merge_audit).resolve())
    model_contract = _model_contract(final_model)
    model_contract["served_name"] = candidate_name
    gate_path = Path(portability_gate).resolve()
    gate = _read_object(gate_path)
    candidate = gate.get("candidate")
    if (
        audit != _MERGE_AUDIT
        or gate.get("schema_version") != 1
        or gate.get("artifact_type") != "v2p11_controller_free_portability_gate"
        or gate.get("status") != "complete"
        or gate.get("passed") is not True
        or gate.get("control_name") != "teacher_sft_v2p10"
        or gate.get("candidate_name") != candidate_name
        or gate.get("population") != 10
        or gate.get("evaluation_overlap") != 0
        or gate.get("criteria") != PORTABILITY_CRITERIA
        or not isinstance(candidate, Mapping)
        or candidate.get("model_contract") != {key: value for key, value in model_contract.items() if key != "served_name"}
    ):
        raise ValueError("v2.11r3 final model or portability gate is incomplete")
    return {
        "model_contract": model_contract,
        "merge_audit": _binding(Path(merge_audit).resolve()),
        "portability_gate": _binding(gate_path),
    }


def _build_report(
    *, candidate_name: str, dataset_path: Path, dataset_context_audit_path: Path,
    adapter_path: Path, init_adapter_path: Path, training_completion_path: Path,
    final_model_path: Path, merge_audit_path: Path, portability_gate_path: Path,
    full_ids_path: Path, v2p10_composite_path: Path,
) -> dict[str, Any]:
    if not candidate_name or candidate_name == "teacher_sft_v2p10":
        raise ValueError("v2.11r3 candidate name is invalid")
    ids = _read_ids(Path(full_ids_path))
    control = _read_object(Path(v2p10_composite_path))
    if (
        len(ids) != EXPECTED_FULL_IDS
        or len(set(ids)) != EXPECTED_FULL_IDS
        or control.get("status") != "complete"
        or control.get("name") != "teacher_sft_v2p10"
        or control.get("expected") != EXPECTED_FULL_IDS
        or control.get("resolved") != EXPECTED_V2P10_RESOLVED
    ):
        raise ValueError("canonical v2.10 full300 control is incomplete")
    dataset = _validate_dataset(dataset_path, dataset_context_audit_path)
    training = _validate_training(
        dataset=dataset_path, adapter=adapter_path, init_adapter=init_adapter_path,
        completion_path=training_completion_path,
    )
    final = _validate_final(
        candidate_name=candidate_name, final_model=final_model_path,
        merge_audit=merge_audit_path, portability_gate=portability_gate_path,
    )
    return {
        "schema_version": 1,
        "artifact_type": "v2p11r3_completion_provenance",
        "status": "complete",
        "candidate_name": candidate_name,
        "lineage": {"kind": "direct_lora_from_v2p10", "reasoned_fable_contract": True},
        "full_ids": _binding(Path(full_ids_path).resolve()),
        "v2p10_full300": _binding(Path(v2p10_composite_path).resolve()),
        "dataset": dataset,
        "training": {"rows": EXPECTED_ROWS, "max_seq": 32768, "optimizer_steps": 79, **training},
        "final_model": final["model_contract"],
        "merge_audit": final["merge_audit"],
        "portability": {
            "gate": final["portability_gate"], "passed": True,
            "population": 10, "criteria": dict(PORTABILITY_CRITERIA),
        },
    }


def publish_completion_provenance(
    *, candidate_name: str, dataset_path: Path, dataset_context_audit_path: Path,
    adapter_path: Path, init_adapter_path: Path, training_completion_path: Path,
    final_model_path: Path, merge_audit_path: Path, portability_gate_path: Path,
    full_ids_path: Path, v2p10_composite_path: Path, output_path: Path,
) -> dict[str, Any]:
    report = _build_report(
        candidate_name=candidate_name, dataset_path=dataset_path,
        dataset_context_audit_path=dataset_context_audit_path, adapter_path=adapter_path,
        init_adapter_path=init_adapter_path, training_completion_path=training_completion_path,
        final_model_path=final_model_path, merge_audit_path=merge_audit_path,
        portability_gate_path=portability_gate_path, full_ids_path=full_ids_path,
        v2p10_composite_path=v2p10_composite_path,
    )
    _publish_json_noreplace(Path(output_path).resolve(), report)
    return report


def validate_completion_provenance(
    provenance_path: Path, *, full_ids_path: Path, v2p10_composite_path: Path,
    candidate_model_contract: Mapping[str, Any], candidate_name: str,
) -> dict[str, Any]:
    report = _read_object(Path(provenance_path).resolve())
    dataset = report.get("dataset")
    training = report.get("training")
    final_model = report.get("final_model")
    portability = report.get("portability")
    if (
        report.get("artifact_type") != "v2p11r3_completion_provenance"
        or report.get("candidate_name") != candidate_name
        or not isinstance(dataset, Mapping) or not isinstance(training, Mapping)
        or not isinstance(final_model, Mapping) or not isinstance(portability, Mapping)
        or not isinstance(portability.get("gate"), Mapping)
    ):
        raise ValueError("v2.11r3 completion provenance is incomplete")
    expected = _build_report(
        candidate_name=candidate_name,
        dataset_path=Path(str(dataset.get("path", ""))),
        dataset_context_audit_path=Path(str(dataset.get("context_audit", {}).get("path", ""))),
        adapter_path=Path(str(training.get("adapter", {}).get("path", ""))).parent,
        init_adapter_path=Path(str(training.get("init_adapter", {}).get("path", ""))).parent,
        training_completion_path=Path(str(training.get("completion", {}).get("path", ""))),
        final_model_path=Path(str(final_model.get("model_path", ""))),
        merge_audit_path=Path(str(report.get("merge_audit", {}).get("path", ""))),
        portability_gate_path=Path(str(portability["gate"].get("path", ""))),
        full_ids_path=Path(full_ids_path), v2p10_composite_path=Path(v2p10_composite_path),
    )
    if report != expected:
        raise ValueError("v2.11r3 completion provenance changed")
    if dict(candidate_model_contract) != expected["final_model"]:
        raise ValueError("v2.11r3 provenance does not bind evaluated model")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-context-audit", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_completion_provenance(
        candidate_name=args.candidate_name, dataset_path=args.dataset,
        dataset_context_audit_path=args.dataset_context_audit, adapter_path=args.adapter,
        init_adapter_path=args.init_adapter, training_completion_path=args.training_completion,
        final_model_path=args.final_model, merge_audit_path=args.merge_audit,
        portability_gate_path=args.portability_gate, full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10, output_path=args.out,
    )
    print(json.dumps({"status": report["status"], "candidate_name": report["candidate_name"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
