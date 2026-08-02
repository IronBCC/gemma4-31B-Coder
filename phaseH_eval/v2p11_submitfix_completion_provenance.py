#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

from phaseD_sft.build_mix_v2p10 import canonical_rows_sha256
from phaseD_sft.build_v2p11_submitfix_mix import SUBMIT_COMMAND
from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
)
import phaseH_eval.v2p11_clean_completion_provenance as clean


EXPECTED_ROWS = 1_262
EXPECTED_BASE_ROWS = 1_211
EXPECTED_STAGE_ROWS = 36
EXPECTED_LATE_ROWS = 15
EXPECTED_FULL_IDS = 300
EXPECTED_V2P10_RESOLVED = 157
EXPECTED_BASE_MODEL = "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
EXPECTED_TRAIN_UNIT = "v2p11-submitfix1262-train-gpu1-v2.service"
EXPECTED_GPU_IDENTITY_ARTIFACT_TYPE = (
    "v2p11_submitfix1262_gpu_training_identity"
)
EXPECTED_TRAINING_COMPLETION_ARTIFACT_TYPE = (
    "v2p11_submitfix1262_training_completion"
)
EXPECTED_MANIFEST_SHA256 = (
    "36a7f3b13ad1dbd89fa5a02af92cdfe9f8581cedf54ea984b73d5749ec52c223"
)
EXPECTED_TRAIN_SHA256 = (
    "04530c4a9237c3af88dcfe1bcce72a3de8ad3982ad2965f96ef436c0c87f254a"
)
EXPECTED_BASE_TRAIN_SHA256 = (
    "6fbd3212ddf200428587025e70f080b0b6250e04dc192149517553c5124c3398"
)
CLEAN_CONTEXT_AUDIT = Path("runs/v2p11_clean_fable51_context_audit.json")
ARTIFACT_TYPE = "v2p11_submitfix_completion_provenance"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid training rows: {path}")
    return rows


def _current_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise ValueError(f"{label} binding is incomplete")
    current = _binding(Path(value["path"]))
    if dict(value) != current:
        raise ValueError(f"{label} binding changed")
    return current


def _terminal_submit_is_exact(message: object, *, call_id: str) -> bool:
    if not isinstance(message, Mapping):
        return False
    calls = message.get("tool_calls")
    if (
        message.get("role") != "assistant"
        or message.get("content") != ""
        or message.get("loss") is not True
        or not isinstance(calls, list)
        or len(calls) != 1
    ):
        return False
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if (
        not isinstance(function, Mapping)
        or call.get("id") != call_id
        or call.get("type") != "function"
        or function.get("name") != "bash"
        or not isinstance(function.get("arguments"), str)
    ):
        return False
    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return False
    return arguments == {"command": SUBMIT_COMMAND}


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
    manifest_binding = _binding(manifest_path)
    train_binding = _binding(train_path)
    base_train_binding = _binding(base_train_path)
    if (
        manifest_binding["sha256"] != EXPECTED_MANIFEST_SHA256
        or train_binding["sha256"] != EXPECTED_TRAIN_SHA256
        or base_train_binding["sha256"] != EXPECTED_BASE_TRAIN_SHA256
    ):
        raise ValueError("submit-fix dataset immutable bindings changed")
    manifest = _read_object(manifest_path)
    rows = _read_rows(train_path)
    base_rows = _read_rows(base_train_path)
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("submit-fix source binding is incomplete")
    source_manifest_binding = _current_binding(
        source.get("manifest"), label="submit-fix source manifest"
    )
    source_train_binding = _current_binding(
        source.get("train"), label="submit-fix source train"
    )
    source_manifest_path = Path(source_manifest_binding["path"])
    source_train_path = Path(source_train_binding["path"])
    if source_manifest_path.parent != source_train_path.parent:
        raise ValueError("submit-fix source files are from different datasets")
    source_rows = _read_rows(source_train_path)
    stage_end = EXPECTED_BASE_ROWS + EXPECTED_STAGE_ROWS
    if (
        len(rows) != EXPECTED_ROWS
        or len(source_rows) != EXPECTED_ROWS
        or len(base_rows) != EXPECTED_BASE_ROWS
        or stage_end + EXPECTED_LATE_ROWS != EXPECTED_ROWS
        or rows[:EXPECTED_BASE_ROWS] != base_rows
        or rows[:EXPECTED_BASE_ROWS] != source_rows[:EXPECTED_BASE_ROWS]
        or rows[stage_end:] != source_rows[stage_end:]
    ):
        raise ValueError("submit-fix dataset partitions changed")
    ids = [row.get("instance_id") for row in rows]
    if (
        any(not isinstance(instance_id, str) or not instance_id for instance_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("submit-fix instance identities are invalid")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, Mapping)
                or type(message.get("loss")) is not bool
                for message in messages
            )
        ):
            raise ValueError("submit-fix row has invalid boolean loss contract")
        if EXPECTED_BASE_ROWS <= index < stage_end:
            source_row = source_rows[index]
            row_without_messages = {key: value for key, value in row.items() if key != "messages"}
            source_without_messages = {
                key: value for key, value in source_row.items() if key != "messages"
            }
            if (
                row_without_messages != source_without_messages
                or messages[:-1] != source_row["messages"][:-1]
                or not _terminal_submit_is_exact(
                    messages[-1],
                    call_id=f"v2p11-submit-{index - EXPECTED_BASE_ROWS + 1}",
                )
            ):
                raise ValueError("submit-fix Stage-A repair is not terminal-only")
    format_gate = manifest.get("standard_native_format_loss_gate")
    report_binding = (
        format_gate.get("report") if isinstance(format_gate, Mapping) else None
    )
    current_report = _current_binding(
        report_binding, label="submit-fix format report"
    )
    source_counts = dict(
        sorted(Counter(row.get("source") for row in rows[EXPECTED_BASE_ROWS:]).items())
    )
    expected_source_counts = {
        "fable5_verified_finalpatch": EXPECTED_LATE_ROWS,
        "teacher:claude:claude-fable-5": EXPECTED_STAGE_ROWS,
    }
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("dataset_variant")
        != "teacher_train_mix_v2p11_submitfix_v1"
        or manifest.get("selection_mode") != "submit_fix"
        or manifest.get("rendered") != EXPECTED_ROWS
        or manifest.get("base_rows") != EXPECTED_BASE_ROWS
        or manifest.get("terminal_repairs") != EXPECTED_STAGE_ROWS
        or manifest.get("unchanged_late_rows") != EXPECTED_LATE_ROWS
        or manifest.get("training_admitted") != EXPECTED_ROWS
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("evaluation_overlap") != 0
        or manifest.get("mutation_sequence_unchanged") is not True
        or manifest.get("train_jsonl_sha256") != train_binding["sha256"]
        or manifest.get("canonical_rows_sha256")
        not in (None, canonical_rows_sha256(rows))
        or manifest.get("base_prefix_canonical_sha256")
        not in (None, canonical_rows_sha256(rows[:EXPECTED_BASE_ROWS]))
        or source_counts != expected_source_counts
        or not isinstance(format_gate, Mapping)
        or format_gate.get("status") != "passed"
        or format_gate.get("failure_count") != 0
        or format_gate.get("samples") != EXPECTED_ROWS
    ):
        raise ValueError("submit-fix dataset manifest contract is incomplete")
    report = _read_object(Path(current_report["path"]))
    counts = report.get("counts")
    if (
        report.get("samples") != EXPECTED_ROWS
        or report.get("failure_count") != 0
        or (
            isinstance(counts, Mapping)
            and (
                counts.get("fallback_spans") != 0
                or counts.get("supervised_fallback_spans") != 0
            )
        )
    ):
        raise ValueError("submit-fix format report is incomplete")
    clean_context = Path(CLEAN_CONTEXT_AUDIT).resolve()
    clean_source = clean._validate_dataset(
        source_manifest_path.parent,
        base_data,
        clean_context,
    )
    context_path = Path(context_audit_path).resolve()
    context = _read_object(context_path)
    max_rendered = context.get("max_rendered_tokens")
    if (
        context.get("schema_version") != 1
        or context.get("artifact_type")
        != "v2p11_submitfix1262_context_audit"
        or context.get("status") != "complete"
        or context.get("dataset_manifest_sha256") != manifest_binding["sha256"]
        or context.get("train_jsonl_sha256") != train_binding["sha256"]
        or context.get("source_manifest_sha256")
        != source_manifest_binding["sha256"]
        or context.get("source_train_jsonl_sha256")
        != source_train_binding["sha256"]
        or context.get("base_train_jsonl_sha256") != base_train_binding["sha256"]
        or context.get("rows") != EXPECTED_ROWS
        or context.get("base_rows") != EXPECTED_BASE_ROWS
        or context.get("terminal_repairs") != EXPECTED_STAGE_ROWS
        or context.get("unchanged_late_rows") != EXPECTED_LATE_ROWS
        or context.get("source_counts") != expected_source_counts
        or context.get("max_seq") != 32768
        or type(max_rendered) is not int
        or not 0 < max_rendered <= 32768
        or context.get("over_limit_rows") != 0
        or context.get("optimizer_steps") != 79
        or context.get("init_adapter") is not None
    ):
        raise ValueError("submit-fix full-context audit is incomplete")
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
        "terminal_repairs": EXPECTED_STAGE_ROWS,
        "unchanged_late_rows": EXPECTED_LATE_ROWS,
        "selection_mode": "submit_fix",
        "source_counts": source_counts,
        "init_adapter": None,
        "max_rendered_tokens": max_rendered,
        "context_audit": _binding(context_path),
        "format_report": current_report,
        "source_clean": clean_source,
    }


@contextmanager
def _training_profile() -> Iterator[None]:
    fields = {
        "EXPECTED_TRAIN_UNIT": EXPECTED_TRAIN_UNIT,
        "EXPECTED_GPU_IDENTITY_ARTIFACT_TYPE": (
            EXPECTED_GPU_IDENTITY_ARTIFACT_TYPE
        ),
        "EXPECTED_TRAINING_COMPLETION_ARTIFACT_TYPE": (
            EXPECTED_TRAINING_COMPLETION_ARTIFACT_TYPE
        ),
    }
    previous = {name: getattr(clean, name) for name in fields}
    try:
        for name, value in fields.items():
            setattr(clean, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(clean, name, value)


def _validate_training(
    *,
    dataset: Path,
    adapter: Path,
    completion_path: Path,
) -> dict[str, Any]:
    with _training_profile():
        return clean._validate_training(
            dataset=dataset,
            adapter=adapter,
            completion_path=completion_path,
        )


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
        raise ValueError("submit-fix candidate name is invalid")
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
    lineage = clean._validate_v2p10_lineage(
        lineage_path=v2p10_lineage_path,
        v2p10_composite_path=v2p10_composite_path,
        base_data_path=base_data_path,
    )
    final = clean._validate_final(
        candidate_name=candidate_name,
        final_model=final_model_path,
        merge_audit=merge_audit_path,
        portability_gate=portability_gate_path,
    )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete",
        "candidate_name": candidate_name,
        "lineage": {
            "kind": "fresh_raw_base_lora",
            "base_model": EXPECTED_BASE_MODEL,
            "init_adapter": None,
            "recovery_sft": False,
            "kto": False,
            "interpolation": False,
            "successor_mode": "submit_fix",
        },
        "full_ids": _binding(full_ids_path),
        "v2p10_full300": _binding(v2p10_composite_path),
        "v2p10_training_lineage": lineage,
        "dataset": dataset,
        "training": {
            "rows": EXPECTED_ROWS,
            "base_rows": EXPECTED_BASE_ROWS,
            "new_fable_rows": EXPECTED_STAGE_ROWS + EXPECTED_LATE_ROWS,
            "terminal_repairs": EXPECTED_STAGE_ROWS,
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
            "criteria": dict(clean.PORTABILITY_CRITERIA),
        },
    }


def publish_completion_provenance(
    *,
    output_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = _build_report(**kwargs)
    _publish_json_noreplace(Path(output_path).resolve(), report)
    return report


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
        report.get("artifact_type") != ARTIFACT_TYPE
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
        raise ValueError("submit-fix completion provenance is incomplete")
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
        merge_audit_path=Path(
            str(report.get("merge_audit", {}).get("path", ""))
        ),
        portability_gate_path=Path(str(portability["gate"].get("path", ""))),
        full_ids_path=Path(full_ids_path),
        v2p10_composite_path=Path(v2p10_composite_path),
        v2p10_lineage_path=Path(v2p10_lineage_path).resolve(),
    )
    if report != expected:
        raise ValueError("submit-fix completion provenance changed")
    if not clean._model_contracts_match(
        candidate_model_contract, expected["final_model"]
    ):
        raise ValueError("submit-fix provenance does not bind evaluated model")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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
    print(json.dumps({"status": report["status"], "candidate_name": report["candidate_name"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
