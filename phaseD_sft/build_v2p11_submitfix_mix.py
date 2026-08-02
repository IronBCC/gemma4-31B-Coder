#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from phaseD_sft.build_mix_v2p10 import (
    _canonical_json,
    _publish_directory_atomic,
    _sha256_path,
    canonical_rows_sha256,
)
from phaseH_eval.empty_retry_composite import _binding


PRODUCTION_ROWS = 1_262
PRODUCTION_BASE_ROWS = 1_211
PRODUCTION_STAGE_ROWS = 36
PRODUCTION_LATE_ROWS = 15
PRODUCTION_SOURCE_MANIFEST_SHA256 = (
    "40531f44c8d5ab1d47a179418aa1adaf1ca31d0f0265f262b15c5fd424b4758a"
)
PRODUCTION_SOURCE_TRAIN_SHA256 = (
    "3d131c531ca060ad2472304b95b423f292cefde94cf4538788ef52ce36b919c1"
)
SUBMIT_COMMAND = (
    "git diff -- . > patch.txt && test -s patch.txt && echo "
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
)
SOURCE_VARIANT = "teacher_train_mix_v2p11_fable_extension"
OUTPUT_VARIANT = "teacher_train_mix_v2p11_submitfix_v1"
STAGE_SOURCE = "teacher:claude:claude-fable-5"
LATE_SOURCE = "fable5_verified_finalpatch"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"missing JSONL: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object row at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json_atomic(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_json_noreplace(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_boolean_loss(rows: list[dict[str, Any]]) -> None:
    for row_number, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {row_number} has invalid messages")
        for message_number, message in enumerate(messages):
            if (
                not isinstance(message, Mapping)
                or type(message.get("loss")) is not bool
            ):
                raise ValueError(
                    f"row {row_number} message {message_number} must have boolean loss"
                )


def _is_supervised_tool_call(message: Mapping[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    return (
        message.get("role") == "assistant"
        and message.get("loss") is True
        and isinstance(tool_calls, list)
        and bool(tool_calls)
    )


def replace_terminal_with_submit(
    row: dict[str, Any],
    *,
    call_id: str,
) -> dict[str, Any]:
    messages = row.get("messages")
    terminal = messages[-1] if isinstance(messages, list) and messages else None
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("role") != "assistant"
        or terminal.get("loss") is not True
        or not isinstance(terminal.get("content"), str)
        or not terminal["content"].strip()
        or terminal.get("tool_calls") not in (None, [])
    ):
        raise ValueError("terminal target must be supervised prose")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("call_id must be non-empty")
    repaired = deepcopy(row)
    repaired["messages"][-1] = {
        "role": "assistant",
        "content": "",
        "loss": True,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": SUBMIT_COMMAND}),
                },
            }
        ],
    }
    return repaired


def _validate_source(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = source / "manifest.json"
    train_path = source / "train.jsonl"
    if _sha256_path(manifest_path) != PRODUCTION_SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest hash mismatch")
    if _sha256_path(train_path) != PRODUCTION_SOURCE_TRAIN_SHA256:
        raise ValueError("source train hash mismatch")
    manifest = _read_object(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("dataset_variant") != SOURCE_VARIANT
        or manifest.get("rendered") != PRODUCTION_ROWS
        or manifest.get("training_admitted") != PRODUCTION_ROWS
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("evaluation_overlap") != 0
        or manifest.get("train_jsonl_sha256") != PRODUCTION_SOURCE_TRAIN_SHA256
    ):
        raise ValueError("source manifest contract mismatch")
    rows = _read_rows(train_path)
    if len(rows) != PRODUCTION_ROWS:
        raise ValueError("source row count mismatch")
    ids = [row.get("instance_id") for row in rows]
    if (
        any(not isinstance(instance_id, str) or not instance_id for instance_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("source instance IDs must be unique strings")
    _validate_boolean_loss(rows)
    stage_end = PRODUCTION_BASE_ROWS + PRODUCTION_STAGE_ROWS
    if stage_end + PRODUCTION_LATE_ROWS != PRODUCTION_ROWS:
        raise ValueError("production partition counts do not sum to total")
    stage_rows = rows[PRODUCTION_BASE_ROWS:stage_end]
    if any(row.get("source") != STAGE_SOURCE for row in stage_rows):
        raise ValueError("Stage-A source contract mismatch")
    for row in stage_rows:
        terminal = row["messages"][-1]
        if (
            terminal.get("role") != "assistant"
            or terminal.get("loss") is not True
            or not isinstance(terminal.get("content"), str)
            or not terminal["content"].strip()
            or terminal.get("tool_calls") not in (None, [])
        ):
            raise ValueError("Stage-A row lacks supervised prose terminal")
    late_rows = rows[stage_end:]
    if any(row.get("source") != LATE_SOURCE for row in late_rows):
        raise ValueError("late Fable source contract mismatch")
    if any(not _is_supervised_tool_call(row["messages"][-1]) for row in late_rows):
        raise ValueError("late Fable row lacks supervised terminal tool call")
    return rows, manifest


def build_submitfix_mix(*, source: Path, out: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    out = Path(out).resolve()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    source_rows, source_manifest = _validate_source(source)
    stage_end = PRODUCTION_BASE_ROWS + PRODUCTION_STAGE_ROWS
    repaired_rows = [
        *source_rows[:PRODUCTION_BASE_ROWS],
        *[
            replace_terminal_with_submit(
                row,
                call_id=f"v2p11-submit-{index + 1}",
            )
            for index, row in enumerate(
                source_rows[PRODUCTION_BASE_ROWS:stage_end]
            )
        ],
        *source_rows[stage_end:],
    ]
    manifest_holder: dict[str, Any] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(repaired_rows).save_to_disk(str(stage))
        train_path = stage / "train.jsonl"
        with train_path.open("wb") as handle:
            for row in repaired_rows:
                handle.write(_canonical_json(row) + b"\n")
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": OUTPUT_VARIANT,
            "source_variant": source_manifest["dataset_variant"],
            "selection_mode": "submit_fix",
            "rendered": len(repaired_rows),
            "base_rows": PRODUCTION_BASE_ROWS,
            "terminal_repairs": PRODUCTION_STAGE_ROWS,
            "unchanged_late_rows": PRODUCTION_LATE_ROWS,
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "canonical_rows_sha256": canonical_rows_sha256(repaired_rows),
            "train_jsonl_sha256": _sha256_path(train_path),
            "base_prefix_canonical_sha256": canonical_rows_sha256(
                repaired_rows[:PRODUCTION_BASE_ROWS]
            ),
            "evaluation_overlap": 0,
            "mutation_sequence_unchanged": True,
            "source": {
                "manifest": _binding(source / "manifest.json"),
                "train": _binding(source / "train.jsonl"),
            },
            "structural_gates": {
                "all_messages_have_boolean_loss": True,
                "base_prefix_unchanged": True,
                "only_stage_a_terminal_targets_repaired": True,
                "late_rows_unchanged": True,
                "evaluation_overlap": 0,
            },
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
                "samples": None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(out, publish)
    return manifest_holder


def seal_submitfix_mix(*, out: Path, format_report: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    format_report = Path(format_report).resolve()
    manifest_path = out / "manifest.json"
    train_path = out / "train.jsonl"
    manifest = _read_object(manifest_path)
    report = _read_object(format_report)
    if (
        manifest.get("dataset_variant") != OUTPUT_VARIANT
        or manifest.get("training_admitted") != 0
        or manifest.get("all_training_gates_complete") is not False
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_path)
    ):
        raise ValueError("submit-fix dataset is not sealable")
    rows = _read_rows(train_path)
    if (
        len(rows) != PRODUCTION_ROWS
        or manifest.get("rendered") != len(rows)
        or manifest.get("canonical_rows_sha256") != canonical_rows_sha256(rows)
    ):
        raise ValueError("submit-fix dataset changed before sealing")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("submit-fix source binding is invalid")
    for field in ("manifest", "train"):
        binding = source.get(field)
        if not isinstance(binding, Mapping):
            raise ValueError("submit-fix source binding is invalid")
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or binding.get("sha256") != _sha256_path(path):
            raise ValueError("submit-fix source changed before sealing")
    report_data = report.get("data")
    if (
        not isinstance(report_data, str)
        or Path(report_data).resolve() != out
        or report.get("samples") != len(rows)
        or report.get("failure_count") != 0
        or report.get("fallback_spans", 0) != 0
    ):
        raise ValueError("format-loss report does not prove the full submit-fix dataset")
    report_name = f"format-gate.{_sha256_path(format_report)}.json"
    report_copy = out / report_name
    _publish_json_noreplace(report_copy, report)
    updated = dict(manifest)
    updated["training_admitted"] = len(rows)
    updated["all_training_gates_complete"] = True
    updated["standard_native_format_loss_gate"] = {
        "status": "passed",
        "failure_count": 0,
        "samples": len(rows),
        "report": _binding(report_copy),
    }
    _write_json_atomic(manifest_path, updated)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--out", type=Path, required=True)
    seal.add_argument("--format-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_submitfix_mix(source=args.source, out=args.out)
    else:
        manifest = seal_submitfix_mix(
            out=args.out,
            format_report=args.format_report,
        )
    print(
        json.dumps(
            {
                "dataset_variant": manifest["dataset_variant"],
                "rendered": manifest["rendered"],
                "terminal_repairs": manifest["terminal_repairs"],
                "all_training_gates_complete": manifest[
                    "all_training_gates_complete"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
