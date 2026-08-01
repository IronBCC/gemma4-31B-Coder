#!/usr/bin/env python3
"""Render admitted Fable tool turns into the production reasoning contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseD_sft.build_mix_v2p10 import (
    _canonical_json,
    _publish_directory_atomic,
    _sha256_path,
    canonical_rows_sha256,
)


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
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSONL row {path}:{line_number}")
        rows.append(value)
    return rows


def _binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(dict(value)))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _is_fable(row: Mapping[str, Any]) -> bool:
    source = row.get("source")
    return isinstance(source, str) and "fable" in source.casefold()


def _canonical_bash_command(call: object, *, instance_id: str) -> str:
    if not isinstance(call, Mapping):
        raise ValueError(f"Fable tool call is invalid for {instance_id}")
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "bash":
        raise ValueError(f"Fable tool call is not canonical bash for {instance_id}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError(f"Fable bash arguments are invalid for {instance_id}")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fable bash arguments are invalid for {instance_id}") from exc
    command = parsed.get("command") if isinstance(parsed, Mapping) else None
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"Fable bash command is invalid for {instance_id}")
    return command


def _rationale_for(command: str) -> str:
    lowered = command.lstrip().casefold()
    if any(marker in lowered for marker in ("pytest", "cargo test", "npm test", "go test")):
        return "I will run the focused test to validate the current approach."
    if any(
        lowered.startswith(marker)
        for marker in ("git apply", "apply_patch", "sed -i", "perl -pi")
    ):
        return "I will make the targeted source edit required by the task."
    if any(
        lowered.startswith(marker)
        for marker in ("git diff", "git status", "rg ", "grep ", "find ", "ls", "sed ", "cat ")
    ):
        return "I will inspect the relevant repository state before making a targeted change."
    return "I will run the next focused shell command and use its output to guide the fix."


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("training row is not JSON serializable") from exc
    if not isinstance(copied, dict):
        raise ValueError("training row is invalid")
    return copied


def _validate_messages(row: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    instance_id = row.get("instance_id")
    messages = row.get("messages")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("training row has no instance ID")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"training row has no messages: {instance_id}")
    if any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        or not isinstance(message.get("loss"), bool)
        for message in messages
    ):
        raise ValueError(f"training row has invalid message contract: {instance_id}")
    return instance_id, messages


def render_fable_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    rendered = _copy_row(row)
    instance_id, messages = _validate_messages(rendered)
    stats = {"canonical_bash_tool_turns": 0, "reasoned_tool_turns": 0}
    if not _is_fable(rendered):
        return rendered, stats
    for message in messages:
        if message["role"] != "assistant":
            continue
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError(f"Fable tool calls are invalid for {instance_id}")
        if not calls:
            continue
        commands = [
            _canonical_bash_command(call, instance_id=instance_id)
            for call in calls
        ]
        if message["loss"] is not True:
            continue
        stats["canonical_bash_tool_turns"] += 1
        if not message["content"].strip():
            message["content"] = _rationale_for(commands[0])
            stats["reasoned_tool_turns"] += 1
    return rendered, stats


def _load_source(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(source).resolve()
    manifest_path = source / "manifest.json"
    train_path = source / "train.jsonl"
    manifest = _read_object(manifest_path)
    rows = _read_rows(train_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("rendered") != len(rows)
        or manifest.get("training_admitted") != len(rows)
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_path)
        or manifest.get("canonical_rows_sha256") != canonical_rows_sha256(rows)
    ):
        raise ValueError("source training dataset is not fully bound and admitted")
    return rows, manifest


def _load_lite_ids(path: Path) -> set[str]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Lite evaluation IDs are invalid") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Lite evaluation IDs are invalid")
    return set(value)


def build_reasoned_mix(
    *,
    source: Path,
    lite_ids: Path,
    out: Path,
) -> dict[str, Any]:
    source = Path(source).resolve()
    lite_ids = Path(lite_ids).resolve()
    out = Path(out).resolve()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    rows, source_manifest = _load_source(source)
    lite = _load_lite_ids(lite_ids)
    source_ids = {
        value
        for row in rows
        for value in (row.get("instance_id"), row.get("source_instance_id"))
        if isinstance(value, str)
    }
    if source_ids & lite:
        raise ValueError("source training dataset overlaps Lite evaluation IDs")
    rendered_rows: list[dict[str, Any]] = []
    fable_rows = 0
    canonical_bash_tool_turns = 0
    reasoned_tool_turns = 0
    for row in rows:
        rendered, stats = render_fable_row(row)
        rendered_rows.append(rendered)
        if _is_fable(row):
            fable_rows += 1
        canonical_bash_tool_turns += stats["canonical_bash_tool_turns"]
        reasoned_tool_turns += stats["reasoned_tool_turns"]
    if not fable_rows or not reasoned_tool_turns:
        raise ValueError("source training dataset has no blank supervised Fable tool turns")
    instance_ids = [row["instance_id"] for row in rendered_rows]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("rendered training dataset has duplicate instance IDs")
    manifest_holder: dict[str, Any] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(rendered_rows).save_to_disk(str(stage))
        train_path = stage / "train.jsonl"
        with train_path.open("wb") as handle:
            for row in rendered_rows:
                handle.write(_canonical_json(row) + b"\n")
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11_fable_reasoned_v1",
            "source_variant": source_manifest["dataset_variant"],
            "rendered": len(rendered_rows),
            "training_admitted": len(rendered_rows),
            "all_training_gates_complete": False,
            "canonical_rows_sha256": canonical_rows_sha256(rendered_rows),
            "train_jsonl_sha256": _sha256_path(train_path),
            "evaluation_overlap": 0,
            "fable_rows": fable_rows,
            "canonical_bash_tool_turns": canonical_bash_tool_turns,
            "reasoned_tool_turns": reasoned_tool_turns,
            "source": {
                "manifest": _binding(source / "manifest.json"),
                "train": _binding(source / "train.jsonl"),
            },
            "lite_ids": _binding(lite_ids),
            "structural_gates": {
                "all_messages_have_boolean_loss": True,
                "all_fable_supervised_tool_calls_are_canonical_bash": True,
                "reasoned_tool_turns": reasoned_tool_turns,
                "evaluation_overlap": 0,
            },
            "format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
                "samples": None,
            },
        }
        _write_json_atomic(stage / "manifest.json", manifest)
        manifest_holder.update(manifest)

    _publish_directory_atomic(out, publish)
    return manifest_holder


def seal_reasoned_mix(*, out: Path, format_report: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    format_report = Path(format_report).resolve()
    manifest_path = out / "manifest.json"
    train_path = out / "train.jsonl"
    manifest = _read_object(manifest_path)
    report = _read_object(format_report)
    if (
        manifest.get("dataset_variant") != "teacher_train_mix_v2p11_fable_reasoned_v1"
        or manifest.get("all_training_gates_complete") is not False
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_path)
    ):
        raise ValueError("reasoned training dataset is not sealable")
    rows = _read_rows(train_path)
    if (
        manifest.get("rendered") != len(rows)
        or manifest.get("training_admitted") != len(rows)
        or manifest.get("canonical_rows_sha256") != canonical_rows_sha256(rows)
    ):
        raise ValueError("reasoned training dataset changed before sealing")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("reasoned training source binding is invalid")
    for field in ("manifest", "train"):
        binding = source.get(field)
        if not isinstance(binding, Mapping):
            raise ValueError("reasoned training source binding is invalid")
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or binding.get("sha256") != _sha256_path(path):
            raise ValueError("reasoned training source changed before sealing")
    report_data = report.get("data")
    if (
        not isinstance(report_data, str)
        or Path(report_data).resolve() != out
        or report.get("samples") != len(rows)
        or report.get("failure_count") != 0
    ):
        raise ValueError("format-loss report does not prove the full reasoned dataset")
    report_name = f"format-gate.{_sha256_path(format_report)}.json"
    report_copy = out / report_name
    _write_json_atomic(report_copy, report)
    updated = dict(manifest)
    updated["all_training_gates_complete"] = True
    updated["format_loss_gate"] = {
        "status": "passed_full_dataset_verification",
        "failure_count": 0,
        "samples": len(rows),
        "report": _binding(report_copy),
    }
    _write_json_atomic(manifest_path, updated)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--lite-ids", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--out", type=Path, required=True)
    seal.add_argument("--format-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_reasoned_mix(
            source=args.source,
            lite_ids=args.lite_ids,
            out=args.out,
        )
    else:
        manifest = seal_reasoned_mix(out=args.out, format_report=args.format_report)
    print(json.dumps({
        "dataset_variant": manifest["dataset_variant"],
        "rendered": manifest["rendered"],
        "fable_rows": manifest["fable_rows"],
        "reasoned_tool_turns": manifest["reasoned_tool_turns"],
        "all_training_gates_complete": manifest["all_training_gates_complete"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
