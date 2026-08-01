#!/usr/bin/env python3
"""Validate all immutable inputs for v2.11 recovery SFT plus behavior KTO."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseD_sft.build_v2p11_recovery_curriculum import (
    PRIVATE_SUBMISSION_MARKER,
    _TEST_RE,
    _is_source_mutation,
    _tool_command,
)
from phaseE_rl.kto_swe_outcome import validate_training_manifest


PRODUCTION_RECOVERY_ROWS = 138
PRODUCTION_RECOVERY_TRAIN_SHA256 = (
    "eda502d05f736426728651953e76330d24addd39471e82bf5fe423c896ba4b66"
)
PRODUCTION_RECOVERY_MANIFEST_SHA256 = (
    "c996953fb810f3e4642bd458dde7226e245bcb99ac02fec249e38a6d9d21fc2a"
)
PRODUCTION_BEHAVIOR_DATA_SHA256 = (
    "525526bffe58a64a3e42b0d733d87b60dd62e3d2dbaa8362de7b74b00a6699e9"
)
PRODUCTION_BEHAVIOR_MANIFEST_SHA256 = (
    "2b77db5de27de7f3880ac711fd0721efb8c00ffe82ce8ca6f219225c5fb43122"
)
PRODUCTION_EXCLUSION_SHA256 = (
    "00171a1e7796fac2103317bcf4f05af42e46ff92db0041fd4a26ccbaaa7e4bd8"
)
PRODUCTION_BEHAVIOR_COUNTS = {
    "empty_terminal": 33,
    "repeated_read_loop": 55,
    "wrong_nonempty_replay": 228,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"invalid JSONL artifact: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row: {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _normalized_repo(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("__", "/")


def _repo_from_instance_id(instance_id: str) -> str:
    if "__" not in instance_id:
        return ""
    owner, remainder = instance_id.split("__", 1)
    repository = remainder.split(".", 1)[0]
    return _normalized_repo(f"{owner}/{repository}")


def _validate_recovery_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_ids: set[str],
    excluded_repos: set[str],
) -> None:
    seen_training_ids: set[str] = set()
    for index, row in enumerate(rows):
        training_id = row.get("instance_id")
        source_id = row.get("source_instance_id")
        repo = _normalized_repo(row.get("repo"))
        messages = row.get("messages")
        if (
            not isinstance(training_id, str)
            or not training_id
            or training_id in seen_training_ids
            or not isinstance(source_id, str)
            or not source_id
            or source_id in excluded_ids
            or repo in excluded_repos
            or not isinstance(messages, list)
        ):
            raise ValueError(f"recovery evaluation overlap or invalid row at {index}")
        seen_training_ids.add(training_id)
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if PRIVATE_SUBMISSION_MARKER in serialized:
            raise ValueError(f"recovery row {index} contains a private marker")
        supervised = [
            message
            for message in messages
            if isinstance(message, Mapping)
            and message.get("role") == "assistant"
            and message.get("loss") is True
        ]
        if len(supervised) != 2:
            raise ValueError(
                f"recovery row {index} must have exactly two supervised actions"
            )
        commands = [_tool_command(message) for message in supervised]
        if not any(_is_source_mutation(command) for command in commands):
            raise ValueError(f"recovery row {index} lacks a source mutation target")
        if not any(_TEST_RE.search(command) for command in commands):
            raise ValueError(f"recovery row {index} lacks a focused test target")


def validate_posttrain_inputs(
    *,
    recovery_data: Path,
    behavior_data: Path,
    behavior_manifest: Path,
    exclusions: Path,
    full_ids: Path,
    expected_recovery_rows: int = PRODUCTION_RECOVERY_ROWS,
    recovery_epochs: int = 3,
    recovery_gradient_accumulation: int = 4,
    require_production_identity: bool = False,
) -> dict[str, Any]:
    recovery_data = Path(recovery_data).resolve()
    behavior_data = Path(behavior_data).resolve()
    behavior_manifest = Path(behavior_manifest).resolve()
    exclusions = Path(exclusions).resolve()
    full_ids = Path(full_ids).resolve()
    if (
        expected_recovery_rows < 1
        or recovery_epochs < 1
        or recovery_gradient_accumulation < 1
    ):
        raise ValueError("posttrain dimensions must be positive")
    recovery_train = recovery_data / "train.jsonl"
    recovery_manifest_path = recovery_data / "manifest.json"
    recovery_rows = _read_jsonl(recovery_train)
    recovery_manifest = _read_json(recovery_manifest_path)
    format_gate = recovery_manifest.get("standard_native_format_loss_gate")
    expected_targets = {
        "rows_with_passing_test_target": expected_recovery_rows,
        "rows_with_source_mutation_target": expected_recovery_rows,
        "supervised_assistant_messages": expected_recovery_rows * 2,
    }
    if (
        len(recovery_rows) != expected_recovery_rows
        or recovery_manifest.get("schema_version") != 2
        or recovery_manifest.get("kind") != "v2p11_recovery_curriculum"
        or recovery_manifest.get("rendered") != expected_recovery_rows
        or recovery_manifest.get("training_admitted") != expected_recovery_rows
        or recovery_manifest.get("all_training_gates_complete") is not True
        or recovery_manifest.get("train_jsonl_sha256") != _sha256(recovery_train)
        or recovery_manifest.get("evaluation_overlap") != 0
        or recovery_manifest.get("private_submission_marker_hits") != 0
        or recovery_manifest.get("target_counts") != expected_targets
        or not isinstance(format_gate, Mapping)
        or format_gate.get("status") != "passed"
        or format_gate.get("samples") != expected_recovery_rows
        or format_gate.get("failure_count") != 0
    ):
        raise ValueError("recovery curriculum manifest is incomplete")

    exclusion = _read_json(exclusions)
    excluded_ids_value = exclusion.get("instance_ids")
    excluded_repos_value = exclusion.get("repo_denylist")
    full_ids_value = json.loads(full_ids.read_text(encoding="utf-8"))
    if (
        not isinstance(excluded_ids_value, list)
        or not isinstance(excluded_repos_value, list)
        or not isinstance(full_ids_value, list)
        or any(not isinstance(value, str) for value in excluded_ids_value)
        or any(not isinstance(value, str) for value in excluded_repos_value)
        or any(not isinstance(value, str) for value in full_ids_value)
        or len(full_ids_value) != len(set(full_ids_value))
        or not set(full_ids_value).issubset(set(excluded_ids_value))
        or recovery_manifest.get("exclusion_sha256") != _sha256(exclusions)
    ):
        raise ValueError("combined evaluation exclusion contract is incomplete")
    excluded_ids = set(excluded_ids_value)
    excluded_repos = {_normalized_repo(value) for value in excluded_repos_value}
    _validate_recovery_rows(
        recovery_rows,
        excluded_ids=excluded_ids,
        excluded_repos=excluded_repos,
    )

    behavior_rows = _read_jsonl(behavior_data)
    behavior_contract = _read_json(behavior_manifest)
    validate_training_manifest(behavior_rows, behavior_contract)
    if (
        behavior_contract.get("output_jsonl_sha256") != _sha256(behavior_data)
        or behavior_contract.get("exclusion_sha256") != _sha256(exclusions)
    ):
        raise ValueError("behavior KTO hashes do not match their bound inputs")
    for index, row in enumerate(behavior_rows):
        source_id = row.get("source_instance_id")
        if (
            not isinstance(source_id, str)
            or source_id in excluded_ids
            or _repo_from_instance_id(source_id) in excluded_repos
        ):
            raise ValueError(f"behavior evaluation overlap at row {index}")

    recovery_manifest_sha256 = _sha256(recovery_manifest_path)
    recovery_train_sha256 = _sha256(recovery_train)
    behavior_data_sha256 = _sha256(behavior_data)
    behavior_manifest_sha256 = _sha256(behavior_manifest)
    exclusion_sha256 = _sha256(exclusions)
    if require_production_identity and (
        expected_recovery_rows != PRODUCTION_RECOVERY_ROWS
        or recovery_epochs != 3
        or recovery_gradient_accumulation != 4
        or len(full_ids_value) != 300
        or len(excluded_ids) != 707
        or len(excluded_repos) != 12
        or recovery_train_sha256 != PRODUCTION_RECOVERY_TRAIN_SHA256
        or recovery_manifest_sha256 != PRODUCTION_RECOVERY_MANIFEST_SHA256
        or behavior_data_sha256 != PRODUCTION_BEHAVIOR_DATA_SHA256
        or behavior_manifest_sha256 != PRODUCTION_BEHAVIOR_MANIFEST_SHA256
        or exclusion_sha256 != PRODUCTION_EXCLUSION_SHA256
        or behavior_contract.get("behavior_negative_counts")
        != PRODUCTION_BEHAVIOR_COUNTS
    ):
        raise ValueError("production v2.11 posttrain identity mismatch")
    return {
        "recovery_rows": len(recovery_rows),
        "recovery_optimizer_steps": (
            math.ceil(len(recovery_rows) / recovery_gradient_accumulation)
            * recovery_epochs
        ),
        "behavior_rows": len(behavior_rows),
        "behavior_negative_counts": behavior_contract[
            "behavior_negative_counts"
        ],
        "full_evaluation_ids": len(full_ids_value),
        "combined_exclusion_ids": len(excluded_ids),
        "excluded_repositories": len(excluded_repos),
        "evaluation_overlap": 0,
        "recovery_train_sha256": recovery_train_sha256,
        "recovery_manifest_sha256": recovery_manifest_sha256,
        "behavior_data_sha256": behavior_data_sha256,
        "behavior_manifest_sha256": behavior_manifest_sha256,
        "exclusion_sha256": exclusion_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-data", type=Path, required=True)
    parser.add_argument("--behavior-data", type=Path, required=True)
    parser.add_argument("--behavior-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--require-production-identity", action="store_true")
    args = parser.parse_args(argv)
    result = validate_posttrain_inputs(
        recovery_data=args.recovery_data,
        behavior_data=args.behavior_data,
        behavior_manifest=args.behavior_manifest,
        exclusions=args.exclusions,
        full_ids=args.full_ids,
        require_production_identity=args.require_production_identity,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
