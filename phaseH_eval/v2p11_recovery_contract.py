#!/usr/bin/env python3
"""Validate the decontaminated model-level recovery continuation for v2.11."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseD_sft.build_v2p11_recovery_curriculum import _is_source_mutation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROWS = 46
PRODUCTION_TRAIN_SHA256 = (
    "c778505ced4c2bd0029508ddb43a4d0c8eed091c8c173fee152436bae94ee1a4"
)
PRODUCTION_MANIFEST_SHA256 = (
    "1715044fa998bed97ab2f9d9e69295f9c86291740383146f31bb70ded9a0b30c"
)
PRODUCTION_SOURCE_EXCLUSION_SHA256 = (
    "864f23c3376c6fca35095262b647ee37b087aec64da3e6c38e677ba49ea28f37"
)
PRODUCTION_COMBINED_EXCLUSION_SHA256 = (
    "00171a1e7796fac2103317bcf4f05af42e46ff92db0041fd4a26ccbaaa7e4bd8"
)
PRODUCTION_SOURCE_MANIFESTS = {
    "data/fable5_recent_verified_revision_v1": (
        "ba2f52279299a64c5c059705371e42c13e12d88e7be322d0c586b10db26e3b2d"
    ),
    "data/fable5_v2p11_frozen47_prepared_safe36": (
        "3708c56a6fa6b2baf68fc69b523c90e9ed30b70936ab67e8bd3e766ce68121e6"
    ),
}
PRODUCTION_SOURCE_COUNTS = {
    "data/fable5_recent_verified_revision_v1": 10,
    "data/fable5_v2p11_frozen47_prepared_safe36": 36,
}
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"invalid JSONL artifact: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL artifact: {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


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


def _tool_command(message: Mapping[str, Any]) -> str:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return ""
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != "bash":
        return ""
    try:
        arguments = json.loads(str(function.get("arguments") or ""))
    except json.JSONDecodeError:
        return ""
    command = arguments.get("command") if isinstance(arguments, Mapping) else None
    return command if isinstance(command, str) else ""


def _portable_target_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    literal_patch_rows = 0
    mutation_rows = 0
    marker_targets = 0
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("recovery row is missing messages")
        verifier_feedback_index = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, Mapping)
                and message.get("role") == "user"
                and "VERIFIER FEEDBACK" in str(message.get("content") or "")
            ),
            None,
        )
        patch_target = False
        mutation_target = False
        row_marker_target = False
        supervised = 0
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if message.get("loss", True) is False:
                continue
            next_content = (
                str(messages[index + 1].get("content") or "")
                if index + 1 < len(messages)
                and isinstance(messages[index + 1], Mapping)
                else ""
            )
            if (
                verifier_feedback_index is not None
                and index < verifier_feedback_index
            ) or "legacy_clean_reference_context" in next_content:
                raise ValueError(
                    "recovery row supervises an assistant before verifier feedback"
                )
            supervised += 1
            command = _tool_command(message)
            content = message.get("content")
            content_text = content if isinstance(content, str) else ""
            patch_target |= (
                "diff --git " in content_text
                or command.lstrip().startswith("git apply -")
            )
            mutation_target |= _is_source_mutation(command)
            row_marker_target |= "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
        if supervised == 0:
            raise ValueError("recovery row has no supervised assistant target")
        if row_marker_target:
            raise ValueError(
                "recovery data supervises a harness-specific submission marker"
            )
        if not mutation_target:
            raise ValueError("recovery row lacks a portable source mutation")
        literal_patch_rows += int(patch_target)
        mutation_rows += int(mutation_target)
        marker_targets += int(row_marker_target)
    return {
        "rows_with_literal_patch_target": literal_patch_rows,
        "rows_with_supervised_source_mutation": mutation_rows,
        "harness_marker_targets": marker_targets,
    }


def validate_recovery_data(
    data: Path,
    *,
    full_ids_path: Path,
    exclusions_path: Path,
    expected_rows: int = PRODUCTION_ROWS,
    require_production_identity: bool = False,
    gradient_accumulation: int = 4,
    epochs: int = 3,
) -> dict[str, Any]:
    data = Path(data).resolve()
    full_ids_path = Path(full_ids_path).resolve()
    exclusions_path = Path(exclusions_path).resolve()
    if expected_rows < 1 or gradient_accumulation < 1 or epochs < 1:
        raise ValueError("recovery dimensions must be positive")
    manifest_path = data / "manifest.json"
    train_path = data / "train.jsonl"
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(train_path)
    full_ids_value = _read_json(full_ids_path)
    combined_exclusions = _read_json(exclusions_path)
    if (
        not isinstance(full_ids_value, list)
        or len(full_ids_value) != len(set(full_ids_value))
        or any(not isinstance(instance_id, str) for instance_id in full_ids_value)
    ):
        raise ValueError("full evaluation IDs must be a unique string list")
    full_ids = set(full_ids_value)
    combined_ids_value = combined_exclusions.get("instance_ids")
    excluded_repos_value = combined_exclusions.get("repo_denylist")
    if (
        not isinstance(combined_ids_value, list)
        or len(combined_ids_value) != len(set(combined_ids_value))
        or any(not isinstance(instance_id, str) for instance_id in combined_ids_value)
        or not isinstance(excluded_repos_value, list)
        or len(excluded_repos_value) != len(set(excluded_repos_value))
        or any(not isinstance(repo, str) for repo in excluded_repos_value)
        or not full_ids.issubset(set(combined_ids_value))
    ):
        raise ValueError("combined evaluation exclusion is incomplete")
    combined_ids = set(combined_ids_value)
    excluded_repos = {
        _normalized_repo(repo)
        for repo in excluded_repos_value
    }
    combined_exclusion_sha256 = _sha256(exclusions_path)
    row_ids = [row.get("instance_id") for row in rows]
    gate = manifest.get("standard_native_format_loss_gate")
    sources = manifest.get("sources")
    per_source = manifest.get("per_source")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "teacher_blend"
        or manifest.get("rendered") != expected_rows
        or manifest.get("training_admitted") != expected_rows
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("train_jsonl_sha256") != _sha256(train_path)
        or not isinstance(gate, Mapping)
        or gate.get("status") != "passed"
        or gate.get("samples") != expected_rows
        or gate.get("failure_count") != 0
        or not isinstance(sources, list)
        or len(sources) != 2
        or not isinstance(per_source, Mapping)
        or len(rows) != expected_rows
        or len(row_ids) != len(set(row_ids))
        or any(not isinstance(instance_id, str) for instance_id in row_ids)
        or full_ids.intersection(row_ids)
        or any(
            instance_id in combined_ids
            or (
                _normalized_repo(row.get("repo"))
                or _repo_from_instance_id(instance_id)
            ) in excluded_repos
            for instance_id, row in zip(row_ids, rows)
        )
    ):
        if any(
            isinstance(instance_id, str)
            and (
                instance_id in combined_ids
                or (
                    _normalized_repo(row.get("repo"))
                    or _repo_from_instance_id(instance_id)
                ) in excluded_repos
            )
            for instance_id, row in zip(row_ids, rows)
        ):
            raise ValueError("recovery row violates combined evaluation exclusion")
        raise ValueError("portable recovery dataset contract is incomplete")

    expected_blend: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_exclusion_sha256: str | None = None
    source_manifest_hashes: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for source in sources:
        if (
            not isinstance(source, Mapping)
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("manifest_sha256"), str)
        ):
            raise ValueError("portable recovery source binding is incomplete")
        source_text = source["path"]
        source_root = _resolve(source_text)
        source_manifest_path = source_root / "manifest.json"
        source_train_path = source_root / "train.jsonl"
        source_manifest = _read_json(source_manifest_path)
        manifest_sha256 = _sha256(source_manifest_path)
        if (
            manifest_sha256 != source["manifest_sha256"]
            or source_manifest.get("schema_version") != 2
            or source_manifest.get("all_training_gates_complete") is not True
            or source_manifest.get("training_admitted")
            != source_manifest.get("rendered")
        ):
            raise ValueError("portable recovery source admission changed")
        exclusions = source_manifest.get("exclusions")
        artifacts = exclusions.get("artifacts") if isinstance(exclusions, Mapping) else None
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != 1
            or not isinstance(artifacts[0], Mapping)
            or not isinstance(artifacts[0].get("path"), str)
            or not isinstance(artifacts[0].get("sha256"), str)
        ):
            raise ValueError("portable recovery source lacks exclusion evidence")
        exclusion_path = _resolve(artifacts[0]["path"])
        current_exclusion_sha256 = _sha256(exclusion_path)
        exclusion = _read_json(exclusion_path)
        excluded_ids = exclusion.get("instance_ids")
        if (
            current_exclusion_sha256 != artifacts[0]["sha256"]
            or not isinstance(excluded_ids, list)
        ):
            raise ValueError("portable recovery source exclusion changed")
        if source_exclusion_sha256 not in (None, current_exclusion_sha256):
            raise ValueError("portable recovery sources use different exclusions")
        source_exclusion_sha256 = current_exclusion_sha256

        source_rows = _read_jsonl(source_train_path)
        selected = 0
        for row in source_rows:
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("portable recovery source has invalid instance ID")
            repo = (
                _normalized_repo(row.get("repo"))
                or _repo_from_instance_id(instance_id)
            )
            if instance_id in combined_ids or repo in excluded_repos:
                raise ValueError(
                    "recovery source violates combined evaluation exclusion"
                )
            if instance_id in seen:
                continue
            seen.add(instance_id)
            expected_blend.append({
                "instance_id": instance_id,
                "messages": row["messages"],
                "repo": row.get("repo", ""),
                "source": row.get("source", source_text),
            })
            selected += 1
        if source.get("rows") != len(source_rows) or source.get("selected") != selected:
            raise ValueError("portable recovery source row counts changed")
        source_manifest_hashes[source_text] = manifest_sha256
        source_counts[source_text] = selected

    if rows != expected_blend or dict(per_source) != source_counts:
        raise ValueError("portable recovery blend differs from bound sources")
    target_counts = _portable_target_counts(rows)
    manifest_sha256 = _sha256(manifest_path)
    train_sha256 = _sha256(train_path)
    if require_production_identity and (
        expected_rows != PRODUCTION_ROWS
        or len(full_ids) != 300
        or manifest_sha256 != PRODUCTION_MANIFEST_SHA256
        or train_sha256 != PRODUCTION_TRAIN_SHA256
        or source_exclusion_sha256 != PRODUCTION_SOURCE_EXCLUSION_SHA256
        or combined_exclusion_sha256 != PRODUCTION_COMBINED_EXCLUSION_SHA256
        or len(combined_ids) != 707
        or len(excluded_repos) != 12
        or target_counts.get("rows_with_supervised_source_mutation")
        != PRODUCTION_ROWS
        or target_counts.get("harness_marker_targets") != 0
        or source_manifest_hashes != PRODUCTION_SOURCE_MANIFESTS
        or source_counts != PRODUCTION_SOURCE_COUNTS
    ):
        raise ValueError("production portable recovery identity mismatch")
    return {
        "data": str(data),
        "rows": len(rows),
        "full_evaluation_ids": len(full_ids),
        "evaluation_overlap": 0,
        "source_counts": source_counts,
        "source_manifest_sha256": source_manifest_hashes,
        "source_exclusion_sha256": source_exclusion_sha256,
        "combined_exclusion_sha256": combined_exclusion_sha256,
        "combined_exclusion_ids": len(combined_ids),
        "excluded_repositories": len(excluded_repos),
        "train_jsonl_sha256": train_sha256,
        "manifest_sha256": manifest_sha256,
        "gradient_accumulation": gradient_accumulation,
        "epochs": epochs,
        "optimizer_steps": math.ceil(expected_rows / gradient_accumulation) * epochs,
        **target_counts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--require-production-identity", action="store_true")
    args = parser.parse_args(argv)
    result = validate_recovery_data(
        args.data,
        full_ids_path=args.full_ids,
        exclusions_path=args.exclusions,
        require_production_identity=args.require_production_identity,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
