#!/usr/bin/env python3
"""Build portable, selective-loss v2.11 recovery curriculum rows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from phaseD_sft.build_mix_v2p10 import _publish_directory_atomic


PRIVATE_SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
RECOVERY_STYLES = {
    "minimal": (
        "You are a software engineer working in a repository checkout. "
        "Inspect the relevant source, make the smallest justified source edit, "
        "run a focused check, and leave a non-empty source diff."
    ),
    "autonomous": (
        "Solve the repository issue autonomously with shell tools. "
        "Do not stop at inspection: make a source change, verify it, and keep "
        "the resulting diff in the worktree."
    ),
    "recovery": (
        "Work directly in the checkout and prefer a small tested source fix. "
        "If a command fails or repeats, change strategy. Finish only after a "
        "source mutation and a passing focused check."
    ),
}
_TEST_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:"
    r"(?:python(?:3)?|/[^;\n ]+/python|\$[A-Za-z_][A-Za-z0-9_]*)"
    r"\s+-m\s+pytest\b|"
    r"pytest\b|"
    r"(?:python(?:3)?|/[^;\n ]+/python)\s+[^;\n]*(?:test|repro)|"
    r"cargo\s+test\b|go\s+test\b|npm\s+test\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_DIRECT_MUTATION_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:"
    r"git\s+apply\b|apply_patch\b|sed\s+-i\b|perl\s+-pi\b|tee\s+"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_REDIRECT_RE = re.compile(r"(?:^|[;&|]\s*)cat\s+[^;\n]*(?:>|>>)\s*(\S+)")
_PYTHON_WRITE_RE = re.compile(
    r"(?:open\s*\([^)]*,\s*['\"][wa]['\"]|"
    r"\.write_text\s*\(|\.write_bytes\s*\()"
)
_TEMP_PATH_RE = re.compile(r"(?:^|[\s'\"])(?:/tmp/|patch\.txt(?:\s|$))")


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
    rows: list[dict[str, Any]] = []
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


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _tool_command(message: Mapping[str, Any]) -> str:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return ""
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != "bash":
        return ""
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return ""
    command = parsed.get("command") if isinstance(parsed, Mapping) else None
    return command if isinstance(command, str) else ""


def _tool_call_id(message: Mapping[str, Any]) -> str:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return ""
    call = calls[0]
    value = call.get("id") if isinstance(call, Mapping) else None
    return value if isinstance(value, str) else ""


def _successful_tool_response(
    messages: Sequence[Mapping[str, Any]],
    assistant_index: int,
) -> bool:
    if assistant_index + 1 >= len(messages):
        return False
    assistant = messages[assistant_index]
    response = messages[assistant_index + 1]
    response_role = response.get("role")
    if response_role not in {"tool", "user"}:
        return False
    call_id = _tool_call_id(assistant)
    if (
        response_role == "tool"
        and call_id
        and response.get("tool_call_id") != call_id
    ):
        return False
    content = response.get("content")
    if not isinstance(content, str):
        return False
    if response_role == "tool":
        return re.search(
            r"<returncode>\s*0\s*</returncode>", content
        ) is not None
    if not content.lstrip().startswith("OBSERVATION:"):
        return False
    tagged = re.search(r"<returncode>\s*([^<]+)\s*</returncode>", content)
    if tagged is not None:
        return tagged.group(1).strip() == "0"
    return (
        re.search(r"\bExit code\s+[1-9][0-9]*\b", content) is None
        and "Traceback (most recent call last)" not in content
    )


def _is_source_mutation(command: str) -> bool:
    if not command or PRIVATE_SUBMISSION_MARKER in command:
        return False
    if _DIRECT_MUTATION_RE.search(command):
        if _TEMP_PATH_RE.search(command) and "git apply" not in command:
            return False
        return True
    redirect = _REDIRECT_RE.search(command)
    if redirect and not redirect.group(1).startswith("/tmp/"):
        return redirect.group(1) != "patch.txt"
    if _PYTHON_WRITE_RE.search(command):
        return not _TEMP_PATH_RE.search(command)
    return False


def _is_passing_test(
    messages: Sequence[Mapping[str, Any]],
    assistant_index: int,
    command: str,
) -> bool:
    return bool(_TEST_RE.search(command)) and _successful_tool_response(
        messages, assistant_index
    )


def _sanitize_content(content: Any, *, style: str, role: str) -> Any:
    if not isinstance(content, str):
        return content
    if role == "system":
        return style
    replacement = "leave a non-empty source diff in the worktree"
    return content.replace(PRIVATE_SUBMISSION_MARKER, replacement)


def _curriculum_messages(
    raw_messages: Sequence[Mapping[str, Any]],
    *,
    style: str,
) -> tuple[list[dict[str, Any]], int]:
    messages = [copy.deepcopy(dict(message)) for message in raw_messages]
    for message in messages:
        role = str(message.get("role") or "")
        message["content"] = _sanitize_content(
            message.get("content"),
            style=style,
            role=role,
        )
        if role == "assistant":
            command = _tool_command(message)
            if command:
                calls = message["tool_calls"]
                function = calls[0]["function"]
                parsed = json.loads(function["arguments"])
                parsed["command"] = parsed["command"].replace(
                    PRIVATE_SUBMISSION_MARKER,
                    "git diff --binary",
                )
                function["arguments"] = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            message["loss"] = False

    mutation_candidates: list[int] = []
    passing_tests: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        command = _tool_command(message)
        if _is_source_mutation(command) and (
            _successful_tool_response(messages, index)
            or (
                command.lstrip().startswith("git apply ")
                and index == len(messages) - 1
            )
        ):
            mutation_candidates.append(index)
        if _is_passing_test(messages, index, command):
            passing_tests.append(index)
    if not mutation_candidates:
        raise ValueError("recovery row lacks a successful source mutation")
    mutation_index = mutation_candidates[-1]
    viable_tests = [
        index
        for index in passing_tests
        if index > mutation_index
    ]
    if not viable_tests:
        raise ValueError(
            "recovery row lacks a passing focused test after source mutation"
        )
    test_index = viable_tests[-1]
    messages[mutation_index]["loss"] = True
    messages[test_index]["loss"] = True
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    if PRIVATE_SUBMISSION_MARKER in serialized:
        raise ValueError("private submission marker survived recovery sanitization")
    return messages, 2


def _normalized_repo(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("__", "/")


def build_recovery_curriculum(
    *,
    source: Path,
    output: Path,
    exclusion_path: Path,
) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    exclusion_path = Path(exclusion_path).resolve()
    source_manifest_path = source / "manifest.json"
    source_train_path = source / "train.jsonl"
    source_manifest = _read_json(source_manifest_path)
    source_rows = _read_jsonl(source_train_path)
    gate = source_manifest.get("standard_native_format_loss_gate")
    if (
        source_manifest.get("schema_version") != 2
        or source_manifest.get("all_training_gates_complete") is not True
        or source_manifest.get("training_admitted") != len(source_rows)
        or source_manifest.get("rendered") != len(source_rows)
        or source_manifest.get("train_jsonl_sha256") != _sha256(source_train_path)
        or not isinstance(gate, Mapping)
        or gate.get("status") != "passed"
        or gate.get("failure_count") != 0
    ):
        raise ValueError("source recovery dataset is not fully admitted")
    exclusions = _read_json(exclusion_path)
    excluded_ids_value = exclusions.get("instance_ids")
    excluded_repos_value = exclusions.get("repo_denylist")
    if (
        not isinstance(excluded_ids_value, list)
        or not isinstance(excluded_repos_value, list)
        or any(not isinstance(value, str) for value in excluded_ids_value)
        or any(not isinstance(value, str) for value in excluded_repos_value)
    ):
        raise ValueError("evaluation exclusion artifact is malformed")
    excluded_ids = set(excluded_ids_value)
    excluded_repos = {_normalized_repo(value) for value in excluded_repos_value}

    output_rows: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for row in source_rows:
        source_id = row.get("instance_id")
        messages = row.get("messages")
        repo = row.get("repo", "")
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in seen_source_ids
            or source_id in excluded_ids
            or _normalized_repo(repo) in excluded_repos
            or not isinstance(messages, list)
        ):
            raise ValueError("source recovery identity overlaps exclusions or is invalid")
        seen_source_ids.add(source_id)
        for style_name, style_prompt in RECOVERY_STYLES.items():
            curriculum, _ = _curriculum_messages(messages, style=style_prompt)
            output_rows.append(
                {
                    "instance_id": f"{source_id}::v2p11-{style_name}",
                    "source_instance_id": source_id,
                    "augmentation_style": style_name,
                    "messages": curriculum,
                    "repo": repo,
                    "source": row.get("source", ""),
                }
            )
    if not output_rows:
        raise ValueError("recovery curriculum is empty")

    train_bytes = _canonical_jsonl_bytes(output_rows)
    target_counts = {
        "rows_with_passing_test_target": len(output_rows),
        "rows_with_source_mutation_target": len(output_rows),
        "supervised_assistant_messages": len(output_rows) * 2,
    }
    manifest_holder: dict[str, Any] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(output_rows).save_to_disk(str(stage))
        (stage / "train.jsonl").write_bytes(train_bytes)
        manifest = {
            "schema_version": 2,
            "kind": "v2p11_recovery_curriculum",
            "source": str(source),
            "source_rows": len(source_rows),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_train_sha256": _sha256(source_train_path),
            "exclusion_path": str(exclusion_path),
            "exclusion_sha256": _sha256(exclusion_path),
            "styles": list(RECOVERY_STYLES),
            "rendered": len(output_rows),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "train_jsonl_sha256": hashlib.sha256(train_bytes).hexdigest(),
            "target_counts": target_counts,
            "private_submission_marker_hits": 0,
            "evaluation_overlap": 0,
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(output, publish)
    return {
        "output": str(output),
        "rows": len(output_rows),
        "source_rows": len(source_rows),
        "styles": list(RECOVERY_STYLES),
        "target_counts": target_counts,
        "train_jsonl_sha256": manifest_holder["train_jsonl_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_recovery_curriculum(
        source=args.source,
        output=args.output,
        exclusion_path=args.exclusions,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
