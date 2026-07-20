#!/usr/bin/env python3
"""Build the leakage-free Rust v3 agentic SFT corpus."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from phaseD_sft.compress_editfirst import READ_COMMAND_RE, edited_paths
from phaseD_sft.convert_external_traces import (
    _extract_bash_command_from_tool,
    _parse_tc_args,
    _translate_sre_command,
)


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
_TRAJECTORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_UPLOADED_FILES_RE = re.compile(
    r"<uploaded_files>\s*(.*?)\s*</uploaded_files>", re.DOTALL | re.IGNORECASE
)
_PATCH_APPLY_OPERATOR_RE = re.compile(
    r"(?:^|&&|\|\||[;|\n])\s*(?:apply_patch|git\s+apply)\b"
)
_EMBEDDED_PATCH_MARKER_RE = re.compile(
    r"diff --git a/|\*\*\* Update File:\s*"
)
_HEREDOC_START_RE = re.compile(
    r"<<(?P<strip>-)?\s*(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P=quote)"
)
_RUST_VERIFY_RE = re.compile(
    r"(?:^|&&|\|\||[;|\n])\s*(?:"
    r"cargo\s+(?:check|test|build|clippy|nextest)\b|"
    r"rustc\b|(?:just|make|ninja)\s+(?:check|test|build)\b"
    r")"
)
_COMPLETE_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_TEST_PATH_PARTS = frozenset(
    {
        "test",
        "tests",
        "testing",
        "bench",
        "benches",
        "benchmarks",
        "examples",
        "fixtures",
    }
)


@dataclass(frozen=True)
class SourceIdentity:
    task_id: str
    repository: str
    trajectory_id: str
    config: str
    split: str


@dataclass(frozen=True)
class TrackedRustPaths:
    """All textual paths in a patch and its safe Rust path subsets."""

    tracked_paths: tuple[str, ...]
    source_paths: tuple[str, ...]
    textual_paths: tuple[str, ...]


@dataclass(frozen=True)
class TrackedMutation:
    command_index: int
    message_index: int
    command: str
    edited_paths: tuple[str, ...]
    tracked_paths: tuple[str, ...]


@dataclass(frozen=True)
class AnalyzedSourceRow:
    identity: SourceIdentity
    trajectory: tuple[Mapping[str, Any], ...]
    patch: str
    patch_paths: TrackedRustPaths
    tracked_mutations: tuple[TrackedMutation, ...]


@dataclass(frozen=True)
class NativeToolPair:
    raw_assistant_index: int
    assistant_index: int
    observation_index: int
    command: str


@dataclass(frozen=True)
class NativeTrajectory:
    messages: tuple[dict[str, Any], ...]
    tool_pairs: tuple[NativeToolPair, ...]
    terminal_index: int


@dataclass(frozen=True)
class CompressedAgenticRow:
    identity: SourceIdentity
    messages: tuple[dict[str, Any], ...]
    original_messages: tuple[dict[str, Any], ...]
    original_suffix_start: int
    compressed_suffix_start: int


@dataclass(frozen=True)
class SourceRowStream:
    """One single-pass source partition and its immutable provenance."""

    source: str
    config: str
    split: str
    rows: Iterable[Mapping[str, Any]]


@dataclass(frozen=True)
class AuditInputs:
    """Preregistered source boundary and frozen holdout inputs."""

    dataset_revision: str
    exclusion_path: Path
    expected_exclusion_count: int
    expected_exclusion_sha256: str
    expected_pre_exclusion_eligible: int


@dataclass(frozen=True)
class BuildInputs(AuditInputs):
    """Audit inputs plus immutable rendering and publication inputs."""

    tokenizer_identity: str
    tokenizer_revision: str
    tokenizer_artifact_sha256: str
    template_identity: str
    template_content: str
    expected_template_sha256: str
    max_tokens: int = 49_152
    max_workers: int = 4


@dataclass
class _PatchSection:
    old_path: str
    new_path: str
    minus_path: str | None = None
    plus_path: str | None = None
    rename_from: str | None = None
    rename_to: str | None = None
    saw_text_hunk: bool = False
    saw_binary: bool = False


def _reject(reason: str) -> dict[str, object]:
    return {"kept": False, "reason": reason}


def _normalize_patch_path(raw: str) -> str:
    value = raw.strip()
    if value in {"", "/dev/null"}:
        return ""
    if value.startswith(("a/", "b/")):
        value = value[2:]
    if "\x00" in value or "\\" in value or value.startswith(("/", "~")):
        raise ValueError("unsafe_patch_path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe_patch_path")
    normalized = path.as_posix()
    if normalized.startswith("../") or normalized == "..":
        raise ValueError("unsafe_patch_path")
    return normalized


def _header_values(line: str, prefix: str, expected: int) -> list[str]:
    try:
        values = shlex.split(line)
    except ValueError as exc:
        raise ValueError("malformed_model_patch") from exc
    if not values or values[0] != prefix or len(values) != expected:
        raise ValueError("malformed_model_patch")
    return values


def _is_test_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    lowered_parts = {part.lower() for part in parsed.parts[:-1]}
    stem = parsed.stem.lower()
    return (
        bool(lowered_parts & _TEST_PATH_PARTS)
        or stem in {"test", "tests"}
        or stem.endswith("_test")
    )


def extract_tracked_rust_paths(
    model_patch: object,
) -> tuple[TrackedRustPaths | None, dict[str, object]]:
    """Extract safe row-local Rust paths from one unified model patch."""
    if not isinstance(model_patch, str) or not model_patch.strip():
        return None, _reject("malformed_model_patch")

    sections: list[_PatchSection] = []
    current: _PatchSection | None = None
    try:
        for line in model_patch.splitlines():
            if line.startswith("diff --git "):
                values = _header_values(line, "diff", 4)
                if values[1] != "--git":
                    raise ValueError("malformed_model_patch")
                current = _PatchSection(
                    old_path=_normalize_patch_path(values[2]),
                    new_path=_normalize_patch_path(values[3]),
                )
                if not current.old_path or not current.new_path:
                    raise ValueError("malformed_model_patch")
                sections.append(current)
            elif current is not None and current.saw_text_hunk:
                # File-header-looking source lines are prefixed by the diff
                # marker and can therefore literally begin ``--- ``/``+++ ``.
                # Once a hunk starts, only the next ``diff --git`` ends it.
                continue
            elif line.startswith("rename from "):
                if current is None or current.rename_from is not None:
                    raise ValueError("malformed_model_patch")
                current.rename_from = _normalize_patch_path(
                    line.removeprefix("rename from ")
                )
            elif line.startswith("rename to "):
                if current is None or current.rename_to is not None:
                    raise ValueError("malformed_model_patch")
                current.rename_to = _normalize_patch_path(line.removeprefix("rename to "))
            elif line.startswith("--- ") or line.startswith("+++ "):
                if current is None:
                    raise ValueError("malformed_model_patch")
                prefix = line[:3]
                values = _header_values(line, prefix, 2)
                normalized = _normalize_patch_path(values[1])
                attribute = "minus_path" if prefix == "---" else "plus_path"
                if getattr(current, attribute) is not None:
                    raise ValueError("malformed_model_patch")
                setattr(current, attribute, normalized)
            elif line.startswith("@@"):
                if current is None:
                    raise ValueError("malformed_model_patch")
                current.saw_text_hunk = True
            elif line.startswith("GIT binary patch") or (
                line.startswith("Binary files ") and line.endswith(" differ")
            ):
                if current is None:
                    raise ValueError("malformed_model_patch")
                current.saw_binary = True

        if not sections:
            return None, _reject("pathless_patch")

        textual_paths: set[str] = set()
        textual_rust_paths: set[str] = set()
        saw_binary_rust = False
        saw_hunkless_rust = False
        for section in sections:
            if (section.rename_from is None) != (section.rename_to is None):
                raise ValueError("malformed_model_patch")
            if section.rename_from is not None and (
                section.rename_from != section.old_path
                or section.rename_to != section.new_path
            ):
                raise ValueError("malformed_model_patch")

            header_paths = {section.old_path, section.new_path}
            header_rust_paths = {
                path for path in header_paths if path.lower().endswith(".rs")
            }
            if section.saw_text_hunk:
                if section.minus_path is None or section.plus_path is None:
                    raise ValueError("malformed_model_patch")
                if section.minus_path not in {"", section.old_path}:
                    raise ValueError("malformed_model_patch")
                if section.plus_path not in {"", section.new_path}:
                    raise ValueError("malformed_model_patch")
                textual_paths.update(header_paths)
                textual_rust_paths.update(header_rust_paths)
            elif header_rust_paths:
                if section.saw_binary:
                    saw_binary_rust = True
                else:
                    saw_hunkless_rust = True
    except ValueError as exc:
        reason = (
            str(exc)
            if str(exc) in {"unsafe_patch_path", "malformed_model_patch"}
            else "malformed_model_patch"
        )
        return None, _reject(reason)

    rust_paths = tuple(sorted(textual_rust_paths))
    if not rust_paths:
        if saw_binary_rust:
            return None, _reject("binary_only_patch")
        if saw_hunkless_rust:
            return None, _reject("rust_path_without_text_hunk")
        return None, _reject("no_rust_paths")
    source_paths = tuple(path for path in rust_paths if not _is_test_path(path))
    if not source_paths:
        return None, _reject("test_only_patch")
    return (
        TrackedRustPaths(rust_paths, source_paths, tuple(sorted(textual_paths))),
        {"kept": True},
    )


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("metadata")
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, Mapping) else None
    return None


def _identity(
    row: Mapping[str, Any], *, config: str, split: str
) -> tuple[SourceIdentity | None, str | None]:
    task_id, task_reason = _exact_task_id(row)
    if task_id is None:
        return None, task_reason
    repository = str(row.get("repo") or "").strip().removesuffix(".git")
    if not _REPOSITORY_RE.fullmatch(repository) or any(
        part in {".", ".."} for part in repository.split("/")
    ):
        return None, "invalid_repository"
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if not _TRAJECTORY_ID_RE.fullmatch(trajectory_id) or ".." in trajectory_id:
        return None, "invalid_trajectory_id"
    if not isinstance(config, str) or not config.strip():
        return None, "invalid_config"
    if not isinstance(split, str) or not split.strip():
        return None, "invalid_split"
    return SourceIdentity(task_id, repository, trajectory_id, config.strip(), split.strip()), None


def _exact_task_id(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Extract only the normalized task identity, before any row transformation."""
    task_id = str(row.get("instance_id") or "").strip()
    if not _TASK_ID_RE.fullmatch(task_id) or ".." in task_id:
        return None, "invalid_task_id"
    return task_id, None


def _trajectory(value: object) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if not value or not all(isinstance(message, Mapping) for message in value):
        return None
    return tuple(value)  # type: ignore[arg-type]


def _declared_root(trajectory: Sequence[Mapping[str, Any]]) -> str | None:
    for message in trajectory:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = _UPLOADED_FILES_RE.search(content)
        if not match:
            continue
        roots = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        if len(roots) == 1 and roots[0].startswith("/"):
            root = roots[0].rstrip("/")
            if root and root != "/" and ".." not in PurePosixPath(root).parts:
                return root
        return None
    return None


def _normalize_command_path(raw: str, *, declared_root: str | None) -> str | None:
    value = raw.strip().strip("'\"")
    for root in ("/testbed", declared_root):
        if root and (value == root or value.startswith(root + "/")):
            value = value[len(root) :].lstrip("/")
            break
    if value.startswith(("/", "~")):
        return None
    try:
        return _normalize_patch_path(value)
    except ValueError:
        return None


def _translated_commands(message: Mapping[str, Any]) -> Iterable[str]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        command: str | None = None
        if name in {"bash", "execute_bash"}:
            command = _extract_bash_command_from_tool(call)
        elif name == "str_replace_editor":
            arguments = _parse_tc_args(call)
            if arguments is not None:
                command, reason = _translate_sre_command(arguments)
                if reason is not None:
                    command = None
        if isinstance(command, str) and command.strip():
            yield command


def _tool_result(message: Mapping[str, Any]) -> bool:
    if message.get("role") == "tool":
        return True
    return message.get("role") == "user" and str(
        message.get("content") or ""
    ).lstrip().startswith("OBSERVATION:")


def _result_matches(call_id: str, message: Mapping[str, Any]) -> bool:
    result_id = message.get("tool_call_id")
    return result_id is None or result_id == call_id


def _assistant_text(
    message: Mapping[str, Any], extra_parts: Sequence[str] = ()
) -> str:
    parts: list[str] = []
    for key in ("reasoning_content", "think", "content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    parts.extend(part for part in extra_parts if part.strip())
    return "\n\n".join(parts)


def _native_bash_call(call_id: str, command: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps(
                {"command": command}, ensure_ascii=False, separators=(",", ":")
            ),
        },
    }


def _empty_or_unset(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _empty_compatibility_field(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _canonical_think_ack_content(message: Mapping[str, Any]) -> bool:
    if message.get("tool_calls") not in (None, []):
        return False
    if not _empty_or_unset(message.get("reasoning_content")) or not _empty_or_unset(
        message.get("think")
    ):
        return False
    allowed = {
        "role",
        "content",
        "tool_call_id",
        "tool_calls",
        "reasoning_content",
        "think",
    }
    if any(
        key not in allowed and not _empty_compatibility_field(value)
        for key, value in message.items()
    ):
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    normalized = re.sub(
        r"^\s*OBSERVATION:\s*", "", content, count=1
    ).strip()
    return normalized == "Your thought has been logged."


def _single_supported_executable(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return False
    call = calls[0]
    if not isinstance(call, Mapping):
        return False
    function = call.get("function")
    return isinstance(function, Mapping) and function.get("name") in {
        "bash",
        "execute_bash",
        "str_replace_editor",
    }


def convert_rich_trajectory(
    trajectory: Sequence[Mapping[str, Any]],
) -> tuple[NativeTrajectory | None, dict[str, object]]:
    """Convert strict rich-tool turns into native Bash/OBSERVATION messages."""
    native: list[dict[str, Any]] = []
    pairs: list[NativeToolPair] = []
    terminal_index: int | None = None
    index = 0
    while index < len(trajectory):
        message = trajectory[index]
        buffered_think_prefix: list[str] = []
        role = message.get("role")
        if terminal_index is not None:
            return None, _reject("finish_not_terminal")
        if _tool_result(message):
            return None, _reject("orphan_tool_result")
        if role in {"system", "user"}:
            content = message.get("content")
            if not isinstance(content, str):
                return None, _reject("malformed_trajectory")
            native.append({"role": role, "content": content, "tool_calls": []})
            index += 1
            continue
        if role != "assistant":
            return None, _reject("malformed_trajectory")

        calls = message.get("tool_calls")
        if isinstance(calls, list) and len(calls) == 1:
            only_call = calls[0]
            function = only_call.get("function") if isinstance(only_call, Mapping) else None
            if isinstance(function, Mapping) and function.get("name") == "think":
                call_id = only_call.get("id")
                arguments = _parse_tc_args(only_call) if isinstance(only_call, dict) else None
                thought = arguments.get("thought") if isinstance(arguments, Mapping) else None
                raw_content = message.get("content")
                canonical_header = (
                    isinstance(call_id, str)
                    and bool(call_id)
                    and isinstance(thought, str)
                    and bool(thought.strip())
                    and _empty_or_unset(message.get("reasoning_content"))
                    and _empty_or_unset(message.get("think"))
                    and (raw_content is None or isinstance(raw_content, str))
                )
                if not canonical_header or index + 3 >= len(trajectory):
                    return None, _reject("noncanonical_think_sequence")
                acknowledgement = trajectory[index + 1]
                executable_message = trajectory[index + 2]
                executable_result = trajectory[index + 3]
                if (
                    not _tool_result(acknowledgement)
                    or not _result_matches(call_id, acknowledgement)
                    or not _canonical_think_ack_content(acknowledgement)
                    or not _single_supported_executable(executable_message)
                    or not _tool_result(executable_result)
                ):
                    return None, _reject("noncanonical_think_sequence")
                executable_calls = executable_message.get("tool_calls")
                assert isinstance(executable_calls, list)
                executable_call = executable_calls[0]
                assert isinstance(executable_call, Mapping)
                executable_id = executable_call.get("id")
                if (
                    not isinstance(executable_id, str)
                    or not executable_id
                    or not _result_matches(executable_id, executable_result)
                    or not isinstance(executable_result.get("content"), str)
                ):
                    return None, _reject("noncanonical_think_sequence")
                buffered_think_prefix.append(thought)
                if isinstance(raw_content, str) and raw_content.strip():
                    buffered_think_prefix.append(raw_content)
                index += 2
                message = executable_message
                calls = executable_calls
        if not isinstance(calls, list) or not calls:
            return None, _reject("no_tool_assistant")
        executable: list[tuple[str, str]] = []
        terminal: list[tuple[str, str]] = []
        thought_parts: list[str] = []
        for call in calls:
            if not isinstance(call, dict):
                return None, _reject("malformed_tool_call")
            call_id = call.get("id")
            function = call.get("function") or {}
            if not isinstance(call_id, str) or not call_id:
                return None, _reject("missing_tool_call_id")
            if not isinstance(function, Mapping):
                return None, _reject("malformed_tool_call")
            name = function.get("name")
            arguments = _parse_tc_args(call)
            if arguments is None:
                return None, _reject("malformed_tool_call")
            if name in {"bash", "execute_bash"}:
                command = _extract_bash_command_from_tool(call)
                if not isinstance(command, str) or not command.strip():
                    return None, _reject("malformed_tool_call")
                executable.append((call_id, command))
            elif name == "str_replace_editor":
                command, reason = _translate_sre_command(arguments)
                if reason is not None or not command:
                    return None, _reject("unsupported_trajectory_tool")
                executable.append((call_id, command))
            elif name == "think":
                thought = arguments.get("thought")
                if isinstance(thought, str) and thought.strip():
                    thought_parts.append(thought)
            elif name in {"finish", "submit"}:
                terminal_text = arguments.get("message") or arguments.get("output") or ""
                if not isinstance(terminal_text, str):
                    return None, _reject("malformed_tool_call")
                terminal.append((call_id, terminal_text))
            else:
                return None, _reject("unsupported_trajectory_tool")

        if terminal:
            if executable or len(terminal) != 1:
                return None, _reject("mixed_terminal_tool_calls")
            call_id, _ = terminal[0]
            index += 1
            if index < len(trajectory) and _tool_result(trajectory[index]):
                if not _result_matches(call_id, trajectory[index]):
                    return None, _reject("mismatched_tool_result")
                index += 1
                # A submit that receives an observation was rejected by the
                # harness.  It and its corrective result are protocol noise;
                # the agent must continue and eventually submit again.
                continue
            if index != len(trajectory):
                return None, _reject("finish_not_terminal")
            terminal_index = len(native)
            native.append(
                {
                    "role": "assistant",
                    "content": _assistant_text(message, thought_parts),
                    "tool_calls": [_native_bash_call(call_id, _COMPLETE_COMMAND)],
                }
            )
            continue

        if len(executable) != 1:
            reason = "multiple_executable_tool_calls" if executable else "no_tool_assistant"
            return None, _reject(reason)
        call_id, command = executable[0]
        if index + 1 >= len(trajectory) or not _tool_result(trajectory[index + 1]):
            return None, _reject("orphan_tool_call")
        result = trajectory[index + 1]
        if not _result_matches(call_id, result):
            return None, _reject("mismatched_tool_result")
        result_content = result.get("content")
        if not isinstance(result_content, str):
            return None, _reject("malformed_tool_result")

        assistant_index = len(native)
        native.append(
            {
                "role": "assistant",
                "content": "\n\n".join(
                    part
                    for part in (
                        *buffered_think_prefix,
                        _assistant_text(message, thought_parts),
                    )
                    if part.strip()
                ),
                "tool_calls": [_native_bash_call(call_id, command)],
            }
        )
        observation_index = len(native)
        observation = (
            result_content
            if result_content.lstrip().startswith("OBSERVATION:")
            else f"OBSERVATION:\n{result_content}"
        )
        native.append({"role": "user", "content": observation, "tool_calls": []})
        pairs.append(
            NativeToolPair(index, assistant_index, observation_index, command)
        )
        index += 2

    if terminal_index is None:
        return None, _reject("missing_terminal_finish")
    return NativeTrajectory(tuple(native), tuple(pairs), terminal_index), {"kept": True}


def _mutation_paths(command: str, *, declared_root: str | None) -> tuple[str, ...]:
    """Extract mutation operands without treating inert diff text as an edit."""
    detectable_command = command.replace("pathlib.Path(", "Path(")
    if not _PATCH_APPLY_OPERATOR_RE.search(_shell_visible_text(detectable_command)):
        # ``edited_paths`` deliberately recognizes unified-diff markers for
        # real patch application.  Remove those marker tokens everywhere when
        # the shell has no executable patch operator; quoted/printed patch text
        # is data, even when it appears inline rather than at line start.
        detectable_command = _EMBEDDED_PATCH_MARKER_RE.sub(
            "INERT_PATCH_MARKER ", detectable_command
        )
    return tuple(
        dict.fromkeys(
            normalized
            for raw_path in edited_paths(detectable_command)
            if (
                normalized := _normalize_command_path(
                    raw_path, declared_root=declared_root
                )
            )
        )
    )


def _shell_visible_text(command: str) -> str:
    """Remove heredoc bodies before detecting executable patch operators."""
    visible: list[str] = []
    delimiter: str | None = None
    strip_tabs = False
    for line in command.splitlines():
        if delimiter is not None:
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                delimiter = None
                strip_tabs = False
            continue
        visible.append(line)
        match = _HEREDOC_START_RE.search(line)
        if match is not None:
            delimiter = match.group("delimiter")
            strip_tabs = bool(match.group("strip"))
    return "\n".join(visible)


def classify_tracked_mutations(
    trajectory: Sequence[Mapping[str, Any]], tracked_source_paths: Sequence[str]
) -> tuple[TrackedMutation, ...]:
    """Return exact-path mutations to Rust source paths tracked by this row's patch."""
    tracked = set(tracked_source_paths)
    declared_root = _declared_root(trajectory)
    mutations: list[TrackedMutation] = []
    command_index = 0
    for message_index, message in enumerate(trajectory):
        if message.get("role") != "assistant":
            continue
        for command in _translated_commands(message):
            command_index += 1
            normalized_paths = _mutation_paths(command, declared_root=declared_root)
            matched = tuple(sorted(tracked.intersection(normalized_paths)))
            if matched:
                mutations.append(
                    TrackedMutation(
                        command_index=command_index,
                        message_index=message_index,
                        command=command,
                        edited_paths=normalized_paths,
                        tracked_paths=matched,
                    )
                )
    return tuple(mutations)


def analyze_source_row(
    row: Mapping[str, Any], *, config: str, split: str
) -> tuple[AnalyzedSourceRow | None, dict[str, object]]:
    """Validate and associate one trajectory only with its own row-level patch."""
    if str(row.get("resolved") or "").strip() != "1":
        return None, _reject("not_resolved")
    if str(row.get("language") or "").strip().lower() != "rust":
        return None, _reject("not_rust")
    identity, identity_reason = _identity(row, config=config, split=split)
    if identity is None:
        return None, _reject(identity_reason or "invalid_identity")
    metadata = _metadata(row)
    if metadata is None or "model_patch" not in metadata:
        return None, _reject("missing_model_patch")
    model_patch_metadata = metadata.get("model_patch")
    if not isinstance(model_patch_metadata, Mapping):
        return None, _reject("malformed_model_patch")
    model_patch = model_patch_metadata.get("patch")
    if not isinstance(model_patch, str) or not model_patch.strip():
        return None, _reject("malformed_model_patch")
    patch_paths, patch_report = extract_tracked_rust_paths(model_patch)
    if patch_paths is None:
        return None, patch_report
    trajectory = _trajectory(row.get("trajectory"))
    if trajectory is None:
        return None, _reject("malformed_trajectory")
    # A test/repro write may count only when that exact path is part of this
    # row's resolved patch.  A patch containing *only* tests was rejected above.
    mutations = classify_tracked_mutations(trajectory, patch_paths.tracked_paths)
    if not mutations:
        return None, _reject("no_tracked_source_mutation")
    return (
        AnalyzedSourceRow(identity, trajectory, model_patch, patch_paths, mutations),
        {"kept": True},
    )


def _is_read_command(command: str) -> bool:
    if _mutation_paths(command, declared_root=None):
        return False
    visible = _shell_visible_text(command)
    fragments = re.split(r"(?:&&|\|\||[;|\n])", visible)
    for fragment in fragments:
        stripped = fragment.strip()
        if not stripped or stripped.startswith("cd "):
            continue
        return bool(READ_COMMAND_RE.search(stripped))
    return False


def _shell_read_fragments(command: str) -> list[list[str]] | None:
    fragments: list[list[str]] = []
    for line in _shell_visible_text(command).splitlines():
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            return None
        fragment: list[str] = []
        for token in tokens:
            if token and all(character in ";&|<>" for character in token):
                if token in {">", ">>", "<", "<<"}:
                    return None
                if fragment:
                    fragments.append(fragment)
                    fragment = []
                continue
            fragment.append(token)
        if fragment:
            fragments.append(fragment)
    return fragments


def _cat_operands(tokens: Sequence[str]) -> list[str] | None:
    operands: list[str] = []
    options_done = False
    for token in tokens[1:]:
        if not options_done and token == "--":
            options_done = True
        elif not options_done and token.startswith("-"):
            continue
        else:
            operands.append(token)
    return operands or None


def _head_tail_operands(tokens: Sequence[str]) -> list[str] | None:
    operands: list[str] = []
    index = 1
    options_done = False
    value_options = {"-n", "--lines", "-c", "--bytes"}
    while index < len(tokens):
        token = tokens[index]
        if not options_done and token == "--":
            options_done = True
            index += 1
        elif not options_done and token in value_options:
            if index + 1 >= len(tokens):
                return None
            index += 2
        elif not options_done and token.startswith("-"):
            index += 1
        else:
            operands.append(token)
            index += 1
    return operands or None


def _sed_operands(tokens: Sequence[str]) -> list[str] | None:
    quiet = False
    expression_supplied = False
    positionals: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-n", "--quiet", "--silent"}:
            quiet = True
            index += 1
        elif token in {"-e", "--expression"}:
            if index + 1 >= len(tokens):
                return None
            expression_supplied = True
            index += 2
        elif token.startswith("--expression=") or (
            token.startswith("-e") and token != "-e"
        ):
            expression_supplied = True
            index += 1
        elif token in {"-f", "--file"} or token.startswith("--file="):
            return None
        elif token.startswith("-"):
            return None
        else:
            positionals.append(token)
            index += 1
    if not quiet:
        return None
    if not expression_supplied:
        if len(positionals) < 2:
            return None
        positionals = positionals[1:]
    return positionals or None


def _grep_operands(tokens: Sequence[str]) -> list[str] | None:
    pattern_supplied = False
    positionals: list[str] = []
    index = 1
    value_options = {
        "-g",
        "--glob",
        "-t",
        "--type",
        "--type-add",
        "-m",
        "--max-count",
        "-A",
        "-B",
        "-C",
        "--context",
        "--encoding",
    }
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            positionals.extend(tokens[index + 1 :])
            break
        if token in {"-e", "--regexp"}:
            if index + 1 >= len(tokens):
                return None
            pattern_supplied = True
            index += 2
            continue
        if token.startswith("--regexp=") or (
            token.startswith("-e") and token != "-e"
        ):
            pattern_supplied = True
            index += 1
            continue
        if token in {"-f", "--file"}:
            if index + 1 >= len(tokens):
                return None
            pattern_supplied = True
            index += 2
            continue
        if token in value_options:
            if index + 1 >= len(tokens):
                return None
            index += 2
            continue
        if any(token.startswith(prefix) for prefix in ("--glob=", "--type=")):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        positionals.append(token)
        index += 1
    if not pattern_supplied:
        if len(positionals) < 2:
            return None
        positionals = positionals[1:]
    return positionals or None


def _find_operands(tokens: Sequence[str]) -> list[str] | None:
    roots: list[str] = []
    for token in tokens[1:]:
        if token.startswith(("-", "!", "(")):
            break
        roots.append(token)
    return roots or None


def _read_operands(tokens: Sequence[str]) -> list[str] | None:
    if not tokens:
        return None
    executable = PurePosixPath(tokens[0]).name
    if executable == "cat":
        return _cat_operands(tokens)
    if executable in {"head", "tail"}:
        return _head_tail_operands(tokens)
    if executable == "sed":
        return _sed_operands(tokens)
    if executable in {"rg", "grep"}:
        return _grep_operands(tokens)
    if executable == "find":
        return _find_operands(tokens)
    return None


def _command_target_paths(command: str, *, declared_root: str | None) -> set[str]:
    fragments = _shell_read_fragments(command)
    if fragments is None:
        return set()
    targets: set[str] = set()
    saw_read = False
    for tokens in fragments:
        if PurePosixPath(tokens[0]).name == "cd":
            continue
        operands = _read_operands(tokens)
        if operands is None:
            return set()
        saw_read = True
        for operand in operands:
            normalized = _normalize_command_path(
                operand, declared_root=declared_root
            )
            if normalized:
                targets.add(normalized)
    if not saw_read:
        return set()
    return targets


def _observation_body(message: Mapping[str, Any]) -> str:
    content = str(message.get("content") or "")
    return re.sub(r"^\s*OBSERVATION:\s*", "", content, count=1).strip()


def _normalized_command(command: str) -> str:
    return " ".join(command.split())


def compress_tracked_source_row(
    analyzed: AnalyzedSourceRow,
) -> tuple[CompressedAgenticRow | None, dict[str, object]]:
    """Compress exploration before the first row-local tracked source mutation."""
    compressed, report, _stages = _compress_tracked_source_row_staged(analyzed)
    return compressed, report


def _compress_tracked_source_row_staged(
    analyzed: AnalyzedSourceRow,
) -> tuple[CompressedAgenticRow | None, dict[str, object], frozenset[str]]:
    """Run compression once and expose only the gates actually completed."""
    stages: set[str] = set()
    converted, conversion_report = convert_rich_trajectory(analyzed.trajectory)
    if converted is None:
        return None, conversion_report, frozenset(stages)
    stages.add("pairing")

    first_mutation = analyzed.tracked_mutations[0]
    edit_pair = next(
        (
            pair
            for pair in converted.tool_pairs
            if pair.raw_assistant_index == first_mutation.message_index
        ),
        None,
    )
    if edit_pair is None:
        return None, _reject("tracked_mutation_conversion_mismatch"), frozenset(stages)

    post_edit_pairs = [
        pair
        for pair in converted.tool_pairs
        if pair.assistant_index > edit_pair.assistant_index
    ]
    if not any(
        _RUST_VERIFY_RE.search(_shell_visible_text(pair.command))
        for pair in post_edit_pairs
    ):
        return None, _reject("missing_post_edit_rust_verification"), frozenset(stages)
    stages.add("verification")

    required_paths = set(first_mutation.tracked_paths)
    declared_root = _declared_root(analyzed.trajectory)
    relevant_reads: list[tuple[NativeToolPair, set[str]]] = []
    for pair in converted.tool_pairs:
        if pair.assistant_index >= edit_pair.assistant_index:
            break
        if not _is_read_command(pair.command):
            continue
        if not _observation_body(converted.messages[pair.observation_index]):
            continue
        targets = _command_target_paths(pair.command, declared_root=declared_root)
        covered = required_paths.intersection(targets)
        if covered:
            relevant_reads.append((pair, covered))
    selected_reads = relevant_reads[-3:]
    grounded = set().union(*(covered for _, covered in selected_reads)) if selected_reads else set()
    ungrounded = sorted(required_paths - grounded)
    if ungrounded:
        return (
            None,
            {
                "kept": False,
                "reason": "edited_paths_not_grounded",
                "ungrounded_paths": ungrounded,
            },
            frozenset(stages),
        )
    stages.add("grounding")

    retained_pairs = [pair for pair, _ in selected_reads] + [
        pair
        for pair in converted.tool_pairs
        if pair.assistant_index >= edit_pair.assistant_index
    ]
    previous_command: str | None = None
    read_streak = 0
    for pair in retained_pairs:
        normalized = _normalized_command(pair.command)
        if normalized == previous_command:
            return None, _reject("consecutive_identical_commands"), frozenset(stages)
        previous_command = normalized
        if _is_read_command(pair.command):
            read_streak += 1
            if read_streak > 5:
                return None, _reject("max_read_streak_exceeded"), frozenset(stages)
        else:
            read_streak = 0
    stages.add("repeat_read_streak")

    first_pair_index = min(
        (pair.assistant_index for pair in converted.tool_pairs),
        default=edit_pair.assistant_index,
    )
    kept_indices = set(range(first_pair_index))
    for pair, _ in selected_reads:
        kept_indices.update((pair.assistant_index, pair.observation_index))
    original_suffix_start = edit_pair.assistant_index
    kept_indices.update(range(original_suffix_start, len(converted.messages)))
    prefix_indices = sorted(index for index in kept_indices if index < original_suffix_start)
    compressed_suffix_start = len(prefix_indices)
    messages = tuple(
        copy.deepcopy(converted.messages[index]) for index in sorted(kept_indices)
    )
    if messages[compressed_suffix_start:] != converted.messages[original_suffix_start:]:
        return None, _reject("suffix_preservation_failed"), frozenset(stages)

    return (
        CompressedAgenticRow(
            identity=analyzed.identity,
            messages=messages,
            original_messages=converted.messages,
            original_suffix_start=original_suffix_start,
            compressed_suffix_start=compressed_suffix_start,
        ),
        {
            "kept": True,
            "kept_pre_edit_reads": len(selected_reads),
            "first_edit_command_index": len(selected_reads) + 1,
        },
        frozenset(stages),
    )


def audit_source_rows(
    rows: Iterable[Mapping[str, Any]], *, config: str, split: str
) -> tuple[list[AnalyzedSourceRow], dict[str, int]]:
    """In-memory Task-1 audit helper; later tasks own streaming and publication."""
    counters: Counter[str] = Counter()
    accepted: list[AnalyzedSourceRow] = []
    for row in rows:
        counters["rows_scanned"] += 1
        analyzed, report = analyze_source_row(row, config=config, split=split)
        if analyzed is None:
            counters[f"dropped_{report['reason']}"] += 1
            continue
        accepted.append(analyzed)
        counters["rows_eligible"] += 1
    counters["unique_task_ids"] = len({row.identity.task_id for row in accepted})
    counters["repositories"] = len({row.identity.repository for row in accepted})
    return accepted, dict(counters)


RenderedTokenCounter = Callable[[list[dict[str, Any]]], int]
HfPublisher = Callable[[list[dict[str, Any]], Path], None]
ProgressSink = Callable[[str], None]


def _validated_flat_token_ids(value: Any) -> int:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("rendered token IDs must be one flat sequence")
    if not value:
        raise ValueError("rendered token IDs must be nonempty")
    if any(
        isinstance(token, bool)
        or not isinstance(token, Integral)
        or int(token) < 0
        for token in value
    ):
        raise ValueError("rendered token IDs must be nonnegative integers")
    return len(value)


def rendered_token_count(rendered: Any) -> int:
    """Count exactly one rendered example across HF/list/tensor return shapes."""
    value = rendered
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("rendered token IDs mapping lacks input_ids")
        value = value["input_ids"]

    shape = getattr(value, "shape", None)
    if shape is not None and callable(getattr(value, "tolist", None)):
        try:
            dimensions = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError) as error:
            raise ValueError("rendered token IDs have an invalid shape") from error
        if len(dimensions) == 1 and dimensions[0] > 0:
            payload = value.tolist()
            count = _validated_flat_token_ids(payload)
            if count != dimensions[0]:
                raise ValueError("rendered token IDs shape does not match payload")
            return count
        if len(dimensions) == 2 and dimensions[0] == 1 and dimensions[1] > 0:
            payload = value.tolist()
            if (
                not isinstance(payload, Sequence)
                or isinstance(payload, (str, bytes, bytearray))
                or len(payload) != 1
            ):
                raise ValueError("rendered token IDs shape does not match payload")
            count = _validated_flat_token_ids(payload[0])
            if count != dimensions[1]:
                raise ValueError("rendered token IDs shape does not match payload")
            return count
        raise ValueError("rendered token IDs must represent exactly one example")

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("rendered token IDs must be a sequence")
    if len(value) == 1 and isinstance(value[0], Sequence) and not isinstance(
        value[0], (str, bytes, bytearray)
    ):
        return _validated_flat_token_ids(value[0])
    return _validated_flat_token_ids(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def tokenizer_artifact_sha256(tokenizer: Any) -> str:
    """Hash the actual files emitted by a loaded tokenizer, not its label."""
    with tempfile.TemporaryDirectory(prefix="rust-v3-tokenizer-") as directory:
        root = Path(directory)
        tokenizer.save_pretrained(root)
        payload = bytearray()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            payload.extend(path.relative_to(root).as_posix().encode())
            payload.extend(b"\0")
            payload.extend(path.read_bytes())
            payload.extend(b"\0")
    return _sha256_bytes(bytes(payload))


def canonical_id_sha256(ids: Iterable[str]) -> str:
    """Hash a unique sorted ID set with an unambiguous trailing newline."""
    normalized = sorted(set(ids))
    return _sha256_bytes("".join(f"{value}\n" for value in normalized).encode())


def _load_frozen_exclusions(inputs: AuditInputs) -> tuple[set[str], dict[str, Any]]:
    path = Path(inputs.exclusion_path).resolve()
    raw = path.read_bytes()
    ids: list[str] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid exclusion JSON at line {line_number}") from error
        if isinstance(value, str):
            task_id = value
        elif isinstance(value, Mapping):
            task_id = value.get("instance_id") or value.get("task_id")
        else:
            task_id = None
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"invalid exclusion ID at line {line_number}")
        normalized, _reason = _exact_task_id({"instance_id": task_id})
        if normalized is None:
            raise ValueError(f"invalid exclusion ID at line {line_number}")
        ids.append(normalized)
    unique = set(ids)
    if len(ids) != len(unique):
        raise ValueError("exclusion file contains duplicate IDs")
    if len(unique) != inputs.expected_exclusion_count:
        raise ValueError(
            "exclusion count mismatch: "
            f"expected {inputs.expected_exclusion_count}, got {len(unique)}"
        )
    digest = canonical_id_sha256(unique)
    if digest != inputs.expected_exclusion_sha256:
        raise ValueError(
            "exclusion SHA256 mismatch: "
            f"expected {inputs.expected_exclusion_sha256}, got {digest}"
        )
    return unique, {
        "path": str(path),
        "file_sha256": _sha256_bytes(raw),
        "count": len(unique),
        "sorted_ids_sha256": digest,
    }


def _empty_config_counters() -> Counter[str]:
    return Counter(
        {
            "scanned": 0,
            "resolved": 0,
            "rust": 0,
            "structurally_eligible": 0,
            "excluded": 0,
            "grounding_passed": 0,
            "verification_passed": 0,
            "pairing_passed": 0,
            "repeat_read_streak_passed": 0,
            "compressed_eligible": 0,
            "task_selected": 0,
            "content_dedup_kept": 0,
            "token_budget_kept": 0,
        }
    )


def _command_from_native_message(message: Mapping[str, Any]) -> str | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "bash":
        return None
    try:
        arguments = json.loads(str(function.get("arguments") or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, Mapping) or not isinstance(arguments.get("command"), str):
        return None
    return arguments["command"]


def _max_read_streak(messages: Sequence[Mapping[str, Any]]) -> int:
    current = maximum = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        command = _command_from_native_message(message)
        if command is not None and _is_read_command(command):
            current += 1
            maximum = max(maximum, current)
        elif command is not None:
            current = 0
    return maximum


def _percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]


def _default_hf_publisher(rows: list[dict[str, Any]], path: Path) -> None:
    from datasets import Dataset

    Dataset.from_list(rows).save_to_disk(path)


def _write_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a complete directory without replacing any target."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace rename unsupported: renameat2 unavailable"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace rename unsupported: renamex_np unavailable"
            ) from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise RuntimeError(
            f"atomic no-replace rename unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _config_key(stream: SourceRowStream) -> str:
    return f"{stream.source}/{stream.config}/{stream.split}"


def _default_progress_sink(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _progress_line(
    stream: SourceRowStream, *, scanned: int, elapsed: float, final: bool
) -> str:
    elapsed = max(0.0, elapsed)
    return json.dumps(
        {
            "event": "rust_v3_stream_progress",
            "source": stream.source,
            "config": stream.config,
            "split": stream.split,
            "scanned": scanned,
            "elapsed_seconds": round(elapsed, 3),
            "rate_rows_per_second": round(scanned / elapsed, 3) if elapsed else 0.0,
            "final": final,
        },
        sort_keys=True,
    )


@dataclass
class _ScanResult:
    candidates: list[dict[str, Any]]
    per_config: dict[str, Counter[str]]
    per_config_drops: dict[str, Counter[str]]
    drop_reasons: Counter[str]
    provenance: list[dict[str, str]]
    exclusion_manifest: dict[str, Any]
    structurally_eligible: int
    structural_task_ids: set[str]
    structural_repositories: set[str]


def _scan_streams(
    streams: Iterable[SourceRowStream],
    *,
    inputs: AuditInputs,
    progress_every: int,
    progress_sink: ProgressSink | None,
    monotonic: Callable[[], float],
) -> _ScanResult:
    if not inputs.dataset_revision:
        raise ValueError("dataset revision is required")
    if progress_every < 0:
        raise ValueError("progress_every must be nonnegative")
    exclusions, exclusion_manifest = _load_frozen_exclusions(inputs)
    emit_progress = progress_sink or _default_progress_sink
    candidates: list[dict[str, Any]] = []
    per_config: dict[str, Counter[str]] = {}
    per_config_drops: dict[str, Counter[str]] = {}
    drop_reasons: Counter[str] = Counter()
    provenance: list[dict[str, str]] = []
    structural_task_ids: set[str] = set()
    structural_repositories: set[str] = set()
    structurally_eligible = 0

    for stream in streams:
        if not stream.source or not stream.config or not stream.split:
            raise ValueError("every source stream needs source/config/split provenance")
        key = _config_key(stream)
        if key in per_config:
            raise ValueError(f"duplicate source partition: {key}")
        counters = per_config[key] = _empty_config_counters()
        config_drops = per_config_drops[key] = Counter()
        provenance.append(
            {"source": stream.source, "config": stream.config, "split": stream.split}
        )
        progress_started = monotonic() if progress_every else 0.0
        for raw in stream.rows:
            counters["scanned"] += 1
            if progress_every and counters["scanned"] % progress_every == 0:
                emit_progress(
                    _progress_line(
                        stream,
                        scanned=counters["scanned"],
                        elapsed=monotonic() - progress_started,
                        final=False,
                    )
                )
            if str(raw.get("resolved") or "").strip() == "1":
                counters["resolved"] += 1
            if str(raw.get("language") or "").strip().lower() == "rust":
                counters["rust"] += 1
            task_id, _task_reason = _exact_task_id(raw)
            if task_id is not None and task_id in exclusions:
                counters["excluded"] += 1
                drop_reasons["excluded_task_id"] += 1
                config_drops["excluded_task_id"] += 1
                continue
            analyzed, report = analyze_source_row(
                raw, config=stream.config, split=stream.split
            )
            if analyzed is None:
                reason = str(report["reason"])
                drop_reasons[reason] += 1
                config_drops[reason] += 1
                continue
            counters["structurally_eligible"] += 1
            structurally_eligible += 1
            structural_task_ids.add(analyzed.identity.task_id)
            structural_repositories.add(analyzed.identity.repository)
            compressed, compression_report, completed_stages = (
                _compress_tracked_source_row_staged(analyzed)
            )
            stage_counter_keys = {
                "pairing": "pairing_passed",
                "verification": "verification_passed",
                "grounding": "grounding_passed",
                "repeat_read_streak": "repeat_read_streak_passed",
            }
            for stage in completed_stages:
                counters[stage_counter_keys[stage]] += 1
            if compressed is None:
                reason = str(compression_report["reason"])
                drop_reasons[reason] += 1
                config_drops[reason] += 1
                continue
            counters["compressed_eligible"] += 1
            messages = [copy.deepcopy(message) for message in compressed.messages]
            candidates.append(
                {
                    "source": stream.source,
                    "config": stream.config,
                    "split": stream.split,
                    "task_id": compressed.identity.task_id,
                    "instance_id": compressed.identity.task_id,
                    "repository": compressed.identity.repository,
                    "trajectory_id": compressed.identity.trajectory_id,
                    "messages": messages,
                    "content_sha256": _sha256_bytes(_canonical_json(messages)),
                    "first_edit_command_index": int(
                        compression_report["first_edit_command_index"]
                    ),
                    "max_read_streak": _max_read_streak(messages),
                    "_config_key": key,
                }
            )
        if progress_every:
            emit_progress(
                _progress_line(
                    stream,
                    scanned=counters["scanned"],
                    elapsed=monotonic() - progress_started,
                    final=True,
                )
            )
    return _ScanResult(
        candidates=candidates,
        per_config=per_config,
        per_config_drops=per_config_drops,
        drop_reasons=drop_reasons,
        provenance=provenance,
        exclusion_manifest=exclusion_manifest,
        structurally_eligible=structurally_eligible,
        structural_task_ids=structural_task_ids,
        structural_repositories=structural_repositories,
    )


def _audit_report(scan: _ScanResult, inputs: AuditInputs) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for counters in scan.per_config.values():
        totals.update(counters)
    matches = scan.structurally_eligible == inputs.expected_pre_exclusion_eligible
    return {
        "event": "rust_v3_audit_final",
        "status": "passed" if matches else "boundary_mismatch",
        "dataset_revision": inputs.dataset_revision,
        "configs": sorted(
            scan.provenance,
            key=lambda item: (item["source"], item["config"], item["split"]),
        ),
        "exclusion": scan.exclusion_manifest,
        "input_boundary": {
            "expected_pre_exclusion_eligible": inputs.expected_pre_exclusion_eligible,
            "actual_pre_exclusion_eligible": scan.structurally_eligible,
            "matches": matches,
            "definition": "structurally eligible rows after frozen task-ID exclusion",
        },
        "totals": dict(sorted(totals.items())),
        "per_config": {
            key: {
                **dict(sorted(value.items())),
                "drop_reasons": dict(sorted(scan.per_config_drops[key].items())),
            }
            for key, value in sorted(scan.per_config.items())
        },
        "drop_reasons": dict(sorted(scan.drop_reasons.items())),
        "unique_structural_task_ids": len(scan.structural_task_ids),
        "unique_structural_repositories": len(scan.structural_repositories),
        "unique_compressed_task_ids": len(
            {candidate["task_id"] for candidate in scan.candidates}
        ),
        "unique_compressed_repositories": len(
            {candidate["repository"] for candidate in scan.candidates}
        ),
    }


def audit_streaming_dataset(
    streams: Iterable[SourceRowStream],
    *,
    inputs: AuditInputs,
    progress_every: int = 1_000,
    progress_sink: ProgressSink | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run every CPU structural/compression audit without tokenizing or publishing."""
    scan = _scan_streams(
        streams,
        inputs=inputs,
        progress_every=progress_every,
        progress_sink=progress_sink,
        monotonic=monotonic,
    )
    return _audit_report(scan, inputs)


def run_audit_cli(
    streams: Iterable[SourceRowStream],
    *,
    inputs: AuditInputs,
    emit: ProgressSink = _default_progress_sink,
    progress_every: int = 1_000,
) -> int:
    """Emit audit evidence before returning a shell status for its boundary gate."""
    report = audit_streaming_dataset(
        streams,
        inputs=inputs,
        progress_every=progress_every,
    )
    emit(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


def build_streaming_dataset(
    streams: Iterable[SourceRowStream],
    *,
    output_dir: Path,
    inputs: BuildInputs,
    count_rendered_tokens: RenderedTokenCounter,
    publish_hf: HfPublisher | None = None,
    progress_every: int = 1_000,
    progress_sink: ProgressSink | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Stream, transform, select, and transactionally publish the Task-3 corpus."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if (
        not inputs.dataset_revision
        or not inputs.tokenizer_identity
        or not inputs.tokenizer_revision
        or not inputs.template_identity
        or not inputs.template_content
    ):
        raise ValueError("dataset, tokenizer, and template identities are required")
    _validated_sha256(inputs.tokenizer_artifact_sha256, "tokenizer artifact SHA256")
    expected_template_sha = _validated_sha256(
        inputs.expected_template_sha256, "expected template SHA256"
    )
    actual_template_sha = _sha256_bytes(inputs.template_content.encode("utf-8"))
    if actual_template_sha != expected_template_sha:
        raise ValueError(
            "template SHA256 mismatch: "
            f"expected {expected_template_sha}, got {actual_template_sha}"
        )
    if not 0 < inputs.max_tokens <= 49_152:
        raise ValueError("max_tokens must be between 1 and 49152")
    if not 1 <= inputs.max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    scan = _scan_streams(
        streams,
        inputs=inputs,
        progress_every=progress_every,
        progress_sink=progress_sink,
        monotonic=monotonic,
    )
    candidates = scan.candidates
    per_config = scan.per_config
    per_config_drops = scan.per_config_drops
    drop_reasons = scan.drop_reasons
    provenance = scan.provenance
    exclusion_manifest = scan.exclusion_manifest
    pre_exclusion_eligible = scan.structurally_eligible
    if pre_exclusion_eligible != inputs.expected_pre_exclusion_eligible:
        raise ValueError(
            "pre-exclusion eligible boundary mismatch: "
            f"expected {inputs.expected_pre_exclusion_eligible}, "
            f"got {pre_exclusion_eligible}"
        )

    def count(candidate: dict[str, Any]) -> int:
        value = count_rendered_tokens(candidate["messages"])
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("rendered token counter must return a positive integer")
        return value

    with ThreadPoolExecutor(max_workers=inputs.max_workers) as executor:
        token_counts = list(executor.map(count, candidates))
    for candidate, tokens in zip(candidates, token_counts, strict=True):
        candidate["tokens"] = tokens

    by_task: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_task.setdefault(candidate["task_id"], []).append(candidate)
    task_winners: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        choices = sorted(
            by_task[task_id],
            key=lambda row: (
                row["tokens"],
                row["source"],
                row["config"],
                row["split"],
                row["content_sha256"],
                row["trajectory_id"],
            ),
        )
        task_winners.append(choices[0])
        per_config[choices[0]["_config_key"]]["task_selected"] += 1
        for rejected in choices[1:]:
            drop_reasons["task_shorter_candidate"] += 1
            per_config_drops[rejected["_config_key"]]["task_shorter_candidate"] += 1

    content_seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in sorted(
        task_winners,
        key=lambda row: (
            row["tokens"], row["source"], row["config"], row["split"],
            row["content_sha256"], row["task_id"],
        ),
    ):
        digest = candidate["content_sha256"]
        if digest in content_seen:
            drop_reasons["content_duplicate"] += 1
            per_config_drops[candidate["_config_key"]]["content_duplicate"] += 1
            continue
        content_seen.add(digest)
        unique.append(candidate)
        per_config[candidate["_config_key"]]["content_dedup_kept"] += 1

    final_internal: list[dict[str, Any]] = []
    for candidate in unique:
        if candidate["tokens"] > inputs.max_tokens:
            drop_reasons["token_budget_exceeded"] += 1
            per_config_drops[candidate["_config_key"]]["token_budget_exceeded"] += 1
            continue
        final_internal.append(candidate)
        per_config[candidate["_config_key"]]["token_budget_kept"] += 1

    final_internal.sort(key=lambda row: (row["task_id"], row["content_sha256"]))
    rows: list[dict[str, Any]] = []
    for candidate in final_internal:
        rows.append({key: value for key, value in candidate.items() if not key.startswith("_")})

    jsonl = b"".join(_canonical_json(row) + b"\n" for row in rows)
    tokens = [row["tokens"] for row in rows]
    per_repo = dict(sorted(Counter(row["repository"] for row in rows).items()))
    first_edit_hist = dict(
        sorted(Counter(str(row["first_edit_command_index"]) for row in rows).items())
    )
    read_streak_hist = dict(
        sorted(Counter(str(row["max_read_streak"]) for row in rows).items())
    )
    rows_scanned = sum(counters["scanned"] for counters in per_config.values())
    rows_dropped = sum(drop_reasons.values())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_revision": inputs.dataset_revision,
        "configs": sorted(provenance, key=lambda item: (item["source"], item["config"], item["split"])),
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "tokenizer": {
            "identity": inputs.tokenizer_identity,
            "revision": inputs.tokenizer_revision,
            "artifact_sha256": inputs.tokenizer_artifact_sha256,
        },
        "template": {
            "identity": inputs.template_identity,
            "sha256": actual_template_sha,
        },
        "exclusion": exclusion_manifest,
        "input_boundary": {
            "expected_pre_exclusion_eligible": inputs.expected_pre_exclusion_eligible,
            "actual_pre_exclusion_eligible": pre_exclusion_eligible,
            "definition": (
                "structurally eligible rows after frozen task-ID exclusion"
            ),
        },
        "max_tokens": inputs.max_tokens,
        "max_workers": inputs.max_workers,
        "rows_output": len(rows),
        "arithmetic": {
            "rows_scanned": rows_scanned,
            "rows_dropped": rows_dropped,
            "rows_output": len(rows),
            "balanced": rows_scanned == rows_dropped + len(rows),
        },
        "output_jsonl_sha256": _sha256_bytes(jsonl),
        "output_rows_sha256": _sha256_bytes(b"".join(_canonical_json(row) for row in rows)),
        "output_task_ids_sha256": canonical_id_sha256(row["task_id"] for row in rows),
        "output_content_hashes_sha256": canonical_id_sha256(row["content_sha256"] for row in rows),
        "per_config": {
            key: {**dict(sorted(value.items())), "drop_reasons": dict(sorted(per_config_drops[key].items()))}
            for key, value in sorted(per_config.items())
        },
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "per_repo": per_repo,
        "token_stats": {
            "min": min(tokens, default=0),
            "median": _percentile(tokens, 50),
            "p95": _percentile(tokens, 95),
            "max": max(tokens, default=0),
        },
        "first_edit_histogram": first_edit_hist,
        "read_streak_histogram": read_streak_hist,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    lock_fd: int | None = None
    temp_dir: Path | None = None
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output: {output_dir}")
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
        _write_durable(temp_dir / "dataset.jsonl", jsonl)
        (publish_hf or _default_hf_publisher)(rows, temp_dir / "hf_dataset")
        if not (temp_dir / "hf_dataset").exists():
            raise RuntimeError("HF publisher did not create its output directory")
        _write_durable(temp_dir / "manifest.json", _canonical_json(manifest) + b"\n")
        _atomic_rename_directory_noreplace(temp_dir, output_dir)
        temp_dir = None
        directory_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
    return manifest


def _parse_config(value: str) -> tuple[str, str]:
    config, separator, split = value.partition(":")
    if not separator or not config or not split:
        raise argparse.ArgumentTypeError("config must be CONFIG:SPLIT")
    return config, split


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dataset", default="nvidia/Open-SWE-Traces")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config", action="append", type=_parse_config, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--expected-exclusion-count", type=int, required=True)
    parser.add_argument("--expected-exclusion-sha256", required=True)
    parser.add_argument("--expected-pre-exclusion-eligible", type=int, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--template-identity")
    parser.add_argument("--template-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-tokens", type=int, default=49_152)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Report per-partition streaming progress every N rows; 0 disables.",
    )
    args = parser.parse_args(argv)
    if not args.audit_only:
        required = {
            "--tokenizer": args.tokenizer,
            "--tokenizer-revision": args.tokenizer_revision,
            "--template-identity": args.template_identity,
            "--template-sha256": args.template_sha256,
            "--out": args.out,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            parser.error("build mode requires " + ", ".join(missing))
    return args


def main() -> int:
    args = _parse_args()
    from datasets import load_dataset

    streams = [
        SourceRowStream(
            args.dataset,
            config,
            split,
            load_dataset(
                args.dataset,
                config,
                split=split,
                revision=args.revision,
                streaming=True,
            ),
        )
        for config, split in args.config
    ]
    audit_inputs = AuditInputs(
        dataset_revision=args.revision,
        exclusion_path=args.exclude,
        expected_exclusion_count=args.expected_exclusion_count,
        expected_exclusion_sha256=args.expected_exclusion_sha256,
        expected_pre_exclusion_eligible=args.expected_pre_exclusion_eligible,
    )
    if args.audit_only:
        return run_audit_cli(
            streams,
            inputs=audit_inputs,
            progress_every=args.progress_every,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    template_content = tokenizer.get_chat_template()
    actual_tokenizer_sha = tokenizer_artifact_sha256(tokenizer)

    def count_rendered(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        return rendered_token_count(rendered)

    manifest = build_streaming_dataset(
        streams,
        output_dir=args.out,
        inputs=BuildInputs(
            dataset_revision=args.revision,
            tokenizer_identity=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            tokenizer_artifact_sha256=actual_tokenizer_sha,
            template_identity=args.template_identity,
            template_content=template_content,
            expected_template_sha256=args.template_sha256,
            exclusion_path=args.exclude,
            expected_exclusion_count=args.expected_exclusion_count,
            expected_exclusion_sha256=args.expected_exclusion_sha256,
            expected_pre_exclusion_eligible=args.expected_pre_exclusion_eligible,
            max_tokens=args.max_tokens,
            max_workers=args.workers,
        ),
        count_rendered_tokens=count_rendered,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
