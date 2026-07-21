"""Pinned Fable 5 source selection and fail-closed structural validation."""

from __future__ import annotations

import base64
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, Iterable, Literal, Mapping, Sequence

from phaseD_sft.agentic_trace_filters import command_trace_quality_report

try:
    # Script-style imports put this directory first on sys.path.
    from teacher_platform import (  # type: ignore[attr-defined]
        UnsupportedTrajectoryTool,
        _normalize_trajectory_bash_command,
        _quoted_python_editor,
        _reject_trajectory_relative_escape,
        _trajectory_shell_tokens,
    )
except ImportError:
    # Module-style imports resolve ``teacher_platform`` as a namespace package.
    from teacher_platform.teacher_platform import (
        UnsupportedTrajectoryTool,
        _normalize_trajectory_bash_command,
        _quoted_python_editor,
        _reject_trajectory_relative_escape,
        _trajectory_shell_tokens,
    )


DATASET_ID: Final = "greghavens/fable-5-coding-and-debugging-traces"
DATASET_REVISION: Final = "aef8506515979988aa5c1a423f5b0fb3cee60382"
SOURCE_LFS_SHA256: Final = (
    "ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3"
)
SOURCE_BYTES: Final = 730_331_947
EXPECTED_ROWS: Final = 12_408
EXPECTED_TERMINAL_TRAJECTORIES: Final = 2_377
MOONSHINER_REVISION: Final = "436316e8f86eb136d5ce3ec95a1a6f48c1d7f940"
MINI_SWE_SYSTEM: Final = (
    "You are a practical software engineer using a shell to fix one repository "
    "bug. Prefer a small correct source edit over extended inspection. Before each "
    "command write 1-3 terse sentences of reasoning, then emit one bash tool call."
)

CanonicalLanguage = Literal["python", "rust", "cpp"]


class DropReason(str, Enum):
    """Stable reason codes for rejecting a source row without repairing it."""

    VALIDATION = "validation"
    NONTERMINAL = "nonterminal"
    LANGUAGE = "language"
    MALFORMED_ROW = "malformed_row"
    ASSISTANT_COUNT = "assistant_count"
    DUPLICATE_TASK = "duplicate_task"
    UNSUPPORTED_ROLE = "unsupported_role"
    MISSING_TOOL_CALL_ID = "missing_tool_call_id"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    ORPHAN_TOOL_RESULT = "orphan_tool_result"
    DUPLICATE_TOOL_RESULT = "duplicate_tool_result"
    TOOL_RESULT_MISMATCH = "tool_result_mismatch"
    MISSING_TOOL_RESULT = "missing_tool_result"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    NON_ASSISTANT_TERMINAL = "non_assistant_terminal"
    CATEGORY = "category_drop"
    UNSUPPORTED_TOOL = "unsupported_tool"
    CONTAMINATION = "contamination_drop"
    BEHAVIOR = "behavior_drop"
    REPLAY = "replay_drop"
    TOKEN_BUDGET = "token_budget"
    DUPLICATE = "duplicate_drop"
    OUTPUT_LIMIT = "output_limit_drop"


class RowRejected(ValueError):
    """Raised when a complete source row or trajectory must be discarded."""

    def __init__(self, reason: DropReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


@dataclass(frozen=True)
class SelectedTrajectory:
    task: str
    language: CanonicalLanguage
    category: str
    messages: tuple[dict[str, Any], ...]
    tools_json: str


@dataclass(frozen=True)
class SourceContract:
    """Immutable provenance contract for the audited source artifact."""

    dataset_id: str = field(default=DATASET_ID, init=False)
    dataset_revision: str = field(default=DATASET_REVISION, init=False)
    source_lfs_sha256: str = field(default=SOURCE_LFS_SHA256, init=False)
    source_bytes: int = field(default=SOURCE_BYTES, init=False)
    expected_rows: int = field(default=EXPECTED_ROWS, init=False)
    expected_terminal_trajectories: int = field(
        default=EXPECTED_TERMINAL_TRAJECTORIES,
        init=False,
    )

    def select_terminal_rows(
        self, rows: Iterable[dict[str, Any]]
    ) -> tuple[SelectedTrajectory, ...]:
        """Validate prefiltered terminal rows and reject a duplicate task.

        Raw-stream drop accounting belongs to the later build pipeline; this
        collection helper intentionally accepts only rows already eligible for
        terminal selection.
        """

        selected: list[SelectedTrajectory] = []
        seen_tasks: set[str] = set()
        for row in rows:
            trajectory = select_terminal_row(row)
            if trajectory.task in seen_tasks:
                raise RowRejected(
                    DropReason.DUPLICATE_TASK,
                    f"duplicate terminal row for task {trajectory.task!r}",
                )
            seen_tasks.add(trajectory.task)
            selected.append(trajectory)
        return tuple(selected)


def canonical_language(value: object) -> CanonicalLanguage | None:
    """Return the canonical language for an exact audited source alias."""

    aliases: dict[str, CanonicalLanguage] = {
        "python": "python",
        "py": "python",
        "rust": "rust",
        "cpp": "cpp",
        "c++": "cpp",
    }
    if type(value) is not str:
        return None
    return aliases.get(value)


def _reject(reason: DropReason, detail: str) -> None:
    raise RowRejected(reason, detail)


def _validate_arguments(arguments: object, *, call_id: str) -> None:
    if type(arguments) is str:
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            _reject(
                DropReason.MALFORMED_ARGUMENTS,
                f"tool call {call_id!r} contains malformed argument JSON: {exc.msg}",
            )
    if type(arguments) is not dict:
        _reject(
            DropReason.MALFORMED_ARGUMENTS,
            f"tool call {call_id!r} arguments must be a JSON object",
        )
    try:
        json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError) as exc:
        _reject(
            DropReason.MALFORMED_ARGUMENTS,
            f"tool call {call_id!r} arguments are not valid JSON: {exc}",
        )


def _assistant_calls(message: dict[str, Any], *, message_index: int) -> list[Any]:
    calls = message.get("tool_calls", [])
    if type(calls) is not list:
        _reject(
            DropReason.MALFORMED_ROW,
            f"assistant message {message_index} tool_calls must be a list",
        )
    return calls


def validate_source_messages(
    messages: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate roles and the ordered tool-call/result ledger atomically."""

    try:
        sequence = tuple(messages)
    except TypeError:
        _reject(DropReason.MALFORMED_ROW, "messages must be iterable")

    if not sequence:
        _reject(DropReason.NON_ASSISTANT_TERMINAL, "trajectory is empty")
    for index, message in enumerate(sequence):
        if type(message) is not dict:
            _reject(DropReason.MALFORMED_ROW, f"message {index} must be an object")
        role = message.get("role")
        if type(role) is not str or role not in {"system", "user", "assistant", "tool"}:
            _reject(
                DropReason.UNSUPPORTED_ROLE,
                f"message {index} has unsupported role {role!r}",
            )
        if type(message.get("content")) is not str:
            _reject(
                DropReason.MALFORMED_ROW,
                f"message {index} content must be a string",
            )

    if sequence[-1]["role"] != "assistant":
        _reject(
            DropReason.NON_ASSISTANT_TERMINAL,
            f"terminal role is {sequence[-1]['role']!r}",
        )

    pending_order: list[str] = []
    pending_ids: set[str] = set()
    called: set[str] = set()
    resolved: set[str] = set()

    for index, message in enumerate(sequence):
        role = message["role"]
        if role == "tool":
            result_id = message.get("tool_call_id")
            if type(result_id) is not str or not result_id:
                _reject(
                    DropReason.ORPHAN_TOOL_RESULT,
                    f"tool result {index} has no nonempty tool_call_id",
                )
            if result_id in resolved:
                _reject(
                    DropReason.DUPLICATE_TOOL_RESULT,
                    f"tool call {result_id!r} has multiple results",
                )
            if result_id not in called:
                _reject(
                    DropReason.ORPHAN_TOOL_RESULT,
                    f"tool result {result_id!r} has no preceding call",
                )
            if result_id not in pending_ids:
                _reject(
                    DropReason.TOOL_RESULT_MISMATCH,
                    f"tool result {result_id!r} belongs to a different call group",
                )
            pending_ids.remove(result_id)
            resolved.add(result_id)
            if not pending_ids:
                pending_order.clear()
            continue

        if pending_ids:
            missing_id = next(
                call_id for call_id in pending_order if call_id in pending_ids
            )
            _reject(
                DropReason.MISSING_TOOL_RESULT,
                f"tool call {missing_id!r} has no result before message {index}",
            )
        if role != "assistant":
            continue

        for call_index, call in enumerate(_assistant_calls(message, message_index=index)):
            if type(call) is not dict:
                _reject(
                    DropReason.MISSING_TOOL_CALL_ID,
                    f"tool call {index}:{call_index} must be an object",
                )
            call_id = call.get("id")
            if type(call_id) is not str or not call_id:
                _reject(
                    DropReason.MISSING_TOOL_CALL_ID,
                    f"tool call {index}:{call_index} has no nonempty id",
                )
            if call_id in called:
                _reject(
                    DropReason.DUPLICATE_TOOL_CALL,
                    f"duplicate tool call id {call_id!r}",
                )
            if call.get("type") != "function":
                _reject(
                    DropReason.MALFORMED_ROW,
                    f"tool call {call_id!r} type must be 'function'",
                )
            function = call.get("function")
            if (
                type(function) is not dict
                or type(function.get("name")) is not str
                or not function["name"]
            ):
                _reject(
                    DropReason.MALFORMED_ROW,
                    f"tool call {call_id!r} has no nonempty function name",
                )
            _validate_arguments(function.get("arguments"), call_id=call_id)
            called.add(call_id)
            pending_order.append(call_id)
            pending_ids.add(call_id)

    if pending_ids:
        missing_id = next(call_id for call_id in pending_order if call_id in pending_ids)
        _reject(
            DropReason.MISSING_TOOL_RESULT,
            f"tool call {missing_id!r} has no result",
        )
    return sequence


def _require_exact_field(row: dict[str, Any], key: str, field_type: type) -> Any:
    value = row.get(key)
    if type(value) is not field_type:
        _reject(
            DropReason.MALFORMED_ROW,
            f"field {key!r} must have exact type {field_type.__name__}",
        )
    return value


def select_terminal_row(row: dict[str, Any]) -> SelectedTrajectory:
    """Select one terminal training row under the pinned structural contract."""

    if type(row) is not dict:
        _reject(DropReason.MALFORMED_ROW, "row must be an object")

    task = _require_exact_field(row, "task", str)
    language_source = _require_exact_field(row, "lang", str)
    category = _require_exact_field(row, "category", str)
    split = _require_exact_field(row, "split", str)
    assistant_step = _require_exact_field(row, "assistant_step", int)
    assistant_steps = _require_exact_field(row, "assistant_steps", int)
    messages = _require_exact_field(row, "messages", list)
    tools_json = _require_exact_field(row, "tools", str)

    if not task or not category or assistant_step < 1 or assistant_steps < 1:
        _reject(DropReason.MALFORMED_ROW, "required row values must be nonempty and positive")
    if split != "train":
        _reject(DropReason.VALIDATION, f"row split is {split!r}")
    if assistant_step != assistant_steps:
        _reject(
            DropReason.NONTERMINAL,
            f"assistant step {assistant_step} of {assistant_steps} is cumulative context",
        )

    language = canonical_language(language_source)
    if language is None:
        _reject(DropReason.LANGUAGE, f"unsupported language {language_source!r}")

    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError as exc:
        _reject(DropReason.MALFORMED_ROW, f"tools contains malformed JSON: {exc.msg}")
    if type(tools) is not list:
        _reject(DropReason.MALFORMED_ROW, "tools JSON must encode a list")

    validated_messages = validate_source_messages(messages)
    assistant_count = sum(
        message["role"] == "assistant" for message in validated_messages
    )
    if assistant_count != assistant_steps:
        _reject(
            DropReason.ASSISTANT_COUNT,
            f"row declares {assistant_steps} assistant turns but contains {assistant_count}",
        )

    return SelectedTrajectory(
        task=task,
        language=language,
        category=category,
        messages=validated_messages,
        tools_json=tools_json,
    )


@dataclass(frozen=True)
class ReadOnlyBashOp:
    command: str
    description: str | None = None
    timeout: int | None = None


@dataclass(frozen=True)
class VerifierEvidenceOp:
    command: str
    description: str | None = None
    timeout: int | None = None

    @property
    def command_sha256(self) -> str:
        return hashlib.sha256(self.command.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FableReadOp:
    path: str
    offset: int
    limit: int


@dataclass(frozen=True)
class FableWriteOp:
    path: str
    content: str
    protected_paths: tuple[str, ...]


@dataclass(frozen=True)
class FableEditOp:
    path: str
    old_string: str
    new_string: str
    replace_all: bool
    protected_paths: tuple[str, ...]


@dataclass(frozen=True)
class FableGlobOp:
    pattern: str
    path: str


@dataclass(frozen=True)
class FableGrepOp:
    pattern: str
    path: str
    glob: str | None
    ignore_case: bool
    head_limit: int


FableOperation = (
    ReadOnlyBashOp
    | VerifierEvidenceOp
    | FableReadOp
    | FableWriteOp
    | FableEditOp
    | FableGlobOp
    | FableGrepOp
)


def _fable_arguments(tool_call: object) -> tuple[str, dict[str, Any]]:
    if type(tool_call) is not dict or tool_call.get("type") != "function":
        raise UnsupportedTrajectoryTool("malformed Fable tool call")
    function = tool_call.get("function")
    if type(function) is not dict or type(function.get("name")) is not str:
        raise UnsupportedTrajectoryTool("malformed Fable tool function")
    name = function["name"]
    raw_arguments = function.get("arguments")
    if type(raw_arguments) is str:
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise UnsupportedTrajectoryTool(
                "Fable tool arguments contain malformed JSON"
            ) from exc
    if type(raw_arguments) is not dict:
        raise UnsupportedTrajectoryTool("Fable tool arguments must be an object")
    if any(type(key) is not str for key in raw_arguments):
        raise UnsupportedTrajectoryTool("Fable tool argument keys must be strings")
    return name, raw_arguments


def _require_keys(
    name: str,
    arguments: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required - arguments.keys())
    if missing:
        raise UnsupportedTrajectoryTool(
            f"{name} missing required arguments: {', '.join(missing)}"
        )
    unknown = sorted(arguments.keys() - required - optional)
    if unknown:
        raise UnsupportedTrajectoryTool(
            f"unsupported {name} arguments: {', '.join(unknown)}"
        )


def _required_string(
    arguments: dict[str, Any], key: str, *, nonempty: bool = False
) -> str:
    value = arguments.get(key)
    if type(value) is not str or (nonempty and not value.strip()):
        qualifier = " nonempty" if nonempty else ""
        raise UnsupportedTrajectoryTool(f"{key} must be a{qualifier} string")
    return value


def _normalize_fable_path(value: object) -> str:
    if type(value) is not str or not value:
        raise UnsupportedTrajectoryTool("file path must be a nonempty string")
    if "\x00" in value:
        raise UnsupportedTrajectoryTool("file path contains NUL")
    if "\n" in value or "\r" in value:
        raise UnsupportedTrajectoryTool("file path contains a newline")

    absolute_testbed = value == "/testbed" or value.startswith("/testbed/")
    if value.startswith("/") and not absolute_testbed:
        raise UnsupportedTrajectoryTool("file path is an unsupported absolute path")
    relative = value.removeprefix("/testbed/") if absolute_testbed else value
    if value == "/testbed":
        relative = "."
    raw_parts = relative.split("/")
    if any(part == "" for part in raw_parts):
        raise UnsupportedTrajectoryTool("file path contains an empty path component")
    if any(part == ".." for part in raw_parts):
        raise UnsupportedTrajectoryTool("file path contains parent traversal")
    if any(part == "." for part in raw_parts) and relative != ".":
        raise UnsupportedTrajectoryTool("file path contains an ambiguous component")
    normalized = PurePosixPath(relative)
    return "/testbed" if normalized == PurePosixPath(".") else f"/testbed/{normalized}"


def _normalize_protected_paths(protected_paths: frozenset[str]) -> frozenset[str]:
    return frozenset(_normalize_fable_path(path) for path in protected_paths)


def _reject_protected(path: str, protected_paths: frozenset[str]) -> None:
    if path in _normalize_protected_paths(protected_paths):
        raise UnsupportedTrajectoryTool(f"write targets protected path {path}")


def _runtime_confinement_lines(
    path: str,
    *,
    kind: Literal["file", "directory", "existing", "mutation"],
    protected_paths: tuple[str, ...] = (),
) -> list[str]:
    lines = [
        "from pathlib import Path",
        "workspace = Path('/testbed')",
        "root = workspace.resolve()",
        f"path = Path({json.dumps(path)})",
    ]
    if kind == "mutation":
        lines.extend(
            [
                "relative = path.relative_to(workspace)",
                "cursor = workspace",
                "for component in relative.parts:",
                "    cursor = cursor / component",
                "    if cursor.is_symlink():",
                "        raise SystemExit(f'mutation path contains symlink {cursor}')",
                "candidate = path.resolve(strict=False)",
            ]
        )
    else:
        lines.append("candidate = path.resolve(strict=True)")
    lines.extend(
        [
            "if candidate != root and root not in candidate.parents:",
            "    raise SystemExit('path escapes /testbed through a symlink')",
        ]
    )
    if kind == "file":
        lines.extend(
            [
                "if not candidate.is_file():",
                "    raise SystemExit('read target is not a regular file')",
            ]
        )
    elif kind == "directory":
        lines.extend(
            [
                "if not candidate.is_dir():",
                "    raise SystemExit('search target is not a directory')",
            ]
        )
    elif kind == "existing":
        lines.extend(
            [
                "if not candidate.exists():",
                "    raise SystemExit('search target does not exist')",
            ]
        )
    if protected_paths:
        lines.extend(
            [
                f"protected_paths = {json.dumps(list(protected_paths))}",
                "for protected_path in protected_paths:",
                "    protected = Path(protected_path).resolve()",
                "    aliases_protected = (",
                "        candidate.exists()",
                "        and protected.exists()",
                "        and candidate.samefile(protected)",
                "    )",
                "    if candidate == protected or aliases_protected:",
                "        raise SystemExit(f'mutation resolves to protected path {protected}')",
            ]
        )
    if kind == "mutation":
        lines.extend(
            [
                "path = candidate",
                "import os",
                "import stat",
                "try:",
                "    pre_stat = os.stat(path, follow_symlinks=False)",
                "except FileNotFoundError:",
                "    pre_stat = None",
                "if pre_stat is not None and not stat.S_ISREG(pre_stat.st_mode):",
                "    raise SystemExit('mutation target is not a regular file')",
                "if pre_stat is not None and pre_stat.st_nlink != 1:",
                "    raise SystemExit('mutation target has a hardlink alias')",
            ]
        )
    return lines


def _guarded_path_command(
    path: str,
    command: str,
    *,
    kind: Literal["file", "directory", "existing"],
) -> str:
    guard = "\n".join(_runtime_confinement_lines(path, kind=kind))
    return (
        "set -o pipefail\n"
        "python3 - <<'FABLE_PATH_GUARD' || exit $?\n"
        f"{guard}\n"
        "FABLE_PATH_GUARD\n"
        f"{command}"
    )


def _bounded_positive_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        raise UnsupportedTrajectoryTool(
            f"{name} must be a positive integer no greater than {maximum}"
        )
    return value


def _safe_search_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise UnsupportedTrajectoryTool(f"{name} must be a nonempty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise UnsupportedTrajectoryTool(f"{name} contains an unsafe control character")
    return value


_BASH_SEPARATORS: Final = frozenset({"&&", "||", ";", "|"})
_BASH_REDIRECTIONS: Final = frozenset(
    {"<", ">", "<<", ">>", "<>", "<&", ">&", ">|"}
)
_BASH_MUTATORS: Final = frozenset(
    {"rm", "mv", "cp", "install", "touch", "tee", "patch", "perl"}
)
_BASH_EXACT_READ_ONLY_ARGV: Final = frozenset(
    {
        ("pwd",),
        ("ls",),
        ("ls", "."),
        ("ls", "-la"),
        ("ls", "-la", "."),
    }
)


def _bash_segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _BASH_SEPARATORS:
            if not segments[-1]:
                raise UnsupportedTrajectoryTool("Bash has an empty command segment")
            segments.append([])
        else:
            segments[-1].append(token)
    if not segments[-1]:
        raise UnsupportedTrajectoryTool("Bash has an empty command segment")
    return segments


def _token_has_external_absolute_path(token: str) -> bool:
    for index, character in enumerate(token):
        if character != "/":
            continue
        if index != 0 and token[index - 1] not in "=,:":
            continue
        suffix = token[index:]
        if suffix == "/testbed" or suffix.startswith("/testbed/"):
            continue
        return True
    return False


def _audit_read_only_bash(command: str) -> str:
    if "\n" in command or "\r" in command:
        if "<<" in command:
            raise UnsupportedTrajectoryTool("Bash redirection is unsupported")
        raise UnsupportedTrajectoryTool("Bash uses multiline shell construction")
    command = _normalize_trajectory_bash_command(command, None).strip()
    _reject_trajectory_relative_escape(command)
    tokens = _trajectory_shell_tokens(command)
    if not tokens:
        raise UnsupportedTrajectoryTool("Bash command is empty")
    if any(
        token in _BASH_REDIRECTIONS or (token and set(token) <= {"<", ">"})
        for token in tokens
    ):
        raise UnsupportedTrajectoryTool("Bash redirection is unsupported")
    if "&" in tokens:
        raise UnsupportedTrajectoryTool("Bash background execution is unsupported")
    if any(token in {"(", ")"} for token in tokens) or any(
        character in command for character in ("$", "`", "~")
    ):
        raise UnsupportedTrajectoryTool("Bash uses dynamic shell construction")
    if any(any(character in token for character in "*?[]{}") for token in tokens):
        raise UnsupportedTrajectoryTool("Bash uses dynamic shell construction")
    if any(_token_has_external_absolute_path(token) for token in tokens):
        raise UnsupportedTrajectoryTool("Bash path is outside /testbed")

    segments = _bash_segments(tokens)
    if len(segments) != 1:
        raise UnsupportedTrajectoryTool("Bash command composition is unsupported")
    segment = segments[0]
    executable = segment[0]
    if "=" in executable or executable.startswith("/"):
        raise UnsupportedTrajectoryTool("Bash has an unsupported executable prefix")
    if executable in _BASH_MUTATORS:
        raise UnsupportedTrajectoryTool(f"Bash uses mutating command {executable}")
    if executable == "sed" and any(
        token.startswith("--in-place") or token.startswith("-i")
        for token in segment[1:]
    ):
        raise UnsupportedTrajectoryTool("Bash uses mutating command sed")
    if executable in {"python", "python3"}:
        raise UnsupportedTrajectoryTool("Bash uses interpreter dynamic execution")
    if executable == "bash":
        raise UnsupportedTrajectoryTool("unsupported Bash executable: bash")
    argv = tuple(segment)
    if argv in _BASH_EXACT_READ_ONLY_ARGV:
        return command
    if executable == "git":
        raise UnsupportedTrajectoryTool("Bash uses mutating git command")
    raise UnsupportedTrajectoryTool("Bash command is outside the exact read-only grammar")


def _normalize_trusted_verifier_commands(
    commands: frozenset[str],
) -> frozenset[str]:
    normalized: set[str] = set()
    for command in commands:
        if type(command) is not str or not command.strip():
            raise UnsupportedTrajectoryTool(
                "trusted verifier commands must be nonempty strings"
            )
        if "\x00" in command or "\n" in command or "\r" in command:
            raise UnsupportedTrajectoryTool(
                "trusted verifier commands must be single-line and NUL-free"
            )
        try:
            command.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise UnsupportedTrajectoryTool(
                "trusted verifier commands must be valid UTF-8"
            ) from exc
        normalized.add(command)
    return frozenset(normalized)


def parse_fable_tool_call(
    tool_call: object,
    protected_paths: frozenset[str],
    *,
    trusted_verifier_commands: frozenset[str] = frozenset(),
) -> FableOperation:
    """Parse one source call once into an immutable canonical operation."""

    name, arguments = _fable_arguments(tool_call)
    canonical_protected = tuple(sorted(_normalize_protected_paths(protected_paths)))

    if name == "Bash":
        _require_keys(
            name,
            arguments,
            required=frozenset({"command"}),
            optional=frozenset({"description", "timeout"}),
        )
        command = _required_string(arguments, "command", nonempty=True)
        if "description" in arguments and type(arguments["description"]) is not str:
            raise UnsupportedTrajectoryTool("description must be a string")
        if "timeout" in arguments:
            timeout = _bounded_positive_int(
                arguments["timeout"], "timeout", maximum=600_000
            )
        else:
            timeout = None
        try:
            command.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise UnsupportedTrajectoryTool("Bash command must be valid UTF-8") from exc
        normalized_command = command
        trusted = _normalize_trusted_verifier_commands(trusted_verifier_commands)
        operation_type: type[ReadOnlyBashOp] | type[VerifierEvidenceOp]
        if normalized_command in trusted:
            operation_type = VerifierEvidenceOp
        else:
            try:
                normalized_command = _audit_read_only_bash(command)
            except UnsupportedTrajectoryTool as exc:
                raise UnsupportedTrajectoryTool(
                    f"ambiguous_bash_mutation: {exc}"
                ) from exc
            operation_type = ReadOnlyBashOp
        return operation_type(
            command=normalized_command,
            description=arguments.get("description"),
            timeout=timeout,
        )

    if name == "Read":
        _require_keys(
            name,
            arguments,
            required=frozenset({"file_path"}),
            optional=frozenset({"offset", "limit"}),
        )
        path = _normalize_fable_path(arguments["file_path"])
        offset = arguments.get("offset", 1)
        limit = arguments.get("limit", 2_000)
        offset = _bounded_positive_int(offset, "offset", maximum=10_000_000)
        limit = _bounded_positive_int(limit, "limit", maximum=2_000)
        return FableReadOp(path=path, offset=offset, limit=limit)

    if name == "Write":
        _require_keys(
            name,
            arguments,
            required=frozenset({"file_path", "content"}),
        )
        path = _normalize_fable_path(arguments["file_path"])
        _reject_protected(path, protected_paths)
        return FableWriteOp(
            path=path,
            content=_required_string(arguments, "content"),
            protected_paths=canonical_protected,
        )

    if name == "Edit":
        _require_keys(
            name,
            arguments,
            required=frozenset({"file_path", "old_string", "new_string"}),
            optional=frozenset({"replace_all"}),
        )
        path = _normalize_fable_path(arguments["file_path"])
        _reject_protected(path, protected_paths)
        old_string = _required_string(arguments, "old_string", nonempty=True)
        new_string = _required_string(arguments, "new_string")
        replace_all = arguments.get("replace_all", False)
        if type(replace_all) is not bool:
            raise UnsupportedTrajectoryTool("replace_all must be a boolean")
        return FableEditOp(
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            protected_paths=canonical_protected,
        )

    if name == "Glob":
        _require_keys(
            name,
            arguments,
            required=frozenset({"pattern"}),
            optional=frozenset({"path"}),
        )
        pattern = _safe_search_string(arguments["pattern"], "pattern")
        path = _normalize_fable_path(arguments.get("path", "."))
        return FableGlobOp(pattern=pattern, path=path)

    if name == "Grep":
        _require_keys(
            name,
            arguments,
            required=frozenset({"pattern"}),
            optional=frozenset(
                {"path", "glob", "output_mode", "head_limit", "-i", "-n"}
            ),
        )
        pattern = _safe_search_string(arguments["pattern"], "pattern")
        path = _normalize_fable_path(arguments.get("path", "."))
        if arguments.get("output_mode", "content") != "content":
            raise UnsupportedTrajectoryTool("unsupported Grep output_mode")
        limit = _bounded_positive_int(
            arguments.get("head_limit", 200), "head_limit", maximum=2_000
        )
        for flag in ("-i", "-n"):
            if flag in arguments and type(arguments[flag]) is not bool:
                raise UnsupportedTrajectoryTool(f"Grep {flag} must be a boolean")
        if arguments.get("-n") is False:
            raise UnsupportedTrajectoryTool("Grep -n cannot be disabled")
        glob = (
            _safe_search_string(arguments["glob"], "glob")
            if "glob" in arguments
            else None
        )
        return FableGrepOp(
            pattern=pattern,
            path=path,
            glob=glob,
            ignore_case=bool(arguments.get("-i", False)),
            head_limit=limit,
        )

    raise UnsupportedTrajectoryTool(f"unsupported Fable tool: {name}")


def lower_fable_operation(operation: FableOperation) -> str:
    """Lower a validated immutable operation into one native bash command."""

    if isinstance(operation, (ReadOnlyBashOp, VerifierEvidenceOp)):
        return operation.command
    if isinstance(operation, FableReadOp):
        end = operation.offset + operation.limit - 1
        quoted_path = shlex.quote(operation.path)
        command = f"sed -n {shlex.quote(f'{operation.offset},{end}p')} {quoted_path}"
        return _guarded_path_command(operation.path, command, kind="file")
    if isinstance(operation, FableWriteOp):
        payload = base64.b64encode(operation.content.encode("utf-8")).decode("ascii")
        return _quoted_python_editor(
            _runtime_confinement_lines(
                operation.path,
                kind="mutation",
                protected_paths=operation.protected_paths,
            )
            + [
                "import base64",
                f"content = base64.b64decode({payload!r})",
                "flags = os.O_WRONLY | os.O_NOFOLLOW",
                "if pre_stat is None:",
                "    flags |= os.O_CREAT | os.O_EXCL",
                "fd = os.open(path, flags, 0o666)",
                "try:",
                "    opened = os.fstat(fd)",
                "    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:",
                "        raise SystemExit('opened mutation target is not a unique regular file')",
                "    if pre_stat is not None and (",
                "        opened.st_dev != pre_stat.st_dev",
                "        or opened.st_ino != pre_stat.st_ino",
                "        or opened.st_nlink != pre_stat.st_nlink",
                "    ):",
                "        raise SystemExit('mutation target identity changed before write')",
                "    os.ftruncate(fd, 0)",
                "    handle = os.fdopen(fd, 'wb')",
                "    fd = None",
                "    with handle:",
                "        handle.write(content)",
                "finally:",
                "    if fd is not None:",
                "        os.close(fd)",
            ]
        )
    if isinstance(operation, FableEditOp):
        old_payload = base64.b64encode(operation.old_string.encode("utf-8")).decode(
            "ascii"
        )
        new_payload = base64.b64encode(operation.new_string.encode("utf-8")).decode(
            "ascii"
        )
        lines = _runtime_confinement_lines(
            operation.path,
            kind="mutation",
            protected_paths=operation.protected_paths,
        ) + [
            "import base64",
            f"old_bytes = base64.b64decode({old_payload!r})",
            f"new_bytes = base64.b64decode({new_payload!r})",
            "if pre_stat is None:",
            "    raise SystemExit('edit target does not exist')",
            "fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)",
            "try:",
            "    opened = os.fstat(fd)",
            "    if (",
            "        not stat.S_ISREG(opened.st_mode)",
            "        or opened.st_nlink != 1",
            "        or opened.st_dev != pre_stat.st_dev",
            "        or opened.st_ino != pre_stat.st_ino",
            "        or opened.st_nlink != pre_stat.st_nlink",
            "    ):",
            "        raise SystemExit('edit target identity changed before read')",
            "    handle = os.fdopen(fd, 'r+b')",
            "    fd = None",
            "    with handle:",
            "        data = handle.read()",
            "        matches = data.count(old_bytes)",
        ]
        if operation.replace_all:
            lines.extend(
                [
                    "        if matches == 0:",
                    "            raise SystemExit('expected at least one old_string match, found 0')",
                ]
            )
        else:
            lines.extend(
                [
                    "        if matches != 1:",
                    "            raise SystemExit(f'expected exactly one old_string match, found {matches}')",
                ]
            )
        lines.extend(
            [
                "        handle.seek(0)",
                "        handle.truncate(0)",
                "        handle.write(data.replace(old_bytes, new_bytes))",
                "finally:",
                "    if fd is not None:",
                "        os.close(fd)",
            ]
        )
        return _quoted_python_editor(lines)
    if isinstance(operation, FableGlobOp):
        command = (
            f"rg --files --glob {shlex.quote(operation.pattern)} -- "
            f"{shlex.quote(operation.path)} | head -n 200"
        )
        return _guarded_path_command(operation.path, command, kind="directory")
    if isinstance(operation, FableGrepOp):
        flags = ["-n"]
        if operation.ignore_case:
            flags.append("-i")
        if operation.glob is not None:
            flags.extend(["--glob", shlex.quote(operation.glob)])
        command = (
            f"rg {' '.join(flags)} -- {shlex.quote(operation.pattern)} "
            f"{shlex.quote(operation.path)} | head -n {operation.head_limit}"
        )
        return _guarded_path_command(operation.path, command, kind="existing")
    raise TypeError(f"unsupported canonical Fable operation: {type(operation).__name__}")


def translate_fable_tool_call(
    tool_call: object,
    protected_paths: frozenset[str],
    *,
    trusted_verifier_commands: frozenset[str] = frozenset(),
) -> str:
    """Parse once, then lower one exact Fable call to native bash."""

    return lower_fable_operation(
        parse_fable_tool_call(
            tool_call,
            protected_paths,
            trusted_verifier_commands=trusted_verifier_commands,
        )
    )


def _native_message(
    role: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    loss: bool | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": role,
        "content": content,
        "tool_calls": tool_calls or [],
    }
    if loss is not None:
        message["loss"] = loss
    return message


def convert_trajectory(
    selected: SelectedTrajectory,
    protected_paths: frozenset[str],
    *,
    trusted_verifier_commands: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Convert one validated terminal Fable trajectory to native mini-SWE SFT."""

    messages = validate_source_messages(selected.messages)
    problem_turns = [
        message
        for message in messages
        if message["role"] == "user"
        and not message["content"].lstrip().startswith("OBSERVATION:")
    ]
    if len(problem_turns) != 1:
        raise UnsupportedTrajectoryTool(
            "trajectory must contain exactly one non-observation user problem turn"
        )
    problem_text = problem_turns[0]["content"]
    results_by_id = {
        message["tool_call_id"]: message["content"]
        for message in messages
        if message["role"] == "tool"
    }

    converted = [
        _native_message("system", MINI_SWE_SYSTEM),
        _native_message(
            "user", f"<pr_description>\n{problem_text}\n</pr_description>"
        ),
    ]
    next_call_id = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        calls = _assistant_calls(message, message_index=-1)
        if not calls:
            converted.append(
                _native_message("assistant", message["content"], loss=True)
            )
            continue
        for call_offset, source_call in enumerate(calls):
            source_call_id = source_call["id"]
            command = translate_fable_tool_call(
                source_call,
                protected_paths,
                trusted_verifier_commands=trusted_verifier_commands,
            )
            call_id = f"fable-tool-{next_call_id}"
            next_call_id += 1
            native_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps(
                        {"command": command}, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
            converted.append(
                _native_message(
                    "assistant",
                    message["content"] if call_offset == 0 else "",
                    tool_calls=[native_call],
                    loss=True,
                )
            )
            converted.append(
                _native_message(
                    "user", f"OBSERVATION:\n{results_by_id[source_call_id]}"
                )
            )

    if converted[-1]["role"] != "assistant":
        raise UnsupportedTrajectoryTool(
            "converted trajectory must end on the real terminal assistant"
        )
    return {
        "instance_id": selected.task,
        "source_instance_id": selected.task,
        "repo": f"moonshiner/{selected.task}",
        "source": f"teacher:fable5:{DATASET_REVISION}:{selected.task}",
        "messages": converted,
    }


@dataclass(frozen=True)
class SourceMetadata:
    dataset_id: str
    dataset_revision: str
    source_lfs_sha256: str
    source_bytes: int


@dataclass(frozen=True)
class SeedContract:
    task: str
    protected_paths: tuple[str, ...]
    verify_cmd: str
    fixture_sha256: str


@dataclass(frozen=True)
class ExclusionRecord:
    instance_ids: frozenset[str] = frozenset()
    content_sha256s: frozenset[str] = frozenset()
    texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayCandidate:
    trajectory_id: str
    source_terminal_sha256: str
    content_sha256: str
    seed: SeedContract


@dataclass(frozen=True)
class ReplayEvidence:
    trajectory_id: str
    source_terminal_sha256: str
    candidate_content_sha256: str
    fixture_sha256: str
    resolved: bool
    control: bool = False
    namespace: str = "candidate"


@dataclass(frozen=True)
class BuildConfig:
    out: Path
    source_metadata: SourceMetadata
    token_counter: Callable[[list[dict[str, Any]]], int]
    seed_contract: Callable[[str], SeedContract]
    exclusions: tuple[ExclusionRecord, ...] = ()
    replay_lookup: Callable[[ReplayCandidate], ReplayEvidence | None] | None = None
    skip_replay: bool = False
    max_output: int = 0
    max_tokens: int = 49_152
    workers: int = 8
    metadata_only: bool = False
    source_contract: Any = field(default_factory=SourceContract)
    progress: Callable[[str], None] | None = None


@dataclass(frozen=True)
class BuildResult:
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    rejected: tuple[dict[str, str], ...]


@dataclass
class _Candidate:
    selected: SelectedTrajectory
    trajectory_id: str
    source_terminal_sha256: str
    seed: SeedContract
    converted: dict[str, Any]
    content_sha256: str
    first_edit_index: int
    max_read_streak: int
    normalized_command_count: int
    canonical_terminal_sha256: str
    token_count: int = 0

    @property
    def representative_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.first_edit_index,
            self.max_read_streak,
            self.normalized_command_count,
            self.token_count,
            self.canonical_terminal_sha256,
        )


_SCHEMA_VERSION: Final = "fable5-agentic-pilot-v1"
_REPRESENTATIVE_VERSION: Final = 1
_CATEGORY_REJECT_RE: Final = re.compile(
    r"(?:^|[-_ ])(?:seed[-_ ]?authoring|non[-_ ]?code|research|scheduling|"
    r"web|memory|instruction[-_ ]?following)(?:$|[-_ ])",
    re.IGNORECASE,
)
_WORD_RE: Final = re.compile(r"[\w]+", re.UNICODE)
_VERIFY_PATTERNS: Final[dict[CanonicalLanguage, re.Pattern[str]]] = {
    "python": re.compile(
        r"(?:(?:env PYTHONDONTWRITEBYTECODE=1 )?(?:python|python3)"
        r"(?=[^\n;&|]*(?:\btest_[A-Za-z0-9_.-]+\.py\b|-m (?:pytest|unittest)\b))"
        r"[^\n;&|]+|pytest(?: [^\n;&|]+)*|bash run_verify\.sh)"
    ),
    "rust": re.compile(
        r"(?:cargo (?:test|check|build)(?: [^\n;&|]+)*|bash run_verify\.sh)"
    ),
    "cpp": re.compile(
        r"(?:(?:make test|cmake [^\n;&|]+|ctest(?: [^\n;&|]+)*|"
        r"ninja [^\n;&|]+)(?: [^\n;&|]+)*)"
    ),
}
_READ_OPERATION_TYPES: Final = (ReadOnlyBashOp, FableReadOp, FableGlobOp, FableGrepOp)
_EDIT_OPERATION_TYPES: Final = (FableWriteOp, FableEditOp)
_SOURCE_SUFFIXES: Final[dict[CanonicalLanguage, frozenset[str]]] = {
    "python": frozenset({".py"}),
    "rust": frozenset({".rs"}),
    "cpp": frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}),
}
_REJECTED_MUTATION_COMPONENT_RE: Final = re.compile(
    r"^(?:tests?|fixtures?|benchmarks?|generated|vendor|third[_-]?party)$",
    re.IGNORECASE,
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_metadata(metadata: SourceMetadata, contract: Any) -> None:
    expected = {
        "dataset_id": contract.dataset_id,
        "dataset_revision": contract.dataset_revision,
        "source_lfs_sha256": contract.source_lfs_sha256,
        "source_bytes": contract.source_bytes,
    }
    for key, expected_value in expected.items():
        actual = getattr(metadata, key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"source metadata {key} mismatch: expected {expected_value!r}, got {actual!r}"
            )


def validate_moonshiner_revision(actual_revision: str) -> None:
    if actual_revision != MOONSHINER_REVISION:
        raise ValueError(
            "Moonshiner revision mismatch: expected "
            f"{MOONSHINER_REVISION}, got {actual_revision!r}"
        )


def token_gate(
    messages: list[dict[str, Any]],
    token_counter: Callable[[list[dict[str, Any]]], int],
    *,
    max_tokens: int = 49_152,
) -> int:
    count = token_counter(messages)
    if type(count) is not int or count < 0:
        raise ValueError("token counter must return a nonnegative exact integer")
    if count > max_tokens:
        raise RowRejected(
            DropReason.TOKEN_BUDGET,
            f"rendered token count {count} exceeds inclusive limit {max_tokens}",
        )
    return count


def _normalized_words(text: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in _WORD_RE.findall(text))


def normalized_word_13gram_overlap(candidate: str, exclusion: str) -> float:
    """Return symmetric normalized 13-word-gram containment overlap."""

    left = _normalized_words(candidate)
    right = _normalized_words(exclusion)
    if not left or not right:
        return 0.0
    if len(left) < 13 or len(right) < 13:
        return 1.0 if left == right else 0.0
    left_grams = {left[index : index + 13] for index in range(len(left) - 12)}
    right_grams = {right[index : index + 13] for index in range(len(right) - 12)}
    denominator = min(len(left_grams), len(right_grams))
    return len(left_grams & right_grams) / denominator if denominator else 0.0


def _problem_text(selected: SelectedTrajectory) -> str:
    problems = [
        message["content"]
        for message in selected.messages
        if message["role"] == "user"
        and not message["content"].lstrip().startswith("OBSERVATION:")
    ]
    if len(problems) != 1:
        raise UnsupportedTrajectoryTool(
            "trajectory must contain exactly one non-observation user problem turn"
        )
    return problems[0]


def _bounded_detail(detail: object) -> str:
    text = " ".join(str(detail).replace("\x00", "").split())
    return text[:240]


def _source_task(row: object) -> str:
    if type(row) is dict and type(row.get("task")) is str:
        return row["task"]
    return "<unknown>"


def _record_rejection(
    rejected: list[dict[str, str]], task: str, reason: str, detail: object
) -> None:
    rejected.append(
        {"task": task[:200], "reason": reason, "detail": _bounded_detail(detail)}
    )


def _raw_terminal(row: object) -> bool:
    return (
        type(row) is dict
        and type(row.get("assistant_step")) is int
        and type(row.get("assistant_steps")) is int
        and row["assistant_step"] == row["assistant_steps"]
    )


def _category_allowed(category: str) -> bool:
    return not _CATEGORY_REJECT_RE.search(category)


def _trajectory_identity(row: dict[str, Any], task: str) -> tuple[str, str, bytes]:
    terminal_bytes = _canonical_json_bytes(row)
    terminal_sha = _sha256_bytes(terminal_bytes)
    digest = hashlib.sha256()
    digest.update(DATASET_REVISION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(task.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(terminal_bytes)
    return digest.hexdigest(), terminal_sha, terminal_bytes


def _source_operations(
    selected: SelectedTrajectory,
    seed: SeedContract,
    tool_names: Counter[str],
    argument_keys: Counter[str],
) -> list[FableOperation]:
    operations: list[FableOperation] = []
    for message in selected.messages:
        if message["role"] != "assistant":
            continue
        for call in _assistant_calls(message, message_index=-1):
            try:
                name, arguments = _fable_arguments(call)
            except UnsupportedTrajectoryTool:
                tool_names["<malformed>"] += 1
                raise
            tool_names[name] += 1
            argument_keys.update(arguments.keys())
            operations.append(
                parse_fable_tool_call(
                    call,
                    frozenset(seed.protected_paths),
                    trusted_verifier_commands=frozenset({seed.verify_cmd}),
                )
            )
    return operations


def _operation_behavior(
    operations: list[FableOperation],
    language: CanonicalLanguage,
    verify_cmd: str,
) -> tuple[int, int, int, tuple[str, ...]]:
    reasons: list[str] = []
    first_edit_index = next(
        (
            index
            for index, operation in enumerate(operations, start=1)
            if isinstance(operation, _EDIT_OPERATION_TYPES)
        ),
        None,
    )
    if first_edit_index is None:
        reasons.append("no_edit")
        first_edit_for_result = 0
    else:
        first_edit_for_result = first_edit_index
        if first_edit_index > 10:
            reasons.append("first_edit_after_limit")

    read_streak = 0
    max_read_streak = 0
    previous: str | None = None
    repeated = False
    for operation in operations:
        if isinstance(operation, _READ_OPERATION_TYPES):
            read_streak += 1
            max_read_streak = max(max_read_streak, read_streak)
        else:
            read_streak = 0
        lowered = lower_fable_operation(operation).strip()
        if previous is not None and lowered == previous:
            repeated = True
        previous = lowered
    if max_read_streak > 5:
        reasons.append("read_streak_exceeded")
    if repeated:
        reasons.append("identical_consecutive_command")

    mutation_paths = [
        operation.path
        for operation in operations
        if isinstance(operation, _EDIT_OPERATION_TYPES)
    ]
    has_source_edit = False
    has_rejected_mutation_path = False
    for mutation_path in mutation_paths:
        path = PurePosixPath(mutation_path)
        relative_parts = path.parts[2:] if path.parts[:2] == ("/", "testbed") else path.parts
        if any(_REJECTED_MUTATION_COMPONENT_RE.fullmatch(part) for part in relative_parts[:-1]):
            has_rejected_mutation_path = True
        if path.suffix.casefold() in _SOURCE_SUFFIXES[language]:
            has_source_edit = True
    if not has_source_edit:
        reasons.append("no_source_edit")
    if has_rejected_mutation_path:
        reasons.append("rejected_mutation_path")

    verify_pattern = _VERIFY_PATTERNS[language]
    has_exact_verify = any(
        index > first_edit_for_result
        and isinstance(operation, VerifierEvidenceOp)
        and operation.command == verify_cmd
        and verify_pattern.fullmatch(operation.command)
        for index, operation in enumerate(operations, start=1)
    )
    if not has_exact_verify:
        reasons.append("no_exact_post_edit_verifier")
    return first_edit_for_result, max_read_streak, len(operations), tuple(reasons)


def _excluded(
    selected: SelectedTrajectory,
    content_sha256: str,
    exclusions: tuple[ExclusionRecord, ...],
) -> str | None:
    prompt = _problem_text(selected)
    for record in exclusions:
        if selected.task in record.instance_ids:
            return "exact_instance_id"
        if content_sha256 in record.content_sha256s:
            return "exact_content_sha256"
        for text in record.texts:
            if normalized_word_13gram_overlap(prompt, text) >= 0.8:
                return "word_13gram_overlap"
    return None


def _replay_matches(candidate: _Candidate, evidence: ReplayEvidence | None) -> bool:
    return bool(
        evidence is not None
        and evidence.resolved
        and not evidence.control
        and evidence.namespace == "candidate"
        and not evidence.namespace.startswith("controls")
        and evidence.trajectory_id == candidate.trajectory_id
        and evidence.source_terminal_sha256 == candidate.source_terminal_sha256
        and evidence.candidate_content_sha256 == candidate.content_sha256
        and evidence.fixture_sha256 == candidate.seed.fixture_sha256
    )


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(values).items(), key=lambda item: item[0])
    }


def _write_fsynced(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(
    out: Path,
    rows: list[dict[str, Any]],
    sidecar_records: Iterable[str],
    rejected: list[dict[str, str]],
    replay_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite build artifact: {out}")
    stage = out.parent / f".{out.name}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(mode=0o700)
    try:
        train_bytes = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
        rejected_bytes = b"".join(
            _canonical_json_bytes(row) + b"\n"
            for row in sorted(
                rejected,
                key=lambda value: (value["task"], value["reason"], value["detail"]),
            )
        )
        replay_bytes = b"".join(
            _canonical_json_bytes(row) + b"\n" for row in replay_records
        )
        sidecar_bytes = "".join(sidecar_records).encode("utf-8")
        if not manifest.get("metadata_only"):
            _write_fsynced(stage / "train.jsonl", train_bytes)
            _write_fsynced(
                stage / "original_terminal_rows.jsonl", sidecar_bytes, mode=0o600
            )
            _write_fsynced(stage / "rejected.jsonl", rejected_bytes)
            _write_fsynced(stage / "replay.jsonl", replay_bytes)
            manifest.update(
                {
                    "output_sha256": _sha256_bytes(train_bytes),
                    "sidecar_sha256": _sha256_bytes(sidecar_bytes),
                    "rejected_sha256": _sha256_bytes(rejected_bytes),
                    "replay_sha256": _sha256_bytes(replay_bytes),
                }
            )
            expected_hashes = {
                "train.jsonl": manifest["output_sha256"],
                "original_terminal_rows.jsonl": manifest["sidecar_sha256"],
                "rejected.jsonl": manifest["rejected_sha256"],
                "replay.jsonl": manifest["replay_sha256"],
            }
            for filename, expected_hash in expected_hashes.items():
                if _sha256_file(stage / filename) != expected_hash:
                    raise RuntimeError(f"staged artifact hash mismatch: {filename}")
            os.chmod(stage / "original_terminal_rows.jsonl", 0o600)
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"
        _write_fsynced(stage / "manifest.json", manifest_bytes)
        _fsync_directory(stage)
        os.rename(stage, out)
        _fsync_directory(out.parent)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "nonterminal": 0,
            "validation": 0,
            "language_drop": 0,
            "category_drop": 0,
            "structure_drop": 0,
            "unsupported_tool": 0,
            "contamination_drop": 0,
            "duplicate_drop": 0,
            "behavior_drop": 0,
            "replay_drop": 0,
            "token_drop": 0,
            "output_limit_drop": 0,
            "metadata_eligible": 0,
            "output": 0,
        }
    )


def build_fable5_pilot(
    rows: Iterable[dict[str, Any]], config: BuildConfig
) -> BuildResult:
    """Stream, gate, and atomically publish one deterministic Fable pilot."""

    _validate_source_metadata(config.source_metadata, config.source_contract)
    if type(config.max_output) is not int or config.max_output < 0:
        raise ValueError("max_output must be a nonnegative integer")
    if type(config.max_tokens) is not int or config.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if type(config.workers) is not int or config.workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not config.skip_replay and config.replay_lookup is None and not config.metadata_only:
        raise ValueError("replay evidence is required unless --skip-replay is set")

    counts = _empty_counts()
    rejected: list[dict[str, str]] = []
    candidates: list[_Candidate] = []
    replay_records: list[dict[str, Any]] = []
    tool_names: Counter[str] = Counter()
    argument_keys: Counter[str] = Counter()
    streamed = 0
    terminal_seen = 0
    start = time.monotonic()
    out = Path(config.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite build artifact: {out}")

    work_root = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.work.", dir=str(out.parent.resolve()))
    ) if out.parent.exists() else None
    if work_root is None:
        out.parent.mkdir(parents=True, exist_ok=True)
        work_root = Path(
            tempfile.mkdtemp(prefix=f".{out.name}.work.", dir=str(out.parent.resolve()))
        )
    database = sqlite3.connect(work_root / "sidecar.sqlite3")
    database.execute(
        "CREATE TABLE sidecar (language TEXT, task TEXT, trajectory_id TEXT, record TEXT)"
    )
    try:
        for raw_row in rows:
            streamed += 1
            is_terminal = _raw_terminal(raw_row)
            if is_terminal:
                terminal_seen += 1
                if terminal_seen % 100 == 0:
                    elapsed = max(time.monotonic() - start, 1e-9)
                    rate = terminal_seen / elapsed
                    remaining = max(
                        int(config.source_contract.expected_terminal_trajectories)
                        - terminal_seen,
                        0,
                    )
                    message = (
                        f"[fable5-import] terminal={terminal_seen} elapsed={elapsed:.1f}s "
                        f"rate={rate:.2f}/s eta={remaining / rate:.1f}s"
                    )
                    (config.progress or print)(message)
            task = _source_task(raw_row)
            if type(raw_row) is not dict:
                counts["structure_drop"] += 1
                _record_rejection(rejected, task, "structure_drop", "row is not an object")
                continue
            split = raw_row.get("split")
            if type(split) is not str:
                counts["structure_drop"] += 1
                _record_rejection(rejected, task, "structure_drop", "split has invalid type")
                continue
            assistant_step = raw_row.get("assistant_step")
            assistant_steps = raw_row.get("assistant_steps")
            if (
                type(assistant_step) is not int
                or type(assistant_steps) is not int
                or assistant_step < 1
                or assistant_steps < 1
            ):
                counts["structure_drop"] += 1
                _record_rejection(
                    rejected, task, "structure_drop", "assistant step has invalid type"
                )
                continue
            if split != "train":
                counts["validation"] += 1
                _record_rejection(rejected, task, "validation", f"split={split!r}")
                continue
            if not is_terminal:
                counts["nonterminal"] += 1
                _record_rejection(rejected, task, "nonterminal", "cumulative prefix")
                continue
            language = canonical_language(raw_row.get("lang"))
            if language is None:
                counts["language_drop"] += 1
                _record_rejection(rejected, task, "language_drop", "unsupported language")
                continue
            category = raw_row.get("category")
            if type(category) is not str or not category:
                counts["structure_drop"] += 1
                _record_rejection(rejected, task, "structure_drop", "category has invalid type")
                continue
            if not _category_allowed(category):
                counts["category_drop"] += 1
                _record_rejection(rejected, task, "category_drop", "non-code category")
                continue
            try:
                selected = select_terminal_row(raw_row)
                trajectory_id, terminal_sha, terminal_bytes = _trajectory_identity(
                    raw_row, selected.task
                )
            except RowRejected as exc:
                counts["structure_drop"] += 1
                _record_rejection(rejected, task, "structure_drop", exc.reason.value)
                continue
            sidecar_record = {
                "trajectory_id": trajectory_id,
                "source_instance_id": selected.task,
                "source_terminal_sha256": terminal_sha,
                "row": json.loads(terminal_bytes),
            }
            database.execute(
                "INSERT INTO sidecar VALUES (?, ?, ?, ?)",
                (
                    selected.language,
                    selected.task,
                    trajectory_id,
                    _canonical_json_bytes(sidecar_record).decode("utf-8") + "\n",
                ),
            )
            if config.metadata_only:
                counts["metadata_eligible"] += 1
                continue
            if any(
                selected.task in record.instance_ids for record in config.exclusions
            ):
                counts["contamination_drop"] += 1
                _record_rejection(
                    rejected,
                    selected.task,
                    "contamination_drop",
                    "exact_instance_id",
                )
                continue
            try:
                seed = config.seed_contract(selected.task)
                if seed.task != selected.task:
                    raise ValueError("seed contract task mismatch")
                if not re.fullmatch(r"[0-9a-f]{64}", seed.fixture_sha256):
                    raise ValueError("seed fixture hash is invalid")
                operations = _source_operations(
                    selected, seed, tool_names, argument_keys
                )
                converted = convert_trajectory(
                    selected,
                    frozenset(seed.protected_paths),
                    trusted_verifier_commands=frozenset({seed.verify_cmd}),
                )
            except (RowRejected, UnsupportedTrajectoryTool, ValueError) as exc:
                counts["unsupported_tool"] += 1
                _record_rejection(
                    rejected, selected.task, "unsupported_tool", type(exc).__name__
                )
                continue
            converted["instance_id"] = trajectory_id
            converted["source_instance_id"] = selected.task
            converted["source"] = (
                f"teacher:fable5:{DATASET_REVISION}:{trajectory_id}"
            )
            converted["trajectory_id"] = trajectory_id
            converted["replay_pending"] = config.skip_replay
            content_sha = _sha256_bytes(_canonical_json_bytes(converted["messages"]))
            contamination = _excluded(selected, content_sha, config.exclusions)
            if contamination is not None:
                counts["contamination_drop"] += 1
                _record_rejection(
                    rejected, selected.task, "contamination_drop", contamination
                )
                continue
            quality = command_trace_quality_report(
                converted["messages"],
                max_first_edit_index=10,
                max_read_streak=5,
                reject_identical_consecutive_commands=True,
            )
            first_edit, max_streak, command_count, operation_reasons = _operation_behavior(
                operations, selected.language, seed.verify_cmd
            )
            reasons = tuple(dict.fromkeys((*quality.reasons, *operation_reasons)))
            if reasons:
                counts["behavior_drop"] += 1
                _record_rejection(
                    rejected, selected.task, "behavior_drop", ",".join(reasons)
                )
                continue
            candidates.append(
                _Candidate(
                    selected=selected,
                    trajectory_id=trajectory_id,
                    source_terminal_sha256=terminal_sha,
                    seed=seed,
                    converted=converted,
                    content_sha256=content_sha,
                    first_edit_index=first_edit,
                    max_read_streak=max_streak,
                    normalized_command_count=command_count,
                    canonical_terminal_sha256=terminal_sha,
                )
            )

        database.commit()
        if streamed != config.source_contract.expected_rows:
            raise ValueError(
                f"raw row count mismatch: expected {config.source_contract.expected_rows}, got {streamed}"
            )
        if terminal_seen != config.source_contract.expected_terminal_trajectories:
            raise ValueError(
                "terminal trajectory count mismatch: expected "
                f"{config.source_contract.expected_terminal_trajectories}, got {terminal_seen}"
            )

        if not config.metadata_only and candidates:
            message_lists = [candidate.converted["messages"] for candidate in candidates]
            with ThreadPoolExecutor(max_workers=config.workers) as executor:
                token_counts = list(executor.map(config.token_counter, message_lists))
            budget_candidates: list[_Candidate] = []
            for candidate, token_count in zip(candidates, token_counts, strict=True):
                if type(token_count) is not int or token_count < 0:
                    raise ValueError(
                        "token counter must return a nonnegative exact integer"
                    )
                if token_count > config.max_tokens:
                    counts["token_drop"] += 1
                    _record_rejection(
                        rejected,
                        candidate.selected.task,
                        "token_drop",
                        f"tokens={token_count}",
                    )
                    continue
                candidate.token_count = token_count
                budget_candidates.append(candidate)
            candidates = budget_candidates

        by_task: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            by_task[candidate.selected.task].append(candidate)
        representatives: list[_Candidate] = []
        representative_manifest: list[dict[str, Any]] = []
        for task in sorted(by_task):
            task_candidates = sorted(
                by_task[task], key=lambda candidate: candidate.representative_key
            )
            chosen = task_candidates[0]
            representatives.append(chosen)
            representative_manifest.append(
                {
                    "task": task,
                    "trajectory_id": chosen.trajectory_id,
                    "first_edit_index": chosen.first_edit_index,
                    "max_read_streak": chosen.max_read_streak,
                    "normalized_command_count": chosen.normalized_command_count,
                    "rendered_token_count": chosen.token_count,
                    "canonical_terminal_sha256": chosen.canonical_terminal_sha256,
                }
            )
            for duplicate in task_candidates[1:]:
                counts["duplicate_drop"] += 1
                _record_rejection(
                    rejected,
                    duplicate.selected.task,
                    "duplicate_drop",
                    "nonrepresentative terminal trajectory",
                )

        content_seen: set[str] = set()
        unique_candidates: list[_Candidate] = []
        for candidate in sorted(
            representatives,
            key=lambda value: (value.selected.language, value.selected.task),
        ):
            if candidate.content_sha256 in content_seen:
                counts["duplicate_drop"] += 1
                _record_rejection(
                    rejected,
                    candidate.selected.task,
                    "duplicate_drop",
                    "duplicate canonical message content",
                )
                continue
            content_seen.add(candidate.content_sha256)
            unique_candidates.append(candidate)
        eligible_unique_task_ceiling = len(unique_candidates)

        replayed: list[_Candidate] = []
        for candidate in unique_candidates:
            if config.skip_replay:
                replayed.append(candidate)
                continue
            evidence = config.replay_lookup(
                ReplayCandidate(
                    trajectory_id=candidate.trajectory_id,
                    source_terminal_sha256=candidate.source_terminal_sha256,
                    content_sha256=candidate.content_sha256,
                    seed=candidate.seed,
                )
            ) if config.replay_lookup else None
            if not _replay_matches(candidate, evidence):
                counts["replay_drop"] += 1
                _record_rejection(
                    rejected, candidate.selected.task, "replay_drop", "evidence mismatch"
                )
                continue
            candidate.converted["replay_pending"] = False
            replayed.append(candidate)
            replay_records.append(
                {
                    "trajectory_id": evidence.trajectory_id,
                    "source_terminal_sha256": evidence.source_terminal_sha256,
                    "candidate_content_sha256": evidence.candidate_content_sha256,
                    "fixture_sha256": evidence.fixture_sha256,
                    "resolved": True,
                    "control": False,
                    "namespace": "candidate",
                }
            )

        replayed.sort(key=lambda value: (value.selected.language, value.selected.task))
        if config.max_output and len(replayed) > config.max_output:
            overflow = replayed[config.max_output :]
            replayed = replayed[: config.max_output]
            counts["output_limit_drop"] += len(overflow)
            for candidate in overflow:
                _record_rejection(
                    rejected,
                    candidate.selected.task,
                    "output_limit_drop",
                    f"max_output={config.max_output}",
                )

        output_rows = [candidate.converted for candidate in replayed]
        counts["output"] = len(output_rows)
        arithmetic_buckets = [
            "nonterminal",
            "validation",
            "language_drop",
            "category_drop",
            "structure_drop",
            "unsupported_tool",
            "contamination_drop",
            "duplicate_drop",
            "behavior_drop",
            "replay_drop",
            "token_drop",
            "output_limit_drop",
            "metadata_eligible" if config.metadata_only else "output",
        ]
        arithmetic_sum = sum(counts[key] for key in arithmetic_buckets)
        if arithmetic_sum != streamed:
            raise RuntimeError(
                f"manifest arithmetic mismatch: streamed={streamed}, buckets={arithmetic_sum}"
            )

        per_language = Counter(candidate.selected.language for candidate in replayed)
        per_category = Counter(candidate.selected.category for candidate in replayed)
        identity_payload = [
            {"task": candidate.selected.task, "trajectory_id": candidate.trajectory_id}
            for candidate in replayed
        ]
        rejection_reason_counts = Counter(row["reason"] for row in rejected)
        exclusion_contract = [
            {
                "instance_ids": sorted(record.instance_ids),
                "content_sha256s": sorted(record.content_sha256s),
                "text_sha256s": sorted(
                    _sha256_bytes(text.encode("utf-8")) for text in record.texts
                ),
            }
            for record in config.exclusions
        ]
        manifest: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "metadata_only": config.metadata_only,
            "source": {
                "dataset_id": config.source_metadata.dataset_id,
                "dataset_revision": config.source_metadata.dataset_revision,
                "source_lfs_sha256": config.source_metadata.source_lfs_sha256,
                "source_bytes": config.source_metadata.source_bytes,
                "expected_rows": config.source_contract.expected_rows,
                "expected_terminal_trajectories": config.source_contract.expected_terminal_trajectories,
                "moonshiner_revision": MOONSHINER_REVISION,
            },
            "config": {
                "max_output": config.max_output,
                "max_tokens": config.max_tokens,
                "workers": config.workers,
                "skip_replay": config.skip_replay,
                "max_first_edit_index": 10,
                "max_read_streak": 5,
                "reject_identical_consecutive_commands": True,
                "word_ngram_size": 13,
                "word_ngram_overlap_reject_at": 0.8,
            },
            "streamed": streamed,
            "terminal_trajectories": terminal_seen,
            **{key: counts[key] for key in _empty_counts()},
            "arithmetic_buckets": arithmetic_buckets,
            "arithmetic_sum": arithmetic_sum,
            "per_language": dict(sorted(per_language.items())),
            "per_category": dict(sorted(per_category.items())),
            "first_edit_histogram": _histogram(
                candidate.first_edit_index for candidate in replayed
            ),
            "read_streak_histogram": _histogram(
                candidate.max_read_streak for candidate in replayed
            ),
            "token_histogram": _histogram(candidate.token_count for candidate in replayed),
            "tool_name_counts": dict(sorted(tool_names.items())),
            "tool_argument_key_counts": dict(sorted(argument_keys.items())),
            "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
            "exclusion_contract_sha256": _sha256_bytes(
                _canonical_json_bytes(exclusion_contract)
            ),
            "eligible_unique_task_ceiling_before_replay": eligible_unique_task_ceiling,
            "representative_version": _REPRESENTATIVE_VERSION,
            "representative_tuple": [
                "first_edit_index",
                "max_read_streak",
                "normalized_command_count",
                "rendered_token_count",
                "canonical_terminal_sha256",
            ],
            "representatives": representative_manifest,
            "trajectory_task_identities_sha256": _sha256_bytes(
                _canonical_json_bytes(identity_payload)
            ),
            "replay_state": (
                "metadata_only"
                if config.metadata_only
                else "pending"
                if config.skip_replay
                else "verified"
            ),
            "controls_in_training": 0,
            "license": "CC BY 4.0",
            "attribution": (
                "greghavens/fable-5-coding-and-debugging-traces and "
                "greghavens/moonshiner"
            ),
        }
        sidecar_records = (
            row[0]
            for row in database.execute(
                "SELECT record FROM sidecar ORDER BY language, task, trajectory_id"
            )
        )
        published_manifest = _publish_directory(
            out,
            output_rows,
            sidecar_records,
            rejected,
            replay_records,
            manifest,
        )
        return BuildResult(
            rows=tuple(output_rows),
            manifest=published_manifest,
            rejected=tuple(
                sorted(
                    rejected,
                    key=lambda value: (
                        value["task"],
                        value["reason"],
                        value["detail"],
                    ),
                )
            ),
        )
    finally:
        database.close()
        shutil.rmtree(work_root, ignore_errors=True)


def seed_contract_from_git_objects(
    task: str,
    task_json_bytes: bytes,
    tree_entries: Iterable[tuple[str, str, str, str]],
) -> SeedContract:
    """Build a seed contract from immutable Git-object metadata only."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task):
        raise ValueError("task ID is not a confined seed name")
    try:
        payload = json.loads(task_json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"task {task} has invalid pinned task.json") from exc
    if payload.get("id") != task:
        raise ValueError(f"task {task} pinned task.json ID mismatch")
    verify_cmd = payload.get("verify_cmd")
    test_files = payload.get("test_files")
    if type(verify_cmd) is not str or not verify_cmd:
        raise ValueError(f"task {task} has no exact verify_cmd")
    if type(test_files) is not list or any(type(path) is not str for path in test_files):
        raise ValueError(f"task {task} has invalid test_files")
    for protected_path in test_files:
        _normalize_fable_path(protected_path)
    prefix = f"tasks/seeds/{task}/"
    inventory: list[dict[str, Any]] = []
    for mode, object_type, object_id, path in sorted(tree_entries, key=lambda item: item[3]):
        if not path.startswith(prefix):
            raise ValueError(f"task {task} tree entry escapes its seed prefix")
        relative = path.removeprefix(prefix)
        if not relative or ".." in PurePosixPath(relative).parts:
            raise ValueError(f"task {task} tree entry has invalid path")
        if mode == "120000":
            raise ValueError(f"task {task} fixture contains a symlink")
        if mode == "160000" or object_type == "commit":
            raise ValueError(f"task {task} fixture contains a submodule")
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"task {task} fixture contains a non-regular Git object")
        if not re.fullmatch(r"[0-9a-f]{40,64}", object_id):
            raise ValueError(f"task {task} fixture has an invalid Git object ID")
        inventory.append(
            {
                "path": relative,
                "type": "file",
                "mode": mode,
                "git_object": object_id,
            }
        )
    if not inventory or not any(item["path"] == "task.json" for item in inventory):
        raise ValueError(f"task {task} pinned tree has no task.json")
    return SeedContract(
        task=task,
        protected_paths=tuple(test_files),
        verify_cmd=verify_cmd,
        fixture_sha256=_sha256_bytes(_canonical_json_bytes(inventory)),
    )


def _load_seed_contract(root: Path, task: str) -> SeedContract:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task):
        raise ValueError("task ID is not a confined seed name")
    prefix = f"tasks/seeds/{task}"
    try:
        task_json_bytes = subprocess.run(
            ["git", "-C", str(root), "show", f"{MOONSHINER_REVISION}:{prefix}/task.json"],
            capture_output=True,
            check=True,
        ).stdout
        tree_raw = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-rz", MOONSHINER_REVISION, "--", prefix],
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"missing pinned Moonshiner task contract for {task}") from exc
    entries: list[tuple[str, str, str, str]] = []
    for raw_entry in tree_raw.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split(" ", 2)
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"task {task} has malformed pinned Git inventory") from exc
        entries.append((mode, object_type, object_id, path))
    return seed_contract_from_git_objects(task, task_json_bytes, entries)


def _load_exclusion(path: Path) -> ExclusionRecord:
    instance_ids: set[str] = set()
    content_hashes: set[str] = set()
    texts: list[str] = []
    raw = path.read_text(encoding="utf-8")
    payloads: list[Any]
    try:
        parsed = json.loads(raw)
        payloads = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        payloads = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("instance_id", "source_instance_id", "task"):
            value = payload.get(key)
            if isinstance(value, str):
                instance_ids.add(value)
        for key in ("problem_statement", "prompt", "text", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                texts.append(value)
        value = payload.get("content_sha256")
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            content_hashes.add(value)
        for key in ("instance_ids", "excluded_instance_ids"):
            values = payload.get(key)
            if isinstance(values, list):
                instance_ids.update(value for value in values if isinstance(value, str))
    return ExclusionRecord(
        instance_ids=frozenset(instance_ids),
        content_sha256s=frozenset(content_hashes),
        texts=tuple(texts),
    )


def _load_replay_ledger(path: Path) -> dict[str, ReplayEvidence]:
    records: dict[str, ReplayEvidence] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            evidence = ReplayEvidence(
                trajectory_id=payload["trajectory_id"],
                source_terminal_sha256=payload["source_terminal_sha256"],
                candidate_content_sha256=payload["candidate_content_sha256"],
                fixture_sha256=payload["fixture_sha256"],
                resolved=payload["resolved"],
                control=payload.get("control", False),
                namespace=payload.get("namespace", "candidate"),
            )
            if evidence.trajectory_id in records:
                raise ValueError(f"duplicate replay trajectory {evidence.trajectory_id}")
            records[evidence.trajectory_id] = evidence
    return records


def _source_jsonl_path(revision: str) -> Path:
    if revision != DATASET_REVISION:
        raise ValueError(f"dataset revision must be the audited pin {DATASET_REVISION}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the CLI source loader") from exc
    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            filename="traces.jsonl",
        )
    )


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed source JSONL at line {line_number}") from exc
            yield value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--max-output", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=49_152)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--exclusion", type=Path, action="append", default=[])
    parser.add_argument("--moonshiner-root", type=Path)
    parser.add_argument("--replay-ledger", type=Path)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = _source_jsonl_path(args.dataset_revision)
    source_metadata = SourceMetadata(
        dataset_id=DATASET_ID,
        dataset_revision=args.dataset_revision,
        source_lfs_sha256=_sha256_file(source_path),
        source_bytes=source_path.stat().st_size,
    )
    if args.metadata_only:
        token_counter: Callable[[list[dict[str, Any]]], int] = lambda _messages: 0
        seed_loader: Callable[[str], SeedContract] = lambda task: (_ for _ in ()).throw(
            RuntimeError(f"metadata-only unexpectedly requested seed {task}")
        )
    else:
        if args.tokenizer is None:
            raise ValueError("--tokenizer is required unless --metadata-only is set")
        if args.moonshiner_root is None:
            raise ValueError("--moonshiner-root is required unless --metadata-only is set")
        moonshiner_head = subprocess.run(
            ["git", "-C", str(args.moonshiner_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        validate_moonshiner_revision(moonshiner_head)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))

        def token_counter(messages: list[dict[str, Any]]) -> int:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            return len(rendered)

        seed_loader = lambda task: _load_seed_contract(args.moonshiner_root, task)
    replay_records = (
        _load_replay_ledger(args.replay_ledger) if args.replay_ledger else {}
    )
    if not args.skip_replay and not args.metadata_only and not replay_records:
        raise ValueError("--replay-ledger is required unless --skip-replay is set")
    config = BuildConfig(
        out=args.out,
        source_metadata=source_metadata,
        token_counter=token_counter,
        seed_contract=seed_loader,
        exclusions=tuple(_load_exclusion(path) for path in args.exclusion),
        replay_lookup=lambda candidate: replay_records.get(candidate.trajectory_id),
        skip_replay=args.skip_replay,
        max_output=args.max_output,
        max_tokens=args.max_tokens,
        workers=args.workers,
        metadata_only=args.metadata_only,
    )
    result = build_fable5_pilot(_jsonl_rows(source_path), config)
    print(json.dumps(result.manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
