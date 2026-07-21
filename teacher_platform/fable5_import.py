"""Pinned Fable 5 source selection and fail-closed structural validation."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final, Iterable, Literal

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
