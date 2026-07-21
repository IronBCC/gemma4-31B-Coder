"""Pinned Fable 5 source selection and fail-closed structural validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Literal


DATASET_ID: Final = "greghavens/fable-5-coding-and-debugging-traces"
DATASET_REVISION: Final = "aef8506515979988aa5c1a423f5b0fb3cee60382"
SOURCE_LFS_SHA256: Final = (
    "ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3"
)
SOURCE_BYTES: Final = 730_331_947
EXPECTED_ROWS: Final = 12_408
EXPECTED_TERMINAL_TRAJECTORIES: Final = 2_377

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
