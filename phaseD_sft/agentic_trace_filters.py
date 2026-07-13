"""Quality filters for agentic SFT traces."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from phaseD_sft.prune_loops import extract_events


# Keep these command patterns aligned with /tmp/audit_edit_behavior.py.  This
# filter is intentionally command-level: the event-level classifier below is
# useful for broad trace quality, but cannot reliably recover shell ordering.
_COMMAND_EDIT_RE = re.compile(
    r"sed -i|perl -i|apply_patch|git apply|cat > |cat >> |tee |>>|"
    r"python3? - <<|str_replace|open\(.+[\"']w[\"']"
)
_COMMAND_READ_RE = re.compile(
    r"^\s*(cat |find |grep |rg |sed -n|head |tail |ls |pwd|git (status|diff|show|log))"
)


@dataclass(frozen=True)
class TraceQualityReport:
    event_count: int
    first_edit_index: int | None
    first_edit_ratio: float
    max_read_streak: int
    has_verify_after_edit: bool
    final_event_class: str | None
    keep: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CommandTraceQualityReport:
    """Exact shell-command quality signals used for edit-decisive mixtures."""

    command_count: int
    first_edit_index: int | None
    max_read_streak: int
    has_identical_consecutive_repeat: bool
    keep: bool
    reasons: tuple[str, ...]


def _assistant_commands(messages: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            command = parsed.get("command")
            if isinstance(command, str) and command:
                commands.append(command)
    return commands


def command_trace_quality_report(
    messages: list[dict[str, Any]],
    *,
    max_first_edit_index: int | None = None,
    max_read_streak: int | None = None,
    reject_identical_consecutive_commands: bool = False,
) -> CommandTraceQualityReport:
    """Evaluate ordered bash calls with the v6 behavior-audit semantics.

    ``first_edit_index`` is one-based to match the human-facing audit table:
    the first tool call is command 1, not index 0.
    """

    commands = _assistant_commands(messages)
    reasons: list[str] = []
    first_edit_index = next(
        (index for index, command in enumerate(commands, start=1) if _COMMAND_EDIT_RE.search(command)),
        None,
    )
    if not commands:
        reasons.append("no_commands")
    elif first_edit_index is None:
        reasons.append("no_edit")
    elif max_first_edit_index is not None and first_edit_index > max_first_edit_index:
        reasons.append("first_edit_after_limit")

    read_run = 0
    max_read_run = 0
    has_repeat = False
    previous: str | None = None
    for command in commands:
        if _COMMAND_READ_RE.search(command):
            read_run += 1
            max_read_run = max(max_read_run, read_run)
        else:
            read_run = 0
        if previous is not None and command.strip() == previous.strip():
            has_repeat = True
        previous = command

    if max_read_streak is not None and max_read_run > max_read_streak:
        reasons.append("read_streak_exceeded")
    if reject_identical_consecutive_commands and has_repeat:
        reasons.append("identical_consecutive_command")

    return CommandTraceQualityReport(
        command_count=len(commands),
        first_edit_index=first_edit_index,
        max_read_streak=max_read_run,
        has_identical_consecutive_repeat=has_repeat,
        keep=not reasons,
        reasons=tuple(reasons),
    )


def trace_quality_report(
    messages: list[dict[str, Any]],
    *,
    max_first_edit_ratio: float = 0.4,
    max_read_streak: int = 6,
    require_verify_tail: bool = True,
) -> TraceQualityReport:
    events = extract_events(messages)
    reasons: list[str] = []

    first_edit_index = next((i for i, event in enumerate(events) if event.event_class == "mutate"), None)
    if not events:
        reasons.append("no_events")

    if first_edit_index is None:
        reasons.append("no_edit")
        first_edit_ratio = 1.0
    else:
        denominator = max(1, len(events))
        first_edit_ratio = first_edit_index / denominator
        if first_edit_ratio > max_first_edit_ratio:
            reasons.append("late_first_edit")

    max_read_run = 0
    read_run = 0
    for event in events:
        if event.event_class == "read":
            read_run += 1
            max_read_run = max(max_read_run, read_run)
        else:
            read_run = 0
    if max_read_run > max_read_streak:
        reasons.append("read_loop")

    has_verify_after_edit = False
    if first_edit_index is not None:
        has_verify_after_edit = any(event.event_class == "verify" for event in events[first_edit_index + 1 :])
        if require_verify_tail and not has_verify_after_edit:
            reasons.append("no_verify_tail")

    if events and events[-1].event_class == "read":
        reasons.append("read_ending")

    return TraceQualityReport(
        event_count=len(events),
        first_edit_index=first_edit_index,
        first_edit_ratio=round(first_edit_ratio, 4),
        max_read_streak=max_read_run,
        has_verify_after_edit=has_verify_after_edit,
        final_event_class=events[-1].event_class if events else None,
        keep=not reasons,
        reasons=tuple(reasons),
    )


def trace_should_keep(
    messages: list[dict[str, Any]],
    *,
    max_first_edit_ratio: float = 0.4,
    max_read_streak: int = 6,
    require_verify_tail: bool = True,
) -> bool:
    return trace_quality_report(
        messages,
        max_first_edit_ratio=max_first_edit_ratio,
        max_read_streak=max_read_streak,
        require_verify_tail=require_verify_tail,
    ).keep
