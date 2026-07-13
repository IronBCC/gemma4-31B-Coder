"""Shared request-clamp helpers for SWE harnesses."""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import re
from typing import Any

# 512 truncated tool calls (29% of gens hit the ceiling -> "No tool calls found" loops).
# 32_768 let degenerate thinking loops burn ~12 min/request (45 tok/s) before returning.
# 4096 fits reasoning + tool call comfortably (observed p90 ~151 tokens without thinking)
# while failing runaway loops fast enough for the harness recovery to act.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_OBSERVATION_CHARS = 2_000
NO_EDIT_PRESSURE_STEP = 25
FORCE_DIFF_STEP = 37
FORCE_SUBMIT_STEP = 38
REPEATED_READ_THRESHOLD = 5
REPEATED_EDIT_THRESHOLD = 3
DEFAULT_STOP_SEQUENCES = (
    "<|turn>",
    "<|turn>model",
    "<end_of_turn>",
    "<|end_of_turn|>",
    "<eot>",
    "<|eot_id|>",
)
_IMPORTANT_RE = re.compile(
    r"error|exception|traceback|failed|failure|warning|warn|panic|assert|undefined|"
    r"not found|no such file|permission denied|denied|segmentation fault|oom|out of memory",
    re.IGNORECASE,
)
_READ_RE = re.compile(r"(^|\b)(cat|find|grep|rg|sed -n|head|tail|ls|pwd)\b")
_EDIT_RE = re.compile(r"(sed -i|perl -pi|python .*write|cat >|tee |apply_patch|>>|> [^&|]*\\.(py|txt|cfg|toml|rst|md))")
_DIFF_RE = re.compile(r"\bgit diff\b|cat patch\.txt")
_SUBMIT_RE = re.compile(r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


def _normalize_stop_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        values: list[str] = []
        for item in value:
            if item is None:
                continue
            values.append(str(item))
        return values
    return [str(value)]


def build_call_kwargs(
    model_kwargs: dict[str, Any],
    *,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    default_stop_sequences: tuple[str, ...] = DEFAULT_STOP_SEQUENCES,
) -> dict[str, Any]:
    """Return OpenAI-compatible call kwargs with output clamps applied."""

    call_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    if "max_tokens" not in call_kwargs:
        call_kwargs["max_tokens"] = default_max_tokens

    merged_stops: list[str] = []
    for value in _normalize_stop_values(call_kwargs.pop("stop", None)):
        if value not in merged_stops:
            merged_stops.append(value)
    for value in default_stop_sequences:
        if value not in merged_stops:
            merged_stops.append(value)
    call_kwargs["stop"] = merged_stops
    return call_kwargs


def compact_live_messages(
    messages: list[dict[str, Any]],
    *,
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
) -> list[dict[str, Any]]:
    """Compact tool observations before sending a live SWE prompt to vLLM."""

    compacted: list[dict[str, Any]] = []
    seen_observations: dict[str, int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            compacted.append(message)
            continue

        cloned = dict(message)
        content = cloned.get("content")
        if not isinstance(content, str) or not _is_observation_message(cloned, content):
            compacted.append(cloned)
            continue

        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        digest = hashlib.sha256(normalized.encode("utf-8", "replace")).hexdigest()
        if digest in seen_observations:
            cloned["content"] = f"[unchanged tool observation omitted; first seen at message {seen_observations[digest]}]"
        elif len(normalized) > max_observation_chars:
            seen_observations[digest] = index
            cloned["content"] = _compact_text(normalized, max_observation_chars)
        else:
            seen_observations[digest] = index
            cloned["content"] = normalized
        compacted.append(cloned)
    return compacted


def add_budget_pressure_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append short live-only steering messages when the agent is looping."""

    commands = extract_bash_commands(messages)
    if not commands:
        return messages

    kinds = [command_kind(command) for command in commands]
    pressure: list[str] = []
    if len(commands) >= NO_EDIT_PRESSURE_STEP and "edit" not in kinds:
        pressure.append(
            "Budget warning: you have inspected enough. Stop reading, make the smallest source edit now, "
            "then run a focused check and submit a git diff."
        )

    repeated_command, repeated_count = repeated_tail_command(commands)
    if repeated_command and repeated_count >= REPEATED_READ_THRESHOLD and command_kind(repeated_command) == "read":
        pressure.append(
            f"Loop warning: you repeated the same read-only command {repeated_count} times. "
            "Do not run it again; make a minimal edit or submit the current patch."
        )

    if not pressure:
        return messages
    return [*messages, {"role": "user", "content": "\n".join(pressure)}]


def forced_command_for_history(messages: list[dict[str, Any]]) -> str | None:
    """Return a harness-enforced bash command when history shows a terminal loop."""

    commands = extract_bash_commands(messages)
    if not commands:
        return None

    last_forced = last_guarded_forced_command(messages)
    if last_forced and command_kind(last_forced) == "diff":
        return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"

    kinds = [command_kind(command) for command in commands]
    repeated_command, repeated_count = repeated_tail_command(commands)
    if repeated_command:
        repeated_kind = command_kind(repeated_command)
        if repeated_kind == "read" and repeated_count >= REPEATED_READ_THRESHOLD:
            return "git diff -- . > patch.txt && cat patch.txt"
        if repeated_kind == "edit" and repeated_count >= REPEATED_EDIT_THRESHOLD:
            return "git diff -- . > patch.txt && cat patch.txt"
        if repeated_kind == "diff" and repeated_count >= 2:
            return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"

    if len(commands) >= FORCE_DIFF_STEP and "diff" not in kinds:
        return "git diff -- . > patch.txt && cat patch.txt"
    if len(commands) >= FORCE_SUBMIT_STEP and "diff" in kinds and "submit" not in kinds:
        return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    return None


def last_guarded_forced_command(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        response = message.get("extra", {}).get("response", {})
        if not isinstance(response, dict):
            continue
        command = response.get("guarded_forced_command")
        return command if isinstance(command, str) else None
    return None


def last_tool_output_contains_diff(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            return False
        if message.get("role") != "tool":
            continue
        text = str(message.get("content") or "")
        raw = str(message.get("extra", {}).get("raw_output") or "")
        return "diff --git " in text or "diff --git " in raw
    return False


def extract_bash_commands(messages: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for action in message.get("extra", {}).get("actions", []) or []:
            command = action.get("command") if isinstance(action, dict) else None
            if isinstance(command, str):
                commands.append(command)
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if function.get("name") != "bash":
                continue
            command = _command_from_tool_arguments(function.get("arguments"))
            if command is not None and (not commands or commands[-1] != command):
                commands.append(command)
    return commands


def command_kind(command: str) -> str:
    stripped = command.strip()
    if _SUBMIT_RE.search(stripped):
        return "submit"
    if _DIFF_RE.search(stripped):
        return "diff"
    if _EDIT_RE.search(stripped):
        return "edit"
    if _READ_RE.search(stripped):
        return "read"
    return "other"


def repeated_tail_command(commands: list[str]) -> tuple[str | None, int]:
    if not commands:
        return None, 0
    last = commands[-1].strip()
    count = 0
    for command in reversed(commands):
        if command.strip() != last:
            break
        count += 1
    return last, count


def _is_observation_message(message: dict[str, Any], content: str) -> bool:
    role = str(message.get("role", "")).lower()
    return role == "tool" or content.lstrip().lower().startswith(("observation:", "tool observation:"))


def _command_from_tool_arguments(arguments: Any) -> str | None:
    if isinstance(arguments, dict):
        command = arguments.get("command")
        return command if isinstance(command, str) else None
    if not isinstance(arguments, str):
        return None
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    command = decoded.get("command")
    return command if isinstance(command, str) else None


def extract_recoverable_command_from_arguments(arguments: Any) -> str | None:
    """Extract a bash command from common malformed tool-call argument shapes."""

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return extract_recoverable_command_from_text(arguments)
    else:
        parsed = arguments
    return _extract_recoverable_command_from_obj(parsed)


def extract_recoverable_command_from_text(text: str) -> str | None:
    match = _COMMAND_JSON_RE.search(text)
    if not match:
        return None
    try:
        command = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        command = match.group(1)
    return command.strip() or None


def _extract_recoverable_command_from_obj(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        commands = [_extract_recoverable_command_from_obj(item) for item in value]
        commands = [command for command in commands if command]
        return "\n".join(commands) if commands else None
    if not isinstance(value, dict):
        return None

    for key in ("command", "cmd", "bash", "shell"):
        command = _extract_recoverable_command_from_obj(value.get(key))
        if command:
            return command
    for key in ("parameters", "arguments", "args", "input"):
        command = _extract_recoverable_command_from_obj(value.get(key))
        if command:
            return command
    for key in ("commands", "command_list"):
        command = _extract_recoverable_command_from_obj(value.get(key))
        if command:
            return command
    return None


def _compact_text(text: str, max_chars: int) -> str:
    if max_chars < 200:
        return text[:max_chars]

    lines = text.splitlines()
    head = lines[:18]
    tail = lines[-10:] if len(lines) > 10 else []
    important = [line for line in lines if _IMPORTANT_RE.search(line)][:32]
    body = _dedup_lines([*head, "[... live observation compacted ...]", *important, *tail])
    compacted = "\n".join(body)
    if len(compacted) <= max_chars:
        return compacted

    marker = "\n[... live observation compacted ...]\n"
    head_budget = max(1, (max_chars - len(marker)) // 2)
    tail_budget = max(1, max_chars - len(marker) - head_budget)
    return text[:head_budget].rstrip() + marker + text[-tail_budget:].lstrip()


_COMMAND_JSON_RE = re.compile(r'"command"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)


def _dedup_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return result
