"""Shared request-clamp helpers for SWE harnesses."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Any

# 512 truncated tool calls (29% of gens hit the ceiling -> "No tool calls found" loops).
# 32_768 let degenerate thinking loops burn ~12 min/request (45 tok/s) before returning.
# 4096 fits reasoning + tool call comfortably (observed p90 ~151 tokens without thinking)
# while failing runaway loops fast enough for the harness recovery to act.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_OBSERVATION_CHARS = 2_000
# The force-diff / force-submit floors are a LAST RESORT, not the step budget.  They used
# to be hardcoded at 37/38, which made the agent config's `step_limit` decorative: every
# run ended at ~39 assistant steps and most "empty patch" outcomes were this injection
# rather than model behaviour.  They now sit just under the real step limit, so raising
# `step_limit` actually buys the agent more steps.
#
#   MSWEA_STEP_LIMIT        real agent step limit (smoke_single.sh exports it from the config)
#   MSWEA_FORCE_DIFF_STEP   explicit override; 0 disables the forced diff
#   MSWEA_FORCE_SUBMIT_STEP explicit override; 0 disables the forced submit
#   MSWEA_NO_EDIT_PRESSURE_STEP explicit override for the "stop reading" nudge
DEFAULT_STEP_LIMIT = 120
FORCE_DIFF_MARGIN = 3
FORCE_SUBMIT_MARGIN = 2
NO_EDIT_PRESSURE_FRACTION = 0.6
REPEATED_READ_THRESHOLD = 5
REPEATED_EDIT_THRESHOLD = 3
REPEATED_FAILURE_THRESHOLD = 3
RECOVERY_DIAGNOSTIC_COMMAND = (
    "git diff --check; git status --short; "
    "git diff -- . | sed -n '1,240p'"
)
_SOURCE_DIFF_EXCLUDES = (
    "**/test/**",
    "**/tests/**",
    "**/testing/**",
    "**/fixture/**",
    "**/fixtures/**",
    "**/benchmark/**",
    "**/benchmarks/**",
    "**/harness/**",
    "**/harnesses/**",
    "**/test_*.py",
    "**/*_test.py",
    "**/*_tests.py",
    "**/conftest.py",
    "**/eval.sh",
    "**/evaluation.sh",
    "**/noxfile.py",
    "**/pyproject.toml",
    "**/pytest.ini",
    "**/run_tests.py",
    "**/run_tests.sh",
    "**/runtests.py",
    "**/setup.cfg",
    "**/setup.py",
    "**/test-requirements.txt",
    "**/tox.ini",
)
_SOURCE_DIFF_PATHS = " ".join(
    f"':(exclude,glob){pattern}'" for pattern in _SOURCE_DIFF_EXCLUDES
)
_UNTRACKED_ARTIFACT_EXCLUDES = (
    "repro*",
    "**/repro*",
    "**/repro*/**",
    "scratch*",
    "**/scratch*",
    "**/scratch*/**",
    "manage.py",
    "**/manage.py",
    "patch.txt",
    "**/patch.txt",
    "*.db",
    "**/*.db",
    "*.sqlite",
    "**/*.sqlite",
    "*.sqlite3",
    "**/*.sqlite3",
    "*.log",
    "**/*.log",
)
_UNTRACKED_SOURCE_DIFF_PATHS = " ".join(
    f"':(exclude,glob){pattern}'"
    for pattern in (*_SOURCE_DIFF_EXCLUDES, *_UNTRACKED_ARTIFACT_EXCLUDES)
)
SOURCE_ONLY_DIFF_COMMAND = (
    f"git add -N -- . {_UNTRACKED_SOURCE_DIFF_PATHS} && "
    f"git diff HEAD -- . {_SOURCE_DIFF_PATHS} > patch.txt && cat patch.txt"
)
RECOVERY_DIFF_COMMAND = SOURCE_ONLY_DIFF_COMMAND
_RETURNCODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)


@dataclass(frozen=True)
class CommandOutcome:
    """One unambiguously paired bash command and its rendered observation."""

    command: str
    returncode: int | None
    output_sha256: str


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def step_limit() -> int:
    """Real agent step limit for this run (env, else the historical default)."""

    value = _env_int("MSWEA_STEP_LIMIT")
    return value if value and value > 0 else DEFAULT_STEP_LIMIT


def force_diff_step() -> int | None:
    """Command count at which the harness injects `git diff`; None disables it."""

    override = _env_int("MSWEA_FORCE_DIFF_STEP")
    if override is not None:
        return override if override > 0 else None
    return max(4, step_limit() - FORCE_DIFF_MARGIN)


def force_submit_step() -> int | None:
    """Command count at which the harness injects submit; None disables it."""

    override = _env_int("MSWEA_FORCE_SUBMIT_STEP")
    if override is not None:
        return override if override > 0 else None
    return max(5, step_limit() - FORCE_SUBMIT_MARGIN)


def no_edit_pressure_step() -> int:
    """Command count at which the agent is told to stop reading and edit."""

    override = _env_int("MSWEA_NO_EDIT_PRESSURE_STEP")
    if override is not None and override > 0:
        return override
    return max(3, int(step_limit() * NO_EDIT_PRESSURE_FRACTION))
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
    recovery_pressure = recovery_pressure_for_history(messages)
    if recovery_pressure:
        pressure.append(recovery_pressure)
    if len(commands) >= no_edit_pressure_step() and "edit" not in kinds:
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

    outcomes = extract_command_outcomes(messages)
    failed_command, after_diagnostic = _repeated_failed_command(outcomes)
    if failed_command and not after_diagnostic:
        return RECOVERY_DIAGNOSTIC_COMMAND
    if failed_command and after_diagnostic:
        return None

    if last_submission_was_rejected(messages):
        return SOURCE_ONLY_DIFF_COMMAND

    last_forced = last_guarded_forced_command(messages)
    if last_forced and command_kind(last_forced) == "diff":
        if last_tool_output_contains_diff(messages):
            return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        if last_forced.strip() == RECOVERY_DIFF_COMMAND:
            return None
        return RECOVERY_DIFF_COMMAND

    kinds = [command_kind(command) for command in commands]
    repeated_command, repeated_count = repeated_tail_command(commands)
    if repeated_command:
        repeated_kind = command_kind(repeated_command)
        if repeated_kind == "read" and repeated_count >= REPEATED_READ_THRESHOLD:
            return SOURCE_ONLY_DIFF_COMMAND
        if repeated_kind == "edit" and repeated_count >= REPEATED_EDIT_THRESHOLD:
            matching_outcomes = _tail_outcomes(outcomes, repeated_command, REPEATED_EDIT_THRESHOLD)
            if not outcomes or (
                len(matching_outcomes) == REPEATED_EDIT_THRESHOLD
                and all(row.returncode == 0 for row in matching_outcomes)
            ):
                return SOURCE_ONLY_DIFF_COMMAND
        if repeated_kind == "diff" and repeated_count >= 2:
            return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"

    diff_step = force_diff_step()
    submit_step = force_submit_step()
    if diff_step is not None and len(commands) >= diff_step and "diff" not in kinds:
        return SOURCE_ONLY_DIFF_COMMAND
    if (
        submit_step is not None
        and len(commands) >= submit_step
        and "diff" in kinds
        and "submit" not in kinds
    ):
        return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    return None


def last_submission_was_rejected(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            return False
        content = str(message.get("content") or "")
        raw = str(message.get("extra", {}).get("raw_output") or "")
        if "SUBMISSION REJECTED:" in content or "SUBMISSION REJECTED:" in raw:
            return True
    return False


def recovery_pressure_for_history(messages: list[dict[str, Any]]) -> str | None:
    """Describe how to recover after an exact command fails three times."""

    failed_command, after_diagnostic = _repeated_failed_command(
        extract_command_outcomes(messages)
    )
    if not failed_command:
        return None
    if after_diagnostic:
        return (
            f"Diagnostic complete. The failed command was `{failed_command}`. "
            "Do not run this command again unchanged; inspect its error and the current diff, "
            "then repair or revert the damaged hunk before running a focused behavior check."
        )
    return (
        f"Recovery required: `{failed_command}` failed repeatedly. "
        "Do not run this command again unchanged. Inspect the failing output and current diff, "
        "then repair or revert the damaged hunk."
    )


def extract_command_outcomes(
    messages: list[dict[str, Any]],
) -> list[CommandOutcome]:
    """Pair only an unambiguous assistant bash call with its adjacent observation."""

    outcomes: list[CommandOutcome] = []
    for index, message in enumerate(messages[:-1]):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        call = _single_bash_call(message)
        if call is None:
            continue
        command, call_id = call
        observation = messages[index + 1]
        if not _is_adjacent_command_observation(observation, call_id):
            continue
        content = str(observation.get("content") or "")
        match = _RETURNCODE_RE.search(content)
        returncode = int(match.group(1)) if match else None
        raw_output = observation.get("extra", {}).get("raw_output")
        hash_input = raw_output if isinstance(raw_output, str) else content
        outcomes.append(CommandOutcome(
            command=command,
            returncode=returncode,
            output_sha256=hashlib.sha256(
                hash_input.encode("utf-8", "replace")
            ).hexdigest(),
        ))
    return outcomes


def _single_bash_call(message: dict[str, Any]) -> tuple[str, str | None] | None:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        if len(tool_calls) != 1 or not isinstance(tool_calls[0], dict):
            return None
        tool_call = tool_calls[0]
        function = tool_call.get("function") or {}
        if not isinstance(function, dict) or function.get("name") != "bash":
            return None
        command = _command_from_tool_arguments(function.get("arguments"))
        if command is None:
            return None
        call_id = tool_call.get("id")
        return command, call_id if isinstance(call_id, str) else None

    actions = message.get("extra", {}).get("actions", []) or []
    bash_actions = [
        action for action in actions
        if isinstance(action, dict) and isinstance(action.get("command"), str)
    ]
    if len(bash_actions) != 1:
        return None
    action = bash_actions[0]
    call_id = action.get("tool_call_id")
    return action["command"], call_id if isinstance(call_id, str) else None


def _is_adjacent_command_observation(
    message: Any,
    call_id: str | None,
) -> bool:
    if not isinstance(message, dict):
        return False
    content = str(message.get("content") or "")
    role = str(message.get("role") or "").lower()
    if role != "tool" and not (
        role == "user" and _RETURNCODE_RE.search(content)
    ):
        return False
    observation_call_id = message.get("tool_call_id")
    if (
        call_id is not None
        and observation_call_id is not None
        and observation_call_id != call_id
    ):
        return False
    return True


def _tail_outcomes(
    outcomes: list[CommandOutcome],
    command: str,
    count: int,
) -> list[CommandOutcome]:
    normalized = command.strip()
    if len(outcomes) < count:
        return []
    tail = outcomes[-count:]
    if any(row.command.strip() != normalized for row in tail):
        return []
    return tail


def _repeated_failed_command(
    outcomes: list[CommandOutcome],
) -> tuple[str | None, bool]:
    if not outcomes:
        return None, False
    after_diagnostic = outcomes[-1].command.strip() == RECOVERY_DIAGNOSTIC_COMMAND
    candidates = outcomes[:-1] if after_diagnostic else outcomes
    if len(candidates) < REPEATED_FAILURE_THRESHOLD:
        return None, after_diagnostic
    tail = candidates[-REPEATED_FAILURE_THRESHOLD:]
    normalized = tail[-1].command.strip()
    if (
        any(row.returncode in (None, 0) for row in tail)
    ):
        return None, after_diagnostic
    same_command = all(row.command.strip() == normalized for row in tail)
    same_failure = len({row.output_sha256 for row in tail}) == 1
    if not same_command and not same_failure:
        return None, after_diagnostic
    return normalized, after_diagnostic


def last_guarded_forced_command(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        response = message.get("extra", {}).get("response", {})
        if not isinstance(response, dict):
            continue
        command = response.get("guarded_forced_command")
        if isinstance(command, str):
            return command
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
