"""Normalize and preflight raw teacher traces before training admission."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

if __package__:
    from .success_trace_distill import (
        ReplayStep,
        ReplayTrace,
        command_mutates_source,
        distill_success_path,
    )
else:
    from success_trace_distill import (  # type: ignore[no-redef]
        ReplayStep,
        ReplayTrace,
        command_mutates_source,
        distill_success_path,
    )


_DOCKER_EXEC_RE = re.compile(
    r"^\s*docker\s+exec\s+\S+\s+bash\s+-c\s+"
    r"(?P<quote>[\"'])(?P<inner>.*)(?P=quote)\s*$",
    re.DOTALL,
)
_DOCKER_EXEC_PREFIX_RE = re.compile(
    r"^\s*docker\s+exec(?:\s+-[A-Za-z]+)*\s+"
    r"[0-9a-f]{12,64}\s+(?P<inner>.+)$",
    re.DOTALL,
)


class TraceNormalizationError(ValueError):
    """A raw teacher stream cannot be joined into an exact message ledger."""


class NormalizedTrace(list):
    """Ordered command steps plus the teacher's real terminal assistant turn."""

    def __init__(self, steps=(), *, terminal_assistant: str = ""):
        super().__init__(steps)
        self.terminal_assistant = terminal_assistant.strip()


class TracePreflightError(ValueError):
    """A strict patch lacks a replayable, trainable raw trajectory."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class TracePreflight:
    retained_steps: int
    source_sha256: str


def _strip_docker_exec(command: str) -> str:
    """Remove only a recognized teacher-runtime Docker wrapper."""

    command = command.strip()
    bash_match = _DOCKER_EXEC_RE.match(command)
    if bash_match:
        inner = bash_match.group("inner")
        if bash_match.group("quote") == '"':
            inner = re.sub(r'\\([$`"\\\n])', r"\1", inner)
    else:
        prefix_match = _DOCKER_EXEC_PREFIX_RE.match(command)
        inner = prefix_match.group("inner") if prefix_match else command
    return re.sub(r"^\s*cd\s+/testbed\s*&&\s*", "", inner).strip()


def _observation_returncode(
    value: Mapping[str, Any],
    text: str,
    *,
    default: int,
) -> int:
    returncode = value.get("returncode")
    if type(returncode) is int:
        return returncode
    is_error = value.get("is_error")
    if type(is_error) is bool:
        return 1 if is_error else 0
    match = re.search(
        r"<returncode>\s*(-?\d+)\s*</returncode>",
        text,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else default


def normalize_steps(backend: str, stream_path: Path) -> NormalizedTrace:
    """Parse a raw teacher stream into exact ordered tool/observation steps."""

    steps: list[dict[str, Any]] = []
    try:
        lines = [
            json.loads(line)
            for line in Path(stream_path).read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            if line.strip().startswith("{")
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceNormalizationError(
            f"unparseable stream: {stream_path}"
        ) from exc
    terminal = ""
    if backend == "openrouter":
        pending_thought = ""
        for value in lines:
            if value.get("role") == "assistant":
                content = (value.get("content") or "").strip()
                if value.get("tool_calls"):
                    pending_thought = content
                elif content:
                    terminal = content
            elif value.get("role") == "tool":
                command = _strip_docker_exec(value.get("command", ""))
                if command:
                    observation = (value.get("observation") or "")[-2000:]
                    steps.append({
                        "thought": pending_thought,
                        "command": command,
                        "observation": observation,
                        "returncode": _observation_returncode(
                            value,
                            observation,
                            default=-1,
                        ),
                        "mutates_source": command_mutates_source(command),
                    })
                    pending_thought = ""
    elif backend == "claude":
        unresolved: dict[str, int] = {}
        resolved: set[str] = set()
        synthetic = 0
        for value in lines:
            if value.get("type") == "assistant":
                blocks = (value.get("message", {}) or {}).get("content", [])
                text = "\n".join(
                    (block.get("text") or "").strip()
                    for block in blocks
                    if block.get("type") == "text"
                    and (block.get("text") or "").strip()
                )
                tools = [
                    block
                    for block in blocks
                    if block.get("type") == "tool_use"
                    and block.get("name") == "Bash"
                ]
                if tools:
                    for block in tools:
                        command = _strip_docker_exec(
                            (block.get("input") or {}).get("command", "")
                        )
                        if not command:
                            raise TraceNormalizationError(
                                "empty Bash tool command"
                            )
                        call_id = block.get("id")
                        if not isinstance(call_id, str) or not call_id:
                            if len(tools) != 1:
                                raise TraceNormalizationError(
                                    "parallel Bash tool call is missing "
                                    "tool_use_id"
                                )
                            call_id = f"unambiguous-{synthetic}"
                            synthetic += 1
                        if call_id in unresolved or call_id in resolved:
                            raise TraceNormalizationError(
                                "duplicate tool_use_id"
                            )
                        unresolved[call_id] = len(steps)
                        steps.append({
                            "thought": text,
                            "command": command,
                            "observation": "",
                            "returncode": None,
                            "mutates_source": command_mutates_source(command),
                            "tool_use_id": call_id,
                        })
                elif text:
                    terminal = text
            elif value.get("type") == "user":
                for block in (value.get("message", {}) or {}).get(
                    "content",
                    [],
                ):
                    if block.get("type") != "tool_result":
                        continue
                    call_id = block.get("tool_use_id")
                    if not isinstance(call_id, str) or not call_id:
                        if len(unresolved) != 1:
                            raise TraceNormalizationError(
                                "tool result is missing unambiguous "
                                "tool_use_id"
                            )
                        call_id = next(iter(unresolved))
                    if call_id in resolved:
                        raise TraceNormalizationError(
                            "duplicate tool result"
                        )
                    if call_id not in unresolved:
                        raise TraceNormalizationError("orphan tool result")
                    content = block.get("content")
                    text = (
                        content
                        if isinstance(content, str)
                        else " ".join(
                            item.get("text", "")
                            for item in (content or [])
                            if isinstance(item, dict)
                        )
                    )
                    step = steps[unresolved.pop(call_id)]
                    observation = (text or "")[-2000:]
                    step["observation"] = observation
                    step["returncode"] = _observation_returncode(
                        block,
                        observation,
                        default=0,
                    )
                    resolved.add(call_id)
        if unresolved:
            raise TraceNormalizationError("missing tool result")
    elif backend == "codex":
        pending_text = ""
        for value in lines:
            if value.get("type") != "item.completed":
                continue
            item = value.get("item") or {}
            if item.get("type") == "agent_message":
                pending_text = (item.get("text") or "").strip()
            elif item.get("type") == "command_execution":
                command = _strip_docker_exec(item.get("command", ""))
                if command:
                    observation = (
                        item.get("aggregated_output") or ""
                    )[-2000:]
                    steps.append({
                        "thought": pending_text,
                        "command": command,
                        "observation": observation,
                        "returncode": _observation_returncode(
                            item,
                            observation,
                            default=-1,
                        ),
                        "mutates_source": command_mutates_source(command),
                    })
                    pending_text = ""
        terminal = pending_text
    else:
        raise TraceNormalizationError(f"unsupported backend: {backend}")
    return NormalizedTrace(steps, terminal_assistant=terminal)


def _classify_preflight_error(error: ValueError) -> TracePreflightError:
    detail = str(error)
    if "focused passing test" in detail:
        code = "missing_focused_test"
    elif (
        "mutation subsequence" in detail
        or "patch identity" in detail
        or "admitted patch" in detail
    ):
        code = "raw_patch_mismatch"
    else:
        code = "normalization_error"
    return TracePreflightError(code, detail)


def preflight_trainable_trace(
    task: Mapping[str, Any],
    backend: str,
    stream_path: Path,
    evidence: Mapping[str, Any],
) -> TracePreflight:
    """Prove a raw trace can distill to the exact admitted success path."""

    try:
        normalized = normalize_steps(backend, stream_path)
    except TraceNormalizationError as exc:
        raise TracePreflightError("normalization_error", str(exc)) from exc
    trace = ReplayTrace(
        steps=tuple(
            ReplayStep(
                command=step["command"],
                observation=step["observation"],
                returncode=step["returncode"],
                mutates_source=step["mutates_source"],
                assistant=step.get("thought", ""),
                loss=True,
            )
            for step in normalized
        ),
        terminal_assistant=normalized.terminal_assistant,
    )
    try:
        distilled = distill_success_path(task, trace, evidence)
    except ValueError as exc:
        raise _classify_preflight_error(exc) from exc
    return TracePreflight(
        retained_steps=len(distilled.steps),
        source_sha256=distilled.source_sha256,
    )
