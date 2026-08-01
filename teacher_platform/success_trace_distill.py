#!/usr/bin/env python3
"""Distill exact-admitted teacher traces to replay-proven success paths."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
import subprocess
from typing import Any

if __package__:
    from .generic_trace_replay import admission_is_exact
else:  # Direct-script import via teacher_platform/teacher_platform.py.
    from generic_trace_replay import admission_is_exact


_MUTATION_RE = re.compile(
    r"(?:^|[;&|]\s*)"
    r"(?:sed\s+-i\b|perl\b(?:(?![;&|]).)*\s-[^\s;&|]*i[^\s;&|]*|"
    r"apply_patch\b|patch\s+-p\d*\b|git\s+apply\b|"
    r"git\s+(?:checkout|restore|reset)\b|"
    r"(?:cp|mv|rm|touch|install)\b|tee\s+\S+|"
    r"python(?:3)?\s+.*(?:write_text|write_bytes|open\s*\())",
    re.IGNORECASE | re.DOTALL,
)
_REDIRECT_WRITE_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|printf|echo)\b"
    r"(?:(?![;&|]).)*?(?<!\d)(?:>>|>)\s*(?P<target>[^\s;&|]+)",
    re.IGNORECASE | re.DOTALL,
)
_TEST_RUNNER_RE = re.compile(
    r"\b(?:pytest|unittest|tox|nox)\b",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATCH_CAPTURE_COMMAND = (
    "set -e; cd /testbed; "
    "idx=$(mktemp); rm -f \"$idx\"; "
    "trap 'rm -f \"$idx\"' EXIT; "
    "GIT_INDEX_FILE=\"$idx\" git read-tree HEAD; "
    "GIT_INDEX_FILE=\"$idx\" git add -A -- .; "
    "GIT_INDEX_FILE=\"$idx\" git diff --cached --binary --no-renames"
)


@dataclass(frozen=True)
class ReplayStep:
    command: str
    observation: str
    returncode: int
    mutates_source: bool
    assistant: str = ""
    loss: bool = True


@dataclass(frozen=True)
class ReplayTrace:
    steps: tuple[ReplayStep, ...]
    terminal_assistant: str


@dataclass(frozen=True)
class DistilledTrace:
    steps: tuple[ReplayStep, ...]
    terminal_assistant: str
    source_sha256: str


def command_mutates_source(command: str) -> bool:
    """Conservatively classify common source-writing shell commands."""

    if _MUTATION_RE.search(command):
        return True
    for match in _REDIRECT_WRITE_RE.finditer(command):
        target = match.group("target").strip("'\"")
        if (
            target not in {"/dev/null", "dev/null"}
            and not target.startswith("&")
            and (
                not target.startswith("/")
                or target == "/testbed"
                or target.startswith("/testbed/")
            )
        ):
            return True
    return False


def _run(
    command: list[str],
    *,
    timeout: int,
    input_text: str | None = None,
) -> tuple[str, int]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
        check=False,
    )
    return result.stdout + result.stderr, result.returncode


def replay_mutation_subsequence(
    task: Mapping[str, Any],
    steps: Sequence[ReplayStep],
) -> str:
    """Replay mutations in a clean task image and return the exact patch SHA-256."""

    image_name = task.get("image_name")
    if not isinstance(image_name, str) or not image_name:
        contract = task.get("task_contract")
        image_name = (
            contract.get("image_name")
            if isinstance(contract, Mapping)
            else None
        )
    if not isinstance(image_name, str) or not image_name:
        raise ValueError("task is missing an exact replay image")

    output, returncode = _run(
        ["docker", "run", "-d", image_name, "sleep", "infinity"],
        timeout=180,
    )
    container_id = next(
        (
            line.strip()
            for line in reversed(output.splitlines())
            if re.fullmatch(r"[0-9a-f]{12,64}", line.strip())
        ),
        "",
    )
    if returncode != 0 or not container_id:
        raise ValueError("clean mutation replay container failed to start")
    try:
        mutation_patch = task.get("patch")
        if not isinstance(mutation_patch, str) or not mutation_patch.strip():
            raise ValueError("task is missing the bug-inducing mutation patch")
        apply_output, apply_returncode = _run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "bash",
                "-c",
                "cd /testbed && git apply --whitespace=nowarn -",
            ],
            timeout=180,
            input_text=mutation_patch,
        )
        if apply_returncode != 0:
            raise ValueError(
                f"mutation replay baseline failed to apply: {apply_output[-300:]}"
            )
        commit_output, commit_returncode = _run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                (
                    "cd /testbed && git add -A -- . && "
                    "git -c user.name=teacher-platform "
                    "-c user.email=teacher-platform@invalid "
                    "commit --no-verify -m teacher-task-mutation >/dev/null && "
                    "git rev-parse HEAD"
                ),
            ],
            timeout=180,
        )
        commit = (
            commit_output.strip().splitlines()[-1]
            if commit_output.strip()
            else ""
        )
        if (
            commit_returncode != 0
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit)
        ):
            raise ValueError("mutation replay baseline commit failed")
        for step in steps:
            if not step.mutates_source:
                raise ValueError("mutation replay received a non-mutation step")
            _run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    f"cd /testbed && {step.command}",
                ],
                timeout=300,
            )
        patch, patch_returncode = _run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                _PATCH_CAPTURE_COMMAND,
            ],
            timeout=180,
        )
        if patch_returncode != 0:
            raise ValueError("clean mutation replay patch capture failed")
        return hashlib.sha256(patch.encode("utf-8")).hexdigest()
    finally:
        _run(["docker", "rm", "-f", container_id], timeout=60)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_test_ids(task: Mapping[str, Any]) -> tuple[str, ...]:
    values = task.get("FAIL_TO_PASS")
    if isinstance(values, str):
        try:
            decoded = json.loads(values)
        except json.JSONDecodeError:
            decoded = [values]
        values = decoded
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(value for value in values if isinstance(value, str) and value)


def _is_focused_passing_test(
    task: Mapping[str, Any],
    step: ReplayStep,
) -> bool:
    if step.returncode != 0 or not _TEST_RUNNER_RE.search(step.command):
        return False
    for test_id in _task_test_ids(task):
        candidates = {
            test_id,
            test_id.split("::", 1)[0],
            test_id.split(" ", 1)[0],
        }
        if any(candidate and candidate in step.command for candidate in candidates):
            return True
    return False


def _repeated_failed_detour_indices(
    steps: Sequence[ReplayStep],
) -> set[int]:
    keys = [
        (step.command.strip(), step.returncode, step.observation)
        for step in steps
    ]
    counts = Counter(
        key
        for key, step in zip(keys, steps, strict=True)
        if step.returncode != 0 and not step.mutates_source
    )
    repeated = {key for key, count in counts.items() if count >= 2}
    return {
        index
        for index, key in enumerate(keys)
        if key in repeated
    }


def distill_success_path(
    task: Mapping[str, Any],
    trace: ReplayTrace,
    evidence: Mapping[str, Any],
) -> DistilledTrace:
    """Remove only detours whose deletion preserves the exact admitted patch."""

    if not admission_is_exact(evidence):
        raise ValueError("exact strict admission evidence is required")
    target_sha256 = evidence.get("candidate_patch_sha256")
    if not isinstance(target_sha256, str) or not _SHA256_RE.fullmatch(
        target_sha256
    ):
        raise ValueError("exact candidate patch SHA-256 is required")
    terminal = trace.terminal_assistant.strip()
    if not terminal:
        raise ValueError("terminal assistant completion is required")

    mutation_indices = [
        index for index, step in enumerate(trace.steps) if step.mutates_source
    ]
    selected = list(mutation_indices)
    original_mutations = tuple(trace.steps[index] for index in selected)
    if replay_mutation_subsequence(task, original_mutations) != target_sha256:
        raise ValueError("raw mutation subsequence does not reproduce admitted patch")

    for index in mutation_indices:
        candidate_indices = [value for value in selected if value != index]
        candidate_steps = tuple(trace.steps[value] for value in candidate_indices)
        if replay_mutation_subsequence(task, candidate_steps) == target_sha256:
            selected = candidate_indices

    if replay_mutation_subsequence(
        task,
        tuple(trace.steps[index] for index in selected),
    ) != target_sha256:
        raise ValueError("distilled mutation subsequence lost admitted patch identity")

    remove = _repeated_failed_detour_indices(trace.steps)
    remove.update(set(mutation_indices) - set(selected))
    retained = tuple(
        replace(step, loss=step.returncode == 0)
        for index, step in enumerate(trace.steps)
        if index not in remove
    )
    if not any(_is_focused_passing_test(task, step) for step in retained):
        raise ValueError("distilled trace lacks a focused passing test from task contract")

    source_sha256 = _canonical_sha256({
        "task": dict(task),
        "trace": {
            "steps": [asdict(step) for step in trace.steps],
            "terminal_assistant": trace.terminal_assistant,
        },
        "evidence": dict(evidence),
        "retained_indices": [
            index for index in range(len(trace.steps)) if index not in remove
        ],
    })
    return DistilledTrace(
        steps=retained,
        terminal_assistant=terminal,
        source_sha256=source_sha256,
    )
