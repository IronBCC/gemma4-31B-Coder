from __future__ import annotations

import hashlib

import pytest

from teacher_platform.success_trace_distill import (
    ReplayStep,
    ReplayTrace,
    command_mutates_source,
    distill_success_path,
    replay_mutation_subsequence,
)


def _task() -> dict:
    return {
        "instance_id": "fixture__repo-1",
        "image_name": "fixture/image:latest",
        "patch": "diff --git a/src/x.py b/src/x.py\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }


@pytest.mark.parametrize(
    "command,expected",
    [
        ("cat src/x.py 2>/dev/null | head", False),
        ("cat > src/x.py <<'EOF'\nnew\nEOF", True),
        ("cat > /tmp/repro.py <<'EOF'\nnew\nEOF", False),
        ("cat > /testbed/src/x.py <<'EOF'\nnew\nEOF", True),
        ("printf '%s\\n' new > src/x.py", True),
        ("git restore src/x.py", True),
        ("perl -0pi -e 's/a/b/' src/x.py", True),
    ],
)
def test_mutation_classifier_distinguishes_null_redirection_and_writes(
    command: str,
    expected: bool,
) -> None:
    assert command_mutates_source(command) is expected


def test_mutation_replay_captures_final_patch_after_nonzero_detour(
    monkeypatch,
) -> None:
    calls = []
    patch = "diff --git a/src/x.py b/src/x.py\n"

    task = _task()

    def run(argv, *, timeout, input_text=None):
        calls.append((argv, input_text))
        if argv[:3] == ["docker", "run", "-d"]:
            return "a" * 12 + "\n", 0
        if argv[:4] == ["docker", "exec", "-i", "a" * 12]:
            assert input_text == task["patch"]
            return "", 0
        if argv[:3] == ["docker", "exec", "a" * 12]:
            if "git diff --cached" in argv[-1]:
                return patch, 0
            if "commit --no-verify" in argv[-1]:
                return "f" * 40 + "\n", 0
            return "command failed after a partial write", 1
        if argv[:3] == ["docker", "rm", "-f"]:
            return "", 0
        raise AssertionError(argv)

    monkeypatch.setattr("teacher_platform.success_trace_distill._run", run)

    result = replay_mutation_subsequence(
        task,
        (
            ReplayStep(
                "python -c \"open('src/x.py','w').write('new')\"; false",
                "failed",
                1,
                True,
            ),
        ),
    )

    assert result == hashlib.sha256(patch.encode()).hexdigest()
    assert any("git diff --cached" in call[0][-1] for call in calls)
    assert calls[1][1] == task["patch"]


def test_distiller_removes_failed_mutation_and_exact_repeated_reproducer(
    monkeypatch,
) -> None:
    trace = ReplayTrace(
        steps=(
            ReplayStep(
                "sed -i 'bad edit' src/x.py",
                "",
                0,
                True,
                assistant="Try the narrow edit.",
            ),
            ReplayStep("python3 reproduce.py", "failure", 1, False),
            ReplayStep("python3 reproduce.py", "failure", 1, False),
            ReplayStep(
                "python3 inspect_once.py",
                "inspection failed",
                1,
                False,
                assistant="Inspect one uncertain path.",
            ),
            ReplayStep(
                "python3 repair.py",
                "",
                0,
                True,
                assistant="Repair the source.",
            ),
            ReplayStep(
                "python -m pytest -q tests/test_x.py::test_fix",
                "1 passed",
                0,
                False,
                assistant="Run the focused contract test.",
            ),
        ),
        terminal_assistant="Implemented and verified.\nDONE",
    )
    target = "a" * 64
    monkeypatch.setattr(
        "teacher_platform.success_trace_distill.admission_is_exact",
        lambda evidence: evidence.get("training_admitted") is True,
    )

    def replay(_task_row, steps):
        commands = [step.command for step in steps]
        if commands in (
            ["sed -i 'bad edit' src/x.py", "python3 repair.py"],
            ["python3 repair.py"],
        ):
            return target
        return "0" * 64

    monkeypatch.setattr(
        "teacher_platform.success_trace_distill.replay_mutation_subsequence",
        replay,
    )

    distilled = distill_success_path(
        _task(),
        trace,
        {
            "training_admitted": True,
            "candidate_patch_sha256": target,
        },
    )

    commands = [step.command for step in distilled.steps]
    assert "sed -i 'bad edit' src/x.py" not in commands
    assert commands.count("python3 reproduce.py") == 0
    failed_inspection = next(
        step for step in distilled.steps
        if step.command == "python3 inspect_once.py"
    )
    assert failed_inspection.loss is False
    assert commands[-1] == "python -m pytest -q tests/test_x.py::test_fix"
    assert all(
        step.loss is True
        for step in distilled.steps
        if step.returncode == 0
    )
    assert distilled.terminal_assistant == "Implemented and verified.\nDONE"
    assert len(distilled.source_sha256) == 64


def test_distiller_rejects_nonexact_admission(monkeypatch) -> None:
    monkeypatch.setattr(
        "teacher_platform.success_trace_distill.admission_is_exact",
        lambda _evidence: False,
    )
    trace = ReplayTrace(
        steps=(
            ReplayStep(
                "python -m pytest -q tests/test_x.py::test_fix",
                "1 passed",
                0,
                False,
            ),
        ),
        terminal_assistant="DONE",
    )

    with pytest.raises(ValueError, match="exact strict admission"):
        distill_success_path(
            _task(),
            trace,
            {"candidate_patch_sha256": "a" * 64},
        )


def test_distiller_requires_passing_focused_contract_test(monkeypatch) -> None:
    monkeypatch.setattr(
        "teacher_platform.success_trace_distill.admission_is_exact",
        lambda _evidence: True,
    )
    monkeypatch.setattr(
        "teacher_platform.success_trace_distill.replay_mutation_subsequence",
        lambda _task_row, _steps: "a" * 64,
    )
    trace = ReplayTrace(
        steps=(
            ReplayStep("sed -i 's/a/b/' src/x.py", "", 0, True),
            ReplayStep("python -m compileall src", "ok", 0, False),
        ),
        terminal_assistant="DONE",
    )

    with pytest.raises(ValueError, match="focused passing test"):
        distill_success_path(
            _task(),
            trace,
            {
                "training_admitted": True,
                "candidate_patch_sha256": "a" * 64,
            },
        )
