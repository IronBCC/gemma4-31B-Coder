from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_platform import trace_gate


def _task() -> dict:
    return {
        "instance_id": "fixture__repo-1",
        "image_name": "fixture/image:latest",
        "patch": "diff --git a/src/x.py b/src/x.py\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }


def _evidence() -> dict:
    return {
        "training_admitted": True,
        "candidate_patch_sha256": "a" * 64,
    }


def _stream(tmp_path: Path) -> Path:
    path = tmp_path / "trace.stream.jsonl"
    rows = [
        {
            "role": "assistant",
            "content": "Apply the narrow repair.",
            "tool_calls": [json.dumps({"command": "sed -i 's/a/b/' src/x.py"})],
        },
        {
            "role": "tool",
            "command": "sed -i 's/a/b/' src/x.py",
            "observation": "",
            "returncode": 0,
        },
        {
            "role": "assistant",
            "content": "Run the focused test.",
            "tool_calls": [
                json.dumps({
                    "command": (
                        "python -m pytest -q "
                        "tests/test_x.py::test_fix"
                    )
                })
            ],
        },
        {
            "role": "tool",
            "command": "python -m pytest -q tests/test_x.py::test_fix",
            "observation": "1 passed",
            "returncode": 0,
        },
        {
            "role": "assistant",
            "content": "DONE",
            "tool_calls": [],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_preflight_rejects_strict_patch_without_focused_f2p_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise ValueError(
            "distilled trace lacks a focused passing test from task contract"
        )

    monkeypatch.setattr(trace_gate, "distill_success_path", reject)

    with pytest.raises(
        trace_gate.TracePreflightError,
        match="missing_focused_test",
    ) as error:
        trace_gate.preflight_trainable_trace(
            _task(),
            "openrouter",
            _stream(tmp_path),
            _evidence(),
        )

    assert error.value.code == "missing_focused_test"


def test_preflight_rejects_trace_whose_edits_do_not_reproduce_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise ValueError(
            "raw mutation subsequence does not reproduce admitted patch"
        )

    monkeypatch.setattr(trace_gate, "distill_success_path", reject)

    with pytest.raises(
        trace_gate.TracePreflightError,
        match="raw_patch_mismatch",
    ) as error:
        trace_gate.preflight_trainable_trace(
            _task(),
            "openrouter",
            _stream(tmp_path),
            _evidence(),
        )

    assert error.value.code == "raw_patch_mismatch"


def test_strip_docker_exec_preserves_heredoc_body_and_removes_wrapper() -> None:
    command = (
        "docker exec -i e77a4f49a0e7 python3 - <<'PY'\n"
        "open('/testbed/x.py', 'w').write('x')\n"
        "PY"
    )

    normalized = trace_gate._strip_docker_exec(command)

    assert normalized.startswith("python3 - <<'PY'")
    assert "open('/testbed/x.py', 'w').write('x')" in normalized
    assert "e77a4f49a0e7" not in normalized


def test_strip_docker_exec_unquotes_double_quoted_bash_c_heredoc() -> None:
    command = (
        'docker exec cef4e7f57f50 bash -c "cd /testbed && '
        "python - <<'EOF'\n"
        "src = open('starlette/routing.py').read()\n"
        "old = '''path_regex += re.escape(path[idx:]) + \\\"\\$\\\"\n"
        "# Align with \\`Host.matches()\\` behavior'''\n"
        "assert old in src\n"
        "EOF\""
    )

    normalized = trace_gate._strip_docker_exec(command)

    assert normalized.startswith("python - <<'EOF'")
    assert 're.escape(path[idx:]) + "$"' in normalized
    assert "`Host.matches()`" in normalized
    assert '\\"' not in normalized
    assert "\\$" not in normalized
