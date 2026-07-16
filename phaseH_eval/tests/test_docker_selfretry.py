"""Unit tests for the submission self-retry docker environment (no docker)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minisweagent.exceptions import Submitted  # noqa: E402

from docker_selfretry import DockerSelfRetryEnv, MARKER  # noqa: E402


def make_env(retries=2):
    env = DockerSelfRetryEnv.__new__(DockerSelfRetryEnv)
    env._submit_retries_left = retries
    env._marker_nudges_left = 1
    return env


def test_valid_submission_still_raises():
    env = make_env()
    out = {"output": f"{MARKER}\ndiff --git a/x b/x\n+fix\n", "returncode": 0}
    with pytest.raises(Submitted):
        env._check_finished(out)


def test_empty_submission_rejected_with_recovery_message():
    env = make_env()
    out = {"output": f"{MARKER}\n\n", "returncode": 0}
    env._check_finished(out)  # no raise
    assert out["returncode"] == 1
    assert "SUBMISSION REJECTED" in out["output"]
    assert "git add -A" in out["output"]


def test_empty_submission_retries_exhausted_then_submits():
    env = make_env(retries=1)
    out1 = {"output": f"{MARKER}\n", "returncode": 0}
    env._check_finished(out1)  # consumes the retry
    out2 = {"output": f"{MARKER}\n", "returncode": 0}
    with pytest.raises(Submitted):  # falls through to parent
        env._check_finished(out2)


def test_marker_not_first_gets_one_nudge():
    env = make_env()
    out = {"output": f"diff --git a/x b/x\n{MARKER}\nmore\n", "returncode": 0}
    env._check_finished(out)
    assert "NOT as the first output line" in out["output"]
    out2 = {"output": f"noise\n{MARKER}\n", "returncode": 0}
    env._check_finished(out2)
    assert "NOT as the first output line" not in out2["output"]  # nudge spent


def test_plain_output_untouched():
    env = make_env()
    out = {"output": "just some ls output\n", "returncode": 0}
    env._check_finished(out)
    assert out["output"] == "just some ls output\n"
    assert out["returncode"] == 0


def test_nonzero_rc_marker_not_submission():
    env = make_env()
    out = {"output": f"{MARKER}\ndiff\n", "returncode": 1}
    env._check_finished(out)  # parent ignores rc!=0; no raise, no mutation
    assert "SUBMISSION REJECTED" not in out["output"]
