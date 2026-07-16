"""Unit tests for phase-2 verified reward (no docker needed — verifier mocked)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grpo_swe_edit_decision import decision_reward, extract_command  # noqa: E402
from verified_reward import VerifiedRewardComputer, f2p_invocations  # noqa: E402

EDIT_COMPLETION = 'call:bash{"command": "sed -i \'s/a/b/\' src/lib.py"}<tool_call|>'
EDIT_CTX_COMPLETION = 'call:bash{"command": "sed -i \'s/a/b/\' src/ctx.py"}<tool_call|>'
READ_COMPLETION = 'call:bash{"command": "cat src/lib.py"}<tool_call|>'
NO_CALL = "just thinking out loud, no call"


class FakeVerifier:
    def __init__(self, tier=0.8, exists=True, raise_exc=False):
        self.tier = tier
        self.exists = exists
        self.raise_exc = raise_exc
        self.calls = []

    def state_exists(self, tag):
        return self.exists

    def verify(self, tag, cmd, f2p, test_patch=""):
        self.calls.append((tag, cmd, test_patch))
        if self.raise_exc:
            raise RuntimeError("docker exploded")
        return self.tier


def make(verifier):
    return VerifiedRewardComputer(decision_reward, extract_command,
                                  verifier=verifier, workers=2, group_timeout=5,
                                  log=lambda *a, **k: None)


def test_unverified_rows_capped_at_06():
    comp = make(FakeVerifier())
    # parse-only 1.0 (edit touching ctx file) must clamp to 0.6 on unverified rows
    out = comp.compute([EDIT_CTX_COMPLETION], [{"src/ctx.py"}], [None], [[]])
    assert out == [0.6]


def test_parse_tiers_preserved_below_edit():
    comp = make(FakeVerifier())
    out = comp.compute([NO_CALL, READ_COMPLETION], [set(), set()], [None, None], [[], []])
    assert out == [0.0, 0.2]


def test_verified_execution_lifts_to_08():
    v = FakeVerifier(tier=0.8)
    out = make(v).compute([EDIT_COMPLETION], [set()], ["rlvr-state:x"], [[]])
    assert out == [0.8]
    assert v.calls and v.calls[0][1].startswith("sed -i")


def test_verified_f2p_pass_lifts_to_10():
    out = make(FakeVerifier(tier=1.0)).compute(
        [EDIT_COMPLETION], [set()], ["rlvr-state:x"], [["t (m.C)"]])
    assert out == [1.0]


def test_failed_execution_stays_at_parse_tier():
    out = make(FakeVerifier(tier=0.6)).compute(
        [EDIT_COMPLETION], [set()], ["rlvr-state:x"], [[]])
    assert out == [0.6]


def test_docker_error_degrades_not_crashes():
    comp = make(FakeVerifier(raise_exc=True))
    out = comp.compute([EDIT_COMPLETION], [set()], ["rlvr-state:x"], [[]])
    assert out == [0.6]
    assert comp.stats["degraded"] == 1


def test_cache_miss_behaves_unverified():
    v = FakeVerifier(exists=False)
    out = make(v).compute([EDIT_CTX_COMPLETION], [{"src/ctx.py"}], ["rlvr-state:x"], [[]])
    assert out == [0.6]
    assert not v.calls


def test_read_completion_never_verified():
    v = FakeVerifier()
    out = make(v).compute([READ_COMPLETION], [set()], ["rlvr-state:x"], [[]])
    assert out == [0.2]
    assert not v.calls


def test_test_patch_reaches_verifier():
    v = FakeVerifier(tier=1.0)
    out = make(v).compute([EDIT_COMPLETION], [set()], ["rlvr-state:x"], [["t (m.C)"]],
                          ["diff --git a/tests/x.py b/tests/x.py"])
    assert out == [1.0]
    assert v.calls[0][2].startswith("diff --git")


def test_f2p_invocations_unittest_style():
    cmds = f2p_invocations(["test_reload (tornado.test.autoreload_test.AutoreloadTest)"])
    assert cmds == ["python -m unittest -q tornado.test.autoreload_test.AutoreloadTest.test_reload"]


def test_f2p_invocations_pytest_style_and_mixed():
    cmds = f2p_invocations([
        "tests/test_x.py::TestA::test_b",
        "test_m (pkg.mod.Cls)",
        "garbage entry with spaces only",
    ])
    assert any(c.startswith("python -m unittest") for c in cmds)
    assert any("pytest" in c and "tests/test_x.py::TestA::test_b" in c for c in cmds)
    assert len(cmds) == 2
