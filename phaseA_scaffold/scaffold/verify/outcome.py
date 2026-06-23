"""T0 — canonical outcome model + VerifyResult data contract.

The single source of truth for "what happened when we verified a patch". Every
language's parser+gate funnels into one VerifyResult so agent.py and the baseline
driver are language-agnostic.

Why distinct buckets (not a bool): for Rust/C++ a compile/link error must give the
agent DIFFERENT feedback than a red test, and must never be mistaken for RESOLVED
because a stale cached build happened to return 0. BUILD_FAILED vs TESTS_FAILED vs
RESOLVED is the core correctness fix over Phase-0's returncode heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    RESOLVED = "RESOLVED"            # all FAIL_TO_PASS flipped to pass AND no PASS_TO_PASS regressed
    TESTS_FAILED = "TESTS_FAILED"    # built/ran, but some target test still failing (or timed out)
    BUILD_FAILED = "BUILD_FAILED"    # compile/link error — covers COMPILE_FAILED + LINK_FAILED
    INFRA_FLAKE = "INFRA_FLAKE"      # docker exit 125/137, ld-killed, OOM, no-space — retry, never a code verdict
    NOT_RESOLVED = "NOT_RESOLVED"    # empty/no-op patch, or task shipped no f2p ids — never a silent pass


def _tail(text: str, n: int) -> str:
    n = max(n, 200)
    return text[-n:] if len(text) > n else text


@dataclass
class VerifyResult:
    """Result of verifying one patch on one instance. Superset of test_loop.TestRun."""
    outcome: Outcome
    lang: str = "python"
    build_ok: bool = False
    f2p_pass: set[str] = field(default_factory=set)
    f2p_fail: set[str] = field(default_factory=set)
    p2p_pass: set[str] = field(default_factory=set)
    p2p_regressed: set[str] = field(default_factory=set)
    all_passed_ids: set[str] = field(default_factory=set)
    all_failed_ids: set[str] = field(default_factory=set)
    returncode: int = 0
    timed_out: bool = False
    infra_flake: bool = False
    build_diag: str = ""        # compiler/linker tail, shown only on BUILD_FAILED
    raw_stdout: str = ""
    raw_stderr: str = ""

    @property
    def resolved(self) -> bool:
        return self.outcome is Outcome.RESOLVED

    def feedback(self, max_chars: int = 4000) -> str:
        """Compact, model-facing summary spliced into the next turn.

        Distinct per bucket so the agent gets the RIGHT signal: a compile error
        shows the diagnostic tail (not test ids); a red test shows which f2p are
        still failing and which p2p regressed.
        """
        o = self.outcome
        if o is Outcome.RESOLVED:
            return "ALL TARGET TESTS PASS (FAIL_TO_PASS flipped, no PASS_TO_PASS regression)."
        if o is Outcome.BUILD_FAILED:
            return f"BUILD FAILED ({self.lang}) — fix the compile/link error before tests can run:\n{_tail(self.build_diag or self.raw_stderr, max_chars - 80)}"
        if o is Outcome.INFRA_FLAKE:
            return f"INFRASTRUCTURE FLAKE (returncode {self.returncode}) — not a code failure; this attempt will be retried."
        if o is Outcome.NOT_RESOLVED:
            return "NO USABLE PATCH — empty/no-op change or task has no target tests; nothing verified."
        # TESTS_FAILED
        if self.timed_out:
            return f"TESTS TIMED OUT after the cap.\n{_tail(self.raw_stdout, 1500)}"
        head = f"TESTS FAILED — {len(self.f2p_fail)} target test(s) still failing"
        if self.p2p_regressed:
            head += f"; {len(self.p2p_regressed)} previously-passing test(s) REGRESSED"
        head += ":\n"
        still = "\n".join(f"  - [f2p] {t}" for t in sorted(self.f2p_fail)[:20])
        regr = "\n".join(f"  - [REGRESSED] {t}" for t in sorted(self.p2p_regressed)[:10])
        body = "\n".join(x for x in (still, regr) if x)
        tail = _tail(self.raw_stdout + "\n" + self.raw_stderr, max(200, max_chars - len(head) - len(body)))
        return f"{head}{body}\n\n--- output tail ---\n{tail}"
