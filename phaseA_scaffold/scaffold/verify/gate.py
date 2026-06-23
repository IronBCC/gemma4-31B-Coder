"""T2 — the id-level resolve gate (the core correctness fix over Phase-0).

RESOLVED iff: build succeeded AND every FAIL_TO_PASS id is now passing AND the
task actually shipped FAIL_TO_PASS ids AND no PASS_TO_PASS id regressed. Everything
else is an explicit non-resolved bucket. `returncode` is NOT consulted here — it is
advisory only and lives on VerifyResult for diagnostics. This replaces
test_loop.run_tests' `returncode==0 and not failures`, which could mark a patch
resolved when it broke an unrelated test or when the target test was already green.
"""
from __future__ import annotations

from .outcome import Outcome, VerifyResult
from .parsers import ParsedTests


def decide(
    parsed: ParsedTests,
    f2p_ids: list[str],
    p2p_ids: list[str],
    *,
    lang: str = "python",
    timed_out: bool = False,
    infra: bool = False,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> VerifyResult:
    f2p = set(f2p_ids or [])
    p2p = set(p2p_ids or [])
    base = dict(
        lang=lang,
        build_ok=parsed.build_ok,
        all_passed_ids=set(parsed.passed_ids),
        all_failed_ids=set(parsed.failed_ids),
        returncode=returncode,
        timed_out=timed_out,
        infra_flake=infra,
        build_diag=parsed.build_diag,
        raw_stdout=stdout,
        raw_stderr=stderr,
    )

    # 1. infra flake — retryable, never a code verdict
    if infra:
        return VerifyResult(outcome=Outcome.INFRA_FLAKE, **base)

    # 2. build/compile/link failure — distinct feedback, p2p untested
    if not parsed.build_ok:
        return VerifyResult(outcome=Outcome.BUILD_FAILED, f2p_fail=set(f2p), **base)

    # 3. task shipped no target tests — never silently pass
    if not f2p:
        vr = VerifyResult(outcome=Outcome.NOT_RESOLVED, **base)
        vr.build_diag = vr.build_diag or "task has no FAIL_TO_PASS ids — cannot verify a fix"
        return vr

    # 4. timeout after building — tests did not complete
    if timed_out:
        return VerifyResult(outcome=Outcome.TESTS_FAILED, f2p_fail=set(f2p), **base)

    # 5. id-level reconciliation
    passed = parsed.passed_ids
    f2p_pass = f2p & passed
    f2p_fail = f2p - passed
    p2p_pass = p2p & passed
    p2p_regressed = p2p - passed  # a previously-passing test that is no longer passing

    outcome = Outcome.RESOLVED if (not f2p_fail and not p2p_regressed) else Outcome.TESTS_FAILED
    return VerifyResult(
        outcome=outcome,
        f2p_pass=f2p_pass,
        f2p_fail=f2p_fail,
        p2p_pass=p2p_pass,
        p2p_regressed=p2p_regressed,
        **base,
    )
