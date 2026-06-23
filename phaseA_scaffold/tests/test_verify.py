"""Unit tests for the T0-T2 verification harness (no docker, no model, no compute).

Covers the runbook's PARSER UNIT PROOF and GATE UNIT PROOF — the correctness core
that every Rust/C++/Python baseline number will depend on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaffold.verify.outcome import Outcome, VerifyResult  # noqa: E402
from scaffold.verify.langspec import LANG_SPECS, TURN_CAP  # noqa: E402
from scaffold.verify import parsers, gate  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


# ----------------------------------------------------------------- T0 contracts
def test_langspecs_present_and_timeouts_scale():
    assert set(LANG_SPECS) == {"python", "rust", "cpp"}
    assert LANG_SPECS["python"].per_turn_timeout_s == 300
    assert LANG_SPECS["rust"].per_turn_timeout_s == 900
    assert LANG_SPECS["cpp"].per_turn_timeout_s == 1200
    assert TURN_CAP == 90  # turn cap is language-independent


def test_feedback_distinct_per_outcome():
    # every Outcome produces a non-empty, distinct-ish feedback string
    seen = set()
    for o in Outcome:
        fb = VerifyResult(outcome=o, lang="rust", build_diag="boom", returncode=125).feedback()
        assert fb and isinstance(fb, str)
        seen.add(fb.split("\n", 1)[0])
    assert len(seen) == len(list(Outcome))  # first lines all differ


# -------------------------------------------------------------- T1 cargo parser
def test_parse_cargo_pass_fail():
    pt = parsers.parse_cargo_ndjson((FIX / "cargo_1pass_1fail.ndjson").read_text())
    assert pt.build_ok is True
    assert pt.passed_ids == {"tests::test_decode_ok"}
    assert pt.failed_ids == {"tests::test_highlight_regression"}


def test_parse_cargo_build_error():
    pt = parsers.parse_cargo_ndjson((FIX / "cargo_build_error.ndjson").read_text())
    assert pt.build_ok is False
    assert "E0308" in pt.build_diag or "mismatched types" in pt.build_diag
    assert not pt.passed_ids and not pt.failed_ids


def test_parse_cargo_malformed_no_raise():
    pt = parsers.parse_cargo_ndjson("not json at all\n{truncated")
    assert pt.build_ok is True  # no explicit build error seen; empty ids
    assert not pt.passed_ids and not pt.failed_ids


# -------------------------------------------------------------- T1 ctest parser
def test_parse_ctest_pass_fail():
    pt = parsers.parse_ctest_junit("", "", str(FIX / "ctest_1pass_1fail.junit.xml"))
    assert pt.build_ok is True
    assert pt.passed_ids == {"parse_valid_json"}
    assert pt.failed_ids == {"parse_regression_unicode"}


def test_parse_ctest_missing_artifact_is_build_failed():
    pt = parsers.parse_ctest_junit("", "", None)
    assert pt.build_ok is False  # no junit => build/configure broke


def test_parse_ctest_malformed_no_raise(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuite><testcase ")  # truncated
    pt = parsers.parse_ctest_junit("", "", str(bad))
    assert pt.build_ok is False


# ------------------------------------------------------------- T1 pytest parser
def test_parse_pytest_summary():
    out = "PASSED tests/test_a.py::test_ok\nFAILED tests/test_b.py::test_bad - AssertionError\n"
    pt = parsers.parse_pytest(out)
    assert pt.passed_ids == {"tests/test_a.py::test_ok"}
    assert pt.failed_ids == {"tests/test_b.py::test_bad"}


def test_parse_pytest_collection_error_is_build_failed():
    pt = parsers.parse_pytest("errors during collection\n")
    assert pt.build_ok is False


def test_dispatch_unknown_lang_raises():
    try:
        parsers.parse("ruby", "")
    except ValueError as e:
        assert "ruby" in str(e)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------- T2 gate
def _pt(passed, failed, build_ok=True):
    return parsers.ParsedTests(passed_ids=set(passed), failed_ids=set(failed), build_ok=build_ok)


def test_gate_resolved():
    r = gate.decide(_pt(["f1", "p1", "p2"], []), ["f1"], ["p1", "p2"], lang="rust")
    assert r.outcome is Outcome.RESOLVED and r.resolved


def test_gate_f2p_still_failing():
    r = gate.decide(_pt(["p1"], ["f1"]), ["f1"], ["p1"], lang="rust")
    assert r.outcome is Outcome.TESTS_FAILED
    assert r.f2p_fail == {"f1"}


def test_gate_p2p_regression_caught():
    # all f2p pass, but a p2p id is no longer passing -> NOT resolved
    r = gate.decide(_pt(["f1"], []), ["f1"], ["p1"], lang="cpp")
    assert r.outcome is Outcome.TESTS_FAILED
    assert r.p2p_regressed == {"p1"}


def test_gate_build_failed_overrides_returncode_zero():
    r = gate.decide(_pt([], [], build_ok=False), ["f1"], [], lang="rust", returncode=0)
    assert r.outcome is Outcome.BUILD_FAILED  # returncode 0 must NOT save it


def test_gate_empty_f2p_never_silent_pass():
    r = gate.decide(_pt(["whatever"], []), [], [], lang="python")
    assert r.outcome is Outcome.NOT_RESOLVED
    assert "no FAIL_TO_PASS" in r.build_diag


def test_gate_infra_flake():
    r = gate.decide(_pt([], []), ["f1"], [], lang="cpp", infra=True, returncode=137)
    assert r.outcome is Outcome.INFRA_FLAKE


def test_gate_end_to_end_cargo_fixture():
    # parse a real cargo fixture, then gate with the failing test as the f2p target
    pt = parsers.parse_cargo_ndjson((FIX / "cargo_1pass_1fail.ndjson").read_text())
    r = gate.decide(pt, ["tests::test_highlight_regression"], ["tests::test_decode_ok"], lang="rust")
    assert r.outcome is Outcome.TESTS_FAILED  # the f2p target is still failing
    # and if the agent fixed it (move it to passed), it resolves:
    pt.passed_ids.add("tests::test_highlight_regression")
    pt.failed_ids.discard("tests::test_highlight_regression")
    r2 = gate.decide(pt, ["tests::test_highlight_regression"], ["tests::test_decode_ok"], lang="rust")
    assert r2.outcome is Outcome.RESOLVED
