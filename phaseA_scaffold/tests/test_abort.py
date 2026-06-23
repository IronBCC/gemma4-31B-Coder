"""T5 LOOP-ABORT UNIT PROOF (no docker, no model).

From the runbook: 8 consecutive read-only -> no-mutation; 3x identical run_tests
feedback signature -> no-fix-progress; A,B,A,B,A,B -> exact-cycle; a healthy
improving sequence -> no abort.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaffold.verify.abort import LoopAbortDetector, failure_signature, canonicalize  # noqa: E402


def test_readonly_spam_aborts():
    d = LoopAbortDetector()
    for _ in range(6):
        d.record("bash", "cd src && ls")
    ab, reason = d.should_abort()
    assert ab and reason == "loop:no-mutation"


def test_exact_cycle_aborts():
    d = LoopAbortDetector()
    for _ in range(3):
        d.record("grep", "foo bar")  # same canonical action repeated
    ab, reason = d.should_abort()
    assert ab and reason in ("loop:exact-cycle", "loop:no-mutation")


def test_stagnant_feedback_aborts():
    d = LoopAbortDetector()
    sig = failure_signature(True, ["t::a", "t::b"])
    # edit then run_tests, same failing set 3x -> thrashing without progress
    for _ in range(3):
        d.record("search_replace", "x->y")
        d.record("run_tests", "", feedback_signature=sig)
    ab, reason = d.should_abort()
    assert ab and reason == "loop:no-fix-progress"


def test_healthy_sequence_does_not_abort():
    d = LoopAbortDetector()
    # localize -> read -> edit -> run(fail A) -> edit -> run(fail B, fewer) : real progress
    d.record("localize", "module")
    d.record("read_file", "src/lib.rs")
    d.record("search_replace", "a->b")
    d.record("run_tests", "", feedback_signature=failure_signature(True, ["t::a", "t::b"]))
    d.record("search_replace", "c->d")
    d.record("run_tests", "", feedback_signature=failure_signature(True, ["t::a"]))
    ab, _ = d.should_abort()
    assert not ab


def test_mutation_breaks_readonly_run():
    d = LoopAbortDetector()
    for _ in range(4):
        d.record("ls", ".")
    d.record("search_replace", "x->y")  # a real edit resets the read-only run
    for _ in range(3):
        d.record("ls", ".")
    ab, _ = d.should_abort()
    assert not ab  # only 3 read-only since the edit, under threshold


def test_canonicalize_collapses_volatile():
    # line numbers and flag variations collapse to the same canon
    assert canonicalize("read_file", "src/lib.rs:42") == canonicalize("read_file", "src/lib.rs:99")
    assert canonicalize("bash", "cd src && ls") == canonicalize("bash", "cd src && ls -la")


def test_shell_sed_classified_mutating():
    d = LoopAbortDetector()
    for _ in range(6):
        d.record("bash", "sed -i s/a/b/ f.rs")  # mutating shell -> NOT read-only spam
    ab, reason = d.should_abort()
    assert not (ab and reason == "loop:no-mutation")
