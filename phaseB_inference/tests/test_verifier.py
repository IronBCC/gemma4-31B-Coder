import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.verifier import Candidate, rerank, select_best, HeuristicORM  # noqa: E402
from inference.best_of_n import best_of_n  # noqa: E402


def test_passing_beats_failing():
    cands = [
        Candidate("t", patch="+ fix\n", tests_passed=False),
        Candidate("t", patch="+ better fix\n", tests_passed=True),
    ]
    best = select_best("issue", cands)
    assert best.candidate.tests_passed


def test_empty_patch_scores_zero():
    orm = HeuristicORM()
    assert orm.score("x", Candidate("t", patch="   \n", tests_passed=False)) == 0.0


def test_orm_breaks_ties_among_passing():
    big = "\n".join("+line" for _ in range(200)) + "\n"
    small = "+one small change\n"
    cands = [
        Candidate("t", patch=big, tests_passed=True, n_turns=35),
        Candidate("t", patch=small, tests_passed=True, n_turns=5),
    ]
    ranked = rerank("issue", cands)
    assert ranked[0].candidate.patch == small  # smaller/faster wins the tie


def test_best_of_n_runs_samples():
    def solve_fn(seed):
        return Candidate("t", patch=f"+ attempt {seed}\n", tests_passed=(seed == 3))
    ranked = best_of_n("issue", solve_fn, n=5, max_workers=2)
    assert len(ranked) == 5
    assert ranked[0].candidate.tests_passed  # the one passing sample ranks first
