"""Hybrid verifier: execution-based signal + execution-free ORM (PLAN §4).

Each candidate trajectory gets two scores:
  - execution score: did the task's tests pass? (the strong, trusted signal)
  - ORM score: an execution-free outcome reward model's estimate in [0, 1]
    (useful when tests are flaky/absent or to break ties among passing patches).

The hybrid score prefers test-passing candidates, then ranks by ORM. Train the
ORM on Phase-D collected trajectories; until then, `HeuristicORM` gives a usable
non-trained fallback so the reranker is runnable end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Candidate:
    """One sampled solution attempt (mirrors phaseA agent.Trajectory's relevant fields)."""

    instance_id: str
    patch: str
    tests_passed: bool
    n_turns: int = 0
    n_tokens: int = 0
    empty_patch: bool = field(init=False, default=False)

    def __post_init__(self):
        self.empty_patch = not self.patch.strip()


class ORM(Protocol):
    """Execution-free outcome reward model: score a candidate in [0, 1]."""

    def score(self, problem_statement: str, candidate: Candidate) -> float: ...


class HeuristicORM:
    """Non-trained fallback ORM.

    Cheap proxy until a real ORM is trained on Phase-D traces: penalize empty
    patches, mildly prefer focused patches and shorter solve paths. Bounded [0,1].
    """

    def score(self, problem_statement: str, candidate: Candidate) -> float:
        if candidate.empty_patch:
            return 0.0
        added = sum(
            1 for ln in candidate.patch.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")
        )
        # prefer small/medium diffs (large diffs correlate with shotgun edits)
        size_term = 1.0 if added <= 40 else max(0.3, 40 / added)
        turn_term = 1.0 if candidate.n_turns <= 20 else max(0.5, 20 / candidate.n_turns)
        return round(0.5 * size_term + 0.5 * turn_term, 4)


@dataclass
class Scored:
    candidate: Candidate
    exec_score: float
    orm_score: float
    hybrid: float


def hybrid_score(exec_passed: bool, orm: float, *, exec_weight: float = 1.0) -> float:
    """Execution dominates; ORM ranks within a tier. exec_weight scales the gap."""
    return exec_weight * (1.0 if exec_passed else 0.0) + orm


def rerank(
    problem_statement: str,
    candidates: list[Candidate],
    orm: ORM | None = None,
    *,
    exec_weight: float = 1.0,
) -> list[Scored]:
    """Rank candidates by hybrid score (highest first). Stable within ties."""
    orm = orm or HeuristicORM()
    scored = [
        Scored(
            candidate=c,
            exec_score=1.0 if c.tests_passed else 0.0,
            orm_score=orm.score(problem_statement, c),
            hybrid=hybrid_score(c.tests_passed, orm.score(problem_statement, c), exec_weight=exec_weight),
        )
        for c in candidates
    ]
    scored.sort(key=lambda s: s.hybrid, reverse=True)
    return scored


def select_best(problem_statement: str, candidates: list[Candidate], orm: ORM | None = None) -> Scored | None:
    ranked = rerank(problem_statement, candidates, orm)
    return ranked[0] if ranked else None
