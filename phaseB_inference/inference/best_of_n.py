"""best-of-n sampling over the Phase-A agent loop + hybrid reranking (PLAN §4).

Samples N trajectories per task (optionally in parallel), converts each to a
`verifier.Candidate`, and returns the reranked list. Decoupled from Phase A via
a `solve_fn` callable so it stays unit-testable without a live model; a helper
wires in the real `phaseA_scaffold.scaffold.agent.solve_task` when available.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .verifier import Candidate, ORM, Scored, rerank

# solve_fn(seed:int) -> Candidate   (one sampled attempt for the task)
SolveFn = Callable[[int], Candidate]


def best_of_n(
    problem_statement: str,
    solve_fn: SolveFn,
    *,
    n: int = 8,
    max_workers: int = 4,
    orm: ORM | None = None,
) -> list[Scored]:
    """Run `solve_fn` n times (distinct seeds), then hybrid-rerank the candidates.

    On hard tasks, raise `n`. Sampling diversity comes from temperature/seed in
    the underlying agent call — keep temperature > 0 for best-of-n to help.
    """
    candidates: list[Candidate] = []
    if max_workers <= 1:
        for seed in range(n):
            candidates.append(_safe_solve(solve_fn, seed))
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, n)) as ex:
            futures = {ex.submit(_safe_solve, solve_fn, seed): seed for seed in range(n)}
            for fut in as_completed(futures):
                candidates.append(fut.result())
    return rerank(problem_statement, candidates, orm)


def _safe_solve(solve_fn: SolveFn, seed: int) -> Candidate:
    try:
        return solve_fn(seed)
    except Exception as e:  # noqa: BLE001 — a crashed sample is just an empty candidate
        return Candidate(instance_id=f"crashed-seed-{seed}", patch="", tests_passed=False)
        # the message is intentionally dropped into an empty patch (scores to 0)


def make_agent_solve_fn(
    *,
    repo_dir: str,
    problem_statement: str,
    test_cmd: str,
    instance_id: str = "task",
    model: str = "google/gemma-4-31B-it",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.7,
    docker_container: str | None = None,
) -> SolveFn:
    """Build a solve_fn that drives the real Phase-A agent (requires phaseA on sys.path).

    Each call runs one full agent trajectory and returns a Candidate. Temperature
    defaults to 0.7 so the N samples actually differ.
    """
    from scaffold.agent import solve_task  # phaseA_scaffold must be importable

    def _solve(seed: int) -> Candidate:
        traj = solve_task(
            repo_dir=repo_dir,
            problem_statement=problem_statement,
            test_cmd=test_cmd,
            instance_id=f"{instance_id}-{seed}",
            model=model,
            base_url=base_url,
            temperature=temperature,
            docker_container=docker_container,
        )
        n_tokens = sum(len(str(m.get("content", ""))) for m in traj.messages)
        return Candidate(
            instance_id=traj.instance_id,
            patch=traj.final_patch,
            tests_passed=traj.resolved,
            n_turns=traj.n_turns,
            n_tokens=n_tokens,
        )

    return _solve
