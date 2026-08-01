from __future__ import annotations

import importlib

import pytest


def _control() -> dict[str, int]:
    return {
        "expected": 150,
        "resolved": 90,
        "empty": 10,
        "wrong_nonempty": 50,
        "repeat_loops": 8,
    }


def _candidate() -> dict[str, int]:
    return {
        "expected": 150,
        "resolved": 91,
        "empty": 9,
        "wrong_nonempty": 50,
        "repeat_loops": 8,
    }


def _paired() -> dict[str, int]:
    return {"wins": 11, "losses": 10, "ties": 129}


def _decide(
    *,
    control: dict[str, int] | None = None,
    candidate: dict[str, int] | None = None,
    paired: dict[str, int] | None = None,
):
    module = importlib.import_module("phaseH_eval.v2p11_successor_gate")
    return module.successor_decision(
        control=control or _control(),
        candidate=candidate or _candidate(),
        paired=paired or _paired(),
    )


def test_allows_full300_only_when_correctness_and_behavior_pass() -> None:
    decision = _decide()

    assert decision["run_full300"] is True
    assert decision["reason"] == "v2.11r4 pre-full300 gate passed"
    assert all(decision["criteria"].values())


@pytest.mark.parametrize(
    ("failure", "criterion"),
    [
        ("resolved", "resolved_floor"),
        ("paired", "paired_wins_no_less_than_losses"),
        ("wrong", "wrong_nonempty_no_regression"),
        ("empty", "empty_no_regression"),
        ("loop", "repeat_loop_no_regression"),
        ("strict", "strict_behavior_improvement"),
    ],
)
def test_rejects_each_prefull_failure(
    failure: str,
    criterion: str,
) -> None:
    control = _control()
    candidate = _candidate()
    paired = _paired()
    if failure == "resolved":
        candidate.update(resolved=89, empty=10, wrong_nonempty=51)
    elif failure == "paired":
        paired.update(wins=10, losses=11)
    elif failure == "wrong":
        candidate.update(resolved=90, empty=9, wrong_nonempty=51)
        paired.update(wins=10, losses=10, ties=130)
    elif failure == "empty":
        candidate.update(resolved=90, empty=11, wrong_nonempty=49)
        paired.update(wins=10, losses=10, ties=130)
    elif failure == "loop":
        candidate["repeat_loops"] = 9
    elif failure == "strict":
        candidate.update(resolved=90, empty=10, wrong_nonempty=50)
        paired.update(wins=10, losses=10, ties=130)

    decision = _decide(
        control=control,
        candidate=candidate,
        paired=paired,
    )

    assert decision["run_full300"] is False
    assert decision["criteria"][criterion] is False
    assert criterion in decision["failed_criteria"]


def test_rejects_incomplete_or_internally_inconsistent_metrics() -> None:
    candidate = _candidate()
    candidate["wrong_nonempty"] = 49

    with pytest.raises(ValueError, match="partition"):
        _decide(candidate=candidate)


def test_rejects_noncanonical_interpolation_model_paths(tmp_path) -> None:
    module = importlib.import_module("phaseH_eval.v2p11_successor_gate")
    paths = {
        "anchor_model": module.CANONICAL_ANCHOR_MODEL,
        "source_model": module.CANONICAL_SOURCE_MODEL,
        "output_model": module.CANONICAL_OUTPUT_MODEL,
        "manifest": module.CANONICAL_OUTPUT_MODEL
        / "interpolation_manifest.json",
    }
    module._require_canonical_model_paths(paths)
    paths["source_model"] = tmp_path / "alternate-source"

    with pytest.raises(ValueError, match="source_model"):
        module._require_canonical_model_paths(paths)
