from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_effective_outcomes_keep_original_wins_and_require_retries_for_failures() -> None:
    try:
        module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    except ModuleNotFoundError:
        module = None
    select_effective_outcomes = (
        getattr(module, "select_effective_outcomes", None) if module is not None else None
    )
    assert callable(select_effective_outcomes), "final report outcome selection is not implemented"

    original = [
        _row("kept-win", resolved=True, patch_len=20, assistant_steps=39),
        _row("retry-win", resolved=False, patch_len=0, assistant_steps=20),
        _row("kept-loss", resolved=False, patch_len=10, assistant_steps=20),
        _row("missing-retry", resolved=False, patch_len=0, assistant_steps=20),
        _row("retry-loss", resolved=False, patch_len=10, assistant_steps=39),
    ]
    retest = [
        _row("retry-win", resolved=True, patch_len=30, assistant_steps=60),
        _row("retry-loss", resolved=False, patch_len=0, assistant_steps=120),
    ]

    selected = select_effective_outcomes(original, retest)

    assert {
        instance_id: (
            row["source_attempt"],
            row["resolved"],
            row["usable_outcome"],
        )
        for instance_id, row in selected.items()
    } == {
        "kept-win": ("original", True, True),
        "retry-win": ("retest", True, True),
        "kept-loss": ("missing_retest", False, False),
        "missing-retry": ("missing_retest", False, False),
        "retry-loss": ("retest", False, True),
    }


def test_effective_outcomes_accept_fresh_retry_for_non_clamp_failure() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    original = [
        _row("early-loss", resolved=False, patch_len=20, assistant_steps=10),
    ]
    retest = [
        _row("early-loss", resolved=True, patch_len=30, assistant_steps=50),
    ]

    selected = module.select_effective_outcomes(original, retest)

    assert selected["early-loss"]["source_attempt"] == "retest"
    assert selected["early-loss"]["resolved"] is True
    assert selected["early-loss"]["usable_outcome"] is True


def test_fixed_harness_comparison_intersects_usable_fixed_attempts_only() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    compare_fixed_harness_outcomes = getattr(
        module,
        "compare_fixed_harness_outcomes",
        None,
    )
    assert callable(compare_fixed_harness_outcomes), "fixed-harness pairing is not implemented"

    left = {
        "old": _selected("old", resolved=True, fixed_harness=False),
        "b": _selected("b", resolved=True, fixed_harness=True),
        "c": _selected("c", resolved=False, fixed_harness=True),
        "f": _selected("f", resolved=False, fixed_harness=True),
        "left-only": _selected("left-only", resolved=False, fixed_harness=True),
        "unusable": _selected(
            "unusable",
            resolved=False,
            fixed_harness=True,
            usable_outcome=False,
        ),
    }
    right = {
        "old": _selected("old", resolved=False, fixed_harness=True),
        "b": _selected("b", resolved=False, fixed_harness=True),
        "c": _selected("c", resolved=True, fixed_harness=True),
        "f": _selected("f", resolved=True, fixed_harness=True),
        "right-only": _selected("right-only", resolved=True, fixed_harness=True),
    }

    report = compare_fixed_harness_outcomes(left, right)

    assert report["pair_ids"] == ["b", "c", "f"]
    assert report["n"] == 3
    assert report["left_fixed_usable"] == 4
    assert report["right_fixed_usable"] == 5
    assert report["left_only_excluded"] == 1
    assert report["right_only_excluded"] == 2
    assert report["contingency"] == {
        "both_resolved": 0,
        "left_only": 1,
        "right_only": 2,
        "neither": 0,
    }
    assert report["right_better_one_sided_p"] == 0.5
    assert report["two_sided_p"] == 1.0


def test_empty_causes_remain_exclusive_and_keep_unknown_missing_visible() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    classify_empty_outcomes = getattr(module, "classify_empty_outcomes", None)
    assert callable(classify_empty_outcomes), "empty-cause classification is not implemented"
    outcomes = {
        "model": _empty_outcome("model"),
        "forced": _empty_outcome("forced", guarded_forced_submit=True),
        "docker": _empty_outcome(
            "docker",
            trajectory_present=False,
            pull_failed=True,
            usable_outcome=False,
        ),
        "unknown": _empty_outcome(
            "unknown",
            trajectory_present=False,
            usable_outcome=False,
        ),
        "patch": {
            **_empty_outcome("patch"),
            "patch_len": 50,
        },
    }

    report = classify_empty_outcomes(outcomes)

    assert report == {
        "total": 4,
        "model": ["model"],
        "harness_forced": ["forced"],
        "docker_failed": ["docker"],
        "unknown_missing": ["unknown"],
    }


def test_step_distribution_uses_assistant_turns_and_nearest_rank_percentiles() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    summarize_step_counts = getattr(module, "summarize_step_counts", None)
    assert callable(summarize_step_counts), "step distribution is not implemented"
    outcomes = {
        str(steps): {
            **_selected(str(steps), resolved=False, fixed_harness=True),
            "assistant_steps": steps,
        }
        for steps in (14, 39, 60, 120)
    }
    outcomes["old"] = {
        **_selected("old", resolved=True, fixed_harness=False),
        "assistant_steps": 1,
    }

    report = summarize_step_counts(outcomes)

    assert report == {
        "n": 4,
        "min": 14,
        "p25": 14,
        "median": 49.5,
        "p75": 60,
        "p90": 120,
        "max": 120,
        "ge39": 3,
        "gt39": 2,
        "ge120": 1,
    }


def test_fixed_panel_selection_is_order_independent_and_hash_preregistered() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    select_fixed_panel = getattr(module, "select_fixed_panel", None)
    assert callable(select_fixed_panel), "fixed-panel selection is not implemented"

    assert select_fixed_panel(
        ["d", "b", "a", "c"],
        size=2,
        namespace="panel-v1",
    ) == ["a", "d"]


def test_attempt_loader_keeps_missing_predictions_and_raw_evidence_explicit(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    load_attempt_artifacts = getattr(module, "load_attempt_artifacts", None)
    assert callable(load_attempt_artifacts), "artifact ingestion is not implemented"

    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            {
                "a": {"instance_id": "a", "model_patch": "diff --git a/a b/a\n"},
                "b": {"instance_id": "b", "model_patch": ""},
            }
        )
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "resolved_ids": ["a"],
                "unresolved_ids": [],
                "error_ids": [],
            }
        )
    )
    trajectory_paths = []
    for instance_id in ("a", "b"):
        trajectory = tmp_path / f"{instance_id}.traj.json"
        first_guard = (
            {"guarded_forced_command": "git diff -- ."}
            if instance_id == "a"
            else {}
        )
        final_guard = (
            {
                "guarded_forced_command": (
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
                )
            }
            if instance_id == "b"
            else {}
        )
        trajectory.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "messages": [
                        {
                            "role": "assistant",
                            "extra": {"response": first_guard},
                        },
                        {"role": "assistant", "extra": {"response": final_guard}},
                    ],
                }
            )
        )
        trajectory_paths.append(trajectory)

    rows, stats = load_attempt_artifacts(
        expected_ids=["a", "b", "c"],
        preds_path=preds_path,
        report_paths=[report_path],
        trajectory_paths=trajectory_paths,
        attempt_kind="retest",
        run_id="fixture",
        fixed_harness=True,
        pull_failed_ids={"c"},
    )

    assert stats == {
        "expected": 3,
        "preds": 2,
        "traj_files": 2,
        "pull_failed": 1,
        "resolved": 1,
        "usable_outcomes": 2,
    }
    assert rows["a"]["resolved"] is True
    assert rows["a"]["patch_len"] == 19
    assert rows["a"]["assistant_steps"] == 2
    assert rows["a"]["guarded_forced"] is True
    assert rows["a"]["guarded_forced_submit"] is False
    assert rows["b"]["resolved"] is False
    assert rows["b"]["score_present"] is True
    assert rows["b"]["guarded_forced"] is True
    assert rows["b"]["guarded_forced_submit"] is True
    assert rows["c"]["prediction_present"] is False
    assert rows["c"]["pull_failed"] is True
    assert rows["c"]["infra_error"] == "docker_pull_failed"
    assert rows["c"]["usable_outcome"] is False


def test_attempt_loader_counts_proven_patch_apply_error_as_usable_loss(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            {
                "bad-patch": {
                    "instance_id": "bad-patch",
                    "model_patch": "malformed diff",
                }
            }
        )
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": ["bad-patch"],
            }
        )
    )
    trajectory = tmp_path / "bad-patch.traj.json"
    trajectory.write_text(
        json.dumps(
            {
                "instance_id": "bad-patch",
                "messages": [{"role": "assistant", "extra": {}}],
            }
        )
    )

    rows, stats = module.load_attempt_artifacts(
        expected_ids=["bad-patch"],
        preds_path=preds_path,
        report_paths=[report_path],
        trajectory_paths=[trajectory],
        attempt_kind="retest",
        run_id="fixture",
        fixed_harness=True,
        model_failure_ids={"bad-patch"},
    )

    assert rows["bad-patch"]["resolved"] is False
    assert rows["bad-patch"]["score_present"] is True
    assert rows["bad-patch"]["infra_error"] is None
    assert rows["bad-patch"]["usable_outcome"] is True
    assert stats["usable_outcomes"] == 1


def test_later_score_and_trajectory_attempt_replace_older_status(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    old_report = tmp_path / "b00-report.json"
    latest_report = tmp_path / "b12-report.json"
    old_report.write_text(
        json.dumps({"resolved_ids": ["case"], "unresolved_ids": [], "error_ids": []})
    )
    latest_report.write_text(
        json.dumps({"resolved_ids": [], "unresolved_ids": ["case"], "error_ids": []})
    )
    old_trajectory = tmp_path / "b00.traj.json"
    latest_trajectory = tmp_path / "b12.traj.json"
    old_trajectory.write_text(
        json.dumps(
            {
                "instance_id": "case",
                "messages": [{"role": "assistant", "content": "old"}],
            }
        )
    )
    latest_trajectory.write_text(
        json.dumps(
            {
                "instance_id": "case",
                "messages": [
                    {"role": "assistant", "content": "latest-1"},
                    {"role": "assistant", "content": "latest-2"},
                ],
            }
        )
    )

    resolved, unresolved, errors = module._load_score_ids(
        [old_report, latest_report]
    )
    trajectories = module._load_trajectory_map(
        [old_trajectory, latest_trajectory]
    )

    assert resolved == set()
    assert unresolved == {"case"}
    assert errors == set()
    assert trajectories["case"]["assistant_steps"] == 2
    assert trajectories["case"]["trajectory_path"] == str(latest_trajectory)


def test_docker_evidence_parsers_map_images_and_only_exit_125_127_blocks() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    instance_id_from_image = getattr(module, "instance_id_from_image", None)
    docker_failed_ids_from_log = getattr(module, "docker_failed_ids_from_log", None)
    assert callable(instance_id_from_image), "pull-failure image mapping is not implemented"
    assert callable(docker_failed_ids_from_log), "Docker log classification is not implemented"

    assert instance_id_from_image(
        "swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-13241:latest"
    ) == "scikit-learn__scikit-learn-13241"
    log = """
minisweagent: ERROR: Error processing instance repo__repo-1: Command
['docker', 'run'] returned non-zero exit status 125.
Traceback...
minisweagent: ERROR: Error processing instance repo__repo-2: Command
['docker', 'run'] returned non-zero exit status 127.
Traceback...
minisweagent: ERROR: Error processing instance repo__repo-3: model request failed.
Traceback...
"""
    assert docker_failed_ids_from_log(log) == {"repo__repo-1", "repo__repo-2"}


def test_model_failure_parser_only_accepts_patch_apply_failures(tmp_path: Path) -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    parser = getattr(module, "model_failure_ids_from_score_logs", None)
    assert callable(parser), "model patch-failure classification is not implemented"

    malformed = tmp_path / "bad-patch" / "run_instance.log"
    malformed.parent.mkdir()
    malformed.write_text("Patch Apply Failed: patch: **** malformed patch at line 15")
    infrastructure = tmp_path / "score-infra" / "run_instance.log"
    infrastructure.parent.mkdir()
    infrastructure.write_text("Error: Docker daemon unavailable")

    assert parser([malformed, infrastructure]) == {"bad-patch"}


def test_recovery_summary_uses_id_union_and_exposes_missing_retries() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    summarize_recovery = getattr(module, "summarize_recovery", None)
    assert callable(summarize_recovery), "recovery summary is not implemented"
    original = [
        _row("old-win", resolved=True, patch_len=10, assistant_steps=20),
        _row("new-win", resolved=False, patch_len=0, assistant_steps=39),
        _row("new-loss", resolved=False, patch_len=10, assistant_steps=20),
        _row("missing", resolved=False, patch_len=0, assistant_steps=39),
    ]
    retest = [
        _row("new-win", resolved=True, patch_len=20, assistant_steps=60),
        _row("new-loss", resolved=False, patch_len=0, assistant_steps=120),
    ]

    report = summarize_recovery(original, retest)

    assert report["original_resolved_ids"] == ["old-win"]
    assert report["retest_resolved_ids"] == ["new-win"]
    assert report["corrected_resolved_ids"] == ["new-win", "old-win"]
    assert report["original_resolved"] == 1
    assert report["retest_expected"] == 3
    assert report["retest_usable"] == 2
    assert report["retest_resolved"] == 1
    assert report["corrected_recovery_lower_bound"] == 2
    assert report["missing_retest_ids"] == ["missing"]


def test_recovery_segments_measure_each_checkpoints_own_resampling_floor() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    summarize_segments = getattr(module, "summarize_recovery_segments", None)
    assert callable(summarize_segments), "segmented recovery summary is not implemented"
    rows = [
        _row("suspect-win", resolved=True, patch_len=12, assistant_steps=60),
        _row("suspect-loss", resolved=False, patch_len=0, assistant_steps=120),
        _row("added-win", resolved=True, patch_len=12, assistant_steps=50),
        _row("added-loss", resolved=False, patch_len=8, assistant_steps=45),
    ]
    rows[-1]["infra_error"] = "score_error"

    report = summarize_segments(
        rows,
        all_unresolved_ids=[
            "suspect-win",
            "suspect-loss",
            "added-win",
            "added-loss",
            "added-missing",
        ],
        clamp_suspect_ids=["suspect-win", "suspect-loss"],
    )

    assert report["clamp_suspect"] == {
        "expected": 2,
        "scored": 2,
        "coverage": 1.0,
        "resolved": 1,
        "rate": 0.5,
        "scored_ids": ["suspect-loss", "suspect-win"],
        "resolved_ids": ["suspect-win"],
        "missing_ids": [],
    }
    assert report["added_non_suspect"] == {
        "expected": 3,
        "scored": 1,
        "coverage": 1 / 3,
        "resolved": 1,
        "rate": 1.0,
        "scored_ids": ["added-win"],
        "resolved_ids": ["added-win"],
        "missing_ids": ["added-loss", "added-missing"],
    }
    assert report["resampling_noise_floor"] == {
        "scored": 1,
        "resolved": 1,
        "rate": 1.0,
    }
    assert report["suspect_excess_over_floor_estimate"] == {
        "rate_difference": -0.5,
        "percentage_points": -50.0,
        "expected_resolved_at_floor": 2.0,
        "excess_resolved": -1.0,
    }
    assert report["total_excess_over_floor_estimate"] == {
        "scored": 3,
        "resolved": 2,
        "expected_resolved_at_floor": 3.0,
        "excess_resolved": -1.0,
    }

    other_checkpoint = summarize_segments(
        [
            _row("s1", resolved=True, patch_len=10, assistant_steps=60),
            _row("n1", resolved=False, patch_len=10, assistant_steps=50),
        ],
        all_unresolved_ids=["s1", "n1"],
        clamp_suspect_ids=["s1"],
    )
    assert other_checkpoint["resampling_noise_floor"]["rate"] == 0.0
    assert other_checkpoint["suspect_excess_over_floor_estimate"][
        "rate_difference"
    ] == 1.0


def test_recovery_segments_reject_drifted_or_duplicate_id_sets() -> None:
    module = importlib.import_module("phaseH_eval.final_teacher_sft_report")
    summarize_segments = getattr(module, "summarize_recovery_segments")
    rows = [_row("a", resolved=False, patch_len=10, assistant_steps=50)]

    with pytest.raises(ValueError, match="duplicate all-unresolved"):
        summarize_segments(
            rows,
            all_unresolved_ids=["a", "a"],
            clamp_suspect_ids=[],
        )
    with pytest.raises(ValueError, match="duplicate clamp-suspect"):
        summarize_segments(
            rows,
            all_unresolved_ids=["a"],
            clamp_suspect_ids=["a", "a"],
        )
    with pytest.raises(ValueError, match="absent from all-unresolved"):
        summarize_segments(
            rows,
            all_unresolved_ids=["a"],
            clamp_suspect_ids=["other"],
        )
    with pytest.raises(ValueError, match="outside all-unresolved"):
        summarize_segments(
            rows + [_row("other", resolved=False, patch_len=10, assistant_steps=50)],
            all_unresolved_ids=["a"],
            clamp_suspect_ids=[],
        )


def _row(
    instance_id: str,
    *,
    resolved: bool,
    patch_len: int,
    assistant_steps: int,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "patch_len": patch_len,
        "assistant_steps": assistant_steps,
        "trajectory_present": True,
        "score_present": True,
        "pull_failed": False,
        "infra_error": None,
    }


def _selected(
    instance_id: str,
    *,
    resolved: bool,
    fixed_harness: bool,
    usable_outcome: bool = True,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "fixed_harness": fixed_harness,
        "usable_outcome": usable_outcome,
    }


def _empty_outcome(
    instance_id: str,
    *,
    trajectory_present: bool = True,
    guarded_forced: bool = False,
    guarded_forced_submit: bool = False,
    pull_failed: bool = False,
    usable_outcome: bool = True,
) -> dict[str, object]:
    return {
        **_selected(
            instance_id,
            resolved=False,
            fixed_harness=True,
            usable_outcome=usable_outcome,
        ),
        "patch_len": 0,
        "trajectory_present": trajectory_present,
        "guarded_forced": guarded_forced,
        "guarded_forced_submit": guarded_forced_submit,
        "pull_failed": pull_failed,
        "infra_error": None,
    }
