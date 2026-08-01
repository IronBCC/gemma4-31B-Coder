#!/usr/bin/env python3
"""Build auditable fixed-harness comparisons for the teacher-SFT adapter lane."""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


def _index_rows(
    rows: Iterable[dict[str, Any]],
    *,
    attempt: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        instance_id = str(row.get("instance_id") or "")
        if not instance_id:
            raise ValueError(f"{attempt} row is missing instance_id")
        if instance_id in indexed:
            raise ValueError(f"duplicate {attempt} outcome: {instance_id}")
        indexed[instance_id] = row
    return indexed


def _is_usable(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("trajectory_present"))
        and not bool(row.get("pull_failed"))
        and row.get("infra_error") is None
        and bool(row.get("score_present"))
        and row.get("resolved") is not None
    )


def _is_clamp_suspect(row: dict[str, Any]) -> bool:
    if bool(row.get("resolved")):
        return False
    patch_len = int(row.get("patch_len") or 0)
    assistant_steps = int(row.get("assistant_steps") or 0)
    return patch_len == 0 or assistant_steps >= 39


def _selected_row(
    row: dict[str, Any],
    *,
    source_attempt: str,
    fixed_harness: bool,
) -> dict[str, Any]:
    selected = dict(row)
    selected["source_attempt"] = source_attempt
    selected["fixed_harness"] = fixed_harness
    selected["usable_outcome"] = _is_usable(row)
    return selected


def select_effective_outcomes(
    original_rows: Iterable[dict[str, Any]],
    retest_rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply the lane's correction precedence without cherry-picking retests."""

    original = _index_rows(original_rows, attempt="original")
    retest = _index_rows(retest_rows, attempt="retest")
    selected: dict[str, dict[str, Any]] = {}

    unexpected = sorted(set(retest) - set(original))
    if unexpected:
        raise ValueError(f"retest IDs absent from original run: {unexpected}")

    for instance_id, old in original.items():
        retry = retest.get(instance_id)
        clamp_suspect = _is_clamp_suspect(old)
        if bool(old.get("resolved")):
            if retry is not None:
                raise ValueError(f"retest ID was already resolved originally: {instance_id}")
            selected[instance_id] = _selected_row(
                old,
                source_attempt="original",
                fixed_harness=False,
            )
        elif retry is not None and _is_usable(retry):
            selected[instance_id] = _selected_row(
                retry,
                source_attempt="retest",
                fixed_harness=True,
            )
        else:
            missing = dict(old)
            missing["resolved"] = False
            missing["trajectory_present"] = False
            missing["score_present"] = False
            selected[instance_id] = _selected_row(
                missing,
                source_attempt="missing_retest",
                fixed_harness=True,
            )
        selected[instance_id]["original_clamp_suspect"] = clamp_suspect

    return selected


def _binomial_upper_tail(n: int, start: int) -> float:
    if n == 0:
        return 1.0
    return sum(math.comb(n, value) for value in range(start, n + 1)) / (2**n)


def compare_fixed_harness_outcomes(
    left_outcomes: dict[str, dict[str, Any]],
    right_outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare only IDs with usable fixed-harness outcomes on both sides."""

    left_ids = {
        instance_id
        for instance_id, row in left_outcomes.items()
        if row.get("fixed_harness") and row.get("usable_outcome")
    }
    right_ids = {
        instance_id
        for instance_id, row in right_outcomes.items()
        if row.get("fixed_harness") and row.get("usable_outcome")
    }
    panel_ids = set(left_outcomes) | set(right_outcomes)
    pair_ids = sorted(left_ids & right_ids)
    contingency = {
        "both_resolved": 0,
        "left_only": 0,
        "right_only": 0,
        "neither": 0,
    }
    for instance_id in pair_ids:
        left_resolved = bool(left_outcomes[instance_id].get("resolved"))
        right_resolved = bool(right_outcomes[instance_id].get("resolved"))
        if left_resolved and right_resolved:
            outcome = "both_resolved"
        elif left_resolved:
            outcome = "left_only"
        elif right_resolved:
            outcome = "right_only"
        else:
            outcome = "neither"
        contingency[outcome] += 1

    left_only = contingency["left_only"]
    right_only = contingency["right_only"]
    discordant = left_only + right_only
    smaller = min(left_only, right_only)
    lower_tail = (
        sum(math.comb(discordant, value) for value in range(smaller + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "n": len(pair_ids),
        "pair_ids": pair_ids,
        "left_fixed_usable": len(left_ids),
        "right_fixed_usable": len(right_ids),
        "left_excluded": len(panel_ids - left_ids),
        "right_excluded": len(panel_ids - right_ids),
        "shared_excluded": len(panel_ids - (left_ids | right_ids)),
        "left_only_excluded": len(left_ids - right_ids),
        "right_only_excluded": len(right_ids - left_ids),
        "left_resolved": contingency["both_resolved"] + left_only,
        "right_resolved": contingency["both_resolved"] + right_only,
        "discordant_pairs": discordant,
        "right_better_one_sided_p": _binomial_upper_tail(discordant, right_only),
        "two_sided_p": min(1.0, 2 * lower_tail),
        "contingency": contingency,
    }


def classify_empty_outcomes(
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Split empty predictions without treating missing artifacts as model behavior."""

    classified = {
        "model": [],
        "harness_forced": [],
        "docker_failed": [],
        "unknown_missing": [],
    }
    for instance_id, row in sorted(outcomes.items()):
        if int(row.get("patch_len") or 0) > 0:
            continue
        infra_error = str(row.get("infra_error") or "")
        if row.get("pull_failed") or infra_error.startswith("docker"):
            cause = "docker_failed"
        elif row.get("trajectory_present") and row.get("guarded_forced_submit"):
            cause = "harness_forced"
        elif row.get("trajectory_present"):
            cause = "model"
        else:
            cause = "unknown_missing"
        classified[cause].append(instance_id)
    return {
        "total": sum(len(ids) for ids in classified.values()),
        **classified,
    }


def _nearest_rank(values: list[int], quantile: float) -> int:
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def summarize_step_counts(
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, int | float]:
    """Summarize assistant-turn counts for usable fixed-harness outcomes."""

    steps = sorted(
        int(row["assistant_steps"])
        for row in outcomes.values()
        if row.get("fixed_harness")
        and row.get("usable_outcome")
        and row.get("assistant_steps") is not None
    )
    if not steps:
        return {
            "n": 0,
            "min": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
            "max": 0,
            "ge39": 0,
            "gt39": 0,
            "ge120": 0,
        }
    return {
        "n": len(steps),
        "min": steps[0],
        "p25": _nearest_rank(steps, 0.25),
        "median": statistics.median(steps),
        "p75": _nearest_rank(steps, 0.75),
        "p90": _nearest_rank(steps, 0.90),
        "max": steps[-1],
        "ge39": sum(step >= 39 for step in steps),
        "gt39": sum(step > 39 for step in steps),
        "ge120": sum(step >= 120 for step in steps),
    }


def select_fixed_panel(
    instance_ids: Iterable[str],
    *,
    size: int,
    namespace: str,
) -> list[str]:
    """Pre-register a deterministic, input-order-independent evaluation panel."""

    ids = [str(instance_id) for instance_id in instance_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("fixed-panel input contains duplicate instance IDs")
    if size <= 0 or size > len(ids):
        raise ValueError(f"fixed-panel size {size} is invalid for {len(ids)} IDs")
    if not namespace:
        raise ValueError("fixed-panel namespace must not be empty")
    return sorted(
        ids,
        key=lambda instance_id: hashlib.sha256(
            f"{namespace}:{instance_id}".encode()
        ).hexdigest(),
    )[:size]


def _load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((row.get("instance_id"), row) for row in data if isinstance(row, dict))
    else:
        raise ValueError(f"{path}: predictions must be an object or list")

    predictions: dict[str, dict[str, Any]] = {}
    for key, source in items:
        if not isinstance(source, dict):
            raise ValueError(f"{path}: prediction {key!r} is not an object")
        instance_id = str(source.get("instance_id") or key or "")
        if not instance_id:
            raise ValueError(f"{path}: prediction is missing instance_id")
        if key and str(key) != instance_id:
            raise ValueError(f"{path}: prediction key/instance_id mismatch for {key}")
        if instance_id in predictions:
            raise ValueError(f"{path}: duplicate prediction {instance_id}")
        predictions[instance_id] = dict(source)
    return predictions


def _load_score_ids(
    report_paths: Iterable[Path],
) -> tuple[set[str], set[str], set[str]]:
    outcomes: dict[str, str] = {}
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        current = {
            "resolved": {
                str(value) for value in report.get("resolved_ids") or []
            },
            "unresolved": {
                str(value) for value in report.get("unresolved_ids") or []
            },
            "error": {
                str(value) for value in report.get("error_ids") or []
            },
        }
        overlap = (
            (current["resolved"] & current["unresolved"])
            | (current["resolved"] & current["error"])
            | (current["unresolved"] & current["error"])
        )
        if overlap:
            raise ValueError(
                f"score report {path} assigns multiple outcomes to: {sorted(overlap)}"
            )
        for status, instance_ids in current.items():
            for instance_id in instance_ids:
                outcomes[instance_id] = status
    return (
        {instance_id for instance_id, status in outcomes.items() if status == "resolved"},
        {instance_id for instance_id, status in outcomes.items() if status == "unresolved"},
        {instance_id for instance_id, status in outcomes.items() if status == "error"},
    )


def _load_trajectory_map(
    paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    """Load trajectories in attempt order, with a later retry replacing an older one."""

    trajectories: dict[str, dict[str, Any]] = {}
    for path in paths:
        raw = path.read_bytes()
        trajectory = json.loads(raw)
        instance_id = str(trajectory.get("instance_id") or "")
        if not instance_id:
            raise ValueError(f"{path}: trajectory is missing instance_id")
        messages = trajectory.get("messages") or []
        assistants = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        guarded_forced = any(
            isinstance(
                ((message.get("extra") or {}).get("response") or {}).get(
                    "guarded_forced_command"
                ),
                str,
            )
            for message in assistants
        )
        final_guarded_command = (
            ((assistants[-1].get("extra") or {}).get("response") or {}).get(
                "guarded_forced_command"
            )
            if assistants
            else None
        )
        guarded_forced_submit = (
            isinstance(final_guarded_command, str)
            and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in final_guarded_command
        )
        trajectories[instance_id] = {
            "trajectory_present": True,
            "trajectory_path": str(path),
            "trajectory_sha256": hashlib.sha256(raw).hexdigest(),
            "assistant_steps": len(assistants),
            "guarded_forced": guarded_forced,
            "guarded_forced_submit": guarded_forced_submit,
        }
    return trajectories


def load_attempt_artifacts(
    *,
    expected_ids: Iterable[str],
    preds_path: Path,
    report_paths: Iterable[Path],
    trajectory_paths: Iterable[Path],
    attempt_kind: str,
    run_id: str,
    fixed_harness: bool,
    pull_failed_ids: set[str] | None = None,
    docker_failed_ids: set[str] | None = None,
    model_failure_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Normalize one attempt from raw artifacts without inventing missing outcomes."""

    expected = [str(instance_id) for instance_id in expected_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected IDs contain duplicates")
    expected_set = set(expected)
    predictions = _load_prediction_map(preds_path)
    resolved_ids, unresolved_ids, score_error_ids = _load_score_ids(report_paths)
    trajectories = _load_trajectory_map(trajectory_paths)
    pull_failed = set(pull_failed_ids or ())
    docker_failed = set(docker_failed_ids or ())
    model_failures = set(model_failure_ids or ())

    observed_ids = set(predictions) | set(trajectories) | resolved_ids | unresolved_ids
    unexpected = sorted(observed_ids - expected_set)
    if unexpected:
        raise ValueError(f"artifacts contain IDs outside expected set: {unexpected}")
    unexpected_model_failures = sorted(model_failures - score_error_ids)
    if unexpected_model_failures:
        raise ValueError(
            "model failures are not scorer error IDs: "
            f"{unexpected_model_failures}"
        )

    rows: dict[str, dict[str, Any]] = {}
    for instance_id in expected:
        prediction = predictions.get(instance_id)
        prediction_present = prediction is not None
        patch = str((prediction or {}).get("model_patch") or "")
        patch_len = len(patch)
        trajectory = trajectories.get(instance_id)
        trajectory_present = trajectory is not None

        if not prediction_present:
            resolved: bool | None = None
            score_present = False
        elif not patch.strip():
            resolved = False
            score_present = True
        elif instance_id in resolved_ids:
            resolved = True
            score_present = True
        elif instance_id in unresolved_ids:
            resolved = False
            score_present = True
        elif instance_id in model_failures:
            resolved = False
            score_present = True
        else:
            resolved = None
            score_present = False

        if instance_id in pull_failed:
            infra_error = "docker_pull_failed"
        elif instance_id in docker_failed:
            infra_error = "docker_run_failed"
        elif not prediction_present:
            infra_error = "missing_prediction"
        elif not trajectory_present:
            infra_error = "missing_trajectory"
        elif instance_id in score_error_ids and instance_id not in model_failures:
            infra_error = "score_error"
        elif not score_present:
            infra_error = "score_missing"
        else:
            infra_error = None

        row = {
            "schema_version": 1,
            "instance_id": instance_id,
            "attempt_kind": attempt_kind,
            "run_id": run_id,
            "fixed_harness": fixed_harness,
            "prediction_present": prediction_present,
            "preds_path": str(preds_path),
            "patch_len": patch_len,
            "trajectory_present": trajectory_present,
            "trajectory_path": None,
            "trajectory_sha256": None,
            "assistant_steps": None,
            "guarded_forced": False,
            "guarded_forced_submit": False,
            "pull_failed": instance_id in pull_failed,
            "infra_error": infra_error,
            "score_present": score_present,
            "resolved": resolved,
        }
        if trajectory is not None:
            row.update(trajectory)
        row["usable_outcome"] = _is_usable(row)
        rows[instance_id] = row

    stats = {
        "expected": len(expected),
        "preds": len(predictions),
        "traj_files": len(trajectories),
        "pull_failed": len(pull_failed),
        "resolved": sum(row["resolved"] is True for row in rows.values()),
        "usable_outcomes": sum(bool(row["usable_outcome"]) for row in rows.values()),
    }
    return rows, stats


def instance_id_from_image(image: str) -> str:
    """Reverse the SWE-bench image naming convention used by the retest driver."""

    marker = "sweb.eval.x86_64."
    if marker not in image:
        raise ValueError(f"unrecognized SWE-bench image: {image}")
    tail = image.split(marker, 1)[1].removesuffix(":latest")
    if "_1776_" not in tail:
        raise ValueError(f"unrecognized SWE-bench image: {image}")
    repo, issue = tail.split("_1776_", 1)
    if not repo or not issue:
        raise ValueError(f"unrecognized SWE-bench image: {image}")
    return f"{repo}__{issue}"


_INSTANCE_ERROR_BLOCK = re.compile(
    r"Error processing instance\s+([^:\s]+):"
    r"(.*?)(?=Error processing instance\s+[^:\s]+:|\Z)",
    re.DOTALL,
)
_DOCKER_EXIT = re.compile(r"returned non-zero exit status (?:125|127)\b")


def docker_failed_ids_from_log(log_text: str) -> set[str]:
    """Return IDs whose own error block proves a Docker exit 125/127."""

    return {
        instance_id
        for instance_id, body in _INSTANCE_ERROR_BLOCK.findall(log_text)
        if _DOCKER_EXIT.search(body)
    }


def model_failure_ids_from_score_logs(log_paths: Iterable[Path]) -> set[str]:
    """Return IDs whose score logs prove the model emitted an invalid patch."""

    return {
        path.parent.name
        for path in log_paths
        if "Patch Apply Failed:" in path.read_text(encoding="utf-8", errors="replace")
    }


def summarize_recovery(
    original_rows: Iterable[dict[str, Any]],
    retest_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize recovery as an ID union, never as an aggregate-count sum."""

    original = _index_rows(original_rows, attempt="original")
    retest = _index_rows(retest_rows, attempt="retest")
    selected = select_effective_outcomes(original.values(), retest.values())
    original_resolved_ids = {
        instance_id for instance_id, row in original.items() if row.get("resolved") is True
    }
    retest_resolved_ids = {
        instance_id
        for instance_id, row in retest.items()
        if _is_usable(row) and row.get("resolved") is True
    }
    corrected_resolved_ids = original_resolved_ids | retest_resolved_ids
    original_unresolved_ids = set(original) - original_resolved_ids
    retest_usable_ids = {
        instance_id for instance_id, row in retest.items() if _is_usable(row)
    }
    missing_retest_ids = {
        instance_id
        for instance_id in original_unresolved_ids
        if selected[instance_id]["source_attempt"] == "missing_retest"
    }
    return {
        "original_resolved": len(original_resolved_ids),
        "original_resolved_ids": sorted(original_resolved_ids),
        "retest_expected": len(original_unresolved_ids),
        "retest_usable": len(retest_usable_ids),
        "retest_resolved": len(retest_resolved_ids),
        "retest_resolved_ids": sorted(retest_resolved_ids),
        "corrected_recovery_lower_bound": len(corrected_resolved_ids),
        "corrected_resolved_ids": sorted(corrected_resolved_ids),
        "missing_retest_ids": sorted(missing_retest_ids),
    }


def summarize_recovery_segments(
    retest_rows: Iterable[dict[str, Any]],
    *,
    all_unresolved_ids: Iterable[str],
    clamp_suspect_ids: Iterable[str],
) -> dict[str, Any]:
    """Separate harness-suspect recovery from ordinary resampling flips.

    The preserved pre-retest ID lists define the segments. New trajectories must
    never be used to reconstruct which historical outcomes were clamp-suspect.
    """

    all_ids = [str(instance_id) for instance_id in all_unresolved_ids]
    suspect_ids = [str(instance_id) for instance_id in clamp_suspect_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate all-unresolved IDs")
    if len(suspect_ids) != len(set(suspect_ids)):
        raise ValueError("duplicate clamp-suspect IDs")

    all_set = set(all_ids)
    suspect_set = set(suspect_ids)
    suspect_outside_all = sorted(suspect_set - all_set)
    if suspect_outside_all:
        raise ValueError(
            "clamp-suspect IDs absent from all-unresolved set: "
            f"{suspect_outside_all}"
        )

    retest = _index_rows(retest_rows, attempt="retest")
    unexpected_retests = sorted(set(retest) - all_set)
    if unexpected_retests:
        raise ValueError(
            "retest IDs outside all-unresolved set: "
            f"{unexpected_retests}"
        )

    def summarize_segment(segment_ids: set[str]) -> dict[str, Any]:
        scored_ids = sorted(
            instance_id
            for instance_id in segment_ids
            if instance_id in retest and _is_usable(retest[instance_id])
        )
        resolved_ids = [
            instance_id
            for instance_id in scored_ids
            if retest[instance_id].get("resolved") is True
        ]
        expected = len(segment_ids)
        scored = len(scored_ids)
        resolved = len(resolved_ids)
        return {
            "expected": expected,
            "scored": scored,
            "coverage": scored / expected if expected else 1.0,
            "resolved": resolved,
            "rate": resolved / scored if scored else None,
            "scored_ids": scored_ids,
            "resolved_ids": resolved_ids,
            "missing_ids": sorted(segment_ids - set(scored_ids)),
        }

    suspect = summarize_segment(suspect_set)
    added = summarize_segment(all_set - suspect_set)
    floor_rate = added["rate"]
    total_scored = suspect["scored"] + added["scored"]
    total_resolved = suspect["resolved"] + added["resolved"]

    if floor_rate is None:
        suspect_excess = {
            "rate_difference": None,
            "percentage_points": None,
            "expected_resolved_at_floor": None,
            "excess_resolved": None,
        }
        total_excess = {
            "scored": total_scored,
            "resolved": total_resolved,
            "expected_resolved_at_floor": None,
            "excess_resolved": None,
        }
    else:
        rate_difference = suspect["rate"] - floor_rate
        expected_suspect_at_floor = suspect["scored"] * floor_rate
        expected_total_at_floor = total_scored * floor_rate
        suspect_excess = {
            "rate_difference": rate_difference,
            "percentage_points": rate_difference * 100,
            "expected_resolved_at_floor": expected_suspect_at_floor,
            "excess_resolved": suspect["resolved"] - expected_suspect_at_floor,
        }
        total_excess = {
            "scored": total_scored,
            "resolved": total_resolved,
            "expected_resolved_at_floor": expected_total_at_floor,
            "excess_resolved": total_resolved - expected_total_at_floor,
        }

    return {
        "clamp_suspect": suspect,
        "added_non_suspect": added,
        "resampling_noise_floor": {
            "scored": added["scored"],
            "resolved": added["resolved"],
            "rate": floor_rate,
        },
        "suspect_excess_over_floor_estimate": suspect_excess,
        "total_excess_over_floor_estimate": total_excess,
    }
