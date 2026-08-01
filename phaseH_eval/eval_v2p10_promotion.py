#!/usr/bin/env python3
"""Fail-closed promotion decisions and trajectory-health summaries for v2.10."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from phaseH_eval.audit_v2p9_failure_modes import extract_command_outcomes
from phaseH_eval.final_teacher_sft_report import (
    classify_empty_outcomes,
    docker_failed_ids_from_log,
    instance_id_from_image,
    load_attempt_artifacts,
    model_failure_ids_from_score_logs,
    summarize_step_counts,
)


_FORMAT_ERROR_RE = re.compile(r"tool call error", re.IGNORECASE)
_HARD30_RESOLUTION_CONTRACT = (
    "clean F2P passes; mutation reproduces exact F2P failure; "
    "candidate passes all frozen F2P IDs; P2P not checked"
)


def _complete_run(metrics: Mapping[str, Any], expected: int) -> bool:
    return bool(
        metrics.get("status") == "complete"
        and metrics.get("expected") == expected
        and metrics.get("preds") == expected
        and metrics.get("traj_files") == expected
        and metrics.get("usable_outcomes") == expected
    )


def hard30_decision(
    *,
    candidate: Mapping[str, Any],
    v2p8_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide whether a matched, non-Lite candidate may advance to fixed150."""

    if (
        not _complete_run(candidate, 30)
        or not _complete_run(v2p8_control, 30)
        or candidate.get("schema_version") != 2
        or v2p8_control.get("schema_version") != 2
        or candidate.get("resolution_contract")
        != _HARD30_RESOLUTION_CONTRACT
        or v2p8_control.get("resolution_contract")
        != _HARD30_RESOLUTION_CONTRACT
        or not isinstance(candidate.get("tasks_sha256"), str)
        or candidate.get("tasks_sha256")
        != v2p8_control.get("tasks_sha256")
        or int(candidate.get("infra_failed", -1)) != 0
        or int(v2p8_control.get("infra_failed", -1)) != 0
    ):
        return {"run_fixed150": False, "reason": "non-Lite hard30 incomplete"}
    if int(candidate.get("nonempty", -1)) < 18:
        return {"run_fixed150": False, "reason": "nonempty patch floor"}
    format_error_rate = candidate.get("format_error_rate")
    if (
        isinstance(format_error_rate, bool)
        or not isinstance(format_error_rate, (int, float))
        or not 0 <= float(format_error_rate) < 0.10
    ):
        return {"run_fixed150": False, "reason": "format-error ceiling"}
    if int(candidate.get("repeat_loops", -1)) != 0:
        return {"run_fixed150": False, "reason": "repeated-failure loop"}
    if int(candidate.get("resolved", -1)) < int(
        v2p8_control.get("resolved", -1)
    ):
        return {"run_fixed150": False, "reason": "v2.8 hard30 regression"}
    return {
        "run_fixed150": True,
        "reason": "non-Lite hard30 gate passed",
        "no_adapter_selected": True,
    }


def promotion_decision(
    *,
    candidate: Mapping[str, Any],
    best_incumbent: Mapping[str, Any],
    v2p9: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide whether fixed150 evidence permits the expensive full-300 run."""

    if (
        not _complete_run(candidate, 150)
        or not _complete_run(best_incumbent, 150)
        or not _complete_run(v2p9, 150)
        or int(candidate.get("pull_failed", -1)) != 0
        or int(candidate.get("docker_failed", -1)) != 0
    ):
        return {"run_full300": False, "reason": "fixed150 incomplete"}
    if int(candidate.get("resolved", -1)) < int(
        best_incumbent.get("resolved", -1)
    ):
        return {"run_full300": False, "reason": "fixed150 regression"}
    wins = int(paired.get("wins", -1))
    losses = int(paired.get("losses", -1))
    ties = int(paired.get("ties", -1))
    if min(wins, losses, ties) < 0 or wins + losses + ties != 150:
        return {"run_full300": False, "reason": "paired evidence incomplete"}
    if wins < losses:
        return {
            "run_full300": False,
            "reason": "paired regression versus v2.9",
        }
    if int(candidate.get("empty", -1)) > int(v2p9.get("empty", -1)):
        return {"run_full300": False, "reason": "empty-patch regression"}
    if int(candidate.get("repeat_loops", -1)) > int(
        v2p9.get("repeat_loops", -1)
    ):
        return {
            "run_full300": False,
            "reason": "repeated-failure-loop regression",
        }
    return {
        "run_full300": True,
        "reason": "fixed150 gate passed",
        "no_adapter_selected": True,
    }


def exact_repeated_failure_loops(trajectory: Mapping[str, Any]) -> int:
    """Count maximal runs of 3+ adjacent identical commands with nonzero exits."""

    messages = trajectory.get("messages")
    rows = (
        [row for row in messages if isinstance(row, Mapping)]
        if isinstance(messages, list)
        else []
    )
    outcomes = extract_command_outcomes(rows)
    loops = 0
    index = 0
    while index < len(outcomes):
        current = outcomes[index]
        end = index + 1
        while (
            end < len(outcomes)
            and outcomes[end].command.strip() == current.command.strip()
            and outcomes[end].returncode not in (None, 0)
            and current.returncode not in (None, 0)
        ):
            end += 1
        if (
            current.returncode not in (None, 0)
            and end - index >= 3
        ):
            loops += 1
        index = end
    return loops


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_map(
    run_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    trajectories: dict[str, dict[str, Any]] = {}
    paths = sorted(run_root.glob("**/*.traj.json"))
    for path in paths:
        row = _read_json(path)
        instance_id = row.get("instance_id") or path.parent.name
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"trajectory is missing instance_id: {path}")
        if instance_id in trajectories:
            raise ValueError(f"duplicate trajectory identity: {instance_id}")
        trajectories[instance_id] = row
    return trajectories, paths


def summarize_run_health(run_root: Path) -> dict[str, Any]:
    """Bind acceptance, predictions, and trajectories into promotion metrics."""

    run_root = Path(run_root)
    acceptance_path = run_root / "acceptance.json"
    predictions_path = run_root / "preds_all.json"
    acceptance = _read_json(acceptance_path)
    predictions = _read_json(predictions_path)
    trajectories, trajectory_paths = _trajectory_map(run_root)
    if set(predictions) != set(trajectories):
        raise ValueError("prediction and trajectory identities differ")
    if acceptance.get("traj_files") != len(trajectory_paths):
        raise ValueError("acceptance trajectory count differs from artifacts")
    repeat_loops = sum(
        exact_repeated_failure_loops(trajectory)
        for trajectory in trajectories.values()
    )
    format_errors = 0
    assistant_responses = 0
    for trajectory in trajectories.values():
        messages = trajectory.get("messages")
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "assistant":
                assistant_responses += 1
            content = message.get("content")
            if isinstance(content, str) and _FORMAT_ERROR_RE.search(content):
                format_errors += 1
    empty_split = acceptance.get("empty_split")
    empty = (
        int(empty_split.get("total", -1))
        if isinstance(empty_split, Mapping)
        else sum(
            not str(row.get("model_patch") or "").strip()
            for row in predictions.values()
            if isinstance(row, Mapping)
        )
    )
    metrics = {
        key: acceptance.get(key)
        for key in (
            "status",
            "expected",
            "preds",
            "traj_files",
            "usable_outcomes",
            "resolved",
            "pull_failed",
            "docker_failed",
        )
    }
    metrics.update({
        "empty": empty,
        "nonempty": len(predictions) - empty,
        "repeat_loops": repeat_loops,
        "tool_format_errors": format_errors,
        "assistant_responses": assistant_responses,
        "format_error_rate": (
            format_errors / assistant_responses
            if assistant_responses
            else None
        ),
        "artifact_bindings": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                acceptance_path,
                predictions_path,
                *trajectory_paths,
            )
        ],
    })
    return metrics


def paired_resolved(
    candidate_acceptance: Mapping[str, Any],
    reference_acceptance: Mapping[str, Any],
) -> dict[str, int]:
    if (
        not _complete_run(candidate_acceptance, 150)
        or not _complete_run(reference_acceptance, 150)
    ):
        raise ValueError("paired fixed150 inputs are incomplete")
    candidate = set(candidate_acceptance.get("resolved_ids") or [])
    reference = set(reference_acceptance.get("resolved_ids") or [])
    return {
        "wins": len(candidate - reference),
        "losses": len(reference - candidate),
        "ties": 150 - len(candidate ^ reference),
    }


def validate_fixed_run(
    *,
    name: str,
    ids_path: Path,
    run_root: Path,
    score_log_root: Path = Path("logs/run_evaluation"),
) -> dict[str, Any]:
    """Build the same strict acceptance ledger for fixed150 or full300."""

    ids_value = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    if (
        not isinstance(ids_value, list)
        or not ids_value
        or len(ids_value) != len(set(ids_value))
        or any(not isinstance(value, str) for value in ids_value)
    ):
        raise ValueError("fixed-eval IDs must be a unique nonempty string list")
    run_root = Path(run_root)
    summary = _read_json(run_root / "summary.json")
    batch_dirs = sorted(
        (
            path
            for path in run_root.iterdir()
            if path.is_dir()
            and path.name.startswith("b")
            and path.name[1:].isdigit()
        ),
        key=lambda path: int(path.name[1:]),
    )
    report_paths = [
        path
        for batch_dir in batch_dirs
        if (path := batch_dir / "report" / "official_report.json").is_file()
    ]
    trajectory_paths = [
        path
        for batch_dir in batch_dirs
        for path in sorted(batch_dir.glob("**/*.traj.json"))
    ]
    pull_failed_ids = {
        instance_id_from_image(image)
        for image in summary.get("pull_failed") or []
    }
    docker_failed_ids = set()
    model_failure_ids = set()
    for batch_dir in batch_dirs:
        for gen_log in batch_dir.glob("*.gen.log"):
            docker_failed_ids.update(
                docker_failed_ids_from_log(
                    gen_log.read_text(encoding="utf-8", errors="replace")
                )
            )
        model_failure_ids.update(
            model_failure_ids_from_score_logs(
                (
                    score_log_root / f"{run_root.name}_{batch_dir.name}"
                ).glob("**/run_instance.log")
            )
        )
    rows, stats = load_attempt_artifacts(
        expected_ids=ids_value,
        preds_path=run_root / "preds_all.json",
        report_paths=report_paths,
        trajectory_paths=trajectory_paths,
        attempt_kind="fixed_eval",
        run_id=run_root.name,
        fixed_harness=True,
        pull_failed_ids=pull_failed_ids,
        docker_failed_ids=docker_failed_ids,
        model_failure_ids=model_failure_ids,
    )
    resolved_ids = sorted(
        instance_id
        for instance_id, row in rows.items()
        if row.get("usable_outcome") and row.get("resolved") is True
    )
    expected = len(ids_value)
    problems = []
    if stats["preds"] != expected:
        problems.append(f"preds={stats['preds']}/{expected}")
    if stats["traj_files"] != expected:
        problems.append(f"unique_trajectories={stats['traj_files']}/{expected}")
    if stats["usable_outcomes"] != expected:
        problems.append(f"usable={stats['usable_outcomes']}/{expected}")
    if len(trajectory_paths) != expected:
        problems.append(f"trajectory_files={len(trajectory_paths)}/{expected}")
    if int(summary.get("scored") or 0) != stats["preds"]:
        problems.append(
            f"summary_scored={summary.get('scored')} "
            f"normalized_preds={stats['preds']}"
        )
    if set(summary.get("resolved") or []) != set(resolved_ids):
        problems.append(
            f"summary_resolved={len(summary.get('resolved') or [])} "
            f"normalized_resolved={len(resolved_ids)}"
        )
    if pull_failed_ids:
        problems.append(f"pull_failed={len(pull_failed_ids)}")
    if docker_failed_ids:
        problems.append(f"docker_failed={len(docker_failed_ids)}")
    unexpected = sorted(set(rows) - set(ids_value))
    if unexpected:
        problems.append(f"unexpected_ids={unexpected[:10]}")
    return {
        "schema_version": 1,
        "run_id": run_root.name,
        "name": name,
        "status": "complete" if not problems else "incomplete",
        "expected": expected,
        "preds": stats["preds"],
        "traj_files": len(trajectory_paths),
        "unique_trajectories": stats["traj_files"],
        "usable_outcomes": stats["usable_outcomes"],
        "resolved": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "pull_failed": len(pull_failed_ids),
        "docker_failed": len(docker_failed_ids),
        "empty_split": classify_empty_outcomes(rows),
        "steps": summarize_step_counts(rows),
        "problems": problems,
    }


def _publish_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _bound_decision(
    decision: Mapping[str, Any],
    inputs: Sequence[Path],
) -> dict[str, Any]:
    return {
        **decision,
        "schema_version": 1,
        "complete": True,
        "input_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        ],
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--candidate", type=Path, required=True)
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--out", type=Path, required=True)
    validate = subparsers.add_parser("validate-fixed")
    validate.add_argument("--name", required=True)
    validate.add_argument("--ids", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    hard = subparsers.add_parser("decide-hard30")
    hard.add_argument("--candidate", type=Path, required=True)
    hard.add_argument("--v2p8-control", type=Path, required=True)
    hard.add_argument("--out", type=Path, required=True)
    fixed = subparsers.add_parser("decide-fixed150")
    fixed.add_argument("--candidate", type=Path, required=True)
    fixed.add_argument("--best-incumbent", type=Path, required=True)
    fixed.add_argument("--v2p9", type=Path, required=True)
    fixed.add_argument("--paired", type=Path, required=True)
    fixed.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        _publish_json_noreplace(args.out, summarize_run_health(args.run_root))
        return 0
    if args.command == "pair":
        _publish_json_noreplace(
            args.out,
            _bound_decision(
                paired_resolved(
                    _read_json(args.candidate),
                    _read_json(args.reference),
                ),
                (args.candidate, args.reference),
            ),
        )
        return 0
    if args.command == "decide-hard30":
        _publish_json_noreplace(
            args.out,
            _bound_decision(
                hard30_decision(
                    candidate=_read_json(args.candidate),
                    v2p8_control=_read_json(args.v2p8_control),
                ),
                (args.candidate, args.v2p8_control),
            ),
        )
        return 0
    if args.command == "decide-fixed150":
        _publish_json_noreplace(
            args.out,
            _bound_decision(
                promotion_decision(
                    candidate=_read_json(args.candidate),
                    best_incumbent=_read_json(args.best_incumbent),
                    v2p9=_read_json(args.v2p9),
                    paired=_read_json(args.paired),
                ),
                (
                    args.candidate,
                    args.best_incumbent,
                    args.v2p9,
                    args.paired,
                ),
            ),
        )
        return 0
    acceptance = validate_fixed_run(
        name=args.name,
        ids_path=args.ids,
        run_root=args.run_root,
    )
    _publish_json_noreplace(args.out, acceptance)
    return 0 if acceptance["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
