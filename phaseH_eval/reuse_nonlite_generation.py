#!/usr/bin/env python3
"""Carry exact unchanged non-Lite generation artifacts into a new run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from phaseH_eval.freeze_nonlite_hard30 import _publish_directory_atomic


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_tasks(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 30 or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected exactly 30 task objects: {path}")
    tasks = {row.get("instance_id"): row for row in rows}
    if len(tasks) != 30 or any(
        not isinstance(instance_id, str) or not instance_id
        for instance_id in tasks
    ):
        raise ValueError(f"task identities are not exact: {path}")
    return tasks


def reuse_generation(
    *,
    source_run: Path,
    target_run: Path,
    name: str,
    source_tasks: Path,
    target_tasks: Path,
    expected_reused: int,
) -> dict[str, Any]:
    source_run = Path(source_run)
    target_run = Path(target_run)
    source_tasks = Path(source_tasks)
    target_tasks = Path(target_tasks)
    if target_run.exists():
        raise FileExistsError(f"refusing to overwrite {target_run}")
    old_tasks = _read_tasks(source_tasks)
    new_tasks = _read_tasks(target_tasks)
    reused_ids = sorted(
        instance_id
        for instance_id in set(old_tasks) & set(new_tasks)
        if _canonical_json(old_tasks[instance_id])
        == _canonical_json(new_tasks[instance_id])
    )
    if len(reused_ids) != expected_reused:
        raise ValueError(
            f"expected {expected_reused} reusable tasks, got {len(reused_ids)}"
        )
    source_model = source_run / name
    source_predictions_path = source_model / "preds.json"
    source_predictions = _read_json(source_predictions_path)
    copied_predictions = {}
    trajectory_bindings = []
    source_trajectories: dict[str, Path] = {}
    for instance_id in reused_ids:
        prediction = source_predictions.get(instance_id)
        if (
            not isinstance(prediction, dict)
            or prediction.get("instance_id", instance_id) != instance_id
        ):
            raise ValueError(f"missing source prediction: {instance_id}")
        trajectory_path = (
            source_model / instance_id / f"{instance_id}.traj.json"
        )
        trajectory = _read_json(trajectory_path)
        evidence = (
            trajectory.get("info", {}).get("task_mutation")
            if isinstance(trajectory.get("info"), dict)
            else None
        )
        if (
            trajectory.get("instance_id", instance_id) != instance_id
            or not isinstance(evidence, dict)
            or evidence.get("instance_id") != instance_id
            or evidence.get("mutation_patch_sha256")
            != new_tasks[instance_id].get("mutation_patch_sha256")
        ):
            raise ValueError(
                f"trajectory mutation binding mismatch: {instance_id}"
            )
        copied_predictions[instance_id] = prediction
        source_trajectories[instance_id] = trajectory_path
        trajectory_bindings.append({
            "instance_id": instance_id,
            "source_path": str(trajectory_path.resolve()),
            "source_sha256": _sha256_path(trajectory_path),
            "target_path": f"{name}/{instance_id}/{instance_id}.traj.json",
        })

    holder: dict[str, Any] = {}

    def build(stage: Path) -> None:
        target_model = stage / name
        target_model.mkdir(parents=True)
        target_predictions_path = target_model / "preds.json"
        target_predictions_path.write_text(
            json.dumps(copied_predictions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for instance_id, source_path in source_trajectories.items():
            destination = (
                target_model / instance_id / f"{instance_id}.traj.json"
            )
            destination.parent.mkdir()
            shutil.copy2(source_path, destination)
        manifest = {
            "schema_version": 1,
            "complete": True,
            "training_eligible": False,
            "source_run": str(source_run.resolve()),
            "target_run": str(target_run.resolve()),
            "name": name,
            "source_tasks_sha256": _sha256_path(source_tasks),
            "target_tasks_sha256": _sha256_path(target_tasks),
            "source_predictions_sha256": _sha256_path(source_predictions_path),
            "target_predictions_sha256": _sha256_path(target_predictions_path),
            "reused": len(reused_ids),
            "missing": 30 - len(reused_ids),
            "reused_ids": reused_ids,
            "trajectory_bindings": trajectory_bindings,
        }
        acceptance = source_run / "acceptance.json"
        if acceptance.exists():
            manifest["source_acceptance_sha256"] = _sha256_path(acceptance)
        (stage / "generation_reuse_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        holder.update(manifest)

    _publish_directory_atomic(target_run, build)
    return holder


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--target-tasks", type=Path, required=True)
    parser.add_argument("--expected-reused", type=int, required=True)
    args = parser.parse_args()
    result = reuse_generation(
        source_run=args.source_run,
        target_run=args.target_run,
        name=args.name,
        source_tasks=args.source_tasks,
        target_tasks=args.target_tasks,
        expected_reused=args.expected_reused,
    )
    print(json.dumps({
        "reused": result["reused"],
        "missing": result["missing"],
        "target_run": result["target_run"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
