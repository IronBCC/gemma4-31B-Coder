from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseH_eval.reuse_nonlite_generation import reuse_generation


def _tasks(prefix: str, count: int = 30) -> list[dict]:
    return [
        {
            "instance_id": f"fixture__repo.{prefix}_{index:02d}",
            "mutation_patch_sha256": f"{index:064x}",
        }
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reuse_generation_copies_only_exact_shared_tasks_atomically(
    tmp_path: Path,
) -> None:
    source_tasks = _tasks("old")
    target_tasks = source_tasks[:27] + _tasks("new", 3)
    source_tasks_path = tmp_path / "old.jsonl"
    target_tasks_path = tmp_path / "new.jsonl"
    _write_jsonl(source_tasks_path, source_tasks)
    _write_jsonl(target_tasks_path, target_tasks)
    source = tmp_path / "source"
    model = source / "model"
    model.mkdir(parents=True)
    predictions = {}
    for row in source_tasks:
        instance_id = row["instance_id"]
        predictions[instance_id] = {
            "instance_id": instance_id,
            "model_patch": f"patch-{instance_id}",
        }
        trajectory = model / instance_id / f"{instance_id}.traj.json"
        trajectory.parent.mkdir()
        trajectory.write_text(json.dumps({
            "instance_id": instance_id,
            "info": {
                "task_mutation": {
                    "instance_id": instance_id,
                    "mutation_patch_sha256": row["mutation_patch_sha256"],
                }
            },
        }))
    (model / "preds.json").write_text(json.dumps(predictions))
    target = tmp_path / "target"

    manifest = reuse_generation(
        source_run=source,
        target_run=target,
        name="model",
        source_tasks=source_tasks_path,
        target_tasks=target_tasks_path,
        expected_reused=27,
    )

    copied_predictions = json.loads((target / "model" / "preds.json").read_text())
    assert set(copied_predictions) == {
        row["instance_id"] for row in source_tasks[:27]
    }
    assert manifest["reused"] == 27
    assert manifest["missing"] == 3
    assert len(list(target.glob("model/**/*.traj.json"))) == 27
    assert json.loads(
        (target / "generation_reuse_manifest.json").read_text()
    ) == manifest
    assert not list(tmp_path.glob(".target.*"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reuse_generation(
            source_run=source,
            target_run=target,
            name="model",
            source_tasks=source_tasks_path,
            target_tasks=target_tasks_path,
            expected_reused=27,
        )


def test_reuse_generation_rejects_trajectory_mutation_mismatch(
    tmp_path: Path,
) -> None:
    rows = _tasks("old")
    source_tasks = tmp_path / "old.jsonl"
    target_tasks = tmp_path / "new.jsonl"
    _write_jsonl(source_tasks, rows)
    _write_jsonl(target_tasks, rows)
    source = tmp_path / "source"
    model = source / "model"
    model.mkdir(parents=True)
    (model / "preds.json").write_text(json.dumps({
        row["instance_id"]: {
            "instance_id": row["instance_id"],
            "model_patch": "patch",
        }
        for row in rows
    }))
    for row in rows:
        instance_id = row["instance_id"]
        trajectory = model / instance_id / f"{instance_id}.traj.json"
        trajectory.parent.mkdir()
        trajectory.write_text(json.dumps({
            "instance_id": instance_id,
            "info": {
                "task_mutation": {
                    "instance_id": instance_id,
                    "mutation_patch_sha256": (
                        "bad" if instance_id == rows[0]["instance_id"]
                        else row["mutation_patch_sha256"]
                    ),
                }
            },
        }))

    with pytest.raises(ValueError, match="mutation binding mismatch"):
        reuse_generation(
            source_run=source,
            target_run=tmp_path / "target",
            name="model",
            source_tasks=source_tasks,
            target_tasks=target_tasks,
            expected_reused=30,
        )
