from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from phaseH_eval import retest_empties


def _artifact_binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_scoreable_diff_predictions_excludes_empty_and_non_diff_rows() -> None:
    predictions = {
        "empty": {"instance_id": "empty", "model_patch": ""},
        "prose": {"instance_id": "prose", "model_patch": "I changed src/x.py"},
        "diff": {
            "instance_id": "diff",
            "model_patch": "diff --git a/src/x.py b/src/x.py\n+fixed\n",
        },
    }

    assert retest_empties.scoreable_diff_predictions(predictions) == {
        "diff": predictions["diff"],
    }


def test_pull_refuses_to_start_without_reserved_disk_headroom(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        raise AssertionError("docker pull must not start below the admission floor")

    monkeypatch.setattr(retest_empties, "free_gb", lambda: 49.0)
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)

    image, ok, error = retest_empties.pull("repo/image:tag")

    assert image == "repo/image:tag"
    assert ok is False
    assert "disk admission" in error
    assert calls == []


def test_pull_all_serializes_different_repositories(monkeypatch) -> None:
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **_kwargs):
        nonlocal active, max_active
        assert command[:3] == ["docker", "pull", "-q"]
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return Result()

    monkeypatch.setattr(retest_empties, "free_gb", lambda: 100.0)
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)

    failed = retest_empties.pull_all(
        [
            "swebench/sweb.eval.x86_64.django_1776_django-1:latest",
            "swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-1:latest",
        ],
        workers=2,
    )

    assert failed == []
    assert max_active == 1


def test_global_docker_transaction_lock_excludes_another_process(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "docker-transaction.lock"

    with retest_empties.docker_transaction_lock(lock_path):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import fcntl,sys; "
                "f=open(sys.argv[1], 'a+'); "
                "\ntry:\n fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "except BlockingIOError:\n print('blocked')\n"
                "else:\n print('acquired')\n",
                str(lock_path),
            ],
            text=True,
            capture_output=True,
            check=True,
        )

    assert probe.stdout.strip() == "blocked"


def test_pull_removes_exact_image_if_success_crosses_hard_disk_floor(
    monkeypatch,
) -> None:
    free_values = iter([60.0, 30.0])
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(retest_empties, "free_gb", lambda: next(free_values))
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)

    image, ok, error = retest_empties.pull("repo/image:tag")

    assert image == "repo/image:tag"
    assert ok is False
    assert "hard disk floor" in error
    assert calls == [
        ["docker", "pull", "-q", "repo/image:tag"],
        ["docker", "rmi", "repo/image:tag"],
    ]


def test_load_valid_predictions_excludes_rows_without_trajectories(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "b00"
    model = batch / "model"
    model.mkdir(parents=True)
    (model / "preds.json").write_text(
        json.dumps(
            {
                "valid": {"instance_id": "valid", "model_patch": "diff"},
                "missing": {"instance_id": "missing", "model_patch": ""},
            }
        )
    )
    trajectory = model / "valid" / "valid.traj.json"
    trajectory.parent.mkdir()
    trajectory.write_text(json.dumps({"instance_id": "valid", "messages": []}))

    predictions, missing = retest_empties.load_valid_predictions(batch, "model")

    assert predictions == {
        "valid": {"instance_id": "valid", "model_patch": "diff"},
    }
    assert missing == ["missing"]


def test_find_existing_report_ignores_stale_root_run_id(tmp_path: Path) -> None:
    find_existing_report = getattr(retest_empties, "find_existing_report", None)
    assert callable(find_existing_report), "durable report reuse is not implemented"
    report_dir = tmp_path / "batch" / "report"
    report_dir.mkdir(parents=True)
    expected = {
        "resolved_ids": ["a"],
        "unresolved_ids": ["b"],
        "resolved_instances": 1,
    }
    (tmp_path / "openai__model.other-run.json").write_text(json.dumps(expected))
    (tmp_path / "openai__model.target-run.json").write_text(json.dumps(expected))

    report = find_existing_report(
        report_dir,
        "target-run",
        root=tmp_path,
    )

    assert report == {}

    prediction = tmp_path / "batch" / "model" / "preds_scored.json"
    prediction.parent.mkdir()
    prediction.write_text(
        json.dumps({
            "a": {"model_patch": "diff-a"},
            "b": {"model_patch": "diff-b"},
        })
    )
    report_path = report_dir / "official_report.json"
    report_path.write_text(json.dumps(expected))
    report = find_existing_report(report_dir, "target-run", root=tmp_path)

    assert report == {}

    (report_dir / "official_report_binding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": "target-run",
            "predictions": _artifact_binding(prediction),
            "official_report": _artifact_binding(report_path),
        })
    )
    report = find_existing_report(report_dir, "target-run", root=tmp_path)

    assert report == expected


def test_find_existing_report_rejects_changed_scored_predictions(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "batch" / "report"
    report_dir.mkdir(parents=True)
    prediction = tmp_path / "batch" / "model" / "preds_scored.json"
    prediction.parent.mkdir()
    prediction.write_text(json.dumps({"case": {"model_patch": "diff-old"}}))
    report_path = report_dir / "official_report.json"
    report_path.write_text(
        json.dumps({
            "resolved_ids": ["case"],
            "unresolved_ids": [],
            "resolved_instances": 1,
        })
    )
    (report_dir / "official_report_binding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": "target-run",
            "predictions": _artifact_binding(prediction),
            "official_report": _artifact_binding(report_path),
        })
    )

    prediction.write_text(json.dumps({"case": {"model_patch": "diff-new"}}))

    assert retest_empties.find_existing_report(
        report_dir,
        "target-run",
    ) == {}


def test_find_existing_report_rejects_non_object_binding(tmp_path: Path) -> None:
    report_dir = tmp_path / "batch" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "official_report.json").write_text(
        json.dumps({"resolved_ids": [], "resolved_instances": 0})
    )
    (report_dir / "official_report_binding.json").write_text("[]")

    try:
        report = retest_empties.find_existing_report(
            report_dir,
            "target-run",
        )
    except Exception as exc:
        pytest.fail(f"malformed binding must fail closed, not raise: {exc}")

    assert report == {}


def test_find_existing_report_rejects_non_list_outcome_ids(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "batch" / "report"
    report_dir.mkdir(parents=True)
    prediction = tmp_path / "batch" / "model" / "preds_scored.json"
    prediction.parent.mkdir()
    prediction.write_text(
        json.dumps({"case": {"instance_id": "case", "model_patch": "diff"}})
    )
    report_path = report_dir / "official_report.json"
    report_path.write_text(
        json.dumps({
            "resolved_ids": {"case": True},
            "unresolved_ids": [],
            "resolved_instances": 1,
            "unresolved_instances": 0,
        })
    )
    (report_dir / "official_report_binding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": "target-run",
            "predictions": _artifact_binding(prediction),
            "official_report": _artifact_binding(report_path),
        })
    )

    assert retest_empties.find_existing_report(
        report_dir,
        "target-run",
        predictions_path=prediction,
    ) == {}


def test_official_report_rejects_present_null_submitted_ids(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "preds_scored.json"
    prediction.write_text(json.dumps({"case": {"model_patch": "diff"}}))
    report = {
        "resolved_ids": ["case"],
        "unresolved_ids": [],
        "resolved_instances": 1,
        "unresolved_instances": 0,
        "submitted_ids": None,
    }

    assert not retest_empties.official_report_valid(report, prediction)


def test_official_report_rejects_submitted_count_mismatch(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "preds_scored.json"
    prediction.write_text(json.dumps({"case": {"model_patch": "diff"}}))
    report = {
        "resolved_ids": ["case"],
        "unresolved_ids": [],
        "resolved_instances": 1,
        "unresolved_instances": 0,
        "submitted_ids": ["case"],
        "submitted_instances": 0,
    }

    assert not retest_empties.official_report_valid(report, prediction)


def test_score_rejects_unchanged_root_report(monkeypatch, tmp_path: Path) -> None:
    stale = {
        "resolved_ids": ["stale"],
        "unresolved_ids": [],
        "resolved_instances": 1,
    }
    (tmp_path / "openai__model.target-run.json").write_text(json.dumps(stale))
    prediction = tmp_path / "preds.json"
    prediction.write_text(json.dumps({"new": {"model_patch": "diff"}}))

    class Result:
        stdout = ""
        stderr = ""

    monkeypatch.setattr(retest_empties, "ROOT", tmp_path)
    monkeypatch.setattr(retest_empties.subprocess, "run", lambda *_args, **_kwargs: Result())

    report = retest_empties.score(
        prediction,
        "target-run",
        tmp_path / "batch" / "report",
    )

    assert report == {}


def test_score_retains_fresh_root_report_in_batch(monkeypatch, tmp_path: Path) -> None:
    expected = {
        "resolved_ids": ["new"],
        "unresolved_ids": [],
        "resolved_instances": 1,
    }
    prediction = tmp_path / "preds.json"
    prediction.write_text(json.dumps({"new": {"model_patch": "diff"}}))

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(*_args, **_kwargs):
        (tmp_path / "openai__model.target-run.json").write_text(json.dumps(expected))
        return Result()

    monkeypatch.setattr(retest_empties, "ROOT", tmp_path)
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)
    report_dir = tmp_path / "batch" / "report"

    report = retest_empties.score(prediction, "target-run", report_dir)

    assert report == expected
    assert json.loads((report_dir / "official_report.json").read_text()) == expected
    binding_path = report_dir / "official_report_binding.json"
    assert binding_path.is_file()
    assert json.loads(
        binding_path.read_text()
    ) == {
        "schema_version": 1,
        "run_id": "target-run",
        "predictions": _artifact_binding(prediction),
        "official_report": _artifact_binding(
            report_dir / "official_report.json"
        ),
    }
    assert retest_empties.find_existing_report(
        report_dir,
        "target-run",
    ) == expected


def test_score_rejects_predictions_changed_during_scoring(
    monkeypatch,
    tmp_path: Path,
) -> None:
    expected = {
        "resolved_ids": ["new"],
        "unresolved_ids": [],
        "resolved_instances": 1,
    }
    prediction = tmp_path / "preds.json"
    prediction.write_text(json.dumps({"new": {"model_patch": "diff-old"}}))

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(*_args, **_kwargs):
        prediction.write_text(json.dumps({"new": {"model_patch": "diff-new"}}))
        (tmp_path / "openai__model.target-run.json").write_text(
            json.dumps(expected)
        )
        return Result()

    monkeypatch.setattr(retest_empties, "ROOT", tmp_path)
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)
    report_dir = tmp_path / "batch" / "report"

    assert retest_empties.score(
        prediction,
        "target-run",
        report_dir,
    ) == {}
    assert not (report_dir / "official_report_binding.json").exists()


def test_score_uses_private_snapshot_if_predictions_change_and_restore(
    monkeypatch,
    tmp_path: Path,
) -> None:
    expected = {
        "resolved_ids": ["new"],
        "unresolved_ids": [],
        "resolved_instances": 1,
    }
    original = json.dumps({"new": {"model_patch": "diff-old"}})
    prediction = tmp_path / "preds.json"
    prediction.write_text(original)
    scored_bytes: list[str] = []

    class Result:
        stdout = ""
        stderr = ""

    def fake_run(command, **_kwargs):
        scored_path = Path(command[command.index("--predictions_path") + 1])
        prediction.write_text(json.dumps({"new": {"model_patch": "diff-new"}}))
        scored_bytes.append(scored_path.read_text())
        prediction.write_text(original)
        (tmp_path / "openai__model.target-run.json").write_text(
            json.dumps(expected)
        )
        return Result()

    monkeypatch.setattr(retest_empties, "ROOT", tmp_path)
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)
    report_dir = tmp_path / "batch" / "report"

    assert retest_empties.score(
        prediction,
        "target-run",
        report_dir,
    ) == expected
    assert scored_bytes == [original]


def test_score_or_reuse_materializes_current_predictions_before_report_check(
    monkeypatch,
    tmp_path: Path,
) -> None:
    score_or_reuse = getattr(retest_empties, "score_or_reuse_predictions", None)
    assert callable(score_or_reuse)
    run_out = tmp_path / "batch"
    name = "model"
    report_dir = run_out / "report"
    report_dir.mkdir(parents=True)
    old_scored_path = run_out / "old-model" / "preds_scored.json"
    old_scored_path.parent.mkdir()
    old_scored_path.write_text(
        json.dumps({
            "case": {
                "instance_id": "case",
                "model_patch": "diff --git a/old b/old\n",
            },
        })
    )
    old_report = {
        "resolved_ids": ["case"],
        "unresolved_ids": [],
        "resolved_instances": 1,
    }
    report_path = report_dir / "official_report.json"
    report_path.write_text(json.dumps(old_report))
    (report_dir / "official_report_binding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": "target-run",
            "predictions": _artifact_binding(old_scored_path),
            "official_report": _artifact_binding(report_path),
        })
    )
    scored_path = run_out / name / "preds_scored.json"
    scored_path.parent.mkdir()
    current = {
        "case": {
            "instance_id": "case",
            "model_patch": "diff --git a/new b/new\n",
        },
    }
    rescored = {
        "resolved_ids": [],
        "unresolved_ids": ["case"],
        "resolved_instances": 0,
    }
    calls: list[Path] = []

    def fake_score(prediction: Path, run_id: str, output: Path) -> dict:
        calls.append(prediction)
        assert run_id == "target-run"
        assert output == report_dir
        assert json.loads(prediction.read_text()) == current
        return rescored

    monkeypatch.setattr(retest_empties, "score", fake_score)

    assert score_or_reuse(
        run_out,
        name,
        current,
        "target-run",
        report_dir,
    ) == rescored
    assert calls == [scored_path]


def test_later_unresolved_attempt_replaces_older_resolution_status() -> None:
    merge = getattr(retest_empties, "merge_batch_aggregate", None)
    assert callable(merge), "batch aggregation helper is not implemented"
    predictions: dict[str, dict] = {}
    resolved: set[str] = set()
    steps: dict[str, int] = {}

    merge(
        predictions,
        resolved,
        steps,
        {"case": {"instance_id": "case", "model_patch": "old patch"}},
        ["case"],
        {"case": 40},
    )
    merge(
        predictions,
        resolved,
        steps,
        {"case": {"instance_id": "case", "model_patch": "latest patch"}},
        [],
        {"case": 75},
    )

    assert predictions["case"]["model_patch"] == "latest patch"
    assert "case" not in resolved
    assert steps["case"] == 75


def test_cleanup_uses_only_non_forced_exact_image_removals(monkeypatch) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)

    removed = retest_empties.cleanup_pulled_images(["exact-b", "exact-a", "exact-a"])

    assert removed == 2
    assert calls == [
        ["docker", "rmi", "exact-a"],
        ["docker", "rmi", "exact-b"],
    ]


def test_run_manifest_allows_identical_resume_and_rejects_drift(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    expected = {
        "schema_version": 1,
        "name": "model",
        "ids_sha256": "abc",
        "instances": 150,
        "temperature": 0.7,
        "seed": 1,
        "step_limit": 120,
    }

    retest_empties.ensure_run_manifest(run_root, expected)
    retest_empties.ensure_run_manifest(run_root, expected)

    assert json.loads((run_root / "run_manifest.json").read_text()) == expected
    with pytest.raises(ValueError, match="run manifest mismatch.*ids_sha256"):
        retest_empties.ensure_run_manifest(
            run_root,
            expected | {"ids_sha256": "different"},
        )


def test_generate_passes_the_selected_seed_to_the_harness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    class Result:
        returncode = 0

    def fake_run(command, *, cwd, env, check):
        assert cwd == tmp_path
        assert check is False
        calls.append((command, env))
        return Result()

    monkeypatch.setattr(retest_empties, "ROOT", tmp_path)
    monkeypatch.setattr(retest_empties.subprocess, "run", fake_run)

    retest_empties.generate(
        "candidate",
        8013,
        ["repo__case"],
        tmp_path / "out",
        16,
        seed=2,
    )

    assert len(calls) == 1
    assert calls[0][0] == ["bash", "phaseH_eval/smoke_single.sh"]
    assert calls[0][1]["SEED"] == "2"
