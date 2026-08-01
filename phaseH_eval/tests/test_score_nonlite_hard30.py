from __future__ import annotations

import hashlib
import json

import pytest

from phaseH_eval.score_nonlite_hard30 import (
    _trajectory_map,
    build_nonlite_acceptance,
    score_prediction,
)


def _task(instance_id: str = "fixture__repo.case_1") -> dict:
    mutation = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n"
        "-clean\n"
        "+buggy\n"
    )
    return {
        "instance_id": instance_id,
        "image_name": "fixture/image:latest",
        "task_patch_role": "bug_inducing_mutation",
        "mutation_patch_sha256": hashlib.sha256(mutation.encode()).hexdigest(),
        "patch": mutation,
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }


def test_score_prediction_runs_full_f2p_in_the_bound_image() -> None:
    calls = []
    container_id = "a" * 12
    image_id = "sha256:" + "b" * 64
    patch = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    mutation = _task()["patch"]
    suite_calls = 0

    def run(argv, timeout, stdin):
        nonlocal suite_calls
        calls.append((argv, timeout, stdin))
        if argv[:3] == ["docker", "image", "inspect"]:
            return image_id + "\n", 0
        if argv[:3] == ["docker", "run", "-d"]:
            return container_id + "\n", 0
        if argv[:4] == ["docker", "exec", "-i", container_id]:
            if stdin == mutation:
                return "", 0
            if stdin == patch:
                return "", 0
            ids = json.loads(stdin)
            assert ids == ["tests/test_x.py::test_fix"]
            suite_calls += 1
            if suite_calls == 2:
                return "1 failed in 0.01s\n", 1
            return "1 passed in 0.01s\n", 0
        if argv[:3] == ["docker", "exec", container_id]:
            assert "commit --no-verify" in argv[-1]
            return "c" * 40 + "\n", 0
        if argv[:3] == ["docker", "rm", "-f"]:
            return "", 0
        raise AssertionError(argv)

    result = score_prediction(
        _task(),
        {"instance_id": "fixture__repo.case_1", "model_patch": patch},
        expected_image_id=image_id,
        run=run,
    )

    assert result["resolved"] is True
    assert result["task_baseline_valid"] is True
    assert result["reference_f2p_pass"] is True
    assert result["mutation_f2p_pass"] is False
    assert result["mutation_patch_applied"] is True
    assert result["mutation_baseline_commit"] == "c" * 40
    assert result["f2p_pass"] is True
    assert result["f2p_commands"] == [
        "python -m pytest -x -q tests/test_x.py::test_fix"
    ]
    mutation_apply = next(call for call in calls if call[2] == mutation)
    candidate_apply = next(call for call in calls if call[2] == patch)
    assert mutation_apply[0][:5] == ["docker", "exec", "-i", container_id, "bash"]
    assert "git apply --whitespace" in mutation_apply[0][-1]
    assert "git add -A" in mutation_apply[0][-1]
    assert "git apply --whitespace" in candidate_apply[0][-1]
    assert calls[-1][0] == ["docker", "rm", "-f", container_id]


def test_score_prediction_marks_invalid_mutation_as_infrastructure() -> None:
    task = _task()
    image_id = "sha256:" + "b" * 64
    container_id = "a" * 12

    def run(argv, _timeout, stdin):
        if argv[:3] == ["docker", "image", "inspect"]:
            return image_id + "\n", 0
        if argv[:3] == ["docker", "run", "-d"]:
            return container_id + "\n", 0
        if argv[:4] == ["docker", "exec", "-i", container_id]:
            if stdin == task["patch"]:
                return "does not apply", 1
            return "1 passed\n", 0
        if argv[:3] == ["docker", "rm", "-f"]:
            return "", 0
        raise AssertionError(argv)

    result = score_prediction(
        task,
        {"instance_id": task["instance_id"], "model_patch": ""},
        expected_image_id=image_id,
        run=run,
    )

    assert result["status"] == "invalid_task"
    assert result["infra_error"] == "task mutation apply failed"
    assert result["mutation_patch_applied"] is False


def test_score_prediction_rejects_test_mutation_after_task_preflight() -> None:
    patch = "diff --git a/tests/test_x.py b/tests/test_x.py\n"
    task = _task()
    image_id = "sha256:" + "b" * 64
    container_id = "a" * 12
    suite_calls = 0

    def run(argv, _timeout, stdin):
        nonlocal suite_calls
        if argv[:3] == ["docker", "image", "inspect"]:
            return image_id + "\n", 0
        if argv[:3] == ["docker", "run", "-d"]:
            return container_id + "\n", 0
        if argv[:4] == ["docker", "exec", "-i", container_id]:
            assert stdin != patch
            if stdin == task["patch"]:
                return "", 0
            suite_calls += 1
            return ("1 failed\n", 1) if suite_calls == 2 else ("1 passed\n", 0)
        if argv[:3] == ["docker", "exec", container_id]:
            return "c" * 40 + "\n", 0
        if argv[:3] == ["docker", "rm", "-f"]:
            return "", 0
        raise AssertionError(argv)

    result = score_prediction(
        task,
        {"instance_id": "fixture__repo.case_1", "model_patch": patch},
        expected_image_id=image_id,
        run=run,
    )

    assert result["resolved"] is False
    assert result["status"] == "unsafe_patch"
    assert result["task_baseline_valid"] is True
    assert result["protected_patch_paths"] == ["tests/test_x.py"]


def test_build_nonlite_acceptance_counts_health_and_infra_exactly() -> None:
    tasks = {
        f"fixture__repo.case_{index}": _task(f"fixture__repo.case_{index}")
        for index in range(30)
    }
    predictions = {
        instance_id: {
            "instance_id": instance_id,
            "model_patch": (
                "diff --git a/src/x.py b/src/x.py\n" if index < 20 else ""
            ),
        }
        for index, instance_id in enumerate(tasks)
    }
    trajectories = {
        instance_id: {
            "instance_id": instance_id,
            "info": {
                "task_mutation": {
                    "schema_version": 1,
                    "instance_id": instance_id,
                    "task_patch_role": "bug_inducing_mutation",
                    "mutation_patch_sha256": tasks[instance_id][
                        "mutation_patch_sha256"
                    ],
                    "mutation_baseline_commit": "c" * 40,
                },
            },
            "messages": [],
        }
        for instance_id in tasks
    }
    scores = {
        instance_id: {
            "instance_id": instance_id,
            "status": "scored" if index < 20 else "empty_patch",
            "resolved": index < 12,
            "infra_error": None,
            "task_baseline_valid": True,
        }
        for index, instance_id in enumerate(tasks)
    }

    acceptance = build_nonlite_acceptance(
        tasks=tasks,
        predictions=predictions,
        trajectories=trajectories,
        scores=scores,
    )

    assert acceptance["status"] == "complete"
    assert acceptance["expected"] == acceptance["preds"] == 30
    assert acceptance["traj_files"] == acceptance["usable_outcomes"] == 30
    assert acceptance["resolved"] == 12
    assert acceptance["nonempty"] == 20
    assert acceptance["infra_failed"] == 0
    assert acceptance["mutation_bound_trajectories"] == 30
    assert acceptance["repeat_loops"] == 0

    scores["fixture__repo.case_0"]["infra_error"] = "container start failed"
    assert build_nonlite_acceptance(
        tasks=tasks,
        predictions=predictions,
        trajectories=trajectories,
        scores=scores,
    )["status"] == "incomplete"

    scores["fixture__repo.case_0"]["infra_error"] = None
    trajectories["fixture__repo.case_0"]["info"]["task_mutation"][
        "mutation_patch_sha256"
    ] = "0" * 64
    assert build_nonlite_acceptance(
        tasks=tasks,
        predictions=predictions,
        trajectories=trajectories,
        scores=scores,
    )["status"] == "incomplete"


def test_trajectory_map_rejects_duplicate_identity(tmp_path) -> None:
    for batch in ("b00", "b01"):
        path = tmp_path / batch / "case" / f"{batch}.traj.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "instance_id": "case",
            "messages": [],
        }))

    with pytest.raises(ValueError, match="duplicate trajectory"):
        _trajectory_map(tmp_path)
