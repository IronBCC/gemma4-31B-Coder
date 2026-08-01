#!/usr/bin/env python3
"""Score a frozen SWE-smith hard30 with exact F2P execution evidence."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from phaseH_eval.eval_v2p10_promotion import exact_repeated_failure_loops
from teacher_platform.generic_trace_replay import (
    _run_suite,
    parse_patch_paths,
    protected_paths,
    strict_test_run_failed,
    test_invocations,
)


ENV_BOOTSTRAP = (
    "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true; "
)
_FORMAT_ERROR_RE = re.compile(r"tool call error", re.IGNORECASE)
RunCommand = Callable[[list[str], int, str | None], tuple[str, int]]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subprocess_run(
    argv: list[str],
    timeout: int,
    stdin: str | None,
) -> tuple[str, int]:
    result = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.stdout + result.stderr, result.returncode


def score_prediction(
    task: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    expected_image_id: str,
    run: RunCommand = _subprocess_run,
) -> dict[str, Any]:
    """Establish the exact mutated task baseline, then score one candidate."""

    instance_id = task.get("instance_id")
    if (
        not isinstance(instance_id, str)
        or prediction.get("instance_id", instance_id) != instance_id
    ):
        raise ValueError("prediction identity mismatch")
    patch = prediction.get("model_patch")
    patch = patch if isinstance(patch, str) else ""
    result: dict[str, Any] = {
        "schema_version": 2,
        "instance_id": instance_id,
        "task_sha256": _sha256_bytes(_canonical_json(dict(task))),
        "patch_sha256": _sha256_bytes(patch.encode("utf-8")),
        "task_patch_role": task.get("task_patch_role"),
        "mutation_patch_sha256": task.get("mutation_patch_sha256"),
        "image_name": task.get("image_name"),
        "image_id": expected_image_id,
        "status": "empty_patch",
        "resolved": False,
        "patch_applied": False,
        "mutation_patch_applied": False,
        "mutation_baseline_commit": None,
        "task_baseline_valid": False,
        "reference_f2p_pass": False,
        "mutation_f2p_pass": False,
        "f2p_pass": False,
        "p2p_checked": False,
        "protected_patch_paths": [],
        "f2p_commands": [],
        "f2p_runs": [],
        "infra_error": None,
    }

    mutation_patch = task.get("patch")
    try:
        if task.get("task_patch_role") != "bug_inducing_mutation":
            raise ValueError("task patch role is not bug_inducing_mutation")
        if not isinstance(mutation_patch, str) or not mutation_patch.strip():
            raise ValueError("task mutation patch is missing")
        mutation_sha256 = _sha256_bytes(mutation_patch.encode("utf-8"))
        if task.get("mutation_patch_sha256") != mutation_sha256:
            raise ValueError("task mutation SHA-256 binding mismatch")
        mutation_paths = parse_patch_paths(mutation_patch)
        mutation_protected = protected_paths(mutation_paths)
        if mutation_protected:
            raise ValueError(
                f"task mutation changes protected paths: {mutation_protected}"
            )
        commands = test_invocations(task.get("FAIL_TO_PASS"))
    except ValueError as exc:
        result.update(status="invalid_task", infra_error=str(exc)[:500])
        return result

    result["f2p_commands"] = commands
    candidate_status = "candidate_ready"
    if not patch.strip():
        candidate_status = "empty_patch"
    else:
        try:
            paths = parse_patch_paths(patch)
        except ValueError as exc:
            candidate_status = "malformed_patch"
            result["model_error"] = str(exc)[:500]
        else:
            protected = list(protected_paths(paths))
            result["protected_patch_paths"] = protected
            if protected:
                candidate_status = "unsafe_patch"

    container_id = ""
    try:
        image_output, image_rc = run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                str(task.get("image_name") or ""),
            ],
            60,
            None,
        )
        actual_image_id = image_output.strip()
        if image_rc != 0 or actual_image_id != expected_image_id:
            result.update(
                status="infra_error",
                infra_error="local image identity mismatch",
            )
            return result
        created, create_rc = run(
            [
                "docker",
                "run",
                "-d",
                "--network",
                "none",
                "--pids-limit",
                "512",
                str(task["image_name"]),
                "sleep",
                "infinity",
            ],
            180,
            None,
        )
        container_id = created.strip().splitlines()[-1][:12]
        if create_rc != 0 or not re.fullmatch(r"[0-9a-f]{12}", container_id):
            result.update(
                status="infra_error",
                infra_error=f"container start failed: {created[-300:]}",
            )
            return result

        reference_f2p = _run_suite(
            container_id,
            commands,
            run=run,
            timeout=900,
            env_bootstrap=ENV_BOOTSTRAP,
        )
        result["reference_f2p_pass"] = reference_f2p["passed"]
        result["reference_f2p_runs"] = reference_f2p["runs"]
        if not reference_f2p["passed"]:
            result.update(
                status="invalid_task",
                infra_error="clean reference F2P control failed",
            )
            return result

        mutation_output, mutation_rc = run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "bash",
                "-c",
                (
                    "set -euo pipefail; cd /testbed; "
                    'test -z "$(git status --porcelain=v1 '
                    '--untracked-files=all)"; '
                    "git apply --whitespace=nowarn -; "
                    "git add -A"
                ),
            ],
            180,
            mutation_patch,
        )
        result["mutation_apply_returncode"] = mutation_rc
        result["mutation_apply_output_sha256"] = _sha256_bytes(
            mutation_output.encode("utf-8", errors="replace")
        )
        if mutation_rc != 0:
            result.update(
                status="invalid_task",
                infra_error="task mutation apply failed",
                mutation_apply_output_tail=mutation_output[-2000:],
            )
            return result
        result["mutation_patch_applied"] = True

        commit_output, commit_rc = run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                (
                    "set -euo pipefail; cd /testbed; "
                    "git -c user.name='SWE-smith Task' "
                    "-c user.email='swe-smith-task@invalid' "
                    "-c commit.gpgsign=false commit --no-verify "
                    "-m swe-smith-task-mutation >/dev/null; "
                    'test -z "$(git status --porcelain=v1 '
                    '--untracked-files=all)"; '
                    "git rev-parse HEAD"
                ),
            ],
            180,
            None,
        )
        mutation_commit = (
            commit_output.strip().splitlines()[-1]
            if commit_output.strip()
            else ""
        )
        if commit_rc != 0 or not re.fullmatch(
            r"[0-9a-f]{40,64}",
            mutation_commit,
        ):
            result.update(
                status="invalid_task",
                infra_error="task mutation commit failed",
                mutation_commit_output_tail=commit_output[-2000:],
            )
            return result
        result["mutation_baseline_commit"] = mutation_commit

        mutation_f2p = _run_suite(
            container_id,
            commands,
            run=run,
            timeout=900,
            env_bootstrap=ENV_BOOTSTRAP,
        )
        result["mutation_f2p_pass"] = mutation_f2p["passed"]
        result["mutation_f2p_runs"] = mutation_f2p["runs"]
        exact_mutation_failure = bool(mutation_f2p["runs"]) and all(
            strict_test_run_failed(
                run_row["command"],
                run_row["output_tail"],
                run_row["returncode"],
            )
            for run_row in mutation_f2p["runs"]
        )
        if mutation_f2p["passed"] or not exact_mutation_failure:
            result.update(
                status="invalid_task",
                infra_error="task mutation did not reproduce exact F2P failure",
            )
            return result
        result["task_baseline_valid"] = True

        if candidate_status != "candidate_ready":
            result["status"] = candidate_status
            return result

        apply_output, apply_rc = run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "bash",
                "-c",
                "cd /testbed && git apply --whitespace=nowarn -",
            ],
            180,
            patch,
        )
        if apply_rc != 0:
            result.update(
                status="patch_apply_failed",
                patch_apply_returncode=apply_rc,
                patch_apply_output_sha256=_sha256_bytes(
                    apply_output.encode("utf-8", errors="replace")
                ),
                patch_apply_output_tail=apply_output[-2000:],
            )
            return result
        result["patch_applied"] = True
        suite = _run_suite(
            container_id,
            commands,
            run=run,
            timeout=900,
            env_bootstrap=ENV_BOOTSTRAP,
        )
        f2p_runs = suite["runs"]
        result["f2p_runs"] = f2p_runs
        result["f2p_pass"] = suite["passed"]
        result["resolved"] = result["f2p_pass"]
        result["status"] = "scored"
        return result
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(
            status="infra_error",
            infra_error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        return result
    finally:
        if container_id:
            try:
                run(["docker", "rm", "-f", container_id], 60, None)
            except (OSError, subprocess.SubprocessError):
                pass


def build_nonlite_acceptance(
    *,
    tasks: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one exact 30-task run without counting missing evidence as failure."""

    expected_ids = set(tasks)
    if len(expected_ids) != 30:
        raise ValueError(f"expected exactly 30 frozen tasks, got {len(expected_ids)}")
    for name, rows in (
        ("predictions", predictions),
        ("trajectories", trajectories),
        ("scores", scores),
    ):
        unexpected = set(rows) - expected_ids
        if unexpected:
            raise ValueError(f"{name} contain unexpected IDs: {sorted(unexpected)}")

    def mutation_evidence_is_exact(instance_id: str) -> bool:
        trajectory = trajectories.get(instance_id)
        task = tasks.get(instance_id)
        if not isinstance(trajectory, Mapping) or not isinstance(task, Mapping):
            return False
        info = trajectory.get("info")
        evidence = (
            info.get("task_mutation")
            if isinstance(info, Mapping)
            else None
        )
        return bool(
            isinstance(evidence, Mapping)
            and evidence.get("schema_version") == 1
            and evidence.get("instance_id") == instance_id
            and evidence.get("task_patch_role") == "bug_inducing_mutation"
            and evidence.get("mutation_patch_sha256")
            == task.get("mutation_patch_sha256")
            and isinstance(evidence.get("mutation_baseline_commit"), str)
            and re.fullmatch(
                r"[0-9a-f]{40,64}",
                str(evidence.get("mutation_baseline_commit")),
            )
        )

    mutation_bound = sum(
        mutation_evidence_is_exact(instance_id)
        for instance_id in expected_ids
    )
    infra_failed = sum(
        bool(row.get("infra_error"))
        for row in scores.values()
    )
    usable = sum(
        instance_id in predictions
        and instance_id in trajectories
        and instance_id in scores
        and not scores[instance_id].get("infra_error")
        and scores[instance_id].get("task_baseline_valid") is True
        and mutation_evidence_is_exact(instance_id)
        for instance_id in expected_ids
    )
    resolved_ids = sorted(
        instance_id
        for instance_id, row in scores.items()
        if row.get("resolved") is True and not row.get("infra_error")
    )
    nonempty = sum(
        bool(str(row.get("model_patch") or "").strip())
        for row in predictions.values()
    )
    format_errors = 0
    assistant_responses = 0
    repeat_loops = 0
    for trajectory in trajectories.values():
        repeat_loops += exact_repeated_failure_loops(trajectory)
        messages = trajectory.get("messages")
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "assistant":
                assistant_responses += 1
            content = message.get("content")
            if isinstance(content, str) and _FORMAT_ERROR_RE.search(content):
                format_errors += 1
    problems = []
    if len(predictions) != 30:
        problems.append(f"preds={len(predictions)}/30")
    if len(trajectories) != 30:
        problems.append(f"traj_files={len(trajectories)}/30")
    if usable != 30:
        problems.append(f"usable={usable}/30")
    if infra_failed:
        problems.append(f"infra_failed={infra_failed}")
    return {
        "schema_version": 2,
        "training_eligible": False,
        "resolution_contract": (
            "clean F2P passes; mutation reproduces exact F2P failure; "
            "candidate passes all frozen F2P IDs; P2P not checked"
        ),
        "status": "complete" if not problems else "incomplete",
        "expected": 30,
        "preds": len(predictions),
        "traj_files": len(trajectories),
        "usable_outcomes": usable,
        "mutation_bound_trajectories": mutation_bound,
        "resolved": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "empty": len(predictions) - nonempty,
        "nonempty": nonempty,
        "infra_failed": infra_failed,
        "repeat_loops": repeat_loops,
        "tool_format_errors": format_errors,
        "assistant_responses": assistant_responses,
        "format_error_rate": (
            format_errors / assistant_responses
            if assistant_responses
            else None
        ),
        "problems": problems,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(row)
    return rows


def _trajectory_map(root: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    rows = {}
    paths = sorted(root.glob("**/*.traj.json"))
    for path in paths:
        row = _read_json(path)
        instance_id = row.get("instance_id") or path.parent.name
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"trajectory is missing instance ID: {path}")
        if instance_id in rows:
            raise ValueError(f"duplicate trajectory identity: {instance_id}")
        rows[instance_id] = row
    return rows, paths


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


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in [1,8]")

    manifest = _read_json(args.task_manifest)
    task_rows = _read_jsonl(args.tasks)
    tasks = {row["instance_id"]: row for row in task_rows}
    if (
        manifest.get("schema_version") != 3
        or manifest.get("complete") is not True
        or manifest.get("training_eligible") is not False
        or manifest.get("selected_rows") != 30
        or manifest.get("tasks_sha256") != _sha256_path(args.tasks)
        or len(tasks) != len(task_rows)
        or len(task_rows) != 30
    ):
        raise ValueError("frozen hard30 manifest does not bind exactly 30 tasks")
    bindings = {
        row["instance_id"]: row
        for row in manifest.get("task_bindings") or []
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
    }
    if set(bindings) != set(tasks):
        raise ValueError("hard30 task bindings differ from task rows")
    preflight = manifest.get("baseline_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version") != 1
        or preflight.get("valid_rows", 0) < 30
        or preflight.get("eligible_rows")
        != preflight.get("valid_rows", 0) + preflight.get("invalid_rows", 0)
    ):
        raise ValueError("hard30 baseline preflight is incomplete")
    preflight_summary_path = (
        args.task_manifest.parent / "baseline_preflight" / "summary.json"
    )
    if _read_json(preflight_summary_path) != preflight:
        raise ValueError("hard30 baseline preflight summary mismatch")
    preflight_bindings = {
        row["instance_id"]: row
        for row in preflight.get("result_bindings") or []
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
    }
    if len(preflight_bindings) != preflight.get("eligible_rows"):
        raise ValueError("hard30 baseline preflight bindings are incomplete")
    selected_preflight_paths = []
    for instance_id, task in tasks.items():
        preflight_binding = preflight_bindings.get(instance_id)
        if not isinstance(preflight_binding, dict):
            raise ValueError(f"missing baseline preflight: {instance_id}")
        preflight_path = (
            args.task_manifest.parent / str(preflight_binding.get("path") or "")
        ).resolve()
        try:
            preflight_path.relative_to(args.task_manifest.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"baseline preflight path escapes task root: {instance_id}"
            ) from exc
        preflight_result = _read_json(preflight_path)
        if (
            bindings[instance_id].get("task_sha256")
            != _sha256_bytes(_canonical_json(task))
            or bindings[instance_id].get("task_patch_role")
            != "bug_inducing_mutation"
            or task.get("task_patch_role") != "bug_inducing_mutation"
            or bindings[instance_id].get("mutation_patch_sha256")
            != task.get("mutation_patch_sha256")
            or task.get("mutation_patch_sha256")
            != _sha256_bytes(str(task.get("patch") or "").encode("utf-8"))
            or bindings[instance_id].get(
                "baseline_preflight_result_sha256"
            )
            != preflight_binding.get("sha256")
            or preflight_binding.get("sha256") != _sha256_path(preflight_path)
            or preflight_result.get("instance_id") != instance_id
            or preflight_result.get("task_sha256")
            != bindings[instance_id].get("task_sha256")
            or preflight_result.get("status") != "empty_patch"
            or preflight_result.get("reference_f2p_pass") is not True
            or preflight_result.get("mutation_f2p_pass") is not False
            or preflight_result.get("mutation_patch_applied") is not True
            or preflight_result.get("task_baseline_valid") is not True
            or preflight_result.get("infra_error") is not None
        ):
            raise ValueError(f"hard30 task hash mismatch: {instance_id}")
        selected_preflight_paths.append(preflight_path)

    generation_root = args.generation_root or args.run_root
    model_root = generation_root / args.name
    predictions_path = model_root / "preds.json"
    predictions = _read_json(predictions_path)
    trajectories, trajectory_paths = _trajectory_map(model_root)
    score_root = args.run_root / "nonlite_score"
    result_root = score_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    implementation_sha256 = _sha256_path(Path(__file__))
    score_manifest = score_root / "score_manifest.json"
    expected_score_manifest = {
        "schema_version": 3,
        "name": args.name,
        "generation_root": str(generation_root.resolve()),
        "tasks_sha256": _sha256_path(args.tasks),
        "task_manifest_sha256": _sha256_path(args.task_manifest),
        "implementation_sha256": implementation_sha256,
        "resolution_contract": (
            "clean F2P passes; mutation reproduces exact F2P failure; "
            "candidate passes all frozen F2P IDs; P2P not checked"
        ),
    }
    if score_manifest.exists():
        if _read_json(score_manifest) != expected_score_manifest:
            raise ValueError("score manifest mismatch")
    else:
        _publish_json_noreplace(score_manifest, expected_score_manifest)

    def score_one(instance_id: str) -> dict[str, Any]:
        result_path = result_root / (
            hashlib.sha256(instance_id.encode("utf-8")).hexdigest() + ".json"
        )
        if result_path.exists():
            existing = _read_json(result_path)
            if (
                existing.get("instance_id") != instance_id
                or existing.get("schema_version") != 2
                or existing.get("task_sha256")
                != _sha256_bytes(_canonical_json(tasks[instance_id]))
                or existing.get("patch_sha256")
                != _sha256_bytes(
                    str(predictions[instance_id].get("model_patch") or "").encode(
                        "utf-8"
                    )
                )
            ):
                raise ValueError(f"stale score result: {instance_id}")
            return existing
        result = score_prediction(
            tasks[instance_id],
            predictions[instance_id],
            expected_image_id=bindings[instance_id]["image_id"],
        )
        _publish_json_noreplace(result_path, result)
        return result

    score_ids = sorted(set(tasks) & set(predictions) & set(trajectories))
    with ThreadPoolExecutor(max_workers=min(args.workers, len(score_ids) or 1)) as pool:
        score_rows = list(pool.map(score_one, score_ids))
    scores = {row["instance_id"]: row for row in score_rows}
    acceptance = build_nonlite_acceptance(
        tasks=tasks,
        predictions=predictions,
        trajectories=trajectories,
        scores=scores,
    )
    acceptance["run_id"] = args.run_root.name
    acceptance["generation_run_id"] = generation_root.name
    acceptance["name"] = args.name
    acceptance["tasks_sha256"] = _sha256_path(args.tasks)
    acceptance["task_manifest_sha256"] = _sha256_path(args.task_manifest)
    acceptance["artifact_bindings"] = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in [
            args.tasks,
            args.task_manifest,
            preflight_summary_path,
            *selected_preflight_paths,
            predictions_path,
            score_manifest,
            *trajectory_paths,
            *sorted(result_root.glob("*.json")),
        ]
    ]
    _publish_json_noreplace(args.run_root / "acceptance.json", acceptance)
    print(json.dumps({
        "status": acceptance["status"],
        "resolved": acceptance["resolved"],
        "nonempty": acceptance["nonempty"],
        "repeat_loops": acceptance["repeat_loops"],
        "format_error_rate": acceptance["format_error_rate"],
        "infra_failed": acceptance["infra_failed"],
    }, sort_keys=True))
    return 0 if acceptance["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
