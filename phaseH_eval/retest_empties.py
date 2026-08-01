#!/usr/bin/env python3
"""Re-run the empty-patch instances of one adapter with the step-limit fix in place.

Why batched: 135 of the 172 union instances have no local docker image, each costs ~2.6 GB
on /var/lib/docker, and only ~107 GB is free (the rest is Codex's live build cache).  Pulling
all of them at once is what produced the 12 exit-127 "empty patches" in the selfverify probe.
So: pull a batch sequentially, generate, score it, drop only the images this batch pulled.

Usage:
  python3 retest_empties.py --name teacher_sft_v2p8 --port 8013 \
      --ids data/empties_v2p8.json --runid retest_v2p8 [--batch 20] [--pull-workers 4]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path.home() / "projects" / "gemma4-31B-Coder"
CONFIG = "swebench_edit_first_selfretry_s120.yaml"
MIN_FREE_GB = 35.0
PULL_HEADROOM_GB = 15.0
MIN_PULL_START_GB = MIN_FREE_GB + PULL_HEADROOM_GB
_PULL_LOCK = threading.Lock()
DOCKER_TRANSACTION_LOCK_PATH = Path(
    os.environ.get("SWE_DOCKER_TRANSACTION_LOCK", "/tmp/swebench-docker-transaction.lock")
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def image_for(instance: str) -> str:
    repo, num = instance.split("__")
    return f"swebench/sweb.eval.x86_64.{repo}_1776_{num}:latest"


def local_images() -> set[str]:
    out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                         capture_output=True, text=True).stdout.split()
    return set(out)


def free_gb(path: str = "/var/lib/docker") -> float:
    st = shutil.disk_usage(path)
    return st.free / 2**30


@contextmanager
def docker_transaction_lock(lock_path: Path | None = None):
    path = lock_path or DOCKER_TRANSACTION_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def pull(image: str) -> tuple[str, bool, str]:
    """Pull with one retry — a first-attempt failure under concurrency is what produced the
    exit-127 'empty patches' when the harness pulled on demand instead."""
    with _PULL_LOCK, docker_transaction_lock():
        err = ""
        for attempt in range(2):
            available = free_gb()
            if available < MIN_PULL_START_GB:
                return (
                    image,
                    False,
                    f"disk admission refused: {available:.1f} GiB free, "
                    f"need {MIN_PULL_START_GB:.1f} GiB",
                )
            proc = subprocess.run(
                ["docker", "pull", "-q", image],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                remaining = free_gb()
                if remaining < MIN_FREE_GB:
                    cleanup = subprocess.run(
                        ["docker", "rmi", image],
                        capture_output=True,
                        text=True,
                    )
                    cleanup_note = (
                        "exact image removed"
                        if cleanup.returncode == 0
                        else f"exact-image cleanup failed with exit {cleanup.returncode}"
                    )
                    return (
                        image,
                        False,
                        f"hard disk floor crossed after pull: {remaining:.1f} GiB free; "
                        f"{cleanup_note}",
                    )
                return image, True, ""
            lines = (proc.stderr or "").strip().splitlines()
            err = lines[-1] if lines else f"exit {proc.returncode}"
            if attempt == 0:
                time.sleep(10)
        return image, False, err


def repo_of(image: str) -> str:
    """swebench/sweb.eval.x86_64.<repo>_1776_<num>:latest -> <repo>"""
    tail = image.split("sweb.eval.x86_64.", 1)[-1]
    return tail.split("_1776_", 1)[0]


def pull_all(images: list[str], workers: int) -> list[tuple[str, str]]:
    """Pull with NO two images of the same repo in flight at once.

    Images from one repo (e.g. 15 matplotlib tasks) share large layers, and concurrent pulls of
    those layers corrupt containerd's snapshotter — observed as 'failed to commit snapshot',
    'commit failed: rename', and 'lease does not exist: not found'.  So: one worker per repo,
    sequential within a repo, then a fully serialized retry for whatever still failed.
    """
    by_repo: dict[str, list[str]] = {}
    for image in images:
        by_repo.setdefault(repo_of(image), []).append(image)

    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(by_repo)))) as pool:
        for results in pool.map(lambda group: [pull(i) for i in group], by_repo.values()):
            for image, ok, err in results:
                if not ok:
                    failed.append((image, err))

    if failed:
        log(f"  {len(failed)} pulls failed concurrently; retrying serialized")
        still: list[tuple[str, str]] = []
        for image, _ in failed:
            _, ok, err = pull(image)
            if not ok:
                still.append((image, err))
                log(f"  pull FAILED (serialized too) {image}: {err[:120]}")
        return still
    return []


def batches(ids: list[str], size: int) -> list[list[str]]:
    """Group by repo first so shared docker layers land in the same batch."""
    ordered = sorted(ids, key=lambda i: (i.split("__")[0], i))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def generate(
    name: str,
    port: int,
    ids: list[str],
    out: Path,
    workers: int,
    *,
    seed: int,
) -> None:
    env = dict(os.environ)
    env.update(
        NAME=name, PORT=str(port), SUBSET="lite", SPLIT="test", SLICE="0:300",
        FILTER="^(" + "|".join(ids) + ")$", WORKERS=str(workers), TEMPERATURE="0.7",
        SEED=str(seed), CONFIG=CONFIG, SCORE="0", OUT=str(out),
    )
    subprocess.run(["bash", "phaseH_eval/smoke_single.sh"], cwd=ROOT, env=env, check=False)


def find_existing_report(
    report_dir: Path,
    run_id: str,
    *,
    predictions_path: Path | None = None,
    root: Path | None = None,
) -> dict:
    """Load only the batch-retained report; root reports may belong to an old attempt."""

    del root
    candidate = report_dir / "official_report.json"
    try:
        data = json.loads(candidate.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    binding_path = report_dir / "official_report_binding.json"
    try:
        binding = json.loads(binding_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(binding, dict):
        return {}
    predictions_binding = binding.get("predictions")
    report_binding = binding.get("official_report")
    if (
        binding.get("schema_version") != 1
        or binding.get("run_id") != run_id
        or not isinstance(predictions_binding, dict)
        or not isinstance(report_binding, dict)
        or not artifact_binding_current(predictions_binding)
        or not artifact_binding_current(report_binding)
        or (
            predictions_path is not None
            and predictions_binding.get("path")
            != str(predictions_path.resolve())
        )
        or report_binding.get("path") != str(candidate.resolve())
    ):
        return {}
    bound_predictions_path = Path(str(predictions_binding["path"]))
    if official_report_valid(
        data,
        predictions_path or bound_predictions_path,
    ):
        return data
    return {}


def artifact_binding(path: Path) -> dict[str, object]:
    return artifact_binding_for_bytes(path, path.read_bytes())


def artifact_binding_for_bytes(
    path: Path,
    data: bytes,
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def artifact_binding_current(binding: object) -> bool:
    if not isinstance(binding, dict):
        return False
    path = Path(str(binding.get("path") or ""))
    try:
        return (
            path.is_file()
            and path.stat().st_size == binding.get("bytes")
            and hashlib.sha256(path.read_bytes()).hexdigest()
            == binding.get("sha256")
        )
    except OSError:
        return False


def official_report_valid(data: object, predictions_path: Path) -> bool:
    if not isinstance(data, dict):
        return False
    try:
        predictions = json.loads(predictions_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(predictions, dict):
        return False
    prediction_ids = set(predictions)
    outcomes: dict[str, list[str]] = {}
    for key in ("resolved_ids", "unresolved_ids"):
        ids = data.get(key)
        if (
            not isinstance(ids, list)
            or any(not isinstance(instance_id, str) or not instance_id for instance_id in ids)
            or len(set(ids)) != len(ids)
        ):
            return False
        outcomes[key] = ids
    resolved = set(outcomes["resolved_ids"])
    unresolved = set(outcomes["unresolved_ids"])
    if resolved & unresolved or not (resolved | unresolved) <= prediction_ids:
        return False
    for ids_key, count_key in (
        ("resolved_ids", "resolved_instances"),
        ("unresolved_ids", "unresolved_instances"),
    ):
        if count_key in data and (
            type(data[count_key]) is not int
            or data[count_key] != len(outcomes[ids_key])
        ):
            return False
    has_submitted_ids = "submitted_ids" in data
    has_submitted_count = "submitted_instances" in data
    if has_submitted_ids or has_submitted_count:
        submitted = data.get("submitted_ids")
        if (
            not has_submitted_ids
            or not has_submitted_count
            or not isinstance(submitted, list)
            or any(
                not isinstance(instance_id, str) or not instance_id
                for instance_id in submitted
            )
            or len(set(submitted)) != len(submitted)
            or set(submitted) != prediction_ids
            or type(data.get("submitted_instances")) is not int
            or data["submitted_instances"] != len(submitted)
        ):
            return False
    return True


def write_official_report_binding(
    report_dir: Path,
    run_id: str,
    predictions_binding: dict[str, object],
) -> None:
    report_path = report_dir / "official_report.json"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "predictions": predictions_binding,
        "official_report": artifact_binding(report_path),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=".official_report_binding.",
        dir=report_dir,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, report_dir / "official_report_binding.json")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def report_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def scorer_report_candidates(report_dir: Path, run_id: str) -> list[Path]:
    return sorted(report_dir.glob("*.json")) + sorted(ROOT.glob(f"*.{run_id}.json"))


def score(pred: Path, run_id: str, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    scored_predictions_bytes = pred.read_bytes()
    scored_predictions_binding = artifact_binding_for_bytes(
        pred,
        scored_predictions_bytes,
    )
    before = {
        candidate: report_signature(candidate)
        for candidate in scorer_report_candidates(report_dir, run_id)
    }
    descriptor, snapshot_name = tempfile.mkstemp(
        prefix=".scored_predictions.",
        suffix=".json",
        dir=report_dir,
    )
    snapshot = Path(snapshot_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(scored_predictions_bytes)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        proc = subprocess.run(
            [str(ROOT / ".venv-eval/bin/python"), "-m", "swebench.harness.run_evaluation",
             "--dataset_name", "princeton-nlp/SWE-bench_Lite",
             "--predictions_path", str(snapshot), "--run_id", run_id,
             "--max_workers", "16", "--cache_level", "env",
             "--report_dir", str(report_dir)],
            cwd=ROOT, capture_output=True, text=True)
    finally:
        snapshot.unlink(missing_ok=True)
    (report_dir / "score.log").write_text(proc.stdout + "\n" + proc.stderr)
    if not artifact_binding_current(scored_predictions_binding):
        return {}
    for candidate in scorer_report_candidates(report_dir, run_id):
        signature = report_signature(candidate)
        if signature is None or signature == before.get(candidate):
            continue
        try:
            data = json.loads(candidate.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not official_report_valid(data, pred):
            continue
        retained = report_dir / "official_report.json"
        if candidate != retained:
            temporary = report_dir / f".official_report.json.tmp.{os.getpid()}"
            shutil.copy2(candidate, temporary)
            os.replace(temporary, retained)
        write_official_report_binding(
            report_dir,
            run_id,
            scored_predictions_binding,
        )
        return data
    return {}


def steps_of(run_out: Path, name: str) -> dict[str, int]:
    steps = {}
    for traj in run_out.glob(f"{name}/*/*.traj.json"):
        try:
            msgs = json.loads(traj.read_text()).get("messages") or []
        except (json.JSONDecodeError, OSError):
            continue
        steps[traj.parent.name] = len([m for m in msgs if m.get("role") == "assistant"])
    return steps


def load_valid_predictions(
    run_out: Path,
    name: str,
) -> tuple[dict[str, dict], list[str]]:
    """Load only predictions backed by a trajectory from the same batch."""

    pred_path = run_out / name / "preds.json"
    if not pred_path.exists():
        return {}, []
    try:
        predictions = json.loads(pred_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, []
    if not isinstance(predictions, dict):
        return {}, []

    trajectory_ids: set[str] = set()
    for trajectory_path in (run_out / name).glob("*/*.traj.json"):
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        instance_id = str(trajectory.get("instance_id") or trajectory_path.parent.name)
        if instance_id:
            trajectory_ids.add(instance_id)

    missing_trajectories = sorted(set(predictions) - trajectory_ids)
    valid = {
        instance_id: row
        for instance_id, row in predictions.items()
        if instance_id in trajectory_ids
    }
    return valid, missing_trajectories


def score_predictions_path(
    run_out: Path,
    name: str,
    predictions: dict[str, dict],
) -> Path:
    """Persist the trajectory-backed scoring subset for an auditable denominator."""

    path = run_out / name / "preds_scored.json"
    path.write_text(json.dumps(predictions, indent=2))
    return path


def scoreable_diff_predictions(
    predictions: dict[str, dict],
) -> dict[str, dict]:
    """Return only predictions containing an actual unified git diff."""

    return {
        instance_id: row
        for instance_id, row in predictions.items()
        if "diff --git " in str(row.get("model_patch") or "")
    }


def score_or_reuse_predictions(
    run_out: Path,
    name: str,
    predictions: dict[str, dict],
    run_id: str,
    report_dir: Path,
) -> dict:
    scoring_predictions = scoreable_diff_predictions(predictions)
    scoring_path = score_predictions_path(run_out, name, scoring_predictions)
    if not scoring_predictions:
        return {}
    report = find_existing_report(
        report_dir,
        run_id,
        predictions_path=scoring_path,
    )
    if report:
        return report
    return score(scoring_path, run_id, report_dir)


def batch_dirs(run_root: Path) -> list[Path]:
    """Return batch directories in numeric order, including indices above 99."""

    return sorted(
        (
            path
            for path in run_root.iterdir()
            if path.is_dir() and path.name.startswith("b") and path.name[1:].isdigit()
        ),
        key=lambda path: int(path.name[1:]),
    )


def merge_batch_aggregate(
    all_predictions: dict[str, dict],
    resolved: set[str],
    all_steps: dict[str, int],
    predictions: dict[str, dict],
    resolved_ids: list[str] | set[str],
    batch_steps: dict[str, int],
) -> None:
    """Make prediction, step, and resolution state all follow the latest batch."""

    prediction_ids = set(predictions)
    all_predictions.update(predictions)
    all_steps.update(
        {
            instance_id: steps
            for instance_id, steps in batch_steps.items()
            if instance_id in prediction_ids
        }
    )
    resolved.difference_update(prediction_ids)
    resolved.update(set(resolved_ids) & prediction_ids)


def cleanup_pulled_images(images: list[str]) -> int:
    """Remove only exact images successfully pulled for this batch, never forcibly."""

    removed = 0
    with docker_transaction_lock():
        for image in sorted(set(images)):
            proc = subprocess.run(
                ["docker", "rmi", image],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                removed += 1
    return removed


def ensure_run_manifest(run_root: Path, expected: dict) -> None:
    """Make resume semantics explicit and reject any harness or denominator drift."""

    path = run_root / "run_manifest.json"
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"invalid run manifest {path}: {exc}") from exc
        mismatches = sorted(
            key
            for key in set(current) | set(expected)
            if current.get(key) != expected.get(key)
        )
        if mismatches:
            raise ValueError(
                f"run manifest mismatch in {path}: {', '.join(mismatches)}"
            )
        return

    temporary = run_root / f".run_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(expected, indent=2))
    os.replace(temporary, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--runid", required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pull-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.seed < 1:
        ap.error("--seed must be a positive integer")

    ids_path = ROOT / args.ids
    ids = json.loads(ids_path.read_text())
    run_root = ROOT / "runs" / args.runid
    run_root.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(
        run_root,
        {
            "schema_version": 1,
            "name": args.name,
            "port": args.port,
            "ids_path": args.ids,
            "ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
            "instances": len(ids),
            "batch": args.batch,
            "workers": args.workers,
            "pull_workers": args.pull_workers,
            "config": CONFIG,
            "temperature": 0.7,
            "seed": args.seed,
            "step_limit": 120,
            "environment_class": "docker_selfretry.DockerSelfRetryEnv",
        },
    )

    # Resume at INSTANCE level, not batch level: the id list grows as we learn which runs were
    # clamp-truncated, and batch boundaries shift when it does.  Reusing "b00" positionally after
    # that would silently score an old batch's predictions as if they were the new batch's.
    done: dict[str, dict] = {}
    for batch_dir in batch_dirs(run_root):
        pred_file = batch_dir / args.name / "preds.json"
        if not pred_file.exists():
            continue
        valid, _ = load_valid_predictions(pred_file.parents[1], args.name)
        done.update(valid)
    if done:
        log(f"resume: {len(done)} instances already generated in runs/{args.runid}")
    preexisting = local_images()
    todo = [i for i in ids if i not in done]
    log(f"{args.name}: {len(ids)} instances ({len(todo)} to run), batch={args.batch}, "
        f"{sum(1 for i in todo if image_for(i) not in preexisting)} images to pull, "
        f"{free_gb():.0f} GiB free")

    all_preds: dict[str, dict] = {}
    resolved: set[str] = set()
    all_steps: dict[str, int] = {}
    pull_failed: list[str] = []
    start_index = 1 + max((int(path.name[1:]) for path in batch_dirs(run_root)), default=-1)

    for offset, batch in enumerate(batches(todo, args.batch)):
        index = start_index + offset
        out = run_root / f"b{index:02d}"
        expected_generated = batch
        pulled_for_batch: list[str] = []
        if (out / args.name / "preds.json").exists():
            log(f"batch {index}: already generated, reusing")
        else:
            need = [image_for(i) for i in batch if image_for(i) not in local_images()]
            if need and free_gb() < MIN_FREE_GB:
                log(f"ABORT: only {free_gb():.0f} GiB free on /var/lib/docker, need >{MIN_FREE_GB}")
                break
            if need:
                log(f"batch {index}: pulling {len(need)} images "
                    f"({len({repo_of(i) for i in need})} repos, {free_gb():.0f} GiB free)")
                failures = pull_all(need, args.pull_workers)
                failed_images = {image for image, _ in failures}
                pulled_for_batch = sorted(set(need) - failed_images)
                for image, _ in failures:
                    pull_failed.append(image)
            # An instance whose image never arrived would just record an exit-127 empty patch,
            # which is exactly the artefact this retest exists to remove — so skip it instead.
            runnable = [i for i in batch if image_for(i) in local_images()]
            skipped = [i for i in batch if i not in runnable]
            if skipped:
                log(f"batch {index}: SKIPPING {len(skipped)} instances with no image: {skipped}")
            if not runnable:
                continue
            expected_generated = runnable
            log(f"batch {index}: generating {len(runnable)} instances")
            generate(
                args.name,
                args.port,
                runnable,
                out,
                args.workers,
                seed=args.seed,
            )

        pred = out / args.name / "preds.json"
        if not pred.exists():
            log(f"batch {index}: NO preds.json — generation failed, see {out}")
            cleanup_pulled_images(pulled_for_batch)
            continue
        preds, missing_trajectories = load_valid_predictions(out, args.name)
        if missing_trajectories:
            log(
                f"batch {index}: EXCLUDING {len(missing_trajectories)} predictions "
                f"without trajectories: {missing_trajectories}"
            )
        missing_generated = sorted(set(expected_generated) - set(preds))
        if missing_generated:
            log(
                f"batch {index}: generation incomplete "
                f"usable={len(preds)}/{len(expected_generated)} "
                f"missing={missing_generated}"
            )
        if not preds:
            cleanup_pulled_images(pulled_for_batch)
            continue
        batch_steps = steps_of(out, args.name)

        report = score_or_reuse_predictions(
            out,
            args.name,
            preds,
            f"{args.runid}_b{index:02d}",
            out / "report",
        )
        got = report.get("resolved_ids") or report.get("resolved_instances") or []
        if isinstance(got, int):
            got = []
        merge_batch_aggregate(all_preds, resolved, all_steps, preds, got, batch_steps)
        empties = [k for k, v in preds.items() if not (v.get("model_patch") or "").strip()]
        log(f"batch {index}: n={len(preds)} resolved={len(got)} empty={len(empties)} "
            f"cumulative_resolved={len(resolved)}")

        if pulled_for_batch:
            removed = cleanup_pulled_images(pulled_for_batch)
            log(
                f"batch {index}: freed {removed}/{len(pulled_for_batch)} "
                f"batch-pulled images, {free_gb():.0f} GiB free"
            )

    # Final aggregation walks EVERY batch dir, including ones generated by an earlier invocation
    # with a shorter id list, and scores any that never got a report.
    for out in batch_dirs(run_root):
        pred = out / args.name / "preds.json"
        if not pred.exists():
            continue
        preds, missing_trajectories = load_valid_predictions(out, args.name)
        if missing_trajectories:
            log(
                f"{out.name}: excluding {len(missing_trajectories)} predictions "
                "without trajectories from final aggregation"
            )
        if not preds:
            continue
        batch_steps = steps_of(out, args.name)
        report = score_or_reuse_predictions(
            out,
            args.name,
            preds,
            f"{args.runid}_{out.name}",
            out / "report",
        )
        got = report.get("resolved_ids") or report.get("resolved_instances") or []
        if isinstance(got, int):
            got = []
        merge_batch_aggregate(all_preds, resolved, all_steps, preds, got, batch_steps)

    summary = {
        "name": args.name,
        "instances": len(ids),
        "scored": len(all_preds),
        "diff_scored": len(scoreable_diff_predictions(all_preds)),
        "resolved": sorted(resolved),
        "empty": sorted(k for k, v in all_preds.items() if not (v.get("model_patch") or "").strip()),
        "steps": all_steps,
        "pull_failed": pull_failed,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_root / "preds_all.json").write_text(json.dumps(all_preds, indent=2))
    steps = sorted(all_steps.values())
    log(f"DONE {args.name}: scored={len(all_preds)}/{len(ids)} "
        f"resolved={len(resolved)} still_empty={len(summary['empty'])} "
        f"pull_failed={len(pull_failed)} "
        f"steps median={steps[len(steps)//2] if steps else 0} max={max(steps) if steps else 0} "
        f"over39={sum(1 for s in steps if s > 39)}/{len(steps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
