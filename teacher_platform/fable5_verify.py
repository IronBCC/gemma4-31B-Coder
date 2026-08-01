"""Verify the pinned Fable-5 structural subset with trusted Docker controls.

This runner deliberately executes only each Moonshiner seed's pinned verifier.
It reconstructs candidates from the importer's typed ``Write``/``Edit``
operations; transcript Bash commands are never executed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .fable5_import import (
    DATASET_REVISION,
    SOURCE_BYTES,
    SOURCE_LFS_SHA256,
    assess_converted_trajectory,
    select_terminal_row,
)
from .fable5_replay import (
    MOONSHINER_REVISION,
    DockerExecutor,
    GitSeedSource,
    ReplayContractError,
    ReplayLedger,
    _atomic_write_0600,
    _policy_from_artifact,
    canonical_mutation_plan,
    materialize_seed,
    preflight_reference_patch,
    reconstruct_candidate,
    replay_evidence_payload,
    run_control_set,
    validate_admission_artifact,
    validate_seed_repository,
    validate_structural_sidecar,
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open JSONL artifact: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL row {path}:{line_number}"
                ) from exc
            if type(row) is not dict:
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            yield row


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_language(value: object) -> str:
    canonical = {
        "py": "python",
        "python": "python",
        "rust": "rust",
        "c++": "cpp",
        "cpp": "cpp",
    }.get(value)
    if canonical is None:
        raise ValueError(f"unsupported selected language: {value!r}")
    return canonical


def missing_admission_trajectories(
    selected_rows: Sequence[Mapping[str, Any]],
    admitted_languages: Iterable[str],
) -> frozenset[str]:
    """Return selected trajectory IDs whose language lacks a live admission."""

    admitted = frozenset(admitted_languages)
    return frozenset(
        str(row["trajectory_id"])
        for row in selected_rows
        if _canonical_language(row["row"].get("lang")) not in admitted
    )


def preflight_candidate_plan(
    contract: object,
    plan: object,
    destination: Path,
    *,
    reconstructor: Any = reconstruct_candidate,
) -> str | None:
    """Return a stable row-level reason when typed reconstruction is impossible."""

    try:
        reconstructor(contract, plan, destination)
    except ReplayContractError as exc:
        return str(exc)
    return None


def row_level_control_failure(error: ReplayContractError) -> str | None:
    """Classify only reviewed, cleanup-safe verifier behavior as row-local."""

    message = str(error)
    marker = "restricted Docker verifier process quiescence failed"
    if message == marker:
        return "verifier process quiescence failed"
    return None


def select_representative_rows(
    sidecar_rows: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Join the manifest's reviewed representatives to structural sidecar rows."""

    source = candidate_manifest.get("source")
    representatives = candidate_manifest.get("representatives")
    ceiling = candidate_manifest.get("eligible_unique_task_ceiling_before_replay")
    if (
        type(source) is not dict
        or source.get("dataset_revision") != DATASET_REVISION
        or type(representatives) is not list
        or type(ceiling) is not int
        or ceiling != len(representatives)
        or ceiling <= 0
    ):
        raise ValueError("candidate manifest does not bind the reviewed Fable subset")

    by_trajectory: dict[str, Mapping[str, Any]] = {}
    for row in sidecar_rows:
        trajectory_id = row.get("trajectory_id")
        if type(trajectory_id) is not str or trajectory_id in by_trajectory:
            raise ValueError("structural sidecar has an invalid or duplicate trajectory")
        by_trajectory[trajectory_id] = row

    selected: list[Mapping[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_trajectories: set[str] = set()
    for representative in representatives:
        if type(representative) is not dict:
            raise ValueError("candidate manifest representative is not an object")
        task = representative.get("task")
        trajectory_id = representative.get("trajectory_id")
        if (
            type(task) is not str
            or type(trajectory_id) is not str
            or task in seen_tasks
            or trajectory_id in seen_trajectories
        ):
            raise ValueError("candidate manifest representative identity is invalid")
        row = by_trajectory.get(trajectory_id)
        if row is None:
            raise ValueError(
                f"representative {trajectory_id} is missing from structural sidecar"
            )
        sidecar_task = row.get("task", row.get("source_instance_id"))
        if sidecar_task != task:
            raise ValueError("candidate manifest task does not match structural sidecar")
        seen_tasks.add(task)
        seen_trajectories.add(trajectory_id)
        selected.append(row)
    return tuple(selected)


def build_verification_manifest(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    ledger_records: Sequence[Mapping[str, Any]],
    ledger_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Summarize exact ledger outcomes without trusting external counters."""

    selected_by_id = {
        row["trajectory_id"]: row
        for row in selected_rows
        if type(row.get("trajectory_id")) is str
    }
    if len(selected_by_id) != len(selected_rows):
        raise ValueError("selected rows have duplicate or invalid trajectory IDs")

    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    verified_by_language: Counter[str] = Counter()
    completed: set[str] = set()
    for record in ledger_records:
        trajectory_id = record.get("trajectory_id")
        status = record.get("status")
        if trajectory_id not in selected_by_id or trajectory_id in completed:
            raise ValueError("ledger record is outside or duplicates the selected subset")
        if status not in {"verified", "rejected", "timeout"}:
            raise ValueError("ledger record has an invalid status")
        completed.add(trajectory_id)
        status_counts[status] += 1
        if status == "verified":
            language = _canonical_language(
                selected_by_id[trajectory_id]["row"].get("lang")
            )
            verified_by_language[language] += 1
        else:
            failure = record.get("failure_class")
            if type(failure) is not str:
                raise ValueError("failed ledger record has no failure class")
            failure_counts[failure] += 1

    selected_by_language: Counter[str] = Counter()
    for row in selected_rows:
        selected_by_language[_canonical_language(row["row"].get("lang"))] += 1

    languages = sorted(selected_by_language)
    return {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "moonshiner_revision": MOONSHINER_REVISION,
        "selected": len(selected_rows),
        "completed": len(completed),
        "verified": status_counts["verified"],
        "rejected": status_counts["rejected"],
        "timeout": status_counts["timeout"],
        "pending": len(selected_rows) - len(completed),
        "per_language": {
            language: {
                "selected": selected_by_language[language],
                "verified": verified_by_language[language],
            }
            for language in languages
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "ledger_sha256": _sha256_path(ledger_path),
    }


def _control_failure_code(failure_class: str | None) -> str:
    if failure_class == "reference_invalid_evidence":
        return "invalid_reference_patch"
    if failure_class == "candidate_not_repeatable":
        return "candidate_not_repeatable"
    if failure_class == "verifier_timeout":
        return "verifier_timeout"
    return "verifier_failed"


def _canonical_log(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_manifest(
    out: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    started: float,
) -> dict[str, Any]:
    manifest = build_verification_manifest(
        selected_rows=selected_rows,
        ledger_records=records,
        ledger_path=out / "replay.jsonl",
        elapsed_seconds=time.monotonic() - started,
    )
    _atomic_write_0600(out / "manifest.json", _canonical_log(manifest))
    return manifest


def verify_subset(
    *,
    source: Path,
    sidecar: Path,
    candidate_manifest_path: Path,
    seed_repo: Path,
    policy_path: Path,
    admission_path: Path,
    out: Path,
) -> dict[str, Any]:
    """Verify or resume every representative in one exact reviewed subset."""

    source = Path(source)
    if source.stat().st_size != SOURCE_BYTES or _sha256_path(source) != SOURCE_LFS_SHA256:
        raise ValueError("source JSONL does not match the pinned Fable-5 artifact")

    candidate_manifest = _read_json(candidate_manifest_path)
    structural = validate_structural_sidecar(
        _iter_jsonl(source), _iter_jsonl(sidecar)
    )
    selected_rows = select_representative_rows(structural, candidate_manifest)

    policy_document = _read_json(policy_path)
    admission_document = _read_json(admission_path)
    policy, _policy_sha = _policy_from_artifact(policy_document)
    admitted_languages = validate_admission_artifact(
        policy_document, admission_document
    )
    missing_admission = missing_admission_trajectories(
        selected_rows, admitted_languages
    )

    seed_source = GitSeedSource(Path(seed_repo), MOONSHINER_REVISION)
    validate_seed_repository(seed_source)
    executor = DockerExecutor(policy)

    out = Path(out)
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(out, 0o700)
    logs = out / "logs"
    started = time.monotonic()

    with ReplayLedger(out / "replay.jsonl", logs) as ledger:
        completed = ledger.completed_trajectory_ids
        total = len(selected_rows)
        for index, item in enumerate(selected_rows, start=1):
            trajectory_id = str(item["trajectory_id"])
            if trajectory_id in completed:
                continue
            task = str(item.get("task", item.get("source_instance_id")))
            row = item["row"]
            row_started = time.monotonic()
            if trajectory_id in missing_admission:
                ledger.publish_failure(
                    trajectory_id=trajectory_id,
                    status="rejected",
                    failure_class="admission_failed",
                    log=_canonical_log(
                        {
                            "task": task,
                            "failure": "selected language lacks fresh functional admission",
                        }
                    ),
                )
                continue

            with tempfile.TemporaryDirectory(
                prefix=f"fable-verify-{task}-"
            ) as temporary:
                workspace = Path(temporary)
                try:
                    contract = materialize_seed(
                        seed_source, task, workspace / "seed"
                    )
                    reference = preflight_reference_patch(contract)
                    if not reference.eligible:
                        ledger.publish_failure(
                            trajectory_id=trajectory_id,
                            status="rejected",
                            failure_class="invalid_reference_patch",
                            log=_canonical_log(
                                {
                                    "task": task,
                                    "failure": reference.exclusion_reason,
                                }
                            ),
                        )
                        continue
                    selected = select_terminal_row(row)
                    plan = canonical_mutation_plan(
                        selected.messages,
                        frozenset(contract.protected_paths),
                        contract.verify_cmd,
                    )
                    if not plan.operations:
                        ledger.publish_failure(
                            trajectory_id=trajectory_id,
                            status="rejected",
                            failure_class="unsupported_operation",
                            log=_canonical_log(
                                {"task": task, "failure": "empty mutation plan"}
                            ),
                        )
                        continue
                    assessment = assess_converted_trajectory(
                        selected,
                        protected_paths=contract.protected_paths,
                        verify_cmd=contract.verify_cmd,
                        exclusions=(),
                    )
                    preflight_failure = preflight_candidate_plan(
                        contract,
                        plan,
                        workspace / "candidate-preflight",
                    )
                    if preflight_failure is not None:
                        ledger.publish_failure(
                            trajectory_id=trajectory_id,
                            status="rejected",
                            failure_class="replay_rejected",
                            log=_canonical_log(
                                {
                                    "task": task,
                                    "failure": preflight_failure,
                                }
                            ),
                        )
                        continue
                except (ReplayContractError, ValueError) as exc:
                    ledger.publish_failure(
                        trajectory_id=trajectory_id,
                        status="rejected",
                        failure_class=(
                            "invalid_seed"
                            if "seed" in str(exc).casefold()
                            else "unsupported_operation"
                        ),
                        log=_canonical_log(
                            {
                                "task": task,
                                "exception": type(exc).__name__,
                                "failure": str(exc),
                            }
                        ),
                    )
                    continue

                try:
                    controls = run_control_set(
                        contract,
                        plan,
                        trajectory_id=trajectory_id,
                        source_terminal_sha256=str(
                            item["source_terminal_sha256"]
                        ),
                        executor=executor,
                        workspace=workspace / "controls",
                    )
                except ReplayContractError as exc:
                    control_failure = row_level_control_failure(exc)
                    if control_failure is None:
                        raise
                    ledger.publish_failure(
                        trajectory_id=trajectory_id,
                        status="rejected",
                        failure_class="replay_rejected",
                        log=_canonical_log(
                            {
                                "task": task,
                                "failure": control_failure,
                            }
                        ),
                    )
                    continue
                control_payload = json.loads(
                    _canonical_log(dataclasses.asdict(controls))
                )
                if controls.admitted:
                    candidate = controls.candidate
                    # ``run_control_set`` is the only producer allowed to make
                    # a candidate trainable; the ledger revalidates all fields.
                    ledger.publish_verified(
                        candidate,
                        source_content_sha256=assessment.content_sha256,
                        fixture_sha256=contract.fixture_sha256,
                        log=_canonical_log(control_payload),
                    )
                else:
                    ledger.publish_failure(
                        trajectory_id=trajectory_id,
                        status="rejected",
                        failure_class=_control_failure_code(
                            controls.failure_class
                        ),
                        log=_canonical_log(control_payload),
                    )

            completed_count = len(ledger.records)
            elapsed = time.monotonic() - started
            rate = completed_count / elapsed if elapsed > 0 else 0.0
            eta = (total - completed_count) / rate if rate > 0 else 0.0
            result = ledger.records[-1]
            print(
                f"[fable-verify {completed_count}/{total} "
                f"status={result['status']} task={task} "
                f"row_s={time.monotonic() - row_started:.1f} eta_s={eta:.0f}]",
                flush=True,
            )
            _write_manifest(
                out, selected_rows, ledger.records, started=started
            )

        return _write_manifest(
            out, selected_rows, ledger.records, started=started
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--seed-repo", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = verify_subset(
        source=args.source,
        sidecar=args.sidecar,
        candidate_manifest_path=args.candidate_manifest,
        seed_repo=args.seed_repo,
        policy_path=args.policy,
        admission_path=args.admission,
        out=args.out,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
