#!/usr/bin/env python3
"""Recover legacy Fable traces as verifier-corrected revision supervision.

Legacy collectors ran the teacher against the clean SWE-smith image instead of
the bug-mutated task.  Their edits are therefore never positive targets here.
Safe inspection turns may retain loss, while mutations, test runs, failures,
and repeated commands are conditioning-only.  Every rendered row ends in a
separately replay-verified repair of the actual mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from phaseD_sft.build_verified_teacher_finalpatch_sft import (
    portable_git_apply_command,
)
from phaseD_sft.teacher_trace_driver import ENV_BOOTSTRAP, sh
from teacher_platform.generic_trace_replay import (
    ReplayContractError,
    admission_is_exact,
    canonical_json_bytes,
    parse_patch_paths,
    protected_paths,
    sha256_bytes,
    verify_candidate_patch,
)
from teacher_platform.success_trace_distill import command_mutates_source
from teacher_platform.teacher_platform import (
    NormalizedTrace,
    _load_exclusion_contract,
    _publish_directory_atomic,
    _write_jsonl,
    normalize_steps,
)


SOURCE = "teacher:claude:claude-fable-5:verified-revision"
DATASET_VARIANT = "fable5_recent_verified_revision_v1"
_TEST_RE = re.compile(
    r"\b(?:pytest|unittest|tox|nox|cargo\s+test|go\s+test|npm\s+test)\b",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATCH_CAPTURE_COMMAND = (
    "set -e; cd /testbed; "
    "idx=$(mktemp); rm -f \"$idx\"; "
    "trap 'rm -f \"$idx\"' EXIT; "
    "GIT_INDEX_FILE=\"$idx\" git read-tree HEAD; "
    "GIT_INDEX_FILE=\"$idx\" git add -A -- .; "
    "GIT_INDEX_FILE=\"$idx\" git diff --cached --binary --no-renames"
)


@dataclass(frozen=True)
class RevisionStep:
    thought: str
    command: str
    observation: str
    returncode: int
    supervise: bool = False


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = canonical_json_bytes(dict(row))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _is_safe_inspection(step: RevisionStep, repeated: set[str]) -> bool:
    command = step.command.strip()
    return bool(
        command
        and step.returncode == 0
        and command not in repeated
        and not command_mutates_source(command)
        and not _TEST_RE.search(command)
    )


def select_revision_steps(
    steps: Sequence[RevisionStep],
    *,
    max_steps: int = 12,
) -> list[RevisionStep]:
    """Trim a legacy trace and label only non-repeated successful inspections."""

    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    counts = Counter(step.command.strip() for step in steps)
    repeated = {command for command, count in counts.items() if command and count > 1}
    if len(steps) <= max_steps:
        retained = list(steps)
    else:
        head = min(4, max_steps)
        indices = list(range(head))
        for index in range(max(head, len(steps) - (max_steps - head)), len(steps)):
            if index not in indices:
                indices.append(index)
        retained = [steps[index] for index in indices[:max_steps]]
    return [
        replace(step, supervise=_is_safe_inspection(step, repeated))
        for step in retained
    ]


def _message(
    role: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    loss: bool,
) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls or [],
        "loss": loss,
    }


def _assistant(step: RevisionStep, index: int) -> dict[str, Any]:
    return _message(
        "assistant",
        step.thought,
        tool_calls=[{
            "function": {
                "arguments": json.dumps(
                    {"command": step.command},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "name": "bash",
            },
            "id": f"fable-revision-{index}",
            "type": "function",
        }],
        loss=step.supervise,
    )


def render_revision_row(
    *,
    task: Mapping[str, Any],
    steps: Sequence[RevisionStep],
    repair_patch: str,
    evidence: Mapping[str, Any],
    stream_sha256: str,
    legacy_patch_sha256: str,
) -> dict[str, Any]:
    """Render one Fable-conditioned row with one exact verified repair target."""

    if not steps:
        raise ValueError("at least one Fable trace step is required")
    required_hashes = (
        "candidate_patch_sha256",
        "admission_evidence_sha256",
        "task_contract_sha256",
    )
    if (
        evidence.get("training_admitted") is not True
        or evidence.get("candidate_passed_twice") is not True
        or any(
            not isinstance(evidence.get(key), str)
            or not _SHA256_RE.fullmatch(str(evidence[key]))
            for key in required_hashes
        )
    ):
        raise ValueError("exact replay admission is required")
    repair_sha256 = sha256_bytes(repair_patch.encode("utf-8"))
    if evidence["candidate_patch_sha256"] != repair_sha256:
        raise ValueError("verified repair patch identity mismatch")
    if not _SHA256_RE.fullmatch(stream_sha256) or not _SHA256_RE.fullmatch(
        legacy_patch_sha256
    ):
        raise ValueError("legacy Fable artifact hashes are required")
    instance_id = task.get("instance_id")
    problem = task.get("problem_statement")
    repo = task.get("repo")
    if not all(isinstance(value, str) and value for value in (instance_id, problem, repo)):
        raise ValueError("task metadata is incomplete")

    messages = [
        _message(
            "system",
            (
                "You are a practical software engineer repairing one repository bug. "
                "The legacy Fable attempt below was collected against a clean reference "
                "because of a collector defect. Treat its observations and unsafe actions "
                "as historical revision context. The actual bug mutation is present now."
            ),
            loss=False,
        ),
        _message(
            "user",
            f"<pr_description>\n{problem}\n</pr_description>",
            loss=False,
        ),
    ]
    for index, step in enumerate(steps):
        messages.append(_assistant(step, index))
        messages.append(
            _message(
                "user",
                (
                    "OBSERVATION:\n"
                    "<legacy_clean_reference_context>true</legacy_clean_reference_context>\n"
                    f"<returncode>{step.returncode}</returncode>\n"
                    f"<output>\n{step.observation.rstrip()}\n</output>"
                ),
                loss=False,
            )
        )
    messages.append(
        _message(
            "user",
            (
                "VERIFIER FEEDBACK:\n"
                "The legacy candidate does not repair the bug-mutated task. "
                "Apply the independently replay-verified source correction now."
            ),
            loss=False,
        )
    )
    final_step = RevisionStep(
        thought="Apply the verifier-proven correction to the mutated task.",
        command=portable_git_apply_command(repair_patch),
        observation="",
        returncode=0,
        supervise=True,
    )
    messages.append(_assistant(final_step, len(steps)))
    content_sha256 = sha256_bytes(canonical_json_bytes(messages))
    return {
        "instance_id": f"fable5-revision::{instance_id}",
        "source_instance_id": instance_id,
        "messages": messages,
        "repo": repo,
        "source": SOURCE,
        "content_sha256": content_sha256,
        "legacy_fable_conditioning": True,
        "oracle_repair_target": True,
        "legacy_stream_sha256": stream_sha256,
        "legacy_patch_sha256": legacy_patch_sha256,
        "repair_patch_sha256": repair_sha256,
        "repair_admission_evidence_sha256": evidence[
            "admission_evidence_sha256"
        ],
        "task_contract_sha256": evidence["task_contract_sha256"],
        "supervised_fable_inspection_turns": sum(
            step.supervise for step in steps
        ),
        "masked_legacy_turns": sum(not step.supervise for step in steps),
    }


def generate_reverse_mutation_patch(
    task: Mapping[str, Any],
    *,
    run: Callable[..., tuple[str, int]] = sh,
) -> str:
    """Construct the clean-reference repair without trusting legacy edits."""

    image = task.get("image_name")
    mutation = task.get("patch")
    if not isinstance(image, str) or not image:
        raise ValueError("task image is missing")
    if not isinstance(mutation, str) or not mutation.strip():
        raise ValueError("task mutation is missing")
    paths = parse_patch_paths(mutation)
    if protected_paths(paths):
        raise ValueError("task mutation touches protected paths")
    output, returncode = run(
        ["docker", "run", "-d", image, "sleep", "infinity"],
        180,
    )
    container_id = next(
        (
            line.strip()
            for line in reversed(output.splitlines())
            if re.fullmatch(r"[0-9a-f]{12,64}", line.strip())
        ),
        "",
    )
    if returncode != 0 or not container_id:
        raise ValueError("repair container failed to start")
    container_id = container_id[:12]
    try:
        apply_output, apply_returncode = run(
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
            mutation,
        )
        if apply_returncode != 0:
            raise ValueError(
                f"task mutation failed to apply: {apply_output[-300:]}"
            )
        commit_output, commit_returncode = run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                (
                    "cd /testbed && git add -A -- . && "
                    "git -c user.name=teacher-platform "
                    "-c user.email=teacher-platform@invalid "
                    "commit --no-verify -m teacher-task-mutation >/dev/null"
                ),
            ],
            180,
        )
        if commit_returncode != 0:
            raise ValueError(
                f"task mutation failed to commit: {commit_output[-300:]}"
            )
        reverse_output, reverse_returncode = run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "bash",
                "-c",
                "cd /testbed && git apply -R --whitespace=nowarn -",
            ],
            180,
            mutation,
        )
        if reverse_returncode != 0:
            raise ValueError(
                f"task mutation failed to reverse: {reverse_output[-300:]}"
            )
        patch, patch_returncode = run(
            ["docker", "exec", container_id, "bash", "-c", _PATCH_CAPTURE_COMMAND],
            180,
        )
        if patch_returncode != 0 or not patch.strip():
            raise ValueError("reverse mutation patch capture failed")
        if protected_paths(parse_patch_paths(patch)):
            raise ValueError("repair patch touches protected paths")
        return patch
    finally:
        run(["docker", "rm", "-f", container_id], 60)


def _load_tasks(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    tasks: dict[str, dict[str, Any]] = {}
    bindings = []
    for path in paths:
        bindings.append({"path": str(path), "sha256": _sha256_path(path)})
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"task metadata lacks instance_id: {path}")
            previous = tasks.get(instance_id)
            if previous is not None and canonical_json_bytes(previous) != canonical_json_bytes(row):
                raise ValueError(f"conflicting task metadata: {instance_id}")
            tasks[instance_id] = row
    return tasks, bindings


def _load_attempts(run_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: dict[str, dict[str, Any]] = {}
    ledger_bindings = []
    for run_dir in run_dirs:
        ledger = run_dir / "results.jsonl"
        ledger_bindings.append({"path": str(ledger), "sha256": _sha256_path(ledger)})
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            instance_id = result.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"legacy result lacks instance_id: {ledger}")
            if result.get("error") == "out_of_credits":
                continue
            stream = run_dir / f"{instance_id}.stream.jsonl"
            patch = run_dir / f"{instance_id}.patch"
            if not stream.is_file() or not patch.is_file() or not patch.read_text().strip():
                continue
            if instance_id in attempts:
                raise ValueError(f"duplicate legacy Fable attempt: {instance_id}")
            attempts[instance_id] = {
                "result": result,
                "stream": stream,
                "patch": patch,
            }
    return [attempts[key] for key in sorted(attempts)], ledger_bindings


def _normalized_revision_steps(
    trace: NormalizedTrace,
    *,
    max_steps: int,
) -> list[RevisionStep]:
    return select_revision_steps(
        [
            RevisionStep(
                thought=str(step.get("thought") or ""),
                command=str(step["command"]),
                observation=str(step.get("observation") or ""),
                returncode=int(step["returncode"]),
            )
            for step in trace
        ],
        max_steps=max_steps,
    )


def bind_replay_artifact_identity(
    evidence: Mapping[str, Any],
    *,
    stream_sha256: str,
) -> dict[str, Any]:
    """Attach the artifact identity fields required by exact replay admission."""

    candidate_patch_sha256 = evidence.get("candidate_patch_sha256")
    if (
        not isinstance(candidate_patch_sha256, str)
        or not _SHA256_RE.fullmatch(candidate_patch_sha256)
        or not _SHA256_RE.fullmatch(stream_sha256)
    ):
        raise ValueError("replay artifact identity is incomplete")
    return {
        **evidence,
        "patch_sha256": candidate_patch_sha256,
        "stream_sha256": stream_sha256,
    }


def validate_verified_row_floor(
    row_count: int,
    *,
    minimum_rows: int,
) -> None:
    """Validate the standard or explicitly exhausted collection floor."""

    if type(minimum_rows) is not int or not 1 <= minimum_rows <= 30:
        raise ValueError("minimum_rows must be an integer in [1,30]")
    if row_count < minimum_rows:
        raise ValueError(
            f"fewer than {minimum_rows} verified Fable revision rows: "
            f"{row_count}"
        )


def repository_is_operator_excluded(
    task: Mapping[str, Any],
    repositories: set[str],
) -> bool:
    """Return whether a task repository is in an explicit operator skip set."""

    repo = task.get("repo")
    return bool(
        isinstance(repo, str)
        and repo
        and repo.casefold() in {value.casefold() for value in repositories}
    )


def build_revision_dataset(
    *,
    run_dirs: Sequence[Path],
    task_paths: Sequence[Path],
    exclusion_paths: Sequence[str],
    output: Path,
    workers: int = 2,
    max_steps: int = 12,
    minimum_rows: int = 30,
    skip_repositories: Sequence[str] = (),
    verify: Callable[..., dict[str, Any]] = verify_candidate_patch,
) -> dict[str, Any]:
    """Build an atomic, pending-format-gate Fable revision dataset."""

    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    tasks, task_bindings = _load_tasks(task_paths)
    attempts, ledger_bindings = _load_attempts(run_dirs)
    operator_excluded_repositories = {
        str(repo).casefold()
        for repo in skip_repositories
        if isinstance(repo, str) and repo.strip()
    }
    if len(operator_excluded_repositories) != len(skip_repositories):
        raise ValueError("skip_repositories must contain unique non-empty strings")
    excluded_ids, excluded_repos, exclusions = _load_exclusion_contract(
        list(exclusion_paths)
    )

    def one(attempt: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        result = attempt["result"]
        instance_id = result["instance_id"]
        task = tasks.get(instance_id)
        if task is None:
            return None, {"instance_id": instance_id, "reason": "missing_task_metadata"}
        if instance_id in excluded_ids or str(task.get("repo", "")).casefold() in excluded_repos:
            return None, {"instance_id": instance_id, "reason": "evaluation_overlap"}
        if repository_is_operator_excluded(
            task,
            operator_excluded_repositories,
        ):
            return None, {
                "instance_id": instance_id,
                "reason": "operator_excluded_repository",
                "repository": task["repo"],
            }
        stream = Path(attempt["stream"])
        legacy_patch = Path(attempt["patch"])
        try:
            stream_sha256 = _sha256_path(stream)
            trace = normalize_steps(str(result.get("backend") or "claude"), stream)
            steps = _normalized_revision_steps(trace, max_steps=max_steps)
            repair_patch = generate_reverse_mutation_patch(task)
            evidence = verify(
                task,
                repair_patch,
                run=sh,
                env_bootstrap=ENV_BOOTSTRAP,
            )
            evidence = bind_replay_artifact_identity(
                evidence,
                stream_sha256=stream_sha256,
            )
            if not admission_is_exact(evidence):
                raise ValueError("repair did not receive exact replay admission")
            row = render_revision_row(
                task=task,
                steps=steps,
                repair_patch=repair_patch,
                evidence=evidence,
                stream_sha256=stream_sha256,
                legacy_patch_sha256=_sha256_path(legacy_patch),
            )
        except (
            OSError,
            ValueError,
            ReplayContractError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        ) as exc:
            return None, {
                "instance_id": instance_id,
                "reason": f"{type(exc).__name__}: {str(exc)[:1000]}",
                "stream_sha256": _sha256_path(stream),
                "legacy_patch_sha256": _sha256_path(legacy_patch),
            }
        binding = {
            key: row[key]
            for key in (
                "instance_id",
                "source_instance_id",
                "content_sha256",
                "legacy_stream_sha256",
                "legacy_patch_sha256",
                "repair_patch_sha256",
                "repair_admission_evidence_sha256",
                "task_contract_sha256",
                "supervised_fable_inspection_turns",
                "masked_legacy_turns",
            )
        }
        return row, binding

    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, attempt): attempt for attempt in attempts}
        for future in as_completed(futures):
            row, report = future.result()
            if row is None:
                rejected.append(report)
                status = "rejected"
            else:
                rows.append(row)
                bindings.append(report)
                status = "accepted"
            print(
                "[fable-revision] "
                f"completed={len(rows) + len(rejected)}/{len(attempts)} "
                f"accepted={len(rows)} rejected={len(rejected)} "
                f"status={status} instance_id={report['instance_id']}",
                flush=True,
            )
    rows.sort(key=lambda row: row["instance_id"])
    bindings.sort(key=lambda row: row["instance_id"])
    rejected.sort(key=lambda row: row["instance_id"])
    validate_verified_row_floor(len(rows), minimum_rows=minimum_rows)
    if len({row["source_instance_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate Fable revision source identity")

    manifest_holder: dict[str, Any] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(rows).save_to_disk(str(stage))
        _write_jsonl(stage / "train.jsonl", rows)
        _write_jsonl(stage / "rejected.jsonl", rejected)
        manifest = {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": DATASET_VARIANT,
            "source": SOURCE,
            "legacy_attempts": len(attempts),
            "rendered": len(rows),
            "rejected": len(rejected),
            "minimum_verified_rows": minimum_rows,
            "verified_row_policy": (
                "standard_30_row_floor"
                if minimum_rows == 30
                else "explicit_exhausted_collection_override"
            ),
            "operator_excluded_repositories": sorted(
                operator_excluded_repositories
            ),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "verified_revision_supervision": True,
            "success_path_distilled": False,
            "legacy_mutations_supervised": 0,
            "legacy_test_runs_supervised": 0,
            "repair_targets_exact_replay": True,
            "oracle_repair_targets": True,
            "task_inputs": task_bindings,
            "input_ledgers": ledger_bindings,
            "exclusions": {
                "instance_ids": len(excluded_ids),
                "repositories": len(excluded_repos),
                "artifacts": exclusions,
            },
            "artifact_bindings": bindings,
            "canonical_rows_sha256": _canonical_rows_sha256(rows),
            "train_jsonl_sha256": _sha256_path(stage / "train.jsonl"),
            "rejected_sha256": _sha256_path(stage / "rejected.jsonl"),
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(output, publish)
    return manifest_holder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--tasks", action="append", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--minimum-rows",
        type=int,
        default=30,
        help=(
            "Verified-row floor in [1,30]. Lower values are only for an "
            "explicit exhausted-collection override when another verified "
            "source satisfies the final mix floor."
        ),
    )
    parser.add_argument(
        "--skip-repository",
        action="append",
        default=[],
        help=(
            "Explicitly reject attempts from this repository without replay; "
            "the repository and rejection are recorded in the manifest."
        ),
    )
    args = parser.parse_args()
    manifest = build_revision_dataset(
        run_dirs=args.run_dir,
        task_paths=args.tasks,
        exclusion_paths=args.exclude,
        output=args.out,
        workers=args.workers,
        max_steps=args.max_steps,
        minimum_rows=args.minimum_rows,
        skip_repositories=args.skip_repository,
    )
    print(json.dumps({
        "legacy_attempts": manifest["legacy_attempts"],
        "rendered": manifest["rendered"],
        "rejected": manifest["rejected"],
        "training_admitted": manifest["training_admitted"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
