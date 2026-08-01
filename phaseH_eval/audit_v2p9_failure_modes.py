#!/usr/bin/env python3
"""Build an immutable, evaluation-only audit of v2.9 SWE patch failures."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


_RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
_SED_EDIT_RE = re.compile(r"\bsed\s+-i\b")
_STANDARD_TEST_RE = re.compile(r"\b(?:pytest|unittest|tox)\b")
_FORMAT_ERROR_RE = re.compile(r"tool call error", re.IGNORECASE)


@dataclass(frozen=True)
class CommandOutcome:
    command: str
    returncode: int | None
    output_sha256: str


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


def artifact_binding(paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
    """Bind exact existing input files in stable path order."""

    bindings = []
    for raw_path in sorted({Path(path) for path in paths}, key=lambda path: str(path)):
        path = raw_path.resolve()
        if not path.is_file():
            raise ValueError(f"artifact is missing: {path}")
        bindings.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        })
    return tuple(bindings)


def validate_artifact_binding(binding: Iterable[Mapping[str, object]]) -> None:
    """Reject missing or changed artifacts before publication or reuse."""

    for row in binding:
        path = Path(str(row.get("path") or ""))
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or _sha256_path(path) != row.get("sha256")
        ):
            raise ValueError(f"artifact changed: {path}")


def _command_from_call(call: Mapping[str, Any]) -> str | None:
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "bash":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, Mapping):
        return None
    command = arguments.get("command")
    return command if isinstance(command, str) and command.strip() else None


def _tool_output_for_call(
    messages: Sequence[Mapping[str, Any]],
    assistant_index: int,
    call_id: str | None,
) -> tuple[str, str] | None:
    if assistant_index + 1 >= len(messages):
        return None
    message = messages[assistant_index + 1]
    content = message.get("content")
    role = message.get("role")
    if (
        role not in {"tool", "user"}
        or not isinstance(content, str)
        or (role == "user" and _RETURNCODE_RE.search(content) is None)
    ):
        return None
    message_call_id = message.get("tool_call_id")
    if call_id and message_call_id and message_call_id != call_id:
        return None
    raw_output = message.get("extra", {}).get("raw_output")
    hash_input = raw_output if isinstance(raw_output, str) else content
    return content, hash_input


def extract_command_outcomes(
    messages: Sequence[Mapping[str, Any]],
) -> list[CommandOutcome]:
    outcomes: list[CommandOutcome] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            continue
        raw_call = calls[0]
        if not isinstance(raw_call, Mapping):
            continue
        command = _command_from_call(raw_call)
        if command is None:
            continue
        call_id = raw_call.get("id")
        observation = _tool_output_for_call(
            messages,
            index,
            call_id if isinstance(call_id, str) else None,
        )
        if observation is None:
            continue
        output, hash_input = observation
        match = _RETURNCODE_RE.search(output)
        outcomes.append(CommandOutcome(
            command=command,
            returncode=int(match.group(1)) if match else None,
            output_sha256=hashlib.sha256(
                hash_input.encode("utf-8", errors="replace")
            ).hexdigest(),
        ))
    return outcomes


def _failure_features(trajectory: Mapping[str, Any]) -> dict[str, object]:
    messages = trajectory.get("messages")
    message_rows = (
        [message for message in messages if isinstance(message, Mapping)]
        if isinstance(messages, list)
        else []
    )
    outcomes = extract_command_outcomes(message_rows)
    counts = Counter(row.command.strip() for row in outcomes)
    format_errors = sum(
        bool(
            isinstance(message.get("content"), str)
            and _FORMAT_ERROR_RE.search(message["content"])
        )
        for message in message_rows
    )
    return {
        "uses_sed_i": any(_SED_EDIT_RE.search(row.command) for row in outcomes),
        "standard_test_run": any(
            _STANDARD_TEST_RE.search(row.command) for row in outcomes
        ),
        "nonzero_commands": sum(
            row.returncode is not None and row.returncode != 0
            for row in outcomes
        ),
        "max_exact_repeat": max(counts.values(), default=0),
        "tool_format_errors": format_errors,
        "command_count": len(outcomes),
    }


def audit_rows(
    predictions: Mapping[str, Mapping[str, Any]],
    resolved_ids: set[str],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    expected_ids: Sequence[str] | None = None,
    scored_ids: set[str] | None = None,
) -> dict[str, object]:
    """Partition expected cases without turning absent evidence into model failure."""

    expected = list(expected_ids if expected_ids is not None else predictions)
    if len(expected) != len(set(expected)):
        raise ValueError("expected IDs contain duplicates")
    expected_set = set(expected)
    unexpected = (set(predictions) | set(trajectories) | resolved_ids) - expected_set
    if unexpected:
        raise ValueError(f"artifacts contain unexpected IDs: {sorted(unexpected)}")
    if not resolved_ids <= set(predictions):
        raise ValueError("resolved IDs are missing predictions")
    scored = set(predictions) if scored_ids is None else set(scored_ids)
    if not resolved_ids <= scored:
        raise ValueError("resolved IDs are missing scorer evidence")

    counts = Counter({
        "expected": len(expected),
        "usable": 0,
        "resolved": 0,
        "empty": 0,
        "wrong_nonempty": 0,
        "missing_prediction": 0,
        "missing_trajectory": 0,
        "missing_score": 0,
    })
    rows = []
    for instance_id in expected:
        prediction = predictions.get(instance_id)
        trajectory = trajectories.get(instance_id)
        patch = (
            str(prediction.get("model_patch") or "")
            if isinstance(prediction, Mapping)
            else ""
        )
        if prediction is None:
            category = "missing_prediction"
        elif trajectory is None:
            category = "missing_trajectory"
        elif not patch.strip():
            category = "empty"
        elif instance_id not in scored:
            category = "missing_score"
        elif instance_id in resolved_ids:
            category = "resolved"
        else:
            category = "wrong_nonempty"

        if category in {"resolved", "empty", "wrong_nonempty"}:
            counts["usable"] += 1
        counts[category] += 1
        row: dict[str, object] = {
            "instance_id": instance_id,
            "category": category,
            "training_eligible": False,
            "prediction_present": prediction is not None,
            "trajectory_present": trajectory is not None,
            "score_present": instance_id in scored,
            "patch_bytes": len(patch.encode("utf-8")),
            "patch_sha256": (
                hashlib.sha256(patch.encode("utf-8")).hexdigest()
                if prediction is not None
                else None
            ),
        }
        if category == "wrong_nonempty":
            row["failure_features"] = _failure_features(trajectory or {})
        rows.append(row)

    wrong_rows = [
        row["failure_features"]
        for row in rows
        if row["category"] == "wrong_nonempty"
    ]
    return {
        "schema_version": 1,
        "training_eligible": False,
        "counts": dict(counts),
        "wrong_nonempty_features": {
            "uses_sed_i": sum(bool(row["uses_sed_i"]) for row in wrong_rows),
            "without_standard_test": sum(
                not bool(row["standard_test_run"]) for row in wrong_rows
            ),
            "with_tool_format_error": sum(
                int(row["tool_format_errors"]) > 0 for row in wrong_rows
            ),
            "with_ten_nonzero_commands": sum(
                int(row["nonzero_commands"]) >= 10 for row in wrong_rows
            ),
            "with_half_or_more_exact_repeats": sum(
                int(row["command_count"]) > 0
                and int(row["max_exact_repeat"]) * 2 >= int(row["command_count"])
                for row in wrong_rows
            ),
        },
        "rows": rows,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _numeric_batch(path: Path) -> int:
    for part in path.parts:
        if part.startswith("b") and part[1:].isdigit():
            return int(part[1:])
    return -1


def _score_statuses(report_paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    statuses: dict[str, str] = {}
    for path in sorted(report_paths, key=lambda value: (_numeric_batch(value), str(value))):
        report = _read_json(path)
        if not isinstance(report, Mapping):
            raise ValueError(f"score report is not an object: {path}")
        for key, status in (
            ("resolved_ids", "resolved"),
            ("unresolved_ids", "unresolved"),
            ("error_ids", "error"),
        ):
            values = report.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise ValueError(f"invalid {key}: {path}")
            for instance_id in values:
                statuses[instance_id] = status
    return (
        {instance_id for instance_id, status in statuses.items() if status == "resolved"},
        set(statuses),
    )


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"predictions are not an object: {path}")
    predictions: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError(f"invalid prediction row: {path}")
        instance_id = value.get("instance_id", key)
        if instance_id != key:
            raise ValueError(f"prediction identity mismatch: {path}: {key}")
        predictions[key] = dict(value)
    return predictions


def _trajectory_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda value: (_numeric_batch(value), str(value))):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"trajectory is not an object: {path}")
        instance_id = payload.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"trajectory is missing instance_id: {path}")
        trajectories[instance_id] = dict(payload)
    return trajectories


def _markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    features = report["wrong_nonempty_features"]
    return "\n".join([
        "# v2.9 wrong-edit audit",
        "",
        f"Source status: `{report['source_run_status']}`.",
        "",
        "| expected | usable | resolved | empty | wrong non-empty | missing prediction | missing trajectory | missing score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {counts['expected']} | {counts['usable']} | {counts['resolved']} | "
            f"{counts['empty']} | {counts['wrong_nonempty']} | "
            f"{counts['missing_prediction']} | {counts['missing_trajectory']} | "
            f"{counts['missing_score']} |"
        ),
        "",
        "## Wrong non-empty features",
        "",
        f"- `sed -i`: {features['uses_sed_i']}",
        f"- no standard test command: {features['without_standard_test']}",
        f"- tool-format error: {features['with_tool_format_error']}",
        f"- at least ten nonzero commands: {features['with_ten_nonzero_commands']}",
        (
            "- one command accounts for at least half of commands: "
            f"{features['with_half_or_more_exact_repeats']}"
        ),
        "",
        "**Evaluation-only artifact. No row is eligible for training.**",
        "",
    ])


def audit_run(run_root: Path, output: Path) -> dict[str, object]:
    """Audit one batched run and atomically publish immutable evidence."""

    run_root = Path(run_root).resolve()
    output = Path(output).resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    if run_root.parent.name != "runs":
        raise ValueError(f"run root must be under a runs directory: {run_root}")
    repository_root = run_root.parent.parent
    eval_manifest_path = run_root / "eval_manifest.json"
    acceptance_path = run_root / "acceptance.json"
    summary_path = run_root / "summary.json"
    predictions_path = run_root / "preds_all.json"
    required_paths = [
        eval_manifest_path,
        acceptance_path,
        summary_path,
        predictions_path,
    ]
    if any(not path.is_file() for path in required_paths):
        missing = [str(path) for path in required_paths if not path.is_file()]
        raise ValueError(f"run is missing required artifacts: {missing}")

    eval_manifest = _read_json(eval_manifest_path)
    acceptance = _read_json(acceptance_path)
    summary = _read_json(summary_path)
    if not all(isinstance(value, Mapping) for value in (eval_manifest, acceptance, summary)):
        raise ValueError("run manifests must be JSON objects")
    ids_value = eval_manifest.get("ids_path")
    if not isinstance(ids_value, str):
        raise ValueError("eval manifest is missing ids_path")
    ids_path = (repository_root / ids_value).resolve()
    try:
        ids_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("eval IDs path escapes repository root") from exc
    expected_ids = _read_json(ids_path)
    if not isinstance(expected_ids, list) or any(
        not isinstance(instance_id, str) for instance_id in expected_ids
    ):
        raise ValueError("eval IDs must be a JSON string list")
    if len(expected_ids) != eval_manifest.get("instances"):
        raise ValueError("eval IDs count does not match manifest")
    if _sha256_path(ids_path) != eval_manifest.get("ids_sha256"):
        raise ValueError("eval IDs hash does not match manifest")

    report_paths = sorted(
        run_root.glob("b*/report/official_report.json"),
        key=lambda value: (_numeric_batch(value), str(value)),
    )
    trajectory_paths = sorted(
        run_root.glob("b*/**/*.traj.json"),
        key=lambda value: (_numeric_batch(value), str(value)),
    )
    if not report_paths:
        raise ValueError("run has no scorer reports")
    resolved_ids, scored_ids = _score_statuses(report_paths)
    predictions = _prediction_map(predictions_path)
    trajectories = _trajectory_map(trajectory_paths)
    report = audit_rows(
        predictions,
        resolved_ids,
        trajectories,
        expected_ids=expected_ids,
        scored_ids=scored_ids,
    )
    report["source_run"] = str(run_root)
    report["source_run_status"] = acceptance.get("status")
    report["source_acceptance"] = dict(acceptance)
    report["source_summary"] = dict(summary)

    inputs = artifact_binding([
        *required_paths,
        ids_path,
        *report_paths,
        *trajectory_paths,
    ])
    implementation = artifact_binding([Path(__file__)])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        audit_path = stage / "audit.json"
        markdown_path = stage / "audit.md"
        audit_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "complete": True,
            "training_eligible": False,
            "source_run_status": report["source_run_status"],
            "counts": report["counts"],
            "input_artifacts": list(inputs),
            "implementation": list(implementation),
            "audit_sha256": _sha256_path(audit_path),
            "markdown_sha256": _sha256_path(markdown_path),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_artifact_binding(inputs)
        validate_artifact_binding(implementation)
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite {output}")
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run_root, args.output)
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
