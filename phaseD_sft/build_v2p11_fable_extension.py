#!/usr/bin/env python3
"""Add the late strict Fable-5 batch to the frozen v2.11 rehearsal mix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseD_sft.build_mix_v2p10 import (
    _canonical_json,
    _publish_directory_atomic,
    _sha256_path,
    _training_projection,
    canonical_rows_sha256,
)
from phaseH_eval.empty_retry_composite import _binding
from teacher_platform.generic_trace_replay import admission_is_exact


PRODUCTION_BASE_ROWS = 1_247
PRODUCTION_BASE_MANIFEST_SHA256 = (
    "23090fb8664134f3b8ec2c891b25fef1ca368abdba4bdc6fe2ab114b99f420a0"
)
PRODUCTION_BASE_TRAIN_SHA256 = (
    "e671ca08b9cdefcea59ca767c6d6a5158101ccdcc3b2257dd797d9bdfceca8cd"
)
PRODUCTION_DELTA_ROWS = 15
PRODUCTION_DELTA_SHA256 = (
    "b4eaf4580ca9ccb21d8d4caca69cbed23476d225e1079ce494379b3791c91820"
)
PRODUCTION_DELTA_MANIFEST_SHA256 = (
    "c7d0580900893f3c25e1309e4f6d46282eec35b1c82b5763d566a27ce63bcf17"
)
PRODUCTION_DELTA_REPORT_SHA256 = (
    "69632d235e231a9186dec5722ae5dfadea42a252e88919426111ad1b2249487b"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_RAW_PASSES = (
    "training_admitted",
    "executed",
    "f2p_pass",
    "p2p_pass",
    "resolved",
    "candidate_passed_twice",
    "protected_stable",
    "reference_controls_passed",
    "mutation_f2p_reproduced",
    "mutation_p2p_passed",
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"missing JSONL: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL row {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL row {path}:{line_number}")
        rows.append(row)
    return rows


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def _repo_from_source_instance_id(instance_id: str) -> str:
    if "." in instance_id:
        family = instance_id.split(".", 1)[0]
    else:
        match = re.fullmatch(r"(?P<family>.+)-[0-9]+", instance_id)
        if match is None:
            raise ValueError(
                f"cannot derive repository from source ID: {instance_id}"
            )
        family = match.group("family")
    if family.count("__") != 1:
        raise ValueError(
            f"cannot derive repository from source ID: {instance_id}"
        )
    owner, repo = family.split("__", 1)
    if not owner or not repo:
        raise ValueError(
            f"cannot derive repository from source ID: {instance_id}"
        )
    return f"{owner}/{repo}".casefold()


def _validate_base(base: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = base / "manifest.json"
    train_path = base / "train.jsonl"
    if (
        _sha256_path(manifest_path) != PRODUCTION_BASE_MANIFEST_SHA256
        or _sha256_path(train_path) != PRODUCTION_BASE_TRAIN_SHA256
    ):
        raise ValueError("frozen v2.11 rehearsal mix hash changed")
    manifest = _read_object(manifest_path)
    rows = _read_rows(train_path)
    if (
        len(rows) != PRODUCTION_BASE_ROWS
        or manifest.get("schema_version") != 2
        or manifest.get("dataset_variant") != "teacher_train_mix_v2p11"
        or manifest.get("rendered") != len(rows)
        or manifest.get("training_admitted") != len(rows)
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_path)
    ):
        raise ValueError("frozen v2.11 rehearsal mix is incomplete")
    return rows, manifest


def _validate_delta_contract(
    *,
    delta: Path,
    delta_manifest: Path,
    delta_report: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if (
        _sha256_path(delta) != PRODUCTION_DELTA_SHA256
        or _sha256_path(delta_manifest)
        != PRODUCTION_DELTA_MANIFEST_SHA256
        or _sha256_path(delta_report)
        != PRODUCTION_DELTA_REPORT_SHA256
    ):
        raise ValueError("late Fable batch hash changed")
    rows = _read_rows(delta)
    manifest = _read_object(delta_manifest)
    report = _read_object(delta_report)
    decontam = manifest.get("decontam")
    source_ids = [row.get("source_instance_id") for row in rows]
    selected = report.get("selected_per_instance_cost_usd")
    if (
        len(rows) != PRODUCTION_DELTA_ROWS
        or len(set(source_ids)) != len(rows)
        or any(not isinstance(value, str) or not value for value in source_ids)
        or manifest.get("rows") != len(rows)
        or manifest.get("strict_admitted_selected") != len(rows)
        or manifest.get("all_messages_have_boolean_loss") is not True
        or manifest.get("terminal_assistant_loss_true") is not True
        or not isinstance(manifest.get("source_runs"), list)
        or not manifest["source_runs"]
        or not isinstance(decontam, Mapping)
        or decontam.get("assertions_pass") is not True
        or decontam.get("rows") != len(rows)
        or decontam.get("unique_source_ids") != len(rows)
        or decontam.get("existing_overlap") != []
        or decontam.get("lite_overlap") != []
        or decontam.get("pool_manifest_lite_overlap") != 0
        or not isinstance(selected, Mapping)
        or set(selected) != set(source_ids)
        or report.get("strict_admitted", 0) < len(rows)
    ):
        raise ValueError("late Fable admission contract is incomplete")
    for row in rows:
        messages = row.get("messages")
        assistants = (
            [
                message
                for message in messages
                if isinstance(message, Mapping)
                and message.get("role") == "assistant"
            ]
            if isinstance(messages, list)
            else []
        )
        if (
            row.get("source") != "fable5_verified_finalpatch"
            or row.get("instance_id")
            != f"verified-teacher-finalpatch-{row.get('source_instance_id')}"
            or not messages
            or any(
                not isinstance(message, Mapping)
                or not isinstance(message.get("loss"), bool)
                for message in messages
            )
            or not assistants
            or assistants[-1].get("loss") is not True
            or not assistants[-1].get("tool_calls")
        ):
            raise ValueError("late Fable rendered row is incomplete")
    return rows, manifest, report


def _validate_raw_admission(
    *,
    source_id: str,
    source_runs: Sequence[object],
) -> dict[str, Any]:
    matches: list[tuple[Path, int, dict[str, Any]]] = []
    for source_run in source_runs:
        if not isinstance(source_run, str) or not source_run:
            raise ValueError("late Fable source run is invalid")
        results_path = Path(source_run).resolve() / "results.jsonl"
        for line_number, result in enumerate(_read_rows(results_path), 1):
            if (
                result.get("instance_id") == source_id
                and result.get("training_admitted") is True
            ):
                matches.append((results_path, line_number, result))
    if len(matches) != 1:
        raise ValueError(
            f"raw admission is not unique for {source_id}"
        )
    results_path, line_number, result = matches[0]
    if (
        not admission_is_exact(result)
        or result.get("admission_schema_version") != 2
        or any(result.get(field) is not True for field in _REQUIRED_RAW_PASSES)
        or not _SHA256.fullmatch(
            str(result.get("trace_preflight_source_sha256", ""))
        )
        or not isinstance(result.get("trace_preflight_retained_steps"), int)
        or result["trace_preflight_retained_steps"] <= 0
    ):
        raise ValueError(f"raw admission did not pass for {source_id}")
    for field in (
        "task_contract_sha256",
        "admission_evidence_sha256",
        "stream_sha256",
        "patch_sha256",
    ):
        _require_sha(result.get(field), label=f"{source_id} {field}")
    stream = results_path.parent / f"{source_id}.stream.jsonl"
    patch = results_path.parent / f"{source_id}.patch"
    if (
        _sha256_path(stream) != result["stream_sha256"]
        or _sha256_path(patch) != result["patch_sha256"]
        or not patch.read_text(encoding="utf-8").strip()
        or not stream.read_text(encoding="utf-8").strip()
    ):
        raise ValueError(f"raw admission artifact changed for {source_id}")
    return {
        "source_instance_id": source_id,
        "results": _binding(results_path),
        "result_line": line_number,
        "result_sha256": hashlib.sha256(_canonical_json(result)).hexdigest(),
        "task_contract_sha256": result["task_contract_sha256"],
        "admission_evidence_sha256": result[
            "admission_evidence_sha256"
        ],
        "stream": _binding(stream),
        "patch": _binding(patch),
    }


def _validate_training_target(
    *,
    row: Mapping[str, Any],
    raw_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = row.get("source_instance_id")
    messages = row.get("messages")
    if not isinstance(source_id, str) or not isinstance(messages, list):
        raise ValueError("late Fable training target is invalid")
    assistants = [
        message
        for message in messages
        if isinstance(message, Mapping)
        and message.get("role") == "assistant"
        and message.get("loss") is True
    ]
    if not assistants:
        raise ValueError(f"late Fable target has no supervised assistant: {source_id}")
    calls = assistants[-1].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError(f"late Fable target has no unique patch call: {source_id}")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        raise ValueError(f"late Fable patch call is invalid: {source_id}")
    arguments = function.get("arguments")
    if function.get("name") != "bash" or not isinstance(arguments, str):
        raise ValueError(f"late Fable patch call is invalid: {source_id}")
    try:
        command_value = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"late Fable patch arguments are invalid: {source_id}") from exc
    command = command_value.get("command") if isinstance(command_value, Mapping) else None
    match = (
        re.fullmatch(r"git apply - <<'([^']+)'\n(.*)\n\1", command, flags=re.DOTALL)
        if isinstance(command, str)
        else None
    )
    if match is None:
        raise ValueError(
            f"late Fable target does not reproduce raw patch: {source_id}"
        )
    patch_binding = raw_evidence.get("patch")
    if not isinstance(patch_binding, Mapping):
        raise ValueError(f"late Fable raw patch binding is invalid: {source_id}")
    patch_path = Path(str(patch_binding.get("path", "")))
    expected = patch_path.read_text(encoding="utf-8")
    candidate = match.group(2) + "\n"
    if candidate != expected:
        raise ValueError(
            f"late Fable target does not reproduce raw patch: {source_id}"
        )
    return {
        "source_instance_id": source_id,
        "raw_patch": _binding(patch_path),
        "supervised_patch_sha256": hashlib.sha256(
            candidate.encode("utf-8")
        ).hexdigest(),
        "exact_patch": True,
    }


def build_extension(
    *,
    base: Path,
    delta: Path,
    delta_manifest: Path,
    delta_report: Path,
    lite_ids: Path,
    out: Path,
) -> dict[str, Any]:
    base = Path(base).resolve()
    delta = Path(delta).resolve()
    delta_manifest = Path(delta_manifest).resolve()
    delta_report = Path(delta_report).resolve()
    lite_ids = Path(lite_ids).resolve()
    out = Path(out).resolve()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")

    base_rows, base_manifest = _validate_base(base)
    delta_rows, manifest, report = _validate_delta_contract(
        delta=delta,
        delta_manifest=delta_manifest,
        delta_report=delta_report,
    )
    lite_value = json.loads(lite_ids.read_text(encoding="utf-8"))
    if (
        not isinstance(lite_value, list)
        or any(not isinstance(value, str) for value in lite_value)
    ):
        raise ValueError("Lite evaluation ID artifact is invalid")
    lite = set(lite_value)
    base_ids = {
        value
        for row in base_rows
        for value in (row.get("instance_id"), row.get("source_instance_id"))
        if isinstance(value, str)
    }
    delta_ids = {
        value
        for row in delta_rows
        for value in (row.get("instance_id"), row.get("source_instance_id"))
        if isinstance(value, str)
    }
    if base_ids & delta_ids:
        raise ValueError("late Fable batch overlaps frozen training data")
    if delta_ids & lite:
        raise ValueError("late Fable evaluation overlap is nonzero")

    raw_evidence = [
        _validate_raw_admission(
            source_id=str(row["source_instance_id"]),
            source_runs=manifest["source_runs"],
        )
        for row in delta_rows
    ]
    training_targets = [
        _validate_training_target(row=row, raw_evidence=evidence)
        for row, evidence in zip(delta_rows, raw_evidence, strict=True)
    ]
    projected_base = [_training_projection(row) for row in base_rows]
    delta_with_repos: list[dict[str, Any]] = []
    for row in delta_rows:
        source_id = str(row["source_instance_id"])
        derived_repo = _repo_from_source_instance_id(source_id)
        explicit_repo = row.get("repo")
        if (
            explicit_repo is not None
            and (
                not isinstance(explicit_repo, str)
                or explicit_repo.casefold().rstrip("/") != derived_repo
            )
        ):
            raise ValueError(
                f"late Fable repository does not match source ID: {source_id}"
            )
        delta_with_repos.append({**row, "repo": derived_repo})
    projected_delta = [
        _training_projection(row) for row in delta_with_repos
    ]
    combined = [*projected_base, *projected_delta]
    instance_ids = [row["instance_id"] for row in combined]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("extended v2.11 mix has duplicate instance IDs")
    manifest_holder: dict[str, Any] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(combined).save_to_disk(str(stage))
        train_path = stage / "train.jsonl"
        with train_path.open("wb") as handle:
            for row in combined:
                handle.write(_canonical_json(row) + b"\n")
        output_manifest: dict[str, Any] = {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11_fable_extension",
            "base_variant": base_manifest["dataset_variant"],
            "base_rows": len(projected_base),
            "new_fable_rows": len(projected_delta),
            "rendered": len(combined),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "canonical_rows_sha256": canonical_rows_sha256(combined),
            "train_jsonl_sha256": _sha256_path(train_path),
            "evaluation_overlap": 0,
            "base": {
                "path": str(base),
                "manifest": _binding(base / "manifest.json"),
                "train": _binding(base / "train.jsonl"),
            },
            "new_fable": {
                "train": _binding(delta),
                "manifest": _binding(delta_manifest),
                "report": _binding(delta_report),
                "source_instance_ids": [
                    row["source_instance_id"] for row in delta_rows
                ],
            },
            "lite_ids": _binding(lite_ids),
            "new_fable_admission_evidence": raw_evidence,
            "new_fable_training_target_evidence": training_targets,
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_holder.update(output_manifest)

    _publish_directory_atomic(out, publish)
    return manifest_holder


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--delta-manifest", type=Path, required=True)
    parser.add_argument("--delta-report", type=Path, required=True)
    parser.add_argument("--lite-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_extension(
        base=args.base,
        delta=args.delta,
        delta_manifest=args.delta_manifest,
        delta_report=args.delta_report,
        lite_ids=args.lite_ids,
        out=args.out,
    )
    print(json.dumps({
        "base_rows": manifest["base_rows"],
        "new_fable_rows": manifest["new_fable_rows"],
        "rendered": manifest["rendered"],
        "evaluation_overlap": manifest["evaluation_overlap"],
        "all_training_gates_complete": manifest[
            "all_training_gates_complete"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
