#!/usr/bin/env python3
"""Validate a stock Mini-SWE controller-free v2.10/v2.11 holdout gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from phaseH_eval.empty_retry_composite import (
    _binding,
    _prediction_map,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
    _summarize_behavior,
    _trajectory_bindings,
    _trajectory_health,
)
from phaseH_eval.summarize_smoke import _exit_statuses


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_HARNESS = {
    "mini_swe_agent_version": "2.4.1",
    "environment_class": "docker",
    "model_class": (
        "minisweagent.models.litellm_model.LitellmModel"
    ),
    "temperature": 0.7,
    "seed": 1,
    "max_tokens": 4096,
    "workers": 4,
    "step_limit": 250,
    "subset": "verified",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_harness(
    value: object,
    *,
    ids_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("run is missing harness contract")
    harness = dict(value)
    if any(
        harness.get(key) != expected
        for key, expected in _REQUIRED_HARNESS.items()
    ):
        raise ValueError("run is not the stock controller-free harness")
    if (
        harness.get("ids_sha256") != _sha256(ids_path)
        or not isinstance(harness.get("stock_config_sha256"), str)
        or not _SHA256_RE.fullmatch(harness["stock_config_sha256"])
    ):
        raise ValueError("stock harness artifact binding is invalid")
    return harness


def _validate_model(
    manifest: Mapping[str, Any],
    *,
    expected_name: str,
) -> dict[str, Any]:
    model_path_value = manifest.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise ValueError(f"{expected_name} model path is missing")
    model_path = Path(model_path_value)
    config = model_path / "config.json"
    if (
        not config.is_file()
        or manifest.get("model_config_sha256") != _sha256(config)
    ):
        raise ValueError(f"{expected_name} model config binding changed")
    artifacts = manifest.get("model_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{expected_name} model weights are not bound")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("path"), str)
            or dict(artifact) != _binding(Path(artifact["path"]))
        ):
            raise ValueError(
                f"{expected_name} model weight binding changed"
            )
    index = model_path / "model.safetensors.index.json"
    single = model_path / "model.safetensors"
    if index.is_file():
        if manifest.get("model_index_sha256") != _sha256(index):
            raise ValueError(
                f"{expected_name} model index binding changed"
            )
    elif single.is_file():
        if manifest.get("model_safetensors_sha256") != _sha256(
            single
        ):
            raise ValueError(
                f"{expected_name} model weight binding changed"
            )
    else:
        raise ValueError(f"{expected_name} model weights are missing")
    return {
        "model_path": str(model_path.resolve()),
        "model_config_sha256": manifest["model_config_sha256"],
        "model_index_sha256": manifest.get("model_index_sha256"),
        "model_safetensors_sha256": manifest.get(
            "model_safetensors_sha256"
        ),
        "model_artifacts": [dict(row) for row in artifacts],
    }


def _validate_canary(
    path: Path,
    *,
    expected_name: str,
    eval_manifest_sha256: str,
) -> dict[str, Any]:
    canary = _read_object(path)
    if (
        canary.get("schema_version") != 1
        or canary.get("model") != expected_name
        or canary.get("tool_name") != "bash"
        or not isinstance(canary.get("command"), str)
        or not canary["command"].strip()
        or canary.get("finish_reason") != "tool_calls"
        or canary.get("eval_manifest_sha256")
        != eval_manifest_sha256
    ):
        raise ValueError(
            f"{expected_name} raw tool-call canary did not pass"
        )
    return canary


def _validate_run(
    *,
    ids_path: Path,
    run_root: Path,
    expected_name: str,
) -> dict[str, Any]:
    ids = _read_ids(ids_path)
    expected_ids = set(ids)
    manifest_path = run_root / "eval_manifest.json"
    predictions_path = run_root / "preds.json"
    report_path = run_root / "official_report.json"
    report_binding_path = run_root / "official_report_binding.json"
    canary_path = run_root / "tool_canary.json"
    manifest = _read_object(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_root.name
        or manifest.get("served_name") != expected_name
        or manifest.get("ids_sha256") != _sha256(ids_path)
    ):
        raise ValueError(f"{expected_name} eval manifest is incomplete")
    harness = _validate_harness(
        manifest.get("harness_contract"),
        ids_path=ids_path,
    )
    model = _validate_model(manifest, expected_name=expected_name)
    canary = _validate_canary(
        canary_path,
        expected_name=expected_name,
        eval_manifest_sha256=_sha256(manifest_path),
    )
    predictions = _prediction_map(predictions_path)
    trajectories, trajectory_artifacts = _trajectory_bindings(run_root)
    if set(predictions) != expected_ids:
        raise ValueError(
            f"{expected_name} prediction IDs are incomplete"
        )
    if set(trajectories) != expected_ids:
        raise ValueError(
            f"{expected_name} trajectory IDs are incomplete"
        )
    wrong_models = sorted(
        instance_id
        for instance_id, row in predictions.items()
        if row.get("model_name_or_path")
        not in {expected_name, f"openai/{expected_name}"}
    )
    if wrong_models:
        raise ValueError(
            f"{expected_name} prediction model identity differs"
        )

    empty_ids = {
        instance_id
        for instance_id, row in predictions.items()
        if not str(row.get("model_patch") or "").strip()
    }
    official = _read_object(report_path)
    if not report_binding_path.is_file():
        raise ValueError(
            f"{expected_name} official report binding is missing"
        )
    report_binding = _read_object(report_binding_path)
    if (
        report_binding.get("schema_version") != 1
        or report_binding.get("eval_manifest_sha256")
        != _sha256(manifest_path)
        or report_binding.get("predictions_sha256")
        != _sha256(predictions_path)
        or report_binding.get("official_report_sha256")
        != _sha256(report_path)
    ):
        raise ValueError(
            f"{expected_name} official report binding changed"
        )
    submitted_ids = official.get("submitted_ids")
    resolved_ids = official.get("resolved_ids")
    official_empty = official.get("empty_patch_ids")
    error_ids = official.get("error_ids")
    if (
        official.get("schema_version") != 2
        or official.get("submitted_instances") != len(ids)
        or not isinstance(submitted_ids, list)
        or set(submitted_ids) != expected_ids
        or not isinstance(resolved_ids, list)
        or len(resolved_ids) != len(set(resolved_ids))
        or not set(resolved_ids).issubset(expected_ids)
        or official.get("resolved_instances") != len(resolved_ids)
        or not isinstance(official_empty, list)
        or set(official_empty) != empty_ids
        or official.get("empty_patch_instances") != len(empty_ids)
        or not isinstance(error_ids, list)
        or error_ids
        or official.get("error_instances") != 0
    ):
        raise ValueError(
            f"{expected_name} official holdout report is incomplete"
        )
    statuses = _exit_statuses(run_root)
    attempted = set().union(*statuses.values()) if statuses else set()
    if attempted != expected_ids:
        raise ValueError(
            f"{expected_name} Mini-SWE exit statuses are incomplete"
        )
    format_error_ids = {
        instance_id
        for status, instance_ids in statuses.items()
        if "format" in status.casefold()
        for instance_id in instance_ids
    }
    behavior_by_instance = {
        instance_id: _trajectory_health(trajectory)
        for instance_id, trajectory in trajectories.items()
    }
    behavior = _summarize_behavior(behavior_by_instance)
    return {
        "name": expected_name,
        "resolved": len(resolved_ids),
        "resolved_ids": sorted(resolved_ids),
        "empty": len(empty_ids),
        "empty_ids": sorted(empty_ids),
        "format_errors": len(format_error_ids),
        "format_error_ids": sorted(format_error_ids),
        "behavior_health": behavior,
        "harness_contract": harness,
        "model_contract": model,
        "artifacts": [
            _binding(manifest_path),
            _binding(predictions_path),
            _binding(report_path),
            _binding(report_binding_path),
            _binding(canary_path),
            *trajectory_artifacts,
        ],
        "canary": canary,
    }


def evaluate_portability(
    *,
    ids_path: Path,
    verified_exclusions_path: Path,
    lite_ids_path: Path,
    control_root: Path,
    candidate_root: Path,
    candidate_name: str = "teacher_sft_v2p11",
) -> dict[str, Any]:
    ids_path = Path(ids_path).resolve()
    verified_exclusions_path = Path(
        verified_exclusions_path
    ).resolve()
    lite_ids_path = Path(lite_ids_path).resolve()
    control_root = Path(control_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    ids = _read_ids(ids_path)
    if len(ids) != 10:
        raise ValueError("portability gate requires exactly 10 IDs")
    verified = _read_object(verified_exclusions_path)
    verified_ids = verified.get("instance_ids")
    if (
        not isinstance(verified_ids, list)
        or not set(ids).issubset(verified_ids)
    ):
        raise ValueError(
            "portability IDs are not contained in Verified exclusions"
        )
    lite_ids = set(_read_ids(lite_ids_path))
    if lite_ids.intersection(ids):
        raise ValueError(
            "portability IDs overlap SWE-bench Lite evaluation"
        )

    control = _validate_run(
        ids_path=ids_path,
        run_root=control_root,
        expected_name="teacher_sft_v2p10",
    )
    candidate = _validate_run(
        ids_path=ids_path,
        run_root=candidate_root,
        expected_name=candidate_name,
    )
    if control["harness_contract"] != candidate["harness_contract"]:
        raise ValueError("controller-free harness contract mismatch")

    candidate_repeat_loops = int(
        candidate["behavior_health"]["repeat_loops"]
    )
    control_repeat_loops = int(
        control["behavior_health"]["repeat_loops"]
    )
    criteria = {
        "candidate_empty_at_most_one": candidate["empty"] <= 1,
        "candidate_empty_no_regression": (
            candidate["empty"] <= control["empty"]
        ),
        "candidate_no_format_regression": (
            candidate["format_errors"] <= control["format_errors"]
            and candidate["format_errors"] <= 1
        ),
        "candidate_no_loop_regression": (
            candidate_repeat_loops <= control_repeat_loops
        ),
        "candidate_resolution_floor": (
            candidate["resolved"] >= control["resolved"] - 1
        ),
    }
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "artifact_type": "v2p11_controller_free_portability_gate",
        "status": "complete",
        "passed": passed,
        "control_name": "teacher_sft_v2p10",
        "candidate_name": candidate_name,
        "population": len(ids),
        "ids": _binding(ids_path),
        "verified_exclusions": _binding(verified_exclusions_path),
        "lite_ids": _binding(lite_ids_path),
        "evaluation_overlap": 0,
        "harness_contract": candidate["harness_contract"],
        "control": control,
        "candidate": candidate,
        "criteria": criteria,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument(
        "--verified-exclusions",
        type=Path,
        required=True,
    )
    parser.add_argument("--lite-ids", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-name",
        default="teacher_sft_v2p11",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_portability(
        ids_path=args.ids,
        verified_exclusions_path=args.verified_exclusions,
        lite_ids_path=args.lite_ids,
        control_root=args.control_root,
        candidate_root=args.candidate_root,
        candidate_name=args.candidate_name,
    )
    if os.path.lexists(args.out):
        if _read_object(args.out) != report:
            raise ValueError(
                "existing portability gate differs from current inputs"
            )
    else:
        _publish_json_noreplace(args.out, report)
    print(json.dumps({
        "passed": report["passed"],
        "control": {
            "resolved": report["control"]["resolved"],
            "empty": report["control"]["empty"],
        },
        "candidate": {
            "resolved": report["candidate"]["resolved"],
            "empty": report["candidate"]["empty"],
        },
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
