#!/usr/bin/env python3
"""Freeze empty-patch retries and combine them without outcome cherry-picking."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_EMPTY_CATEGORIES = (
    "model",
    "harness_forced",
    "docker_failed",
    "unknown_missing",
)
_FORMAT_ERROR_RE = re.compile(r"tool call error", re.IGNORECASE)
_HARNESS_FIELDS = (
    "batch",
    "workers",
    "pull_workers",
    "config",
    "temperature",
    "seed",
    "step_limit",
    "environment_class",
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_ids(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid IDs artifact: {path}") from exc
    if (
        not isinstance(value, list)
        or len(value) != len(set(value))
        or any(not isinstance(instance_id, str) or not instance_id for instance_id in value)
    ):
        raise ValueError(f"IDs must be a unique string list: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _publish_bytes_noreplace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_json_noreplace(path: Path, value: object) -> None:
    _publish_bytes_noreplace(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    value = _read_object(path)
    predictions: dict[str, dict[str, Any]] = {}
    for key, source in value.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"prediction is not an object: {key}")
        instance_id = source.get("instance_id") or key
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"prediction is missing instance_id: {key}")
        if key != instance_id:
            raise ValueError(f"prediction key/instance_id mismatch: {key}")
        predictions[instance_id] = dict(source)
    return predictions


def _trajectory_bindings(
    run_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    trajectories: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("**/*.traj.json")):
        row = _read_object(path)
        instance_id = row.get("instance_id") or path.parent.name
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"trajectory is missing instance_id: {path}")
        if instance_id in trajectories:
            raise ValueError(f"duplicate trajectory identity: {instance_id}")
        trajectories[instance_id] = row
        bindings.append(_binding(path))
    return trajectories, bindings


def _trajectory_health(trajectory: Mapping[str, Any]) -> dict[str, int]:
    from phaseH_eval.eval_v2p10_promotion import exact_repeated_failure_loops

    messages = trajectory.get("messages")
    rows = (
        [row for row in messages if isinstance(row, Mapping)]
        if isinstance(messages, list)
        else []
    )
    return {
        "repeat_loops": exact_repeated_failure_loops(trajectory),
        "tool_format_errors": sum(
            isinstance(row.get("content"), str)
            and bool(_FORMAT_ERROR_RE.search(row["content"]))
            for row in rows
        ),
        "assistant_responses": sum(row.get("role") == "assistant" for row in rows),
    }


def _summarize_behavior(
    by_instance: Mapping[str, Mapping[str, int]],
) -> dict[str, int | float | None]:
    repeat_loops = sum(row["repeat_loops"] for row in by_instance.values())
    format_errors = sum(row["tool_format_errors"] for row in by_instance.values())
    assistant_responses = sum(
        row["assistant_responses"] for row in by_instance.values()
    )
    return {
        "repeat_loops": repeat_loops,
        "tool_format_errors": format_errors,
        "assistant_responses": assistant_responses,
        "format_error_rate": (
            format_errors / assistant_responses if assistant_responses else None
        ),
    }


def _acceptance_empty_ids(acceptance: Mapping[str, Any]) -> dict[str, list[str]]:
    empty_split = acceptance.get("empty_split")
    if not isinstance(empty_split, Mapping):
        raise ValueError("acceptance is missing empty_split")
    categories: dict[str, list[str]] = {}
    seen: set[str] = set()
    for category in _EMPTY_CATEGORIES:
        values = empty_split.get(category)
        if (
            not isinstance(values, list)
            or any(not isinstance(instance_id, str) for instance_id in values)
        ):
            raise ValueError(f"acceptance empty_split.{category} must be a string list")
        duplicates = seen.intersection(values)
        if duplicates:
            raise ValueError(f"acceptance empty categories overlap: {sorted(duplicates)}")
        categories[category] = sorted(values)
        seen.update(values)
    if empty_split.get("total") != len(seen):
        raise ValueError("acceptance empty total differs from category IDs")
    return categories


def _complete_run(
    *,
    ids_path: Path,
    run_root: Path,
    label: str,
) -> dict[str, Any]:
    ids = _read_ids(ids_path)
    expected_ids = set(ids)
    manifest_path = run_root / "run_manifest.json"
    eval_manifest_path = run_root / "eval_manifest.json"
    acceptance_path = run_root / "acceptance.json"
    predictions_path = run_root / "preds_all.json"
    manifest = _read_object(manifest_path)
    eval_manifest = _read_object(eval_manifest_path)
    acceptance = _read_object(acceptance_path)
    predictions = _prediction_map(predictions_path)
    trajectories, trajectory_bindings = _trajectory_bindings(run_root)
    expected = len(ids)
    name = acceptance.get("name")
    model_path_value = eval_manifest.get("model_path")
    model_path = (
        Path(model_path_value)
        if isinstance(model_path_value, str) and model_path_value
        else None
    )
    eval_harness_contract = {
        "batch": eval_manifest.get("batch"),
        "workers": eval_manifest.get("generation_workers"),
        "pull_workers": eval_manifest.get("pull_workers"),
        "config": eval_manifest.get("config"),
        "temperature": eval_manifest.get("temperature"),
        "seed": eval_manifest.get("seed"),
        "step_limit": eval_manifest.get("step_limit"),
        "environment_class": eval_manifest.get("environment_class"),
    }
    model_config = model_path / "config.json" if model_path is not None else None
    model_index = (
        model_path / "model.safetensors.index.json"
        if model_path is not None
        else None
    )
    model_single = (
        model_path / "model.safetensors" if model_path is not None else None
    )
    if model_index is not None and model_index.is_file():
        index_value = _read_object(model_index)
        weight_map = index_value.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"{label} model index is missing weight_map")
        shard_paths = sorted(
            {
                model_path / filename
                for filename in weight_map.values()
                if isinstance(filename, str)
            }
        )
        if not shard_paths or any(not path.is_file() for path in shard_paths):
            raise ValueError(f"{label} model index references missing shards")
        model_artifacts = [_binding(path) for path in shard_paths]
        weights_contract = {
            "model_index_sha256": eval_manifest.get("model_index_sha256"),
            "model_artifacts": model_artifacts,
        }
        weights_valid = (
            eval_manifest.get("model_index_sha256") == _sha256(model_index)
            and eval_manifest.get("model_artifacts") in (None, model_artifacts)
        )
    elif model_single is not None and model_single.is_file():
        model_artifacts = [_binding(model_single)]
        weights_contract = {
            "model_safetensors_sha256": eval_manifest.get(
                "model_safetensors_sha256"
            ),
            "model_artifacts": model_artifacts,
        }
        weights_valid = (
            eval_manifest.get("model_safetensors_sha256") == _sha256(model_single)
            and eval_manifest.get("model_artifacts") in (None, model_artifacts)
        )
    else:
        model_artifacts = []
        weights_contract = {}
        weights_valid = False
    complete = (
        acceptance.get("status") == "complete"
        and acceptance.get("schema_version") == 1
        and acceptance.get("run_id") == run_root.name
        and isinstance(name, str)
        and bool(name)
        and manifest.get("schema_version") == 1
        and manifest.get("name") == name
        and manifest.get("instances") == expected
        and manifest.get("ids_sha256") == _sha256(ids_path)
        and eval_manifest.get("schema_version") == 1
        and eval_manifest.get("run_id") == run_root.name
        and eval_manifest.get("served_name") == name
        and eval_manifest.get("instances") == expected
        and eval_manifest.get("ids_sha256") == _sha256(ids_path)
        and eval_manifest.get("scorer_workers") == manifest.get("workers")
        and eval_harness_contract
        == {field: manifest.get(field) for field in _HARNESS_FIELDS}
        and model_path is not None
        and model_config is not None
        and model_config.is_file()
        and eval_manifest.get("model_config_sha256") == _sha256(model_config)
        and weights_valid
        and acceptance.get("expected") == expected
        and acceptance.get("preds") == expected
        and acceptance.get("traj_files") == expected
        and acceptance.get("unique_trajectories") == expected
        and acceptance.get("usable_outcomes") == expected
        and acceptance.get("pull_failed") == 0
        and acceptance.get("docker_failed") == 0
        and acceptance.get("problems") == []
        and set(predictions) == expected_ids
        and set(trajectories) == expected_ids
    )
    if not complete:
        raise ValueError(f"{label} run is incomplete")
    wrong_prediction_models = sorted(
        instance_id
        for instance_id, row in predictions.items()
        if row.get("model_name_or_path") not in {name, f"openai/{name}"}
    )
    if wrong_prediction_models:
        raise ValueError(
            f"{label} prediction model identity differs from acceptance: "
            f"{wrong_prediction_models}"
        )
    resolved_ids = acceptance.get("resolved_ids")
    if (
        not isinstance(resolved_ids, list)
        or len(resolved_ids) != len(set(resolved_ids))
        or any(not isinstance(instance_id, str) for instance_id in resolved_ids)
        or not set(resolved_ids).issubset(expected_ids)
        or acceptance.get("resolved") != len(resolved_ids)
    ):
        raise ValueError(f"{label} acceptance has invalid resolved IDs")
    empty_categories = _acceptance_empty_ids(acceptance)
    if empty_categories["docker_failed"] or empty_categories["unknown_missing"]:
        raise ValueError(f"{label} run contains an infrastructure empty outcome")
    acceptance_empty = {
        instance_id
        for category in _EMPTY_CATEGORIES
        for instance_id in empty_categories[category]
    }
    prediction_empty = {
        instance_id
        for instance_id, row in predictions.items()
        if not str(row.get("model_patch") or "").strip()
    }
    if acceptance_empty != prediction_empty:
        raise ValueError(f"{label} acceptance empty IDs differ from predictions")
    if acceptance_empty.intersection(resolved_ids):
        raise ValueError(f"{label} empty predictions cannot be resolved")
    return {
        "ids": ids,
        "acceptance": acceptance,
        "manifest": manifest,
        "eval_manifest": eval_manifest,
        "harness_contract": {
            field: manifest.get(field) for field in _HARNESS_FIELDS
        },
        "model_contract": {
            "served_name": name,
            "model_path": str(model_path.resolve()),
            "model_config_sha256": eval_manifest["model_config_sha256"],
            **weights_contract,
        },
        "predictions": predictions,
        "trajectories": trajectories,
        "empty_categories": empty_categories,
        "empty_ids": sorted(prediction_empty),
        "resolved_ids": sorted(resolved_ids),
        "artifacts": [
            _binding(ids_path),
            _binding(manifest_path),
            _binding(eval_manifest_path),
            _binding(acceptance_path),
            _binding(predictions_path),
            *model_artifacts,
            *trajectory_bindings,
        ],
    }


def freeze_empty_retry(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    source = _complete_run(
        ids_path=source_ids_path,
        run_root=source_run_root,
        label="source",
    )
    if not source["empty_ids"]:
        raise ValueError("no usable empty patches; retry not required")
    retry_payload = (
        json.dumps(source["empty_ids"], indent=2, sort_keys=False) + "\n"
    ).encode()
    plan = {
        "schema_version": 1,
        "artifact_type": "empty_patch_retry_plan",
        "status": "complete",
        "source_run_id": source["acceptance"].get("run_id") or source_run_root.name,
        "source_name": source["acceptance"].get("name"),
        "source_expected": len(source["ids"]),
        "source_empty": len(source["empty_ids"]),
        "empty_ids": source["empty_ids"],
        "source_harness_contract": source["harness_contract"],
        "source_model_contract": source["model_contract"],
        "retry_ids_path": str(retry_ids_path.resolve()),
        "retry_ids_sha256": hashlib.sha256(retry_payload).hexdigest(),
        "retry_contract": (
            "one fresh attempt for every preregistered usable source empty; "
            "the retry always supersedes the source outcome"
        ),
        "source_artifacts": source["artifacts"],
    }
    if retry_ids_path.exists():
        if retry_ids_path.read_bytes() != retry_payload:
            raise ValueError("existing retry IDs differ from frozen empty set")
    else:
        _publish_bytes_noreplace(retry_ids_path, retry_payload)
    if plan_path.exists():
        if _read_object(plan_path) != plan:
            raise ValueError("existing retry plan differs from current source")
    else:
        _publish_json_noreplace(plan_path, plan)
    return plan


def _verify_source_bindings(
    plan: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> None:
    frozen = plan.get("source_artifacts")
    if not isinstance(frozen, list) or len(frozen) != len(source_artifacts):
        raise ValueError("source artifact changed after freeze")
    for expected, current in zip(frozen, source_artifacts, strict=True):
        if (
            not isinstance(expected, Mapping)
            or expected.get("path") != current.get("path")
            or expected.get("sha256") != current.get("sha256")
            or expected.get("bytes") != current.get("bytes")
        ):
            raise ValueError("source artifact changed after freeze")


def _verify_frozen_source_files(plan: Mapping[str, Any]) -> None:
    frozen = plan.get("source_artifacts")
    if not isinstance(frozen, list):
        raise ValueError("source artifact changed after freeze")
    for expected in frozen:
        if not isinstance(expected, Mapping) or not isinstance(
            expected.get("path"), str
        ):
            raise ValueError("source artifact changed after freeze")
        path = Path(expected["path"])
        if (
            not path.is_file()
            or expected.get("sha256") != _sha256(path)
            or expected.get("bytes") != path.stat().st_size
        ):
            raise ValueError("source artifact changed after freeze")


def _retry_harness_matches_generation(
    source: Mapping[str, Any],
    retry: Mapping[str, Any],
    generation: int,
) -> bool:
    return (
        type(generation) is int
        and generation > 1
        and retry.get("seed") == generation
        and all(
            source.get(field) == retry.get(field)
            for field in _HARNESS_FIELDS
            if field != "seed"
        )
    )


def _build_empty_retry_composite(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
) -> dict[str, Any]:
    plan = _read_object(plan_path)
    if source_run_root.resolve() == retry_run_root.resolve():
        raise ValueError("source and retry runs must be distinct")
    _verify_frozen_source_files(plan)
    source = _complete_run(
        ids_path=source_ids_path,
        run_root=source_run_root,
        label="source",
    )
    _verify_source_bindings(plan, source["artifacts"])
    retry_ids = _read_ids(retry_ids_path)
    if (
        plan.get("status") != "complete"
        or plan.get("artifact_type") != "empty_patch_retry_plan"
        or plan.get("empty_ids") != retry_ids
        or plan.get("source_empty") != len(retry_ids)
        or plan.get("retry_ids_path") != str(retry_ids_path.resolve())
        or plan.get("retry_ids_sha256") != _sha256(retry_ids_path)
        or source["empty_ids"] != retry_ids
        or plan.get("source_harness_contract") != source["harness_contract"]
        or plan.get("source_model_contract") != source["model_contract"]
    ):
        raise ValueError("empty-retry plan does not match source and retry IDs")
    retry = _complete_run(
        ids_path=retry_ids_path,
        run_root=retry_run_root,
        label="retry",
    )
    if source["acceptance"].get("name") != retry["acceptance"].get("name"):
        raise ValueError("source and retry model names differ")
    if source["acceptance"].get("run_id") == retry["acceptance"].get("run_id"):
        raise ValueError("source and retry run IDs must be distinct")
    if source["harness_contract"] != retry["harness_contract"]:
        raise ValueError("source and retry harness contracts differ")
    if source["model_contract"] != retry["model_contract"]:
        raise ValueError("source and retry model contracts differ")

    source_resolved = set(source["resolved_ids"])
    retry_resolved = set(retry["resolved_ids"])
    composite_resolved = sorted((source_resolved - set(retry_ids)) | retry_resolved)
    retry_empty = set(retry["empty_ids"])
    selected_trajectories = {
        instance_id: (
            retry["trajectories"][instance_id]
            if instance_id in retry_ids
            else source["trajectories"][instance_id]
        )
        for instance_id in source["ids"]
    }
    behavior_by_instance = {
        instance_id: _trajectory_health(trajectory)
        for instance_id, trajectory in selected_trajectories.items()
    }
    composite = {
        "schema_version": 1,
        "artifact_type": "empty_patch_retry_composite",
        "run_id": f"{source_run_root.name}+{retry_run_root.name}",
        "source_run_id": source_run_root.name,
        "retry_run_id": retry_run_root.name,
        "name": source["acceptance"].get("name"),
        "model_contract": source["model_contract"],
        "harness_contract": source["harness_contract"],
        "source_ids_sha256": _sha256(source_ids_path),
        "retry_ids_sha256": _sha256(retry_ids_path),
        "status": "complete",
        "expected": len(source["ids"]),
        "preds": len(source["ids"]),
        "traj_files": len(source["ids"]),
        "unique_trajectories": len(source["ids"]),
        "usable_outcomes": len(source["ids"]),
        "resolved": len(composite_resolved),
        "resolved_ids": composite_resolved,
        "pull_failed": 0,
        "docker_failed": 0,
        "empty_split": {
            "total": len(retry_empty),
            **retry["empty_categories"],
        },
        "empty_resampling": {
            "attempted": len(retry_ids),
            "became_nonempty": len(retry_ids) - len(retry_empty),
            "became_resolved": len(retry_resolved),
            "still_empty": len(retry_empty),
            "nonempty_rate": (
                (len(retry_ids) - len(retry_empty)) / len(retry_ids)
                if retry_ids
                else None
            ),
            "resolved_rate": (
                len(retry_resolved) / len(retry_ids) if retry_ids else None
            ),
        },
        "selected_attempts": {
            "source": len(source["ids"]) - len(retry_ids),
            "empty_retry": len(retry_ids),
        },
        "behavior_health": _summarize_behavior(behavior_by_instance),
        "behavior_health_by_instance": behavior_by_instance,
        "retry_ids": retry_ids,
        "retry_precedence": "retry always supersedes source for every retry_id",
        "input_artifacts": [
            _binding(plan_path),
            *source["artifacts"],
            *retry["artifacts"],
        ],
    }
    return composite


def combine_empty_retry(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    output_path: Path,
    predictions_output_path: Path | None = None,
) -> dict[str, Any]:
    composite = _build_empty_retry_composite(
        source_ids_path=source_ids_path,
        source_run_root=source_run_root,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
    )
    prediction_payload = None
    if predictions_output_path is not None:
        merged_predictions = _merged_retry_predictions(
            source_ids_path=source_ids_path,
            source_run_root=source_run_root,
            retry_ids_path=retry_ids_path,
            retry_run_root=retry_run_root,
        )
        prediction_payload = (
            json.dumps(
                merged_predictions,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        composite["predictions_artifact"] = {
            "path": str(predictions_output_path.resolve()),
            "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            "bytes": len(prediction_payload),
        }
    composite_payload = (
        json.dumps(composite, indent=2, sort_keys=True) + "\n"
    ).encode()
    predictions_created = False
    try:
        if predictions_output_path is not None:
            _publish_bytes_noreplace(
                predictions_output_path,
                prediction_payload,
            )
            predictions_created = True
        _publish_bytes_noreplace(output_path, composite_payload)
    except BaseException:
        if (
            predictions_created
            and predictions_output_path is not None
            and not output_path.exists()
        ):
            predictions_output_path.unlink(missing_ok=True)
        raise
    return composite


def _merged_retry_predictions(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
) -> dict[str, dict[str, Any]]:
    source_ids = _read_ids(source_ids_path)
    retry_ids = _read_ids(retry_ids_path)
    source_predictions = _prediction_map(
        source_run_root / "preds_all.json"
    )
    retry_predictions = _prediction_map(
        retry_run_root / "preds_all.json"
    )
    if set(source_predictions) != set(source_ids):
        raise ValueError("source prediction IDs differ from source IDs")
    if set(retry_predictions) != set(retry_ids):
        raise ValueError("retry prediction IDs differ from frozen retry IDs")
    merged_predictions = {
        instance_id: dict(source_predictions[instance_id])
        for instance_id in source_ids
    }
    for instance_id in retry_ids:
        merged_predictions[instance_id] = dict(
            retry_predictions[instance_id]
        )
    return merged_predictions


def verify_empty_retry_composite(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    output_path: Path,
    predictions_output_path: Path | None = None,
) -> dict[str, Any]:
    expected = _build_empty_retry_composite(
        source_ids_path=source_ids_path,
        source_run_root=source_run_root,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
    )
    existing = _read_object(output_path)
    prediction_artifact = existing.get("predictions_artifact")
    if prediction_artifact is not None:
        if not isinstance(prediction_artifact, Mapping):
            raise ValueError("existing composite has invalid predictions binding")
        bound_path = prediction_artifact.get("path")
        if predictions_output_path is None:
            if not isinstance(bound_path, str):
                raise ValueError(
                    "existing composite has invalid predictions binding"
                )
            predictions_output_path = Path(bound_path)
        merged_predictions = _merged_retry_predictions(
            source_ids_path=source_ids_path,
            source_run_root=source_run_root,
            retry_ids_path=retry_ids_path,
            retry_run_root=retry_run_root,
        )
        prediction_payload = (
            json.dumps(
                merged_predictions,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        expected["predictions_artifact"] = {
            "path": str(predictions_output_path.resolve()),
            "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            "bytes": len(prediction_payload),
        }
        if (
            not predictions_output_path.is_file()
            or predictions_output_path.read_bytes()
            != prediction_payload
        ):
            raise ValueError(
                "existing predictions differ from current bound inputs"
            )
    if existing != expected:
        raise ValueError("existing composite differs from current bound inputs")
    return expected


def promote_complete_run_without_retry(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    output_path: Path,
    predictions_output_path: Path,
) -> dict[str, Any]:
    source = _complete_run(
        ids_path=source_ids_path,
        run_root=source_run_root,
        label="source",
    )
    if source["empty_ids"]:
        raise ValueError(
            "source contains usable empty patches; retry is required"
        )
    ordered_predictions = {
        instance_id: source["predictions"][instance_id]
        for instance_id in source["ids"]
    }
    behavior_by_instance = {
        instance_id: _trajectory_health(
            source["trajectories"][instance_id]
        )
        for instance_id in source["ids"]
    }
    prediction_payload = (
        json.dumps(
            ordered_predictions,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    composite = {
        "schema_version": 1,
        "artifact_type": "empty_patch_retry_composite",
        "run_id": source_run_root.name,
        "source_run_id": source_run_root.name,
        "retry_run_id": None,
        "name": source["acceptance"].get("name"),
        "model_contract": source["model_contract"],
        "harness_contract": source["harness_contract"],
        "source_ids_sha256": _sha256(source_ids_path),
        "retry_ids_sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "status": "complete",
        "expected": len(source["ids"]),
        "preds": len(source["ids"]),
        "traj_files": len(source["ids"]),
        "unique_trajectories": len(source["ids"]),
        "usable_outcomes": len(source["ids"]),
        "resolved": len(source["resolved_ids"]),
        "resolved_ids": source["resolved_ids"],
        "pull_failed": 0,
        "docker_failed": 0,
        "empty_split": {
            "total": 0,
            **source["empty_categories"],
        },
        "empty_resampling": {
            "attempted": 0,
            "became_nonempty": 0,
            "became_resolved": 0,
            "still_empty": 0,
            "nonempty_rate": None,
            "resolved_rate": None,
        },
        "selected_attempts": {
            "source": len(source["ids"]),
            "empty_retry": 0,
        },
        "behavior_health": _summarize_behavior(
            behavior_by_instance
        ),
        "behavior_health_by_instance": behavior_by_instance,
        "retry_ids": [],
        "retry_precedence": "no retry required because source had no empty patches",
        "input_artifacts": source["artifacts"],
        "predictions_artifact": {
            "path": str(predictions_output_path.resolve()),
            "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            "bytes": len(prediction_payload),
        },
    }
    composite_payload = (
        json.dumps(composite, indent=2, sort_keys=True) + "\n"
    ).encode()
    predictions_created = False
    try:
        _publish_bytes_noreplace(
            predictions_output_path,
            prediction_payload,
        )
        predictions_created = True
        _publish_bytes_noreplace(output_path, composite_payload)
    except BaseException:
        if predictions_created and not output_path.exists():
            predictions_output_path.unlink(missing_ok=True)
        raise
    return composite


def _validate_composite_source(
    *,
    ids_path: Path,
    composite_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    ids = _read_ids(ids_path)
    expected_ids = set(ids)
    composite = _read_object(composite_path)
    predictions = _prediction_map(predictions_path)
    predictions_artifact = composite.get("predictions_artifact")
    if (
        not isinstance(predictions_artifact, Mapping)
        or predictions_artifact != _binding(predictions_path)
    ):
        raise ValueError("source composite predictions artifact changed")
    input_artifacts = composite.get("input_artifacts")
    if not isinstance(input_artifacts, list) or not input_artifacts:
        raise ValueError("source composite input artifacts are missing")
    for artifact in input_artifacts:
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("path"), str)
            or dict(artifact) != _binding(Path(artifact["path"]))
        ):
            raise ValueError("source composite input artifact changed")

    empty_categories = _acceptance_empty_ids(composite)
    if empty_categories["docker_failed"] or empty_categories["unknown_missing"]:
        raise ValueError("source composite contains an infrastructure empty outcome")
    empty_ids = {
        instance_id
        for category in _EMPTY_CATEGORIES
        for instance_id in empty_categories[category]
    }
    prediction_empty = {
        instance_id
        for instance_id, row in predictions.items()
        if not str(row.get("model_patch") or "").strip()
    }
    if empty_ids != prediction_empty:
        raise ValueError("source composite empty IDs differ from predictions")

    resolved_ids = composite.get("resolved_ids")
    selected_attempts = composite.get("selected_attempts")
    behavior = composite.get("behavior_health_by_instance")
    resampling = composite.get("empty_resampling")
    resampling_keys = (
        "attempted",
        "became_nonempty",
        "became_resolved",
        "still_empty",
    )
    if (
        not isinstance(resampling, Mapping)
        or any(
            type(resampling.get(key)) is not int or resampling[key] < 0
            for key in resampling_keys
        )
    ):
        raise ValueError("source composite empty resampling is invalid")
    attempted = resampling["attempted"]
    became_nonempty = resampling["became_nonempty"]
    became_resolved = resampling["became_resolved"]
    still_empty = resampling["still_empty"]
    if (
        became_nonempty > attempted
        or became_resolved > became_nonempty
        or still_empty > attempted
        or became_nonempty + still_empty > attempted
        or still_empty != len(empty_ids)
        or resampling.get("nonempty_rate")
        != (became_nonempty / attempted if attempted else None)
        or resampling.get("resolved_rate")
        != (became_resolved / attempted if attempted else None)
    ):
        raise ValueError("source composite empty resampling is inconsistent")
    if (
        composite.get("schema_version") != 1
        or composite.get("artifact_type") != "empty_patch_retry_composite"
        or composite.get("status") != "complete"
        or any(
            composite.get(key) != len(ids)
            for key in (
                "expected",
                "preds",
                "traj_files",
                "unique_trajectories",
                "usable_outcomes",
            )
        )
        or composite.get("pull_failed") != 0
        or composite.get("docker_failed") != 0
        or set(predictions) != expected_ids
        or not isinstance(resolved_ids, list)
        or len(resolved_ids) != len(set(resolved_ids))
        or not set(resolved_ids).issubset(expected_ids)
        or composite.get("resolved") != len(resolved_ids)
        or not isinstance(selected_attempts, Mapping)
        or not selected_attempts
        or any(
            not isinstance(key, str)
            or not key
            or type(value) is not int
            or value < 0
            for key, value in selected_attempts.items()
        )
        or sum(selected_attempts.values()) != len(ids)
        or not isinstance(behavior, Mapping)
        or set(behavior) != expected_ids
        or not isinstance(composite.get("model_contract"), Mapping)
        or not isinstance(composite.get("harness_contract"), Mapping)
        or not isinstance(composite.get("name"), str)
        or not composite["name"]
    ):
        raise ValueError("source composite is incomplete")

    generation = composite.get("retry_generation", 1)
    if type(generation) is not int or generation < 1:
        raise ValueError("source composite retry generation is invalid")
    if generation > 1 and not _retry_harness_matches_generation(
        composite["harness_contract"],
        composite.get("retry_harness_contract", {}),
        generation,
    ):
        raise ValueError("source composite retry harness contract is invalid")
    retry_ids = composite.get("retry_ids")
    if (
        not isinstance(retry_ids, list)
        or len(retry_ids) != len(set(retry_ids))
        or any(
            not isinstance(instance_id, str) or instance_id not in expected_ids
            for instance_id in retry_ids
        )
        or not empty_ids.issubset(retry_ids)
    ):
        raise ValueError("source composite retry IDs are invalid")
    current_retry_label = (
        "empty_retry" if generation == 1 else f"empty_retry_{generation}"
    )
    if selected_attempts.get(current_retry_label) != len(retry_ids):
        raise ValueError("source composite retry provenance is invalid")
    retry_lineage = None
    if composite.get("retry_run_id") is not None:
        plan_path_value = input_artifacts[0].get("path")
        if not isinstance(plan_path_value, str):
            raise ValueError("source composite retry plan is missing")
        retry_lineage = verify_retry_plan_empty_only(
            Path(plan_path_value)
        )
        if (
            retry_lineage["retry_generation"] != generation
            or retry_lineage["empty_ids"] != retry_ids
        ):
            raise ValueError(
                "source composite retry IDs differ from actual source empties"
            )
    return {
        "ids": ids,
        "composite": composite,
        "predictions": predictions,
        "empty_categories": empty_categories,
        "empty_ids": sorted(empty_ids),
        "resolved_ids": sorted(resolved_ids),
        "selected_attempts": dict(selected_attempts),
        "behavior": dict(behavior),
        "retry_generation": generation,
        "current_retry_label": current_retry_label,
        "retry_lineage": retry_lineage,
        "artifacts": [
            _binding(ids_path),
            _binding(composite_path),
            _binding(predictions_path),
        ],
    }


def verify_retry_plan_empty_only(plan_path: Path) -> dict[str, Any]:
    """Reopen a frozen retry plan and prove it contains only source empties."""
    plan_path = Path(plan_path).resolve()
    plan = _read_object(plan_path)
    _verify_frozen_source_files(plan)
    retry_ids_path_value = plan.get("retry_ids_path")
    source_artifacts = plan.get("source_artifacts")
    if (
        not isinstance(retry_ids_path_value, str)
        or not isinstance(source_artifacts, list)
        or not source_artifacts
    ):
        raise ValueError("retry plan source bindings are incomplete")
    retry_ids_path = Path(retry_ids_path_value)
    retry_ids = _read_ids(retry_ids_path)
    if (
        plan.get("retry_ids_sha256") != _sha256(retry_ids_path)
        or plan.get("empty_ids") != retry_ids
        or plan.get("source_empty") != len(retry_ids)
    ):
        raise ValueError("retry plan IDs differ from its frozen contract")

    artifact_type = plan.get("artifact_type")
    if artifact_type == "empty_patch_retry_plan":
        ids_binding = source_artifacts[0]
        manifest_bindings = [
            artifact
            for artifact in source_artifacts
            if isinstance(artifact, Mapping)
            and isinstance(artifact.get("path"), str)
            and Path(artifact["path"]).name == "run_manifest.json"
        ]
        if (
            not isinstance(ids_binding, Mapping)
            or not isinstance(ids_binding.get("path"), str)
            or len(manifest_bindings) != 1
        ):
            raise ValueError("retry plan source run bindings are incomplete")
        source = _complete_run(
            ids_path=Path(ids_binding["path"]),
            run_root=Path(manifest_bindings[0]["path"]).parent,
            label="retry_plan_source",
        )
        _verify_source_bindings(plan, source["artifacts"])
        actual_empty_ids = source["empty_ids"]
        generation = 1
    elif artifact_type == "composite_empty_patch_retry_plan":
        if (
            len(source_artifacts) != 3
            or any(
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("path"), str)
                for artifact in source_artifacts
            )
        ):
            raise ValueError(
                "composite retry plan source bindings are incomplete"
            )
        source = _validate_composite_source(
            ids_path=Path(source_artifacts[0]["path"]),
            composite_path=Path(source_artifacts[1]["path"]),
            predictions_path=Path(source_artifacts[2]["path"]),
        )
        _verify_source_bindings(plan, source["artifacts"])
        actual_empty_ids = source["empty_ids"]
        generation = source["retry_generation"] + 1
        if plan.get("retry_generation") != generation:
            raise ValueError("composite retry generation is inconsistent")
    else:
        raise ValueError("retry plan artifact type is invalid")

    if actual_empty_ids != retry_ids:
        raise ValueError(
            "retry plan IDs differ from actual source empty outcomes"
        )
    return {
        "plan": _binding(plan_path),
        "retry_generation": generation,
        "empty_ids": retry_ids,
        "source_artifacts": [dict(row) for row in source_artifacts],
    }


def freeze_composite_retry(
    *,
    source_ids_path: Path,
    source_composite_path: Path,
    source_predictions_path: Path,
    retry_ids_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    source = _validate_composite_source(
        ids_path=source_ids_path,
        composite_path=source_composite_path,
        predictions_path=source_predictions_path,
    )
    if not source["empty_ids"]:
        raise ValueError("no usable empty patches; retry not required")
    retry_payload = (
        json.dumps(source["empty_ids"], indent=2, sort_keys=False) + "\n"
    ).encode()
    plan = {
        "schema_version": 1,
        "artifact_type": "composite_empty_patch_retry_plan",
        "status": "complete",
        "retry_generation": source["retry_generation"] + 1,
        "source_run_id": source["composite"]["run_id"],
        "source_name": source["composite"]["name"],
        "source_expected": len(source["ids"]),
        "source_empty": len(source["empty_ids"]),
        "empty_ids": source["empty_ids"],
        "source_harness_contract": source["composite"]["harness_contract"],
        "source_model_contract": source["composite"]["model_contract"],
        "retry_ids_path": str(retry_ids_path.resolve()),
        "retry_ids_sha256": hashlib.sha256(retry_payload).hexdigest(),
        "retry_contract": (
            "one fresh attempt for every preregistered current composite empty; "
            "the retry always supersedes the parent composite outcome"
        ),
        "source_artifacts": source["artifacts"],
    }
    if retry_ids_path.exists():
        if retry_ids_path.read_bytes() != retry_payload:
            raise ValueError("existing retry IDs differ from frozen empty set")
    else:
        _publish_bytes_noreplace(retry_ids_path, retry_payload)
    if plan_path.exists():
        if _read_object(plan_path) != plan:
            raise ValueError("existing retry plan differs from current source")
    else:
        _publish_json_noreplace(plan_path, plan)
    return plan


def _merged_composite_predictions(
    *,
    source_ids_path: Path,
    source_predictions_path: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
) -> dict[str, dict[str, Any]]:
    source_ids = _read_ids(source_ids_path)
    retry_ids = _read_ids(retry_ids_path)
    source_predictions = _prediction_map(source_predictions_path)
    retry_predictions = _prediction_map(retry_run_root / "preds_all.json")
    if set(source_predictions) != set(source_ids):
        raise ValueError("source prediction IDs differ from source IDs")
    if set(retry_predictions) != set(retry_ids):
        raise ValueError("retry prediction IDs differ from frozen retry IDs")
    merged = {
        instance_id: dict(source_predictions[instance_id])
        for instance_id in source_ids
    }
    for instance_id in retry_ids:
        merged[instance_id] = dict(retry_predictions[instance_id])
    return merged


def _build_composite_retry(
    *,
    source_ids_path: Path,
    source_composite_path: Path,
    source_predictions_path: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
) -> dict[str, Any]:
    plan = _read_object(plan_path)
    _verify_frozen_source_files(plan)
    source = _validate_composite_source(
        ids_path=source_ids_path,
        composite_path=source_composite_path,
        predictions_path=source_predictions_path,
    )
    _verify_source_bindings(plan, source["artifacts"])
    retry_ids = _read_ids(retry_ids_path)
    generation = source["retry_generation"] + 1
    if (
        plan.get("status") != "complete"
        or plan.get("artifact_type")
        != "composite_empty_patch_retry_plan"
        or plan.get("retry_generation") != generation
        or plan.get("source_run_id") != source["composite"]["run_id"]
        or plan.get("source_name") != source["composite"]["name"]
        or plan.get("source_expected") != len(source["ids"])
        or plan.get("source_empty") != len(retry_ids)
        or plan.get("empty_ids") != retry_ids
        or source["empty_ids"] != retry_ids
        or plan.get("retry_ids_path") != str(retry_ids_path.resolve())
        or plan.get("retry_ids_sha256") != _sha256(retry_ids_path)
        or plan.get("source_harness_contract")
        != source["composite"]["harness_contract"]
        or plan.get("source_model_contract")
        != source["composite"]["model_contract"]
    ):
        raise ValueError("composite retry plan does not match source and retry IDs")

    retry = _complete_run(
        ids_path=retry_ids_path,
        run_root=retry_run_root,
        label="retry",
    )
    if source["composite"]["name"] != retry["acceptance"].get("name"):
        raise ValueError("source and retry model names differ")
    if not _retry_harness_matches_generation(
        source["composite"]["harness_contract"],
        retry["harness_contract"],
        generation,
    ):
        raise ValueError("source and retry harness contracts differ")
    if source["composite"]["model_contract"] != retry["model_contract"]:
        raise ValueError("source and retry model contracts differ")

    retry_id_set = set(retry_ids)
    resolved_ids = sorted(
        (set(source["resolved_ids"]) - retry_id_set)
        | set(retry["resolved_ids"])
    )
    empty_categories = {
        category: sorted(
            (set(source["empty_categories"][category]) - retry_id_set)
            | set(retry["empty_categories"][category])
        )
        for category in _EMPTY_CATEGORIES
    }
    empty_ids = {
        instance_id
        for values in empty_categories.values()
        for instance_id in values
    }
    behavior = dict(source["behavior"])
    for instance_id in retry_ids:
        behavior[instance_id] = _trajectory_health(
            retry["trajectories"][instance_id]
        )
    selected_attempts = dict(source["selected_attempts"])
    prior_label = source["current_retry_label"]
    selected_attempts[prior_label] -= len(retry_ids)
    selected_attempts[f"empty_retry_{generation}"] = len(retry_ids)
    previous_resampling = source["composite"]["empty_resampling"]
    attempted = previous_resampling["attempted"] + len(retry_ids)
    became_nonempty = (
        previous_resampling["became_nonempty"]
        + len(retry_ids)
        - len(retry["empty_ids"])
    )
    became_resolved = (
        previous_resampling["became_resolved"]
        + len(retry["resolved_ids"])
    )
    composite = {
        "schema_version": 1,
        "artifact_type": "empty_patch_retry_composite",
        "retry_generation": generation,
        "run_id": (
            f"{source['composite']['run_id']}+{retry_run_root.name}"
        ),
        "source_run_id": source["composite"].get("source_run_id"),
        "retry_run_id": retry_run_root.name,
        "parent_composite": _binding(source_composite_path),
        "name": source["composite"]["name"],
        "model_contract": source["composite"]["model_contract"],
        "harness_contract": source["composite"]["harness_contract"],
        "retry_harness_contract": retry["harness_contract"],
        "source_ids_sha256": _sha256(source_ids_path),
        "retry_ids_sha256": _sha256(retry_ids_path),
        "status": "complete",
        "expected": len(source["ids"]),
        "preds": len(source["ids"]),
        "traj_files": len(source["ids"]),
        "unique_trajectories": len(source["ids"]),
        "usable_outcomes": len(source["ids"]),
        "resolved": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "pull_failed": 0,
        "docker_failed": 0,
        "empty_split": {
            "total": len(empty_ids),
            **empty_categories,
        },
        "empty_resampling": {
            "attempted": attempted,
            "became_nonempty": became_nonempty,
            "became_resolved": became_resolved,
            "still_empty": len(empty_ids),
            "nonempty_rate": (
                became_nonempty / attempted if attempted else None
            ),
            "resolved_rate": (
                became_resolved / attempted if attempted else None
            ),
        },
        "selected_attempts": selected_attempts,
        "behavior_health": _summarize_behavior(behavior),
        "behavior_health_by_instance": behavior,
        "retry_ids": retry_ids,
        "retry_precedence": (
            "newest retry always supersedes the parent composite "
            "for every retry_id"
        ),
        "input_artifacts": [
            _binding(plan_path),
            *source["artifacts"],
            *retry["artifacts"],
        ],
    }
    return composite


def combine_composite_retry(
    *,
    source_ids_path: Path,
    source_composite_path: Path,
    source_predictions_path: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    output_path: Path,
    predictions_output_path: Path | None = None,
) -> dict[str, Any]:
    composite = _build_composite_retry(
        source_ids_path=source_ids_path,
        source_composite_path=source_composite_path,
        source_predictions_path=source_predictions_path,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
    )
    prediction_payload = None
    if predictions_output_path is not None:
        merged_predictions = _merged_composite_predictions(
            source_ids_path=source_ids_path,
            source_predictions_path=source_predictions_path,
            retry_ids_path=retry_ids_path,
            retry_run_root=retry_run_root,
        )
        prediction_payload = (
            json.dumps(merged_predictions, indent=2, sort_keys=True) + "\n"
        ).encode()
        composite["predictions_artifact"] = {
            "path": str(predictions_output_path.resolve()),
            "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            "bytes": len(prediction_payload),
        }
    composite_payload = (
        json.dumps(composite, indent=2, sort_keys=True) + "\n"
    ).encode()
    predictions_created = False
    try:
        if predictions_output_path is not None:
            _publish_bytes_noreplace(
                predictions_output_path,
                prediction_payload,
            )
            predictions_created = True
        _publish_bytes_noreplace(output_path, composite_payload)
    except BaseException:
        if (
            predictions_created
            and predictions_output_path is not None
            and not output_path.exists()
        ):
            predictions_output_path.unlink(missing_ok=True)
        raise
    return composite


def verify_composite_retry(
    *,
    source_ids_path: Path,
    source_composite_path: Path,
    source_predictions_path: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    output_path: Path,
    predictions_output_path: Path | None = None,
) -> dict[str, Any]:
    expected = _build_composite_retry(
        source_ids_path=source_ids_path,
        source_composite_path=source_composite_path,
        source_predictions_path=source_predictions_path,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
    )
    existing = _read_object(output_path)
    prediction_artifact = existing.get("predictions_artifact")
    if prediction_artifact is not None:
        if not isinstance(prediction_artifact, Mapping):
            raise ValueError("existing composite has invalid predictions binding")
        bound_path = prediction_artifact.get("path")
        if predictions_output_path is None:
            if not isinstance(bound_path, str):
                raise ValueError(
                    "existing composite has invalid predictions binding"
                )
            predictions_output_path = Path(bound_path)
        merged_predictions = _merged_composite_predictions(
            source_ids_path=source_ids_path,
            source_predictions_path=source_predictions_path,
            retry_ids_path=retry_ids_path,
            retry_run_root=retry_run_root,
        )
        prediction_payload = (
            json.dumps(merged_predictions, indent=2, sort_keys=True) + "\n"
        ).encode()
        expected["predictions_artifact"] = {
            "path": str(predictions_output_path.resolve()),
            "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            "bytes": len(prediction_payload),
        }
        if (
            not predictions_output_path.is_file()
            or predictions_output_path.read_bytes() != prediction_payload
        ):
            raise ValueError(
                "existing predictions differ from current bound inputs"
            )
    if existing != expected:
        raise ValueError("existing composite differs from current bound inputs")
    return expected


def _build_subset_composite(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    full_composite_path: Path,
    subset_ids_path: Path,
) -> dict[str, Any]:
    parent = _build_empty_retry_composite(
        source_ids_path=source_ids_path,
        source_run_root=source_run_root,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
    )
    current_parent = _read_object(full_composite_path)
    parent_without_predictions = dict(current_parent)
    prediction_artifact = parent_without_predictions.pop(
        "predictions_artifact",
        None,
    )
    if prediction_artifact is not None:
        if (
            not isinstance(prediction_artifact, Mapping)
            or not isinstance(prediction_artifact.get("path"), str)
            or _binding(Path(prediction_artifact["path"]))
            != prediction_artifact
        ):
            raise ValueError(
                "full composite predictions binding is invalid"
            )
    if parent_without_predictions != parent:
        raise ValueError("full composite differs from current bound inputs")
    parent = current_parent
    subset_ids = _read_ids(subset_ids_path)
    subset = set(subset_ids)
    source_ids = set(_read_ids(source_ids_path))
    if not subset or not subset.issubset(source_ids):
        raise ValueError("subset IDs must be a nonempty subset of source IDs")
    parent_retry_ids = set(parent["retry_ids"])
    subset_retry_ids = sorted(parent_retry_ids & subset)
    retry_empty_categories = {
        category: sorted(set(values) & subset)
        for category, values in parent["empty_split"].items()
        if category in _EMPTY_CATEGORIES
    }
    retry_empty = {
        instance_id
        for values in retry_empty_categories.values()
        for instance_id in values
    }
    retry_resolved = set(parent["resolved_ids"]) & set(subset_retry_ids)
    resolved_ids = sorted(set(parent["resolved_ids"]) & subset)
    behavior_by_instance = {
        instance_id: parent["behavior_health_by_instance"][instance_id]
        for instance_id in subset_ids
    }
    retry_payload = (
        json.dumps(subset_retry_ids, indent=2, sort_keys=False) + "\n"
    ).encode()
    return {
        "schema_version": 1,
        "artifact_type": "empty_patch_retry_composite",
        "run_id": (
            f"{source_run_root.name}+{retry_run_root.name}:"
            f"subset:{subset_ids_path.stem}"
        ),
        "source_run_id": source_run_root.name,
        "retry_run_id": retry_run_root.name,
        "name": parent["name"],
        "model_contract": parent["model_contract"],
        "harness_contract": parent["harness_contract"],
        "panel_derivation": "fixed150_from_full300",
        "source_ids_sha256": _sha256(subset_ids_path),
        "parent_source_ids_sha256": parent["source_ids_sha256"],
        "retry_ids_sha256": hashlib.sha256(retry_payload).hexdigest(),
        "status": "complete",
        "expected": len(subset_ids),
        "preds": len(subset_ids),
        "traj_files": len(subset_ids),
        "unique_trajectories": len(subset_ids),
        "usable_outcomes": len(subset_ids),
        "resolved": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "pull_failed": 0,
        "docker_failed": 0,
        "empty_split": {
            "total": len(retry_empty),
            **retry_empty_categories,
        },
        "empty_resampling": {
            "attempted": len(subset_retry_ids),
            "became_nonempty": len(subset_retry_ids) - len(retry_empty),
            "became_resolved": len(retry_resolved),
            "still_empty": len(retry_empty),
            "nonempty_rate": (
                (len(subset_retry_ids) - len(retry_empty))
                / len(subset_retry_ids)
                if subset_retry_ids
                else None
            ),
            "resolved_rate": (
                len(retry_resolved) / len(subset_retry_ids)
                if subset_retry_ids
                else None
            ),
        },
        "selected_attempts": {
            "source": len(subset_ids) - len(subset_retry_ids),
            "empty_retry": len(subset_retry_ids),
        },
        "behavior_health": _summarize_behavior(behavior_by_instance),
        "behavior_health_by_instance": behavior_by_instance,
        "retry_ids": subset_retry_ids,
        "retry_precedence": "retry always supersedes source for every retry_id",
        "input_artifacts": [
            _binding(full_composite_path),
            _binding(subset_ids_path),
        ],
    }


def subset_empty_retry_composite(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    full_composite_path: Path,
    subset_ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    composite = _build_subset_composite(
        source_ids_path=source_ids_path,
        source_run_root=source_run_root,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
        full_composite_path=full_composite_path,
        subset_ids_path=subset_ids_path,
    )
    _publish_json_noreplace(output_path, composite)
    return composite


def verify_subset_composite(
    *,
    source_ids_path: Path,
    source_run_root: Path,
    retry_ids_path: Path,
    retry_run_root: Path,
    plan_path: Path,
    full_composite_path: Path,
    subset_ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    expected = _build_subset_composite(
        source_ids_path=source_ids_path,
        source_run_root=source_run_root,
        retry_ids_path=retry_ids_path,
        retry_run_root=retry_run_root,
        plan_path=plan_path,
        full_composite_path=full_composite_path,
        subset_ids_path=subset_ids_path,
    )
    if _read_object(output_path) != expected:
        raise ValueError("existing subset composite differs from current bound inputs")
    return expected


def _add_combine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--retry-ids", type=Path, required=True)
    parser.add_argument("--retry-run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)


def _add_subset_arguments(parser: argparse.ArgumentParser) -> None:
    _add_combine_arguments(parser)
    parser.add_argument("--full-composite", type=Path, required=True)
    parser.add_argument("--subset-ids", type=Path, required=True)


def _add_composite_retry_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--source-composite", type=Path, required=True)
    parser.add_argument("--source-preds", type=Path, required=True)
    parser.add_argument("--retry-ids", type=Path, required=True)
    parser.add_argument("--retry-run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--source-ids", type=Path, required=True)
    freeze.add_argument("--source-run-root", type=Path, required=True)
    freeze.add_argument("--retry-ids-out", type=Path, required=True)
    freeze.add_argument("--plan-out", type=Path, required=True)
    freeze_composite = commands.add_parser("freeze-composite")
    freeze_composite.add_argument("--source-ids", type=Path, required=True)
    freeze_composite.add_argument(
        "--source-composite",
        type=Path,
        required=True,
    )
    freeze_composite.add_argument("--source-preds", type=Path, required=True)
    freeze_composite.add_argument(
        "--retry-ids-out",
        type=Path,
        required=True,
    )
    freeze_composite.add_argument("--plan-out", type=Path, required=True)
    combine = commands.add_parser("combine")
    _add_combine_arguments(combine)
    combine.add_argument("--preds-out", type=Path)
    verify = commands.add_parser("verify")
    _add_combine_arguments(verify)
    verify.add_argument("--preds-out", type=Path)
    combine_composite = commands.add_parser("combine-composite")
    _add_composite_retry_arguments(combine_composite)
    verify_composite = commands.add_parser("verify-composite")
    _add_composite_retry_arguments(verify_composite)
    promote_no_retry = commands.add_parser("promote-no-retry")
    promote_no_retry.add_argument(
        "--source-ids",
        type=Path,
        required=True,
    )
    promote_no_retry.add_argument(
        "--source-run-root",
        type=Path,
        required=True,
    )
    promote_no_retry.add_argument("--out", type=Path, required=True)
    promote_no_retry.add_argument(
        "--preds-out",
        type=Path,
        required=True,
    )
    subset = commands.add_parser("subset")
    _add_subset_arguments(subset)
    verify_subset = commands.add_parser("verify-subset")
    _add_subset_arguments(verify_subset)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_empty_retry(
            source_ids_path=args.source_ids,
            source_run_root=args.source_run_root,
            retry_ids_path=args.retry_ids_out,
            plan_path=args.plan_out,
        )
        return 0
    if args.command == "freeze-composite":
        freeze_composite_retry(
            source_ids_path=args.source_ids,
            source_composite_path=args.source_composite,
            source_predictions_path=args.source_preds,
            retry_ids_path=args.retry_ids_out,
            plan_path=args.plan_out,
        )
        return 0
    if args.command == "promote-no-retry":
        promote_complete_run_without_retry(
            source_ids_path=args.source_ids,
            source_run_root=args.source_run_root,
            output_path=args.out,
            predictions_output_path=args.preds_out,
        )
        return 0
    if args.command in {"subset", "verify-subset"}:
        function = (
            subset_empty_retry_composite
            if args.command == "subset"
            else verify_subset_composite
        )
        function(
            source_ids_path=args.source_ids,
            source_run_root=args.source_run_root,
            retry_ids_path=args.retry_ids,
            retry_run_root=args.retry_run_root,
            plan_path=args.plan,
            full_composite_path=args.full_composite,
            subset_ids_path=args.subset_ids,
            output_path=args.out,
        )
        return 0
    if args.command in {"combine-composite", "verify-composite"}:
        function = (
            combine_composite_retry
            if args.command == "combine-composite"
            else verify_composite_retry
        )
        function(
            source_ids_path=args.source_ids,
            source_composite_path=args.source_composite,
            source_predictions_path=args.source_preds,
            retry_ids_path=args.retry_ids,
            retry_run_root=args.retry_run_root,
            plan_path=args.plan,
            output_path=args.out,
            predictions_output_path=args.preds_out,
        )
    elif args.command == "combine":
        combine_empty_retry(
            source_ids_path=args.source_ids,
            source_run_root=args.source_run_root,
            retry_ids_path=args.retry_ids,
            retry_run_root=args.retry_run_root,
            plan_path=args.plan,
            output_path=args.out,
            predictions_output_path=args.preds_out,
        )
    else:
        verify_empty_retry_composite(
            source_ids_path=args.source_ids,
            source_run_root=args.source_run_root,
            retry_ids_path=args.retry_ids,
            retry_run_root=args.retry_run_root,
            plan_path=args.plan,
            output_path=args.out,
            predictions_output_path=args.preds_out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
