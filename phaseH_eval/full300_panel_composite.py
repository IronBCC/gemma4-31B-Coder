#!/usr/bin/env python3
"""Join corrected disjoint panels into one checksum-bound full-300 result."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from phaseH_eval.empty_retry_composite import (
    _acceptance_empty_ids,
    _binding,
    _prediction_map,
    _publish_bytes_noreplace,
    _read_ids,
    _read_object,
    _summarize_behavior,
)


@dataclass(frozen=True)
class PanelInput:
    tag: str
    ids_path: Path
    composite_path: Path
    predictions_path: Path


def _validate_panel(panel: PanelInput) -> dict[str, Any]:
    ids = _read_ids(panel.ids_path)
    expected_ids = set(ids)
    composite = _read_object(panel.composite_path)
    predictions = _prediction_map(panel.predictions_path)
    empty_categories = _acceptance_empty_ids(composite)
    resolved_ids = composite.get("resolved_ids")
    selected_attempts = composite.get("selected_attempts")
    behavior = composite.get("behavior_health_by_instance")
    predictions_artifact = composite.get("predictions_artifact")
    if (
        not isinstance(predictions_artifact, dict)
        or predictions_artifact != _binding(panel.predictions_path)
    ):
        raise ValueError(
            f"panel predictions artifact changed: {panel.tag}"
        )
    input_artifacts = composite.get("input_artifacts")
    if not isinstance(input_artifacts, list) or not input_artifacts:
        raise ValueError(f"panel input artifacts are missing: {panel.tag}")
    for artifact in input_artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or artifact != _binding(Path(artifact["path"]))
        ):
            raise ValueError(
                f"panel input artifact changed: {panel.tag}"
            )
    resampling = composite.get("empty_resampling")
    resampling_keys = (
        "attempted",
        "became_nonempty",
        "became_resolved",
        "still_empty",
    )
    resampling_counts_valid = bool(
        isinstance(resampling, dict)
        and all(
            type(resampling.get(key)) is int
            and resampling[key] >= 0
            for key in resampling_keys
        )
    )
    if resampling_counts_valid:
        attempted = resampling["attempted"]
        became_nonempty = resampling["became_nonempty"]
        became_resolved = resampling["became_resolved"]
        still_empty = resampling["still_empty"]
        expected_nonempty_rate = (
            became_nonempty / attempted if attempted else None
        )
        expected_resolved_rate = (
            became_resolved / attempted if attempted else None
        )
        resampling_counts_valid = bool(
            became_nonempty <= attempted
            and became_resolved <= became_nonempty
            and still_empty <= attempted
            and became_nonempty + still_empty <= attempted
            and still_empty
            == sum(len(values) for values in empty_categories.values())
            and resampling.get("nonempty_rate")
            == expected_nonempty_rate
            and resampling.get("resolved_rate")
            == expected_resolved_rate
        )
    if not resampling_counts_valid:
        raise ValueError(
            f"panel empty resampling is inconsistent: {panel.tag}"
        )
    required_complete_counts = (
        "expected",
        "preds",
        "traj_files",
        "unique_trajectories",
        "usable_outcomes",
    )
    selected_attempts_valid = bool(
        isinstance(selected_attempts, dict)
        and selected_attempts
        and all(
            isinstance(key, str)
            and key
            and type(value) is int
            and value >= 0
            for key, value in selected_attempts.items()
        )
        and sum(selected_attempts.values()) == len(ids)
    )
    if (
        not panel.tag
        or composite.get("schema_version") != 1
        or composite.get("artifact_type")
        != "empty_patch_retry_composite"
        or composite.get("status") != "complete"
        or any(
            composite.get(key) != len(ids)
            for key in required_complete_counts
        )
        or composite.get("pull_failed") != 0
        or composite.get("docker_failed") != 0
        or set(predictions) != expected_ids
        or not isinstance(resolved_ids, list)
        or len(resolved_ids) != len(set(resolved_ids))
        or not set(resolved_ids).issubset(expected_ids)
        or composite.get("resolved") != len(resolved_ids)
        or not selected_attempts_valid
        or not isinstance(behavior, dict)
        or set(behavior) != expected_ids
    ):
        raise ValueError(f"incomplete panel composite: {panel.tag}")
    prediction_empty = {
        instance_id
        for instance_id, row in predictions.items()
        if not str(row.get("model_patch") or "").strip()
    }
    composite_empty = {
        instance_id
        for category in (
            "model",
            "harness_forced",
            "docker_failed",
            "unknown_missing",
        )
        for instance_id in empty_categories[category]
    }
    if prediction_empty != composite_empty:
        raise ValueError(
            f"panel empty predictions differ from composite: {panel.tag}"
        )
    name = composite.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"panel model name is missing: {panel.tag}")
    wrong_models = [
        instance_id
        for instance_id, row in predictions.items()
        if row.get("model_name_or_path") not in {
            name,
            f"openai/{name}",
        }
    ]
    if wrong_models:
        raise ValueError(
            f"panel prediction model identity differs: {panel.tag}"
        )
    return {
        "input": panel,
        "ids": ids,
        "composite": composite,
        "predictions": predictions,
        "empty_categories": empty_categories,
        "resolved_ids": set(resolved_ids),
        "selected_attempts": selected_attempts,
        "behavior": behavior,
    }


def _sum_resampling(
    validated: Sequence[dict[str, Any]],
) -> dict[str, int | float | None]:
    keys = (
        "attempted",
        "became_nonempty",
        "became_resolved",
        "still_empty",
    )
    values = {
        key: sum(
            int(row["composite"].get("empty_resampling", {}).get(key, 0))
            for row in validated
        )
        for key in keys
    }
    attempted = values["attempted"]
    return {
        **values,
        "nonempty_rate": (
            values["became_nonempty"] / attempted
            if attempted
            else None
        ),
        "resolved_rate": (
            values["became_resolved"] / attempted
            if attempted
            else None
        ),
    }


def _build_join(
    *,
    full_ids_path: Path,
    panels: Sequence[PanelInput],
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    full_ids = _read_ids(full_ids_path)
    if len(panels) < 2 or len({panel.tag for panel in panels}) != len(
        panels
    ):
        raise ValueError("at least two uniquely tagged panels are required")
    validated = [_validate_panel(panel) for panel in panels]
    concatenated_ids = [
        instance_id
        for row in validated
        for instance_id in row["ids"]
    ]
    if (
        len(concatenated_ids) != len(set(concatenated_ids))
        or set(concatenated_ids) != set(full_ids)
    ):
        raise ValueError(
            "panel IDs must be an exact disjoint union of full IDs"
        )

    model_contract = validated[0]["composite"].get("model_contract")
    harness_contract = validated[0]["composite"].get(
        "harness_contract"
    )
    for row in validated[1:]:
        if row["composite"].get("model_contract") != model_contract:
            raise ValueError("panel model contract mismatch")
        if row["composite"].get("harness_contract") != harness_contract:
            raise ValueError("panel harness contract mismatch")

    predictions_by_id = {
        instance_id: prediction
        for row in validated
        for instance_id, prediction in row["predictions"].items()
    }
    ordered_predictions = {
        instance_id: predictions_by_id[instance_id]
        for instance_id in full_ids
    }
    resolved_ids = sorted(
        set().union(*(row["resolved_ids"] for row in validated))
    )
    categories = {
        category: sorted({
            instance_id
            for row in validated
            for instance_id in row["empty_categories"][category]
        })
        for category in (
            "model",
            "harness_forced",
            "docker_failed",
            "unknown_missing",
        )
    }
    categories["total"] = sum(
        len(categories[category])
        for category in (
            "model",
            "harness_forced",
            "docker_failed",
            "unknown_missing",
        )
    )
    behavior = {
        instance_id: value
        for row in validated
        for instance_id, value in row["behavior"].items()
    }
    selected_attempts = {
        row["input"].tag: row["selected_attempts"]
        for row in validated
    }
    prediction_payload = (
        json.dumps(ordered_predictions, indent=2) + "\n"
    ).encode()
    prediction_artifact = {
        "path": str(predictions_path.resolve()),
        "sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "bytes": len(prediction_payload),
    }
    composite = {
        "schema_version": 1,
        "artifact_type": "disjoint_panel_full300_composite",
        "status": "complete",
        "run_id": "+".join(
            row["composite"]["run_id"] for row in validated
        ),
        "name": validated[0]["composite"]["name"],
        "expected": len(full_ids),
        "preds": len(full_ids),
        "traj_files": len(full_ids),
        "unique_trajectories": len(full_ids),
        "usable_outcomes": len(full_ids),
        "pull_failed": 0,
        "docker_failed": 0,
        "resolved": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "empty_split": categories,
        "empty_resampling": _sum_resampling(validated),
        "selected_attempts": selected_attempts,
        "behavior_health_by_instance": behavior,
        "behavior_health": _summarize_behavior(behavior),
        "model_contract": model_contract,
        "harness_contract": harness_contract,
        "full_ids_sha256": _binding(full_ids_path)["sha256"],
        "predictions_artifact": prediction_artifact,
        "panels": [
            {
                "tag": row["input"].tag,
                "expected": len(row["ids"]),
                "resolved": len(row["resolved_ids"]),
                "empty": sum(
                    len(values)
                    for values in row["empty_categories"].values()
                ),
                "ids_sha256": _binding(
                    row["input"].ids_path
                )["sha256"],
                "composite_sha256": _binding(
                    row["input"].composite_path
                )["sha256"],
                "predictions_sha256": _binding(
                    row["input"].predictions_path
                )["sha256"],
            }
            for row in validated
        ],
        "input_artifacts": [
            _binding(full_ids_path),
            *[
                binding
                for row in validated
                for binding in (
                    _binding(row["input"].ids_path),
                    _binding(row["input"].composite_path),
                    _binding(row["input"].predictions_path),
                )
            ],
        ],
    }
    return composite, ordered_predictions


def join_disjoint_panels(
    *,
    full_ids_path: Path,
    panels: Sequence[PanelInput],
    output_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    if os.path.lexists(output_path) or os.path.lexists(predictions_path):
        raise FileExistsError("refusing to overwrite full-300 artifacts")
    composite, predictions = _build_join(
        full_ids_path=full_ids_path,
        panels=panels,
        predictions_path=predictions_path,
    )
    prediction_payload = (
        json.dumps(predictions, indent=2) + "\n"
    ).encode()
    composite_payload = (
        json.dumps(composite, indent=2, sort_keys=True) + "\n"
    ).encode()
    predictions_created = False
    try:
        _publish_bytes_noreplace(predictions_path, prediction_payload)
        predictions_created = True
        _publish_bytes_noreplace(output_path, composite_payload)
    except BaseException:
        if predictions_created and not output_path.exists():
            predictions_path.unlink(missing_ok=True)
        raise
    return composite


def verify_disjoint_panels(
    *,
    full_ids_path: Path,
    panels: Sequence[PanelInput],
    output_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    expected, predictions = _build_join(
        full_ids_path=full_ids_path,
        panels=panels,
        predictions_path=predictions_path,
    )
    if _read_object(output_path) != expected:
        raise ValueError("full-300 composite differs from bound panels")
    if _prediction_map(predictions_path) != predictions:
        raise ValueError("full-300 predictions differ from bound panels")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("join", "verify"),
    )
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument(
        "--panel",
        action="append",
        nargs=4,
        metavar=("TAG", "IDS", "COMPOSITE", "PREDS"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path, required=True)
    args = parser.parse_args()
    panels = [
        PanelInput(
            tag=tag,
            ids_path=Path(ids),
            composite_path=Path(composite),
            predictions_path=Path(predictions),
        )
        for tag, ids, composite, predictions in args.panel
    ]
    function = (
        join_disjoint_panels
        if args.command == "join"
        else verify_disjoint_panels
    )
    result = function(
        full_ids_path=args.full_ids,
        panels=panels,
        output_path=args.out,
        predictions_path=args.preds_out,
    )
    print(json.dumps({
        "expected": result["expected"],
        "resolved": result["resolved"],
        "empty": result["empty_split"]["total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
