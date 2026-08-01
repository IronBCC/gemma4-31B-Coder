#!/usr/bin/env python3
"""Bind a full-300 composite to replayed official SWE-bench reports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseH_eval.empty_retry_composite import (
    _binding,
    _prediction_map,
    _read_ids,
    _read_object,
)
from phaseH_eval.eval_v2p10_promotion import validate_fixed_run
from phaseH_eval.final_teacher_sft_report import (
    model_failure_ids_from_score_logs,
)
from phaseH_eval.full300_panel_composite import (
    PanelInput,
    verify_disjoint_panels,
)


@dataclass(frozen=True)
class ScorePanel:
    tag: str
    ids_path: Path
    composite_path: Path
    predictions_path: Path


def _string_set(
    report: Mapping[str, Any],
    key: str,
    *,
    full_set: set[str],
    path: Path,
) -> set[str]:
    values = report.get(key)
    if (
        not isinstance(values, list)
        or len(values) != len(set(values))
        or any(not isinstance(value, str) for value in values)
        or not set(values).issubset(full_set)
    ):
        raise ValueError(f"official report {path} has invalid {key}")
    return set(values)


def _validate_official_reports(
    *,
    full_ids: Sequence[str],
    run_ids: set[str],
    prediction_empty_ids: set[str],
    acceptance_resolved_ids: set[str],
    report_paths: Sequence[Path],
    model_failure_ids: set[str],
) -> dict[str, list[str]]:
    full_set = set(full_ids)
    if not report_paths:
        raise ValueError("official score reports are missing")
    unions = {
        "resolved": set(),
        "unresolved": set(),
        "empty": set(),
        "error": set(),
        "submitted": set(),
    }
    for path in report_paths:
        report = _read_object(path)
        sets = {
            "resolved": _string_set(
                report,
                "resolved_ids",
                full_set=full_set,
                path=path,
            ),
            "unresolved": _string_set(
                report,
                "unresolved_ids",
                full_set=full_set,
                path=path,
            ),
            "empty": _string_set(
                report,
                "empty_patch_ids",
                full_set=full_set,
                path=path,
            ),
            "error": _string_set(
                report,
                "error_ids",
                full_set=full_set,
                path=path,
            ),
            "submitted": _string_set(
                report,
                "submitted_ids",
                full_set=full_set,
                path=path,
            ),
            "completed": _string_set(
                report,
                "completed_ids",
                full_set=full_set,
                path=path,
            ),
            "incomplete": _string_set(
                report,
                "incomplete_ids",
                full_set=full_set,
                path=path,
            ),
        }
        outcome_sets = (
            sets["resolved"],
            sets["unresolved"],
            sets["empty"],
            sets["error"],
        )
        if any(
            left & right
            for index, left in enumerate(outcome_sets)
            for right in outcome_sets[index + 1 :]
        ):
            raise ValueError(
                f"official report {path} assigns overlapping outcomes"
            )
        if (
            report.get("schema_version") != 2
            or report.get("total_instances") != len(full_ids)
            or report.get("resolved_instances")
            != len(sets["resolved"])
            or report.get("unresolved_instances")
            != len(sets["unresolved"])
            or report.get("empty_patch_instances")
            != len(sets["empty"])
            or report.get("error_instances") != len(sets["error"])
            or report.get("submitted_instances")
            != len(sets["submitted"])
            or report.get("completed_instances")
            != len(sets["completed"])
        ):
            raise ValueError(
                f"official report {path} counts are inconsistent"
            )
        if (
            sets["completed"]
            != sets["resolved"] | sets["unresolved"]
            or sets["submitted"]
            != sets["completed"] | sets["empty"] | sets["error"]
            or sets["incomplete"] != full_set - sets["submitted"]
        ):
            raise ValueError(
                f"official report {path} set relations are inconsistent"
            )
        if unions["submitted"] & sets["submitted"]:
            raise ValueError("official reports submit an instance twice")
        for key in unions:
            unions[key].update(sets[key])
    expected_submitted = (
        run_ids - prediction_empty_ids
    ) | unions["empty"]
    if unions["submitted"] != expected_submitted:
        raise ValueError(
            "official submitted IDs differ from predictions"
        )
    if not unions["empty"].issubset(prediction_empty_ids):
        raise ValueError(
            "official empty IDs differ from empty predictions"
        )
    if unions["resolved"] != acceptance_resolved_ids:
        raise ValueError(
            "official resolved IDs differ from acceptance"
        )
    if unions["error"] != model_failure_ids:
        raise ValueError(
            "official errors lack exact model-failure classification"
        )
    return {
        "resolved_ids": sorted(unions["resolved"]),
        "unresolved_ids": sorted(unions["unresolved"]),
        "empty_patch_ids": sorted(unions["empty"]),
        "error_ids": sorted(unions["error"]),
        "model_failure_ids": sorted(model_failure_ids),
        "submitted_ids": sorted(unions["submitted"]),
    }


def _bindings_root(bindings: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(bindings),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _replay_run(
    run_root: Path,
    *,
    full_ids: Sequence[str],
    score_log_root: Path,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    run_manifest_path = run_root / "run_manifest.json"
    manifest = _read_object(run_manifest_path)
    ids_value = manifest.get("ids_path")
    if not isinstance(ids_value, str) or not ids_value:
        raise ValueError(f"{run_root} run manifest has no IDs path")
    ids_path = Path(ids_value).resolve()
    if (
        manifest.get("ids_sha256") != _binding(ids_path)["sha256"]
        or manifest.get("instances") != len(_read_ids(ids_path))
    ):
        raise ValueError(f"{run_root} run manifest IDs changed")
    acceptance_path = run_root / "acceptance.json"
    acceptance = _read_object(acceptance_path)
    name = acceptance.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{run_root} acceptance has no model name")
    replayed = validate_fixed_run(
        name=name,
        ids_path=ids_path,
        run_root=run_root,
        score_log_root=score_log_root,
    )
    if replayed != acceptance:
        raise ValueError(
            f"{run_root} acceptance differs from official replay"
        )
    batch_dirs = sorted(
        (
            path
            for path in run_root.iterdir()
            if path.is_dir()
            and path.name.startswith("b")
            and path.name[1:].isdigit()
        ),
        key=lambda path: int(path.name[1:]),
    )
    report_paths = [
        batch / "report" / "official_report.json"
        for batch in batch_dirs
        if (batch / "report" / "official_report.json").is_file()
    ]
    gen_logs = [
        path
        for batch in batch_dirs
        for path in sorted(batch.glob("*.gen.log"))
    ]
    trajectories = [
        path
        for batch in batch_dirs
        for path in sorted(batch.glob("**/*.traj.json"))
    ]
    score_logs = [
        path
        for batch in batch_dirs
        for path in sorted(
            (
                score_log_root / f"{run_root.name}_{batch.name}"
            ).glob("**/run_instance.log")
        )
    ]
    predictions_path = run_root / "preds_all.json"
    predictions = _prediction_map(predictions_path)
    run_ids = set(_read_ids(ids_path))
    prediction_empty_ids = {
        instance_id
        for instance_id, row in predictions.items()
        if not str(row.get("model_patch") or "").strip()
    }
    official = _validate_official_reports(
        full_ids=full_ids,
        run_ids=run_ids,
        prediction_empty_ids=prediction_empty_ids,
        acceptance_resolved_ids=set(acceptance["resolved_ids"]),
        report_paths=report_paths,
        model_failure_ids=model_failure_ids_from_score_logs(score_logs),
    )
    fixed_paths = [
        ids_path,
        run_root / "summary.json",
        run_manifest_path,
        run_root / "eval_manifest.json",
        acceptance_path,
        predictions_path,
    ]
    evidence_paths = sorted({
        path.resolve()
        for path in (
            fixed_paths
            + report_paths
            + gen_logs
            + trajectories
            + score_logs
        )
    })
    if any(not path.is_file() for path in evidence_paths):
        raise ValueError(f"{run_root} official evidence is incomplete")
    evidence = [_binding(path) for path in evidence_paths]
    report_bindings = [_binding(path) for path in report_paths]
    return {
        "run_id": run_root.name,
        "run_root": str(run_root),
        "ids": _binding(ids_path),
        "acceptance": _binding(acceptance_path),
        "expected": len(run_ids),
        "resolved": acceptance["resolved"],
        "prediction_empty_ids": sorted(prediction_empty_ids),
        "official": official,
        "official_report_set_sha256": _bindings_root(
            report_bindings
        ),
        "evidence_root_sha256": _bindings_root(evidence),
        "evidence_artifacts": evidence,
    }


def _panel_run_roots(panel: ScorePanel) -> list[Path]:
    composite = _read_object(panel.composite_path)
    run_id = composite.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{panel.tag} panel has no run lineage")
    names = run_id.split("+")
    if any(
        not name
        or name in {".", ".."}
        or "/" in name
        for name in names
    ):
        raise ValueError(f"{panel.tag} panel run lineage is invalid")
    return [panel.composite_path.resolve().parent / name for name in names]


def build_official_score_binding(
    *,
    full_ids_path: Path,
    composite_path: Path,
    predictions_path: Path,
    panels: Sequence[ScorePanel],
    score_log_root: Path = Path("logs/run_evaluation"),
    verify_panels: bool = True,
) -> dict[str, Any]:
    full_ids_path = Path(full_ids_path).resolve()
    composite_path = Path(composite_path).resolve()
    predictions_path = Path(predictions_path).resolve()
    score_log_root = Path(score_log_root).resolve()
    normalized = [
        ScorePanel(
            tag=panel.tag,
            ids_path=Path(panel.ids_path).resolve(),
            composite_path=Path(panel.composite_path).resolve(),
            predictions_path=Path(panel.predictions_path).resolve(),
        )
        for panel in panels
    ]
    full_ids = _read_ids(full_ids_path)
    if len(full_ids) != 300:
        raise ValueError("official score binding requires 300 IDs")
    if (
        len(normalized) != 2
        or {panel.tag for panel in normalized}
        != {"fixed150", "complement150"}
    ):
        raise ValueError(
            "official score binding requires two canonical panels"
        )
    if verify_panels:
        verify_disjoint_panels(
            full_ids_path=full_ids_path,
            panels=[
                PanelInput(
                    tag=panel.tag,
                    ids_path=panel.ids_path,
                    composite_path=panel.composite_path,
                    predictions_path=panel.predictions_path,
                )
                for panel in normalized
            ],
            output_path=composite_path,
            predictions_path=predictions_path,
        )
    composite = _read_object(composite_path)
    predictions = _prediction_map(predictions_path)
    empty_ids = sorted(
        instance_id
        for instance_id, row in predictions.items()
        if not str(row.get("model_patch") or "").strip()
    )
    if (
        composite.get("status") != "complete"
        or composite.get("expected") != 300
        or set(predictions) != set(full_ids)
        or composite.get("predictions_artifact")
        != _binding(predictions_path)
    ):
        raise ValueError("full300 artifacts are incomplete")
    composite_panels = {
        row.get("tag"): row
        for row in composite.get("panels", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("tag"), str)
    }
    if set(composite_panels) != {panel.tag for panel in normalized}:
        raise ValueError("full300 panel bindings are incomplete")
    for panel in normalized:
        row = composite_panels[panel.tag]
        if (
            row.get("ids_sha256") != _binding(panel.ids_path)["sha256"]
            or row.get("composite_sha256")
            != _binding(panel.composite_path)["sha256"]
            or row.get("predictions_sha256")
            != _binding(panel.predictions_path)["sha256"]
        ):
            raise ValueError(
                f"full300 panel binding changed: {panel.tag}"
            )
    panel_rows = []
    for panel in normalized:
        panel_composite = _read_object(panel.composite_path)
        runs = [
            _replay_run(
                run_root,
                full_ids=full_ids,
                score_log_root=score_log_root,
            )
            for run_root in _panel_run_roots(panel)
        ]
        panel_rows.append({
            "tag": panel.tag,
            "ids": _binding(panel.ids_path),
            "composite": _binding(panel.composite_path),
            "predictions": _binding(panel.predictions_path),
            "retry_generation": panel_composite.get(
                "retry_generation", 1
            ),
            "selected_attempts": panel_composite.get(
                "selected_attempts"
            ),
            "runs": runs,
        })
    return {
        "schema_version": 1,
        "artifact_type": "full300_official_score_binding",
        "status": "complete",
        "name": composite["name"],
        "full_ids": _binding(full_ids_path),
        "composite": _binding(composite_path),
        "predictions": _binding(predictions_path),
        "score_log_root": str(score_log_root),
        "model_contract": composite["model_contract"],
        "harness_contract": composite["harness_contract"],
        "panels": panel_rows,
        "final": {
            "resolved_ids": composite["resolved_ids"],
            "empty_ids": empty_ids,
            "resolved": composite["resolved"],
            "empty": len(empty_ids),
        },
    }


def validate_official_score_binding(
    contract_path: Path,
    *,
    full_ids_path: Path,
    composite_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    report = _read_object(contract_path)
    panel_values = report.get("panels")
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type")
        != "full300_official_score_binding"
        or report.get("status") != "complete"
        or not isinstance(panel_values, list)
        or len(panel_values) != 2
        or not isinstance(report.get("score_log_root"), str)
    ):
        raise ValueError("official score binding is incomplete")
    panels = []
    for row in panel_values:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("tag"), str)
        ):
            raise ValueError("official score panel binding is incomplete")
        bindings = []
        for key in ("ids", "composite", "predictions"):
            binding = row.get(key)
            if (
                not isinstance(binding, Mapping)
                or not isinstance(binding.get("path"), str)
                or dict(binding) != _binding(Path(binding["path"]))
            ):
                raise ValueError(
                    f"official score panel {row.get('tag')} changed"
                )
            bindings.append(Path(binding["path"]))
        panels.append(
            ScorePanel(
                tag=row["tag"],
                ids_path=bindings[0],
                composite_path=bindings[1],
                predictions_path=bindings[2],
            )
        )
    expected = build_official_score_binding(
        full_ids_path=full_ids_path,
        composite_path=composite_path,
        predictions_path=predictions_path,
        panels=panels,
        score_log_root=Path(report["score_log_root"]),
        verify_panels=False,
    )
    if report != expected:
        raise ValueError(
            "official score binding differs from current evidence"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--panel",
        action="append",
        nargs=4,
        required=True,
        metavar=("TAG", "IDS", "COMPOSITE", "PREDICTIONS"),
    )
    parser.add_argument(
        "--score-log-root",
        type=Path,
        default=Path("logs/run_evaluation"),
    )
    args = parser.parse_args()
    report = build_official_score_binding(
        full_ids_path=args.full_ids,
        composite_path=args.composite,
        predictions_path=args.predictions,
        panels=[
            ScorePanel(
                tag=tag,
                ids_path=Path(ids),
                composite_path=Path(composite),
                predictions_path=Path(predictions),
            )
            for tag, ids, composite, predictions in args.panel
        ],
        score_log_root=args.score_log_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
