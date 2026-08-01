#!/usr/bin/env python3
"""Publish the final requirement-level audit for a trustworthy v2.11r3 win."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phaseH_eval.compare_v2p11_full300 import validate_completion_provenance
from phaseH_eval.empty_retry_composite import _binding, _publish_json_noreplace, _read_ids, _read_object
from phaseH_eval.full300_official_score_binding import validate_official_score_binding
from phaseH_eval.v2p11_completion_provenance import _model_contract
from phaseH_eval.v2p11r2_goal_completion_audit import _validate_trustworthy_verdict


def _validate_reasoned_lineage(provenance: Mapping[str, Any]) -> None:
    training = provenance.get("training")
    dataset = provenance.get("dataset")
    if not isinstance(training, Mapping) or not isinstance(dataset, Mapping):
        raise ValueError("r3 training provenance is incomplete")
    stage = dataset.get("stage_a")
    source_fable = dataset.get("source_fable")
    stage_summary = stage.get("summary") if isinstance(stage, Mapping) else None
    source_ids = source_fable.get("source_ids") if isinstance(source_fable, Mapping) else None
    if (
        training.get("rows") != 1262
        or training.get("max_seq") != 32768
        or training.get("optimizer_steps") != 79
        or type(dataset.get("max_rendered_tokens")) is not int
        or not 1 <= dataset["max_rendered_tokens"] <= 32768
        or dataset.get("fable_rows") != 92
        or dataset.get("canonical_bash_tool_turns") != 517
        or dataset.get("reasoned_tool_turns") != 468
        or stage_summary != {"rows": 1247, "base_rows": 1211, "new_fable_rows": 36}
        or not isinstance(source_ids, list)
        or len(source_ids) != 15
        or len(set(source_ids)) != 15
        or any(not isinstance(item, str) or not item for item in source_ids)
    ):
        raise ValueError("r3 Fable reasoning-contract evidence is incomplete")


def _validate_behavior_poststage(provenance: Mapping[str, Any]) -> None:
    poststage = provenance.get("behavior_poststage")
    if not isinstance(poststage, Mapping) or {
        "recovery_rows": poststage.get("recovery_rows"),
        "recovery_optimizer_steps": poststage.get(
            "recovery_optimizer_steps"
        ),
        "behavior_rows": poststage.get("behavior_rows"),
        "kto_optimizer_steps": poststage.get("kto_optimizer_steps"),
        "coverage_counts": poststage.get("coverage_counts"),
        "negative_counts": poststage.get("negative_counts"),
    } != {
        "recovery_rows": 138,
        "recovery_optimizer_steps": 105,
        "behavior_rows": 606,
        "kto_optimizer_steps": 25,
        "coverage_counts": {
            "desirable_correct_patch": 25,
            "empty_terminal": 8,
            "repeated_read_loop": 8,
            "wrong_nonempty_replay": 9,
        },
        "negative_counts": {
            "empty_terminal": 33,
            "repeated_read_loop": 55,
            "wrong_nonempty_replay": 228,
        },
    }:
        raise ValueError("r3 behavior poststage evidence is incomplete")


def publish_goal_audit(
    *, verdict_path: Path, full_ids_path: Path,
    v2p10_composite_path: Path, v2p10_predictions_path: Path,
    v2p10_score_path: Path, v2p11_composite_path: Path,
    v2p11_predictions_path: Path, v2p11_score_path: Path,
    provenance_path: Path, candidate_model_path: Path,
    candidate_name: str, output_path: Path,
) -> dict[str, Any]:
    verdict_path = Path(verdict_path).resolve()
    full_ids_path = Path(full_ids_path).resolve()
    v2p10_composite_path = Path(v2p10_composite_path).resolve()
    v2p10_predictions_path = Path(v2p10_predictions_path).resolve()
    v2p10_score_path = Path(v2p10_score_path).resolve()
    v2p11_composite_path = Path(v2p11_composite_path).resolve()
    v2p11_predictions_path = Path(v2p11_predictions_path).resolve()
    v2p11_score_path = Path(v2p11_score_path).resolve()
    provenance_path = Path(provenance_path).resolve()
    candidate_model_path = Path(candidate_model_path).resolve()
    full_ids = _read_ids(full_ids_path)
    if len(full_ids) != 300 or len(set(full_ids)) != 300:
        raise ValueError("final audit requires the exact Lite300 population")
    verdict = _read_object(verdict_path)
    _validate_trustworthy_verdict(
        verdict, candidate_name=candidate_name, full_ids=full_ids,
    )
    v2p10_score = validate_official_score_binding(
        v2p10_score_path, full_ids_path=full_ids_path,
        composite_path=v2p10_composite_path,
        predictions_path=v2p10_predictions_path,
    )
    v2p11_score = validate_official_score_binding(
        v2p11_score_path, full_ids_path=full_ids_path,
        composite_path=v2p11_composite_path,
        predictions_path=v2p11_predictions_path,
    )
    model_contract = _model_contract(candidate_model_path)
    model_contract["served_name"] = candidate_name
    provenance = validate_completion_provenance(
        provenance_path, full_ids_path=full_ids_path,
        v2p10_composite_path=v2p10_composite_path,
        candidate_model_contract=model_contract, candidate_name=candidate_name,
    )
    _validate_reasoned_lineage(provenance)
    _validate_behavior_poststage(provenance)
    control = verdict["v2p10"]
    candidate = verdict["v2p11"]
    assert isinstance(control, Mapping) and isinstance(candidate, Mapping)
    report = {
        "schema_version": 1,
        "artifact_type": "v2p11r3_goal_completion_audit",
        "status": "complete",
        "scope": ["v2p10", "v2p11r3"],
        "requirements": {
            "same_complete_lite300": True,
            "v2p11_beats_v2p10": True,
            "empty_patch_no_regression": True,
            "behavior_healthy": True,
            "stage_a_strict_fable_rows": 36,
            "recent_strict_fable_rows": 15,
            "reasoned_fable_tool_turns": 468,
            "portable_recovery_rows": 138,
            "recovery_optimizer_steps": 105,
            "behavior_kto_rows": 606,
            "behavior_kto_optimizer_steps": 25,
            "behavior_kto_coverage": {
                "desirable_correct_patch": 25,
                "empty_terminal": 8,
                "repeated_read_loop": 8,
                "wrong_nonempty_replay": 9,
            },
            "full_context_window": 32768,
            "official_scores_bound": True,
            "failure_analysis_bound": True,
        },
        "score": {
            "v2p10": {"resolved": control["resolved"], "empty": control["empty"]},
            "v2p11r3": {"resolved": candidate["resolved"], "empty": candidate["empty"]},
            "resolved_delta": candidate["resolved"] - control["resolved"],
        },
        "artifacts": {
            "full_ids": _binding(full_ids_path),
            "verdict": _binding(verdict_path),
            "v2p10_official_score": _binding(v2p10_score_path),
            "v2p11r3_official_score": _binding(v2p11_score_path),
            "provenance": _binding(provenance_path),
        },
    }
    output_path = Path(output_path).resolve()
    if output_path.exists():
        if _read_object(output_path) != report:
            raise ValueError(
                "existing goal audit differs from current inputs"
            )
    else:
        _publish_json_noreplace(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--v2p10-preds", type=Path, required=True)
    parser.add_argument("--v2p10-score", type=Path, required=True)
    parser.add_argument("--v2p11", type=Path, required=True)
    parser.add_argument("--v2p11-preds", type=Path, required=True)
    parser.add_argument("--v2p11-score", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_goal_audit(
        verdict_path=args.verdict, full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10, v2p10_predictions_path=args.v2p10_preds,
        v2p10_score_path=args.v2p10_score, v2p11_composite_path=args.v2p11,
        v2p11_predictions_path=args.v2p11_preds, v2p11_score_path=args.v2p11_score,
        provenance_path=args.provenance, candidate_model_path=args.candidate_model,
        candidate_name=args.candidate_name, output_path=args.out,
    )
    print(json.dumps(report["score"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
