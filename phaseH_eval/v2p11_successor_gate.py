#!/usr/bin/env python3
"""Gate the v2.11r4 successor before a full SWE-bench Lite evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseH_eval.empty_retry_composite import (
    _binding,
    _complete_run,
    _publish_json_noreplace,
    _read_object,
    _trajectory_health,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract
from phaseH_eval.v2p11_interpolation_lineage import (
    validate_interpolation_lineage,
)
from phaseH_eval.v2p11_portability_gate import evaluate_portability
from phaseH_eval.v2p11r3_behavior_completion_provenance import (
    validate_completion_provenance as validate_source_provenance,
)


ARTIFACT_TYPE = "v2p11r4_successor_prefull_gate"
EXPECTED_CANDIDATE_NAME = "teacher_sft_v2p11r4_blend25"
SOURCE_CANDIDATE_NAME = "teacher_sft_v2p11r3_behavior"
CANONICAL_ANCHOR_MODEL = Path(
    "/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full"
).resolve()
CANONICAL_SOURCE_MODEL = Path(
    "/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full"
).resolve()
CANONICAL_OUTPUT_MODEL = Path(
    "/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r4_blend25"
).resolve()


def _metric_count(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid successor metric: {key}")
    return value


def successor_decision(
    *,
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    for label, metrics in (("control", control), ("candidate", candidate)):
        expected = _metric_count(metrics, "expected")
        resolved = _metric_count(metrics, "resolved")
        empty = _metric_count(metrics, "empty")
        wrong = _metric_count(metrics, "wrong_nonempty")
        _metric_count(metrics, "repeat_loops")
        if expected != 150 or resolved + empty + wrong != expected:
            raise ValueError(f"{label} fixed150 partition is incomplete")
    wins = _metric_count(paired, "wins")
    losses = _metric_count(paired, "losses")
    ties = _metric_count(paired, "ties")
    if wins + losses + ties != 150:
        raise ValueError("paired fixed150 partition is incomplete")

    criteria = {
        "resolved_floor": (
            candidate["resolved"] >= control["resolved"]
        ),
        "paired_wins_no_less_than_losses": wins >= losses,
        "wrong_nonempty_no_regression": (
            candidate["wrong_nonempty"] <= control["wrong_nonempty"]
        ),
        "empty_no_regression": candidate["empty"] <= control["empty"],
        "repeat_loop_no_regression": (
            candidate["repeat_loops"] <= control["repeat_loops"]
        ),
        "strict_behavior_improvement": (
            candidate["empty"] < control["empty"]
            or candidate["repeat_loops"] < control["repeat_loops"]
        ),
    }
    failed = [key for key, passed in criteria.items() if not passed]
    allowed = not failed
    return {
        "run_full300": allowed,
        "reason": (
            "v2.11r4 pre-full300 gate passed"
            if allowed
            else "v2.11r4 pre-full300 gate rejected"
        ),
        "criteria": criteria,
        "failed_criteria": failed,
    }


def _evaluation_model_contract(
    lineage_contract: Mapping[str, Any],
    *,
    served_name: str,
) -> dict[str, Any]:
    result = {
        "served_name": served_name,
        "model_path": lineage_contract.get("model_path"),
        "model_config_sha256": lineage_contract.get(
            "model_config_sha256"
        ),
    }
    if lineage_contract.get("model_index_sha256") is not None:
        result["model_index_sha256"] = lineage_contract[
            "model_index_sha256"
        ]
    else:
        result["model_safetensors_sha256"] = lineage_contract.get(
            "model_safetensors_sha256"
        )
    result["model_artifacts"] = lineage_contract.get("model_artifacts")
    return result


def _run_metrics(
    *,
    run: Mapping[str, Any],
    ids: Sequence[str],
) -> dict[str, Any]:
    resolved = set(run["resolved_ids"])
    empty = set(run["empty_ids"])
    population = set(ids)
    wrong = population - resolved - empty
    behavior_by_instance = {
        instance_id: _trajectory_health(run["trajectories"][instance_id])
        for instance_id in ids
    }
    return {
        "expected": len(ids),
        "resolved": len(resolved),
        "resolved_ids": sorted(resolved),
        "empty": len(empty),
        "empty_ids": sorted(empty),
        "wrong_nonempty": len(wrong),
        "wrong_nonempty_ids": sorted(wrong),
        "repeat_loops": sum(
            row["repeat_loops"] for row in behavior_by_instance.values()
        ),
        "repeat_loop_ids": sorted(
            instance_id
            for instance_id, row in behavior_by_instance.items()
            if row["repeat_loops"] > 0
        ),
        "tool_format_errors": sum(
            row["tool_format_errors"]
            for row in behavior_by_instance.values()
        ),
        "assistant_responses": sum(
            row["assistant_responses"]
            for row in behavior_by_instance.values()
        ),
        "model_contract": run["model_contract"],
        "harness_contract": run["harness_contract"],
        "artifacts": run["artifacts"],
    }


def _path_inputs(**values: Path | str) -> dict[str, str]:
    return {
        key: (
            str(Path(value).resolve())
            if isinstance(value, Path)
            else value
        )
        for key, value in values.items()
    }


def _require_canonical_model_paths(paths: Mapping[str, Path]) -> None:
    expected = {
        "anchor_model": CANONICAL_ANCHOR_MODEL,
        "source_model": CANONICAL_SOURCE_MODEL,
        "output_model": CANONICAL_OUTPUT_MODEL,
        "manifest": CANONICAL_OUTPUT_MODEL / "interpolation_manifest.json",
    }
    changed = [key for key, value in expected.items() if paths.get(key) != value]
    if changed:
        raise ValueError(
            "v2.11r4 canonical model paths changed: " + ", ".join(changed)
        )


def evaluate_successor_gate(
    *,
    fixed_ids_path: Path,
    control_run_root: Path,
    candidate_run_root: Path,
    manifest_path: Path,
    anchor_model_path: Path,
    source_model_path: Path,
    output_model_path: Path,
    candidate_name: str,
    portability_gate_path: Path,
    portability_ids_path: Path,
    verified_exclusions_path: Path,
    lite_ids_path: Path,
    portability_control_root: Path,
    portability_candidate_root: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
    source_provenance_path: Path,
) -> dict[str, Any]:
    paths = {
        key: Path(value).resolve()
        for key, value in {
            "fixed_ids": fixed_ids_path,
            "control_run_root": control_run_root,
            "candidate_run_root": candidate_run_root,
            "manifest": manifest_path,
            "anchor_model": anchor_model_path,
            "source_model": source_model_path,
            "output_model": output_model_path,
            "portability_gate": portability_gate_path,
            "portability_ids": portability_ids_path,
            "verified_exclusions": verified_exclusions_path,
            "lite_ids": lite_ids_path,
            "portability_control_root": portability_control_root,
            "portability_candidate_root": portability_candidate_root,
            "full_ids": full_ids_path,
            "v2p10_composite": v2p10_composite_path,
            "v2p10_lineage": v2p10_lineage_path,
            "source_provenance": source_provenance_path,
        }.items()
    }
    if candidate_name != EXPECTED_CANDIDATE_NAME:
        raise ValueError("candidate served name changed")
    _require_canonical_model_paths(paths)
    fixed_ids = json.loads(paths["fixed_ids"].read_text(encoding="utf-8"))
    full_ids = json.loads(paths["full_ids"].read_text(encoding="utf-8"))
    if (
        not isinstance(fixed_ids, list)
        or len(fixed_ids) != 150
        or len(fixed_ids) != len(set(fixed_ids))
        or not isinstance(full_ids, list)
        or len(full_ids) != 300
        or len(full_ids) != len(set(full_ids))
        or not set(fixed_ids).issubset(full_ids)
    ):
        raise ValueError("successor evaluation populations changed")

    lineage = validate_interpolation_lineage(
        manifest_path=paths["manifest"],
        anchor_model_path=paths["anchor_model"],
        source_model_path=paths["source_model"],
        output_model_path=paths["output_model"],
        candidate_name=candidate_name,
    )
    portability = evaluate_portability(
        ids_path=paths["portability_ids"],
        verified_exclusions_path=paths["verified_exclusions"],
        lite_ids_path=paths["lite_ids"],
        control_root=paths["portability_control_root"],
        candidate_root=paths["portability_candidate_root"],
        candidate_name=candidate_name,
    )
    if _read_object(paths["portability_gate"]) != portability:
        raise ValueError("portability gate changed")
    if portability.get("passed") is not True:
        raise ValueError("portability gate did not pass")

    source_contract = _model_contract(paths["source_model"])
    source_contract["served_name"] = SOURCE_CANDIDATE_NAME
    source_provenance = validate_source_provenance(
        paths["source_provenance"],
        full_ids_path=paths["full_ids"],
        v2p10_composite_path=paths["v2p10_composite"],
        v2p10_lineage_path=paths["v2p10_lineage"],
        candidate_model_contract=source_contract,
        candidate_name=SOURCE_CANDIDATE_NAME,
    )

    control_run = _complete_run(
        ids_path=paths["fixed_ids"],
        run_root=paths["control_run_root"],
        label="v2p10 fixed150 first pass",
    )
    candidate_run = _complete_run(
        ids_path=paths["fixed_ids"],
        run_root=paths["candidate_run_root"],
        label="v2p11r4 fixed150 first pass",
    )
    if control_run["acceptance"].get("name") != "teacher_sft_v2p10":
        raise ValueError("fixed150 control name changed")
    if candidate_run["acceptance"].get("name") != candidate_name:
        raise ValueError("fixed150 candidate name changed")
    if control_run["harness_contract"] != candidate_run["harness_contract"]:
        raise ValueError("fixed150 harness contract changed")
    if control_run["model_contract"] != _evaluation_model_contract(
        lineage["anchor"], served_name="teacher_sft_v2p10"
    ):
        raise ValueError("fixed150 control model contract changed")
    if candidate_run["model_contract"] != _evaluation_model_contract(
        lineage["output"], served_name=candidate_name
    ):
        raise ValueError("fixed150 candidate model contract changed")

    control = _run_metrics(run=control_run, ids=fixed_ids)
    candidate = _run_metrics(run=candidate_run, ids=fixed_ids)
    control_resolved = set(control["resolved_ids"])
    candidate_resolved = set(candidate["resolved_ids"])
    paired = {
        "wins": len(candidate_resolved - control_resolved),
        "losses": len(control_resolved - candidate_resolved),
        "ties": 150 - len(candidate_resolved ^ control_resolved),
        "candidate_only": sorted(candidate_resolved - control_resolved),
        "control_only": sorted(control_resolved - candidate_resolved),
        "both_resolved": sorted(candidate_resolved & control_resolved),
        "neither_resolved": sorted(
            set(fixed_ids) - candidate_resolved - control_resolved
        ),
    }
    if paired["wins"] - paired["losses"] != (
        candidate["resolved"] - control["resolved"]
    ):
        raise ValueError("paired fixed150 evidence is inconsistent")
    decision = successor_decision(
        control=control,
        candidate=candidate,
        paired=paired,
    )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete",
        "passed": decision["run_full300"],
        "candidate_name": candidate_name,
        "population": 150,
        "full_population": 300,
        "input_paths": _path_inputs(
            **paths,
            candidate_name=candidate_name,
        ),
        "fixed_ids": _binding(paths["fixed_ids"]),
        "full_ids": _binding(paths["full_ids"]),
        "v2p10_full300": _binding(paths["v2p10_composite"]),
        "v2p10_lineage": _binding(paths["v2p10_lineage"]),
        "interpolation_lineage": lineage,
        "source_provenance": source_provenance,
        "portability": portability,
        "control": control,
        "candidate": candidate,
        "paired": paired,
        "empty_patch": {
            "control": control["empty_ids"],
            "candidate": candidate["empty_ids"],
            "eliminated": sorted(
                set(control["empty_ids"]) - set(candidate["empty_ids"])
            ),
            "introduced": sorted(
                set(candidate["empty_ids"]) - set(control["empty_ids"])
            ),
        },
        "wrong_nonempty": {
            "control": control["wrong_nonempty_ids"],
            "candidate": candidate["wrong_nonempty_ids"],
            "delta": (
                candidate["wrong_nonempty"] - control["wrong_nonempty"]
            ),
        },
        "decision": decision,
    }


def validate_interpolation_completion_provenance(
    provenance_path: Path,
    *,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
    candidate_model_contract: Mapping[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    provenance_path = Path(provenance_path).resolve()
    report = _read_object(provenance_path)
    inputs = report.get("input_paths")
    if (
        report.get("artifact_type") != ARTIFACT_TYPE
        or report.get("status") != "complete"
        or report.get("passed") is not True
        or not isinstance(inputs, Mapping)
        or set(inputs)
        != {
            "fixed_ids",
            "control_run_root",
            "candidate_run_root",
            "manifest",
            "anchor_model",
            "source_model",
            "output_model",
            "portability_gate",
            "portability_ids",
            "verified_exclusions",
            "lite_ids",
            "portability_control_root",
            "portability_candidate_root",
            "full_ids",
            "v2p10_composite",
            "v2p10_lineage",
            "source_provenance",
            "candidate_name",
        }
    ):
        raise ValueError("v2.11r4 completion provenance is incomplete")
    expected = evaluate_successor_gate(
        fixed_ids_path=Path(str(inputs["fixed_ids"])),
        control_run_root=Path(str(inputs["control_run_root"])),
        candidate_run_root=Path(str(inputs["candidate_run_root"])),
        manifest_path=Path(str(inputs["manifest"])),
        anchor_model_path=Path(str(inputs["anchor_model"])),
        source_model_path=Path(str(inputs["source_model"])),
        output_model_path=Path(str(inputs["output_model"])),
        candidate_name=str(inputs["candidate_name"]),
        portability_gate_path=Path(str(inputs["portability_gate"])),
        portability_ids_path=Path(str(inputs["portability_ids"])),
        verified_exclusions_path=Path(str(inputs["verified_exclusions"])),
        lite_ids_path=Path(str(inputs["lite_ids"])),
        portability_control_root=Path(
            str(inputs["portability_control_root"])
        ),
        portability_candidate_root=Path(
            str(inputs["portability_candidate_root"])
        ),
        full_ids_path=Path(str(inputs["full_ids"])),
        v2p10_composite_path=Path(str(inputs["v2p10_composite"])),
        v2p10_lineage_path=Path(str(inputs["v2p10_lineage"])),
        source_provenance_path=Path(str(inputs["source_provenance"])),
    )
    if report != expected:
        raise ValueError("v2.11r4 completion provenance changed")
    if report["full_ids"] != _binding(Path(full_ids_path)):
        raise ValueError("v2.11r4 full evaluation population changed")
    if report["v2p10_full300"] != _binding(Path(v2p10_composite_path)):
        raise ValueError("v2.11r4 v2.10 comparison changed")
    if report["v2p10_lineage"] != _binding(Path(v2p10_lineage_path)):
        raise ValueError("v2.11r4 v2.10 lineage changed")
    if report["candidate"]["model_contract"] != dict(
        candidate_model_contract
    ) or candidate_name != report["candidate_name"]:
        raise ValueError("v2.11r4 provenance does not bind evaluated model")
    source = report["source_provenance"]
    return {
        "artifact": _binding(provenance_path),
        "fable": source["fable"],
        "evaluation_exclusion": source["evaluation_exclusion"],
        "dataset": source.get("dataset"),
        "training": source.get("training"),
        "behavior_poststage": source.get("behavior_poststage"),
        "final_model": dict(candidate_model_contract),
        "interpolation_lineage": report["interpolation_lineage"],
        "portability": {
            "gate": _binding(Path(str(inputs["portability_gate"]))),
            "passed": True,
            "population": report["portability"]["population"],
            "criteria": report["portability"]["criteria"],
        },
        "prefull_gate": {
            "passed": True,
            "criteria": report["decision"]["criteria"],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-ids", type=Path, required=True)
    parser.add_argument("--control-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-model", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--portability-ids", type=Path, required=True)
    parser.add_argument("--verified-exclusions", type=Path, required=True)
    parser.add_argument("--lite-ids", type=Path, required=True)
    parser.add_argument("--portability-control-root", type=Path, required=True)
    parser.add_argument("--portability-candidate-root", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10-composite", type=Path, required=True)
    parser.add_argument("--v2p10-lineage", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    values = vars(args).copy()
    output = values.pop("out")
    report = evaluate_successor_gate(
        fixed_ids_path=values.pop("fixed_ids"),
        control_run_root=values.pop("control_run_root"),
        candidate_run_root=values.pop("candidate_run_root"),
        manifest_path=values.pop("manifest"),
        anchor_model_path=values.pop("anchor_model"),
        source_model_path=values.pop("source_model"),
        output_model_path=values.pop("output_model"),
        candidate_name=values.pop("candidate_name"),
        portability_gate_path=values.pop("portability_gate"),
        portability_ids_path=values.pop("portability_ids"),
        verified_exclusions_path=values.pop("verified_exclusions"),
        lite_ids_path=values.pop("lite_ids"),
        portability_control_root=values.pop("portability_control_root"),
        portability_candidate_root=values.pop("portability_candidate_root"),
        full_ids_path=values.pop("full_ids"),
        v2p10_composite_path=values.pop("v2p10_composite"),
        v2p10_lineage_path=values.pop("v2p10_lineage"),
        source_provenance_path=values.pop("source_provenance"),
    )
    if values:
        raise AssertionError(f"unconsumed arguments: {sorted(values)}")
    if os.path.lexists(output):
        if _read_object(output) != report:
            raise ValueError("existing successor gate differs from current inputs")
    else:
        _publish_json_noreplace(output, report)
    print(
        json.dumps(
            {
                "run_full300": report["decision"]["run_full300"],
                "failed_criteria": report["decision"]["failed_criteria"],
                "control_resolved": report["control"]["resolved"],
                "candidate_resolved": report["candidate"]["resolved"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
