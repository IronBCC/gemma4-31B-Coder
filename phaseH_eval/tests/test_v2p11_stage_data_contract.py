from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.v2p11_stage_data_contract import validate_stage_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    campaign = tmp_path / "campaign"
    admitted_evidence = {
        "admission_schema_version": 2,
        "resolved": True,
        "training_admitted": True,
        "executed": True,
        "f2p_pass": True,
        "p2p_pass": True,
        "reference_passed": True,
        "reference_controls_passed": True,
        "baseline_failed": True,
        "candidate_passed_twice": True,
        "mutation_f2p_reproduced": True,
        "mutation_p2p_passed": True,
        "protected_stable": True,
        "cli_rc": 0,
        "patch_len": 42,
        "rejection_reasons": [],
        "admission_evidence_sha256": "a" * 64,
        "patch_sha256": "b" * 64,
        "stream_sha256": "c" * 64,
        "task_contract_sha256": "d" * 64,
    }
    resolved = [
        {"instance_id": "owner__legacy.case"},
        {
            "instance_id": "owner__new.case",
            **admitted_evidence,
        },
    ]
    rejected = [{"instance_id": "owner__rejected.case"}]
    results = [*resolved, *rejected]
    _write_jsonl(campaign / "results.jsonl", results)
    _write_jsonl(campaign / "resolved.jsonl", resolved)
    _write_jsonl(campaign / "rejected.jsonl", rejected)
    _write_json(
        campaign / "manifest.json",
        {
            "schema_version": 2,
            "complete": True,
            "attempted": 3,
            "resolved": 2,
            "rejected": 1,
            "training_admitted": 2,
            "selection_excluded": 0,
            "results_sha256": _sha256(campaign / "results.jsonl"),
            "resolved_sha256": _sha256(campaign / "resolved.jsonl"),
            "rejected_sha256": _sha256(campaign / "rejected.jsonl"),
        },
    )

    strict = tmp_path / "strict"
    strict_rows = [{
        "instance_id": "owner__new.case",
        "repo": "owner/new",
        "source": "teacher:claude:claude-fable-5",
        "messages": [],
    }]
    _write_jsonl(strict / "train.jsonl", strict_rows)
    format_gate = strict / "format-gate.json"
    _write_json(
        format_gate,
        {
            "status": "passed",
            "failure_count": 0,
            "samples": 1,
        },
    )
    _write_json(
        strict / "manifest.json",
        {
            "schema_version": 2,
            "complete": True,
            "rendered": 1,
            "training_admitted": 1,
            "all_training_gates_complete": True,
            "success_path_distilled": True,
            "train_jsonl_sha256": _sha256(strict / "train.jsonl"),
            "resolved_in": 2,
            "merge_manifest_sha256": _sha256(campaign / "manifest.json"),
            "merge_resolved_sha256": _sha256(campaign / "resolved.jsonl"),
            "artifact_bindings": [{
                "instance_id": "owner__new.case",
                "admission_evidence_sha256": "a" * 64,
                "patch_sha256": "b" * 64,
                "stream_sha256": "c" * 64,
                "task_contract_sha256": "d" * 64,
            }],
            "standard_native_format_loss_gate": {
                "status": "passed",
                "failure_count": 0,
                "samples": 1,
                "counts": {
                    "examples": 1,
                    "fallback_spans": 0,
                    "supervised_fallback_spans": 0,
                },
                "artifact": {
                    "path": format_gate.name,
                    "sha256": _sha256(format_gate),
                    "bytes": format_gate.stat().st_size,
                },
            },
            "distillation_exclusions": {
                "count": 1,
                "artifact_bindings": [{
                    "instance_id": "owner__legacy.case",
                    "stage": "success_path_distillation",
                    "reason": "no focused passing test",
                }],
            },
        },
    )

    stage = tmp_path / "stage"
    stage_rows = [
        {
            "instance_id": "safe__inherited.case",
            "repo": "safe/inherited",
            "source": "teacher:fable5:hash:trace",
            "messages": [],
        },
        {
            "instance_id": "safe__revision.case",
            "repo": "safe/revision",
            "source": "teacher:claude:claude-fable-5:verified-revision",
            "messages": [],
        },
        {
            "instance_id": "safe__other.case",
            "repo": "safe/other",
            "source": "teacher:open-swe:sweagent:minimax_m25",
            "messages": [],
        },
        strict_rows[0],
    ]
    _write_jsonl(stage / "train.jsonl", stage_rows)
    strict_manifest = json.loads((strict / "manifest.json").read_text())
    _write_json(
        stage / "manifest.json",
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11",
            "base_rows": 3,
            "fable_rows": 1,
            "rendered": 4,
            "training_admitted": 4,
            "all_training_gates_complete": True,
            "train_jsonl_sha256": _sha256(stage / "train.jsonl"),
            "fable_instance_ids": ["owner__new.case"],
            "fable": {
                "path": str(strict),
                "manifest": strict_manifest,
                "manifest_sha256": _sha256(strict / "manifest.json"),
            },
        },
    )
    training_contract = tmp_path / "training-contract.json"
    _write_json(
        training_contract,
        {
            "rows": 4,
            "base_rows": 3,
            "fable_rows": 1,
            "dataset_manifest_sha256": _sha256(stage / "manifest.json"),
            "train_jsonl_sha256": _sha256(stage / "train.jsonl"),
        },
    )
    exclusions = tmp_path / "exclusions.json"
    _write_json(
        exclusions,
        {
            "instance_ids": ["eval__one.case", "eval__two.case"],
            "repo_denylist": ["eval/repo"],
        },
    )
    return {
        "stage": stage,
        "training_contract": training_contract,
        "exclusions": exclusions,
        "campaign": campaign,
        "strict": strict,
    }


def _refresh_campaign_and_stage_bindings(paths: dict[str, Path]) -> None:
    campaign_manifest_path = paths["campaign"] / "manifest.json"
    campaign_manifest = json.loads(campaign_manifest_path.read_text())
    campaign_manifest["resolved_sha256"] = _sha256(
        paths["campaign"] / "resolved.jsonl"
    )
    _write_json(campaign_manifest_path, campaign_manifest)

    strict_manifest_path = paths["strict"] / "manifest.json"
    strict_manifest = json.loads(strict_manifest_path.read_text())
    strict_manifest["merge_manifest_sha256"] = _sha256(
        campaign_manifest_path
    )
    strict_manifest["merge_resolved_sha256"] = campaign_manifest[
        "resolved_sha256"
    ]
    _write_json(strict_manifest_path, strict_manifest)

    stage_manifest_path = paths["stage"] / "manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text())
    stage_manifest["fable"]["manifest"] = strict_manifest
    stage_manifest["fable"]["manifest_sha256"] = _sha256(
        strict_manifest_path
    )
    _write_json(stage_manifest_path, stage_manifest)

    training = json.loads(paths["training_contract"].read_text())
    training["dataset_manifest_sha256"] = _sha256(stage_manifest_path)
    _write_json(paths["training_contract"], training)


def test_binds_complete_fable_disposition_lineage_and_exclusions(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    contract = validate_stage_data(
        stage_data=paths["stage"],
        training_contract_path=paths["training_contract"],
        exclusions_path=paths["exclusions"],
        campaign_root=paths["campaign"],
    )

    assert contract["fable_lineage"] == {
        "inherited_pinned_rows": 1,
        "historical_verified_revision_rows": 1,
        "new_strict_rows": 1,
        "total_stage_a_fable_rows": 3,
    }
    assert contract["fable_freeze"] == {
        "attempted": 3,
        "resolved": 2,
        "collection_rejected": 1,
        "strict_admitted": 1,
        "distillation_rejected": 1,
    }
    assert contract["evaluation_exclusion"]["overlap"] == 0
    assert contract["evaluation_exclusion"]["ids"] == 2
    assert contract["evaluation_exclusion"]["repositories"] == 1


def test_rejects_stage_row_from_combined_evaluation_population(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    exclusions = json.loads(paths["exclusions"].read_text())
    exclusions["instance_ids"].append("safe__other.case")
    _write_json(paths["exclusions"], exclusions)

    with pytest.raises(ValueError, match="evaluation exclusion"):
        validate_stage_data(
            stage_data=paths["stage"],
            training_contract_path=paths["training_contract"],
            exclusions_path=paths["exclusions"],
            campaign_root=paths["campaign"],
        )


def test_rejects_tampered_fable_disposition_ledger(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    with (paths["campaign"] / "rejected.jsonl").open("a") as handle:
        handle.write(json.dumps({"instance_id": "tampered__case"}) + "\n")

    with pytest.raises(ValueError, match="campaign artifact changed"):
        validate_stage_data(
            stage_data=paths["stage"],
            training_contract_path=paths["training_contract"],
            exclusions_path=paths["exclusions"],
            campaign_root=paths["campaign"],
        )


def test_rejects_malformed_distillation_binding(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    stage_manifest = json.loads(
        (paths["stage"] / "manifest.json").read_text()
    )
    strict_root = Path(stage_manifest["fable"]["path"])
    strict_manifest = json.loads(
        (strict_root / "manifest.json").read_text()
    )
    strict_manifest["distillation_exclusions"][
        "artifact_bindings"
    ].append(None)
    _write_json(strict_root / "manifest.json", strict_manifest)
    stage_manifest["fable"]["manifest"] = strict_manifest
    stage_manifest["fable"]["manifest_sha256"] = _sha256(
        strict_root / "manifest.json"
    )
    _write_json(paths["stage"] / "manifest.json", stage_manifest)
    training = json.loads(paths["training_contract"].read_text())
    training["dataset_manifest_sha256"] = _sha256(
        paths["stage"] / "manifest.json"
    )
    _write_json(paths["training_contract"], training)

    with pytest.raises(ValueError, match="disposition bindings"):
        validate_stage_data(
            stage_data=paths["stage"],
            training_contract_path=paths["training_contract"],
            exclusions_path=paths["exclusions"],
            campaign_root=paths["campaign"],
        )


def test_rejects_admitted_fable_without_execution_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    resolved_path = paths["campaign"] / "resolved.jsonl"
    resolved = [
        json.loads(line)
        for line in resolved_path.read_text().splitlines()
        if line
    ]
    resolved[1]["executed"] = False
    _write_jsonl(resolved_path, resolved)
    _refresh_campaign_and_stage_bindings(paths)

    with pytest.raises(ValueError, match="admission evidence"):
        validate_stage_data(
            stage_data=paths["stage"],
            training_contract_path=paths["training_contract"],
            exclusions_path=paths["exclusions"],
            campaign_root=paths["campaign"],
        )
