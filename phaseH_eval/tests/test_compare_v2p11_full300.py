from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.compare_v2p11_full300 as compare_module
from phaseH_eval.compare_v2p11_full300 import (
    FirstPassPanel,
    compare_full300,
    publish_verdict_outputs,
)
from phaseH_eval.empty_retry_composite import _complete_run
from phaseH_eval.full300_panel_composite import (
    PanelInput,
    join_disjoint_panels,
)
from phaseH_eval.tests.test_empty_retry_composite import _write_run
from phaseH_eval.tests.test_full300_panel_composite import (
    HARNESS_CONTRACT,
    _write_panel,
)
from phaseH_eval.tests.v2p11_lineage_fixture import (
    create_complete_phase_lineage,
)


def _artifact_binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_official_model_failure_uses_latest_retry_attempt(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    retry_ids = tmp_path / "retry_ids.json"
    source_ids.write_text(json.dumps(["case-a", "case-b"]) + "\n")
    retry_ids.write_text(json.dumps(["case-a"]) + "\n")
    report = {
        "panels": [{
            "runs": [
                {
                    "ids": _artifact_binding(source_ids),
                    "official": {
                        "model_failure_ids": ["case-a", "case-b"],
                    },
                },
                {
                    "ids": _artifact_binding(retry_ids),
                    "official": {"model_failure_ids": []},
                },
            ],
        }],
    }

    assert compare_module._official_model_failure_ids(
        report,
        full_ids={"case-a", "case-b"},
        label="candidate",
    ) == {"case-b"}


def _write_v2p10_lineage_contract(
    path: Path,
    *,
    composite: Path,
) -> Path:
    artifacts: dict[str, dict[str, object]] = {
        "composite": _artifact_binding(composite),
    }
    for name in (
        "marker",
        "run_manifest",
        "dataset_manifest",
        "train_jsonl",
        "adapter",
        "merge_audit",
        "stage_data_manifest",
    ):
        artifact = path.parent / f"{path.stem}_{name}.json"
        artifact.write_text("{}\n")
        artifacts[name] = _artifact_binding(artifact)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "v2p10_training_lineage",
                "status": "complete",
                "artifacts": artifacts,
                "model_contract": json.loads(
                    composite.read_text()
                )["model_contract"],
                "training": {
                    "rows": 1211,
                    "max_seq": 32768,
                    "optimizer_steps": 76,
                    "rank": 32,
                    "alpha": 32,
                    "gradient_accumulation": 16,
                },
                "fable": {
                    "historical_verified_revision_rows": 10,
                    "recent_v2p11_rows": 0,
                },
                "stage_a_base": {
                    "rows": 1211,
                    "variant": "teacher_train_mix_v2p10",
                },
            }
        )
        + "\n"
    )
    return path


def _write_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    ids = [f"case-{index:03d}" for index in range(300)]
    full_ids = root / "full_ids.json"
    full_ids.write_text(json.dumps(ids) + "\n")
    fixed = _write_panel(
        root,
        "fixed",
        ids[:150],
        resolved={"case-000"},
        empty={"case-001"},
    )
    complement = _write_panel(
        root,
        "complement",
        ids[150:],
        resolved={"case-150"},
        empty=set(),
    )
    v2p10 = root / "v2p10.json"
    v2p10_predictions = root / "v2p10_preds.json"
    join_disjoint_panels(
        full_ids_path=full_ids,
        panels=[fixed, complement],
        output_path=v2p10,
        predictions_path=v2p10_predictions,
    )

    candidate = _write_panel(
        root,
        "candidate",
        ids,
        resolved={"case-000", "case-001", "case-299"},
        empty={"case-150"},
        model_contract={
            "served_name": "teacher_sft_v2p11",
            "model_path": "/models/teacher_sft_v2p11_full",
            "model_config_sha256": "c" * 64,
            "model_index_sha256": "d" * 64,
            "model_artifacts": [],
        },
    )
    predictions = json.loads(candidate.predictions_path.read_text())
    for row in predictions.values():
        row["model_name_or_path"] = "teacher_sft_v2p11"
    candidate.predictions_path.write_text(json.dumps(predictions) + "\n")
    composite = json.loads(candidate.composite_path.read_text())
    composite["name"] = "teacher_sft_v2p11"
    composite["predictions_artifact"] = {
        "path": str(candidate.predictions_path.resolve()),
        "sha256": hashlib.sha256(
            candidate.predictions_path.read_bytes()
        ).hexdigest(),
        "bytes": candidate.predictions_path.stat().st_size,
    }
    composite["retry_ids"] = ["case-150"]
    candidate.composite_path.write_text(json.dumps(composite) + "\n")
    return (
        full_ids,
        v2p10,
        candidate.composite_path,
        candidate.predictions_path,
    )


def _write_provenance_report(
    path: Path,
    *,
    full_ids: Path,
    v2p10: Path,
    candidate_model_contract: dict,
) -> Path:
    lineage_root = path.parent / f"{path.stem}_lineage"
    lineage_root.mkdir()
    stage_marker = lineage_root / "v2p11_training_complete.json"
    stage_marker.write_text(json.dumps({
        "schema_version": 1,
        "complete": True,
        "data_manifest_sha256": "a" * 64,
        "train_jsonl_sha256": "b" * 64,
        "adapter_sha256": "f" * 64,
        "optimizer_steps": 78,
    }) + "\n")
    posttrain_marker = lineage_root / "v2p11_posttrain_complete.json"
    lineage = create_complete_phase_lineage(
        lineage_root,
        final_model=Path(candidate_model_contract["model_path"]),
        stage_marker=stage_marker,
    )
    contract_values = {
        "training": {
            "rows": 1247,
            "base_rows": 1211,
            "fable_rows": 36,
            "max_seq": 32768,
            "optimizer_steps": 78,
            "dataset_manifest_sha256": "a" * 64,
            "train_jsonl_sha256": "b" * 64,
            "v2p10_full300_sha256": hashlib.sha256(
                v2p10.read_bytes()
            ).hexdigest(),
        },
        "portable_recovery": {
            "rows": 46,
            "evaluation_overlap": 0,
            "source_counts": {
                "data/fable5_recent_verified_revision_v1": 10,
                "data/fable5_v2p11_frozen47_prepared_safe36": 36,
            },
            "combined_exclusion_sha256": "e" * 64,
            "combined_exclusion_ids": 707,
            "excluded_repositories": 12,
        },
        "posttrain": {
            "recovery_rows": 138,
            "recovery_optimizer_steps": 105,
            "behavior_rows": 606,
            "behavior_negative_counts": {
                "empty_terminal": 33,
                "repeated_read_loop": 55,
                "wrong_nonempty_replay": 228,
            },
            "full_evaluation_ids": 300,
            "combined_exclusion_ids": 707,
            "excluded_repositories": 12,
            "evaluation_overlap": 0,
            "recovery_manifest_sha256": _artifact_binding(
                lineage["recovery_manifest"]
            )["sha256"],
            "behavior_manifest_sha256": _artifact_binding(
                lineage["behavior_manifest"]
            )["sha256"],
            "exclusion_sha256": "e" * 64,
        },
        "stage_data": {
            "rows": 1247,
            "base_rows": 1211,
            "new_fable_rows": 36,
            "dataset_manifest_sha256": "a" * 64,
            "train_jsonl_sha256": "b" * 64,
            "strict_fable_manifest_sha256": "1" * 64,
            "campaign_manifest_sha256": "2" * 64,
            "fable_lineage": {
                "inherited_pinned_rows": 31,
                "historical_verified_revision_rows": 10,
                "new_strict_rows": 36,
                "total_stage_a_fable_rows": 77,
            },
            "fable_freeze": {
                "attempted": 60,
                "resolved": 47,
                "collection_rejected": 13,
                "strict_admitted": 36,
                "distillation_rejected": 11,
            },
            "evaluation_exclusion": {
                "ids": 707,
                "repositories": 12,
                "overlap": 0,
                "sha256": "e" * 64,
            },
        },
    }
    bound_artifacts = {}
    for name, value in contract_values.items():
        artifact = path.parent / f"{path.stem}_{name}.json"
        artifact.write_text(json.dumps(value) + "\n")
        bound_artifacts[name] = _artifact_binding(artifact)
    v2p10_lineage = _write_v2p10_lineage_contract(
        path.parent / f"{path.stem}_v2p10_training_lineage.json",
        composite=v2p10,
    )
    bound_artifacts["v2p10_training_lineage"] = _artifact_binding(
        v2p10_lineage
    )
    posttrain_marker.write_text(json.dumps({
        "schema_version": 1,
        "complete": True,
        "stage_a_adapter_sha256": json.loads(
            stage_marker.read_text()
        )["adapter_sha256"],
        "recovery_manifest_sha256": _artifact_binding(
            lineage["recovery_manifest"]
        )["sha256"],
        "behavior_manifest_sha256": _artifact_binding(
            lineage["behavior_manifest"]
        )["sha256"],
        "recovery_optimizer_steps": 105,
        "kto_optimizer_steps": 25,
        "recovery_input_marker_sha256": _artifact_binding(
            lineage["recovery_input"]
        )["sha256"],
        "kto_input_marker_sha256": _artifact_binding(
            lineage["kto_input"]
        )["sha256"],
        "kto_training_evidence_sha256": _artifact_binding(
            lineage["kto_evidence"]
        )["sha256"],
        "final_merge_audit_sha256": _artifact_binding(
            Path(candidate_model_contract["model_path"])
            / "v2p11_final_merge_audit.json"
        )["sha256"],
        "final_merge_marker_sha256": _artifact_binding(
            lineage["final_merge"]
        )["sha256"],
    }) + "\n")
    bound_markers = {
        "stage_a": _artifact_binding(stage_marker),
        "recovery": _artifact_binding(lineage["recovery"]),
        "kto": _artifact_binding(lineage["kto"]),
        "final_merge": _artifact_binding(lineage["final_merge"]),
        "posttrain": _artifact_binding(posttrain_marker),
    }
    criteria = {
        "candidate_empty_at_most_one": True,
        "candidate_empty_no_regression": True,
        "candidate_no_format_regression": True,
        "candidate_no_loop_regression": True,
        "candidate_resolution_floor": True,
    }
    portability_gate = path.parent / f"{path.stem}_portability.json"
    portability_gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": (
                    "v2p11_controller_free_portability_gate"
                ),
                "status": "complete",
                "passed": True,
                "control_name": "teacher_sft_v2p10",
                "candidate_name": "teacher_sft_v2p11",
                "population": 10,
                "evaluation_overlap": 0,
                "criteria": criteria,
                "control": {
                    "model_contract": {"model_path": "/control"}
                },
                "candidate": {
                    "model_contract": {
                        key: value
                        for key, value in candidate_model_contract.items()
                        if key != "served_name"
                    }
                },
            }
        )
        + "\n"
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "v2p11_completion_provenance",
                "status": "complete",
                "full_ids": {
                    "path": str(full_ids.resolve()),
                    "sha256": hashlib.sha256(
                        full_ids.read_bytes()
                    ).hexdigest(),
                    "bytes": full_ids.stat().st_size,
                },
                "v2p10_full300": {
                    "path": str(v2p10.resolve()),
                    "sha256": hashlib.sha256(
                        v2p10.read_bytes()
                    ).hexdigest(),
                    "bytes": v2p10.stat().st_size,
                },
                "contracts": bound_artifacts,
                "markers": bound_markers,
                "portability": {
                    "gate": _artifact_binding(portability_gate),
                    "passed": True,
                    "population": 10,
                    "criteria": criteria,
                },
                "fable": {
                    "stage_a": {
                        "inherited_pinned_rows": 31,
                        "historical_verified_revision_rows": 10,
                        "new_strict_rows": 36,
                        "total_rows": 77,
                    },
                    "extracted_campaign": {
                        "attempted": 60,
                        "resolved": 47,
                        "collection_rejected": 13,
                        "strict_admitted": 36,
                        "distillation_rejected": 11,
                    },
                    "recovery_unique_sources": 46,
                    "targeted_recovery_rows": 138,
                },
                "evaluation_exclusion": {
                    "full_ids": 300,
                    "excluded_ids": 707,
                    "excluded_repositories": 12,
                    "stage_a_overlap": 0,
                    "overlap": 0,
                    "sha256": "e" * 64,
                },
                "final_model": {
                    key: value
                    for key, value in candidate_model_contract.items()
                    if key != "served_name"
                },
            }
        )
        + "\n"
    )
    return path


def test_compare_full300_requires_same_panel_and_reports_paired_delta(
    tmp_path: Path,
) -> None:
    full_ids, v2p10, v2p11, v2p11_predictions = _write_fixture(
        tmp_path
    )

    report = compare_full300(
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        v2p11_composite_path=v2p11,
        v2p11_predictions_path=v2p11_predictions,
    )

    assert report["status"] == "complete"
    assert report["population"] == 300
    assert report["v2p10"]["resolved"] == 2
    assert report["v2p11"]["resolved"] == 3
    assert report["paired"]["v2p11_only"] == ["case-001", "case-299"]
    assert report["paired"]["v2p10_only"] == ["case-150"]
    assert report["paired"]["both_resolved"] == ["case-000"]
    assert len(report["paired"]["neither_resolved"]) == 296
    assert report["empty_patch"]["v2p10"] == ["case-001"]
    assert report["empty_patch"]["v2p11"] == ["case-150"]
    assert report["empty_patch"]["eliminated"] == ["case-001"]
    assert report["empty_patch"]["introduced"] == ["case-150"]
    assert report["verdict"] == {
        "beats_v2p10": True,
        "resolved_delta": 1,
        "paired_win_delta": 1,
        "behavior_healthy": True,
    }
    assert report["harness_contract"] == HARNESS_CONTRACT


def test_compare_full300_rejects_v2p11_harness_drift(
    tmp_path: Path,
) -> None:
    full_ids, v2p10, v2p11, v2p11_predictions = _write_fixture(
        tmp_path
    )
    composite = json.loads(v2p11.read_text())
    composite["harness_contract"]["seed"] = 2
    v2p11.write_text(json.dumps(composite) + "\n")

    with pytest.raises(ValueError, match="harness contract mismatch"):
        compare_full300(
            full_ids_path=full_ids,
            v2p10_composite_path=v2p10,
            v2p11_composite_path=v2p11,
            v2p11_predictions_path=v2p11_predictions,
        )


def test_compare_full300_accepts_explicit_candidate_identity(
    tmp_path: Path,
) -> None:
    full_ids, v2p10, v2p11, v2p11_predictions = _write_fixture(
        tmp_path
    )
    candidate_name = "teacher_sft_v2p11_recovery"
    candidate_model_path = "/models/teacher_sft_v2p11_recovery_base"
    predictions = json.loads(v2p11_predictions.read_text())
    for row in predictions.values():
        row["model_name_or_path"] = candidate_name
    v2p11_predictions.write_text(json.dumps(predictions) + "\n")
    composite = json.loads(v2p11.read_text())
    composite["name"] = candidate_name
    composite["model_contract"]["served_name"] = candidate_name
    composite["model_contract"]["model_path"] = candidate_model_path
    composite["predictions_artifact"] = _artifact_binding(
        v2p11_predictions
    )
    v2p11.write_text(json.dumps(composite) + "\n")

    report = compare_full300(
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        v2p11_composite_path=v2p11,
        v2p11_predictions_path=v2p11_predictions,
        candidate_name=candidate_name,
        candidate_model_path=Path(candidate_model_path),
    )

    assert report["v2p11"]["name"] == candidate_name


def test_first_pass_comparison_uses_explicit_candidate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_ids = tmp_path / "ids.json"
    full_ids.write_text(json.dumps(["case-a"]) + "\n")
    calls: list[str] = []

    def fake_validate_first_pass(
        *,
        expected_name: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append(expected_name)
        return {
            "harness_contract": HARNESS_CONTRACT,
            "model_contract": {
                "served_name": expected_name,
                "model_path": f"/models/{expected_name}",
                "model_config_sha256": "a" * 64,
                "model_index_sha256": "b" * 64,
                "model_artifacts": [],
            },
            "resolved_ids": set(),
            "empty_ids": set(),
            "panels": [],
        }

    monkeypatch.setattr(
        compare_module,
        "_validate_first_pass",
        fake_validate_first_pass,
    )

    compare_module._compare_first_pass(
        full_ids_path=full_ids,
        v2p10_panels=[],
        v2p11_panels=[],
        candidate_name="teacher_sft_v2p11r2",
    )

    assert calls == ["teacher_sft_v2p10", "teacher_sft_v2p11r2"]


def test_portability_provenance_uses_explicit_candidate_identity(
    tmp_path: Path,
) -> None:
    candidate_name = "teacher_sft_v2p11r2"
    model_contract = {
        "served_name": candidate_name,
        "model_path": "/models/teacher_sft_v2p11r2",
        "model_config_sha256": "a" * 64,
        "model_index_sha256": "b" * 64,
        "model_safetensors_sha256": None,
        "model_artifacts": [],
    }
    gate_path = tmp_path / "portability.json"
    gate_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "v2p11_controller_free_portability_gate",
        "status": "complete",
        "passed": True,
        "control_name": "teacher_sft_v2p10",
        "candidate_name": candidate_name,
        "population": 10,
        "evaluation_overlap": 0,
        "criteria": compare_module._PORTABILITY_CRITERIA,
        "control": {},
        "candidate": {"model_contract": model_contract},
    }) + "\n")
    value = {
        "gate": _artifact_binding(gate_path),
        "passed": True,
        "population": 10,
        "criteria": compare_module._PORTABILITY_CRITERIA,
    }

    validated = compare_module._validate_portability_provenance(
        value,
        candidate_model_contract=model_contract,
        candidate_name=candidate_name,
    )

    assert validated == value


def test_completion_provenance_dispatches_direct_lora_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phaseH_eval.v2p11r2_completion_provenance as r2_module

    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "artifact_type": "v2p11r2_completion_provenance",
    }) + "\n")
    captured: dict[str, object] = {}

    def fake_validate(
        path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(
        r2_module,
        "validate_completion_provenance",
        fake_validate,
    )
    full_ids = tmp_path / "ids.json"
    v2p10 = tmp_path / "v2p10.json"
    model_contract = {"served_name": "teacher_sft_v2p11r2"}

    validated = compare_module.validate_completion_provenance(
        provenance,
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        candidate_model_contract=model_contract,
        candidate_name="teacher_sft_v2p11r2",
    )

    assert validated == {"status": "complete"}
    assert captured["path"] == provenance.resolve()
    assert captured["candidate_name"] == "teacher_sft_v2p11r2"
    assert captured["candidate_model_contract"] == model_contract


def test_completion_provenance_dispatches_reasoned_direct_lora_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phaseH_eval.v2p11r3_completion_provenance as r3_module

    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "artifact_type": "v2p11r3_completion_provenance",
    }) + "\n")
    captured: dict[str, object] = {}

    def fake_validate(path: Path, **kwargs: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(r3_module, "validate_completion_provenance", fake_validate)
    full_ids = tmp_path / "ids.json"
    v2p10 = tmp_path / "v2p10.json"
    model_contract = {"served_name": "teacher_sft_v2p11r3"}

    validated = compare_module.validate_completion_provenance(
        provenance,
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        candidate_model_contract=model_contract,
        candidate_name="teacher_sft_v2p11r3",
    )

    assert validated == {"status": "complete"}
    assert captured["path"] == provenance.resolve()
    assert captured["candidate_name"] == "teacher_sft_v2p11r3"
    assert captured["candidate_model_contract"] == model_contract


def test_completion_provenance_dispatches_r3_behavior_poststage_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phaseH_eval.v2p11r3_behavior_completion_provenance as behavior_module

    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "artifact_type": "v2p11r3_behavior_completion_provenance",
    }) + "\n")
    captured: dict[str, object] = {}

    def fake_validate(path: Path, **kwargs: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {"status": "complete", "behavior_poststage": {"kto_optimizer_steps": 25}}

    monkeypatch.setattr(
        behavior_module,
        "validate_completion_provenance",
        fake_validate,
    )
    full_ids = tmp_path / "ids.json"
    v2p10 = tmp_path / "v2p10.json"
    model_contract = {"served_name": "teacher_sft_v2p11r3_behavior"}

    validated = compare_module.validate_completion_provenance(
        provenance,
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        candidate_model_contract=model_contract,
        candidate_name="teacher_sft_v2p11r3_behavior",
    )

    assert validated["behavior_poststage"]["kto_optimizer_steps"] == 25
    assert captured["path"] == provenance.resolve()
    assert captured["candidate_name"] == "teacher_sft_v2p11r3_behavior"
    assert captured["candidate_model_contract"] == model_contract


def test_verdict_publication_rolls_back_json_when_markdown_exists(
    tmp_path: Path,
) -> None:
    json_output = tmp_path / "verdict.json"
    markdown_output = tmp_path / "verdict.md"
    markdown_output.write_text("occupied\n")

    with pytest.raises(FileExistsError):
        publish_verdict_outputs(
            {"verdict": {"beats_v2p10": True}},
            output_path=json_output,
            markdown_output_path=markdown_output,
        )

    assert not json_output.exists()
    assert markdown_output.read_text() == "occupied\n"


def test_cli_publishes_losing_analysis_but_fails_trustworthy_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "verdict": {
            "beats_v2p10": False,
            "trustworthy_beats_v2p10": False,
        }
    }
    published = {}
    monkeypatch.setattr(
        compare_module,
        "compare_full300",
        lambda **_: report,
    )
    monkeypatch.setattr(
        compare_module,
        "publish_verdict_outputs",
        lambda value, **_: published.setdefault("report", value),
    )

    status = compare_module.main([
        "--full-ids",
        str(tmp_path / "ids.json"),
        "--v2p10",
        str(tmp_path / "v2p10.json"),
        "--v2p11",
        str(tmp_path / "v2p11.json"),
        "--v2p11-preds",
        str(tmp_path / "preds.json"),
        "--out",
        str(tmp_path / "verdict.json"),
        "--markdown-out",
        str(tmp_path / "verdict.md"),
        "--require-trustworthy-win",
    ])

    assert status == 2
    assert published["report"] == report


def test_cli_forwards_explicit_candidate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_compare(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"verdict": {"trustworthy_beats_v2p10": False}}

    monkeypatch.setattr(compare_module, "compare_full300", fake_compare)
    monkeypatch.setattr(
        compare_module,
        "publish_verdict_outputs",
        lambda *_args, **_kwargs: None,
    )
    candidate_model = tmp_path / "recovery"

    status = compare_module.main([
        "--full-ids",
        str(tmp_path / "ids.json"),
        "--v2p10",
        str(tmp_path / "v2p10.json"),
        "--v2p11",
        str(tmp_path / "v2p11.json"),
        "--v2p11-preds",
        str(tmp_path / "preds.json"),
        "--candidate-name",
        "teacher_sft_v2p11_recovery",
        "--candidate-model-path",
        str(candidate_model),
        "--out",
        str(tmp_path / "verdict.json"),
        "--markdown-out",
        str(tmp_path / "verdict.md"),
    ])

    assert status == 0
    assert captured["candidate_name"] == "teacher_sft_v2p11_recovery"
    assert captured["candidate_model_path"] == candidate_model


@pytest.mark.parametrize(
    (
        "control_first_empty",
        "candidate_first_empty",
        "control_corrected_empty",
        "candidate_corrected_empty",
        "expected_trustworthy",
    ),
    [
        (set(), set(), set(), set(), True),
        (set(), set(), set(), {"case-299"}, False),
        (set(), set(), {"case-298"}, {"case-299"}, False),
        ({"case-148"}, {"case-149"}, set(), set(), False),
    ],
)
def test_matched_comparison_reports_first_pass_and_retry_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_first_empty: set[str],
    candidate_first_empty: set[str],
    control_corrected_empty: set[str],
    candidate_corrected_empty: set[str],
    expected_trustworthy: bool,
) -> None:
    ids = [f"case-{index:03d}" for index in range(300)]
    full_ids = tmp_path / "full_ids.json"
    fixed_ids = tmp_path / "fixed150.json"
    complement_ids = tmp_path / "complement150.json"
    full_ids.write_text(json.dumps(ids) + "\n")
    fixed_ids.write_text(json.dumps(ids[:150]) + "\n")
    complement_ids.write_text(json.dumps(ids[150:]) + "\n")

    raw_runs: dict[str, list[FirstPassPanel]] = {}
    corrected: dict[str, tuple[Path, Path]] = {}
    for name, resolved in (
        ("teacher_sft_v2p10", {"case-000", "case-150"}),
        (
            "teacher_sft_v2p11",
            {"case-000", "case-001", "case-150"},
        ),
    ):
        first_empty = (
            candidate_first_empty
            if name == "teacher_sft_v2p11"
            else control_first_empty
        )
        corrected_empty = (
            candidate_corrected_empty
            if name == "teacher_sft_v2p11"
            else control_corrected_empty
        )
        model_panels: list[FirstPassPanel] = []
        corrected_panels = []
        for tag, panel_ids, ids_path in (
            ("fixed150", ids[:150], fixed_ids),
            ("complement150", ids[150:], complement_ids),
        ):
            run = tmp_path / f"{name}_{tag}_first"
            _write_run(
                run,
                name=name,
                patches={
                    instance_id: (
                        ""
                        if instance_id in first_empty
                        else f"diff --git a/{instance_id} b/{instance_id}\n"
                    )
                    for instance_id in panel_ids
                },
                resolved=resolved.intersection(panel_ids),
                ids_path=ids_path,
            )
            model_panels.append(
                FirstPassPanel(tag, ids_path, run)
            )
            contract = _complete_run(
                ids_path=ids_path,
                run_root=run,
                label=tag,
            )["model_contract"]
            contract = dict(contract)
            panel_empty = corrected_empty.intersection(panel_ids)
            panel = _write_panel(
                tmp_path,
                f"{name}_{tag}_corrected",
                panel_ids,
                resolved=resolved.intersection(panel_ids),
                empty=panel_empty,
                model_contract=contract,
            )
            predictions = json.loads(
                panel.predictions_path.read_text()
            )
            for row in predictions.values():
                row["model_name_or_path"] = name
            panel.predictions_path.write_text(
                json.dumps(predictions) + "\n"
            )
            composite = json.loads(
                panel.composite_path.read_text()
            )
            composite["name"] = name
            composite["run_id"] = f"{name}-{tag}-source"
            composite["retry_ids"] = sorted(panel_empty)
            composite["selected_attempts"] = {
                "source": len(panel_ids) - len(panel_empty),
                "empty_retry": len(panel_empty),
            }
            composite["empty_resampling"] = {
                "attempted": len(panel_empty),
                "became_nonempty": 0,
                "became_resolved": 0,
                "still_empty": len(panel_empty),
                "nonempty_rate": 0.0 if panel_empty else None,
                "resolved_rate": 0.0 if panel_empty else None,
            }
            composite["predictions_artifact"] = {
                "path": str(panel.predictions_path.resolve()),
                "sha256": hashlib.sha256(
                    panel.predictions_path.read_bytes()
                ).hexdigest(),
                "bytes": panel.predictions_path.stat().st_size,
            }
            panel.composite_path.write_text(
                json.dumps(composite) + "\n"
            )
            corrected_panels.append(
                PanelInput(
                    tag=tag,
                    ids_path=panel.ids_path,
                    composite_path=panel.composite_path,
                    predictions_path=panel.predictions_path,
                )
            )
        raw_runs[name] = model_panels
        out = tmp_path / f"{name}_full.json"
        preds = tmp_path / f"{name}_full_preds.json"
        join_disjoint_panels(
            full_ids_path=full_ids,
            panels=corrected_panels,
            output_path=out,
            predictions_path=preds,
        )
        corrected[name] = (out, preds)

    def fake_validate_official_score(
        _contract_path: Path,
        *,
        full_ids_path: Path,
        composite_path: Path,
        predictions_path: Path,
    ) -> dict[str, object]:
        assert full_ids_path == full_ids.resolve()
        composite = json.loads(composite_path.read_text())
        predictions = json.loads(predictions_path.read_text())
        empty = sum(
            not str(row.get("model_patch") or "").strip()
            for row in predictions.values()
        )
        return {
            "name": composite["name"],
            "final": {
                "resolved_ids": composite["resolved_ids"],
                "resolved": composite["resolved"],
                "empty": empty,
                "model_failure_ids": ["case-250"],
            },
        }

    monkeypatch.setattr(
        compare_module,
        "validate_official_score_binding",
        fake_validate_official_score,
    )
    v2p10_score_binding = tmp_path / "v2p10_score_binding.json"
    v2p11_score_binding = tmp_path / "v2p11_score_binding.json"
    v2p10_score_binding.write_text("{}\n")
    v2p11_score_binding.write_text("{}\n")
    report = compare_full300(
        full_ids_path=full_ids,
        v2p10_composite_path=corrected[
            "teacher_sft_v2p10"
        ][0],
        v2p10_predictions_path=corrected[
            "teacher_sft_v2p10"
        ][1],
        v2p11_composite_path=corrected[
            "teacher_sft_v2p11"
        ][0],
        v2p11_predictions_path=corrected[
            "teacher_sft_v2p11"
        ][1],
        v2p10_score_binding_path=v2p10_score_binding,
        v2p11_score_binding_path=v2p11_score_binding,
        v2p10_first_pass_panels=raw_runs[
            "teacher_sft_v2p10"
        ],
        v2p11_first_pass_panels=raw_runs[
            "teacher_sft_v2p11"
        ],
        provenance_path=_write_provenance_report(
            tmp_path / "provenance.json",
            full_ids=full_ids,
            v2p10=corrected["teacher_sft_v2p10"][0],
            candidate_model_contract=json.loads(
                corrected["teacher_sft_v2p11"][0].read_text()
            )["model_contract"],
        ),
    )

    assert report["first_pass"]["v2p10"]["resolved"] == 2
    assert report["first_pass"]["v2p11"]["resolved"] == 3
    assert report["retry_policy"]["v2p10"]["fixed150"][
        "allowed_retry_generations"
    ] == 2
    assert report["retry_policy"]["v2p11"]["complement150"][
        "actual_retry_generation"
    ] == 1
    assert (
        report["verdict"]["trustworthy_beats_v2p10"]
        is expected_trustworthy
    )
    assert report["provenance"]["fable"]["stage_a"]["new_strict_rows"] == 36
    assert report["failure_analysis"]["v2p11_model_failure_ids"] == [
        "case-250"
    ]
    assert report["failure_analysis"]["by_instance"]["case-250"][
        "v2p11"
    ] == ["model_failure"]
    markdown = compare_module._markdown(report)
    assert "## Matched retry policy" in markdown
    if not any(
        (
            control_first_empty,
            candidate_first_empty,
            control_corrected_empty,
            candidate_corrected_empty,
        )
    ):
        assert "| v2p10 | complement150 | 1 | 1 | 0 |" in markdown
        assert "| v2p11 | fixed150 | 2 | 1 | 0 |" in markdown
    assert "## Failure analysis" in markdown
    assert "## Bound provenance" in markdown
    assert "Official score bindings SHA-256" in markdown
    assert "36 newly strict" in markdown
    assert "Evaluation overlap: 0." in markdown


def test_matched_comparison_requires_completion_provenance(
    tmp_path: Path,
) -> None:
    ids = [f"case-{index:03d}" for index in range(300)]
    full_ids = tmp_path / "full_ids.json"
    full_ids.write_text(json.dumps(ids) + "\n")
    fixed_ids = tmp_path / "fixed.json"
    complement_ids = tmp_path / "complement.json"
    fixed_ids.write_text(json.dumps(ids[:150]) + "\n")
    complement_ids.write_text(json.dumps(ids[150:]) + "\n")
    first_pass = {}
    corrected_panels = []
    for tag, panel_ids, ids_path in (
        ("fixed150", ids[:150], fixed_ids),
        ("complement150", ids[150:], complement_ids),
    ):
        run = tmp_path / f"first-{tag}"
        _write_run(
            run,
            name="teacher_sft_v2p10",
            patches={
                instance_id: f"diff --git a/{instance_id} b/{instance_id}\n"
                for instance_id in panel_ids
            },
            resolved=set(),
            ids_path=ids_path,
        )
        first_pass[tag] = FirstPassPanel(tag, ids_path, run)
        corrected_panels.append(
            _write_panel(
                tmp_path,
                f"corrected-{tag}",
                panel_ids,
                resolved=set(),
                empty=set(),
            )
        )
    v2p10 = tmp_path / "v2p10.json"
    v2p10_preds = tmp_path / "v2p10_preds.json"
    join_disjoint_panels(
        full_ids_path=full_ids,
        panels=corrected_panels,
        output_path=v2p10,
        predictions_path=v2p10_preds,
    )

    with pytest.raises(ValueError, match="completion provenance"):
        compare_full300(
            full_ids_path=full_ids,
            v2p10_composite_path=v2p10,
            v2p10_predictions_path=v2p10_preds,
            v2p11_composite_path=v2p10,
            v2p11_predictions_path=v2p10_preds,
            v2p10_first_pass_panels=list(first_pass.values()),
            v2p11_first_pass_panels=list(first_pass.values()),
        )


def test_matched_comparison_rejects_behavior_regression_as_trustworthy(
    tmp_path: Path,
) -> None:
    ids = [f"case-{index:03d}" for index in range(300)]
    full_ids = tmp_path / "full_ids.json"
    full_ids.write_text(json.dumps(ids) + "\n")
    control_fixed = _write_panel(
        tmp_path,
        "control-fixed",
        ids[:150],
        resolved={"case-000"},
        empty=set(),
    )
    control_complement = _write_panel(
        tmp_path,
        "control-complement",
        ids[150:],
        resolved=set(),
        empty=set(),
    )
    candidate_contract = dict(
        json.loads(control_fixed.composite_path.read_text())[
            "model_contract"
        ],
        served_name="teacher_sft_v2p11",
        model_path="/models/teacher_sft_v2p11_full",
    )
    candidate_fixed = _write_panel(
        tmp_path,
        "candidate-fixed",
        ids[:150],
        resolved={"case-000", "case-001"},
        empty=set(),
        model_contract=candidate_contract,
    )
    candidate_complement = _write_panel(
        tmp_path,
        "candidate-complement",
        ids[150:],
        resolved=set(),
        empty=set(),
        model_contract=candidate_contract,
    )
    candidate_composite = json.loads(
        candidate_fixed.composite_path.read_text()
    )
    candidate_composite["name"] = "teacher_sft_v2p11"
    candidate_composite["behavior_health"]["repeat_loops"] = 1
    candidate_composite["behavior_health_by_instance"]["case-001"][
        "repeat_loops"
    ] = 1
    candidate_fixed.composite_path.write_text(
        json.dumps(candidate_composite) + "\n"
    )
    candidate_predictions = json.loads(
        candidate_fixed.predictions_path.read_text()
    )
    for row in candidate_predictions.values():
        row["model_name_or_path"] = "teacher_sft_v2p11"
    candidate_fixed.predictions_path.write_text(
        json.dumps(candidate_predictions) + "\n"
    )
    candidate_composite["predictions_artifact"] = {
        "path": str(candidate_fixed.predictions_path.resolve()),
        "sha256": hashlib.sha256(
            candidate_fixed.predictions_path.read_bytes()
        ).hexdigest(),
        "bytes": candidate_fixed.predictions_path.stat().st_size,
    }
    candidate_fixed.composite_path.write_text(
        json.dumps(candidate_composite) + "\n"
    )
    complement_composite = json.loads(
        candidate_complement.composite_path.read_text()
    )
    complement_composite["name"] = "teacher_sft_v2p11"
    candidate_complement.composite_path.write_text(
        json.dumps(complement_composite) + "\n"
    )
    complement_predictions = json.loads(
        candidate_complement.predictions_path.read_text()
    )
    for row in complement_predictions.values():
        row["model_name_or_path"] = "teacher_sft_v2p11"
    candidate_complement.predictions_path.write_text(
        json.dumps(complement_predictions) + "\n"
    )
    complement_composite["predictions_artifact"] = {
        "path": str(candidate_complement.predictions_path.resolve()),
        "sha256": hashlib.sha256(
            candidate_complement.predictions_path.read_bytes()
        ).hexdigest(),
        "bytes": candidate_complement.predictions_path.stat().st_size,
    }
    candidate_complement.composite_path.write_text(
        json.dumps(complement_composite) + "\n"
    )
    control_full = tmp_path / "control_full.json"
    control_preds = tmp_path / "control_join_preds.json"
    join_disjoint_panels(
        full_ids_path=full_ids,
        panels=[
            PanelInput(
                "fixed150",
                control_fixed.ids_path,
                control_fixed.composite_path,
                control_fixed.predictions_path,
            ),
            PanelInput(
                "complement150",
                control_complement.ids_path,
                control_complement.composite_path,
                control_complement.predictions_path,
            ),
        ],
        output_path=control_full,
        predictions_path=control_preds,
    )
    candidate_full = tmp_path / "candidate_full.json"
    candidate_preds = tmp_path / "candidate_join_preds.json"
    join_disjoint_panels(
        full_ids_path=full_ids,
        panels=[
            PanelInput(
                "fixed150",
                candidate_fixed.ids_path,
                candidate_fixed.composite_path,
                candidate_fixed.predictions_path,
            ),
            PanelInput(
                "complement150",
                candidate_complement.ids_path,
                candidate_complement.composite_path,
                candidate_complement.predictions_path,
            ),
        ],
        output_path=candidate_full,
        predictions_path=candidate_preds,
    )

    report = compare_full300(
        full_ids_path=full_ids,
        v2p10_composite_path=control_full,
        v2p10_predictions_path=control_preds,
        v2p11_composite_path=candidate_full,
        v2p11_predictions_path=candidate_preds,
    )

    assert report["behavior_regression"]["repeat_loop_delta"] == 1
    assert report["behavior_regression"]["no_repeat_loop_regression"] is False
    assert report["verdict"]["behavior_healthy"] is False


def test_completion_provenance_must_bind_the_evaluated_candidate_weights(
    tmp_path: Path,
) -> None:
    full_ids = tmp_path / "full_ids.json"
    v2p10 = tmp_path / "v2p10.json"
    provenance = tmp_path / "provenance.json"
    model = tmp_path / "candidate"
    model.mkdir()
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    shard = model / "model-00001-of-00001.safetensors"
    full_ids.write_text(
        json.dumps([f"case-{index:03d}" for index in range(300)]) + "\n"
    )
    v2p10.write_text(
        json.dumps(
            {
                "status": "complete",
                "model_contract": {
                    "served_name": "teacher_sft_v2p10",
                    "model_path": "/control",
                    "model_config_sha256": "a" * 64,
                    "model_index_sha256": "b" * 64,
                    "model_artifacts": [],
                },
            }
        )
        + "\n"
    )
    config.write_text(
        json.dumps({
            "architectures": ["Gemma4ForConditionalGeneration"]
        })
        + "\n"
    )
    shard.write_bytes(b"current weights")
    weight_map = {
        (
            f"vision.layer.{tensor_index}"
            if tensor_index < 356
            else f"text.layer.{tensor_index}"
        ): shard.name
        for tensor_index in range(1188)
    }
    index.write_text(
        json.dumps({"weight_map": weight_map}) + "\n"
    )
    model_contract = {
        "served_name": "teacher_sft_v2p11",
        "model_path": str(model),
        "model_config_sha256": hashlib.sha256(
            config.read_bytes()
        ).hexdigest(),
        "model_index_sha256": hashlib.sha256(
            index.read_bytes()
        ).hexdigest(),
        "model_artifacts": [{
            "path": str(shard.resolve()),
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "bytes": shard.stat().st_size,
        }],
    }
    _write_provenance_report(
        provenance,
        full_ids=full_ids,
        v2p10=v2p10,
        candidate_model_contract=model_contract,
    )

    validated = compare_module.validate_completion_provenance(
        provenance,
        full_ids_path=full_ids,
        v2p10_composite_path=v2p10,
        candidate_model_contract=model_contract,
    )
    assert validated["fable"]["stage_a"]["new_strict_rows"] == 36

    original = json.loads(provenance.read_text())
    training_path = Path(original["contracts"]["training"]["path"])
    original_training = training_path.read_text()
    training_path.write_text(json.dumps({"name": "training"}) + "\n")
    changed_contract = dict(original)
    changed_contract["contracts"] = dict(original["contracts"])
    changed_contract["contracts"]["training"] = _artifact_binding(
        training_path
    )
    provenance.write_text(json.dumps(changed_contract) + "\n")
    with pytest.raises(ValueError, match="stage training contract"):
        compare_module.validate_completion_provenance(
            provenance,
            full_ids_path=full_ids,
            v2p10_composite_path=v2p10,
            candidate_model_contract=model_contract,
        )
    training_path.write_text(original_training)
    provenance.write_text(json.dumps(original) + "\n")

    posttrain_marker_path = Path(
        original["markers"]["posttrain"]["path"]
    )
    original_posttrain_marker = posttrain_marker_path.read_text()
    posttrain_marker_value = json.loads(original_posttrain_marker)
    posttrain_marker_value["kto_optimizer_steps"] = 24
    posttrain_marker_path.write_text(
        json.dumps(posttrain_marker_value) + "\n"
    )
    changed_marker = dict(original)
    changed_marker["markers"] = dict(original["markers"])
    changed_marker["markers"]["posttrain"] = _artifact_binding(
        posttrain_marker_path
    )
    provenance.write_text(json.dumps(changed_marker) + "\n")
    with pytest.raises(ValueError, match="posttrain marker"):
        compare_module.validate_completion_provenance(
            provenance,
            full_ids_path=full_ids,
            v2p10_composite_path=v2p10,
            candidate_model_contract=model_contract,
        )
    posttrain_marker_path.write_text(original_posttrain_marker)
    provenance.write_text(json.dumps(original) + "\n")

    changed = dict(original)
    changed["final_model"] = dict(original["final_model"])
    changed["final_model"]["model_artifacts"] = [
        dict(row) for row in original["final_model"]["model_artifacts"]
    ]
    changed["final_model"]["model_artifacts"][0]["sha256"] = "0" * 64
    provenance.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="evaluated candidate weights"):
        compare_module.validate_completion_provenance(
            provenance,
            full_ids_path=full_ids,
            v2p10_composite_path=v2p10,
            candidate_model_contract=model_contract,
        )
    provenance.write_text(json.dumps(original) + "\n")
    portability_path = Path(original["portability"]["gate"]["path"])
    portability = json.loads(portability_path.read_text())
    portability["passed"] = False
    portability_path.write_text(json.dumps(portability) + "\n")
    with pytest.raises(ValueError, match="portability"):
        compare_module.validate_completion_provenance(
            provenance,
            full_ids_path=full_ids,
            v2p10_composite_path=v2p10,
            candidate_model_contract=model_contract,
        )
