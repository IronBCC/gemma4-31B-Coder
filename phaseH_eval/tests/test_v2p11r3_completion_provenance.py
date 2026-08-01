from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import pytest

import phaseH_eval.v2p11r3_completion_provenance as provenance


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _reasoned_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    source_train = source / "train.jsonl"
    source_train.write_text('{"instance_id":"source"}\n', encoding="utf-8")
    _write_json(source / "manifest.json", {"schema_version": 2})
    source_context = tmp_path / "source-context.json"
    _write_json(source_context, {"source": "validated"})
    monkeypatch.setattr(provenance, "SOURCE_CONTEXT_AUDIT", source_context)
    monkeypatch.setattr(
        provenance,
        "_validate_fable_source",
        lambda dataset, context: {
            "path": str(dataset.resolve()),
            "source_ids": ["strict-one"],
            "evidence": [{"source_instance_id": "strict-one"}],
        },
    )
    stage_data = tmp_path / "stage-data"
    stage_data.mkdir()
    stage_contract = tmp_path / "stage-contract.json"
    stage_exclusions = tmp_path / "stage-exclusions.json"
    stage_campaign = tmp_path / "stage-campaign"
    stage_campaign.mkdir()
    _write_json(stage_contract, {"contract": "validated"})
    _write_json(stage_exclusions, {"exclusions": "validated"})
    _write_json(stage_campaign / "manifest.json", {"campaign": "validated"})
    monkeypatch.setattr(provenance, "STAGE_DATA", stage_data)
    monkeypatch.setattr(provenance, "STAGE_TRAINING_CONTRACT", stage_contract)
    monkeypatch.setattr(provenance, "STAGE_EXCLUSIONS", stage_exclusions)
    monkeypatch.setattr(provenance, "STAGE_CAMPAIGN", stage_campaign)
    monkeypatch.setattr(
        provenance,
        "_validate_stage_data",
        lambda **_: {"rows": 1247, "base_rows": 1211, "new_fable_rows": 36},
    )

    data = tmp_path / "reasoned"
    data.mkdir()
    rows = [
        {"instance_id": "one", "messages": []},
        {"instance_id": "two", "messages": []},
    ]
    train = data / "train.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = data / "format-gate.json"
    _write_json(report, {"failure_count": 0, "samples": 2})
    manifest = {
        "schema_version": 2,
        "complete": True,
        "dataset_variant": "teacher_train_mix_v2p11_fable_reasoned_v1",
        "source_variant": "teacher_train_mix_v2p11_fable_extension",
        "rendered": 2,
        "training_admitted": 2,
        "all_training_gates_complete": True,
        "evaluation_overlap": 0,
        "fable_rows": 92,
        "canonical_bash_tool_turns": 517,
        "reasoned_tool_turns": 468,
        "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "source": {
            "manifest": _binding(source / "manifest.json"),
            "train": _binding(source_train),
        },
        "structural_gates": {
            "all_fable_supervised_tool_calls_are_canonical_bash": True,
            "all_messages_have_boolean_loss": True,
            "evaluation_overlap": 0,
            "reasoned_tool_turns": 468,
        },
        "format_loss_gate": {
            "status": "passed_full_dataset_verification",
            "failure_count": 0,
            "samples": 2,
            "report": _binding(report),
        },
    }
    _write_json(data / "manifest.json", manifest)
    context = tmp_path / "context.json"
    _write_json(context, {
        "schema_version": 1,
        "artifact_type": "v2p11r3_dataset_context_audit",
        "status": "complete",
        "dataset": {
            "manifest_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
            "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        },
        "rows": 2,
        "max_seq": 32768,
        "max_rendered_tokens": 17,
        "over_limit_rows": 0,
    })
    monkeypatch.setattr(provenance, "EXPECTED_ROWS", 2)
    return data, context


def test_reasoned_dataset_revalidates_source_and_full_format_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, context = _reasoned_fixture(tmp_path, monkeypatch)

    validated = provenance._validate_dataset(data, context)

    assert validated["reasoned_tool_turns"] == 468
    assert validated["source_fable"]["source_ids"] == ["strict-one"]


def test_reasoned_dataset_rejects_changed_format_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, context = _reasoned_fixture(tmp_path, monkeypatch)
    report = data / "format-gate.json"
    _write_json(report, {"failure_count": 1, "samples": 2})

    with pytest.raises(ValueError, match="format-loss"):
        provenance._validate_dataset(data, context)


def _write_v2p10_lineage(
    root: Path,
    *,
    adapter: Path,
    composite: Path,
) -> Path:
    artifacts = {}
    for name in (
        "marker",
        "run_manifest",
        "dataset_manifest",
        "train_jsonl",
        "merge_audit",
        "stage_data_manifest",
    ):
        artifact = root / f"{name}.json"
        _write_json(artifact, {"artifact": name})
        artifacts[name] = _binding(artifact)
    artifacts["adapter"] = _binding(adapter)
    artifacts["composite"] = _binding(composite)
    lineage = root / "v2p10-lineage.json"
    _write_json(
        lineage,
        {
            "schema_version": 1,
            "artifact_type": "v2p10_training_lineage",
            "status": "complete",
            "artifacts": artifacts,
            "model_contract": json.loads(composite.read_text())[
                "model_contract"
            ],
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
        },
    )
    return lineage


def test_reasoned_provenance_rejects_noncanonical_v2p10_init_adapter(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical-adapter.safetensors"
    substituted = tmp_path / "substituted-adapter.safetensors"
    canonical.write_bytes(b"canonical")
    substituted.write_bytes(b"substituted")
    composite = tmp_path / "v2p10.json"
    _write_json(
        composite,
        {
            "model_contract": {
                "served_name": "teacher_sft_v2p10",
                "model_path": "/models/teacher_sft_v2p10_full",
            }
        },
    )
    lineage = _write_v2p10_lineage(
        tmp_path,
        adapter=canonical,
        composite=composite,
    )

    with pytest.raises(ValueError, match="canonical v2.10 init adapter"):
        provenance._validate_v2p10_init_lineage(
            lineage_path=lineage,
            v2p10_composite_path=composite,
            training={"init_adapter": _binding(substituted)},
        )


def test_reasoned_provenance_rejects_substituted_init_adapter_config(
    tmp_path: Path,
) -> None:
    canonical_dir = tmp_path / "canonical"
    substituted_dir = tmp_path / "substituted"
    canonical_dir.mkdir()
    substituted_dir.mkdir()
    canonical_weights = canonical_dir / "adapter_model.safetensors"
    canonical_config = canonical_dir / "adapter_config.json"
    canonical_weights.write_bytes(b"canonical")
    _write_json(canonical_config, {"r": 32, "lora_alpha": 32})
    substituted_weights = substituted_dir / "adapter_model.safetensors"
    substituted_weights.symlink_to(canonical_weights)
    substituted_config = substituted_dir / "adapter_config.json"
    _write_json(substituted_config, {"r": 16, "lora_alpha": 16})
    composite = tmp_path / "v2p10.json"
    _write_json(
        composite,
        {
            "model_contract": {
                "served_name": "teacher_sft_v2p10",
                "model_path": "/models/teacher_sft_v2p10_full",
            }
        },
    )
    lineage = _write_v2p10_lineage(
        tmp_path,
        adapter=canonical_weights,
        composite=composite,
    )
    assert _binding(substituted_weights) == _binding(canonical_weights)

    with pytest.raises(ValueError, match="canonical v2.10 init adapter"):
        provenance._validate_v2p10_init_lineage(
            lineage_path=lineage,
            v2p10_composite_path=composite,
            training={
                "init_adapter": _binding(substituted_weights),
                "init_adapter_config": _binding(substituted_config),
                "init_adapter_path": str(substituted_dir.resolve()),
            },
        )


def test_reasoned_provenance_validation_uses_external_lineage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_lineage = tmp_path / "canonical-lineage.json"
    attacker_lineage = tmp_path / "attacker-lineage.json"
    canonical_lineage.write_text("{}\n", encoding="utf-8")
    attacker_lineage.write_text("{}\n", encoding="utf-8")
    model_contract = {
        "model_path": str((tmp_path / "model").resolve()),
        "served_name": "teacher_sft_v2p11r3",
    }
    report = {
        "artifact_type": "v2p11r3_completion_provenance",
        "candidate_name": "teacher_sft_v2p11r3",
        "dataset": {},
        "training": {},
        "v2p10_training_lineage": {
            "contract": _binding(attacker_lineage),
        },
        "final_model": model_contract,
        "portability": {"gate": {}},
    }
    monkeypatch.setattr(provenance, "_read_object", lambda _path: report)

    def rebuild(**kwargs: object) -> dict[str, object]:
        rebuilt = copy.deepcopy(report)
        rebuilt["v2p10_training_lineage"] = {
            "contract": _binding(Path(kwargs["v2p10_lineage_path"])),
        }
        return rebuilt

    monkeypatch.setattr(provenance, "_build_report", rebuild)

    with pytest.raises(ValueError, match="provenance changed"):
        provenance.validate_completion_provenance(
            tmp_path / "provenance.json",
            full_ids_path=tmp_path / "ids.json",
            v2p10_composite_path=tmp_path / "v2p10.json",
            v2p10_lineage_path=canonical_lineage,
            candidate_model_contract=model_contract,
            candidate_name="teacher_sft_v2p11r3",
        )
