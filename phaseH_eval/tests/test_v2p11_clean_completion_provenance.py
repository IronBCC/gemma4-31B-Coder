from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.v2p11_clean_completion_provenance as provenance


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _dataset_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    dataset = tmp_path / "dataset"
    stage = tmp_path / "stage"
    campaign = tmp_path / "campaign"
    for path in (base, dataset, stage, campaign):
        path.mkdir()
    base_rows = [
        {"instance_id": "base-one", "source": "base", "messages": []},
        {"instance_id": "base-two", "source": "base", "messages": []},
    ]
    added = [
        {
            "instance_id": "stage-one",
            "source": "teacher:claude:claude-fable-5",
            "messages": [],
        },
        {
            "instance_id": "late-one",
            "source": "fable5_verified_finalpatch",
            "messages": [],
        },
    ]
    base_train = base / "train.jsonl"
    train = dataset / "train.jsonl"
    base_train.write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows),
        encoding="utf-8",
    )
    train.write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows + added),
        encoding="utf-8",
    )
    _write_json(base / "manifest.json", {"variant": "v2p10"})
    _write_json(dataset / "manifest.json", {"variant": "outer"})
    source_context = tmp_path / "source-context.json"
    stage_contract = tmp_path / "stage-contract.json"
    stage_exclusions = tmp_path / "stage-exclusions.json"
    _write_json(source_context, {"validated": True})
    _write_json(stage_contract, {"validated": True})
    _write_json(stage_exclusions, {"validated": True})
    _write_json(stage / "manifest.json", {"validated": True})
    _write_json(campaign / "manifest.json", {"validated": True})
    context = tmp_path / "clean-context.json"
    _write_json(
        context,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_clean_fable51_context_audit",
            "status": "complete",
            "dataset_manifest_sha256": _binding(dataset / "manifest.json")[
                "sha256"
            ],
            "train_jsonl_sha256": _binding(train)["sha256"],
            "base_train_jsonl_sha256": _binding(base_train)["sha256"],
            "rows": 4,
            "base_rows": 2,
            "new_fable_rows": 2,
            "source_counts": {
                "fable5_verified_finalpatch": 1,
                "teacher:claude:claude-fable-5": 1,
            },
            "max_seq": 32768,
            "max_rendered_tokens": 99,
            "over_limit_rows": 0,
            "optimizer_steps": 79,
            "init_adapter": None,
        },
    )
    monkeypatch.setattr(provenance, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(provenance, "EXPECTED_BASE_ROWS", 2)
    monkeypatch.setattr(provenance, "EXPECTED_STAGE_ROWS", 1)
    monkeypatch.setattr(provenance, "EXPECTED_LATE_ROWS", 1)
    monkeypatch.setattr(
        provenance,
        "EXPECTED_MANIFEST_SHA256",
        _binding(dataset / "manifest.json")["sha256"],
    )
    monkeypatch.setattr(
        provenance,
        "EXPECTED_TRAIN_SHA256",
        _binding(train)["sha256"],
    )
    monkeypatch.setattr(
        provenance,
        "EXPECTED_BASE_TRAIN_SHA256",
        _binding(base_train)["sha256"],
    )
    monkeypatch.setattr(provenance, "SOURCE_CONTEXT_AUDIT", source_context)
    monkeypatch.setattr(provenance, "STAGE_DATA", stage)
    monkeypatch.setattr(provenance, "STAGE_TRAINING_CONTRACT", stage_contract)
    monkeypatch.setattr(provenance, "STAGE_EXCLUSIONS", stage_exclusions)
    monkeypatch.setattr(provenance, "STAGE_CAMPAIGN", campaign)
    monkeypatch.setattr(
        provenance,
        "_validate_outer_dataset",
        lambda dataset_path, context_path: {
            "path": str(dataset_path.resolve()),
            "context": str(context_path.resolve()),
            "source_ids": ["late-one"],
        },
    )
    monkeypatch.setattr(
        provenance,
        "_validate_stage_data",
        lambda **_: {"rows": 3, "base_rows": 2, "new_fable_rows": 1},
    )
    return dataset, base, context


def test_clean_dataset_proves_exact_base_prefix_and_fable_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, base, context = _dataset_fixture(tmp_path, monkeypatch)

    result = provenance._validate_dataset(dataset, base, context)

    assert result["base_rows"] == 2
    assert result["new_fable_rows"] == 2
    assert result["source_counts"] == {
        "fable5_verified_finalpatch": 1,
        "teacher:claude:claude-fable-5": 1,
    }
    assert result["init_adapter"] is None


def test_clean_dataset_rejects_nonnull_init_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, base, context = _dataset_fixture(tmp_path, monkeypatch)
    audit = json.loads(context.read_text())
    audit["init_adapter"] = "/adapters/v2p10"
    _write_json(context, audit)

    with pytest.raises(ValueError, match="clean full-context audit"):
        provenance._validate_dataset(dataset, base, context)


def test_clean_dataset_rejects_changed_v2p10_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, base, context = _dataset_fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    rows[0]["instance_id"] = "substituted"
    (dataset / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        provenance,
        "EXPECTED_TRAIN_SHA256",
        _binding(dataset / "train.jsonl")["sha256"],
    )
    audit = json.loads(context.read_text())
    audit["train_jsonl_sha256"] = _binding(dataset / "train.jsonl")["sha256"]
    _write_json(context, audit)

    with pytest.raises(ValueError, match="exact v2.10 prefix"):
        provenance._validate_dataset(dataset, base, context)


def test_clean_training_rejects_adapter_initialization_and_old_lr(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    adapter = tmp_path / "adapter"
    dataset.mkdir()
    adapter.mkdir()
    run = {
        **provenance._RUN_CONTRACT,
        "base": provenance.EXPECTED_BASE_MODEL,
        "data": str(dataset),
        "out": str(adapter),
        "init_adapter": "/adapters/v2p10",
        "lr": 2e-6,
    }
    _write_json(adapter / "run_manifest.json", run)

    with pytest.raises(ValueError, match="clean training run contract"):
        provenance._validate_training(
            dataset=dataset,
            adapter=adapter,
            completion_path=tmp_path / "completion.json",
        )


def test_clean_lineage_rejects_base_data_outside_canonical_v2p10(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    _write_json(base / "manifest.json", {"variant": "substituted"})
    (base / "train.jsonl").write_text('{"instance_id":"one"}\n')
    canonical_manifest = tmp_path / "canonical-manifest.json"
    canonical_train = tmp_path / "canonical-train.jsonl"
    _write_json(canonical_manifest, {"variant": "canonical"})
    canonical_train.write_text('{"instance_id":"one"}\n')
    monkeypatch.setattr(
        provenance,
        "validate_v2p10_training_lineage_contract",
        lambda *_args, **_kwargs: {
            "artifacts": {
                "dataset_manifest": _binding(canonical_manifest),
                "train_jsonl": _binding(canonical_train),
            },
            "training": {"rows": 1211},
            "model_contract": {"served_name": "teacher_sft_v2p10"},
        },
    )

    with pytest.raises(ValueError, match="canonical v2.10 training data"):
        provenance._validate_v2p10_lineage(
            lineage_path=tmp_path / "lineage.json",
            v2p10_composite_path=tmp_path / "v2p10.json",
            base_data_path=base,
        )


def test_clean_training_identity_rejects_non_gpu1_evidence(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "gpu.json"
    _write_json(
        identity,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_clean_gpu_training_identity",
            "status": "complete",
            "train_unit": "v2p11-clean-fable51-train-gpu1-v2.service",
            "train_invocation_id": "a" * 32,
            "train_pid": 123,
            "gpu_index": 0,
            "gpu_uuid": "GPU-substituted",
            "cuda_visible_devices": "0",
            "gpu_compute_pids": [123],
            "train_pid_on_gpu": True,
            "trainer_cmdline_sha256": "b" * 64,
        },
    )

    with pytest.raises(ValueError, match="physical GPU1"):
        provenance._validate_gpu_identity(
            identity,
            train_pid=123,
            invocation_id="a" * 32,
        )


def test_clean_validation_uses_external_v2p10_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical.json"
    attacker = tmp_path / "attacker.json"
    canonical.write_text("{}\n")
    attacker.write_text("{}\n")
    contract = {
        "model_path": str((tmp_path / "model").resolve()),
        "served_name": "teacher_sft_v2p11_clean_fable51",
    }
    report = {
        "artifact_type": "v2p11_clean_completion_provenance",
        "candidate_name": "teacher_sft_v2p11_clean_fable51",
        "dataset": {
            "path": str(tmp_path / "dataset"),
            "base_data": {"path": str(tmp_path / "base")},
        },
        "training": {},
        "v2p10_training_lineage": {"contract": _binding(attacker)},
        "final_model": contract,
        "merge_audit": {},
        "portability": {"gate": {}},
    }
    monkeypatch.setattr(provenance, "_read_object", lambda _path: report)

    def rebuild(**kwargs: object) -> dict[str, object]:
        rebuilt = copy.deepcopy(report)
        rebuilt["v2p10_training_lineage"] = {
            "contract": _binding(Path(kwargs["v2p10_lineage_path"]))
        }
        return rebuilt

    monkeypatch.setattr(provenance, "_build_report", rebuild)

    with pytest.raises(ValueError, match="provenance changed"):
        provenance.validate_completion_provenance(
            tmp_path / "provenance.json",
            full_ids_path=tmp_path / "ids.json",
            v2p10_composite_path=tmp_path / "v2p10.json",
            v2p10_lineage_path=canonical,
            candidate_model_contract=contract,
            candidate_name="teacher_sft_v2p11_clean_fable51",
        )


def test_clean_validation_accepts_omitted_inactive_single_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = {
        "model_path": str((tmp_path / "model").resolve()),
        "served_name": "teacher_sft_v2p11_clean_fable51",
        "model_config_sha256": "a" * 64,
        "model_index_sha256": "b" * 64,
        "model_safetensors_sha256": None,
        "model_artifacts": [
            {
                "path": str((tmp_path / "model" / "shard.safetensors").resolve()),
                "bytes": 1,
                "sha256": "c" * 64,
            }
        ],
    }
    report = {
        "artifact_type": "v2p11_clean_completion_provenance",
        "candidate_name": "teacher_sft_v2p11_clean_fable51",
        "dataset": {
            "path": str(tmp_path / "dataset"),
            "base_data": {"path": str(tmp_path / "base")},
            "context_audit": {"path": str(tmp_path / "context.json")},
        },
        "training": {
            "adapter": {"path": str(tmp_path / "adapter" / "adapter_model.safetensors")},
            "completion": {"path": str(tmp_path / "completion.json")},
        },
        "v2p10_training_lineage": {
            "contract": {"path": str(tmp_path / "lineage.json")}
        },
        "final_model": contract,
        "merge_audit": {"path": str(tmp_path / "merge.json")},
        "portability": {"gate": {"path": str(tmp_path / "gate.json")}},
    }
    monkeypatch.setattr(provenance, "_read_object", lambda _path: report)
    monkeypatch.setattr(provenance, "_build_report", lambda **_kwargs: report)
    legacy_sharded_contract = dict(contract)
    legacy_sharded_contract.pop("model_safetensors_sha256")

    validated = provenance.validate_completion_provenance(
        tmp_path / "provenance.json",
        full_ids_path=tmp_path / "ids.json",
        v2p10_composite_path=tmp_path / "v2p10.json",
        v2p10_lineage_path=tmp_path / "lineage.json",
        candidate_model_contract=legacy_sharded_contract,
        candidate_name="teacher_sft_v2p11_clean_fable51",
    )

    assert validated == report
