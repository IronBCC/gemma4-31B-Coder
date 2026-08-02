from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.v2p11_clean_completion_provenance as clean
import phaseH_eval.compare_v2p11_full300 as compare
import phaseH_eval.v2p11_submitfix_completion_provenance as provenance


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _tool(command: str, *, call_id: str = "call-1") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "loss": True,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _dataset_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    source = tmp_path / "source"
    dataset = tmp_path / "dataset"
    for path in (base, source, dataset):
        path.mkdir()
    base_rows = [
        {"instance_id": "base-one", "source": "base", "messages": [_tool("edit")]},
        {"instance_id": "base-two", "source": "base", "messages": [_tool("test")]},
    ]
    source_rows = [
        *base_rows,
        {
            "instance_id": "stage-one",
            "source": "teacher:claude:claude-fable-5",
            "messages": [
                _tool("edit"),
                {"role": "assistant", "content": "DONE", "loss": True},
            ],
        },
        {
            "instance_id": "late-one",
            "source": "fable5_verified_finalpatch",
            "messages": [_tool("git apply patch")],
        },
    ]
    output_rows = json.loads(json.dumps(source_rows))
    output_rows[2]["messages"][-1] = _tool(
        provenance.SUBMIT_COMMAND,
        call_id="v2p11-submit-1",
    )
    _write_rows(base / "train.jsonl", base_rows)
    _write_json(base / "manifest.json", {"variant": "v2p10"})
    _write_rows(source / "train.jsonl", source_rows)
    _write_json(source / "manifest.json", {"variant": "clean"})
    _write_rows(dataset / "train.jsonl", output_rows)
    report = dataset / "format-gate.fixture.json"
    _write_json(report, {"failure_count": 0, "samples": 4})
    manifest = {
        "schema_version": 2,
        "complete": True,
        "dataset_variant": "teacher_train_mix_v2p11_submitfix_v1",
        "selection_mode": "submit_fix",
        "rendered": 4,
        "base_rows": 2,
        "terminal_repairs": 1,
        "unchanged_late_rows": 1,
        "training_admitted": 4,
        "all_training_gates_complete": True,
        "evaluation_overlap": 0,
        "mutation_sequence_unchanged": True,
        "train_jsonl_sha256": _binding(dataset / "train.jsonl")["sha256"],
        "source": {
            "manifest": _binding(source / "manifest.json"),
            "train": _binding(source / "train.jsonl"),
        },
        "standard_native_format_loss_gate": {
            "status": "passed",
            "failure_count": 0,
            "samples": 4,
            "report": _binding(report),
        },
    }
    _write_json(dataset / "manifest.json", manifest)
    context = tmp_path / "context.json"
    _write_json(
        context,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_submitfix1262_context_audit",
            "status": "complete",
            "dataset_manifest_sha256": _binding(dataset / "manifest.json")["sha256"],
            "train_jsonl_sha256": _binding(dataset / "train.jsonl")["sha256"],
            "source_manifest_sha256": _binding(source / "manifest.json")["sha256"],
            "source_train_jsonl_sha256": _binding(source / "train.jsonl")["sha256"],
            "base_train_jsonl_sha256": _binding(base / "train.jsonl")["sha256"],
            "rows": 4,
            "base_rows": 2,
            "terminal_repairs": 1,
            "unchanged_late_rows": 1,
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
        _binding(dataset / "train.jsonl")["sha256"],
    )
    monkeypatch.setattr(
        provenance,
        "EXPECTED_BASE_TRAIN_SHA256",
        _binding(base / "train.jsonl")["sha256"],
    )
    monkeypatch.setattr(provenance, "CLEAN_CONTEXT_AUDIT", tmp_path / "clean-context.json")
    monkeypatch.setattr(
        clean,
        "_validate_dataset",
        lambda source_path, base_path, context_path: {
            "path": str(source_path.resolve()),
            "base_data": str(base_path.resolve()),
            "context": str(context_path.resolve()),
            "rows": 4,
        },
    )
    return dataset, base, context


def test_submitfix_dataset_proves_only_stage_terminal_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, base, context = _dataset_fixture(tmp_path, monkeypatch)

    result = provenance._validate_dataset(dataset, base, context)

    assert result["base_rows"] == 2
    assert result["terminal_repairs"] == 1
    assert result["unchanged_late_rows"] == 1
    assert result["selection_mode"] == "submit_fix"
    assert result["init_adapter"] is None


def test_submitfix_training_uses_its_exact_unit_and_artifact_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def validate(**kwargs: object) -> dict[str, object]:
        observed.update(
            train_unit=clean.EXPECTED_TRAIN_UNIT,
            gpu_type=clean.EXPECTED_GPU_IDENTITY_ARTIFACT_TYPE,
            completion_type=clean.EXPECTED_TRAINING_COMPLETION_ARTIFACT_TYPE,
        )
        return {"validated": kwargs}

    monkeypatch.setattr(clean, "_validate_training", validate)

    result = provenance._validate_training(
        dataset=tmp_path / "dataset",
        adapter=tmp_path / "adapter",
        completion_path=tmp_path / "completion.json",
    )

    assert result["validated"]
    assert observed == {
        "train_unit": "v2p11-submitfix1262-train-gpu1-v2.service",
        "gpu_type": "v2p11_submitfix1262_gpu_training_identity",
        "completion_type": "v2p11_submitfix1262_training_completion",
    }
    assert clean.EXPECTED_TRAIN_UNIT == "v2p11-clean-fable51-train-gpu1-v2.service"


def test_full300_comparator_dispatches_submitfix_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"validated": True}
    monkeypatch.setattr(
        compare,
        "_read_object",
        lambda _path: {"artifact_type": provenance.ARTIFACT_TYPE},
    )
    monkeypatch.setattr(
        provenance,
        "validate_completion_provenance",
        lambda *_args, **_kwargs: expected,
    )

    result = compare.validate_completion_provenance(
        tmp_path / "provenance.json",
        full_ids_path=tmp_path / "ids.json",
        v2p10_composite_path=tmp_path / "v2p10.json",
        v2p10_lineage_path=tmp_path / "lineage.json",
        candidate_model_contract={"model": "candidate"},
        candidate_name="teacher_sft_v2p11_submitfix1262",
    )

    assert result == expected
