from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.v2p11r2_completion_provenance as provenance_module
from phaseH_eval.empty_retry_composite import _binding
from phaseH_eval.v2p11_completion_provenance import _model_contract
from teacher_platform.generic_trace_replay import (
    assess_controls,
    build_task_contract,
    canonical_json_bytes,
    sha256_bytes,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_run(command: str, passed: bool) -> dict[str, object]:
    output = "1 passed in 0.01s\n" if passed else "1 failed in 0.01s\n"
    return {
        "command": command,
        "returncode": 0 if passed else 1,
        "strict_pass": passed,
        "output_sha256": sha256_bytes(output.encode()),
        "output_tail": output,
    }


def _phase(
    name: str,
    contract: dict[str, object],
    candidate_sha: str,
    *,
    f2p: bool,
) -> dict[str, object]:
    roles = {
        "baseline": [("mutation", contract["mutation_patch_sha256"])],
        "reference": [],
        "candidate_1": [
            ("mutation", contract["mutation_patch_sha256"]),
            ("candidate", candidate_sha),
        ],
        "candidate_2": [
            ("mutation", contract["mutation_patch_sha256"]),
            ("candidate", candidate_sha),
        ],
    }[name]
    protected = "f" * 64
    return {
        "phase": name,
        "patch_applied": True,
        "f2p_pass": f2p,
        "p2p_pass": True,
        "f2p_runs": [
            _exact_run(command, f2p)
            for command in contract["f2p_commands"]
        ],
        "p2p_runs": [
            _exact_run(command, True)
            for command in contract["p2p_commands"]
        ],
        "protected_before": protected,
        "protected_after": protected,
        "protected_stable": True,
        "patch_applications": [
            {
                "role": role,
                "patch_sha256": patch_sha,
                "returncode": 0,
                "output_sha256": "0" * 64,
                "output_tail": "",
            }
            for role, patch_sha in roles
        ],
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path | str]:
    monkeypatch.setattr(provenance_module, "EXPECTED_DATASET_ROWS", 2)
    monkeypatch.setattr(provenance_module, "EXPECTED_BASE_ROWS", 1)
    monkeypatch.setattr(provenance_module, "EXPECTED_NEW_FABLE_ROWS", 1)
    monkeypatch.setattr(provenance_module, "EXPECTED_FULL_IDS", 1)
    monkeypatch.setattr(provenance_module, "EXPECTED_V2P10_RESOLVED", 1)
    monkeypatch.setattr(
        provenance_module,
        "EXPECTED_BASE_MODEL",
        "/models/gemma-4-31B-it",
    )
    monkeypatch.setattr(
        provenance_module,
        "_max_rendered_tokens",
        lambda rows, base: 17,
        raising=False,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    source_id = "owner__repo.abcd.mutation__one"
    stream = raw / f"{source_id}.stream.jsonl"
    patch = raw / f"{source_id}.patch"
    stream.write_text('{"event":"assistant"}\n')
    patch.write_text("diff --git a/a.py b/a.py\n")
    candidate_sha = _sha(patch)
    contract = build_task_contract({
        "instance_id": source_id,
        "image_name": "fixture/image:latest",
        "patch": "mutation",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    })
    controls = {
        "baseline": _phase(
            "baseline", contract, candidate_sha, f2p=False,
        ),
        "reference": _phase(
            "reference", contract, candidate_sha, f2p=True,
        ),
        "candidate_1": _phase(
            "candidate_1", contract, candidate_sha, f2p=True,
        ),
        "candidate_2": _phase(
            "candidate_2", contract, candidate_sha, f2p=True,
        ),
    }
    assessment = assess_controls(controls, protected_patch_paths=())
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": "sha256:" + "c" * 64,
        "candidate_patch_sha256": candidate_sha,
        "candidate_patch_paths": ["a.py"],
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    result = {
        "instance_id": source_id,
        **evidence,
        "admission_evidence_sha256": sha256_bytes(
            canonical_json_bytes(evidence)
        ),
        "stream_sha256": _sha(stream),
        "patch_sha256": candidate_sha,
    }
    results = raw / "results.jsonl"
    _json(results, result)

    base = tmp_path / "base"
    base.mkdir()
    base_train = base / "train.jsonl"
    base_train.write_text('{"instance_id":"base"}\n')
    _json(base / "manifest.json", {"rendered": 1})
    delta = tmp_path / "delta.jsonl"
    delta.write_text(
        json.dumps({"source_instance_id": source_id}) + "\n"
    )
    delta_manifest = tmp_path / "delta.manifest.json"
    delta_report = tmp_path / "delta.report.json"
    _json(delta_manifest, {"rows": 1})
    _json(delta_report, {"selected": 1})

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train = dataset / "train.jsonl"
    apply_command = "git apply - <<'PATCH'\ndiff --git a/a.py b/a.py\nPATCH"
    train.write_text(
        json.dumps({"instance_id": "base", "messages": []}) + "\n"
        + json.dumps({
            "instance_id": "verified-teacher-finalpatch",
            "messages": [{
                "role": "assistant",
                "content": "",
                "loss": True,
                "tool_calls": [{
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": apply_command}),
                    },
                }],
            }],
        })
        + "\n"
    )
    format_gate = dataset / "format-gate.json"
    _json(format_gate, {"failure_count": 0, "samples": 2})
    _json(dataset / "manifest.json", {
        "schema_version": 2,
        "complete": True,
        "dataset_variant": "teacher_train_mix_v2p11_fable_extension",
        "base_rows": 1,
        "new_fable_rows": 1,
        "rendered": 2,
        "training_admitted": 2,
        "all_training_gates_complete": True,
        "evaluation_overlap": 0,
        "train_jsonl_sha256": _sha(train),
        "base": {
            "path": str(base.resolve()),
            "manifest": _binding(base / "manifest.json"),
            "train": _binding(base_train),
        },
        "new_fable": {
            "train": _binding(delta),
            "manifest": _binding(delta_manifest),
            "report": _binding(delta_report),
            "source_instance_ids": [source_id],
        },
        "new_fable_admission_evidence": [{
            "source_instance_id": source_id,
            "results": _binding(results),
            "result_line": 1,
            "result_sha256": hashlib.sha256(
                canonical_json_bytes(result)
            ).hexdigest(),
            "task_contract_sha256": result["task_contract_sha256"],
            "admission_evidence_sha256": result[
                "admission_evidence_sha256"
            ],
            "stream": _binding(stream),
            "patch": _binding(patch),
        }],
        "standard_native_format_loss_gate": {
            "status": "passed",
            "failure_count": 0,
            "samples": 2,
            "artifact": {
                "path": format_gate.name,
                "sha256": _sha(format_gate),
                "bytes": format_gate.stat().st_size,
            },
        },
    })
    context_audit = tmp_path / "context-audit.json"
    _json(context_audit, {
        "schema_version": 1,
        "artifact_type": "v2p11r2_dataset_context_audit",
        "status": "complete",
        "dataset": {
            "manifest": _binding(dataset / "manifest.json"),
            "train": _binding(train),
        },
        "rows": 2,
        "max_seq": 32768,
        "max_rendered_tokens": 17,
        "over_limit_rows": 0,
    })

    init_adapter = tmp_path / "init"
    init_adapter.mkdir()
    (init_adapter / "adapter_model.safetensors").write_bytes(b"init")
    _json(init_adapter / "adapter_config.json", {"r": 32})
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"candidate")
    _json(adapter / "adapter_config.json", {
        "r": 32,
        "lora_alpha": 32,
    })
    _json(adapter / "run_manifest.json", {
        "base": "/models/gemma-4-31B-it",
        "init_adapter": str(init_adapter.resolve()),
        "data": str(dataset.resolve()),
        "out": str(adapter.resolve()),
        "data_len": 2,
        "rank": 32,
        "alpha": 32,
        "lr": 0.000002,
        "epochs": 1.0,
        "bsz": 1,
        "grad_accum": 16,
        "max_seq": 32768,
        "max_steps": 79,
        "warmup_steps": 4,
        "logging_steps": 5,
        "save_steps": 10,
        "save_total_limit": 3,
        "load_4bit": False,
        "gradient_checkpointing": "bounded_unsloth",
        "selective_assistant_loss": True,
    })
    training_journal = tmp_path / "training-journal.log"
    training_journal.write_text(
        "[progress] event=optimizer_step optimizer_step=79/79\n"
        "[done] adapter saved\n"
    )
    watchdog_log = tmp_path / "watchdog.log"
    watchdog_log.write_text(
        "event=target_exited pid=123 timestamp=2026-07-31T00:00:00Z\n"
    )
    training_completion = tmp_path / "training-completion.json"
    _json(training_completion, {
        "schema_version": 1,
        "artifact_type": "v2p11r2_training_completion",
        "status": "complete",
        "train_pid": 123,
        "train_unit": "v2p11r2-fable51-train-gpu0.service",
        "train_invocation_id": "a" * 32,
        "watchdog_unit": "v2p11r2-fable51-watchdog-gpu0.service",
        "optimizer_steps": 79,
        "max_steps": 79,
        "finite_loss_samples": 1,
        "finite_grad_norm_samples": 1,
        "adapter_changed_from_init": True,
        "changed_tensor_count": 1,
        "adapter": _binding(adapter / "adapter_model.safetensors"),
        "init_adapter": _binding(
            init_adapter / "adapter_model.safetensors"
        ),
        "training_journal": _binding(training_journal),
        "watchdog_log": _binding(watchdog_log),
    })

    model = tmp_path / "model"
    model.mkdir()
    _json(model / "config.json", {
        "architectures": ["Gemma4ForConditionalGeneration"],
    })
    (model / "model-00001-of-00001.safetensors").write_bytes(b"model")
    _json(model / "model.safetensors.index.json", {
        "weight_map": {
            "language_model.layer": "model-00001-of-00001.safetensors",
        },
    })
    merge_audit = model / "v2p11r2_merge_audit.json"
    _json(merge_audit, {
        "schema_version": 1,
        "architecture": "Gemma4ForConditionalGeneration",
        "expected_tensors": 1188,
        "actual_tensors": 1188,
        "expected_vision": 356,
        "actual_vision": 356,
        "missing_tensors": [],
        "unexpected_tensors": [],
        "misplaced_tensors": [],
        "nonfinite_tensors": [],
        "complete": True,
    })
    candidate_name = "teacher_sft_v2p11r2"
    gate_model_contract = _model_contract(model)
    portability = tmp_path / "portability.json"
    _json(portability, {
        "schema_version": 1,
        "artifact_type": "v2p11_controller_free_portability_gate",
        "status": "complete",
        "passed": True,
        "control_name": "teacher_sft_v2p10",
        "candidate_name": candidate_name,
        "population": 10,
        "evaluation_overlap": 0,
        "criteria": provenance_module.PORTABILITY_CRITERIA,
        "control": {},
        "candidate": {"model_contract": gate_model_contract},
    })
    full_ids = tmp_path / "full_ids.json"
    _json(full_ids, ["case-a"])
    v2p10 = tmp_path / "v2p10.json"
    _json(v2p10, {
        "status": "complete",
        "name": "teacher_sft_v2p10",
        "expected": 1,
        "resolved": 1,
    })
    return {
        "candidate_name": candidate_name,
        "dataset": dataset,
        "context_audit": context_audit,
        "adapter": adapter,
        "init_adapter": init_adapter,
        "training_completion": training_completion,
        "model": model,
        "merge_audit": merge_audit,
        "portability": portability,
        "full_ids": full_ids,
        "v2p10": v2p10,
    }


def test_publishes_and_revalidates_direct_lora_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "provenance.json"

    report = provenance_module.publish_completion_provenance(
        candidate_name=str(fixture["candidate_name"]),
        dataset_path=Path(fixture["dataset"]),
        dataset_context_audit_path=Path(fixture["context_audit"]),
        adapter_path=Path(fixture["adapter"]),
        init_adapter_path=Path(fixture["init_adapter"]),
        training_completion_path=Path(fixture["training_completion"]),
        final_model_path=Path(fixture["model"]),
        merge_audit_path=Path(fixture["merge_audit"]),
        portability_gate_path=Path(fixture["portability"]),
        full_ids_path=Path(fixture["full_ids"]),
        v2p10_composite_path=Path(fixture["v2p10"]),
        output_path=output,
    )

    assert report["status"] == "complete"
    assert report["training"]["rows"] == 2
    assert report["dataset"]["max_rendered_tokens"] == 17
    assert report["dataset"]["context_audit"]["sha256"] == _sha(
        Path(fixture["context_audit"])
    )
    assert report["recent_fable"]["strict_rows"] == 1
    assert report["recent_fable"]["training_targets"][0]["exact_patch"] is True
    assert provenance_module.validate_completion_provenance(
        output,
        full_ids_path=Path(fixture["full_ids"]),
        v2p10_composite_path=Path(fixture["v2p10"]),
        candidate_model_contract=report["final_model"],
        candidate_name=str(fixture["candidate_name"]),
    ) == report


def test_dataset_rejects_late_fable_target_not_equal_to_admitted_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    dataset = Path(fixture["dataset"])
    train = dataset / "train.jsonl"
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    rows[1]["messages"][0]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "command": "git diff",
    })
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train_jsonl_sha256"] = _sha(train)
    _json(manifest_path, manifest)

    with pytest.raises(ValueError, match="does not reproduce raw patch"):
        provenance_module._validate_dataset(
            dataset,
            Path(fixture["context_audit"]),
        )


def test_revalidation_rejects_changed_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "provenance.json"
    report = provenance_module.publish_completion_provenance(
        candidate_name=str(fixture["candidate_name"]),
        dataset_path=Path(fixture["dataset"]),
        dataset_context_audit_path=Path(fixture["context_audit"]),
        adapter_path=Path(fixture["adapter"]),
        init_adapter_path=Path(fixture["init_adapter"]),
        training_completion_path=Path(fixture["training_completion"]),
        final_model_path=Path(fixture["model"]),
        merge_audit_path=Path(fixture["merge_audit"]),
        portability_gate_path=Path(fixture["portability"]),
        full_ids_path=Path(fixture["full_ids"]),
        v2p10_composite_path=Path(fixture["v2p10"]),
        output_path=output,
    )
    assert report["status"] == "complete"
    (
        Path(fixture["adapter"]) / "adapter_model.safetensors"
    ).write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed"):
        provenance_module.validate_completion_provenance(
            output,
            full_ids_path=Path(fixture["full_ids"]),
            v2p10_composite_path=Path(fixture["v2p10"]),
            candidate_model_contract=report["final_model"],
            candidate_name=str(fixture["candidate_name"]),
        )
