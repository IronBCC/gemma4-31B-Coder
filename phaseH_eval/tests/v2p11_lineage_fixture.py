from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phaseE_rl.kto_swe_outcome import select_behavior_coverage_rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_phase_marker(
    path: Path,
    phase: str,
    artifacts: list[Path],
) -> Path:
    write_json(
        path,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_poststage_phase",
            "phase": phase,
            "status": "complete",
            "artifacts": [binding(artifact) for artifact in artifacts],
        },
    )
    return path


def _weight_map(shard_name: str) -> dict[str, str]:
    return {
        (
            f"vision.layer.{index}"
            if index < 356
            else f"text.layer.{index}"
        ): shard_name
        for index in range(1188)
    }


def _merge_audit() -> dict[str, object]:
    return {
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
    }


def _complete_model(model: Path, audit_name: str) -> list[Path]:
    model.mkdir(parents=True, exist_ok=True)
    config = model / "config.json"
    shard = model / "model-00001-of-00001.safetensors"
    index = model / "model.safetensors.index.json"
    if not config.exists():
        write_json(
            config,
            {"architectures": ["Gemma4ForConditionalGeneration"]},
        )
    if not shard.exists():
        shard.write_bytes(b"weights")
    if not index.exists():
        write_json(index, {"weight_map": _weight_map(shard.name)})
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted({model / name for name in weight_map.values()})
    assets = []
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
    ):
        asset = model / name
        write_json(asset, {"fixture": name})
        assets.append(asset)
    chat_template = model / "chat_template.jinja"
    chat_template.write_text("{{ messages }}\n")
    audit = model / audit_name
    write_json(audit, _merge_audit())
    return [
        audit,
        config,
        index,
        model / "tokenizer.json",
        model / "tokenizer_config.json",
        model / "processor_config.json",
        chat_template,
        *shards,
    ]


def create_complete_phase_lineage(
    root: Path,
    *,
    final_model: Path,
    stage_marker: Path,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stage_adapter = root / "stage_adapter"
    stage_adapter.mkdir()
    stage_weights = stage_adapter / "adapter_model.safetensors"
    stage_weights.write_bytes(b"stage")
    write_json(
        stage_adapter / "adapter_config.json",
        {"r": 32, "lora_alpha": 32},
    )
    write_json(
        stage_adapter / "run_manifest.json",
        {
            "base": "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it",
            "data": "data/teacher_train_mix_v2p11_frozen1247",
            "data_len": 1247,
            "rank": 32,
            "alpha": 32,
            "lr": 2e-5,
            "epochs": 1.0,
            "bsz": 1,
            "grad_accum": 16,
            "max_seq": 32768,
            "warmup_steps": 8,
            "logging_steps": 20,
            "save_steps": 1,
            "save_total_limit": 6,
            "load_4bit": False,
            "gradient_checkpointing": "bounded_unsloth",
            "hybrid_checkpoint_policy": None,
            "bounded_unsloth_host_buffer_policy": {
                "buffer_count": 200,
                "initial_buffer_elements": 128 * 1024,
                "pageable_buffers": 200,
                "recycle_after_backward": True,
                "cuda_synchronized": True,
                "host_cache_drained": True,
            },
            "selective_assistant_loss": True,
            "unsloth_compile_disabled": False,
            "unsloth_double_buffer_disabled": True,
            "torchdynamo_disabled": False,
            "torch_compile_disabled": False,
        },
    )
    stage_value = json.loads(stage_marker.read_text())
    stage_value["adapter_sha256"] = binding(stage_weights)["sha256"]
    stage_value["run_manifest_sha256"] = binding(
        stage_adapter / "run_manifest.json"
    )["sha256"]
    write_json(stage_marker, stage_value)
    recovery_data = root / "recovery_data"
    recovery_data.mkdir()
    (recovery_data / "train.jsonl").write_text("{}\n")
    write_json(recovery_data / "manifest.json", {"rows": 138})
    recovery_input = write_phase_marker(
        root / "v2p11_recovery_sft_inputs.json",
        "recovery_sft_inputs",
        [
            stage_marker,
            stage_weights,
            stage_adapter / "adapter_config.json",
            stage_adapter / "run_manifest.json",
            recovery_data / "train.jsonl",
            recovery_data / "manifest.json",
        ],
    )

    recovery_adapter = root / "recovery_adapter"
    recovery_adapter.mkdir()
    recovery_weights = recovery_adapter / "adapter_model.safetensors"
    recovery_weights.write_bytes(b"recovery")
    write_json(
        recovery_adapter / "adapter_config.json",
        {"r": 32, "lora_alpha": 32},
    )
    write_json(
        recovery_adapter / "run_manifest.json",
        {
            "data": "data/v2p11_portable_recovery138_targeted",
            "init_adapter": "adapters/teacher_sft_v2p11_bf16",
            "max_seq": 32768,
            "load_4bit": False,
        },
    )
    write_json(
        recovery_adapter / "trainer_state.json",
        {"global_step": 105, "max_steps": 105},
    )
    recovery_marker = write_phase_marker(
        root / "v2p11_recovery_sft_complete.json",
        "recovery_sft",
        [
            recovery_input,
            recovery_weights,
            recovery_adapter / "adapter_config.json",
            recovery_adapter / "run_manifest.json",
            recovery_adapter / "trainer_state.json",
        ],
    )

    recovery_model = root / "recovery_model"
    recovery_model_artifacts = _complete_model(
        recovery_model,
        "v2p11_recovery_merge_audit.json",
    )
    recovery_merge_marker = write_phase_marker(
        root / "v2p11_recovery_merge_complete.json",
        "recovery_merge",
        [*recovery_model_artifacts, recovery_marker],
    )

    behavior_data = root / "v2p11_behavior_kto_v2.jsonl"
    behavior_rows = []
    categories = (
        ["desirable_correct_patch"] * 290
        + ["empty_terminal"] * 33
        + ["repeated_read_loop"] * 55
        + ["wrong_nonempty_replay"] * 228
    )
    for index, category in enumerate(categories):
        behavior_rows.append(
            {
                "sample_uid": f"sample-{index:03d}",
                "label": category == "desirable_correct_patch",
                "provenance": {"punishment_category": category},
            }
        )
    behavior_data.write_text(
        "".join(json.dumps(row) + "\n" for row in behavior_rows)
    )
    behavior_manifest = root / "v2p11_behavior_kto_v2_manifest.json"
    write_json(behavior_manifest, {"rows": 606})
    kto_input = write_phase_marker(
        root / "v2p11_kto_inputs.json",
        "kto_full_inputs",
        [recovery_merge_marker, behavior_data, behavior_manifest],
    )

    selected, coverage = select_behavior_coverage_rows(behavior_rows)
    kto_adapter = root / "kto_adapter"
    kto_adapter.mkdir()
    kto_weights = kto_adapter / "adapter_model.safetensors"
    kto_weights.write_bytes(b"kto")
    write_json(
        kto_adapter / "adapter_config.json",
        {"r": 32, "lora_alpha": 32},
    )
    evidence = kto_adapter / "training_evidence.json"
    write_json(
        evidence,
        {
            "schema_version": 1,
            "artifact_type": "v2p11_kto_training_evidence",
            "status": "complete",
            "base_model_path": str(recovery_model.resolve()),
            "source_data_path": str(behavior_data.resolve()),
            "source_data_sha256": binding(behavior_data)["sha256"],
            "source_manifest_path": str(behavior_manifest.resolve()),
            "source_manifest_sha256": binding(behavior_manifest)["sha256"],
            "source_rows": 606,
            "training_rows": 50,
            "coverage_required": True,
            "coverage_counts": coverage,
            "selected_sample_uids": [
                str(row["sample_uid"]) for row in selected
            ],
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "optimizer_steps": 25,
            "seed": 0,
        },
    )
    kto_state = kto_adapter / "trainer_state.json"
    write_json(kto_state, {"global_step": 25, "max_steps": 25})
    kto_marker = write_phase_marker(
        root / "v2p11_kto_complete.json",
        "kto_full",
        [
            kto_input,
            kto_weights,
            kto_adapter / "adapter_config.json",
            evidence,
            kto_state,
        ],
    )

    final_artifacts = _complete_model(
        final_model,
        "v2p11_final_merge_audit.json",
    )
    final_merge_marker = write_phase_marker(
        root / "v2p11_final_merge_complete.json",
        "final_merge",
        [*final_artifacts, recovery_merge_marker, kto_marker],
    )
    return {
        "recovery_input": recovery_input,
        "recovery": recovery_marker,
        "recovery_manifest": recovery_data / "manifest.json",
        "recovery_merge": recovery_merge_marker,
        "behavior_manifest": behavior_manifest,
        "kto_input": kto_input,
        "kto": kto_marker,
        "final_merge": final_merge_marker,
        "kto_evidence": evidence,
    }
