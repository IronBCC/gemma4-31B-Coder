from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "train_merge_v2p10_after_gate.sh"


def test_v2p10_train_contract() -> None:
    script = SCRIPT.read_text()

    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert "--data data/teacher_train_mix_v2p10" in script
    assert "--out adapters/teacher_sft_v2p10_bf16" in script
    assert "--quant" not in script
    assert "--load-4bit" not in script
    assert "all_training_gates_complete" in script
    assert "assert_training_dataset_admitted" in script
    assert 'manifest["minimum_teacher_rows"]' in script
    assert 'manifest["teacher_rows"]' in script
    assert "verified_fable_revision_plus_gpt56sol" in script
    assert 'manifest["base_rows_input"] == 1148' in script
    assert 'manifest["base_rows_quarantined"] == 6' in script
    assert 'manifest["base_rows"] == 1142' in script
    assert 'manifest["fable_revision_rows"]' in script
    assert 'manifest["gpt56sol_rows"] == 59' in script
    assert "1 <= fable_revision_rows <= 41" in script
    assert "teacher_rows == fable_revision_rows + 59" in script
    assert "manifest[\"rendered\"] == 1142 + teacher_rows" in script
    assert 'manifest["verified_revision_supervision"] is True' in script
    assert "fable5_recent_verified_revision_v1" in script
    assert "teacher:claude:claude-fable-5:verified-revision" in script
    assert '"swesmith/pydicom__pydicom.7d361b3d"' in script
    assert "fefc7d83b921a258bc36155e082bda0f4bc9ea5d1c0a2db2b77d645455b07497" in script
    assert "max_rendered_tokens <= 32768" in script
    assert "from collections.abc import Mapping" in script
    assert "isinstance(encoded, Mapping)" in script
    assert "mix_rows >= 1178" not in script
    assert "teacher_sft_v2p9" not in script
    assert "--rank 32 --alpha 32 --lr 2e-5" in script
    assert "--epochs 1 --bsz 1 --grad-accum 16" in script
    assert "--max-seq 32768 --warmup-steps 8" in script
    assert "--gradient-checkpointing bounded_unsloth" in script
    assert "target_modules" in script


def test_v2p10_train_contract_binds_process_and_artifact_identity() -> None:
    script = SCRIPT.read_text()

    assert "WAITER_PID=\"$$\"" in script
    assert 'WAITER_START_TICKS="$(awk \'{print $22}\' /proc/$$/stat)"' in script
    assert "dataset_manifest_sha256" in script
    assert "adapter_sha256" in script
    assert "merge_audit_sha256" in script
    assert "v2p10_train_merge_complete.json" in script
    assert "expected_tensors = 1188" in script
    assert "expected_vision = 356" in script
    assert "nonfinite_tensors" in script
    assert "missing_tensors" in script
    assert "PROD_PORTS=(8000 8101 8103 8104)" in script
    assert "sport = :8013" in script
    assert 'check_gpu1_idle_and_eval_port "pre-train"' in script
    assert 'check_gpu1_idle_and_eval_port "post-merge"' in script
    assert script.count("check_prod_health") == 4
    assert "pkill" not in script
    assert "pgrep" not in script


def test_v2p10_resume_is_checkpoint_bound_and_protects_host_memory() -> None:
    script = SCRIPT.read_text()

    assert 'RESUME_MODE="${RESUME_MODE:-0}"' in script
    assert 'POSTTRAIN_ONLY="${POSTTRAIN_ONLY:-0}"' in script
    assert "POSTTRAIN_ONLY must be 0 or 1" in script
    assert "posttrain-only gate: checkpoint=" in script
    assert "posttrain-only requires checkpoint 76" in script
    assert "trainer_state.json" in script
    assert "checkpoint-" in script
    assert 'TRAIN_SAVE_STEPS="1"' in script
    assert 'TRAIN_RESUME_ARGS=("--resume")' in script
    assert "TORCHINDUCTOR_COMPILE_THREADS=4" in script
    assert "UNSLOTH_COMPILE_DISABLE=0" in script
    assert "UNSLOTH_DISABLE_DOUBLE_BUFFER=1" in script
    assert "TORCHDYNAMO_DISABLE=0" in script
    assert "TORCH_COMPILE_DISABLE=0" in script
    assert '"gradient_checkpointing": "bounded_unsloth"' in script
    assert '"buffer_count": 200' in script
    assert '"pageable_buffers": 200' in script
    assert '"recycle_after_backward": True' in script
    assert '"cuda_synchronized": True' in script
    assert '"host_cache_drained": True' in script
    assert "phaseD_sft/ram_watchdog.py" in script
    assert "--min-available-gib" in script
    assert 'choom -n 750 -p "$trainer_pid"' in script
    assert 'wait "$trainer_pid"' in script
    assert 'wait "$watchdog_pid"' in script
    assert 'MAX_TRAINER_RESTARTS="${MAX_TRAINER_RESTARTS:-6}"' in script
    assert "latest_checkpoint_step" in script
    assert "train_status == 143 && watchdog_status == 3" in script
    assert "latest_step > segment_start_step" in script
    assert "restart_count < MAX_TRAINER_RESTARTS" in script
    assert "watchdog recovery accepted" in script
    assert "--group-gb 3 --max-rss-gb 12" in script
