from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "run_v2p11_clean_posttrain_chain.sh"


def test_clean_chain_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr


def test_clean_chain_binds_fresh_raw_base_training_contract() -> None:
    script = SCRIPT.read_text()

    assert "teacher_train_mix_v2p11_fable1262" in script
    assert "teacher_train_mix_v2p10" in script
    assert "teacher_sft_v2p11_clean_fable51_bf16" in script
    assert '"init_adapter": None' in script
    assert '"lr": 2e-5' in script
    assert '"warmup_steps": 8' in script
    assert '"save_steps": 5' in script
    assert '"max_steps": 79' in script
    assert '"max_seq": 32768' in script
    assert "INIT_ADAPTER" not in script


def test_clean_chain_waits_on_exact_single_unit_processes() -> None:
    script = SCRIPT.read_text()

    assert "v2p11-clean-fable51-train-gpu1-v2.service" in script
    assert 'TRAIN_INVOCATION_ID="${TRAIN_INVOCATION_ID:-' in script
    assert 'TRAIN_WRAPPER_PID="${TRAIN_WRAPPER_PID:-' in script
    assert 'TRAIN_PID="${TRAIN_PID:-' in script
    assert 'WATCHDOG_PID="${WATCHDOG_PID:-' in script
    assert 'systemctl --user show "$TRAIN_UNIT"' in script
    assert 'tr \'\\0\' \' \' < "/proc/$1/cmdline"' in script
    assert 'process_command "$TRAIN_WRAPPER_PID"' in script
    assert 'process_command "$TRAIN_PID"' in script
    assert 'process_command "$WATCHDOG_PID"' in script
    assert "WATCHDOG_UNIT" not in script
    live_check = script[
        script.index("validate_live_processes() {") : script.index(
            "wait_for_training() {"
        )
    ]
    assert 'kill -0 "$TRAIN_WRAPPER_PID" 2>/dev/null || return 0' in live_check
    assert 'kill -0 "$TRAIN_PID" 2>/dev/null || trainer_command=""' in live_check
    assert 'kill -0 "$WATCHDOG_PID" 2>/dev/null || watchdog_command=""' in live_check


def test_clean_chain_captures_immutable_training_evidence() -> None:
    script = SCRIPT.read_text()

    assert "/tmp/train_v2p11_clean_fable51_gpu1.log" in script
    assert "/tmp/ram_watchdog_v2p11_clean_fable51_gpu1.log" in script
    assert 'f"_SYSTEMD_INVOCATION_ID={invocation_id}"' in script
    assert "os.O_EXCL" in script
    assert "v2p11_clean_training_completion" in script
    assert '"unit_journal": binding(unit_journal_path)' in script
    assert '"adapter_tensor_count": tensor_count' in script
    assert '"nonzero_lora_b_tensor_count": nonzero_b_count' in script
    assert '"init_adapter": None' in script
    assert '[[ ! -e "$TRAINING_JOURNAL" &&' in script
    assert "reusing immutable training evidence" not in script


def test_clean_chain_seals_live_physical_gpu1_identity() -> None:
    script = SCRIPT.read_text()

    assert 'GPU_INDEX="${GPU_INDEX:-1}"' in script
    assert '[[ "$GPU_INDEX" == "1" ]]' in script
    assert 'Path(f"/proc/{train_pid}/environ")' in script
    assert 'environment.get("CUDA_VISIBLE_DEVICES") == "1"' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid' in script
    assert 'GPU_IDENTITY_ARTIFACT_TYPE="${GPU_IDENTITY_ARTIFACT_TYPE:-v2p11_clean_gpu_training_identity}"' in script
    assert '"artifact_type": artifact_type' in script
    assert '"train_pid_on_gpu": True' in script
    assert '"gpu_identity": binding(gpu_identity_path)' in script


def test_clean_chain_runs_distinct_merge_portability_provenance_and_full300() -> None:
    script = SCRIPT.read_text()

    assert "teacher_sft_v2p11_clean_fable51_full" in script
    assert "v2p11_clean_merge_audit.json" in script
    assert "--dry-run" in script
    assert script.index("--dry-run") < script.index("--audit-architecture")
    assert "LINEAGE_MODE=direct_lora" in script
    assert "v2p11_clean_completion_provenance.py" in script
    assert 'ARTIFACT_TAG="$ARTIFACT_TAG"' in script
    assert "eval_v2p11_full300_after_merge.sh" in script
    assert 'GPU_INDEX=1' in script
    assert "GPU_INDEX=0" not in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_clean_chain_preflights_nonresumable_outputs_before_waiting() -> None:
    script = SCRIPT.read_text()
    main = script[script.index("main() {") :]
    startup = main[: main.index("wait_for_training")]

    for output in (
        "TRAINING_COMPLETION",
        "GPU_IDENTITY",
        "MERGED",
        "PORTABILITY_ROOT",
        "PORTABILITY_GATE",
        "PROVENANCE",
        "VERDICT",
        "FULL300_MARKDOWN",
    ):
        assert f'! -e "${output}"' in startup
