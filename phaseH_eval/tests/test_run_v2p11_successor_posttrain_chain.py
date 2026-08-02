from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "run_v2p11_successor_posttrain_chain.sh"
ENGINE = ROOT / "phaseH_eval" / "run_v2p11_clean_posttrain_chain.sh"


def _script() -> str:
    return SCRIPT.read_text() + "\n" + ENGINE.read_text()


def test_submitfix_chain_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr


def test_submitfix_chain_binds_fresh_raw_base_training_contract() -> None:
    script = _script()

    assert "teacher_train_mix_v2p11_submitfix1262" in script
    assert "teacher_train_mix_v2p10" in script
    assert "teacher_sft_v2p11_submitfix1262_bf16" in script
    assert '"init_adapter": None' in script
    assert '"lr": 2e-5' in script
    assert '"warmup_steps": 8' in script
    assert '"max_steps": 79' in script
    assert '"max_seq": 32768' in script
    assert "INIT_ADAPTER" not in script
    assert "recovery" not in script.lower()
    assert "kto" not in script.lower()
    assert "interpolation" not in script.lower()


def test_submitfix_chain_waits_on_exact_single_unit_processes() -> None:
    script = _script()

    assert "v2p11-submitfix1262-train-gpu1-v2.service" in script
    assert 'TRAIN_INVOCATION_ID="${TRAIN_INVOCATION_ID:?' in script
    assert 'TRAIN_WRAPPER_PID="${TRAIN_WRAPPER_PID:?' in script
    assert 'TRAIN_PID="${TRAIN_PID:?' in script
    assert 'WATCHDOG_PID="${WATCHDOG_PID:?' in script
    assert 'systemctl --user show "$TRAIN_UNIT"' in script
    assert 'tr \'\\0\' \' \' < "/proc/$1/cmdline"' in script
    assert 'process_command "$TRAIN_WRAPPER_PID"' in script
    assert 'process_command "$TRAIN_PID"' in script
    assert 'process_command "$WATCHDOG_PID"' in script
    assert "WATCHDOG_UNIT" not in script


def test_submitfix_chain_captures_immutable_training_and_gpu1_evidence() -> None:
    script = _script()

    assert "/tmp/train_v2p11_submitfix1262_gpu1.log" in script
    assert "/tmp/ram_watchdog_v2p11_submitfix1262_gpu1.log" in script
    assert 'f"_SYSTEMD_INVOCATION_ID={invocation_id}"' in script
    assert "os.O_EXCL" in script
    assert "v2p11_submitfix1262_training_completion" in script
    assert 'Path(f"/proc/{train_pid}/environ")' in script
    assert 'environment.get("CUDA_VISIBLE_DEVICES") == "1"' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid' in script
    assert 'GPU_IDENTITY_ARTIFACT_TYPE="v2p11_submitfix1262_gpu_training_identity"' in script
    assert '"artifact_type": artifact_type' in script
    assert '"train_pid_on_gpu": True' in script


def test_submitfix_chain_runs_merge_portability_provenance_and_full300() -> None:
    script = _script()

    assert "teacher_sft_v2p11_submitfix1262_full" in script
    assert "v2p11_submitfix1262_merge_audit.json" in script
    assert "--dry-run" in script
    assert script.index("--dry-run") < script.index("--audit-architecture")
    assert "LINEAGE_MODE=direct_lora" in script
    assert "v2p11_submitfix_completion_provenance.py" in script
    assert "eval_v2p11_full300_after_merge.sh" in script
    assert 'GPU_INDEX=1' in script
    assert "GPU_INDEX=0" not in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_submitfix_chain_preflights_nonresumable_outputs_before_waiting() -> None:
    script = _script()
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
