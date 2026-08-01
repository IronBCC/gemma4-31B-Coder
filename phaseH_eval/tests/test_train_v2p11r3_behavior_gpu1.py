from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "train_v2p11r3_behavior_gpu1.sh"


def test_behavior_launcher_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_behavior_launcher_rejects_any_gpu_except_gpu1_before_preflight() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={**os.environ, "GPU_INDEX": "0", "PREFLIGHT_ONLY": "1"},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "GPU_INDEX must be 1" in result.stdout + result.stderr


def test_behavior_launcher_uses_the_r3_stage_and_unique_outputs() -> None:
    script = SCRIPT.read_text()

    assert 'GPU_INDEX="${GPU_INDEX:-1}"' in script
    assert '[[ "$GPU_INDEX" == "1" ]]' in script
    assert "teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16" in script
    assert "v2p11r3_v2p10init_fable_reasoned_training_completion.json" in script
    assert "teacher_sft_v2p11r3_behavior_recovery_bf16" in script
    assert "teacher_sft_v2p11r3_behavior_kto" in script
    assert "teacher_sft_v2p11r3_behavior_full" in script
    assert "v2p11r3_behavior_posttrain_complete.json" in script
    assert 'marker["artifact_type"] == "v2p11r3_training_completion"' in script
    assert 'marker["status"] == "complete"' in script
    assert 'marker["optimizer_steps"] == 79' in script
    assert 'marker["max_steps"] == 79' in script
    assert 'marker["adapter"]["sha256"]' in script


def test_behavior_launcher_trains_recovery_and_coverage_kto_on_gpu1() -> None:
    script = SCRIPT.read_text()

    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in script
    assert "CUDA_VISIBLE_DEVICES=0" not in script
    assert "GPU_INDEX=0" not in script
    assert "GPU0" not in script
    assert "pkill" not in script
    assert "pgrep" not in script
    assert "v2p11_posttrain_contract.py" in script
    assert "--require-production-identity" in script
    assert "--max-seq 32768" in script
    assert "checkpoint-105" in script
    assert "--max-steps 1" in script
    assert "--max-steps 25" in script
    assert "--require-behavior-coverage" in script
    assert "repeated_read_loop" in script
    assert "wrong_nonempty_replay" in script
    assert "v2p11_poststage_phase_marker.py" in script
    assert "output exists without its immutable input marker" in script
    assert "MAX_TRAINER_RESTARTS" in script
    assert "ram_watchdog.py" in script
