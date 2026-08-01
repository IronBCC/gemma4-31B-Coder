from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "run_v2p11r3_posttrain_chain.sh"
TRAIN_SCRIPT = ROOT / "phaseH_eval" / "train_v2p11r3_reasoned_gpu1.sh"


def _shell_default(script: str, variable: str) -> str:
    match = re.search(
        rf'^{variable}="\$\{{{variable}:-([^}}]+)\}}"$',
        script,
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_r3_posttrain_chain_keeps_the_corrected_candidate_isolated() -> None:
    script = SCRIPT.read_text()

    assert "v2p11r3-reasoned-train-gpu1.service" in script
    assert "teacher_train_mix_v2p11_fable1262_reasoned_v1" in script
    assert "teacher_sft_v2p10_bf16" in script
    assert "teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16" in script
    assert "teacher_sft_v2p11r3_v2p10init_fable_reasoned" in script
    assert "v2p11r3_completion_provenance.py" in script
    assert "v2p11r3_goal_completion_audit.py" in script
    assert "LINEAGE_MODE=direct_lora" in script
    assert "GPU_INDEX=1" in script
    assert "CUDA_VISIBLE_DEVICES=0" not in script
    assert "MEMORY_WATCHDOG_GIB" in script
    assert "--group-gb 3 --max-rss-gb 12" in script
    assert "v2p11r3_merge_audit.json" in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_r3_posttrain_chain_requires_completed_training_before_merge() -> None:
    script = SCRIPT.read_text()

    assert "optimizer_step=79/79" in script
    assert "adapter_model.safetensors" in script
    assert "changed_tensor_count" in script
    assert "v2p11r3_training_completion" in script
    assert "ram_watchdog" in script
    assert "refusing to overwrite merged model" in script


def test_r3_training_and_posttrain_evidence_defaults_match() -> None:
    training = TRAIN_SCRIPT.read_text()
    posttrain = SCRIPT.read_text()

    assert _shell_default(training, "TRAIN_LOG") == _shell_default(
        posttrain, "TRAIN_LOG_SOURCE"
    )
    assert _shell_default(training, "WATCHDOG_LOG") == _shell_default(
        posttrain, "WATCHDOG_LOG_SOURCE"
    )
