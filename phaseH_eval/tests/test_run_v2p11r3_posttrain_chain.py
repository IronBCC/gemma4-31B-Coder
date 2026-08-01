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
    assert "train_v2p11r3_behavior_gpu1.sh" in script
    assert "teacher_sft_v2p11r3_behavior_full" in script
    assert "v2p11r3_behavior_completion_provenance.py" in script
    assert "v2p11r3_goal_completion_audit.py" in script
    assert "LINEAGE_MODE=posttrain" in script
    assert "v2p11r3_behavior_posttrain_complete.json" in script
    assert "GPU_INDEX=1" in script
    assert "CUDA_VISIBLE_DEVICES=0" not in script
    assert "MEMORY_WATCHDOG_GIB" in script
    assert "v2p11_final_merge_audit.json" in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_r3_posttrain_chain_requires_completed_training_before_merge() -> None:
    script = SCRIPT.read_text()

    assert "optimizer_step=79/79" in script
    assert "adapter_model.safetensors" in script
    assert "changed_tensor_count" in script
    assert "v2p11r3_training_completion" in script
    assert "ram_watchdog" in script
    assert "behavior poststage ended without completion marker" in script


def test_r3_posttrain_chain_evaluates_only_the_final_behavior_model() -> None:
    script = SCRIPT.read_text()
    main = script[script.index("main() {") :]

    training_evidence = main.index("ensure_training_evidence")
    behavior_poststage = main.index("run_behavior_poststage")
    portability = main.index("run_portability")
    provenance = main.index("publish_provenance")
    full300 = main.index("run_full300")
    assert training_evidence < behavior_poststage < portability < provenance < full300
    assert script.count("eval_v2p11_full300_after_merge.sh") == 1
    assert "merge_model" not in script
    assert "LINEAGE_MODE=direct_lora" not in script
    assert "v2p11r3_merge_audit.json" not in script


def test_r3_training_and_posttrain_evidence_defaults_match() -> None:
    training = TRAIN_SCRIPT.read_text()
    posttrain = SCRIPT.read_text()

    assert _shell_default(training, "TRAIN_LOG") == _shell_default(
        posttrain, "TRAIN_LOG_SOURCE"
    )
    assert _shell_default(training, "WATCHDOG_LOG") == _shell_default(
        posttrain, "WATCHDOG_LOG_SOURCE"
    )


def test_r3_posttrain_chain_reuses_verified_partial_progress() -> None:
    script = SCRIPT.read_text()

    assert "ensure_training_evidence" in script
    assert "validate_training_evidence" in script
    assert "reusing verified r3 training completion" in script
    assert "reusing verified behavior portability gate" in script
    assert "r3 training evidence already exists" not in script
    assert "r3 final-evaluation output already exists" not in script
    assert 'if output.exists():' in script
    assert 'assert output.read_bytes() == payload' in script
    assert "_publish_bytes_noreplace" in script
    assert "_publish_json_noreplace" in script
    assert "evaluate_portability" in script
    assert "published_gate" in script
    assert "current_gate" in script
    assert "existing portability gate differs from current run artifacts" in script
    assert (
        'if [[ -e "$TRAINING_JOURNAL" && -e "$WATCHDOG_EVIDENCE" ]]; then\n'
        "    return\n"
        "  fi"
    ) in script
