from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "train_v2p11r3_reasoned_gpu1.sh"


def test_r3_training_is_a_clean_gpu1_only_v2p10_continuation() -> None:
    script = SCRIPT.read_text()

    assert 'GPU_INDEX="${GPU_INDEX:-1}"' in script
    assert '[[ "$GPU_INDEX" == "1" ]]' in script
    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert "GPU0" not in script
    assert "8000" not in script
    assert "8013" not in script
    assert "teacher_train_mix_v2p11_fable1262_reasoned_v1" in script
    assert "teacher_sft_v2p10_bf16" in script
    assert "teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16" in script
    assert "--init-adapter" in script
    assert "--rank 32 --alpha 32 --lr 2e-6" in script
    assert "--epochs 1 --bsz 1 --grad-accum 16" in script
    assert "--max-seq 32768 --max-steps 79 --warmup-steps 4" in script
    assert "--gradient-checkpointing bounded_unsloth" in script


def test_r3_training_fails_closed_on_the_sealed_reasoned_dataset() -> None:
    script = SCRIPT.read_text()

    assert "all_training_gates_complete" in script
    assert "format_loss_gate" in script
    assert "reasoned_tool_turns" in script
    assert "468" in script
    assert "canonical_bash_tool_turns" in script
    assert "517" in script
    assert "fable_rows" in script
    assert "92" in script
    assert "v2p11r3_dataset_context_audit" in script
    assert "max_rendered_tokens" in script
    assert "32768" in script
    assert "pkill" not in script
    assert "pgrep" not in script
