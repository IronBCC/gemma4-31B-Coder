from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "train_v2p11_successor_gpu1.sh"


def test_submitfix_successor_restarts_from_raw_base_with_v2p10_recipe() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'DATA="${DATA:-data/teacher_train_mix_v2p11_submitfix1262}"' in script
    assert 'ADAPTER="${ADAPTER:-adapters/teacher_sft_v2p11_submitfix1262_bf16}"' in script
    assert "--base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it" in script
    assert "--init-adapter" not in script
    assert "--rank 32 --alpha 32 --lr 2e-5" in script
    assert "--epochs 1 --bsz 1 --grad-accum 16" in script
    assert "--max-seq 32768 --warmup-steps 8" in script
    assert "--max-steps \"$EXPECTED_STEPS\"" in script


def test_submitfix_successor_is_gpu1_only_and_fail_closed() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert '[[ "$GPU_INDEX" == "1" ]]' in script
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid' in script
    assert "--query-gpu=index,uuid" not in script
    assert "phaseD_sft/ram_watchdog.py" in script
    assert "train_status == 0 && watchdog_status == 0" in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_submitfix_successor_binds_sealed_dataset_and_full_context() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'EXPECTED_MANIFEST_SHA256="36a7f3b13ad1dbd89fa5a02af92cdfe9f8581cedf54ea984b73d5749ec52c223"' in script
    assert 'EXPECTED_TRAIN_SHA256="04530c4a9237c3af88dcfe1bcce72a3de8ad3982ad2965f96ef436c0c87f254a"' in script
    assert 'assert manifest["dataset_variant"] == "teacher_train_mix_v2p11_submitfix_v1"' in script
    assert 'assert manifest["selection_mode"] == "submit_fix"' in script
    assert "assert rows[:1211] == base_rows" in script
    assert 'assert manifest["terminal_repairs"] == 36' in script
    assert 'assert manifest["unchanged_late_rows"] == 15' in script
    assert "assert max(lengths) <= 32768" in script
    assert '"optimizer_steps": 79' in script


def test_submitfix_successor_requires_exact_terminal_submit_contract() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in script
    assert 'assert function.get("name") == "bash"' in script
    assert 'assert arguments == {"command": submit_command}' in script
    assert "assert rows[1247:] == source_rows[1247:]" in script
    assert "assert_training_dataset_admitted(str(data))" in script
