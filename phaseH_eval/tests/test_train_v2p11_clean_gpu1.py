from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "train_v2p11_clean_gpu1.sh"


def test_clean_v2p11_restarts_from_raw_base_with_the_v2p10_recipe() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'DATA="${DATA:-data/teacher_train_mix_v2p11_fable1262}"' in script
    assert 'ADAPTER="${ADAPTER:-adapters/teacher_sft_v2p11_clean_fable51_bf16}"' in script
    assert "--base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it" in script
    assert "--init-adapter" not in script
    assert "--rank 32 --alpha 32 --lr 2e-5" in script
    assert "--epochs 1 --bsz 1 --grad-accum 16" in script
    assert "--max-seq 32768 --warmup-steps 8" in script


def test_clean_v2p11_is_gpu1_only_and_fail_closed() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert '[[ "$GPU_INDEX" == "1" ]]' in script
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid' in script
    assert "--query-gpu=index,uuid" not in script
    assert "phaseD_sft/ram_watchdog.py" in script
    assert "train_status == 0 && watchdog_status == 0" in script


def test_clean_v2p11_binds_the_frozen_dataset_and_full_context_audit() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'EXPECTED_MANIFEST_SHA256="40531f44c8d5ab1d47a179418aa1adaf1ca31d0f0265f262b15c5fd424b4758a"' in script
    assert 'EXPECTED_TRAIN_SHA256="3d131c531ca060ad2472304b95b423f292cefde94cf4538788ef52ce36b919c1"' in script
    assert "assert rows[:1211] == base_rows" in script
    assert 'assert source_counts == {"teacher:claude:claude-fable-5": 36, "fable5_verified_finalpatch": 15}' in script
    assert "assert max(lengths) <= 32768" in script
    assert '"optimizer_steps": 79' in script
