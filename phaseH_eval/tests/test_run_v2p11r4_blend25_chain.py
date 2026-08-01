from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "run_v2p11r4_blend25_chain.sh"
FULL_EVAL = ROOT / "phaseH_eval" / "eval_v2p11_full300_after_merge.sh"


def test_chain_is_fixed_to_one_v2p10_r3_blend_and_gpu1() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "teacher_sft_v2p10_full" in script
    assert "teacher_sft_v2p11r3_behavior_full" in script
    assert "teacher_sft_v2p11r4_blend25" in script
    assert 'ANCHOR="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full"' in script
    assert 'SOURCE="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full"' in script
    assert 'OUTPUT="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r4_blend25"' in script
    assert 'ANCHOR="${ANCHOR:-' not in script
    assert 'SOURCE="${SOURCE:-' not in script
    assert 'OUTPUT="${OUTPUT:-' not in script
    assert 'ALPHA="0.25"' in script
    assert "--group-gb 3" in script
    assert "--max-rss-gb 14" in script
    assert "v2p11r4-blend25-materialize.service" in script
    assert "LINEAGE_MODE=interpolation" in script
    assert "GPU_INDEX=1" in script
    assert "PORT=8013" in script
    assert "CUDA_VISIBLE_DEVICES=0" not in script
    assert "nvidia-smi -i 0" not in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_chain_gates_full300_after_portability_and_fixed150() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    main = script[script.index("main() {") :]

    materialize = main.index("materialize_candidate")
    portability = main.index("run_portability")
    fixed150 = main.index("run_fixed150")
    gate = main.index("publish_prefull_gate")
    decision = main.index("gate_allows_full300")
    full300 = main.index("run_full300")
    assert materialize < portability < fixed150 < gate < decision < full300
    assert 'if ! gate_allows_full300; then' in main
    assert "eval_v2p11_full300_after_merge.sh" in script
    assert "fixed150_v2p11r4_blend25_primary" in script
    assert "v2p11r4_blend25_prefull_gate.json" in script


def test_full_evaluator_accepts_only_bound_interpolation_provenance() -> None:
    script = FULL_EVAL.read_text(encoding="utf-8")

    assert "posttrain|direct_lora|interpolation" in script
    assert "phaseH_eval.v2p11_interpolation_lineage" in script
    assert "validate_interpolation_lineage" in script
    assert "v2p11r4_blend25_prefull_gate.json" in script
    assert "validate_interpolation_completion_provenance" in script
    assert '--provenance "$PROVENANCE"' in script
    assert "--require-trustworthy-win" in script
