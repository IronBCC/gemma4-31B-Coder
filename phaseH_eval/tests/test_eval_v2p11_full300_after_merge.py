from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "eval_v2p11_full300_after_merge.sh"


def test_v2p11_full300_driver_is_scoped_and_checksum_bound() -> None:
    script = SCRIPT.read_text()

    assert "teacher_sft_v2p11" in script
    assert "teacher_sft_v2p11_full" in script
    assert "b98fc2b1054dc8fdfcb94f083f43454fd568961a0b3dbf8c388c210b7b868e14" in script
    assert "v2p11_posttrain_complete.json" in script
    assert "v2p11_final_merge_audit.json" in script
    assert "v2p11_train_merge_complete.json" not in script
    assert "v2p10_full300_composite.json" in script
    assert "data/teacher_sft_fixed150_ids.json" in script
    assert "data/swebench_lite_complement150_ids.json" in script
    assert "compare_v2p11_full300.py" in script
    assert "full300_official_score_binding.py" in script
    assert "--v2p10-score-binding" in script
    assert "--v2p11-score-binding" in script
    assert "${ARTIFACT_TAG}_vs_v2p10_full300.json" in script


def test_v2p11_full300_driver_uses_bounded_empty_only_retry_policy() -> None:
    script = SCRIPT.read_text()

    assert "fixed150_${ARTIFACT_TAG}_primary" in script
    assert "complement150_${ARTIFACT_TAG}_primary" in script
    assert "promote-no-retry" in script
    assert 'EMPTY_TOOL="phaseH_eval/empty_retry_composite.py"' in script
    assert '"$EMPTY_TOOL" freeze' in script
    assert "freeze-composite" in script
    assert "combine-composite" in script
    assert "complement150_v2p11_empty_retry2" not in script
    assert "COMPLEMENT_SELECTED_COMPOSITE" not in script
    assert "COMPLEMENT_SELECTED_PREDICTIONS" not in script
    assert '"$COMPLEMENT_GEN1_COMPOSITE"' in script
    assert '"$COMPLEMENT_GEN1_PREDICTIONS"' in script
    assert "MODE=primary" in script
    assert 'run_primary "$retry_ids" "$retry_run" "$empty_count" 2' in script
    assert "full300_panel_composite.py" in script
    assert "v2p11_full300_primary" not in script


def test_v2p11_full300_driver_publishes_first_pass_and_corrected_evidence() -> None:
    script = SCRIPT.read_text()

    assert "--v2p10-first-pass-panel" in script
    assert "--v2p11-first-pass-panel" in script
    assert "--verify-existing" in script
    assert "--require-trustworthy-win" in script


def test_v2p11_full300_driver_preserves_machine_safety() -> None:
    script = SCRIPT.read_text()

    assert "run_v2p10_empty_diff_goal.sh" in script
    assert 'GPU_INDEX is fixed to GPU1; GPU0 is unavailable' in script
    assert "REQUIRE_PRODUCTION_HEALTH" not in script
    assert "wait_for_gpu1_idle" in script
    assert "pkill" not in script
    assert "pgrep" not in script


def test_v2p11_full300_runner_uses_bound_eval_runtime() -> None:
    runner = (ROOT / "phaseH_eval" / "run_v2p10_empty_diff_goal.sh").read_text()

    assert 'VLLM="${VLLM:-' in runner
    assert '"$EVAL_PY" phaseH_eval/retest_empties.py' in runner
    assert "\npython3 phaseH_eval/retest_empties.py" not in runner


def test_v2p11_full300_driver_retries_incomplete_resumable_panels() -> None:
    script = SCRIPT.read_text()

    assert "MAX_RUN_ATTEMPTS" in script
    assert "panel launch attempt=" in script
    assert "remains incomplete after" in script
    assert '[[ -f "$source/acceptance.json" ]] && break' not in script
    assert "archive_incomplete_acceptance" in script
    assert 'complete_run_empty_count "$ids" "$source" >/dev/null 2>&1' in script
    assert "acceptance.incomplete.attempt-" in script


def test_v2p11_full300_driver_accepts_explicit_candidate_identity() -> None:
    script = SCRIPT.read_text()

    assert 'NAME="${NAME:-teacher_sft_v2p11}"' in script
    assert (
        'MODEL="${MODEL:-/media/ironbcc/CrucialX10/models/merged/'
        'teacher_sft_v2p11_full}"'
    ) in script
    assert 'ARTIFACT_TAG="${ARTIFACT_TAG:-v2p11}"' in script
    assert "--candidate-name" in script
    assert '"$NAME"' in script
    assert "--candidate-model-path" in script
    assert '"$MODEL"' in script
