from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseH_eval.eval_v2p10_promotion import (
    _main,
    exact_repeated_failure_loops,
    hard30_decision,
    promotion_decision,
    summarize_run_health,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "eval_v2p10_promotion.sh"
WAITER_SCRIPT = ROOT / "phaseH_eval" / "eval_v2p10_after_train.sh"


def _fixed(
    *,
    resolved: int,
    empty: int,
    repeat_loops: int,
    pull_failed: int = 0,
) -> dict:
    return {
        "status": "complete",
        "expected": 150,
        "preds": 150,
        "traj_files": 150,
        "usable_outcomes": 150,
        "resolved": resolved,
        "empty": empty,
        "repeat_loops": repeat_loops,
        "pull_failed": pull_failed,
        "docker_failed": 0,
    }


def _hard(
    *,
    resolved: int,
    nonempty: int = 20,
    format_error_rate: float = 0.0,
    repeat_loops: int = 0,
) -> dict:
    return {
        "schema_version": 2,
        "status": "complete",
        "expected": 30,
        "preds": 30,
        "traj_files": 30,
        "usable_outcomes": 30,
        "resolved": resolved,
        "nonempty": nonempty,
        "format_error_rate": format_error_rate,
        "repeat_loops": repeat_loops,
        "infra_failed": 0,
        "tasks_sha256": "a" * 64,
        "resolution_contract": (
            "clean F2P passes; mutation reproduces exact F2P failure; "
            "candidate passes all frozen F2P IDs; P2P not checked"
        ),
    }


def test_full300_is_blocked_when_fixed150_regresses() -> None:
    result = promotion_decision(
        candidate=_fixed(resolved=81, empty=4, repeat_loops=0),
        best_incumbent=_fixed(resolved=82, empty=1, repeat_loops=0),
        v2p9=_fixed(resolved=68, empty=25, repeat_loops=1),
        paired={"wins": 13, "losses": 8, "ties": 129},
    )

    assert result == {
        "run_full300": False,
        "reason": "fixed150 regression",
    }


def test_full300_requires_no_wrong_edit_health_regression() -> None:
    result = promotion_decision(
        candidate=_fixed(resolved=83, empty=20, repeat_loops=2),
        best_incumbent=_fixed(resolved=82, empty=1, repeat_loops=0),
        v2p9=_fixed(resolved=68, empty=25, repeat_loops=1),
        paired={"wins": 12, "losses": 11, "ties": 127},
    )

    assert result == {
        "run_full300": False,
        "reason": "repeated-failure-loop regression",
    }


def test_full300_requires_complete_equal_treatment_and_paired_nonregression() -> None:
    candidate = _fixed(resolved=83, empty=20, repeat_loops=0)
    candidate["preds"] = 149
    assert promotion_decision(
        candidate=candidate,
        best_incumbent=_fixed(resolved=82, empty=1, repeat_loops=0),
        v2p9=_fixed(resolved=68, empty=25, repeat_loops=1),
        paired={"wins": 12, "losses": 11, "ties": 127},
    )["reason"] == "fixed150 incomplete"

    assert promotion_decision(
        candidate=_fixed(resolved=83, empty=20, repeat_loops=0),
        best_incumbent=_fixed(resolved=82, empty=1, repeat_loops=0),
        v2p9=_fixed(resolved=68, empty=25, repeat_loops=1),
        paired={"wins": 10, "losses": 11, "ties": 129},
    )["reason"] == "paired regression versus v2.9"


def test_full300_gate_passes_without_selecting_a_winner() -> None:
    result = promotion_decision(
        candidate=_fixed(resolved=83, empty=20, repeat_loops=0),
        best_incumbent=_fixed(resolved=82, empty=1, repeat_loops=0),
        v2p9=_fixed(resolved=68, empty=25, repeat_loops=1),
        paired={"wins": 12, "losses": 11, "ties": 127},
    )

    assert result == {
        "run_full300": True,
        "reason": "fixed150 gate passed",
        "no_adapter_selected": True,
    }


def test_nonlite_hard30_requires_real_behavior_and_v2p8_control() -> None:
    legacy = _hard(resolved=12)
    legacy["schema_version"] = 1
    assert hard30_decision(
        candidate=legacy,
        v2p8_control=_hard(resolved=11),
    )["reason"] == "non-Lite hard30 incomplete"
    assert hard30_decision(
        candidate=_hard(resolved=12, nonempty=17),
        v2p8_control=_hard(resolved=11),
    )["reason"] == "nonempty patch floor"
    assert hard30_decision(
        candidate=_hard(resolved=12, format_error_rate=0.1),
        v2p8_control=_hard(resolved=11),
    )["reason"] == "format-error ceiling"
    assert hard30_decision(
        candidate=_hard(resolved=12, repeat_loops=1),
        v2p8_control=_hard(resolved=11),
    )["reason"] == "repeated-failure loop"
    assert hard30_decision(
        candidate=_hard(resolved=10),
        v2p8_control=_hard(resolved=11),
    )["reason"] == "v2.8 hard30 regression"
    assert hard30_decision(
        candidate=_hard(resolved=12),
        v2p8_control=_hard(resolved=11),
    ) == {
        "run_fixed150": True,
        "reason": "non-Lite hard30 gate passed",
        "no_adapter_selected": True,
    }


def test_exact_repeated_failure_loop_requires_three_adjacent_failed_calls() -> None:
    messages = []
    for index, returncode in enumerate((1, 1, 1)):
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "sed -i s/a/b/ x.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "content": f"<returncode>{returncode}</returncode>\nfailed",
            },
        ])

    assert exact_repeated_failure_loops({"messages": messages}) == 1

    messages[-1]["content"] = "<returncode>0</returncode>\nok"
    assert exact_repeated_failure_loops({"messages": messages}) == 0


def test_health_summary_binds_every_unique_trajectory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    trajectory = run / "b00" / "case" / "case.traj.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(json.dumps({
        "instance_id": "case",
        "messages": [],
    }))
    (run / "preds_all.json").write_text(json.dumps({
        "case": {"instance_id": "case", "model_patch": ""},
    }))
    (run / "acceptance.json").write_text(json.dumps({
        "status": "complete",
        "expected": 1,
        "preds": 1,
        "traj_files": 1,
        "usable_outcomes": 1,
        "resolved": 0,
        "pull_failed": 0,
        "docker_failed": 0,
        "empty_split": {"total": 1},
    }))

    metrics = summarize_run_health(run)

    assert {Path(row["path"]).name for row in metrics["artifact_bindings"]} == {
        "acceptance.json",
        "preds_all.json",
        "case.traj.json",
    }

    duplicate = run / "b01" / "case" / "retry.traj.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(trajectory.read_text())
    with pytest.raises(ValueError, match="duplicate trajectory"):
        summarize_run_health(run)


def test_paired_artifact_binds_both_acceptance_inputs(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    reference = tmp_path / "reference.json"
    output = tmp_path / "paired.json"
    candidate.write_text(json.dumps({
        **_fixed(resolved=1, empty=149, repeat_loops=0),
        "resolved_ids": ["case"],
    }))
    reference.write_text(json.dumps({
        **_fixed(resolved=0, empty=150, repeat_loops=0),
        "resolved_ids": [],
    }))

    assert _main([
        "pair",
        "--candidate",
        str(candidate),
        "--reference",
        str(reference),
        "--out",
        str(output),
    ]) == 0

    paired = json.loads(output.read_text())
    assert paired["complete"] is True
    assert paired["wins"] == 1
    assert {row["path"] for row in paired["input_artifacts"]} == {
        str(candidate.resolve()),
        str(reference.resolve()),
    }


def test_promotion_driver_is_ordered_guarded_and_never_auto_selects() -> None:
    script = SCRIPT.read_text()

    assert "runs/v2p10_train_merge_complete.json" in script
    assert 'export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert 'EVAL_PY="$ROOT/.venv-eval/bin/python"' in script
    assert '"$EVAL_PY" phaseH_eval/score_nonlite_hard30.py' in script
    assert '"$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py' in script
    assert ".venv/bin/python" not in script
    assert "verify_train_marker()" in script
    assert "dataset_manifest_sha256" in script
    assert "adapter_sha256" in script
    assert "merge_audit_sha256" in script
    assert "teacher_sft_v2p10_full" in script
    assert "teacher_sft_v2p8_full" in script
    assert "data/v2p10_nonlite_hard30_v2/tasks.jsonl" in script
    assert "reuse_nonlite_generation.py" in script
    assert "nonlite_hard30_v2p8_retry1" in script
    assert "nonlite_hard30_v2p10_retry1" in script
    assert "--exclude data/fable5_batch60.jsonl" in script
    assert "--exclude data/opus5_batch50.jsonl" in script
    assert "score_nonlite_hard30.py" in script
    assert "decide-hard30" in script
    assert "fixed150_v2p10" in script
    assert "runs/fixed150_base" in script
    assert "runs/fixed150_v2_bf16" in script
    assert "runs/fixed150_v2p8" in script
    assert "runs/fixed150_v2p9" in script
    assert "fixed150_v2p10_vs_base.json" in script
    assert "fixed150_v2p10_vs_v2_bf16.json" in script
    assert "fixed150_v2p10_vs_v2p8.json" in script
    assert "fixed150_v2p10_vs_v2p9.json" in script
    assert "decide-fixed150" in script
    assert "v2p10_full300" in script
    assert "audit_v2p9_failure_modes.py" in script
    assert "abc550841d4a64740da2a9f18d717973a501c4bf138755d3d55883c67d9c1390" in script
    assert "b98fc2b1054dc8fdfcb94f083f43454fd568961a0b3dbf8c388c210b7b868e14" in script
    assert script.index("decide-hard30") < script.index("fixed150_v2p10")
    assert script.index("decide-fixed150") < script.index("v2p10_full300")
    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert 'owned_serve_pid="$!"' in script
    assert 'kill "$owned_serve_pid"' in script
    assert "PROD_PORTS=(8000 8101 8103 8104)" in script
    assert "pkill" not in script
    assert "pgrep" not in script
    assert "no_adapter_selected" in script


def test_post_train_waiter_binds_exact_train_identity_and_marker() -> None:
    script = WAITER_SCRIPT.read_text()

    assert "TRAIN_PID" in script
    assert "TRAIN_START_TICKS" in script
    assert '"/proc/$TRAIN_PID/stat"' in script
    assert "phaseH_eval/train_merge_v2p10_after_gate.sh" in script
    assert "runs/v2p10_train_merge_complete.json" in script
    assert 'marker["waiter_pid"] == int(sys.argv[2])' in script
    assert 'marker["waiter_start_ticks"] == sys.argv[3]' in script
    assert "exec env LOG=" in script
    assert "phaseH_eval/eval_v2p10_promotion.sh" in script
    assert "pkill" not in script
    assert "pgrep" not in script
