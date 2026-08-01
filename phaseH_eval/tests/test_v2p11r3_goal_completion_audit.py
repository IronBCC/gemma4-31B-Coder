import pytest

from phaseH_eval.v2p11r3_goal_completion_audit import _validate_reasoned_lineage


def _provenance() -> dict[str, object]:
    return {
        "training": {"rows": 1262, "max_seq": 32768, "optimizer_steps": 79},
        "dataset": {
            "max_rendered_tokens": 26594,
            "fable_rows": 92,
            "canonical_bash_tool_turns": 517,
            "reasoned_tool_turns": 468,
            "stage_a": {"summary": {"rows": 1247, "base_rows": 1211, "new_fable_rows": 36}},
            "source_fable": {"source_ids": [str(index) for index in range(15)]},
        },
    }


def test_reasoned_lineage_requires_all_admitted_fable_evidence() -> None:
    _validate_reasoned_lineage(_provenance())

    broken = _provenance()
    broken["dataset"]["reasoned_tool_turns"] = 467
    with pytest.raises(ValueError, match="reasoning-contract"):
        _validate_reasoned_lineage(broken)
