from __future__ import annotations

import json
import hashlib

import pytest

from teacher_platform.fable5_revision import (
    RevisionStep,
    bind_replay_artifact_identity,
    repository_is_operator_excluded,
    render_revision_row,
    select_revision_steps,
    validate_verified_row_floor,
)


MUTATION = """diff --git a/src/value.py b/src/value.py
--- a/src/value.py
+++ b/src/value.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 0
"""
REPAIR = """diff --git a/src/value.py b/src/value.py
--- a/src/value.py
+++ b/src/value.py
@@ -1 +1 @@
-VALUE = 0
+VALUE = 1
"""


def _command(message: dict) -> str:
    return json.loads(
        message["tool_calls"][0]["function"]["arguments"]
    )["command"]


def test_step_selection_supervises_only_safe_fable_inspection() -> None:
    steps = [
        RevisionStep(
            thought="Inspect the implementation.",
            command="sed -n '1,120p' src/value.py",
            observation="VALUE = 1",
            returncode=0,
        ),
        RevisionStep(
            thought="Try a source edit.",
            command="sed -i 's/VALUE = 1/VALUE = 2/' src/value.py",
            observation="",
            returncode=0,
        ),
        RevisionStep(
            thought="Run the focused test.",
            command="python -m pytest -q tests/test_value.py::test_value",
            observation="1 passed",
            returncode=0,
        ),
        RevisionStep(
            thought="Inspect a missing file.",
            command="cat src/missing.py",
            observation="No such file",
            returncode=1,
        ),
    ]

    selected = select_revision_steps(steps, max_steps=12)

    assert [step.supervise for step in selected] == [
        True,
        False,
        False,
        False,
    ]


def test_revision_row_masks_bad_actions_and_supervises_verified_repair() -> None:
    task = {
        "instance_id": "org__repo.mutation__one",
        "problem_statement": "Restore VALUE.",
        "repo": "org/repo",
        "patch": MUTATION,
    }
    steps = select_revision_steps(
        [
            RevisionStep(
                thought="Inspect the implementation.",
                command="sed -n '1,120p' src/value.py",
                observation="VALUE = 1",
                returncode=0,
            ),
            RevisionStep(
                thought="Change the value.",
                command="sed -i 's/VALUE = 1/VALUE = 2/' src/value.py",
                observation="",
                returncode=0,
            ),
        ],
        max_steps=12,
    )
    evidence = {
        "training_admitted": True,
        "candidate_passed_twice": True,
        "candidate_patch_sha256": hashlib.sha256(REPAIR.encode()).hexdigest(),
        "admission_evidence_sha256": "b" * 64,
        "task_contract_sha256": "c" * 64,
    }

    row = render_revision_row(
        task=task,
        steps=steps,
        repair_patch=REPAIR,
        evidence=evidence,
        stream_sha256="d" * 64,
        legacy_patch_sha256="e" * 64,
    )

    assistants = [
        message for message in row["messages"]
        if message["role"] == "assistant"
    ]
    assert [message["loss"] for message in assistants] == [True, False, True]
    assert _command(assistants[-1]).startswith("git apply - <<'PATCH_")
    assert REPAIR.rstrip() in _command(assistants[-1])
    assert row["source"] == "teacher:claude:claude-fable-5:verified-revision"
    assert row["legacy_fable_conditioning"] is True
    assert row["oracle_repair_target"] is True
    assert row["repair_admission_evidence_sha256"] == "b" * 64


def test_revision_row_rejects_unverified_repair() -> None:
    task = {
        "instance_id": "org__repo.mutation__one",
        "problem_statement": "Restore VALUE.",
        "repo": "org/repo",
        "patch": MUTATION,
    }

    with pytest.raises(ValueError, match="exact replay admission"):
        render_revision_row(
            task=task,
            steps=[
                RevisionStep(
                    thought="Inspect.",
                    command="cat src/value.py",
                    observation="VALUE = 1",
                    returncode=0,
                    supervise=True,
                )
            ],
            repair_patch=REPAIR,
            evidence={
                "training_admitted": False,
                "candidate_passed_twice": False,
            },
            stream_sha256="d" * 64,
            legacy_patch_sha256="e" * 64,
        )


def test_revision_row_requires_original_fable_trace_context() -> None:
    task = {
        "instance_id": "org__repo.mutation__one",
        "problem_statement": "Restore VALUE.",
        "repo": "org/repo",
        "patch": MUTATION,
    }

    with pytest.raises(ValueError, match="at least one Fable trace step"):
        render_revision_row(
            task=task,
            steps=[],
            repair_patch=REPAIR,
            evidence={
                "training_admitted": True,
                "candidate_passed_twice": True,
                "candidate_patch_sha256": "a" * 64,
                "admission_evidence_sha256": "b" * 64,
                "task_contract_sha256": "c" * 64,
            },
            stream_sha256="d" * 64,
            legacy_patch_sha256="e" * 64,
        )


def test_replay_identity_binding_attaches_fields_required_by_exact_admission() -> None:
    evidence = {
        "candidate_patch_sha256": "a" * 64,
        "admission_evidence_sha256": "b" * 64,
    }

    bound = bind_replay_artifact_identity(
        evidence,
        stream_sha256="c" * 64,
    )

    assert bound == {
        **evidence,
        "patch_sha256": "a" * 64,
        "stream_sha256": "c" * 64,
    }
    assert evidence == {
        "candidate_patch_sha256": "a" * 64,
        "admission_evidence_sha256": "b" * 64,
    }


def test_verified_row_floor_allows_explicit_exhausted_collection_override() -> None:
    validate_verified_row_floor(7, minimum_rows=1)

    with pytest.raises(ValueError, match="fewer than 30"):
        validate_verified_row_floor(7, minimum_rows=30)
    with pytest.raises(ValueError, match=r"integer in \[1,30\]"):
        validate_verified_row_floor(7, minimum_rows=0)


def test_operator_repository_exclusion_is_explicit_and_case_insensitive() -> None:
    task = {"repo": "swesmith/pydicom__pydicom.7d361b3d"}

    assert repository_is_operator_excluded(
        task,
        {"SWESMITH/PYDICOM__PYDICOM.7D361B3D"},
    )
    assert not repository_is_operator_excluded(task, set())
