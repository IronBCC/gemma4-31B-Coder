from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from phaseD_sft.build_verified_teacher_finalpatch_sft import (
    build_verified_teacher_finalpatch_rows,
    portable_git_apply_command,
    render_verified_teacher_finalpatch_sft,
)


PATCH = """diff --git a/pkg/target.py b/pkg/target.py
index 1234567..89abcde 100644
--- a/pkg/target.py
+++ b/pkg/target.py
@@ -1,2 +1,2 @@
 def target():
-    return 'old'
+    return 'new'
"""

MULTI_PATCH = PATCH + """diff --git a/pkg/helper.py b/pkg/helper.py
index 1111111..2222222 100644
--- a/pkg/helper.py
+++ b/pkg/helper.py
@@ -1 +1 @@
-OLD = True
+OLD = False
"""


def _completed(item: dict[str, object]) -> str:
    return json.dumps({"type": "item.completed", "item": item})


def _command(command: str, output: str, *, exit_code: int = 0) -> str:
    payload = f"docker exec abc123def456 bash -c {shlex.quote(f'cd /testbed && {command}')}"
    wrapped = f"/bin/bash -lc {shlex.quote(payload)}"
    return _completed(
        {
            "type": "command_execution",
            "command": wrapped,
            "aggregated_output": output,
            "exit_code": exit_code,
        }
    )


def _record(*, patch: str = PATCH) -> dict[str, object]:
    return {
        "record_id": "teacher/task-a",
        "task": {
            "instance_id": "task-a",
            "problem_statement": "Fix target() in pkg/target.py.",
            "repo": "repo-a",
            "FAIL_TO_PASS": ["tests/test_target.py::test_target"],
            "patch": "HIDDEN_GOLD_PATCH_MUST_NOT_APPEAR",
            "test_patch": "HIDDEN_TEST_PATCH_MUST_NOT_APPEAR",
        },
        "result": {
            "instance_id": "task-a",
            "backend": "codex",
            "model": "gpt-5.6-terra",
            "resolved": True,
        },
        "patch": patch,
        "stream_lines": [
            _command("ls", "README.md\npkg"),
            _command(
                "sed -n '1,80p' pkg/target.py",
                "pkg/target.py\ndef target():\n    return 'old'",
            ),
            _command(
                "apply_patch <<'PATCH'\n*** Begin Patch\n*** Update File: pkg/target.py\n"
                "@@\n-    return 'old'\n+    return 'new'\n*** End Patch\nPATCH",
                "Done!",
            ),
            _command("python -m pytest -q tests/test_target.py", "1 passed"),
        ],
        "system_template": "You are a practical software engineer.",
        "instance_template": "<pr_description>\n{{task}}\n</pr_description>",
    }


def _assistant_commands(row: dict[str, object]) -> list[str]:
    commands: list[str] = []
    for message in row["messages"]:
        if message["role"] != "assistant":
            continue
        call = message["tool_calls"][0]
        commands.append(json.loads(call["function"]["arguments"])["command"])
    return commands


def test_renders_portable_final_patch_instead_of_harness_specific_first_edit() -> None:
    row, report = render_verified_teacher_finalpatch_sft(_record())

    assert report["kept"] is True
    assert row is not None
    commands = _assistant_commands(row)
    assert commands[:-1] == ["sed -n '1,80p' pkg/target.py"]
    assert commands[-1].startswith("git apply - <<'PATCH_")
    assert PATCH.strip() in commands[-1]
    assert "apply_patch <<" not in commands[-1]
    assert [m["loss"] for m in row["messages"] if m["role"] == "assistant"] == [False, True]

    rendered = json.dumps(row)
    assert "HIDDEN_GOLD_PATCH_MUST_NOT_APPEAR" not in rendered
    assert "HIDDEN_TEST_PATCH_MUST_NOT_APPEAR" not in rendered
    assert "python -m pytest" not in rendered


def test_keeps_one_pre_edit_observation_for_every_edited_file() -> None:
    record = _record(patch=MULTI_PATCH)
    record["stream_lines"] = [
        _command("sed -n '1,80p' pkg/target.py", "pkg/target.py\ndef target():"),
        _command("sed -n '1,80p' pkg/helper.py", "pkg/helper.py\nOLD = True"),
        _command("sed -i s/old/new/ pkg/target.py", ""),
    ]

    row, report = render_verified_teacher_finalpatch_sft(record, max_grounded_reads=3)

    assert report["kept"] is True
    assert report["covered_patch_paths"] == ["pkg/helper.py", "pkg/target.py"]
    assert row is not None
    assert _assistant_commands(row)[:-1] == [
        "sed -n '1,80p' pkg/target.py",
        "sed -n '1,80p' pkg/helper.py",
    ]
    for index, message in enumerate(row["messages"][:-1]):
        if message["role"] == "assistant":
            assert row["messages"][index + 1]["role"] == "user"


def test_rejects_when_any_edited_file_was_not_grounded_before_first_edit() -> None:
    record = _record(patch=MULTI_PATCH)

    row, report = render_verified_teacher_finalpatch_sft(record)

    assert row is None
    assert report["reason"] == "edited_path_not_seen"
    assert report["missing_patch_paths"] == ["pkg/helper.py"]


def test_requires_a_bounded_nonempty_f2p_contract() -> None:
    over_cap = _record()
    over_cap["task"] = {**over_cap["task"], "FAIL_TO_PASS": [f"test_{i}" for i in range(21)]}
    row, report = render_verified_teacher_finalpatch_sft(over_cap, max_f2p_tests=20)
    assert row is None
    assert report["reason"] == "f2p_over_cap"

    missing = _record()
    missing["task"] = {**missing["task"], "FAIL_TO_PASS": []}
    row, report = render_verified_teacher_finalpatch_sft(missing)
    assert row is None
    assert report["reason"] == "no_f2p_tests"


def test_portable_command_uses_a_noncolliding_quoted_heredoc() -> None:
    patch = PATCH + "\nPATCH_deadbeef\n"
    command = portable_git_apply_command(patch)

    first_line, *body, delimiter = command.splitlines()
    assert first_line == f"git apply - <<'{delimiter}'"
    assert "\n".join(body) == patch.strip()
    assert delimiter not in body


def test_portable_command_preserves_unified_diff_trailing_whitespace(tmp_path: Path) -> None:
    patch = PATCH.replace("+    return 'new'", "+    return 'new'  ")
    command = portable_git_apply_command(patch)

    assert "+    return 'new'  \n" in command

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    repo.joinpath("pkg").mkdir()
    repo.joinpath("pkg/target.py").write_text("def target():\n    return 'old'\n")
    subprocess.run(["bash", "-c", command], cwd=repo, check=True)
    assert repo.joinpath("pkg/target.py").read_text() == "def target():\n    return 'new'  \n"


def test_basename_only_observation_does_not_ground_a_different_path() -> None:
    record = _record()
    record["stream_lines"] = [
        _command("sed -n '1,80p' other/target.py", "other/target.py\ndef target():"),
        _command("sed -i s/old/new/ pkg/target.py", ""),
    ]

    row, report = render_verified_teacher_finalpatch_sft(record)

    assert row is None
    assert report["reason"] == "edited_path_not_seen"
    assert report["missing_patch_paths"] == ["pkg/target.py"]


def test_compaction_preserves_truthful_exact_path_grounding() -> None:
    record = _record()
    record["stream_lines"] = [
        _command(
            "sed -n '1,800p' pkg/target.py",
            "A" * 500 + "\npkg/target.py\n" + "B" * 500,
        ),
        _command("sed -i s/old/new/ pkg/target.py", ""),
    ]

    row, report = render_verified_teacher_finalpatch_sft(record, max_observation_chars=200)

    assert report["kept"] is True
    assert row is not None
    observations = [message["content"] for message in row["messages"] if message["role"] == "user"][1:]
    assert any("<grounded_paths>pkg/target.py</grounded_paths>" in text for text in observations)


@pytest.mark.parametrize(
    "bad_patch",
    [
        PATCH.replace("diff --git a/pkg/target.py b/pkg/target.py", "diff --git a/pkg/target.py b/pkg/renamed.py"),
        PATCH.replace("diff --git a/pkg/target.py b/pkg/target.py\n", "diff --git a/pkg/target.py b/pkg/target.py\nnew file mode 100644\n"),
        PATCH.replace("pkg/target.py", "../pkg/target.py"),
        PATCH + "GIT binary patch\nliteral 1\nAcmZQz\n",
    ],
)
def test_rejects_unsupported_or_unsafe_patch_shapes(bad_patch: str) -> None:
    row, report = render_verified_teacher_finalpatch_sft(_record(patch=bad_patch))

    assert row is None
    assert report["reason"] == "unsupported_patch_shape"


def test_build_filters_to_exact_matching_independent_tier_one_credit() -> None:
    row, _ = render_verified_teacher_finalpatch_sft(_record())
    assert row is not None
    command = _assistant_commands(row)[-1]
    ledger = {
        "task-a": {
            "instance_id": "task-a",
            "tier": 1.0,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        }
    }

    rows, manifest, rejected = build_verified_teacher_finalpatch_rows(
        [_record()],
        token_counter=lambda messages: 123,
        credit_ledger=ledger,
    )

    assert [item["source_instance_id"] for item in rows] == ["task-a"]
    assert manifest["rows_kept"] == 1
    assert manifest["tier_one_credit_rows"] == 1
    assert rejected == []


def test_build_rejects_missing_or_nonpassing_credit_and_stale_hashes() -> None:
    row, _ = render_verified_teacher_finalpatch_sft(_record())
    assert row is not None
    command = _assistant_commands(row)[-1]
    command_sha = hashlib.sha256(command.encode()).hexdigest()

    rows, manifest, rejected = build_verified_teacher_finalpatch_rows(
        [_record()],
        token_counter=lambda messages: 10,
        credit_ledger={"task-a": {"instance_id": "task-a", "tier": 0.8, "command_sha256": command_sha}},
    )
    assert rows == []
    assert manifest["drop_reasons"] == {"independent_credit_below_1.0": 1}
    assert rejected[0]["reason"] == "independent_credit_below_1.0"

    with pytest.raises(ValueError, match="stale credit ledger"):
        build_verified_teacher_finalpatch_rows(
            [_record()],
            token_counter=lambda messages: 10,
            credit_ledger={"task-a": {"instance_id": "task-a", "tier": 1.0, "command_sha256": "bad"}},
        )


def test_build_deduplicates_content_and_applies_token_budget() -> None:
    duplicate = _record()
    duplicate["record_id"] = "teacher/task-b"
    duplicate["task"] = {**duplicate["task"], "instance_id": "task-b"}
    duplicate["result"] = {**duplicate["result"], "instance_id": "task-b"}
    oversized = _record()
    oversized["record_id"] = "teacher/task-c"
    oversized["task"] = {
        **oversized["task"],
        "instance_id": "task-c",
        "problem_statement": "OVERSIZED Fix target() in pkg/target.py.",
    }
    oversized["result"] = {**oversized["result"], "instance_id": "task-c"}

    rows, manifest, rejected = build_verified_teacher_finalpatch_rows(
        [_record(), duplicate, oversized],
        token_counter=lambda messages: 100 if "OVERSIZED" in json.dumps(messages) else 10,
        max_tokens=50,
    )

    assert [item["source_instance_id"] for item in rows] == ["task-a"]
    assert manifest["content_duplicates"] == 1
    assert manifest["budget_rejected"] == 1
    assert {item["reason"] for item in rejected} == {"content_duplicate", "token_budget_exceeded"}
