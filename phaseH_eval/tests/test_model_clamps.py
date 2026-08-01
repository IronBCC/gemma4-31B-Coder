import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from phaseH_eval.model_clamps import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STEP_LIMIT,
    DEFAULT_STOP_SEQUENCES,
    SOURCE_ONLY_DIFF_COMMAND,
    force_diff_step,
    force_submit_step,
    no_edit_pressure_step,
    step_limit,
    add_budget_pressure_messages,
    build_call_kwargs,
    compact_live_messages,
    command_kind,
    extract_command_outcomes,
    extract_recoverable_command_from_arguments,
    extract_recoverable_command_from_text,
    forced_command_for_history,
    recovery_pressure_for_history,
)


class ModelClampTests(unittest.TestCase):
    def test_build_call_kwargs_applies_defaults(self):
        call_kwargs = build_call_kwargs({"temperature": 0})

        self.assertEqual(call_kwargs["max_tokens"], DEFAULT_MAX_TOKENS)
        self.assertEqual(call_kwargs["stop"], list(DEFAULT_STOP_SEQUENCES))
        self.assertEqual(call_kwargs["temperature"], 0)

    def test_build_call_kwargs_preserves_existing_values(self):
        call_kwargs = build_call_kwargs({"max_tokens": 99, "stop": ["<x>", "<|turn>"], "top_p": 0.5})

        self.assertEqual(call_kwargs["max_tokens"], 99)
        self.assertEqual(call_kwargs["stop"], ["<x>", "<|turn>", *DEFAULT_STOP_SEQUENCES[1:]])
        self.assertEqual(call_kwargs["top_p"], 0.5)

    def test_compact_live_messages_compacts_and_deduplicates_tool_observations(self):
        long_observation = "line 0\n" + "\n".join(f"noise {idx}" for idx in range(200)) + "\nfailed at tail"
        messages = [
            {"role": "system", "content": "fix bug"},
            {"role": "tool", "content": long_observation},
            {"role": "assistant", "content": "inspect"},
            {"role": "tool", "content": long_observation},
        ]

        compacted = compact_live_messages(messages, max_observation_chars=200)

        self.assertEqual(messages[1]["content"], long_observation)
        self.assertLessEqual(len(compacted[1]["content"]), 200)
        self.assertIn("failed at tail", compacted[1]["content"])
        self.assertIn("unchanged tool observation omitted", compacted[3]["content"])

    def test_command_kind_classifies_edit_before_read(self):
        self.assertEqual(command_kind("sed -i 's/a/b/' src/x.py"), "edit")
        self.assertEqual(command_kind("sed -n '1,20p' src/x.py"), "read")

    def test_add_budget_pressure_for_repeated_read_loop(self):
        messages = [
            _assistant_command("sed -n '1,20p' src/x.py")
            for _ in range(6)
        ]

        pressured = add_budget_pressure_messages(messages)

        self.assertEqual(pressured[-1]["role"], "user")
        self.assertIn("Loop warning", pressured[-1]["content"])

    def test_no_edit_budget_pressure_tracks_the_step_limit(self):
        with mock.patch.dict(os.environ, {"MSWEA_STEP_LIMIT": "40"}, clear=False):
            self.assertEqual(no_edit_pressure_step(), 24)
            before = [_assistant_command(f"rg pattern-{index} src") for index in range(23)]
            at_limit = [*before, _assistant_command("rg pattern-23 src")]

            self.assertIs(add_budget_pressure_messages(before), before)
            pressured = add_budget_pressure_messages(at_limit)
            self.assertEqual(pressured[-1]["role"], "user")
            self.assertIn("Stop reading", pressured[-1]["content"])

    def test_forced_command_repeated_edit_moves_to_diff(self):
        messages = [
            _assistant_command("sed -i 's/a/b/' src/x.py")
            for _ in range(3)
        ]

        self.assertEqual(forced_command_for_history(messages), SOURCE_ONLY_DIFF_COMMAND)

    def test_repeated_failed_reproducer_requests_diagnosis_not_submission(self):
        messages = _command_history("python3 reproduce_issue.py", returncode=1, count=3)

        pressured = add_budget_pressure_messages(messages)

        self.assertIn("Do not run this command again", pressured[-1]["content"])
        self.assertEqual(
            forced_command_for_history(messages),
            "git diff --check; git status --short; git diff -- . | sed -n '1,240p'",
        )

    def test_repeated_failed_edit_requests_diagnosis_not_diff(self):
        messages = _command_history("sed -i 's/old/new/' src/x.py", returncode=2, count=3)

        self.assertEqual(
            forced_command_for_history(messages),
            "git diff --check; git status --short; git diff -- . | sed -n '1,240p'",
        )

    def test_repeated_successful_edit_may_advance_to_diff(self):
        messages = _command_history("sed -i 's/old/new/' src/x.py", returncode=0, count=3)

        self.assertEqual(forced_command_for_history(messages), SOURCE_ONLY_DIFF_COMMAND)

    def test_diagnostic_returns_control_with_failed_command_named(self):
        failed = "python3 reproduce_issue.py"
        messages = _command_history(failed, returncode=1, count=3)
        messages.extend(_command_exchange(
            "git diff --check; git status --short; git diff -- . | sed -n '1,240p'",
            returncode=0,
            index=3,
        ))

        self.assertIsNone(forced_command_for_history(messages))
        pressure = recovery_pressure_for_history(messages)
        self.assertIsNotNone(pressure)
        self.assertIn(failed, pressure)
        self.assertIn("repair or revert", pressure)

    def test_extract_command_outcomes_deduplicates_actions_and_tool_calls(self):
        messages = _command_history("python3 reproduce_issue.py", returncode=1, count=1)

        outcomes = extract_command_outcomes(messages)

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].command, "python3 reproduce_issue.py")
        self.assertEqual(outcomes[0].returncode, 1)
        self.assertEqual(len(outcomes[0].output_sha256), 64)

    def test_extract_command_outcomes_does_not_pair_nonadjacent_observation(self):
        messages = [
            _assistant_command("python3 reproduce_issue.py", call_id="call-0"),
            {"role": "assistant", "content": "I should inspect the error."},
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "content": "<returncode>1</returncode>\n<output>failure</output>",
            },
        ]

        self.assertEqual(extract_command_outcomes(messages), [])

    def test_extract_command_outcomes_rejects_ambiguous_multi_call_turn(self):
        first = _assistant_command("python one.py", call_id="call-0")
        second = _assistant_command("python two.py", call_id="call-1")
        first["tool_calls"].extend(second["tool_calls"])
        first["extra"]["actions"].extend(second["extra"]["actions"])
        messages = [
            first,
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "content": "<returncode>1</returncode>\n<output>failure</output>",
            },
        ]

        self.assertEqual(extract_command_outcomes(messages), [])

    def test_forced_command_repeated_read_moves_to_diff(self):
        messages = [
            _assistant_command("sed -n '1,20p' src/x.py")
            for _ in range(5)
        ]

        self.assertEqual(forced_command_for_history(messages), SOURCE_ONLY_DIFF_COMMAND)

    def test_forced_diff_fires_just_below_the_step_limit(self):
        with mock.patch.dict(os.environ, {"MSWEA_STEP_LIMIT": "40"}, clear=False):
            self.assertEqual(force_diff_step(), 37)
            self.assertEqual(force_submit_step(), 38)
            before = [_assistant_command(f"rg pattern-{index} src") for index in range(36)]
            at_limit = [*before, _assistant_command("rg pattern-36 src")]

            self.assertIsNone(forced_command_for_history(before))
            self.assertEqual(
                forced_command_for_history(at_limit),
                SOURCE_ONLY_DIFF_COMMAND,
            )

    def test_forced_steps_scale_with_a_larger_step_limit(self):
        with mock.patch.dict(os.environ, {"MSWEA_STEP_LIMIT": "120"}, clear=False):
            self.assertEqual(force_diff_step(), 117)
            self.assertEqual(force_submit_step(), 118)
            history = [_assistant_command(f"rg pattern-{index} src") for index in range(60)]
            self.assertIsNone(forced_command_for_history(history))

    def test_forced_steps_can_be_disabled_and_overridden(self):
        with mock.patch.dict(os.environ, {"MSWEA_FORCE_SUBMIT_STEP": "0"}, clear=False):
            self.assertIsNone(force_submit_step())
        with mock.patch.dict(os.environ, {"MSWEA_FORCE_DIFF_STEP": "9"}, clear=False):
            self.assertEqual(force_diff_step(), 9)

    def test_step_limit_defaults_when_env_is_absent_or_junk(self):
        with mock.patch.dict(os.environ, {"MSWEA_STEP_LIMIT": ""}, clear=False):
            self.assertEqual(step_limit(), DEFAULT_STEP_LIMIT)
        with mock.patch.dict(os.environ, {"MSWEA_STEP_LIMIT": "not-a-number"}, clear=False):
            self.assertEqual(step_limit(), DEFAULT_STEP_LIMIT)

    def test_forced_command_repeated_diff_moves_to_submit(self):
        messages = [
            _assistant_command("git diff -- src/x.py > patch.txt && cat patch.txt")
            for _ in range(2)
        ]

        self.assertEqual(forced_command_for_history(messages), "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")

    def test_forced_diff_with_diff_output_moves_immediately_to_submit(self):
        messages = [
            _assistant_command("sed -i 's/a/b/' src/x.py"),
            _guarded_command("git diff -- . > patch.txt && cat patch.txt"),
            {"role": "tool", "content": "<returncode>0</returncode>\n<output>\ndiff --git a/src/x.py b/src/x.py\n</output>"},
        ]

        self.assertEqual(forced_command_for_history(messages), "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")

    def test_forced_diff_with_empty_output_refreshes_staged_diff_before_submit(self):
        messages = [
            _assistant_command("sed -n '1,20p' src/x.py"),
            _guarded_command("git diff -- . > patch.txt && cat patch.txt"),
            {"role": "tool", "content": "<returncode>0</returncode>\n<output>\n</output>"},
        ]

        self.assertEqual(
            forced_command_for_history(messages),
            SOURCE_ONLY_DIFF_COMMAND,
        )

    def test_edit_after_forced_diff_refreshes_patch_before_submit(self):
        messages = [
            _assistant_command("sed -i 's/a/b/' src/x.py"),
            _guarded_command("git diff -- . > patch.txt && cat patch.txt"),
            {
                "role": "tool",
                "content": (
                    "<returncode>0</returncode>\n<output>\n"
                    "diff --git a/src/x.py b/src/x.py\n</output>"
                ),
            },
            _assistant_command("sed -i 's/b/c/' src/x.py"),
            {"role": "tool", "content": "<returncode>0</returncode>\n<output></output>"},
        ]

        self.assertEqual(
            forced_command_for_history(messages),
            SOURCE_ONLY_DIFF_COMMAND,
        )

    def test_source_only_diff_omits_protected_dirt_and_includes_new_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=repo,
                check=True,
            )
            (repo / "src").mkdir()
            (repo / "src" / "old.py").write_text("old = 1\n")
            (repo / "setup.py").write_text("name = 'before'\n")
            (repo / "tox.ini").write_text("[tox]\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
            (repo / "src" / "old.py").write_text("old = 2\n")
            (repo / "src" / "new.py").write_text("new = 1\n")
            (repo / "setup.py").write_text("name = 'dirty'\n")
            (repo / "tox.ini").write_text("[tox]\nenvlist = py\n")

            completed = subprocess.run(
                SOURCE_ONLY_DIFF_COMMAND,
                cwd=repo,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            patch = (repo / "patch.txt").read_text()
            self.assertIn("diff --git a/src/old.py b/src/old.py", patch)
            self.assertIn("diff --git a/src/new.py b/src/new.py", patch)
            self.assertNotIn("setup.py", patch)
            self.assertNotIn("tox.ini", patch)

    def test_protected_submission_rejection_rebuilds_source_only_patch(self):
        messages = [
            _assistant_command("sed -i 's/a/b/' src/x.py"),
            _guarded_command(
                "git add -A && git diff --cached > patch.txt && cat patch.txt"
            ),
            {
                "role": "tool",
                "content": (
                    "<returncode>0</returncode>\n<output>\n"
                    "diff --git a/src/x.py b/src/x.py\n"
                    "diff --git a/setup.py b/setup.py\n</output>"
                ),
            },
            _guarded_command(
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
            ),
            {
                "role": "tool",
                "content": (
                    "<returncode>1</returncode>\n"
                    "SUBMISSION REJECTED: protected path is not allowed: "
                    "setup.py, tox.ini"
                ),
            },
        ]

        self.assertEqual(
            forced_command_for_history(messages),
            SOURCE_ONLY_DIFF_COMMAND,
        )

    def test_source_only_recovery_diff_can_be_submitted(self):
        messages = [
            _assistant_command("sed -i 's/a/b/' src/x.py"),
            _guarded_command(
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
            ),
            {
                "role": "tool",
                "content": (
                    "<returncode>1</returncode>\n"
                    "SUBMISSION REJECTED: protected path is not allowed: setup.py"
                ),
            },
            _guarded_command(SOURCE_ONLY_DIFF_COMMAND),
            {
                "role": "tool",
                "content": (
                    "<returncode>0</returncode>\n<output>\n"
                    "diff --git a/src/x.py b/src/x.py\n</output>"
                ),
            },
        ]

        self.assertEqual(
            forced_command_for_history(messages),
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        )

    def test_same_failed_output_from_different_commands_triggers_diagnosis(self):
        messages = []
        for index, command in enumerate(
            ("pytest -q", "python -m pytest -q", "./test.sh")
        ):
            call_id = f"call-{index}"
            messages.extend(
                [
                    _assistant_command(command, call_id=call_id),
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            "<returncode>1</returncode>\n"
                            "<output>same collection failure</output>"
                        ),
                        "extra": {"raw_output": "same collection failure"},
                    },
                ]
            )

        self.assertEqual(
            forced_command_for_history(messages),
            "git diff --check; git status --short; git diff -- . | sed -n '1,240p'",
        )

    def test_extract_command_from_nested_tool_arguments(self):
        arguments = '{"description": "repro", "parameters": {"command": "python repro.py"}}'

        self.assertEqual(extract_recoverable_command_from_arguments(arguments), "python repro.py")

    def test_extract_command_from_command_list(self):
        arguments = '{"description": "inspect", "commands": ["pwd", "sed -n \\"1,20p\\" src/x.py"]}'

        self.assertEqual(extract_recoverable_command_from_arguments(arguments), 'pwd\nsed -n "1,20p" src/x.py')

    def test_extract_command_from_text_pseudo_tool_call(self):
        text = 'I will inspect. [Makes bash tool call: {"command": "find . -name \\"*.py\\""}]'

        self.assertEqual(extract_recoverable_command_from_text(text), 'find . -name "*.py"')

def _assistant_command(command: str, *, call_id: str = "call-1") -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        "extra": {"actions": [{"command": command, "tool_call_id": call_id}]},
    }


def _command_exchange(command: str, *, returncode: int, index: int) -> list[dict]:
    call_id = f"call-{index}"
    return [
        _assistant_command(command, call_id=call_id),
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"<returncode>{returncode}</returncode>\n<output>failure</output>",
            "extra": {"raw_output": "failure"},
        },
    ]


def _command_history(command: str, *, returncode: int, count: int) -> list[dict]:
    messages: list[dict] = []
    for index in range(count):
        messages.extend(_command_exchange(command, returncode=returncode, index=index))
    return messages


def _guarded_command(command: str) -> dict:
    message = _assistant_command(command)
    message["extra"]["response"] = {"guarded_forced_command": command}
    return message


if __name__ == "__main__":
    unittest.main()
