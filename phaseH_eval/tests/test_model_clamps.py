import unittest

from phaseH_eval.model_clamps import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STOP_SEQUENCES,
    add_budget_pressure_messages,
    build_call_kwargs,
    compact_live_messages,
    command_kind,
    extract_recoverable_command_from_arguments,
    extract_recoverable_command_from_text,
    forced_command_for_history,
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

    def test_no_edit_budget_pressure_starts_at_command_25(self):
        before = [_assistant_command(f"rg pattern-{index} src") for index in range(24)]
        at_limit = [*before, _assistant_command("rg pattern-24 src")]

        self.assertIs(add_budget_pressure_messages(before), before)
        pressured = add_budget_pressure_messages(at_limit)
        self.assertEqual(pressured[-1]["role"], "user")
        self.assertIn("Stop reading", pressured[-1]["content"])

    def test_forced_command_repeated_edit_moves_to_diff(self):
        messages = [
            _assistant_command("sed -i 's/a/b/' src/x.py")
            for _ in range(3)
        ]

        self.assertEqual(forced_command_for_history(messages), "git diff -- . > patch.txt && cat patch.txt")

    def test_forced_command_repeated_read_moves_to_diff(self):
        messages = [
            _assistant_command("sed -n '1,20p' src/x.py")
            for _ in range(5)
        ]

        self.assertEqual(forced_command_for_history(messages), "git diff -- . > patch.txt && cat patch.txt")

    def test_command_37_forces_diff_without_an_edit(self):
        before = [_assistant_command(f"rg pattern-{index} src") for index in range(36)]
        at_limit = [*before, _assistant_command("rg pattern-36 src")]

        self.assertIsNone(forced_command_for_history(before))
        self.assertEqual(
            forced_command_for_history(at_limit),
            "git diff -- . > patch.txt && cat patch.txt",
        )

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

    def test_forced_diff_with_empty_output_moves_immediately_to_submit(self):
        messages = [
            _assistant_command("sed -n '1,20p' src/x.py"),
            _guarded_command("git diff -- . > patch.txt && cat patch.txt"),
            {"role": "tool", "content": "<returncode>0</returncode>\n<output>\n</output>"},
        ]

        self.assertEqual(forced_command_for_history(messages), "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")

    def test_extract_command_from_nested_tool_arguments(self):
        arguments = '{"description": "repro", "parameters": {"command": "python repro.py"}}'

        self.assertEqual(extract_recoverable_command_from_arguments(arguments), "python repro.py")

    def test_extract_command_from_command_list(self):
        arguments = '{"description": "inspect", "commands": ["pwd", "sed -n \\"1,20p\\" src/x.py"]}'

        self.assertEqual(extract_recoverable_command_from_arguments(arguments), 'pwd\nsed -n "1,20p" src/x.py')

    def test_extract_command_from_text_pseudo_tool_call(self):
        text = 'I will inspect. [Makes bash tool call: {"command": "find . -name \\"*.py\\""}]'

        self.assertEqual(extract_recoverable_command_from_text(text), 'find . -name "*.py"')

def _assistant_command(command: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": f'{{"command": {command!r}}}'},
            }
        ],
        "extra": {"actions": [{"command": command, "tool_call_id": "call-1"}]},
    }


def _guarded_command(command: str) -> dict:
    message = _assistant_command(command)
    message["extra"]["response"] = {"guarded_forced_command": command}
    return message


if __name__ == "__main__":
    unittest.main()
