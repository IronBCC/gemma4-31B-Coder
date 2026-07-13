from __future__ import annotations

import unittest

from phaseD_sft.sanitize_agentic_empty_assistant import sanitize_row


class SanitizeEmptyAssistantTests(unittest.TestCase):
    def test_removes_empty_assistant_and_merges_adjacent_observations(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "system", "tool_calls": []},
                {"role": "user", "content": "task", "tool_calls": []},
                {"role": "assistant", "content": "inspect", "tool_calls": []},
                {"role": "user", "content": "OBSERVATION:\nfirst", "tool_calls": []},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "user", "content": "OBSERVATION:\nsecond", "tool_calls": []},
                {"role": "assistant", "content": "patch", "tool_calls": []},
            ]
        }

        sanitized, stats = sanitize_row(row)

        self.assertEqual(stats["empty_assistant_removed"], 1)
        self.assertEqual(stats["adjacent_observations_merged"], 1)
        self.assertEqual(sanitized["messages"][3]["content"], "OBSERVATION:\nfirst\n\nsecond")
        self.assertEqual(sanitized["messages"][-1]["content"], "patch")
        self.assertFalse(any(message["role"] == "assistant" and not message["content"] for message in sanitized["messages"]))

    def test_preserves_nonempty_assistant_tool_call(self) -> None:
        tool_call = {"id": "call", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        row = {"messages": [{"role": "assistant", "content": "", "tool_calls": [tool_call]}]}

        sanitized, stats = sanitize_row(row)

        self.assertEqual(stats["empty_assistant_removed"], 0)
        self.assertEqual(sanitized, row)


if __name__ == "__main__":
    unittest.main()
