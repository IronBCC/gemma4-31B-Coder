import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phaseE_rl.grpo_swe_edit_decision import decision_reward, extract_command


class RewardTests(unittest.TestCase):
    def test_no_tool_call(self):
        self.assertEqual(decision_reward("let me think about this...", {"a.py"}), 0.0)

    def test_read_only_call(self):
        t = 'thought\n<|tool_call>call:bash{"command": "cat -n src/a.py"}<tool_call|>'
        self.assertEqual(decision_reward(t, {"src/a.py"}), 0.2)

    def test_edit_not_in_context(self):
        t = '<|tool_call>call:bash{"command": "sed -i \'s/x/y/\' other/b.py"}<tool_call|>'
        self.assertEqual(decision_reward(t, {"src/a.py"}), 0.6)

    def test_edit_in_context(self):
        t = '<|tool_call>call:bash{"command": "sed -i \'s/x/y/\' src/a.py"}<tool_call|>'
        self.assertEqual(decision_reward(t, {"src/a.py"}), 1.0)

    def test_heredoc_edit_in_context(self):
        t = ('<|channel>thought\nfix it\n<channel|>'
             '<|tool_call>call:bash{"command": "cat > src/a.py <<EOF\\nnew\\nEOF"}<tool_call|>')
        self.assertEqual(decision_reward(t, {"src/a.py"}), 1.0)

    def test_extract_gemma_compact_form(self):
        t = '<|tool_call>call:bash{command:<|"|>grep -rn foo .<|"|>}<tool_call|>'
        self.assertEqual(extract_command(t), "grep -rn foo .")

    def test_unterminated_call_still_extracts(self):
        t = 'call:bash{"command": "sed -i \'s/a/b/\' src/a.py"'
        self.assertEqual(decision_reward(t, {"src/a.py"}), 1.0)

    def test_empty_command(self):
        t = '<|tool_call>call:bash{"command": ""}<tool_call|>'
        self.assertEqual(decision_reward(t, {"a.py"}), 0.0)


if __name__ == "__main__":
    unittest.main()
