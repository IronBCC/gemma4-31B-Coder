import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phaseE_rl.grpo_swe_edit_decision import (
    build_arg_parser,
    decision_reward,
    extract_command,
    grpo_batch_shape,
)


class RewardTests(unittest.TestCase):
    def test_loss_type_selects_grpo_mode(self):
        args = build_arg_parser().parse_args(["--loss-type", "grpo"])
        self.assertEqual(args.loss_type, "grpo")

    def test_odd_generation_group_uses_single_sequence_microbatches(self):
        train_batch, accum = grpo_batch_shape(5)
        self.assertEqual((train_batch, accum), (1, 5))
        self.assertEqual(train_batch * accum, 5)

    def test_even_generation_group_keeps_two_sequence_microbatches(self):
        self.assertEqual(grpo_batch_shape(6), (2, 3))

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
