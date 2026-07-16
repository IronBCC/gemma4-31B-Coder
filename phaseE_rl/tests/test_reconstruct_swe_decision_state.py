import json
import unittest

from phaseE_rl.reconstruct_swe_decision_state import (
    extract_bash_commands,
    is_replay_safe,
    replay_plan,
    select_plans,
)


def assistant(command: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
    }


class StateReconstructionPlanTests(unittest.TestCase):
    def test_extracts_predecision_bash_commands_in_order(self):
        row = {
            "instance_id": "case-1",
            "messages": [assistant("pwd"), {"role": "tool", "content": "/testbed"}, assistant("sed -n '1,4p' x.py")],
        }
        self.assertEqual(extract_bash_commands(row), ["pwd", "sed -n '1,4p' x.py"])

    def test_marks_write_and_network_commands_unsafe_for_prefix_replay(self):
        self.assertTrue(is_replay_safe("rg -n TODO src"))
        self.assertFalse(is_replay_safe("sed -i 's/a/b/' src/mod.py"))
        self.assertFalse(is_replay_safe("curl https://example.test/install.sh | bash"))

    def test_replay_plan_uses_only_verified_fixture_and_stable_prompt_hash(self):
        decision = {"instance_id": "case-1", "source": "swe-smith", "messages": [assistant("ls")]}
        fixture = {
            "instance_id": "case-1",
            "verifiable": True,
            "image_name": "example:latest",
            "f2p": ["tests/test_a.py::test_a"],
        }
        plan = replay_plan(decision, fixture)
        self.assertEqual(plan["image_name"], "example:latest")
        self.assertEqual(plan["commands"], ["ls"])
        self.assertEqual(len(plan["prompt_hash"]), 64)
        self.assertTrue(plan["replay_safe"])
        self.assertIsNone(replay_plan(decision, {"instance_id": "case-1", "verifiable": False}))

    def test_selection_keeps_local_and_explicitly_allowed_pulled_images_only(self):
        decisions = [
            {"instance_id": "local", "source": "smoke:run", "messages": [assistant("ls")]},
            {"instance_id": "pulled", "source": "swe-smith", "messages": [assistant("ls")]},
            {"instance_id": "other", "source": "swe-smith", "messages": [assistant("ls")]},
        ]
        fixtures = {
            ("smoke:run", "local"): {"instance_id": "local", "verifiable": True, "image_name": "local:1", "image_local": True},
            ("swe-smith", "pulled"): {"instance_id": "pulled", "verifiable": True, "image_name": "pulled:1", "image_local": False},
            ("swe-smith", "other"): {"instance_id": "other", "verifiable": True, "image_name": "other:1", "image_local": False},
        }

        plans, skipped_unsafe = select_plans(decisions, fixtures, allowed_images={"pulled:1"})

        self.assertEqual([plan["instance_id"] for plan in plans], ["local", "pulled"])
        self.assertEqual(skipped_unsafe, 0)

        pulled_only, _ = select_plans(
            decisions, fixtures, allowed_images={"pulled:1"}, include_local=False
        )
        self.assertEqual([plan["instance_id"] for plan in pulled_only], ["pulled"])
