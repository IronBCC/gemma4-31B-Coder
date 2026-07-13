import unittest

from phaseD_sft.agent_smoke_eval import Case, score_full, score_semantic


class AgentSmokeEvalTests(unittest.TestCase):
    def test_semantic_score_ignores_full_output_contract(self) -> None:
        case = Case(
            name="wrapped_but_logical",
            prompt="patch only",
            must_contain=("return value is not None",),
            must_not_contain=("```",),
            semantic_must_contain=("return value is not None",),
        )

        text = "```patch\n+    return value is not None\n```"

        self.assertEqual(score_semantic(case, text), (True, []))
        full_ok, full_failures = score_full(case, text)
        self.assertFalse(full_ok)
        self.assertIn("not_raw_diff", full_failures)
        self.assertIn("forbidden:```", full_failures)

    def test_semantic_score_can_use_different_requirements(self) -> None:
        case = Case(
            name="specific_full_permissive_semantic",
            prompt="patch only",
            must_contain=("path.relative_to(ROOT.resolve())",),
            semantic_must_contain=(".resolve()", "ValueError"),
            semantic_must_not_contain=("startswith",),
        )

        text = """
--- src/files.py
+++ src/files.py
+    path = (ROOT / name).resolve()
+    if ROOT.resolve() not in path.parents:
+        raise ValueError("outside root")
"""

        self.assertEqual(score_semantic(case, text), (True, []))
        full_ok, full_failures = score_full(case, text)
        self.assertFalse(full_ok)
        self.assertIn("missing:path.relative_to(ROOT.resolve())", full_failures)

    def test_semantic_forbidden_ignores_removed_diff_lines(self) -> None:
        case = Case(
            name="remove_forbidden_code",
            prompt="patch only",
            semantic_must_contain=("copy.deepcopy(DEFAULTS)",),
            semantic_must_not_contain=("DEFAULTS.copy()",),
        )

        text = """
--- src/config.py
+++ src/config.py
+import copy
 def build_config(overrides):
-    config = DEFAULTS.copy()
+    config = copy.deepcopy(DEFAULTS)
"""

        self.assertEqual(score_semantic(case, text), (True, []))


if __name__ == "__main__":
    unittest.main()
