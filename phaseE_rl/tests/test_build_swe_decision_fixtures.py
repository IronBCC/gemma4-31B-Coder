import unittest

from phaseE_rl.build_swe_decision_fixtures import build_fixture_rows, summarize_coverage


class DecisionFixtureJoinTests(unittest.TestCase):
    def test_join_prefers_exact_instance_id_and_preserves_f2p(self):
        decisions = [
            {"source": "swe-smith", "instance_id": "smith-1"},
            {"source": "smoke:run", "instance_id": "lite-1"},
            {"source": "oracle", "instance_id": "unknown"},
        ]
        swe_smith = [
            {
                "instance_id": "smith-1",
                "image_name": "jyangballin/swesmith.x86_64.smith-1",
                "FAIL_TO_PASS": ["tests/test_a.py::test_a"],
            }
        ]
        swe_lite = [
            {
                "instance_id": "lite-1",
                "image_name": "swebench/sweb.eval.x86_64.lite-1:latest",
                "FAIL_TO_PASS": '["tests/test_b.py::test_b"]',
                "test_patch": "diff --git a/tests/test_b.py b/tests/test_b.py",
            }
        ]

        rows = build_fixture_rows(decisions, swe_smith, swe_lite, local_images=set())

        self.assertEqual(rows[0]["fixture_source"], "swe-smith")
        self.assertTrue(rows[0]["verifiable"])
        self.assertEqual(rows[0]["f2p"], ["tests/test_a.py::test_a"])
        self.assertFalse(rows[0]["image_local"])
        self.assertEqual(rows[1]["fixture_source"], "swe-lite")
        self.assertEqual(rows[1]["f2p"], ["tests/test_b.py::test_b"])
        self.assertEqual(rows[1]["test_patch"], "diff --git a/tests/test_b.py b/tests/test_b.py")
        self.assertFalse(rows[2]["verifiable"])
        self.assertTrue(rows[2]["unverified"])

    def test_coverage_counts_resolvable_and_local_rows_separately(self):
        rows = [
            {"source": "swe-smith", "verifiable": True, "image_local": False},
            {"source": "swe-smith", "verifiable": True, "image_local": True},
            {"source": "openswe", "verifiable": False, "image_local": False},
        ]

        summary = summarize_coverage(rows)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["verifiable_rows"], 2)
        self.assertEqual(summary["local_image_rows"], 1)
        self.assertEqual(summary["by_source"]["swe-smith"]["verifiable_rows"], 2)
        self.assertEqual(summary["by_source"]["openswe"]["unverified_rows"], 1)

