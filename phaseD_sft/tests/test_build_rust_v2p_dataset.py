import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phaseD_sft.build_rust_v2p_dataset import (
    ASSERT_BUCKETS,
    select_streaming_rows,
)


def row(row_id: str, asserts: int, problem: str | None = None, solution: str | None = None):
    return {
        "id": row_id,
        "n_asserts": asserts,
        "problem": problem or f"problem {row_id}",
        "solution": solution or f"fn {row_id}() {{}}",
        "tests": ["assert_eq!(1, 1);"],
    }


class RustV2pSelectionTests(unittest.TestCase):
    def test_stratifies_deterministically_and_excludes_v1_ids(self):
        rows = [
            row("old", 1), row("a", 1), row("b", 5), row("c", 11), row("d", 21),
            row("e", 21),
        ]
        quotas = {"1-4": 1, "5-10": 1, "11-20": 1, "21+": 1}
        selected, stats = select_streaming_rows(
            rows, v1_ids={"old"}, v1_problem_keys=set(), v1_solution_keys=set(),
            quotas=quotas, seed=7,
        )
        self.assertEqual({item["n_asserts"] for item in selected}, {1, 5, 11, 21})
        self.assertNotIn("old", {item["id"] for item in selected})
        self.assertEqual(stats["excluded_v1_id"], 1)
        self.assertEqual(stats["selected"], 4)

    def test_excludes_v1_content_and_deduplicates_selected_content(self):
        rows = [
            row("old-problem", 1, problem="same v1 problem"),
            row("old-solution", 5, solution="fn prior_solution() {}"),
            row("low", 1),
            row("middle", 5),
            row("keep", 11),
            row("same-problem", 11, problem="problem keep"),
            row("other-11", 11),
            row("same-solution", 21, solution="fn keep() {}"),
            row("high", 21),
            row("other-21", 21),
        ]
        quotas = {"1-4": 1, "5-10": 1, "11-20": 2, "21+": 2}
        selected, stats = select_streaming_rows(
            rows,
            v1_ids=set(),
            v1_problem_keys={"same v1 problem"},
            v1_solution_keys={"fn prior_solution() {}"},
            quotas=quotas,
            seed=0,
        )
        ids = {item["id"] for item in selected}
        self.assertNotIn("old-problem", ids)
        self.assertNotIn("old-solution", ids)
        self.assertEqual(len(selected), 6)
        self.assertGreaterEqual(stats["excluded_v1_problem"], 1)
        self.assertGreaterEqual(stats["excluded_v1_solution"], 1)
        self.assertGreaterEqual(stats["dedup_selected"], 1)

    def test_declares_all_four_expected_assert_buckets(self):
        self.assertEqual(tuple(ASSERT_BUCKETS), ("1-4", "5-10", "11-20", "21+"))


if __name__ == "__main__":
    unittest.main()
