import unittest

from phaseE_rl.build_swe_decision_dataset import Candidate
from phaseE_rl.build_swe_decision_v3 import cap_external_share, select_v3_rows


def candidate(source: str, instance_id: str) -> Candidate:
    return Candidate(
        {"source": source, "instance_id": instance_id, "messages": [{"role": "user", "content": instance_id}]},
        "smoke" if source.startswith("smoke:") else "sft",
    )


class V3SelectionTests(unittest.TestCase):
    def test_actual_shortfall_keeps_external_rows_at_twenty_percent(self):
        selected = [
            (candidate("swe-smith", f"smith-{number}"), 10, {"verifiable": True})
            for number in range(4)
        ] + [
            (candidate("swe_train_oracle_edit_trace", "oracle"), 10, None),
            (candidate("open_swe_traces_qwen35", "openswe"), 10, None),
        ]

        capped = cap_external_share(selected)

        self.assertEqual(len(capped), 5)
        self.assertEqual(
            sum(item[0].row["source"] in {"swe_train_oracle_edit_trace", "open_swe_traces_qwen35"} for item in capped),
            1,
        )

    def test_prefers_fixture_backed_edit_sources_and_caps_external_diversity(self):
        accepted = [
            (candidate("swe-smith", "smith-a"), 30, {"verifiable": True}),
            (candidate("swe-smith", "smith-b"), 10, {"verifiable": True}),
            (candidate("kwai_klear_miniswe", "kwai-a"), 20, {"verifiable": True}),
            (candidate("smoke:run", "lite-a"), 25, {"verifiable": True}),
            (candidate("swe-smith", "unverified"), 1, {"verifiable": False}),
            (candidate("swe_train_oracle_edit_trace", "oracle-a"), 5, None),
            (candidate("open_swe_traces_qwen35", "openswe-a"), 6, None),
        ]

        selected = select_v3_rows(
            accepted,
            target=5,
            fixture_caps={"swe-smith": 2, "kwai": 1, "smoke": 1},
            external_caps={"oracle": 1, "openswe": 0},
        )

        self.assertEqual(len(selected), 5)
        sources = [item[0].row["source"] for item in selected]
        self.assertIn("swe-smith", sources)
        self.assertIn("kwai_klear_miniswe", sources)
        self.assertIn("smoke:run", sources)
        self.assertIn("swe_train_oracle_edit_trace", sources)
        self.assertNotIn("open_swe_traces_qwen35", sources)
        self.assertEqual(sum(source in {"swe_train_oracle_edit_trace", "open_swe_traces_qwen35"} for source in sources), 1)
