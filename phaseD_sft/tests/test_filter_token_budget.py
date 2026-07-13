import json
import tempfile
import unittest
from pathlib import Path

from phaseD_sft.filter_token_budget import filter_jsonl_by_budget


def char_tokens(text: str) -> int:
    return len(text)


class FilterTokenBudgetTests(unittest.TestCase):
    def test_filter_jsonl_splits_rows_by_budget_without_mutating_rows(self):
        small = {"id": "small", "messages": [{"role": "assistant", "content": "aaaa"}]}
        large = {"id": "large", "messages": [{"role": "assistant", "content": "aaaaaaaaaa"}]}

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            accepted = Path(td) / "accepted.jsonl"
            rejected = Path(td) / "rejected.jsonl"
            source.write_text(
                json.dumps(small) + "\n" + json.dumps(large) + "\n{bad json\n",
                encoding="utf-8",
            )

            manifest = filter_jsonl_by_budget(source, accepted, rejected, char_tokens, max_tokens=5)
            accepted_rows = [json.loads(line) for line in accepted.read_text(encoding="utf-8").splitlines()]
            rejected_rows = [json.loads(line) for line in rejected.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(accepted_rows, [small])
        self.assertEqual(rejected_rows, [large])
        self.assertEqual(manifest["rows_read"], 2)
        self.assertEqual(manifest["rows_accepted"], 1)
        self.assertEqual(manifest["rows_rejected"], 1)
        self.assertEqual(manifest["malformed_rows"], 1)
        self.assertEqual(manifest["max_tokens"], 5)
        self.assertEqual(manifest["max_accepted_tokens"], 4)
        self.assertEqual(manifest["min_rejected_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
