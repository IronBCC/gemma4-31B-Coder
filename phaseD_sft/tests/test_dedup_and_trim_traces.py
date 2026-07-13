import copy
import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from phaseD_sft.dedup_and_trim_traces import (
    dedup_traces,
    load_rows_from_jsonl,
    process,
    trim_trace,
)


def _make_row(messages: list[dict], instance_id: str = "t1", source: str = "src_a") -> dict:
    return {"messages": copy.deepcopy(messages), "instance_id": instance_id, "source": source}


class TrimTraceTests(unittest.TestCase):
    def test_trailing_user_message_is_dropped(self):
        messages = [
            {"role": "user", "content": "fix this"},
            {"role": "assistant", "content": "ok doing it"},
            {"role": "user", "content": "what about the other part"},
        ]
        row = _make_row(messages)
        kept, stats = trim_trace(row)

        self.assertIsNotNone(kept)
        self.assertEqual(len(kept["messages"]), 2)
        self.assertEqual(kept["messages"][-1]["role"], "assistant")
        self.assertEqual(stats["trailing_trimmed_messages"], 1)

    def test_multiple_trailing_user_messages_dropped(self):
        messages = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        row = _make_row(messages)
        kept, stats = trim_trace(row)

        self.assertIsNotNone(kept)
        self.assertEqual(len(kept["messages"]), 2)
        self.assertEqual(stats["trailing_trimmed_messages"], 2)

    def test_no_assistant_message_drops_trace(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
        row = _make_row(messages)
        kept, stats = trim_trace(row)

        self.assertIsNone(kept)
        self.assertEqual(stats["traces_dropped_no_assistant"], 1)

    def test_single_message_with_assistant_dropped(self):
        messages = [
            {"role": "assistant", "content": "only message"},
        ]
        row = _make_row(messages)
        kept, stats = trim_trace(row)

        self.assertIsNone(kept)
        self.assertEqual(stats["traces_dropped_no_assistant"], 1)

    def test_two_messages_with_assistant_kept(self):
        messages = [
            {"role": "user", "content": "fix this"},
            {"role": "assistant", "content": "ok doing it"},
        ]
        row = _make_row(messages)
        kept, stats = trim_trace(row)

        self.assertIsNotNone(kept)
        self.assertEqual(len(kept["messages"]), 2)


class DedupTracesTests(unittest.TestCase):
    def test_exact_duplicate_content_keeps_first(self):
        messages_a = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "ok"},
        ]
        messages_b = copy.deepcopy(messages_a)

        rows = [_make_row(messages_a, instance_id="t1"), _make_row(messages_b, instance_id="t2")]
        kept, stats = dedup_traces(rows)

        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["exact_dups_removed"], 1)
        # Should keep the first occurrence
        self.assertEqual(kept[0]["instance_id"], "t1")

    def test_same_instance_id_different_messages_preserved(self):
        messages_a = [
            {"role": "user", "content": "fix part A"},
            {"role": "assistant", "content": "ok A"},
        ]
        messages_b = [
            {"role": "user", "content": "fix part B"},
            {"role": "assistant", "content": "ok B"},
        ]

        # Same instance_id but different content (window splits)
        rows = [_make_row(messages_a, instance_id="split-1"), _make_row(messages_b, instance_id="split-1")]
        kept, stats = dedup_traces(rows)

        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["exact_dups_removed"], 0)

    def test_different_content_not_deduplicated(self):
        messages_a = [
            {"role": "user", "content": "fix A"},
            {"role": "assistant", "content": "ok A"},
        ]
        messages_b = [
            {"role": "user", "content": "fix B"},
            {"role": "assistant", "content": "ok B"},
        ]

        rows = [_make_row(messages_a, instance_id="t1"), _make_row(messages_b, instance_id="t2")]
        kept, stats = dedup_traces(rows)

        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["exact_dups_removed"], 0)

    def test_progress_reaches_total_for_passthrough_rows(self):
        rows = [
            _make_row([{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}], source="anchor"),
            _make_row([{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}], source="anchor"),
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            dedup_traces(rows, frozenset({"anchor"}), progress_every=1)

        self.assertIn("dedup-traces 2/2", output.getvalue())

    def test_sort_keys_matters_for_hash(self):
        messages_a = [
            {"content": "fix", "role": "user"},
            {"content": "ok", "role": "assistant"},
        ]
        messages_b = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "ok"},
        ]

        rows = [_make_row(messages_a, instance_id="t1"), _make_row(messages_b, instance_id="t2")]
        kept, stats = dedup_traces(rows)

        # Same content but different key ordering - should still be considered same due to sort_keys=True
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["exact_dups_removed"], 1)


class ProcessEndToEndTests(unittest.TestCase):
    def test_process_jsonl_full_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            output = Path(td) / "out.jsonl"

            rows_data = [
                {
                    "messages": [
                        {"role": "user", "content": "fix"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "instance_id": "t1",
                    "source": "src_a",
                },
                # Duplicate content (different instance_id)
                {
                    "messages": [
                        {"role": "user", "content": "fix"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "instance_id": "t2",
                    "source": "src_b",
                },
                # Trailing user message that should be trimmed
                {
                    "messages": [
                        {"role": "user", "content": "fix C"},
                        {"role": "assistant", "content": "ok C"},
                        {"role": "user", "content": "what about D?"},
                    ],
                    "instance_id": "t3",
                    "source": "src_a",
                },
            ]

            source.write_text(
                "\n".join(json.dumps(r) for r in rows_data), encoding="utf-8"
            )

            manifest = process(source, output)

        self.assertEqual(manifest["rows_in"], 3)
        # After trim: t3 survives (still has assistant at end after trim).
        # After dedup: t1 and t2 are exact dups -> remove t2. t3 is unique.
        self.assertEqual(manifest["rows_out"], 2)
        self.assertEqual(manifest["exact_dups_removed"], 1)
        self.assertEqual(manifest["trailing_trimmed_messages"], 1)
        self.assertEqual(manifest["traces_dropped_no_assistant"], 0)
        self.assertEqual(manifest["unique_content_count"], 2)

    def test_process_jsonl_drops_traces_without_assistant(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            output = Path(td) / "out.jsonl"

            rows_data = [
                {
                    "messages": [
                        {"role": "user", "content": "fix"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "instance_id": "t1",
                    "source": "src_a",
                },
                # Only user messages -> should be dropped
                {
                    "messages": [
                        {"role": "user", "content": "fix only user"},
                    ],
                    "instance_id": "t2",
                    "source": "src_b",
                },
            ]

            source.write_text(
                "\n".join(json.dumps(r) for r in rows_data), encoding="utf-8"
            )

            manifest = process(source, output)

        self.assertEqual(manifest["rows_in"], 2)
        self.assertEqual(manifest["traces_dropped_no_assistant"], 1)


class LoadJsonlTests(unittest.TestCase):
    def test_load_rows_from_jsonl_skips_malformed(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            content = (
                json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
                + "{bad json\n"
                + "\n"
                + json.dumps({"messages": [{"role": "assistant", "content": "hello"}]}) + "\n"
            )
            source.write_text(content, encoding="utf-8")

            rows, stats = load_rows_from_jsonl(source)

        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["malformed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
