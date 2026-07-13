import json
import unittest

from phaseD_sft.build_v7_edit_decisive_mixture import build_v7_mixture


def command(value: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({"command": value})}}],
    }


def row(source: str, commands: list[str]) -> dict:
    return {"source": source, "messages": [command(value) for value in commands]}


class BuildV7EditDecisiveMixtureTests(unittest.TestCase):
    def test_filters_only_externals_and_doubles_oracle_anchor(self):
        rows = [
            row("kwai_klear_miniswe", ["rg target src", "sed -i 's/a/b/' src/mod.py"]),
            row("open_swe_traces_qwen35", [f"pytest -q test_{index}" for index in range(10)] + ["apply_patch <<'PATCH'\nPATCH"]),
            row("swe-smith", ["cat src/mod.py"]),
            row("coder_repair_synthetic", ["cat src/repair.py"]),
            row("swe_train_oracle_edit_trace", ["apply_patch <<'PATCH'\nPATCH"]),
        ]

        output, manifest = build_v7_mixture(rows)

        counts = {source: sum(item["source"] == source for item in output) for source in {
            "kwai_klear_miniswe",
            "open_swe_traces_qwen35",
            "swe-smith",
            "coder_repair_synthetic",
            "swe_train_oracle_edit_trace",
        }}
        self.assertEqual(counts["kwai_klear_miniswe"], 1)
        self.assertEqual(counts["open_swe_traces_qwen35"], 0)
        self.assertEqual(counts["swe-smith"], 1)
        self.assertEqual(counts["coder_repair_synthetic"], 1)
        self.assertEqual(counts["swe_train_oracle_edit_trace"], 2)
        self.assertEqual(manifest["external"]["open_swe_traces_qwen35"]["dropped"], 1)
        self.assertEqual(manifest["oracle_edit_upweight_factor"], 2)


if __name__ == "__main__":
    unittest.main()
