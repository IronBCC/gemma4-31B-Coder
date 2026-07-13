import unittest

from phaseD_sft.build_v8_editfirst_mixture import build_v8_mixture


def row(source: str, instance_id: str, tokens: int) -> dict:
    return {
        "source": source,
        "instance_id": instance_id,
        "messages": [{"role": "assistant", "content": "x" * tokens}],
    }


def count_tokens(text: str) -> int:
    return len(text)


class BuildV8EditFirstMixtureTests(unittest.TestCase):
    def test_selects_one_per_repo_before_second_trace_and_keeps_oracle_once(self):
        externals = [
            row("kwai_klear_miniswe", "alpha__repo.one", 30),
            row("kwai_klear_miniswe", "alpha__repo.two", 1),
            row("kwai_klear_miniswe", "beta__repo.one", 20),
            row("open_swe_traces_qwen35", "gamma__repo.one", 10),
        ]
        anchors = [
            row("swe-smith", "smith__one.x", 5),
            row("swe_train_oracle_edit_trace", "oracle__one.x", 5),
            row("coder_repair_synthetic", "repair__one.x", 5),
        ]

        output, manifest = build_v8_mixture(
            externals, anchors, count_tokens, external_cap=3, workers=1
        )

        selected = output[3:]
        self.assertEqual({item["instance_id"] for item in selected}, {
            "alpha__repo.two",
            "beta__repo.one",
            "gamma__repo.one",
        })
        self.assertEqual(
            sum(item["source"] == "swe_train_oracle_edit_trace" for item in output), 1
        )
        self.assertEqual(manifest["external_selection"]["selected"], 3)
        self.assertEqual(manifest["external_selection"]["unique_repos_selected"], 3)
        self.assertEqual(manifest["oracle_edit_upweight_factor"], 1)


if __name__ == "__main__":
    unittest.main()
