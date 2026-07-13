import unittest

from phaseI_deepspeed_longctx.longctx_plan import (
    DEFAULT_FULL_TRACE_DATASET,
    DEFAULT_WINDOWED_DATASET,
    build_deepspeed_config,
    build_stage_ladder,
    dataset_for_max_seq,
    reject_invalid_parallelism,
)


class LongContextPlanTests(unittest.TestCase):
    def test_stage_ladder_starts_small_and_reaches_target(self):
        stages = build_stage_ladder(target_seq=90_112)

        self.assertEqual(stages[0].name, "smoke-4k")
        self.assertEqual(stages[0].max_seq, 4_096)
        self.assertEqual(stages[-1].name, "target-90k")
        self.assertEqual(stages[-1].max_seq, 90_112)
        self.assertTrue(all(a.max_seq < b.max_seq for a, b in zip(stages, stages[1:])))
        self.assertEqual([stage.max_steps for stage in stages[:3]], [1, 2, 2])

    def test_dataset_selection_uses_windowed_until_long_context_is_proven(self):
        self.assertEqual(dataset_for_max_seq(32_768), DEFAULT_WINDOWED_DATASET)
        self.assertEqual(dataset_for_max_seq(65_536), DEFAULT_FULL_TRACE_DATASET)
        self.assertEqual(dataset_for_max_seq(90_112), DEFAULT_FULL_TRACE_DATASET)

    def test_deepspeed_config_uses_sequence_parallelism_and_zero3(self):
        config = build_deepspeed_config(
            sequence_parallel_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
        )

        self.assertEqual(config["sequence_parallel_size"], 2)
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertEqual(
            config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"],
            True,
        )
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 16)
        self.assertEqual(config["bf16"]["enabled"], True)

    def test_rejects_parallelism_that_does_not_extend_sequence_length(self):
        for mode in ("auto", "balanced", "ddp", "fsdp-only"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    reject_invalid_parallelism(mode)

        self.assertEqual(reject_invalid_parallelism("ulysses"), "ulysses")


if __name__ == "__main__":
    unittest.main()
