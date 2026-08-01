import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phaseD_sft import train_rust_lora as trainer_module
from phaseD_sft.train_rust_lora import (
    assert_training_dataset_admitted,
    configure_attention_implementation,
    label_tokens_for_spans,
    rendered_assistant_turn_spans,
    supervised_assistant_turn_spans,
)


class TrainRustLoraLossMaskTests(unittest.TestCase):
    def test_schema_v2_teacher_dataset_requires_bound_passing_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "teacher"
            dataset.mkdir()
            train_jsonl = dataset / "train.jsonl"
            train_jsonl.write_text('{"instance_id":"one"}\n', encoding="utf-8")
            digest = __import__("hashlib").sha256(train_jsonl.read_bytes()).hexdigest()
            manifest = {
                "schema_version": 2,
                "rendered": 1,
                "training_admitted": 0,
                "all_training_gates_complete": False,
                "train_jsonl_sha256": digest,
                "standard_native_format_loss_gate": {
                    "status": "pending_full_dataset_verification",
                    "failure_count": None,
                },
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "admission"):
                assert_training_dataset_admitted(dataset)

            manifest.update(
                training_admitted=1,
                all_training_gates_complete=True,
                standard_native_format_loss_gate={
                    "status": "passed",
                    "failure_count": 0,
                },
            )
            (dataset / "manifest.json").write_text(json.dumps(manifest))

            assert_training_dataset_admitted(dataset)

    def test_schema_v2_reasoned_fable_dataset_requires_its_bound_full_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "teacher"
            dataset.mkdir()
            train_jsonl = dataset / "train.jsonl"
            train_jsonl.write_text('{"instance_id":"one"}\n', encoding="utf-8")
            digest = __import__("hashlib").sha256(train_jsonl.read_bytes()).hexdigest()
            report = dataset / "format-gate.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = {
                "schema_version": 2,
                "dataset_variant": "teacher_train_mix_v2p11_fable_reasoned_v1",
                "complete": True,
                "rendered": 1,
                "training_admitted": 1,
                "all_training_gates_complete": True,
                "train_jsonl_sha256": digest,
                "evaluation_overlap": 0,
                "fable_rows": 92,
                "canonical_bash_tool_turns": 517,
                "reasoned_tool_turns": 468,
                "structural_gates": {
                    "all_fable_supervised_tool_calls_are_canonical_bash": True,
                    "all_messages_have_boolean_loss": True,
                    "evaluation_overlap": 0,
                    "reasoned_tool_turns": 468,
                },
                "format_loss_gate": {
                    "status": "passed_full_dataset_verification",
                    "failure_count": 0,
                    "samples": 1,
                    "report": {
                        "path": str(report),
                        "sha256": __import__("hashlib").sha256(report.read_bytes()).hexdigest(),
                    },
                },
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest))

            assert_training_dataset_admitted(dataset)

            manifest["format_loss_gate"]["report"]["sha256"] = "0" * 64
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "admission"):
                assert_training_dataset_admitted(dataset)

    def test_schema_v2_loader_reads_bound_jsonl_not_unbound_arrow(self):
        from datasets import Dataset

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "teacher"
            Dataset.from_list([{
                "instance_id": "unbound-arrow",
                "messages": [{"role": "user", "content": "wrong"}],
            }]).save_to_disk(dataset)
            bound = {
                "instance_id": "bound-jsonl",
                "messages": [{"role": "user", "content": "right"}],
            }
            (dataset / "train.jsonl").write_text(json.dumps(bound) + "\n")
            (dataset / "manifest.json").write_text(json.dumps({
                "schema_version": 2,
            }))

            loaded = trainer_module.load_training_dataset(dataset)

            self.assertEqual(loaded.to_list(), [bound])

    def test_load_training_dataset_accepts_jsonl_file(self):
        rows = [
            {"instance_id": "one", "messages": [{"role": "user", "content": "first"}]},
            {"instance_id": "two", "messages": [{"role": "user", "content": "second"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mix"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loader = getattr(trainer_module, "load_training_dataset", None)

            self.assertIsNotNone(loader, "shared dataset loader is missing")
            loaded = loader(path)

        self.assertEqual(loaded.to_list(), rows)

    def test_load_training_dataset_preserves_hf_save_to_disk_behavior(self):
        from datasets import Dataset

        rows = [
            {"instance_id": "hf-one", "messages": [{"role": "user", "content": "first"}]},
            {"instance_id": "hf-two", "messages": [{"role": "user", "content": "second"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hf-dataset"
            Dataset.from_list(rows).save_to_disk(path)
            loader = getattr(trainer_module, "load_training_dataset", None)

            self.assertIsNotNone(loader, "shared dataset loader is missing")
            loaded = loader(path)

        self.assertEqual(loaded.to_list(), rows)

    def test_load_training_dataset_rejects_missing_path_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-data"
            loader = getattr(trainer_module, "load_training_dataset", None)

            self.assertIsNotNone(loader, "shared dataset loader is missing")
            with self.assertRaisesRegex(FileNotFoundError, "dataset path does not exist"):
                loader(missing)

    def test_trainer_uses_shared_dataset_loader(self):
        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("ds = load_training_dataset(a.data)", source)
        self.assertNotIn("ds = load_from_disk(a.data)", source)

    def test_tokenizer_preprocessing_does_not_spawn_worker_processes(self):
        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        process_counts = []
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            for keyword in call.keywords:
                if keyword.arg in {"num_proc", "dataset_num_proc"}:
                    process_counts.append(ast.literal_eval(keyword.value))

        self.assertTrue(process_counts)
        self.assertEqual(process_counts, [1] * len(process_counts))

    def test_trainer_emits_microstep_step_and_eta_progress(self):
        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")

        self.assertIn("class TrainingProgressCallback(TrainerCallback):", source)
        self.assertIn("def on_substep_end", source)
        self.assertIn("def on_step_end", source)
        self.assertIn("event=optimizer_step", source)
        self.assertIn("eta_seconds=", source)
        self.assertIn("epoch_seconds=", source)
        self.assertIn("event=data_ready", source)

    def test_resume_restores_exact_dataset_position(self):
        policy = getattr(trainer_module, "ignore_data_skip_for_run", None)

        self.assertIsNotNone(policy, "resume data-position policy is missing")
        self.assertTrue(policy(resume=False))
        self.assertFalse(policy(resume=True))

    def test_requested_save_cadence_overrides_restored_trainer_state(self):
        policy = getattr(trainer_module, "should_force_checkpoint", None)

        self.assertIsNotNone(policy, "absolute checkpoint cadence policy is missing")
        self.assertFalse(policy(global_step=0, save_steps=4))
        self.assertTrue(policy(global_step=64, save_steps=4))
        self.assertFalse(policy(global_step=65, save_steps=4))
        self.assertTrue(policy(global_step=68, save_steps=4))

        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")
        self.assertIn("control.should_save = True", source)

    def test_cuda_cache_release_preserves_live_allocations_and_releases_unused_reservations(self):
        release = getattr(trainer_module, "release_cuda_cache", None)
        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated():
                return 70

            @staticmethod
            def memory_reserved():
                return 90 if not calls else 74

            @staticmethod
            def empty_cache():
                calls.append("empty_cache")

        fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})()

        self.assertIsNotNone(release, "CUDA cache release policy is missing")
        self.assertEqual(
            release(fake_torch),
            {
                "allocated_bytes": 70,
                "reserved_before_bytes": 90,
                "reserved_after_bytes": 74,
            },
        )
        self.assertEqual(calls, ["empty_cache"])

        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("release_cuda_cache(torch)"), 2)
        self.assertIn("event=cuda_cache_release", source)

    def test_gradient_checkpointing_mode_can_disable_unsloth_host_offload(self):
        resolve = getattr(trainer_module, "resolve_gradient_checkpointing", None)

        self.assertIsNotNone(resolve, "gradient-checkpointing policy is missing")
        self.assertEqual(resolve("unsloth"), "unsloth")
        self.assertEqual(resolve("bounded_unsloth"), "unsloth")
        self.assertIs(resolve("standard"), True)
        self.assertEqual(resolve("hybrid"), "unsloth")
        with self.assertRaisesRegex(ValueError, "gradient checkpointing"):
            resolve("invalid")

        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('use_gradient_checkpointing=gradient_checkpointing', source)

    def test_bounded_unsloth_recycles_exact_host_pool_as_pageable_memory(self):
        recycle = getattr(
            trainer_module,
            "recycle_bounded_unsloth_host_buffers",
            None,
        )
        calls = []

        class FakeStorage:
            def __init__(self, size):
                self._size = size

            def nbytes(self):
                return self._size

        class FakeTensor:
            dtype = "bfloat16"

            def __init__(self, size, *, pinned):
                self._storage = FakeStorage(size)
                self._pinned = pinned

            def untyped_storage(self):
                return self._storage

            def is_pinned(self):
                return self._pinned

        class FakeC:
            @staticmethod
            def _host_emptyCache():
                calls.append("host_empty_cache")

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize():
                calls.append("cuda_synchronize")

        class FakeTorch:
            _C = FakeC
            cuda = FakeCuda

            @staticmethod
            def empty(numel, *, dtype, device):
                calls.append(("empty", numel, dtype, device))
                return FakeTensor(numel * 2, pinned=False)

        checkpointing = type(
            "GradientCheckpointing",
            (),
            {
                "INITIAL_CPU_BUFFER_COUNT": 200,
                "INITIAL_CPU_BUFFER_SIZE": 128 * 1024,
                "CPU_BUFFERS": [
                    FakeTensor(256 * 1024, pinned=True)
                    for _ in range(200)
                ],
                "CPU_INDEX": 60,
                "BACKWARD_PASS": True,
            },
        )()

        self.assertIsNotNone(recycle, "bounded Unsloth host-buffer policy is missing")
        observed = recycle(FakeTorch, checkpointing)

        self.assertEqual(
            observed,
            {
                "buffer_count": 200,
                "initial_buffer_elements": 128 * 1024,
                "retained_bytes": 200 * 256 * 1024,
                "pinned_buffers": 200,
                "pageable_buffers": 200,
                "cuda_synchronized": True,
                "host_cache_drained": True,
            },
        )
        self.assertEqual(calls[0], "cuda_synchronize")
        self.assertEqual(calls.count("host_empty_cache"), 1)
        self.assertEqual(
            calls.count(("empty", 128 * 1024, "bfloat16", "cpu")),
            200,
        )
        self.assertTrue(all(not tensor.is_pinned() for tensor in checkpointing.CPU_BUFFERS))

    def test_bounded_unsloth_fails_closed_outside_completed_backward(self):
        recycle = getattr(
            trainer_module,
            "recycle_bounded_unsloth_host_buffers",
            None,
        )
        checkpointing = type(
            "GradientCheckpointing",
            (),
            {
                "INITIAL_CPU_BUFFER_COUNT": 200,
                "INITIAL_CPU_BUFFER_SIZE": 128 * 1024,
                "CPU_BUFFERS": [object()] * 200,
                "CPU_INDEX": 1,
                "BACKWARD_PASS": False,
            },
        )()

        self.assertIsNotNone(recycle, "bounded Unsloth host-buffer policy is missing")
        with self.assertRaisesRegex(RuntimeError, "completed backward"):
            recycle(object(), checkpointing)

    def test_hybrid_gradient_checkpointing_limits_standard_layers_to_six(self):
        configure = getattr(
            trainer_module,
            "configure_hybrid_gradient_checkpointing",
            None,
        )
        offloaded = lambda *args, **kwargs: None
        standard = lambda *args, **kwargs: None
        layer_type = type("Gemma4TextDecoderLayer", (), {})
        layers = []
        for _ in range(60):
            layer = layer_type()
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = offloaded
            layers.append(layer)
        model = type("Model", (), {"modules": lambda self: iter(layers)})()
        checkpoint_module = type(
            "CheckpointModule",
            (),
            {"_unsloth_pristine_checkpoint": staticmethod(standard)},
        )()

        self.assertIsNotNone(configure, "hybrid checkpointing policy is missing")
        observed = configure(model, checkpoint_module, standard_stride=10)

        self.assertEqual(
            observed,
            {
                "text_layers": 60,
                "offloaded_layers": 54,
                "standard_layers": 6,
                "standard_stride": 10,
            },
        )
        self.assertIs(layers[0]._gradient_checkpointing_func.func, standard)
        self.assertTrue(layers[0]._gradient_checkpointing_func.keywords["use_reentrant"])
        self.assertIs(layers[1]._gradient_checkpointing_func, offloaded)
        self.assertIs(layers[9]._gradient_checkpointing_func, offloaded)
        self.assertIs(layers[10]._gradient_checkpointing_func.func, standard)

    def test_inductor_compile_threads_override_unsloth_cached_options(self):
        configure = getattr(trainer_module, "configure_inductor_compile_threads", None)

        class Config:
            compile_threads = 24

        class Common:
            torch_compile_options = {"compile_threads": 24, "epilogue_fusion": True}

            @staticmethod
            def determine_compile_threads():
                return 24

            @staticmethod
            def get_torch_compile_options():
                return {"compile_threads": Common.determine_compile_threads()}

        fake_torch = type(
            "FakeTorch",
            (),
            {"_inductor": type("Inductor", (), {"config": Config})()},
        )()

        self.assertIsNotNone(configure, "bounded Inductor compile policy is missing")
        with mock.patch.dict(
            os.environ,
            {"TORCHINDUCTOR_COMPILE_THREADS": "4"},
            clear=False,
        ):
            observed = configure(fake_torch, Common)

            self.assertEqual(Config.compile_threads, 4)
            self.assertEqual(Common.determine_compile_threads(), 4)
            self.assertEqual(Common.torch_compile_options["compile_threads"], 4)
            self.assertEqual(Common.get_torch_compile_options()["compile_threads"], 4)
            self.assertEqual(
                observed,
                {
                    "torch_inductor_compile_threads": 4,
                    "unsloth_compile_threads": 4,
                },
            )

        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("event=compile_policy", source)
        self.assertIn("unsloth_compile_disabled=", source)
        self.assertLess(
            source.index("compile_policy = configure_inductor_compile_threads("),
            source.index("FastLanguageModel.from_pretrained("),
        )
        self.assertLess(
            source.index('requested_compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"'),
            source.index("from unsloth import FastLanguageModel"),
        )

    def test_hf_flex_routing_debug_is_explicit_opt_in(self):
        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")

        self.assertIn("def install_hf_flex_routing_debug", source)
        self.assertIn("torch.compiler.disable(recursive=False)", source)
        self.assertIn("WrappedFlexAttention._is_flex_compiled", source)
        self.assertIn('"--debug-hf-flex-routing"', source)

    def test_configure_attention_implementation_updates_root_and_nested_configs(self):
        class Config:
            def __init__(self):
                self._attn_implementation = "sdpa"

        class Model:
            def __init__(self):
                self.config = Config()
                self.config.text_config = Config()
                self.base_model = type("Base", (), {"config": Config()})()

        model = Model()

        changed = configure_attention_implementation(model, "flex_attention")

        self.assertEqual(changed, 3)
        self.assertEqual(model.config._attn_implementation, "flex_attention")
        self.assertEqual(model.config.text_config._attn_implementation, "flex_attention")
        self.assertEqual(model.base_model.config._attn_implementation, "flex_attention")

    def test_configure_attention_implementation_is_noop_when_not_requested(self):
        model = type("Model", (), {"config": object()})()

        self.assertEqual(configure_attention_implementation(model, None), 0)

    def test_assistant_turn_span_stops_before_tool_observation_turn(self):
        messages = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "calling pytest"},
            {"role": "tool", "content": "FAILED tests/test_mod.py::test_case"},
            {"role": "user", "content": "fix it"},
        ]
        text = (
            "<bos><|turn>user\nrun tests"
            "<|turn>model\ncalling pytest<|tool_call>{}</tool_call>"
            "<|turn>tool\nFAILED tests/test_mod.py::test_case"
            "<|turn>user\nfix it"
        )

        spans, fallback_used = rendered_assistant_turn_spans(object(), messages, text)

        self.assertFalse(fallback_used)
        self.assertEqual(spans, [(len("<bos><|turn>user\nrun tests"), text.index("<|turn>tool\n"))])

    def test_label_tokens_for_spans_masks_tokens_outside_assistant_spans(self):
        input_ids = [10, 11, 12, 13]
        offsets = [(0, 4), (4, 8), (8, 12), (12, 16)]

        labels, supervised = label_tokens_for_spans(input_ids, offsets, [(4, 12)])

        self.assertEqual(labels, [-100, 11, 12, -100])
        self.assertEqual(supervised, 2)

    def test_selective_assistant_loss_keeps_only_explicit_targets(self):
        messages = [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "", "loss": False},
            {"role": "user", "content": "OBSERVATION:\nsource"},
            {"role": "assistant", "content": "", "loss": True},
        ]
        text = (
            "<bos><|turn>user\nfix it"
            "<|turn>model\n<|tool_call>{}</tool_call>"
            "<|turn>user\nOBSERVATION:\nsource"
            "<|turn>model\n<|tool_call>{}</tool_call>"
        )
        all_spans, _ = rendered_assistant_turn_spans(object(), messages, text)

        spans, fallback_used = supervised_assistant_turn_spans(object(), messages, text)

        self.assertFalse(fallback_used)
        self.assertEqual(spans, [all_spans[1]])

    def test_selective_assistant_loss_defaults_to_all_assistant_turns(self):
        messages = [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        text = (
            "<bos><|turn>user\nfix it"
            "<|turn>model\nfirst"
            "<|turn>model\nsecond"
        )

        self.assertEqual(
            supervised_assistant_turn_spans(object(), messages, text)[0],
            rendered_assistant_turn_spans(object(), messages, text)[0],
        )


if __name__ == "__main__":
    unittest.main()
