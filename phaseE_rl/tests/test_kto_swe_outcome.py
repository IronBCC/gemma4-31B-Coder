from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import phaseE_rl.kto_swe_outcome as outcome_kto
from phaseE_rl.kto_swe_outcome import (
    BEHAVIOR_MANIFEST_SCHEMA,
    balanced_kto_weights,
    build_arg_parser,
    build_kto_config,
    main,
    select_behavior_coverage_rows,
    validate_training_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PYTHON_ENV = os.environ.get("TRAIN_PYTHON")
TRAIN_PYTHON = (
    Path(TRAIN_PYTHON_ENV)
    if TRAIN_PYTHON_ENV is not None
    else REPO_ROOT / ".venv-train/bin/python"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _training_contract(
    *,
    row_count: int = 500,
    undesirable_count: int = 150,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    stock_hash = "a" * 64
    tool_hash = "b" * 64
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        prompt = f"prompt-{index}"
        completion = f"completion-{index}"
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "label": index >= undesirable_count,
                "raw_token_count": 10 + index % 7,
                "token_count": 12 + index % 7,
                "provenance_hashes": {
                    "prompt_sha256": _sha256(prompt),
                    "completion_sha256": _sha256(completion),
                    "stock_chat_template_sha256": stock_hash,
                    "tool_schema_sha256": tool_hash,
                },
            }
        )
    manifest: dict[str, object] = {
        "schema": "python_swe_outcome_kto_v1",
        "row_count": row_count,
        "desirable_count": row_count - undesirable_count,
        "undesirable_count": undesirable_count,
        "replay_verified_rows": row_count,
        "immutable_max_tokens": 4096,
        "max_token_count": max(row["token_count"] for row in rows),
        "truncated_rows": 0,
        "render_failures": 0,
        "format_failures": 0,
        "duplicate_rows": 0,
        "evaluation_leakage_hits": 0,
        "format_loss_checked_rows": row_count,
        "format_loss_failures": 0,
        "pre_rendered": True,
        "stock_native_template": True,
        "stock_chat_template_sha256": stock_hash,
        "tool_schema_sha256": tool_hash,
        "output_jsonl_sha256": "c" * 64,
    }
    return rows, manifest


class _FakeKTOConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_select_kto_processing_class_unwraps_callable_nested_text_tokenizer() -> None:
    text_tokenizer = lambda *_args, **_kwargs: {"input_ids": [1]}
    processor = SimpleNamespace(tokenizer=text_tokenizer)

    assert outcome_kto.select_kto_processing_class(processor) is text_tokenizer


def test_select_kto_processing_class_preserves_plain_callable_tokenizer() -> None:
    tokenizer = lambda *_args, **_kwargs: {"input_ids": [1]}

    assert outcome_kto.select_kto_processing_class(tokenizer) is tokenizer


@pytest.mark.parametrize(
    "loaded_processing_class",
    [SimpleNamespace(tokenizer=object()), object()],
    ids=["noncallable-nested-tokenizer", "noncallable-input"],
)
def test_select_kto_processing_class_fails_closed_on_noncallable_contract(
    loaded_processing_class: object,
) -> None:
    with pytest.raises(RuntimeError, match="callable"):
        outcome_kto.select_kto_processing_class(loaded_processing_class)


def test_initial_lora_b_state_accepts_substantial_trainable_exact_zero_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _LoraStateModel(_substantial_lora_b_parameters())

    state = outcome_kto.validate_initial_lora_b_state(
        model,
        tensor_runtime=_ProbeTorch(),
    )

    assert state.tensor_count == 100
    assert state.element_count == 100
    assert json.loads(capsys.readouterr().out) == {
        "event": "lora-b-initial",
        "tensor_count": 100,
        "element_count": 100,
        "nonzero_element_count": 0,
        "max_abs": 0,
    }


def test_initial_lora_b_state_fails_closed_without_substantial_lora_b_set() -> None:
    model = _LoraStateModel([])

    with pytest.raises(RuntimeError, match="at least 100"):
        outcome_kto.validate_initial_lora_b_state(
            model,
            tensor_runtime=_ProbeTorch(),
        )


def test_initial_lora_b_state_fails_closed_on_nontrainable_tensor() -> None:
    parameters = _substantial_lora_b_parameters()
    parameters[7][1].requires_grad = False

    with pytest.raises(RuntimeError, match="trainable"):
        outcome_kto.validate_initial_lora_b_state(
            _LoraStateModel(parameters),
            tensor_runtime=_ProbeTorch(),
        )


def test_initial_lora_b_state_fails_closed_on_nonzero_tensor() -> None:
    parameters = _substantial_lora_b_parameters()
    parameters[3][1].values = (0.25,)

    with pytest.raises(RuntimeError, match="exactly zero"):
        outcome_kto.validate_initial_lora_b_state(
            _LoraStateModel(parameters),
            tensor_runtime=_ProbeTorch(),
        )


def test_updated_lora_b_state_requires_real_nonzero_update_without_snapshotting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parameters = _substantial_lora_b_parameters()
    model = _LoraStateModel(parameters)
    state = outcome_kto.validate_initial_lora_b_state(
        model,
        tensor_runtime=_ProbeTorch(),
    )
    capsys.readouterr()
    parameters[51][1].values = (-0.5,)

    outcome_kto.validate_updated_lora_b_state(
        model,
        state,
        tensor_runtime=_ProbeTorch(),
    )

    assert json.loads(capsys.readouterr().out) == {
        "event": "lora-b-updated",
        "tensor_count": 100,
        "element_count": 100,
        "nonzero_element_count": 1,
        "max_abs": 0.5,
    }


def test_updated_lora_b_state_fails_closed_when_all_tensors_remain_zero() -> None:
    model = _LoraStateModel(_substantial_lora_b_parameters())
    state = outcome_kto.validate_initial_lora_b_state(
        model,
        tensor_runtime=_ProbeTorch(),
    )

    with pytest.raises(RuntimeError, match="at least one nonzero"):
        outcome_kto.validate_updated_lora_b_state(
            model,
            state,
            tensor_runtime=_ProbeTorch(),
        )


@pytest.mark.parametrize("phase", ["initial", "updated"])
def test_lora_b_state_fails_closed_on_nonfinite_tensor(phase: str) -> None:
    parameters = _substantial_lora_b_parameters()
    model = _LoraStateModel(parameters)
    if phase == "initial":
        parameters[0][1].values = (float("nan"),)
        with pytest.raises(RuntimeError, match="finite"):
            outcome_kto.validate_initial_lora_b_state(
                model,
                tensor_runtime=_ProbeTorch(),
            )
    else:
        state = outcome_kto.validate_initial_lora_b_state(
            model,
            tensor_runtime=_ProbeTorch(),
        )
        parameters[0][1].values = (float("inf"),)
        with pytest.raises(RuntimeError, match="finite"):
            outcome_kto.validate_updated_lora_b_state(
                model,
                state,
                tensor_runtime=_ProbeTorch(),
            )


@pytest.mark.parametrize("drift", ["identity", "count", "elements"])
def test_updated_lora_b_state_fails_closed_on_tensor_set_drift(drift: str) -> None:
    parameters = _substantial_lora_b_parameters()
    model = _LoraStateModel(parameters)
    state = outcome_kto.validate_initial_lora_b_state(
        model,
        tensor_runtime=_ProbeTorch(),
    )
    if drift == "identity":
        name, _tensor = parameters[0]
        parameters[0] = (name, _LoraStateTensor())
    elif drift == "count":
        parameters.pop()
    else:
        parameters[0][1].values = (0.0, 0.0)

    with pytest.raises(RuntimeError, match="identity|count|element"):
        outcome_kto.validate_updated_lora_b_state(
            model,
            state,
            tensor_runtime=_ProbeTorch(),
        )


class _ProbeTensor:
    def __init__(
        self,
        values: tuple[float, ...],
        *,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.values = values
        self.shape = shape or (len(values),)
        self.moved_to: object | None = None
        self.requires_grad = False

    def numel(self) -> int:
        return len(self.values)

    def to(self, device: object) -> _ProbeTensor:
        self.moved_to = device
        return self

    def detach(self) -> _ProbeTensor:
        return self

    def clone(self) -> _ProbeTensor:
        return _ProbeTensor(self.values, shape=self.shape)


class _GradProbeTensor(_ProbeTensor):
    def __init__(
        self,
        values: tuple[float, ...],
        *,
        shape: tuple[int, ...],
        requires_grad: bool,
        name: str,
        events: list[str],
    ) -> None:
        super().__init__(values, shape=shape)
        self.requires_grad = requires_grad
        self.name = name
        self.events = events

    def detach(self) -> _GradProbeTensor:
        self.events.append(f"{self.name}-detach")
        return _GradProbeTensor(
            self.values,
            shape=self.shape,
            requires_grad=False,
            name=self.name,
            events=self.events,
        )

    def clone(self) -> _GradProbeTensor:
        self.events.append(f"{self.name}-clone")
        return _GradProbeTensor(
            self.values,
            shape=self.shape,
            requires_grad=self.requires_grad,
            name=self.name,
            events=self.events,
        )


class _ProbeBool:
    def __init__(self, value: bool) -> None:
        self._value = value

    def all(self) -> _ProbeBool:
        return self

    def item(self) -> bool:
        return self._value


class _ProbeScalar:
    def __init__(self, value: int | float) -> None:
        self._value = value

    def item(self) -> int | float:
        return self._value


class _LoraStateTensor:
    def __init__(
        self,
        values: tuple[float, ...] = (0.0,),
        *,
        requires_grad: object = True,
    ) -> None:
        self.values = values
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return len(self.values)

    def detach(self) -> _LoraStateTensor:
        return _LoraStateTensor(self.values, requires_grad=False)

    def abs(self) -> _LoraStateTensor:
        return _LoraStateTensor(
            tuple(abs(value) for value in self.values),
            requires_grad=False,
        )

    def max(self) -> _ProbeScalar:
        return _ProbeScalar(max(self.values))


class _LoraStateModel:
    def __init__(self, parameters: list[tuple[str, _LoraStateTensor]]) -> None:
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


def _substantial_lora_b_parameters(
    *,
    values: tuple[float, ...] = (0.0,),
    requires_grad: object = True,
) -> list[tuple[str, _LoraStateTensor]]:
    return [
        (
            f"model.layers.{index}.self_attn.q_proj.lora_B.default.weight",
            _LoraStateTensor(values, requires_grad=requires_grad),
        )
        for index in range(100)
    ]


class _ProbeTorch:
    def __init__(self) -> None:
        self.inference_entries = 0

    @contextmanager
    def inference_mode(self):
        self.inference_entries += 1
        yield

    @staticmethod
    def isfinite(tensor: _ProbeTensor) -> _ProbeBool:
        return _ProbeBool(all(value == value and abs(value) != float("inf") for value in tensor.values))

    @staticmethod
    def equal(left: _ProbeTensor, right: _ProbeTensor) -> bool:
        return left.shape == right.shape and left.values == right.values

    @staticmethod
    def count_nonzero(tensor: _LoraStateTensor) -> _ProbeScalar:
        return _ProbeScalar(sum(value != 0 for value in tensor.values))


class _GradRejectingProbeTorch(_ProbeTorch):
    @staticmethod
    def isfinite(tensor: _GradProbeTensor) -> _ProbeBool:
        if tensor.requires_grad:
            raise RuntimeError("patched isfinite rejects grad-bearing logits")
        return _ProbeTorch.isfinite(tensor)


class _ProbeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.input_ids = _ProbeTensor((1.0, 2.0), shape=(1, 2))

    def __call__(self, text: str, **kwargs: object) -> dict[str, _ProbeTensor]:
        self.calls.append((text, kwargs))
        return {"input_ids": self.input_ids}


class _KeywordOnlyProbeProcessor(_ProbeTokenizer):
    def __call__(self, *, text: str, **kwargs: object) -> dict[str, _ProbeTensor]:
        return super().__call__(text, **kwargs)


class _ProbeModel:
    device = "cuda:1"

    def __init__(
        self,
        *,
        enabled_logits: _ProbeTensor,
        disabled_logits: _ProbeTensor,
    ) -> None:
        self.training = True
        self._adapter_disabled = False
        self.enabled_logits = enabled_logits
        self.disabled_logits = disabled_logits
        self.forward_training_modes: list[bool] = []
        self.train_calls: list[bool] = []
        self.disable_entries = 0

    def train(self, mode: bool = True) -> _ProbeModel:
        self.training = mode
        self.train_calls.append(mode)
        return self

    def eval(self) -> _ProbeModel:
        return self.train(False)

    def __call__(self, **inputs: object) -> SimpleNamespace:
        assert inputs["input_ids"] is not None
        self.forward_training_modes.append(self.training)
        logits = self.disabled_logits if self._adapter_disabled else self.enabled_logits
        return SimpleNamespace(logits=logits)

    @contextmanager
    def disable_adapter(self):
        self.disable_entries += 1
        self._adapter_disabled = True
        try:
            yield
        finally:
            self._adapter_disabled = False


def test_fresh_lora_reference_probe_checks_exact_logits_and_restores_training_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokenizer = _ProbeTokenizer()
    model = _ProbeModel(
        enabled_logits=_ProbeTensor((1.0, -2.0, 3.0), shape=(1, 1, 3)),
        disabled_logits=_ProbeTensor((1.0, -2.0, 3.0), shape=(1, 1, 3)),
    )
    tensor_runtime = _ProbeTorch()

    outcome_kto.verify_fresh_lora_reference_equality(
        model,
        tokenizer,
        tensor_runtime=tensor_runtime,
    )

    assert model.training is True
    assert model.train_calls == [False, True]
    assert model.forward_training_modes == [False, False]
    assert model.disable_entries == 1
    assert tensor_runtime.inference_entries == 1
    assert tokenizer.input_ids.moved_to == "cuda:1"
    assert len(tokenizer.calls) == 1
    prompt, tokenizer_kwargs = tokenizer.calls[0]
    assert prompt
    assert tokenizer_kwargs == {"return_tensors": "pt", "add_special_tokens": True}
    event = capsys.readouterr().out.strip()
    assert json.loads(event) == {
        "event": "reference-equality",
        "checked_elements": 3,
        "max_abs_diff": 0,
    }
    assert prompt not in event


def test_fresh_lora_reference_probe_calls_gemma_processor_with_text_keyword() -> None:
    processor = _KeywordOnlyProbeProcessor()
    model = _ProbeModel(
        enabled_logits=_ProbeTensor((1.0,), shape=(1, 1, 1)),
        disabled_logits=_ProbeTensor((1.0,), shape=(1, 1, 1)),
    )

    outcome_kto.verify_fresh_lora_reference_equality(
        model,
        processor,
        tensor_runtime=_ProbeTorch(),
    )

    assert len(processor.calls) == 1
    assert processor.calls[0][0]


def test_fresh_lora_reference_probe_snapshots_each_grad_logit_before_next_forward() -> None:
    events: list[str] = []
    enabled_logits = _GradProbeTensor(
        (1.0, 2.0),
        shape=(1, 1, 2),
        requires_grad=True,
        name="enabled",
        events=events,
    )
    disabled_logits = _GradProbeTensor(
        (1.0, 2.0),
        shape=(1, 1, 2),
        requires_grad=True,
        name="disabled",
        events=events,
    )

    class OrderedProbeModel(_ProbeModel):
        def __call__(self, **inputs: object) -> SimpleNamespace:
            events.append("disabled-forward" if self._adapter_disabled else "enabled-forward")
            return super().__call__(**inputs)

        @contextmanager
        def disable_adapter(self):
            events.append("disable-enter")
            with super().disable_adapter():
                yield

    model = OrderedProbeModel(
        enabled_logits=enabled_logits,
        disabled_logits=disabled_logits,
    )

    outcome_kto.verify_fresh_lora_reference_equality(
        model,
        _ProbeTokenizer(),
        tensor_runtime=_GradRejectingProbeTorch(),
    )

    assert events == [
        "enabled-forward",
        "enabled-detach",
        "enabled-clone",
        "disable-enter",
        "disabled-forward",
        "disabled-detach",
        "disabled-clone",
    ]


@pytest.mark.parametrize("missing", ["detach", "clone"])
def test_fresh_lora_reference_probe_fails_closed_without_logit_snapshot_method(
    missing: str,
) -> None:
    class IncompleteSnapshotTensor(_ProbeTensor):
        if missing == "detach":
            detach = None  # type: ignore[assignment]
        else:
            clone = None  # type: ignore[assignment]

    logits = IncompleteSnapshotTensor((1.0,), shape=(1, 1, 1))
    model = _ProbeModel(enabled_logits=logits, disabled_logits=logits)

    with pytest.raises(RuntimeError, match=missing):
        outcome_kto.verify_fresh_lora_reference_equality(
            model,
            _ProbeTokenizer(),
            tensor_runtime=_ProbeTorch(),
        )

    assert model.training is True


@pytest.mark.parametrize("requires_grad", [None, 0], ids=["missing", "non-boolean"])
def test_fresh_lora_reference_probe_requires_explicit_false_snapshot_grad_flag(
    requires_grad: object,
) -> None:
    class InvalidGradFlagTensor(_ProbeTensor):
        def __init__(self) -> None:
            super().__init__((1.0,), shape=(1, 1, 1))
            if requires_grad is None:
                del self.requires_grad
            else:
                self.requires_grad = requires_grad  # type: ignore[assignment]

        def detach(self) -> InvalidGradFlagTensor:
            return self

        def clone(self) -> InvalidGradFlagTensor:
            return self

    logits = InvalidGradFlagTensor()
    model = _ProbeModel(enabled_logits=logits, disabled_logits=logits)

    with pytest.raises(RuntimeError, match="requires_grad.*boolean|requires_grad.*false"):
        outcome_kto.verify_fresh_lora_reference_equality(
            model,
            _ProbeTokenizer(),
            tensor_runtime=_ProbeTorch(),
        )

    assert model.training is True
    assert model.train_calls == [False, True]


def test_fresh_lora_reference_probe_fails_closed_on_logit_mismatch_and_restores_mode() -> None:
    model = _ProbeModel(
        enabled_logits=_ProbeTensor((1.0, 2.0), shape=(1, 1, 2)),
        disabled_logits=_ProbeTensor((1.0, 2.5), shape=(1, 1, 2)),
    )

    with pytest.raises(RuntimeError, match="exactly equal"):
        outcome_kto.verify_fresh_lora_reference_equality(
            model,
            _ProbeTokenizer(),
            tensor_runtime=_ProbeTorch(),
        )

    assert model.training is True
    assert model.train_calls[-1] is True


@pytest.mark.parametrize(
    ("enabled_logits", "disabled_logits", "message"),
    [
        (
            _ProbeTensor((1.0, 2.0), shape=(1, 1, 2)),
            _ProbeTensor((1.0, 2.0), shape=(1, 2, 1)),
            "matching shapes",
        ),
        (
            _ProbeTensor((1.0, float("nan")), shape=(1, 1, 2)),
            _ProbeTensor((1.0, float("nan")), shape=(1, 1, 2)),
            "finite",
        ),
    ],
)
def test_fresh_lora_reference_probe_fails_closed_on_invalid_logits(
    enabled_logits: _ProbeTensor,
    disabled_logits: _ProbeTensor,
    message: str,
) -> None:
    model = _ProbeModel(
        enabled_logits=enabled_logits,
        disabled_logits=disabled_logits,
    )

    with pytest.raises(RuntimeError, match=message):
        outcome_kto.verify_fresh_lora_reference_equality(
            model,
            _ProbeTokenizer(),
            tensor_runtime=_ProbeTorch(),
        )

    assert model.training is True


def test_fresh_lora_reference_probe_fails_closed_without_disable_adapter() -> None:
    model = SimpleNamespace(training=True, eval=lambda: None, train=lambda _mode: None)

    with pytest.raises(RuntimeError, match="disable_adapter"):
        outcome_kto.verify_fresh_lora_reference_equality(
            model,
            _ProbeTokenizer(),
            tensor_runtime=_ProbeTorch(),
        )


@pytest.mark.parametrize("missing", ["inference_context", "logits"])
def test_fresh_lora_reference_probe_fails_closed_when_runtime_contract_is_unavailable(
    missing: str,
) -> None:
    model = _ProbeModel(
        enabled_logits=_ProbeTensor((1.0,), shape=(1, 1, 1)),
        disabled_logits=_ProbeTensor((1.0,), shape=(1, 1, 1)),
    )
    tensor_runtime: object = _ProbeTorch()
    if missing == "inference_context":
        tensor_runtime = SimpleNamespace(
            isfinite=_ProbeTorch.isfinite,
            equal=_ProbeTorch.equal,
        )
    else:
        class MissingLogitsModel(_ProbeModel):
            def __call__(self, **_inputs: object) -> SimpleNamespace:
                return SimpleNamespace()

        model = MissingLogitsModel(
            enabled_logits=_ProbeTensor((1.0,), shape=(1, 1, 1)),
            disabled_logits=_ProbeTensor((1.0,), shape=(1, 1, 1)),
        )

    with pytest.raises(RuntimeError, match="inference_mode|logits"):
        outcome_kto.verify_fresh_lora_reference_equality(
            model,
            _ProbeTokenizer(),
            tensor_runtime=tensor_runtime,
        )

    assert model.training is True


def test_main_defaults_to_the_live_fresh_lora_reference_probe() -> None:
    parameter = inspect.signature(main).parameters["reference_equality_probe"]

    assert parameter.default is outcome_kto.verify_fresh_lora_reference_equality


def test_parser_fixes_default_length_and_has_no_init_adapter_path() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.max_seq_length == 4096
    assert args.per_device_train_batch_size == 2
    assert args.max_steps == 1
    assert args.save_steps == 1
    assert args.seed == 0
    assert not any(action.dest == "init_adapter" for action in parser._actions)


def test_main_imports_unsloth_before_trl_without_module_level_training_imports() -> None:
    source = inspect.getsource(main)

    assert "from unsloth import FastLanguageModel" in source
    assert source.index("from unsloth") < source.index("from trl")


@pytest.mark.parametrize(
    "unsafe_args",
    [
        ["--max-seq-length", "8192"],
        ["--load-in-4bit", "false"],
        ["--init-adapter", "checkpoint"],
        ["--seed", "1"],
    ],
)
def test_parser_has_no_override_for_immutable_model_safety(
    unsafe_args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(unsafe_args)


def test_balanced_kto_weights_downweights_the_majority_label() -> None:
    assert balanced_kto_weights(350, 150) == pytest.approx((150 / 350, 1.0))


@pytest.mark.parametrize(
    ("desirable_count", "undesirable_count"),
    [(0, 1), (1, 0), (-1, 1), (1, -1), (True, 1), (1, False)],
)
def test_balanced_kto_weights_requires_positive_exact_counts(
    desirable_count: int,
    undesirable_count: int,
) -> None:
    with pytest.raises(ValueError, match="positive exact integers"):
        balanced_kto_weights(desirable_count, undesirable_count)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"per_device_train_batch_size": 1}, "physical batch size"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": 26}, "max_steps"),
        ({"save_steps": 0}, "save_steps"),
        ({"save_steps": 2}, "save_steps"),
        ({"max_steps": 25, "save_steps": 26}, "save_steps"),
        ({"seed": 1}, "seed"),
    ],
)
def test_kto_config_rejects_unsafe_batch_and_step_bounds(
    overrides: dict[str, int],
    message: str,
) -> None:
    kwargs = {
        "output_dir": "out",
        "desirable_count": 350,
        "undesirable_count": 150,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": 1,
        "save_steps": 1,
        "seed": 0,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_kto_config(_FakeKTOConfig, **kwargs)


def test_kto_config_locks_the_audited_trl_024_contract() -> None:
    cfg = build_kto_config(
        _FakeKTOConfig,
        output_dir="adapter-out",
        desirable_count=350,
        undesirable_count=150,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
        max_steps=25,
        save_steps=25,
        seed=0,
    )

    assert cfg.kwargs == {
        "output_dir": "adapter-out",
        "max_length": 4096,
        "max_prompt_length": 4096,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 3,
        "gradient_checkpointing": True,
        "beta": 0.1,
        "learning_rate": 5e-7,
        "warmup_steps": 0,
        "desirable_weight": pytest.approx(150 / 350),
        "undesirable_weight": 1.0,
        "precompute_ref_log_probs": False,
        "dataset_num_proc": 1,
        "remove_unused_columns": False,
        "use_liger_loss": False,
        "dataloader_drop_last": True,
        "logging_steps": 1,
        "report_to": "none",
        "max_steps": 25,
        "save_steps": 25,
        "seed": 0,
    }


@pytest.mark.parametrize(
    ("row_count", "undesirable_count", "message"),
    [
        (499, 150, "at least 500"),
        (500, 149, "at least 150 undesirable"),
        (500, 500, "at least one desirable"),
    ],
)
def test_manifest_enforces_minimum_total_and_undesirable_counts(
    row_count: int,
    undesirable_count: int,
    message: str,
) -> None:
    rows, manifest = _training_contract(
        row_count=row_count,
        undesirable_count=undesirable_count,
    )

    with pytest.raises(ValueError, match=message):
        validate_training_manifest(rows, manifest)


def test_manifest_accepts_the_exact_task5_training_contract() -> None:
    rows, manifest = _training_contract()

    validate_training_manifest(rows, manifest)


def test_manifest_accepts_behavior_categories_with_structural_negatives() -> None:
    rows, manifest = _training_contract(row_count=500, undesirable_count=210)
    categories = []
    for index, row in enumerate(rows):
        if row["label"] is True:
            category = "desirable_correct_patch"
        elif index < 170:
            category = "empty_terminal"
        elif index < 190:
            category = "repeated_read_loop"
        else:
            category = "wrong_nonempty_replay"
        row["provenance"] = {"punishment_category": category}
        categories.append(category)
    counts = {
        category: categories.count(category)
        for category in (
            "empty_terminal",
            "repeated_read_loop",
            "wrong_nonempty_replay",
        )
    }
    manifest.update(
        {
            "schema": BEHAVIOR_MANIFEST_SCHEMA,
            "replay_verified_rows": (
                categories.count("desirable_correct_patch")
                + categories.count("wrong_nonempty_replay")
            ),
            "structural_negative_rows": (
                counts["empty_terminal"] + counts["repeated_read_loop"]
            ),
            "behavior_negative_counts": counts,
            "private_marker_positive_rows": 0,
            "full_evaluation_ids": 707,
            "full_evaluation_repositories": 12,
        }
    )

    validate_training_manifest(rows, manifest)


def test_behavior_coverage_selection_is_balanced_and_deterministic() -> None:
    rows, _manifest = _training_contract(
        row_count=500,
        undesirable_count=210,
    )
    for index, row in enumerate(rows):
        if row["label"] is True:
            category = "desirable_correct_patch"
        elif index < 170:
            category = "empty_terminal"
        elif index < 190:
            category = "repeated_read_loop"
        else:
            category = "wrong_nonempty_replay"
        row["sample_uid"] = f"sample-{index:03d}"
        row["provenance"] = {"punishment_category": category}

    selected, counts = select_behavior_coverage_rows(rows)

    assert len(selected) == 50
    assert counts == {
        "desirable_correct_patch": 25,
        "empty_terminal": 8,
        "repeated_read_loop": 8,
        "wrong_nonempty_replay": 9,
    }
    assert {
        category: sum(
            row["provenance"]["punishment_category"] == category
            for row in selected
        )
        for category in counts
    } == counts
    assert select_behavior_coverage_rows(list(reversed(rows))) == (
        selected,
        counts,
    )


def test_behavior_manifest_rejects_negative_category_sign_reversal() -> None:
    rows, manifest = _training_contract(row_count=500, undesirable_count=210)
    categories = []
    for index, row in enumerate(rows):
        if row["label"] is True:
            category = "desirable_correct_patch"
        elif index < 170:
            category = "empty_terminal"
        elif index < 190:
            category = "repeated_read_loop"
        else:
            category = "wrong_nonempty_replay"
        row["provenance"] = {"punishment_category": category}
        categories.append(category)
    counts = {
        category: categories.count(category)
        for category in (
            "empty_terminal",
            "repeated_read_loop",
            "wrong_nonempty_replay",
        )
    }
    manifest.update(
        {
            "schema": BEHAVIOR_MANIFEST_SCHEMA,
            "replay_verified_rows": (
                categories.count("desirable_correct_patch")
                + categories.count("wrong_nonempty_replay")
            ),
            "structural_negative_rows": (
                counts["empty_terminal"] + counts["repeated_read_loop"]
            ),
            "behavior_negative_counts": counts,
            "private_marker_positive_rows": 0,
            "full_evaluation_ids": 707,
            "full_evaluation_repositories": 12,
        }
    )
    negative_index = next(
        index
        for index, row in enumerate(rows)
        if row["provenance"]["punishment_category"] == "empty_terminal"
    )
    rows[negative_index]["label"] = True

    with pytest.raises(ValueError, match="category disagrees with its label"):
        validate_training_manifest(rows, manifest)


def test_behavior_manifest_rejects_a_positive_private_submission_marker() -> None:
    rows, manifest = _training_contract(row_count=500, undesirable_count=150)
    for row in rows:
        row["provenance"] = {
            "punishment_category": (
                "desirable_correct_patch"
                if row["label"]
                else "wrong_nonempty_replay"
            )
        }
    manifest.update(
        {
            "schema": BEHAVIOR_MANIFEST_SCHEMA,
            "structural_negative_rows": 0,
            "behavior_negative_counts": {
                "empty_terminal": 0,
                "repeated_read_loop": 0,
                "wrong_nonempty_replay": 150,
            },
            "private_marker_positive_rows": 1,
            "full_evaluation_ids": 500,
            "full_evaluation_repositories": 12,
        }
    )

    with pytest.raises(ValueError, match="private_marker_positive_rows"):
        validate_training_manifest(rows, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows, _manifest: rows[0].__setitem__("label", 1), "exact booleans"),
        (lambda rows, _manifest: rows[0].__setitem__("prompt", ""), "nonempty"),
        (
            lambda rows, _manifest: rows[0].__setitem__("raw_token_count", 99),
            "raw_token_count",
        ),
        (
            lambda rows, _manifest: rows[0].__setitem__("raw_token_count", False),
            "raw_token_count",
        ),
        (
            lambda rows, _manifest: rows[0].__setitem__("token_count", True),
            "token_count",
        ),
        (
            lambda rows, _manifest: rows[0]["provenance_hashes"].__setitem__(
                "prompt_sha256", "0" * 64
            ),
            "prompt_sha256",
        ),
        (
            lambda rows, _manifest: rows[0]["provenance_hashes"].__setitem__(
                "stock_chat_template_sha256", "c" * 64
            ),
            "stock_chat_template_sha256",
        ),
        (
            lambda rows, _manifest: rows[0]["provenance_hashes"].__setitem__(
                "tool_schema_sha256", "c" * 64
            ),
            "tool_schema_sha256",
        ),
        (
            lambda rows, _manifest: rows[1].__setitem__(
                "completion", rows[0]["completion"]
            ),
            "completion_sha256",
        ),
    ],
)
def test_manifest_recomputes_row_content_and_provenance(
    mutation: object,
    message: str,
) -> None:
    rows, manifest = _training_contract()
    mutation(rows, manifest)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        validate_training_manifest(rows, manifest)


def test_manifest_detects_actual_duplicate_prompt_completion_rows() -> None:
    rows, manifest = _training_contract()
    rows[1]["prompt"] = rows[0]["prompt"]
    rows[1]["completion"] = rows[0]["completion"]
    rows[1]["provenance_hashes"]["prompt_sha256"] = rows[0]["provenance_hashes"][
        "prompt_sha256"
    ]
    rows[1]["provenance_hashes"]["completion_sha256"] = rows[0][
        "provenance_hashes"
    ]["completion_sha256"]

    with pytest.raises(ValueError, match="duplicate prompt/completion"):
        validate_training_manifest(rows, manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "schema"),
        ("row_count", 501, "row_count"),
        ("desirable_count", 351, "desirable_count"),
        ("undesirable_count", 151, "undesirable_count"),
        ("replay_verified_rows", 499, "replay_verified_rows"),
        ("immutable_max_tokens", 8192, "immutable_max_tokens"),
        ("max_token_count", 4097, "max_token_count"),
        ("truncated_rows", 1, "truncated_rows"),
        ("render_failures", 1, "render_failures"),
        ("format_failures", 1, "format_failures"),
        ("duplicate_rows", 1, "duplicate_rows"),
        ("evaluation_leakage_hits", 1, "evaluation_leakage_hits"),
        ("format_loss_checked_rows", 499, "format_loss_checked_rows"),
        ("format_loss_failures", 1, "format_loss_failures"),
        ("pre_rendered", False, "pre_rendered"),
        ("stock_native_template", False, "stock_native_template"),
        ("stock_chat_template_sha256", "", "stock_chat_template_sha256"),
        ("tool_schema_sha256", "", "tool_schema_sha256"),
        ("output_jsonl_sha256", "", "output_jsonl_sha256"),
    ],
)
def test_manifest_rejects_drift_from_the_frozen_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    rows, manifest = _training_contract()
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        validate_training_manifest(rows, manifest)


def test_main_builds_one_fresh_4bit_lora_and_saves_only_the_adapter(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    rows, manifest = _training_contract()
    calls: dict[str, object] = {}
    ordering: list[str] = []
    initial_lora_b_state = object()
    canonical_out = tmp_path / "adapter"
    quarantine_out = tmp_path / "adapter.inprogress"

    class Model:
        def save_pretrained(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("main must never save the base model directly")

        def merge_and_unload(self) -> None:
            raise AssertionError("main must never merge or unload the adapter")

    model = Model()
    text_tokenizer = lambda *_args, **_kwargs: {"input_ids": [1]}
    tokenizer = SimpleNamespace(tokenizer=text_tokenizer)

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs: object) -> tuple[object, object]:
            calls["from_pretrained"] = kwargs
            return model, tokenizer

        @staticmethod
        def get_peft_model(received_model: object, **kwargs: object) -> object:
            calls["get_peft_model"] = (received_model, kwargs)
            return received_model

    class Dataset:
        @staticmethod
        def from_list(received_rows: list[dict[str, object]]) -> object:
            calls["dataset_rows"] = received_rows
            return "dataset"

    class KTOConfig(_FakeKTOConfig):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            calls["config"] = kwargs

    class KTOTrainer:
        def __init__(self, **kwargs: object) -> None:
            assert calls.get("reference_probe") == (model, tokenizer)
            assert ordering == ["initial-lora-b", "reference"]
            ordering.append("trainer-init")
            calls["trainer_init"] = kwargs
            self.callbacks = kwargs["callbacks"]
            self.output_dir = Path(kwargs["args"].kwargs["output_dir"])

        def train(self) -> None:
            assert self.output_dir == quarantine_out
            assert not canonical_out.exists()
            self.output_dir.mkdir()
            (self.output_dir / "checkpoint-1").mkdir()
            ordering.append("train")
            calls["train"] = int(calls.get("train", 0)) + 1
            state = SimpleNamespace(global_step=1, max_steps=1)
            self.callbacks[0].on_log(
                None,
                state,
                None,
                logs={"loss": 0.25, "grad_norm": 0.5},
            )
            self.callbacks[0].on_log(
                None,
                state,
                None,
                logs={"train_loss": 0.25},
            )

        def save_model(self, output_dir: str) -> None:
            assert ordering[-1] == "updated-lora-b"
            assert Path(output_dir) == quarantine_out
            assert not canonical_out.exists()
            (quarantine_out / "adapter_model.safetensors").write_text("adapter")
            ordering.append("save")
            calls.setdefault("save_model", []).append(output_dir)

    runtime = SimpleNamespace(
        FastLanguageModel=FastLanguageModel,
        Dataset=Dataset,
        KTOConfig=KTOConfig,
        KTOTrainer=KTOTrainer,
        gpu_peak_bytes=lambda: 1234,
    )

    def reference_equality_probe(received_model: object, received_tokenizer: object) -> None:
        assert ordering == ["initial-lora-b"]
        ordering.append("reference")
        calls["reference_probe"] = (received_model, received_tokenizer)

    def initial_lora_b_probe(received_model: object) -> object:
        assert received_model is model
        ordering.append("initial-lora-b")
        return initial_lora_b_state

    def updated_lora_b_probe(received_model: object, received_state: object) -> None:
        assert received_model is model
        assert received_state is initial_lora_b_state
        assert ordering[-1] == "train"
        ordering.append("updated-lora-b")

    assert (
        main(
            [
                "--data",
                "rows.jsonl",
                "--manifest",
                "manifest.json",
                "--out",
                str(canonical_out),
            ],
            runtime_loader=lambda: runtime,
            row_loader=lambda _path: rows,
            manifest_loader=lambda _path: manifest,
            data_hash_loader=lambda _path: "c" * 64,
            reference_equality_probe=reference_equality_probe,
            initial_lora_b_probe=initial_lora_b_probe,
            updated_lora_b_probe=updated_lora_b_probe,
        )
        == 0
    )

    assert calls["from_pretrained"] == {
        "model_name": build_arg_parser().parse_args([]).base,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "full_finetuning": False,
    }
    received_model, lora_kwargs = calls["get_peft_model"]
    assert received_model is model
    assert lora_kwargs == {
        "r": 32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_alpha": 32,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 0,
    }
    assert calls["dataset_rows"] is rows
    assert calls["reference_probe"] == (model, tokenizer)
    assert calls["train"] == 1
    assert calls["save_model"] == [str(quarantine_out)]
    assert calls["config"]["output_dir"] == str(quarantine_out)
    assert canonical_out.is_dir()
    assert (canonical_out / "checkpoint-1").is_dir()
    assert (canonical_out / "adapter_model.safetensors").is_file()
    evidence = json.loads(
        (canonical_out / "training_evidence.json").read_text()
    )
    assert evidence["optimizer_steps"] == 1
    assert evidence["coverage_required"] is False
    assert evidence["source_data_sha256"] == "c" * 64
    assert evidence["source_manifest_sha256"] == "c" * 64
    assert not quarantine_out.exists()
    assert ordering == [
        "initial-lora-b",
        "reference",
        "trainer-init",
        "train",
        "updated-lora-b",
        "save",
    ]

    trainer_kwargs = calls["trainer_init"]
    assert trainer_kwargs["model"] is model
    assert trainer_kwargs["ref_model"] is None
    assert trainer_kwargs["train_dataset"] == "dataset"
    assert trainer_kwargs["processing_class"] is text_tokenizer
    assert "peft_config" not in trainer_kwargs
    assert len(trainer_kwargs["callbacks"]) == 1
    assert calls["config"]["precompute_ref_log_probs"] is False

    output = capsys.readouterr().out
    assert '"event": "model-ready"' in output
    assert '"model_load_seconds"' in output
    assert '"event": "data-ready"' in output
    assert '"data_validation_seconds"' in output
    assert output.count('"event": "optimizer-step"') == 1
    assert '"elapsed_training_seconds"' in output
    assert '"seconds_per_step_avg"' in output
    assert '"eta_seconds"' in output
    assert '"estimated_training_total_seconds"' in output
    assert '"loss": 0.25' in output
    assert '"grad_norm": 0.5' in output
    assert '"gpu_peak_bytes": 1234' in output
    assert '"event": "adapter-save"' in output
    assert f'"output_dir": "{canonical_out}"' in output
    assert '"total_wall_seconds"' in output


def test_main_trains_exactly_one_balanced_behavior_coverage_epoch(
    tmp_path: Path,
) -> None:
    rows, manifest = _training_contract(
        row_count=500,
        undesirable_count=210,
    )
    categories = []
    for index, row in enumerate(rows):
        if row["label"] is True:
            category = "desirable_correct_patch"
        elif index < 170:
            category = "empty_terminal"
        elif index < 190:
            category = "repeated_read_loop"
        else:
            category = "wrong_nonempty_replay"
        row["sample_uid"] = f"sample-{index:03d}"
        row["provenance"] = {"punishment_category": category}
        categories.append(category)
    counts = {
        category: categories.count(category)
        for category in (
            "empty_terminal",
            "repeated_read_loop",
            "wrong_nonempty_replay",
        )
    }
    manifest.update(
        {
            "schema": BEHAVIOR_MANIFEST_SCHEMA,
            "replay_verified_rows": (
                categories.count("desirable_correct_patch")
                + counts["wrong_nonempty_replay"]
            ),
            "structural_negative_rows": (
                counts["empty_terminal"] + counts["repeated_read_loop"]
            ),
            "behavior_negative_counts": counts,
            "private_marker_positive_rows": 0,
            "full_evaluation_ids": 707,
            "full_evaluation_repositories": 12,
        }
    )
    calls: dict[str, object] = {}
    output = tmp_path / "coverage-adapter"

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**_kwargs: object) -> tuple[object, object]:
            return object(), lambda *_args, **_kwargs: {"input_ids": [1]}

        @staticmethod
        def get_peft_model(model: object, **_kwargs: object) -> object:
            return model

    class Dataset:
        @staticmethod
        def from_list(selected: list[dict[str, object]]) -> object:
            calls["rows"] = selected
            return "dataset"

    class Trainer:
        def __init__(self, **kwargs: object) -> None:
            self.output = Path(kwargs["args"].kwargs["output_dir"])
            self.callbacks = kwargs["callbacks"]

        def train(self) -> None:
            self.output.mkdir()
            self.state = SimpleNamespace(global_step=25)
            state = SimpleNamespace(global_step=25, max_steps=25)
            self.callbacks[0].on_log(
                None,
                state,
                None,
                logs={"loss": 0.1},
            )

        def save_model(self, output_dir: str) -> None:
            Path(output_dir, "adapter_model.safetensors").write_bytes(
                b"adapter"
            )

    runtime = SimpleNamespace(
        FastLanguageModel=FastLanguageModel,
        Dataset=Dataset,
        KTOConfig=_FakeKTOConfig,
        KTOTrainer=Trainer,
        gpu_peak_bytes=lambda: 0,
    )

    assert main(
        [
            "--data",
            "rows.jsonl",
            "--manifest",
            "manifest.json",
            "--out",
            str(output),
            "--max-steps",
            "25",
            "--save-steps",
            "25",
            "--require-behavior-coverage",
        ],
        runtime_loader=lambda: runtime,
        row_loader=lambda _path: rows,
        manifest_loader=lambda _path: manifest,
        data_hash_loader=lambda _path: "c" * 64,
        reference_equality_probe=lambda _model, _tokenizer: None,
        initial_lora_b_probe=lambda _model: object(),
        updated_lora_b_probe=lambda _model, _state: None,
    ) == 0

    selected = calls["rows"]
    assert len(selected) == 50
    evidence = json.loads(
        (output / "training_evidence.json").read_text()
    )
    assert evidence["coverage_required"] is True
    assert evidence["training_rows"] == 50
    assert evidence["coverage_counts"] == {
        "desirable_correct_patch": 25,
        "empty_terminal": 8,
        "repeated_read_loop": 8,
        "wrong_nonempty_replay": 9,
    }
    assert evidence["optimizer_steps"] == 25


@pytest.mark.parametrize("existing", ["canonical", "quarantine"])
def test_main_fails_before_runtime_or_model_load_on_output_collision(
    tmp_path: Path,
    existing: str,
) -> None:
    rows, manifest = _training_contract()
    canonical_out = tmp_path / "adapter"
    quarantine_out = tmp_path / "adapter.inprogress"
    (canonical_out if existing == "canonical" else quarantine_out).mkdir()

    def never_load_runtime() -> object:
        raise AssertionError("output collision reached runtime/model loading")

    with pytest.raises(FileExistsError, match=existing):
        main(
            [
                "--data",
                "rows.jsonl",
                "--manifest",
                "manifest.json",
                "--out",
                str(canonical_out),
            ],
            runtime_loader=never_load_runtime,
            row_loader=lambda _path: rows,
            manifest_loader=lambda _path: manifest,
            data_hash_loader=lambda _path: "c" * 64,
        )


@pytest.mark.parametrize("failure_point", ["post-update-gate", "explicit-save"])
def test_main_never_publishes_canonical_output_after_gate_or_save_failure(
    tmp_path: Path,
    failure_point: str,
) -> None:
    rows, manifest = _training_contract()
    canonical_out = tmp_path / "adapter"
    quarantine_out = tmp_path / "adapter.inprogress"
    model = object()
    tokenizer = lambda *_args, **_kwargs: {"input_ids": [1]}

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**_kwargs: object) -> tuple[object, object]:
            return model, tokenizer

        @staticmethod
        def get_peft_model(received_model: object, **_kwargs: object) -> object:
            return received_model

    class Dataset:
        @staticmethod
        def from_list(_rows: list[dict[str, object]]) -> object:
            return "dataset"

    class KTOTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.output_dir = Path(kwargs["args"].kwargs["output_dir"])

        def train(self) -> None:
            self.output_dir.mkdir()
            (self.output_dir / "checkpoint-1").mkdir()
            self.state = SimpleNamespace(global_step=1)

        def save_model(self, output_dir: str) -> None:
            assert Path(output_dir) == quarantine_out
            (quarantine_out / "partial-save").write_text("forensic")
            if failure_point == "explicit-save":
                raise RuntimeError("save failed")

    runtime = SimpleNamespace(
        FastLanguageModel=FastLanguageModel,
        Dataset=Dataset,
        KTOConfig=_FakeKTOConfig,
        KTOTrainer=KTOTrainer,
        gpu_peak_bytes=lambda: 0,
    )

    def updated_probe(_model: object, _state: object) -> None:
        if failure_point == "post-update-gate":
            raise RuntimeError("post-update gate failed")

    with pytest.raises(RuntimeError, match="gate failed|save failed"):
        main(
            [
                "--data",
                "rows.jsonl",
                "--manifest",
                "manifest.json",
                "--out",
                str(canonical_out),
            ],
            runtime_loader=lambda: runtime,
            row_loader=lambda _path: rows,
            manifest_loader=lambda _path: manifest,
            data_hash_loader=lambda _path: "c" * 64,
            reference_equality_probe=lambda _model, _tokenizer: None,
            initial_lora_b_probe=lambda _model: object(),
            updated_lora_b_probe=updated_probe,
        )

    assert not canonical_out.exists()
    assert quarantine_out.is_dir()
    assert (quarantine_out / "checkpoint-1").is_dir()
    if failure_point == "post-update-gate":
        assert not (quarantine_out / "partial-save").exists()
    else:
        assert (quarantine_out / "partial-save").is_file()


def test_main_rejects_unsafe_config_before_loading_the_model(tmp_path: Path) -> None:
    rows, manifest = _training_contract()

    class NeverLoadModel:
        @staticmethod
        def from_pretrained(**_kwargs: object) -> tuple[object, object]:
            raise AssertionError("unsafe config reached the model loader")

    runtime = SimpleNamespace(
        FastLanguageModel=NeverLoadModel,
        Dataset=object(),
        KTOConfig=_FakeKTOConfig,
        KTOTrainer=object(),
        gpu_peak_bytes=lambda: 0,
    )

    with pytest.raises(ValueError, match="max_steps"):
        main(
            [
                "--data",
                "rows.jsonl",
                "--manifest",
                "manifest.json",
                "--max-steps",
                "26",
                "--out",
                str(tmp_path / "unused-unsafe-config-output"),
            ],
            runtime_loader=lambda: runtime,
            row_loader=lambda _path: rows,
            manifest_loader=lambda _path: manifest,
            data_hash_loader=lambda _path: "c" * 64,
        )


def test_main_rejects_data_hash_mismatch_before_loading_the_model(
    tmp_path: Path,
) -> None:
    rows, manifest = _training_contract()

    def never_load_runtime() -> object:
        raise AssertionError("mismatched data reached the model loader")

    with pytest.raises(ValueError, match="output_jsonl_sha256"):
        main(
            [
                "--data",
                "rows.jsonl",
                "--manifest",
                "manifest.json",
                "--out",
                str(tmp_path / "unused-hash-mismatch-output"),
            ],
            runtime_loader=never_load_runtime,
            row_loader=lambda _path: rows,
            manifest_loader=lambda _path: manifest,
            data_hash_loader=lambda _path: "d" * 64,
        )


class KtoImportTests(unittest.TestCase):
    def test_unsloth_then_kto_imports_without_optional_judge_crash(self) -> None:
        if TRAIN_PYTHON_ENV is None and not TRAIN_PYTHON.exists():
            self.skipTest(f"default local training interpreter absent: {TRAIN_PYTHON}")
        completed = subprocess.run(
            [
                str(TRAIN_PYTHON),
                "-c",
                (
                    "import unsloth\n"
                    "from trl import KTOConfig, KTOTrainer\n"
                    "import trl.trainer.judges\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
