#!/usr/bin/env python3
"""Guarded 4-bit KTO training for replay-verified SWE outcome rows.

Heavy training dependencies are imported only by :func:`main`.  Manifest
validation is intentionally pure: the builder has already recorded TRL's
effective token counts, including missing BOS/EOS insertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMMUTABLE_MAX_TOKENS = 4096
MANIFEST_SCHEMA = "python_swe_outcome_kto_v1"
BEHAVIOR_MANIFEST_SCHEMA = "v2p11_behavior_kto_v2"
MIN_ROWS = 500
MIN_UNDESIRABLE_ROWS = 150
_SHA256_HEX = frozenset("0123456789abcdef")
_MIN_LORA_B_TENSORS = 100
_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
_BEHAVIOR_COVERAGE_QUOTAS = {
    "desirable_correct_patch": 25,
    "empty_terminal": 8,
    "repeated_read_loop": 8,
    "wrong_nonempty_replay": 9,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one fresh 4-bit KTO LoRA on verified outcome rows."
    )
    parser.add_argument("--data", default="data/python_swe_outcome_kto_v1.jsonl")
    parser.add_argument(
        "--manifest",
        default="data/python_swe_outcome_kto_v1_manifest.json",
    )
    parser.add_argument(
        "--base",
        default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it",
    )
    parser.add_argument("--out", default="adapters/python_swe_outcome_kto_v1_smoke")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument(
        "--require-behavior-coverage",
        action="store_true",
    )
    # Deliberately no CLI switches for length, quantization, or an init adapter.
    parser.set_defaults(max_seq_length=IMMUTABLE_MAX_TOKENS, seed=0)
    return parser


def _is_positive_exact_int(value: object) -> bool:
    return type(value) is int and value > 0


def balanced_kto_weights(
    desirable_count: int,
    undesirable_count: int,
) -> tuple[float, float]:
    """Return ``(desirable_weight, undesirable_weight)`` for label balance."""
    if not (
        _is_positive_exact_int(desirable_count)
        and _is_positive_exact_int(undesirable_count)
    ):
        raise ValueError("label counts must be positive exact integers")
    return undesirable_count / desirable_count, 1.0


def select_behavior_coverage_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    buckets = {
        category: []
        for category in _BEHAVIOR_COVERAGE_QUOTAS
    }
    seen_uids: set[str] = set()
    for row in rows:
        provenance = row.get("provenance")
        category = (
            provenance.get("punishment_category")
            if isinstance(provenance, Mapping)
            else None
        )
        uid = row.get("sample_uid")
        if category not in buckets or not isinstance(uid, str) or not uid:
            raise ValueError(
                "behavior coverage requires categorized unique sample_uid rows"
            )
        if uid in seen_uids:
            raise ValueError(
                "behavior coverage requires categorized unique sample_uid rows"
            )
        seen_uids.add(uid)
        buckets[str(category)].append(dict(row))

    selected: list[dict[str, Any]] = []
    for category, quota in _BEHAVIOR_COVERAGE_QUOTAS.items():
        candidates = sorted(
            buckets[category],
            key=lambda row: hashlib.sha256(
                str(row["sample_uid"]).encode()
            ).hexdigest(),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"behavior coverage category {category} has "
                f"{len(candidates)} rows, needs {quota}"
            )
        selected.extend(candidates[:quota])
    selected.sort(
        key=lambda row: hashlib.sha256(
            str(row["sample_uid"]).encode()
        ).hexdigest()
    )
    return selected, dict(_BEHAVIOR_COVERAGE_QUOTAS)


def select_kto_processing_class(loaded_processing_class: Any) -> Any:
    """Select TRL's text-only callable without passing a multimodal processor."""
    missing = object()
    nested_tokenizer = getattr(loaded_processing_class, "tokenizer", missing)
    if nested_tokenizer is not missing:
        if not callable(nested_tokenizer):
            raise RuntimeError("nested KTO tokenizer must be callable")
        return nested_tokenizer
    if not callable(loaded_processing_class):
        raise RuntimeError("KTO processing class must be callable")
    return loaded_processing_class


@dataclass(frozen=True)
class _LoraBTensorRecord:
    name: str
    tensor: Any
    element_count: int


@dataclass(frozen=True)
class LoraBInitialState:
    records: tuple[_LoraBTensorRecord, ...]

    @property
    def tensor_count(self) -> int:
        return len(self.records)

    @property
    def element_count(self) -> int:
        return sum(record.element_count for record in self.records)


def _collect_trainable_lora_b_tensors(model: Any) -> tuple[_LoraBTensorRecord, ...]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("LoRA-B state gate requires named_parameters")
    records: list[_LoraBTensorRecord] = []
    seen_names: set[str] = set()
    seen_tensors: set[int] = set()
    for name, tensor in named_parameters():
        if not isinstance(name, str) or "lora_B" not in name:
            continue
        if name in seen_names or id(tensor) in seen_tensors:
            raise RuntimeError("LoRA-B tensor names and identities must be unique")
        seen_names.add(name)
        seen_tensors.add(id(tensor))
        if getattr(tensor, "requires_grad", None) is not True:
            raise RuntimeError("every LoRA-B tensor must be trainable")
        numel = getattr(tensor, "numel", None)
        if not callable(numel):
            raise RuntimeError("LoRA-B tensor element count is unavailable")
        element_count = numel()
        if type(element_count) is not int or element_count < 1:
            raise RuntimeError("LoRA-B tensor element count must be a positive integer")
        records.append(
            _LoraBTensorRecord(
                name=name,
                tensor=tensor,
                element_count=element_count,
            )
        )
    if len(records) < _MIN_LORA_B_TENSORS:
        raise RuntimeError(
            f"LoRA-B tensor count requires at least {_MIN_LORA_B_TENSORS} tensors"
        )
    return tuple(records)


def _lora_b_metrics(
    records: Sequence[_LoraBTensorRecord],
    *,
    tensor_runtime: Any,
) -> tuple[int, float]:
    isfinite = getattr(tensor_runtime, "isfinite", None)
    count_nonzero = getattr(tensor_runtime, "count_nonzero", None)
    if not callable(isfinite) or not callable(count_nonzero):
        raise RuntimeError("LoRA-B tensor diagnostics are unavailable")
    nonzero_element_count = 0
    max_abs = 0.0
    for record in records:
        detach = getattr(record.tensor, "detach", None)
        if not callable(detach):
            raise RuntimeError("LoRA-B tensor detach is unavailable")
        detached = detach()
        if getattr(detached, "requires_grad", None) is not False:
            raise RuntimeError("detached LoRA-B tensor must not require grad")
        try:
            finite = bool(isfinite(detached).all().item())
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("LoRA-B finite check is unavailable") from error
        if not finite:
            raise RuntimeError("LoRA-B tensors must be finite")
        try:
            tensor_nonzero = count_nonzero(detached).item()
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("LoRA-B nonzero count is unavailable") from error
        if type(tensor_nonzero) is not int or not 0 <= tensor_nonzero <= record.element_count:
            raise RuntimeError("LoRA-B nonzero count must be an exact valid integer")
        nonzero_element_count += tensor_nonzero
        abs_tensor = getattr(detached, "abs", None)
        if not callable(abs_tensor):
            raise RuntimeError("LoRA-B maximum-absolute diagnostic is unavailable")
        absolute = abs_tensor()
        maximum = getattr(absolute, "max", None)
        if not callable(maximum):
            raise RuntimeError("LoRA-B maximum-absolute diagnostic is unavailable")
        try:
            tensor_max_abs = float(maximum().item())
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("LoRA-B maximum-absolute diagnostic is unavailable") from error
        max_abs = max(max_abs, tensor_max_abs)
    return nonzero_element_count, max_abs


def validate_initial_lora_b_state(
    model: Any,
    *,
    tensor_runtime: Any | None = None,
) -> LoraBInitialState:
    """Require a substantial, trainable, finite, exactly-zero fresh LoRA-B set."""
    if tensor_runtime is None:
        import torch as tensor_runtime

    records = _collect_trainable_lora_b_tensors(model)
    nonzero_element_count, max_abs = _lora_b_metrics(
        records,
        tensor_runtime=tensor_runtime,
    )
    if nonzero_element_count != 0 or max_abs != 0.0:
        raise RuntimeError("fresh LoRA-B tensors must be exactly zero")
    state = LoraBInitialState(records=records)
    _emit(
        "lora-b-initial",
        tensor_count=state.tensor_count,
        element_count=state.element_count,
        nonzero_element_count=nonzero_element_count,
        max_abs=max_abs,
    )
    return state


def validate_updated_lora_b_state(
    model: Any,
    initial_state: LoraBInitialState,
    *,
    tensor_runtime: Any | None = None,
) -> None:
    """Require the same LoRA-B tensors to contain a finite nonzero update."""
    if not isinstance(initial_state, LoraBInitialState):
        raise RuntimeError("initial LoRA-B state is unavailable")
    if tensor_runtime is None:
        import torch as tensor_runtime

    current_records = _collect_trainable_lora_b_tensors(model)
    if len(current_records) != initial_state.tensor_count:
        raise RuntimeError("LoRA-B tensor count drifted during training")
    current_by_name = {record.name: record for record in current_records}
    if set(current_by_name) != {record.name for record in initial_state.records}:
        raise RuntimeError("LoRA-B tensor identity set drifted during training")
    for initial_record in initial_state.records:
        current = current_by_name[initial_record.name]
        if current.tensor is not initial_record.tensor:
            raise RuntimeError("LoRA-B tensor identity drifted during training")
        if current.element_count != initial_record.element_count:
            raise RuntimeError("LoRA-B tensor element count drifted during training")

    nonzero_element_count, max_abs = _lora_b_metrics(
        current_records,
        tensor_runtime=tensor_runtime,
    )
    if nonzero_element_count < 1:
        raise RuntimeError("trained LoRA-B tensors require at least one nonzero value")
    _emit(
        "lora-b-updated",
        tensor_count=len(current_records),
        element_count=sum(record.element_count for record in current_records),
        nonzero_element_count=nonzero_element_count,
        max_abs=max_abs,
    )


def _require_exact_int(
    container: Mapping[str, Any],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    value = container.get(field)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer >= {minimum}")
    return value


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be one nonempty lowercase SHA-256")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_manifest(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Validate the immutable Task-5 manifest and its loaded JSONL rows.

    This function never imports a tokenizer or model and never retokenizes.
    It verifies the builder-recorded effective token counts and content hashes.
    """
    schema = manifest.get("schema")
    if schema not in {MANIFEST_SCHEMA, BEHAVIOR_MANIFEST_SCHEMA}:
        raise ValueError(
            f"schema must equal {MANIFEST_SCHEMA} or {BEHAVIOR_MANIFEST_SCHEMA}"
        )
    behavior_schema = schema == BEHAVIOR_MANIFEST_SCHEMA

    row_count = _require_exact_int(manifest, "row_count")
    if row_count != len(rows):
        raise ValueError("manifest row_count must exactly equal loaded rows")

    desirable_manifest = _require_exact_int(manifest, "desirable_count")
    undesirable_manifest = _require_exact_int(manifest, "undesirable_count")
    replay_verified_rows = _require_exact_int(manifest, "replay_verified_rows")
    behavior_negative_counts: Mapping[str, Any] | None = None
    structural_negative_rows = 0
    if behavior_schema:
        structural_negative_rows = _require_exact_int(
            manifest, "structural_negative_rows", minimum=0
        )
        if replay_verified_rows + structural_negative_rows != row_count:
            raise ValueError(
                "replay_verified_rows plus structural_negative_rows must equal row_count"
            )
        behavior_negative_counts = manifest.get("behavior_negative_counts")
        if not isinstance(behavior_negative_counts, Mapping):
            raise ValueError("behavior_negative_counts must be a mapping")
        expected_behavior_keys = {
            "empty_terminal",
            "repeated_read_loop",
            "wrong_nonempty_replay",
        }
        if set(behavior_negative_counts) != expected_behavior_keys:
            raise ValueError("behavior_negative_counts has unexpected categories")
        for key in expected_behavior_keys:
            _require_exact_int(behavior_negative_counts, key, minimum=0)
        if (
            _require_exact_int(
                behavior_negative_counts, "empty_terminal", minimum=0
            )
            + _require_exact_int(
                behavior_negative_counts, "repeated_read_loop", minimum=0
            )
            != structural_negative_rows
        ):
            raise ValueError(
                "structural_negative_rows does not match behavior categories"
            )
        if _require_exact_int(manifest, "private_marker_positive_rows", minimum=0) != 0:
            raise ValueError("private_marker_positive_rows must be zero")
        if _require_exact_int(manifest, "full_evaluation_ids") < 500:
            raise ValueError("full_evaluation_ids must cover at least 500 tasks")
        if _require_exact_int(manifest, "full_evaluation_repositories") != 12:
            raise ValueError("full_evaluation_repositories must equal 12")
    elif replay_verified_rows != row_count:
        raise ValueError("replay_verified_rows must exactly equal row_count")
    if _require_exact_int(manifest, "immutable_max_tokens") != IMMUTABLE_MAX_TOKENS:
        raise ValueError("immutable_max_tokens must equal 4096")

    for zero_field in (
        "truncated_rows",
        "render_failures",
        "format_failures",
        "duplicate_rows",
        "evaluation_leakage_hits",
    ):
        if _require_exact_int(manifest, zero_field) != 0:
            raise ValueError(f"{zero_field} must be zero")
    if _require_exact_int(manifest, "format_loss_checked_rows") != row_count:
        raise ValueError("format_loss_checked_rows must exactly equal row_count")
    if _require_exact_int(manifest, "format_loss_failures") != 0:
        raise ValueError("format_loss_failures must be zero")
    if manifest.get("pre_rendered") is not True:
        raise ValueError("pre_rendered must be true")
    if manifest.get("stock_native_template") is not True:
        raise ValueError("stock_native_template must be true")

    stock_template_hash = _require_sha256(
        manifest.get("stock_chat_template_sha256"),
        "stock_chat_template_sha256",
    )
    tool_schema_hash = _require_sha256(
        manifest.get("tool_schema_sha256"),
        "tool_schema_sha256",
    )
    _require_sha256(
        manifest.get("output_jsonl_sha256"),
        "output_jsonl_sha256",
    )

    desirable_actual = 0
    undesirable_actual = 0
    category_counts: dict[str, int] = {
        "desirable_correct_patch": 0,
        "empty_terminal": 0,
        "repeated_read_loop": 0,
        "wrong_nonempty_replay": 0,
    }
    max_token_count = 0
    seen_rendered: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"row {index} prompt must be a nonempty string")
        if not isinstance(completion, str) or not completion:
            raise ValueError(f"row {index} completion must be a nonempty string")
        if type(row.get("label")) is not bool:
            raise ValueError(f"row {index} labels must be exact booleans")

        if row["label"]:
            desirable_actual += 1
        else:
            undesirable_actual += 1
        if behavior_schema:
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"row {index} behavior provenance must be a mapping")
            category = provenance.get("punishment_category")
            if category not in category_counts:
                raise ValueError(f"row {index} has an invalid punishment category")
            if (category == "desirable_correct_patch") is not row["label"]:
                raise ValueError(
                    f"row {index} punishment category disagrees with its label"
                )
            category_counts[str(category)] += 1
            if (
                row["label"]
                and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                in f"{prompt}\n{completion}"
            ):
                raise ValueError(
                    f"row {index} positive completion contains a private marker"
                )

        raw_token_count = _require_exact_int(row, "raw_token_count")
        token_count = _require_exact_int(row, "token_count", minimum=1)
        if raw_token_count > token_count:
            raise ValueError(
                f"row {index} raw_token_count must not exceed token_count"
            )
        if token_count > IMMUTABLE_MAX_TOKENS:
            raise ValueError(f"row {index} token_count exceeds immutable 4096 cap")
        max_token_count = max(max_token_count, token_count)

        provenance_hashes = row.get("provenance_hashes")
        if not isinstance(provenance_hashes, Mapping):
            raise ValueError(f"row {index} provenance_hashes must be a mapping")
        prompt_hash = _require_sha256(
            provenance_hashes.get("prompt_sha256"),
            f"row {index} prompt_sha256",
        )
        completion_hash = _require_sha256(
            provenance_hashes.get("completion_sha256"),
            f"row {index} completion_sha256",
        )
        if prompt_hash != _sha256_text(prompt):
            raise ValueError(f"row {index} prompt_sha256 does not match prompt")
        if completion_hash != _sha256_text(completion):
            raise ValueError(
                f"row {index} completion_sha256 does not match completion"
            )
        if provenance_hashes.get("stock_chat_template_sha256") != stock_template_hash:
            raise ValueError(
                f"row {index} stock_chat_template_sha256 does not match manifest"
            )
        if provenance_hashes.get("tool_schema_sha256") != tool_schema_hash:
            raise ValueError(
                f"row {index} tool_schema_sha256 does not match manifest"
            )

        rendered_identity = (prompt, completion)
        if rendered_identity in seen_rendered:
            raise ValueError(f"row {index} is a duplicate prompt/completion pair")
        seen_rendered.add(rendered_identity)

    if row_count < MIN_ROWS:
        raise ValueError(f"training data must contain at least {MIN_ROWS} rows")
    if undesirable_actual < MIN_UNDESIRABLE_ROWS:
        raise ValueError(
            f"training data must contain at least {MIN_UNDESIRABLE_ROWS} undesirable rows"
        )
    if desirable_actual < 1:
        raise ValueError("training data must contain at least one desirable row")
    if desirable_manifest != desirable_actual:
        raise ValueError("manifest desirable_count must exactly equal loaded labels")
    if undesirable_manifest != undesirable_actual:
        raise ValueError("manifest undesirable_count must exactly equal loaded labels")
    if desirable_actual + undesirable_actual != row_count:
        raise ValueError("manifest label counts must exactly equal row_count")
    if behavior_schema:
        assert behavior_negative_counts is not None
        for category in (
            "empty_terminal",
            "repeated_read_loop",
            "wrong_nonempty_replay",
        ):
            if (
                _require_exact_int(
                    behavior_negative_counts, category, minimum=0
                )
                != category_counts[category]
            ):
                raise ValueError(
                    f"behavior_negative_counts does not match {category}"
                )
        if category_counts["desirable_correct_patch"] != desirable_actual:
            raise ValueError(
                "desirable_correct_patch category count does not match labels"
            )
        if (
            category_counts["desirable_correct_patch"]
            + category_counts["wrong_nonempty_replay"]
            != replay_verified_rows
        ):
            raise ValueError(
                "replay_verified_rows does not match replay-backed categories"
            )

    manifest_max = _require_exact_int(manifest, "max_token_count", minimum=1)
    if manifest_max != max_token_count:
        raise ValueError(
            "manifest max_token_count must equal the maximum loaded row token_count"
        )
    if manifest_max > IMMUTABLE_MAX_TOKENS:
        raise ValueError("manifest max_token_count exceeds immutable 4096 cap")


def build_kto_config(
    config_cls: Callable[..., Any],
    *,
    output_dir: str,
    desirable_count: int,
    undesirable_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    save_steps: int,
    seed: int,
) -> Any:
    """Construct the exact audited TRL 0.24 KTO configuration."""
    if type(per_device_train_batch_size) is not int or per_device_train_batch_size < 2:
        raise ValueError("physical batch size must be at least two")
    if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")
    if type(max_steps) is not int or not 1 <= max_steps <= 25:
        raise ValueError("max_steps must be between 1 and 25")
    if type(save_steps) is not int or not 1 <= save_steps <= min(max_steps, 25):
        raise ValueError("save_steps must be between 1 and min(max_steps, 25)")
    if type(seed) is not int or seed != 0:
        raise ValueError("seed must remain the lineage value 0")

    desirable_weight, undesirable_weight = balanced_kto_weights(
        desirable_count,
        undesirable_count,
    )
    return config_cls(
        output_dir=output_dir,
        max_length=IMMUTABLE_MAX_TOKENS,
        max_prompt_length=IMMUTABLE_MAX_TOKENS,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        beta=0.1,
        learning_rate=5e-7,
        warmup_steps=0,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        precompute_ref_log_probs=False,
        dataset_num_proc=1,
        remove_unused_columns=False,
        use_liger_loss=False,
        dataloader_drop_last=True,
        logging_steps=1,
        report_to="none",
        max_steps=max_steps,
        save_steps=save_steps,
        seed=seed,
    )


def _emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def verify_fresh_lora_reference_equality(
    model: Any,
    tokenizer: Any,
    *,
    tensor_runtime: Any | None = None,
) -> None:
    """Fail closed unless a fresh enabled LoRA exactly matches its base logits."""
    if tensor_runtime is None:
        import torch as tensor_runtime

    inference_mode = getattr(tensor_runtime, "inference_mode", None)
    if not callable(inference_mode):
        raise RuntimeError("reference equality requires inference_mode context")
    disable_adapter = getattr(model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise RuntimeError("reference equality requires disable_adapter context")
    model_eval = getattr(model, "eval", None)
    model_train = getattr(model, "train", None)
    original_training = getattr(model, "training", None)
    if not callable(model_eval) or not callable(model_train) or type(original_training) is not bool:
        raise RuntimeError("reference equality requires restorable model training mode")

    encoded = tokenizer(
        text="Reference adapter equality check.",
        return_tensors="pt",
        add_special_tokens=True,
    )
    if not isinstance(encoded, Mapping) or not encoded:
        raise RuntimeError("reference equality tokenizer output is unavailable")
    input_ids = encoded.get("input_ids")
    input_numel = getattr(input_ids, "numel", None)
    if not callable(input_numel) or int(input_numel()) < 1:
        raise RuntimeError("reference equality tokenizer input must be nonempty")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {
            key: value.to(device) if callable(getattr(value, "to", None)) else value
            for key, value in encoded.items()
        }

    def snapshot_logits(output: object) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError("reference equality logits are unavailable")
        detach = getattr(logits, "detach", None)
        if not callable(detach):
            raise RuntimeError("reference equality logits detach is unavailable")
        detached_logits = detach()
        clone = getattr(detached_logits, "clone", None)
        if not callable(clone):
            raise RuntimeError("reference equality logits clone is unavailable")
        snapshot = clone()
        requires_grad = getattr(snapshot, "requires_grad", None)
        if type(requires_grad) is not bool:
            raise RuntimeError(
                "reference equality snapshot requires_grad must be boolean"
            )
        if requires_grad:
            raise RuntimeError("reference equality snapshot requires_grad must be false")
        return snapshot

    try:
        model_eval()
        with inference_mode():
            enabled_logits = snapshot_logits(model(**encoded))
            with disable_adapter():
                disabled_logits = snapshot_logits(model(**encoded))

        try:
            enabled_shape = tuple(enabled_logits.shape)
            disabled_shape = tuple(disabled_logits.shape)
        except (AttributeError, TypeError) as error:
            raise RuntimeError("reference equality logits shapes are unavailable") from error
        if enabled_shape != disabled_shape:
            raise RuntimeError("reference equality logits must have matching shapes")

        isfinite = getattr(tensor_runtime, "isfinite", None)
        equal = getattr(tensor_runtime, "equal", None)
        if not callable(isfinite) or not callable(equal):
            raise RuntimeError("reference equality tensor operations are unavailable")
        try:
            finite = bool(isfinite(enabled_logits).all().item()) and bool(
                isfinite(disabled_logits).all().item()
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("reference equality finite-logit check is unavailable") from error
        if not finite:
            raise RuntimeError("reference equality logits must be finite")
        if not bool(equal(enabled_logits, disabled_logits)):
            raise RuntimeError("fresh-LoRA and disabled-adapter logits must be exactly equal")

        numel = getattr(enabled_logits, "numel", None)
        if not callable(numel) or int(numel()) < 1:
            raise RuntimeError("reference equality checked element count is unavailable")
        _emit(
            "reference-equality",
            checked_elements=int(numel()),
            max_abs_diff=0,
        )
    finally:
        model_train(original_training)


class _KtoTelemetryCallback:
    def __init__(self, gpu_peak_bytes: Callable[[], int]) -> None:
        self._gpu_peak_bytes = gpu_peak_bytes
        self._started_at = time.monotonic()
        self._last_logged_step = 0

    def on_train_begin(
        self,
        _args: object,
        _state: object,
        control: object,
        **_kwargs: object,
    ) -> object:
        self._started_at = time.monotonic()
        self._last_logged_step = 0
        return control

    def on_log(
        self,
        _args: object,
        state: object,
        control: object,
        *,
        logs: Mapping[str, object] | None = None,
        **_kwargs: object,
    ) -> object:
        step = int(getattr(state, "global_step", 0))
        max_steps = int(getattr(state, "max_steps", 0))
        values = logs or {}
        if (
            step <= self._last_logged_step
            or step <= 0
            or ("loss" not in values and "grad_norm" not in values)
        ):
            return control
        elapsed = max(0.0, time.monotonic() - self._started_at)
        seconds_per_step_avg = elapsed / step
        eta_seconds = None
        estimated_training_total_seconds = None
        if step > 0 and max_steps >= step:
            eta_seconds = seconds_per_step_avg * (max_steps - step)
            estimated_training_total_seconds = seconds_per_step_avg * max_steps
        _emit(
            "optimizer-step",
            step=step,
            max_steps=max_steps,
            elapsed_training_seconds=elapsed,
            seconds_per_step_avg=seconds_per_step_avg,
            eta_seconds=eta_seconds,
            estimated_training_total_seconds=estimated_training_total_seconds,
            loss=values.get("loss"),
            grad_norm=values.get("grad_norm"),
            gpu_peak_bytes=int(self._gpu_peak_bytes()),
        )
        self._last_logged_step = step
        return control

    def __getattr__(self, name: str) -> Callable[..., object]:
        if not name.startswith("on_"):
            raise AttributeError(name)

        def passthrough(
            _args: object,
            _state: object,
            control: object,
            **_kwargs: object,
        ) -> object:
            return control

        return passthrough


@dataclass(frozen=True)
class _RuntimeDependencies:
    FastLanguageModel: Any
    Dataset: Any
    KTOConfig: Any
    KTOTrainer: Any
    gpu_peak_bytes: Callable[[], int]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("training manifest must be a JSON object")
    return value


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_loader: Callable[[], Any] | None = None,
    row_loader: Callable[[str | Path], list[dict[str, Any]]] = _load_jsonl,
    manifest_loader: Callable[[str | Path], dict[str, Any]] = _load_json,
    data_hash_loader: Callable[[str | Path], str] = _file_sha256,
    reference_equality_probe: Callable[[Any, Any], None] = (
        verify_fresh_lora_reference_equality
    ),
    initial_lora_b_probe: Callable[[Any], Any] = validate_initial_lora_b_state,
    updated_lora_b_probe: Callable[[Any, Any], None] = validate_updated_lora_b_state,
) -> int:
    wall_started_at = time.monotonic()
    args = build_arg_parser().parse_args(argv)
    canonical_output = Path(args.out)
    quarantine_output = canonical_output.with_name(
        f"{canonical_output.name}.inprogress"
    )
    if canonical_output.exists():
        raise FileExistsError(f"canonical output already exists: {canonical_output}")
    if quarantine_output.exists():
        raise FileExistsError(f"quarantine output already exists: {quarantine_output}")
    validation_started_at = time.monotonic()
    rows = row_loader(args.data)
    manifest = manifest_loader(args.manifest)
    validate_training_manifest(rows, manifest)
    source_data_sha256 = data_hash_loader(args.data)
    if source_data_sha256 != manifest["output_jsonl_sha256"]:
        raise ValueError("output_jsonl_sha256 does not match training data")
    source_manifest_sha256 = data_hash_loader(args.manifest)
    training_rows = rows
    coverage_counts: dict[str, int] | None = None
    if args.require_behavior_coverage:
        if manifest.get("schema") != BEHAVIOR_MANIFEST_SCHEMA:
            raise ValueError(
                "behavior coverage requires the behavior KTO schema"
            )
        training_rows, coverage_counts = (
            select_behavior_coverage_rows(rows)
        )
        scheduled_examples = (
            args.max_steps
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        if scheduled_examples != len(training_rows):
            raise ValueError(
                "behavior coverage requires exactly one full selected epoch"
            )
    data_validation_seconds = time.monotonic() - validation_started_at

    if runtime_loader is None:
        # Unsloth must patch TRL before KTOConfig/KTOTrainer are imported.
        from unsloth import FastLanguageModel

        from datasets import Dataset
        from trl import KTOConfig, KTOTrainer
        import torch

        def gpu_peak_bytes() -> int:
            if not torch.cuda.is_available():
                return 0
            return int(torch.cuda.max_memory_allocated())

        runtime = _RuntimeDependencies(
            FastLanguageModel=FastLanguageModel,
            Dataset=Dataset,
            KTOConfig=KTOConfig,
            KTOTrainer=KTOTrainer,
            gpu_peak_bytes=gpu_peak_bytes,
        )
    else:
        runtime = runtime_loader()

    config = build_kto_config(
        runtime.KTOConfig,
        output_dir=str(quarantine_output),
        desirable_count=sum(
            row["label"] is True for row in training_rows
        ),
        undesirable_count=sum(
            row["label"] is False for row in training_rows
        ),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )
    model_load_started_at = time.monotonic()
    model, tokenizer = runtime.FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=IMMUTABLE_MAX_TOKENS,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = runtime.FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=list(_LORA_TARGETS),
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model_load_seconds = time.monotonic() - model_load_started_at
    _emit(
        "model-ready",
        max_seq_length=IMMUTABLE_MAX_TOKENS,
        load_in_4bit=True,
        fresh_lora=True,
        model_load_seconds=model_load_seconds,
    )
    initial_lora_b_state = initial_lora_b_probe(model)
    reference_equality_probe(model, tokenizer)
    kto_processing_class = select_kto_processing_class(tokenizer)

    train_dataset = runtime.Dataset.from_list(training_rows)
    _emit(
        "data-ready",
        source_rows=len(rows),
        training_rows=len(training_rows),
        desirable_rows=sum(
            row["label"] is True for row in training_rows
        ),
        undesirable_rows=sum(
            row["label"] is False for row in training_rows
        ),
        behavior_coverage=coverage_counts,
        max_token_count=manifest["max_token_count"],
        data_validation_seconds=data_validation_seconds,
    )
    callback = _KtoTelemetryCallback(runtime.gpu_peak_bytes)
    trainer = runtime.KTOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_dataset,
        processing_class=kto_processing_class,
        callbacks=[callback],
    )
    trainer.train()
    completed_steps = int(
        getattr(
            getattr(trainer, "state", None),
            "global_step",
            callback._last_logged_step,
        )
    )
    if completed_steps != args.max_steps:
        raise RuntimeError(
            "KTO trainer did not complete the required optimizer steps"
        )
    updated_lora_b_probe(model, initial_lora_b_state)
    trainer.save_model(str(quarantine_output))
    evidence = {
        "schema_version": 1,
        "artifact_type": "v2p11_kto_training_evidence",
        "status": "complete",
        "base_model_path": str(Path(args.base).resolve()),
        "source_data_path": str(Path(args.data).resolve()),
        "source_data_sha256": source_data_sha256,
        "source_manifest_path": str(Path(args.manifest).resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_rows": len(rows),
        "training_rows": len(training_rows),
        "coverage_required": args.require_behavior_coverage,
        "coverage_counts": coverage_counts,
        "selected_sample_uids": (
            [str(row["sample_uid"]) for row in training_rows]
            if args.require_behavior_coverage
            else None
        ),
        "per_device_train_batch_size": (
            args.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "optimizer_steps": completed_steps,
        "seed": args.seed,
    }
    (quarantine_output / "training_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if canonical_output.exists():
        raise FileExistsError(
            f"canonical output appeared before publication: {canonical_output}"
        )
    quarantine_output.rename(canonical_output)
    _emit(
        "adapter-save",
        output_dir=str(canonical_output),
        total_wall_seconds=time.monotonic() - wall_started_at,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
