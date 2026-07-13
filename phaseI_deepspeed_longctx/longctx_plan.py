"""Planning utilities for the DeepSpeed/Ulysses long-context path.

This module is dependency-light on purpose. It lets the smoke ladder, dataset
choice, and DeepSpeed config be tested on a laptop before touching the GPU box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_FULL_TRACE_DATASET = "data/agentic_sft_bashtools.jsonl"
DEFAULT_WINDOWED_DATASET = "data/agentic_sft_24k_windowed.jsonl"

_INVALID_PARALLELISM = {
    "auto": "device_map=auto shards layers, not sequence activations; this path produced NaN before.",
    "balanced": "device_map=balanced shards layers, not sequence activations; this path produced NaN before.",
    "ddp": "DDP replicates the sequence on each GPU, so it improves throughput but not max context.",
    "fsdp-only": "FSDP/ZeRO shard model state, but the 90K blocker is activation/attention memory.",
}


@dataclass(frozen=True)
class TrainingStage:
    name: str
    max_seq: int
    max_steps: int
    dataset: str


def dataset_for_max_seq(
    max_seq: int,
    *,
    full_trace_dataset: str = DEFAULT_FULL_TRACE_DATASET,
    windowed_dataset: str = DEFAULT_WINDOWED_DATASET,
    full_trace_threshold: int = 65_536,
) -> str:
    """Select the safer dataset for a sequence length.

    Below the long-context threshold, use turn-boundary windowed samples so the
    run trains every token without destructive front truncation. Once the probe
    can really fit long samples, switch to full traces.
    """
    return full_trace_dataset if max_seq >= full_trace_threshold else windowed_dataset


def build_stage_ladder(
    *,
    target_seq: int,
    start_seq: int = 4_096,
    full_trace_dataset: str = DEFAULT_FULL_TRACE_DATASET,
    windowed_dataset: str = DEFAULT_WINDOWED_DATASET,
) -> list[TrainingStage]:
    """Build a conservative smoke-to-target ladder.

    The first stages are intentionally tiny: they prove imports, Ulysses
    registration, LoRA attachment, finite loss, backward, and optimizer step
    before expensive contexts are attempted.
    """
    if target_seq < start_seq:
        raise ValueError("target_seq must be >= start_seq")

    seqs: list[int] = []
    current = start_seq
    while current < target_seq:
        seqs.append(current)
        current *= 2
    if not seqs or seqs[-1] != target_seq:
        seqs.append(target_seq)

    stages: list[TrainingStage] = []
    for idx, seq in enumerate(seqs):
        if idx == 0:
            name = "smoke-4k" if seq == 4_096 else f"smoke-{seq // 1024}k"
            max_steps = 1
        elif seq == target_seq:
            name = f"target-{round(seq / 1000)}k"
            max_steps = 3
        else:
            name = f"ramp-{seq // 1024}k"
            max_steps = 2
        stages.append(
            TrainingStage(
                name=name,
                max_seq=seq,
                max_steps=max_steps,
                dataset=dataset_for_max_seq(
                    seq,
                    full_trace_dataset=full_trace_dataset,
                    windowed_dataset=windowed_dataset,
                ),
            )
        )
    return stages


def build_deepspeed_config(
    *,
    sequence_parallel_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    zero_stage: int = 3,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    activation_checkpointing: bool = True,
) -> dict[str, Any]:
    """Return the DeepSpeed config for Ulysses sequence-parallel probing."""
    if sequence_parallel_size < 2:
        raise ValueError("sequence_parallel_size must be >= 2 for the 2-GPU long-context path")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    zero_optimization: dict[str, Any] = {
        "stage": zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
    if offload_optimizer:
        zero_optimization["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if offload_param:
        zero_optimization["offload_param"] = {"device": "cpu", "pin_memory": True}

    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {"enabled": True},
        "zero_optimization": zero_optimization,
        "sequence_parallel_size": sequence_parallel_size,
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
    }
    if activation_checkpointing:
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False,
            "synchronize_checkpoint_boundary": False,
        }
    return config


def reject_invalid_parallelism(mode: str) -> str:
    """Reject modes that do not prove 2-GPU long-context training."""
    normalized = mode.strip().lower()
    if normalized in _INVALID_PARALLELISM:
        raise ValueError(_INVALID_PARALLELISM[normalized])
    return normalized
