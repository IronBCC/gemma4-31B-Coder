#!/usr/bin/env python3
"""DeepSpeed-Ulysses LoRA probe for Gemma-4 long-context training.

This is path 2: a real sequence-parallel experiment. It is intentionally
small-first and acceptance-test driven. The run is not considered valid unless a
real sample produces finite loss, backward succeeds, optimizer step succeeds,
and a checkpoint can be written.

Launch on the box, keeping prod GPU0 out of scope by remapping visible devices:

  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
    phaseI_deepspeed_longctx/train_ulysses_lora.py \
    --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
    --max-seq 4096 --max-steps 1 --limit-samples 2 --out adapters/ulysses-smoke-4k
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from phaseI_deepspeed_longctx.longctx_plan import (
    build_deepspeed_config,
    build_stage_ladder,
    dataset_for_max_seq,
    reject_invalid_parallelism,
)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_rank0() -> bool:
    return _rank() == 0


def _print_rank0(message: str) -> None:
    if _is_rank0():
        print(message, flush=True)


def _extract_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    messages = row.get("messages")
    if isinstance(messages, list):
        chunks = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "unknown")
            content = message.get("content", "")
            reasoning = message.get("reasoning_content")
            if reasoning:
                chunks.append(f"<{role}:reasoning>\n{reasoning}")
            chunks.append(f"<{role}>\n{content}")
        if chunks:
            return "\n\n".join(chunks)
    for key in ("prompt", "instruction", "problem", "input"):
        if isinstance(row.get(key), str):
            return row[key]
    raise ValueError("row has no supported text/messages/prompt field")


class JsonlTextDataset:
    """Small JSONL-backed dataset for smoke and ramp probes."""

    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        max_seq: int,
        limit_samples: int = 0,
        require_full_length: bool = False,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.rows = self._load_rows(path, tokenizer, max_seq, limit_samples, require_full_length)
        if not self.rows:
            raise ValueError(f"no usable rows found in {path}")

    @staticmethod
    def _load_rows(path: str, tokenizer, max_seq: int, limit_samples: int, require_full_length: bool) -> list[str]:
        rows: list[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = _extract_text(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
                if require_full_length:
                    encoded = tokenizer(text, truncation=True, max_length=max_seq, padding=False)
                    if len(encoded["input_ids"]) < max_seq:
                        continue
                rows.append(text)
                if limit_samples and len(rows) >= limit_samples:
                    break
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        encoded = self.tokenizer(
            self.rows[idx],
            truncation=True,
            max_length=self.max_seq,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        mm_token_type_ids = torch.zeros_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "position_ids": position_ids,
            "labels": labels,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def _manual_sequence_shard(
    batch: dict[str, Any],
    *,
    sp_rank: int,
    sp_world_size: int,
) -> dict[str, Any]:
    """Shard a fixed-length batch on the sequence axis for direct Ulysses probes."""

    import torch

    seq_length = batch["input_ids"].shape[1]
    if seq_length % sp_world_size != 0:
        raise ValueError(f"batch seqlen={seq_length} is not divisible by sp-size={sp_world_size}")
    chunk_len = seq_length // sp_world_size
    start = chunk_len * sp_rank
    end = chunk_len * (sp_rank + 1)

    sharded: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == seq_length:
            sharded[key] = value[:, start:end].contiguous()
        else:
            sharded[key] = value
    return sharded


def _finite_grad_norm(model) -> float:
    import torch

    first_param = next(model.parameters())
    sq_sum = torch.zeros((), device=first_param.device)
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        sq_sum = sq_sum + torch.sum(grad.float() * grad.float())
    return float(torch.sqrt(sq_sum).item())


def _lora_target_modules(scope: str) -> str:
    if scope == "attention":
        return r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    if scope == "all-linear":
        return (
            r"model\.language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        )
    raise ValueError(f"unsupported LoRA target scope: {scope}")


def _causal_lm_loss_from_logits(logits, shift_labels):
    import torch
    import torch.nn.functional as F

    if shift_labels is None:
        raise RuntimeError("model did not return loss and batch has no shift_labels")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1).to(logits.device),
        ignore_index=-100,
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _sdpa_kernel_context(backend: str):
    if backend == "auto":
        return nullcontext()

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel(backends[backend])


def _drop_all_one_attention_mask(batch: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return batch
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return batch
    if not bool(attention_mask.all().item()):
        raise RuntimeError(
            "--drop-all-one-attention-mask was requested, but the batch contains padding; "
            "pack or filter samples before using flash SDPA"
        )
    batch = dict(batch)
    batch.pop("attention_mask", None)
    return batch


def _patch_transformers_sdpa_force_causal_no_mask() -> None:
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def forced_sdpa_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        sdpa_kwargs = {}
        if hasattr(module, "num_key_value_groups"):
            if use_gqa_in_sdpa(None, key):
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key = repeat_kv(key, module.num_key_value_groups)
                value = repeat_kv(value, module.num_key_value_groups)
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        is_causal = bool(query.shape[2] > 1 and is_causal)
        output_dtype = query.dtype
        if query.dtype not in (torch.float16, torch.bfloat16):
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        attn_output = attn_output.to(output_dtype)
        return attn_output.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = forced_sdpa_attention_forward
    _print_rank0("[stage] patched Transformers SDPA to force no-mask causal attention")


def _patch_ulysses_for_variable_gemma4_heads(attn_implementation: str) -> bool:
    """Let DeepSpeed's single HF Ulysses wrapper handle Gemma-4's mixed attention shapes."""

    import types
    import torch
    import deepspeed.comm as dist
    from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    wrapper = ALL_ATTENTION_FUNCTIONS[attn_implementation]
    uattn = None
    for cell in wrapper.__closure__ or ():
        candidate = cell.cell_contents
        if isinstance(candidate, UlyssesSPAttentionHF):
            uattn = candidate
            break
    if uattn is None:
        return False

    original_forward = uattn.forward

    def forward_with_runtime_heads(self, module, query, key, value, attention_mask, *args, **kwargs):
        q_heads = query.shape[1]
        kv_heads = key.shape[1]
        head_dim = query.shape[-1]
        local_seq_length = query.shape[2]

        self.attn_head_size = head_dim
        self.attn_head_count = q_heads
        self.global_kv_head_count = kv_heads
        self.local_q_head_count = q_heads // self.world_size
        self.kv_replication_factor = self.world_size // kv_heads
        if self.kv_replication_factor > 1:
            self.local_kv_head_count = 1
        else:
            self.local_kv_head_count = kv_heads // self.world_size
        self.local_seq_length = local_seq_length
        self.global_seq_length = local_seq_length * self.world_size
        self.required_query_shape = torch.Size(
            [self.local_seq_length, self.batch_size, self.attn_head_count, self.attn_head_size]
        )
        self.required_key_value_shape = torch.Size(
            [self.local_seq_length, self.batch_size, self.global_kv_head_count, self.attn_head_size]
        )
        self.required_context_shape = torch.Size(
            [
                self.global_seq_length,
                self.batch_size,
                self.attn_head_size * self.attn_head_count // self.world_size,
            ]
        )
        return original_forward(module, query, key, value, attention_mask, *args, **kwargs)

    uattn.forward = types.MethodType(forward_with_runtime_heads, uattn)
    _print_rank0("[stage] patched Ulysses for Gemma-4 variable attention heads")
    return True


def _build_lora_model(args: argparse.Namespace, *, device: str):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if args.load_4bit:
        _print_rank0("[stage] enabling 4-bit NF4 loading")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        device_map={"": device},
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.load_4bit:
        _print_rank0("[stage] preparing k-bit model for LoRA training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif args.gradient_checkpointing:
        _print_rank0("[stage] enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    _print_rank0("[stage] attaching LoRA")
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_lora_target_modules(args.lora_targets),
    )
    model = get_peft_model(model, lora)
    _print_rank0("[stage] LoRA attached")
    return model


def run_single_gpu_training(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "single-GPU mode expects exactly one visible GPU; set CUDA_VISIBLE_DEVICES to the physical GPU to use"
        )
    torch.cuda.set_device(0)

    dataset_path = args.data or dataset_for_max_seq(args.max_seq)
    _print_rank0(
        "[probe] "
        f"single-gpu base={args.base} data={dataset_path} max_seq={args.max_seq} "
        f"steps={args.max_steps} sdpa_backend={args.sdpa_backend}"
    )

    _print_rank0("[stage] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _print_rank0("[stage] tokenizer loaded")

    if args.force_causal_flash_no_mask:
        _patch_transformers_sdpa_force_causal_no_mask()

    _print_rank0("[stage] loading model")
    model = _build_lora_model(args, device="cuda:0")
    _print_rank0("[stage] model loaded")

    _print_rank0("[stage] loading JSONL dataset")
    dataset = JsonlTextDataset(
        dataset_path,
        tokenizer,
        max_seq=args.max_seq,
        limit_samples=args.limit_samples,
        require_full_length=args.require_full_length,
    )
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, collate_fn=_collate)
    _print_rank0(f"[stage] dataset ready rows={len(dataset)}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    model.train()

    accepted_steps = 0
    _print_rank0("[stage] entering single-GPU train loop")
    for step, batch in enumerate(loader, start=1):
        if accepted_steps >= args.max_steps:
            break
        _print_rank0(f"[stage] step {step} moving batch to cuda")
        batch = {key: value.to("cuda:0") for key, value in batch.items()}
        batch = _drop_all_one_attention_mask(batch, enabled=args.drop_all_one_attention_mask)
        shift_labels = batch.pop("shift_labels", None)
        _print_rank0(f"[stage] step {step} forward")
        with _sdpa_kernel_context(args.sdpa_backend):
            outputs = model(**batch)
        loss = outputs.loss if outputs.loss is not None else _causal_lm_loss_from_logits(outputs.logits, shift_labels)
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
        _print_rank0(f"[stage] step {step} backward")
        with _sdpa_kernel_context(args.sdpa_backend):
            loss.backward()
        grad_norm = _finite_grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm}")
        _print_rank0(f"[stage] step {step} optimizer")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accepted_steps += 1
        peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        _print_rank0(
            json.dumps(
                {"step": step, "loss": loss_value, "grad_norm": grad_norm, "peak_mem_gb": peak_gb},
                sort_keys=True,
            )
        )

    if accepted_steps < args.max_steps:
        raise RuntimeError(f"only completed {accepted_steps}/{args.max_steps} steps")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    _write_json(
        str(out / "single_gpu_probe_result.json"),
        {
            "engine": "single-gpu",
            "max_seq": args.max_seq,
            "steps": accepted_steps,
            "dataset": dataset_path,
            "sdpa_backend": args.sdpa_backend,
            "attn_implementation": args.attn_implementation,
        },
    )
    print(f"[probe] adapter saved -> {out}", flush=True)
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    dataset = args.data or dataset_for_max_seq(args.max_seq)
    config = build_deepspeed_config(
        sequence_parallel_size=args.sequence_parallel_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        offload_optimizer=args.offload_optimizer,
        offload_param=args.offload_param,
        activation_checkpointing=args.gradient_checkpointing,
    )
    ladder = [stage.__dict__ for stage in build_stage_ladder(target_seq=args.target_seq)]
    payload = {
        "mode": "dry-run",
        "selected_dataset": dataset,
        "stage_ladder": ladder,
        "deepspeed_config": config,
        "acceptance_gate": [
            "real JSONL row tokenized",
            "Ulysses registration succeeds",
            "LoRA adapters attached",
            "loss is finite",
            "backward succeeds",
            "optimizer step succeeds",
            "rank0 checkpoint saved",
        ],
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.write_config:
        _write_json(args.write_config, config)
    return 0


def run_training(args: argparse.Namespace) -> int:
    reject_invalid_parallelism(args.parallelism)

    if args.single_gpu:
        return run_single_gpu_training(args)

    import torch
    import deepspeed
    import deepspeed.comm as dist
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        UlyssesSPAttentionHF,
        UlyssesSPDataLoaderAdapter,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("launch with torchrun so WORLD_SIZE/RANK/LOCAL_RANK are set")

    torch.cuda.set_device(_local_rank())
    dist.init_distributed(dist_backend="nccl", dist_init_required=True)
    if args.load_4bit and not args.no_zero_init:
        raise ValueError("--load-4bit currently requires --no-zero-init; use the direct Ulysses QLoRA path")

    dataset_path = args.data or dataset_for_max_seq(args.max_seq)
    ds_config = build_deepspeed_config(
        sequence_parallel_size=args.sequence_parallel_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        offload_optimizer=args.offload_optimizer,
        offload_param=args.offload_param,
        activation_checkpointing=args.gradient_checkpointing,
    )
    if args.write_config and _is_rank0():
        _write_json(args.write_config, ds_config)

    _print_rank0(
        "[probe] "
        f"base={args.base} data={dataset_path} max_seq={args.max_seq} "
        f"sp={args.sequence_parallel_size} steps={args.max_steps}"
    )

    _print_rank0("[stage] registering Ulysses attention")
    mpu = UlyssesSPAttentionHF.register_with_transformers(
        model_name_or_path=args.base,
        core_attn_implementation=args.attn_implementation,
        sequence_parallel_size=args.sequence_parallel_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.max_seq,
        seq_length_is_variable=False,
    )
    _patch_ulysses_for_variable_gemma4_heads(args.attn_implementation)
    _print_rank0("[stage] Ulysses attention registered")

    _print_rank0("[stage] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _print_rank0("[stage] tokenizer loaded")

    if args.force_causal_flash_no_mask:
        _patch_transformers_sdpa_force_causal_no_mask()

    if args.no_zero_init:
        _print_rank0("[stage] loading model without deepspeed.zero.Init")
        model = _build_lora_model(args, device=f"cuda:{_local_rank()}")
    else:
        _print_rank0("[stage] loading model through Transformers DeepSpeed ZeRO-3 integration")
        from transformers.integrations.deepspeed import HfDeepSpeedConfig

        hf_ds_config = HfDeepSpeedConfig(ds_config)
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
        )
        # Keep the weak-ref-backed HF DeepSpeed config alive until loading finishes.
        model._hf_ds_config = hf_ds_config
    _print_rank0("[stage] model loaded")

    if not args.no_zero_init:
        model.config.use_cache = False
        if args.gradient_checkpointing:
            _print_rank0("[stage] enabling gradient checkpointing")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        _print_rank0("[stage] attaching LoRA")
        lora = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_lora_target_modules(args.lora_targets),
        )
        model = get_peft_model(model, lora)
        _print_rank0("[stage] LoRA attached")

    _print_rank0("[stage] loading JSONL dataset")
    dataset = JsonlTextDataset(
        dataset_path,
        tokenizer,
        max_seq=args.max_seq,
        limit_samples=args.limit_samples,
        require_full_length=args.require_full_length,
    )
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, collate_fn=_collate)
    _print_rank0(f"[stage] dataset ready rows={len(dataset)}")

    _print_rank0("[stage] initializing DeepSpeed engine")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    if args.skip_deepspeed_engine:
        _print_rank0("[stage] skipping DeepSpeed engine; using direct Ulysses step")
        sp_rank = mpu.get_sequence_parallel_rank()
        sp_world_size = mpu.get_sequence_parallel_world_size()
        if args.direct_manual_shard:
            _print_rank0("[stage] using manual sequence sharding for direct probe")
        else:
            loader = UlyssesSPDataLoaderAdapter(
                loader,
                sp_rank=sp_rank,
                sp_group=mpu.get_sequence_parallel_group(),
                sp_world_size=sp_world_size,
                device=torch.device(f"cuda:{_local_rank()}"),
            )
            _print_rank0("[stage] Ulysses dataloader adapter initialized")
        model.train()
        accepted_steps = 0
        _print_rank0("[stage] entering direct train loop")
        for step, batch in enumerate(loader, start=1):
            if accepted_steps >= args.max_steps:
                break
            _print_rank0(f"[stage] step {step} batch received")
            if args.direct_manual_shard:
                batch = _manual_sequence_shard(batch, sp_rank=sp_rank, sp_world_size=sp_world_size)
            _print_rank0(f"[stage] step {step} moving batch to cuda")
            batch = {key: value.to(f"cuda:{_local_rank()}") for key, value in batch.items()}
            batch = _drop_all_one_attention_mask(batch, enabled=args.drop_all_one_attention_mask)
            shift_labels = batch.pop("shift_labels", None)
            _print_rank0(f"[stage] step {step} forward")
            with _sdpa_kernel_context(args.sdpa_backend):
                outputs = model(**batch)
            loss = outputs.loss if outputs.loss is not None else _causal_lm_loss_from_logits(outputs.logits, shift_labels)
            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
            _print_rank0(f"[stage] step {step} backward")
            loss.backward()
            grad_norm = _finite_grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm}")
            _print_rank0(f"[stage] step {step} optimizer")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accepted_steps += 1
            _print_rank0(json.dumps({"step": step, "loss": loss_value, "grad_norm": grad_norm}, sort_keys=True))
        if accepted_steps < args.max_steps:
            raise RuntimeError(f"only completed {accepted_steps}/{args.max_steps} steps")
        if _is_rank0():
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            _write_json(
                str(out / "ulysses_probe_result.json"),
                {
                    "engine": "direct",
                    "max_seq": args.max_seq,
                    "steps": accepted_steps,
                    "dataset": dataset_path,
                    "sequence_parallel_size": args.sequence_parallel_size,
                    "parallelism": args.parallelism,
                },
            )
            print(f"[probe] adapter saved -> {out}", flush=True)
        return 0

    engine, optimizer, _engine_loader, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=None,
        config=ds_config,
        mpu=mpu,
    )
    _print_rank0("[stage] DeepSpeed engine initialized")
    loader = UlyssesSPDataLoaderAdapter(
        loader,
        sp_rank=mpu.get_sequence_parallel_rank(),
        sp_group=mpu.get_sequence_parallel_group(),
        sp_world_size=mpu.get_sequence_parallel_world_size(),
        device=torch.device(f"cuda:{_local_rank()}"),
    )
    _print_rank0("[stage] Ulysses dataloader adapter initialized")

    accepted_steps = 0
    for step, batch in enumerate(loader, start=1):
        if accepted_steps >= args.max_steps:
            break
        _print_rank0(f"[stage] step {step} forward")
        batch = {key: value.to(engine.device) for key, value in batch.items()}
        batch = _drop_all_one_attention_mask(batch, enabled=args.drop_all_one_attention_mask)
        shift_labels = batch.pop("shift_labels", None)
        with _sdpa_kernel_context(args.sdpa_backend):
            outputs = engine(**batch)
        loss = outputs.loss if outputs.loss is not None else _causal_lm_loss_from_logits(outputs.logits, shift_labels)
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
        _print_rank0(f"[stage] step {step} backward")
        engine.backward(loss)
        grad_norm = _finite_grad_norm(engine.module)
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm}")
        _print_rank0(f"[stage] step {step} optimizer")
        engine.step()
        accepted_steps += 1
        _print_rank0(json.dumps({"step": step, "loss": loss_value, "grad_norm": grad_norm}, sort_keys=True))

    if accepted_steps < args.max_steps:
        raise RuntimeError(f"only completed {accepted_steps}/{args.max_steps} steps")

    if _is_rank0():
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        engine.module.save_pretrained(out)
        tokenizer.save_pretrained(out)
        _write_json(
            str(out / "ulysses_probe_result.json"),
            {
                "max_seq": args.max_seq,
                "steps": accepted_steps,
                "dataset": dataset_path,
                "sequence_parallel_size": args.sequence_parallel_size,
                "parallelism": args.parallelism,
            },
        )
        print(f"[probe] adapter saved -> {out}", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    parser.add_argument("--data", default="")
    parser.add_argument("--out", default="adapters/ulysses-longctx-probe")
    parser.add_argument("--target-seq", type=int, default=90_112)
    parser.add_argument("--max-seq", type=int, default=4_096)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=2)
    parser.add_argument(
        "--require-full-length",
        action="store_true",
        help="Skip JSONL rows that tokenize shorter than --max-seq. Useful when dropping attention masks for flash SDPA.",
    )
    parser.add_argument("--sequence-parallel-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora-targets", choices=("all-linear", "attention"), default="all-linear")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "flash", "cudnn", "efficient", "math"),
        default="auto",
        help="Force a PyTorch SDPA backend during model forward. Use 'flash' on Blackwell to avoid math-SDPA OOM.",
    )
    parser.add_argument(
        "--drop-all-one-attention-mask",
        action="store_true",
        help="Drop attention_mask before forward only if it contains no padding, allowing PyTorch flash SDPA.",
    )
    parser.add_argument(
        "--force-causal-flash-no-mask",
        action="store_true",
        help=(
            "Experimental probe only: patch Transformers SDPA to ignore internal masks and use full causal attention. "
            "This is not faithful Gemma sliding-window training."
        ),
    )
    parser.add_argument("--parallelism", default="ulysses")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--offload-optimizer", action="store_true")
    parser.add_argument("--offload-param", action="store_true")
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable HF gradient checkpointing and DeepSpeed activation checkpointing.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--zero-remote-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Where DeepSpeed zero.Init should create parameters before partitioning.",
    )
    parser.add_argument(
        "--no-zero-init",
        action="store_true",
        help="Load a full model replica per rank for tiny smoke debugging before reintroducing ZeRO-3 init.",
    )
    parser.add_argument(
        "--skip-deepspeed-engine",
        action="store_true",
        help="Use Ulysses sequence parallel attention directly without deepspeed.initialize for compatibility probing.",
    )
    parser.add_argument(
        "--direct-manual-shard",
        action="store_true",
        help="In direct mode, bypass UlyssesSPDataLoaderAdapter and shard a fixed batch locally for hang isolation.",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Run a non-Ulysses single-GPU QLoRA probe on the one visible CUDA device.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-config", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.dry_run:
        return _dry_run(args)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
