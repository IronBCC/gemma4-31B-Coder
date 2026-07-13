# Phase I — DeepSpeed/Ulysses Long-Context Probe

This is the experimental path for testing whether two GPUs can train Gemma-4-31B
past the Unsloth single-GPU ceiling. It is not the default production path.

## Current verdict

The 90K+ target is not validated on the current `ironbccllm` stack.

Measured on 2x RTX PRO 6000 Blackwell 96GB:

- BF16 DeepSpeed ZeRO-3 + Ulysses is mechanically real but not useful for this
  model size: 512, 1K, and 2K complete one optimizer step; 3K OOMs.
- QLoRA + direct Ulysses sequence sharding is the only productive path tested so
  far: 4K, 8K, and 12K complete one optimizer step with finite loss and finite
  gradient norm; 16K forward reaches backward and then OOMs.
- `flash-attn` is not installed. Source builds failed on the CUDA 13 / PyTorch
  2.12 environment because the PyPI build pulled a mismatched CUDA toolkit/header
  stack. Without a working memory-efficient attention backward kernel, 90K+ on
  two 96GB GPUs is not a realistic claim.

Additional GPU1-only kernel work:

- `flash-attn-4` can be installed from the upstream `flash_attn/cute[dev,cu13]`
  subpackage, but the raw FA4 CuTe path fails during kernel lowering on this RTX
  PRO Blackwell host, including tiny MHA cases.
- PyTorch 2.12 `SDPBackend.FLASH_ATTENTION` does work for an isolated
  Gemma-like GQA tensor at 16K when there is no attention mask.
- The full Gemma4 Transformers path still cannot use that backend faithfully:
  Gemma4 creates internal sliding-window masks, and PyTorch flash SDPA rejects
  non-null masks. An experimental no-mask causal bypass then reaches another
  real model shape with effective head_dim 512, which PyTorch flash SDPA also
  rejects.
- Therefore the proper long-context fix is still an attention backend that
  supports Gemma4's sliding masks and effective attention shapes on Blackwell.
  The current PyTorch flash backend is useful evidence, not a production
  training solution for this model.

Required NCCL environment on this host:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
```

Default NCCL collectives hung on the PHB-only topology.

## Why this exists

The measured failures rule out the earlier easy answers:

- `device_map="balanced"` / `device_map="auto"` can shard layers, but it produced
  `loss=nan` on this Gemma-4 stack.
- DDP gives throughput, not longer context, because each rank still sees the full
  sequence.
- FSDP/ZeRO alone shards model state. The blocker for 90K+ is activation and
  attention memory, so model-state sharding is not sufficient.

The only class worth testing for 90K+ on two GPUs is real sequence/context
parallelism. This path uses DeepSpeed-Ulysses, which splits the sequence across
participating GPUs.

## Success criteria

A run is valid only if all of these are true:

1. It runs on real Gemma-4-31B weights.
2. LoRA adapters are attached.
3. It uses Ulysses sequence parallelism with `sequence_parallel_size=2`.
4. A real JSONL training row is tokenized at the requested `--max-seq`.
5. Loss is finite.
6. Backward succeeds.
7. Gradient norm is finite.
8. Optimizer step succeeds.
9. Rank 0 saves an adapter checkpoint.

Anything short of that is a probe failure, even if the process prints progress.

## Datasets

Use the existing agentic datasets from the handoff:

- `data/agentic_sft_24k_windowed.jsonl` for 4K-32K/48K ramp stages.
- `data/agentic_sft_bashtools.jsonl` once 64K+ is proven.

The windowed set is the safe fallback below long context because every token is
still trained at turn boundaries. Do not return to destructive front truncation.

## Dry run

Dry run prints the planned stage ladder and the generated DeepSpeed config. It
does not import torch or DeepSpeed.

```bash
python3 phaseI_deepspeed_longctx/train_ulysses_lora.py \
  --dry-run \
  --target-seq 90112 \
  --max-seq 4096 \
  --write-config /tmp/ulysses-smoke-ds.json
```

## Smoke ladder

Use two idle GPUs only. On this host, NCCL needs P2P/IB disabled.

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 \
  phaseI_deepspeed_longctx/train_ulysses_lora.py \
  --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --data data/agentic_sft_24k_windowed.jsonl \
  --out adapters/ulysses-smoke-4k \
  --max-seq 4096 \
  --max-steps 1 \
  --limit-samples 2 \
  --write-config /tmp/ulysses-smoke-4k.json
```

The currently working QLoRA smoke shape is:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 \
  phaseI_deepspeed_longctx/train_ulysses_lora.py \
  --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --data data/agentic_sft_24k_windowed.jsonl \
  --out adapters/ulysses-q4-smoke-12k \
  --max-seq 12288 \
  --max-steps 1 \
  --limit-samples 2 \
  --no-zero-init \
  --skip-deepspeed-engine \
  --direct-manual-shard \
  --load-4bit
```

Then ramp only after the previous stage satisfies the full success criteria:

```bash
# 8K / 12K sanity verified with QLoRA direct Ulysses
--max-seq 8192  --max-steps 2 --data data/agentic_sft_24k_windowed.jsonl
--max-seq 12288 --max-steps 2 --data data/agentic_sft_24k_windowed.jsonl

# next unresolved rung; currently OOMs in backward without memory-efficient attention
--max-seq 16384 --max-steps 1 --data data/agentic_sft_24k_windowed.jsonl

# long-context proof, not yet achieved
--max-seq 32768 --max-steps 2 --data data/agentic_sft_24k_windowed.jsonl
--max-seq 65536 --max-steps 2 --data data/agentic_sft_bashtools.jsonl
--max-seq 90112 --max-steps 3 --data data/agentic_sft_bashtools.jsonl
```

If any stage OOMs, hangs, or produces non-finite loss/grad norm, stop. The
single-GPU Unsloth path remains the production route.

## Full run shape

Only after the 90K proof passes. This has not happened yet.

```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
  phaseI_deepspeed_longctx/train_ulysses_lora.py \
  --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --data data/agentic_sft_bashtools.jsonl \
  --out adapters/python-agentic-v2-ulysses-90k \
  --max-seq 90112 \
  --max-steps 1200 \
  --limit-samples 0 \
  --rank 32 \
  --alpha 32 \
  --grad-accum 16
```

The post-run verdict is still the fair fight:

- Base Gemma-4-31B vs adapter on SWE-bench Lite same-50.
- Then Verified/Pro only if Lite moves.
- Compare BF16 capability separately from NVFP4 deployment.

## Dependencies

This path needs the box training environment, not the lightweight local test
environment:

- `torch` with CUDA/NCCL
- `deepspeed`
- `transformers`
- `peft`
- `datasets` optional, only for other data tooling

The pure planning module is covered by local `unittest` tests and does not import
the GPU stack.

## Next engineering steps

1. Get a compatible memory-efficient attention backward stack working for
   Gemma-4 on Blackwell, preferably FlashAttention 4 or an equivalent kernel
   that supports this model's sliding-window masks and effective attention
   shapes.
2. Re-run the QLoRA Ulysses ladder from 16K upward.
3. Before full training, replace the direct manual sequence-shard loss path with
   a globally correct Ulysses data/loss adapter, or explicitly handle the
   one-token label boundary between shards.
4. If 16K still fails after the attention kernel is fixed, use the 12K windowed
   dataset path as the honest production fallback and evaluate it before spending
   more GPU time.
