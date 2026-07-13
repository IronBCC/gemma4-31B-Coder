#!/usr/bin/env python3
"""sm_120 long-context attention + fused-loss stack for Gemma-4-31B LoRA SFT.

Why this exists (see phaseH_eval/SESSION_STATE.md "49k attention saga"): Gemma-4's
10 global layers use head_dim 512. NO fused attention kernel (flash/cutlass/flex)
can tile head_dim 512 on sm_120 (RTX PRO 6000 Blackwell WS, ~101 KB shared mem/SM),
so every backend silently degrades to dense O(S^2) math and OOMs at 49k. The 50
sliding layers (head_dim 256) also can't run flex backward within the sm_120 SMEM
budget under unsloth's runtime.

This module packages the VERIFIED 49,152-context solution (Gate B PASS 2026-07-11,
GPU1 peak 82.3/97.9 GB, one optimizer step end-to-end):

  * sliding layers (hd 256)  -> xformers memory_efficient_attention + local mask
  * global  layers (hd 512)  -> exact chunked_global_attention (this repo)
  * loss                     -> fused cut_cross_entropy (never materializes the
                                262k-vocab x S logits tensor that OOM'd at 45 GiB)

Registered as a single Transformers attention backend "xformers_sm120" that routes
per-layer by head_dim. Requires env UNSLOTH_RETURN_HIDDEN_STATES=1 so the model
returns hidden states (not logits) and the fused loss can pair with them.

All heavy imports are deferred into the functions so this module imports cleanly on
a torch-less data-pipeline box; it only *runs* on the training box.
"""
from __future__ import annotations

SM120_ATTN_IMPL = "xformers_sm120"
_GEMMA_FINAL_LOGIT_SOFTCAP = 30.0


def _sm120_attention_forward(module, query, key, value, attention_mask,
                             dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    """Transformers attention-interface fn routing by head_dim.

    query/key/value: (B, H, S, D). Returns (out (B, S, H, D), None) per contract.
    hd>256 -> exact chunked global attention (no fused kernel exists on sm_120).
    hd<=256 -> xformers local/causal memory-efficient attention.
    """
    if query.shape[-1] > 256:
        from phaseD_sft.chunked_global_attention import gemma4_global_attention_forward
        return gemma4_global_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
        )

    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import (
        LocalAttentionFromBottomRightMask,
        LowerTriangularMask,
    )

    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    q = query.transpose(1, 2)  # (B, S, H, D)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if sliding_window:
        bias = LocalAttentionFromBottomRightMask(window_left=sliding_window - 1, window_right=0)
    else:
        bias = LowerTriangularMask()
    out = xops.memory_efficient_attention(
        q, k, v, attn_bias=bias, p=dropout if module.training else 0.0, scale=scaling
    )
    return out, None


def register_sm120_attention() -> None:
    """Register the "xformers_sm120" backend in the Transformers attention registry."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    try:
        ALL_ATTENTION_FUNCTIONS.register(SM120_ATTN_IMPL, _sm120_attention_forward)
    except Exception:
        ALL_ATTENTION_FUNCTIONS[SM120_ATTN_IMPL] = _sm120_attention_forward
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        got = ALL_ATTENTION_FUNCTIONS.get_interface(SM120_ATTN_IMPL, None)
    else:
        got = ALL_ATTENTION_FUNCTIONS[SM120_ATTN_IMPL]
    assert got is _sm120_attention_forward, "FATAL: xformers_sm120 registration failed"
    print("[sm120] xformers_sm120 attention interface registered", flush=True)


def swap_to_nonreentrant_checkpoint(model) -> int:
    """Replace unsloth's reentrant autograd checkpoint with native use_reentrant=False.

    This is the checkpoint configuration under which Gate B passed; the exact chunked
    backward needs no reentrant semantics and non-reentrant is the supported combo.
    """
    import functools
    import torch.utils.checkpoint as tuc
    native = functools.partial(tuc.checkpoint, use_reentrant=False)
    swapped = 0
    for mod in model.modules():
        if hasattr(mod, "_gradient_checkpointing_func"):
            mod._gradient_checkpointing_func = native
            swapped += 1
    print(f"[sm120] swapped checkpoint fn on {swapped} modules -> native use_reentrant=False",
          flush=True)
    return swapped


def install_fused_ce_loss(trainer_cls, softcap: float = _GEMMA_FINAL_LOGIT_SOFTCAP) -> None:
    """Patch a Trainer class's compute_loss to use fused cut_cross_entropy.

    Pairs with UNSLOTH_RETURN_HIDDEN_STATES=1: the model returns hidden states under
    `.logits`; we project + cross-entropy in one fused op via linear_cross_entropy,
    so the (S x 262k) logits tensor is never materialized (it OOM'd at 45.28 GiB).
    """
    from cut_cross_entropy import linear_cross_entropy

    _seen = {"n": 0}

    def _fused_ce_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        if num_items_in_batch is None:
            num_items_in_batch = inputs.pop("num_items_in_batch", None)
        else:
            inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        hidden = outputs.logits  # hidden states (B, S, H) under RETURN_HIDDEN_STATES=1
        base = model
        while hasattr(base, "module"):
            base = base.module
        try:
            lm_w = base.get_output_embeddings().weight
        except Exception:
            lm_w = next(p for n, p in base.named_parameters() if "lm_head" in n)
        assert hidden.shape[-1] == lm_w.shape[-1], (
            f"FATAL: got real logits (vocab={hidden.shape[-1]}), not hidden states — "
            "set UNSLOTH_RETURN_HIDDEN_STATES=1"
        )
        loss = linear_cross_entropy(
            hidden.to(lm_w.dtype), lm_w, labels,
            ignore_index=-100, softcap=softcap, shift=True,
            reduction="sum" if num_items_in_batch is not None else "mean",
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        if _seen["n"] < 3:
            _seen["n"] += 1
            print(f"[sm120] fused-CE loss #{_seen['n']}: {loss.item():.5f} "
                  f"(n_items={num_items_in_batch})", flush=True)
        return (loss, outputs) if return_outputs else loss

    trainer_cls.compute_loss = _fused_ce_compute_loss
    print("[sm120] fused cut-cross-entropy compute_loss installed", flush=True)


def require_return_hidden_states() -> None:
    """Fail fast if the env flag the fused loss depends on is not set."""
    import os
    if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES") != "1":
        raise RuntimeError(
            "--sm120-attn requires env UNSLOTH_RETURN_HIDDEN_STATES=1 (model must return "
            "hidden states for the fused cut_cross_entropy loss). Set it and relaunch."
        )
