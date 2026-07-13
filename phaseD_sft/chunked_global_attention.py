#!/usr/bin/env python3
"""Exact chunked causal attention for Gemma-4 global layers (head_dim 512) on sm_120.

Why this exists: sm_120 (RTX PRO 6000 Blackwell WS) has ~101 KB shared memory/SM —
no fused attention kernel (flash/cutlass/flex) can tile head_dim 512 there, so every
backend silently degrades to dense math (O(S^2) score matrix = 128 GiB at 49k).

This module computes attention in query-row blocks. Row-block decomposition of
softmax-attention is mathematically EXACT (softmax normalizes per query row), not an
approximation. Peak memory = one block's score tile (fp32), independent of the fused-
kernel shared-memory ceiling because it uses plain GEMMs.

Backward is a custom FA2-style recompute: nothing is saved except q/k/v/out and the
per-row log-sum-exp, so autograd memory stays flat regardless of sequence length.
Works under any checkpoint wrapper (no dynamo, no torch.compile, no HOPs).
"""
from __future__ import annotations

import torch

DEFAULT_Q_BLOCK = 1024


def _causal_scores(q_blk, k_ctx, scale, q_start):
    # q_blk: (B,H,bq,D), k_ctx: (B,H,S_ctx,D) where S_ctx = q_start + bq (causal range)
    scores = torch.matmul(q_blk.float(), k_ctx.float().transpose(-1, -2)) * scale
    bq, s_ctx = scores.shape[-2], scores.shape[-1]
    qi = torch.arange(q_start, q_start + bq, device=scores.device).unsqueeze(-1)
    ki = torch.arange(s_ctx, device=scores.device).unsqueeze(0)
    scores.masked_fill_(ki > qi, float("-inf"))
    return scores


class ChunkedCausalAttention(torch.autograd.Function):
    """Exact causal attention, q-blockwise, fp32 softmax, recomputing backward."""

    @staticmethod
    def forward(ctx, q, k, v, scale, q_block):
        # q,k,v: (B,H,S,D) bf16/fp16/fp32, same heads (expand GQA before calling)
        B, H, S, D = q.shape
        out = torch.empty_like(q)
        lse = torch.empty(B, H, S, dtype=torch.float32, device=q.device)
        for s in range(0, S, q_block):
            e = min(s + q_block, S)
            scores = _causal_scores(q[:, :, s:e], k[:, :, :e], scale, s)
            blk_lse = torch.logsumexp(scores, dim=-1)
            p = torch.exp(scores - blk_lse.unsqueeze(-1))
            out[:, :, s:e] = torch.matmul(p.to(v.dtype), v[:, :, :e])
            lse[:, :, s:e] = blk_lse
            del scores, p
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.q_block = q_block
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        scale, q_block = ctx.scale, ctx.q_block
        B, H, S, D = q.shape
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        # D_i = rowsum(dout * out) per query row (fp32)
        delta = (dout.float() * out.float()).sum(dim=-1)  # (B,H,S)
        for s in range(0, S, q_block):
            e = min(s + q_block, S)
            scores = _causal_scores(q[:, :, s:e], k[:, :, :e], scale, s)
            p = torch.exp(scores - lse[:, :, s:e].unsqueeze(-1))  # (B,H,bq,e) fp32
            do_blk = dout[:, :, s:e].float()
            dv[:, :, :e] += torch.matmul(p.transpose(-1, -2), do_blk)
            dp = torch.matmul(do_blk, v[:, :, :e].float().transpose(-1, -2))
            ds = p * (dp - delta[:, :, s:e].unsqueeze(-1))  # softmax backward
            dq[:, :, s:e] = torch.matmul(ds, k[:, :, :e].float()) * scale
            dk[:, :, :e] += torch.matmul(ds.transpose(-1, -2), q[:, :, s:e].float()) * scale
            del scores, p, dp, ds
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def chunked_causal_attention(q, k, v, scale=None, q_block=DEFAULT_Q_BLOCK):
    """q,k,v: (B, H, S, D), equal head counts. Returns (B, H, S, D)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ChunkedCausalAttention.apply(q, k, v, float(scale), int(q_block))


def gemma4_global_attention_forward(module, query, key, value, attention_mask,
                                    dropout=0.0, scaling=None, sliding_window=None,
                                    q_block=DEFAULT_Q_BLOCK, **kwargs):
    """transformers attention-interface adapter for hd-512 global layers.

    query/key/value: (B, H_q, S, D) / (B, H_kv, S, D). GQA expanded here; grads for
    repeated KV flow back correctly because repeat_interleave is autograd-aware.
    Ignores attention_mask by design: training uses bsz=1 unpadded rows, so pure
    causal is exact. Returns (out (B, S, H, D), None) per interface contract.
    """
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    out = chunked_causal_attention(query, key, value, scale=scale, q_block=q_block)
    return out.transpose(1, 2).contiguous(), None
