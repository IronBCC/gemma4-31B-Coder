# CODEX GO — v6 49,152-context training (Gate B PASSED, execute Phase 5→6)

**Status: the 49k blocker is fully solved, verified end-to-end, and promoted to committed-ready code.**
You are cleared to execute training. Read this whole file first. Source of truth for state:
`phaseH_eval/SESSION_STATE.md` → section "49k attention saga — resolution status".

---

## 0. Updated goal

Produce the first v6 LoRA adapter trained at **49,152 context** on the frozen Gemma-4-31B base,
then pass the eval gates. Concretely, in order, STOP at the first failure:

1. Run a **1-step smoke** through the promoted `--sm120-attn` flag path (confirms the in-repo flag
   reproduces the verified /tmp recipe → checkpoint written, no OOM).
2. Run a **5-step training** from cp20 init at lr 2e-5 on the full 49k dataset.
3. **hard30** eval of the resulting checkpoint vs the same-day cp20 baseline: **≥ cp20 + 3** semantic.
4. **30-case SWE-Lite smoke**: ≥ 18/30 non-empty patches, < 10% format errors, median first source
   edit before command 10.
5. Only on passing all gates: report GO for a broader run. Do NOT scale before that.

North star (unchanged): one solo model strong in Python + Rust + C++, frozen base + LoRA, no regression.

---

## 1. What changed since your last dispatch (already done, verified — do NOT redo)

- **Root cause of every 49k OOM found**: Gemma-4's 10 global layers use `global_head_dim: 512`.
  No fused attention kernel (flash/cutlass/flex) can tile head_dim 512 on sm_120 (RTX PRO 6000
  Blackwell WS, ~101 KB shared mem/SM) → every backend silently degrades to dense O(S²) math → OOM.
  (H100's 228 KB SMEM is why unsloth's 60k-context claims hold there; ours is a hardware ceiling.)
- **Solution stack built + verified (Gate B PASS 2026-07-11 19:36, one full optimizer step at 49,152)**:
  - sliding layers (50×, hd 256): xformers `memory_efficient_attention` + local mask.
  - global layers (10×, hd 512): exact chunked attention `phaseD_sft/chunked_global_attention.py`
    (custom autograd.Function, FA2-style recompute backward, mathematically exact; 7/7 unit tests).
  - loss: fused `cut_cross_entropy.linear_cross_entropy` (softcap 30.0, shift, ignore_index −100) —
    never materializes the 262k-vocab × S logits tensor (that OOM'd at 45.28 GiB).
- **Promoted to code** (this is what you use — the /tmp wrapper is retired):
  - NEW `phaseD_sft/sm120_attention.py`: `register_sm120_attention`, `swap_to_nonreentrant_checkpoint`,
    `install_fused_ce_loss`, `require_return_hidden_states`.
  - `phaseD_sft/train_rust_lora.py`: opt-in flag **`--sm120-attn`** wires all of the above.
    Inert unless the flag is set. Requires env `UNSLOTH_RETURN_HIDDEN_STATES=1` (guarded — errors if unset).
  - Verified on host: py_compile clean; import/registration/patch/xformers/cce all wire; env-guard raises.
  - Files are SYNCED to the host repo already (`/home/ironbcc/projects/gemma4-31B-Coder/phaseD_sft/`).
- **Verified Gate B telemetry** (reference — your run should land near these):
  VRAM peak GPU1 **82,300 / 97,900 MiB**; **~48 min per optimizer step** (16 microsteps @ ~180 s on
  worst-case-length rows); loss 0.017–0.022; grad_norm 0.076; host RAM peak **43 / 61 GB**;
  prod ports 8000/8101/8103/8104 green throughout.

---

## 2. Host + hard constraints (NON-NEGOTIABLE)

- Host: `ssh ironbcc@192.168.50.148`, repo `/home/ironbcc/projects/gemma4-31B-Coder`,
  python `.venv-train/bin/python`.
- **GPU0 = prod vLLM, OFF-LIMITS.** Train GPU1 only (`CUDA_VISIBLE_DEVICES=1`).
- Ports **8000 / 8101 / 8103 / 8104 must stay green** the entire time — check before and after each launch.
- **Kill only by exact numeric PID** (from `ps -eo pid,args | grep … | grep -v grep`). **NEVER `pkill -f`**
  (matches your own ssh command string → kills the session).
- Host RAM is 64 GB. The 49k run peaks ~43 GB — fine on bare RAM. **Never raise any cgroup cap.**
  If you use a bounded systemd unit and it OOM-kills at ~43 GB, fall back to the plain nohup launch below
  (that is the exact configuration that PASSED). Do not fight the launcher.
- Smoke-first always. Verify one artifact before scaling.

---

## 3. Exact commands

### Step A — 1-step smoke via the promoted flag (≈48 min)

Reproduces the verified recipe through `--sm120-attn`. Launch (this is run-8's exact env + launch,
only swapping the /tmp wrapper for the in-repo flag):

```bash
ssh ironbcc@192.168.50.148
cd /home/ironbcc/projects/gemma4-31B-Coder

# preflight: prod green + GPU1 free + RAM
for p in 8000 8101 8103 8104; do (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "$p ok" || echo "$p DOWN"; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
awk '/MemAvailable/{print $2/1048576" GiB avail"}' /proc/meminfo   # abort if < 15

rm -rf adapters/swe_edit_v6_49k_smoke   # clean target
nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  UNSLOTH_COMPILE_DISABLE=1 \
  UNSLOTH_RETURN_HIDDEN_STATES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$PWD \
  .venv-train/bin/python phaseD_sft/train_rust_lora.py \
    --data data/unsloth_agentic_train_swe_edit_trace_v6_budget49152_long16 \
    --out adapters/swe_edit_v6_49k_smoke \
    --init-adapter adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10 \
    --max-seq 49152 --max-steps 1 --bsz 1 --grad-accum 16 --warmup-steps 1 \
    --lr 2e-5 --load-4bit --save-steps 1 --logging-steps 1 \
    --sm120-attn \
  > /tmp/v6_49k_smoke.log 2>&1 &
echo "launched pid $!"
```

PASS gate A: `adapters/swe_edit_v6_49k_smoke/checkpoint-1/adapter_model.safetensors` exists;
log shows `[sm120] xformers_sm120 attention interface registered`,
`[sm120] fused cut-cross-entropy compute_loss installed`, `[sm120] fused-CE loss #…`, and `[done] adapter saved`;
GPU1 VRAM peak ≤ ~85 GB; prod ports still green. **If Step A fails, STOP and report — do not run Step B.**

### Step B — 5-step training on the full 49k dataset (≈4 h)

Same env, full dataset, 5 optimizer steps, save every step:

```bash
rm -rf adapters/swe_edit_v6_49k_s1
nohup env \
  CUDA_VISIBLE_DEVICES=1 UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_RETURN_HIDDEN_STATES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD \
  .venv-train/bin/python phaseD_sft/train_rust_lora.py \
    --data data/unsloth_agentic_train_swe_edit_trace_v6_budget49152 \
    --out adapters/swe_edit_v6_49k_s1 \
    --init-adapter adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10 \
    --max-seq 49152 --max-steps 5 --bsz 1 --grad-accum 16 --warmup-steps 1 \
    --lr 2e-5 --load-4bit --save-steps 1 --logging-steps 1 \
    --sm120-attn \
  > /tmp/v6_49k_s1.log 2>&1 &
echo "launched pid $!"
```

Monitor by tailing `/tmp/v6_49k_s1.log` for `event=optimizer_step` lines (~48 min apart) and by
polling GPU1 + prod ports. Do NOT kill on log-mtime staleness alone — check the actual PID + GPU util
(one optimizer step is 16 microsteps ≈ 48 min of apparent silence between step lines is NORMAL).
PASS gate B: checkpoint-5 written, loss trending sane, prod green.

### Step C — hard30 + SWE-Lite smoke

Follow `phaseD_sft/V6_PIPELINE_RUNBOOK.md` Phase 6 for serving + eval (256k serve, FP8 KV, isolated
eval unit; distinct `--served-model-name` from the LoRA module name — a name collision invalidated a
prior eval). Gate: hard30 ≥ same-day cp20 + 3; then 30-case SWE-Lite smoke ≥ 18/30 non-empty,
< 10% format errors, median first edit < command 10.

---

## 4. Reporting

- Post a short `status` when Step A passes, when Step B launches, and when each optimizer step lands.
- Send `worker_done` to your coordinator handle on completion (or full stop on any gate failure) with a
  3-line summary + the checkpoint path + eval numbers.
- If blocked, `ask` — do not guess around a hard constraint in §2.

---

## 5. Exactly how 49k was verified (method + code + evidence)

### 5.1 Method
The verification was a **real one training step** (not a synthetic probe) at `--max-seq 49152`, cp20 init,
on the worst-case-length shard `data/unsloth_agentic_train_swe_edit_trace_v6_budget49152_long16` (the 16
longest rows by rendered tokens — longest is 46,365 tokens). It exercised the full path that a real run
uses: model load → attention forward (both layer types) → backward → optimizer step → checkpoint save.
This is stronger than a forward-only kernel probe — the sm_120 memory limits bite in the **backward** pass,
so any probe that skips backward is worthless here (a lesson from an earlier over-claim). Driver was
`/tmp/verify_49k_step.py` on host (runtime monkeypatch wrapper); its logic is now the in-repo
`phaseD_sft/sm120_attention.py` you run via `--sm120-attn`.

### 5.2 The three components, with the actual code

**(a) Global layers (hd 512) — exact chunked attention** (`phaseD_sft/chunked_global_attention.py`).
Row-block softmax decomposition is mathematically exact (softmax normalizes per query row), so peak memory
is one q-block's score tile instead of the full S² matrix. Forward + custom recompute backward:

```python
# forward: loop q-blocks, fp32 softmax, save only q,k,v,out,lse
for s in range(0, S, q_block):
    e = min(s + q_block, S)
    scores = _causal_scores(q[:, :, s:e], k[:, :, :e], scale, s)   # (B,H,bq,e) fp32, causal-masked
    blk_lse = torch.logsumexp(scores, dim=-1)
    p = torch.exp(scores - blk_lse.unsqueeze(-1))
    out[:, :, s:e] = torch.matmul(p.to(v.dtype), v[:, :, :e])
    lse[:, :, s:e] = blk_lse
# backward: FA2-style, delta = rowsum(dout*out); ds = p*(dp - delta); accumulate dq/dk/dv fp32
```

**(b) Sliding layers (hd 256) — xformers memory-efficient attention** (`phaseD_sft/sm120_attention.py`):

```python
if sliding_window:
    bias = LocalAttentionFromBottomRightMask(window_left=sliding_window - 1, window_right=0)
else:
    bias = LowerTriangularMask()
out = xops.memory_efficient_attention(q, k, v, attn_bias=bias, p=..., scale=scaling)
```

Both are dispatched by one registered backend `"xformers_sm120"` that routes on `query.shape[-1] > 256`.

**(c) Loss — fused cross-entropy** (`phaseD_sft/sm120_attention.py::install_fused_ce_loss`), pairs with
env `UNSLOTH_RETURN_HIDDEN_STATES=1` (model returns hidden states, not logits):

```python
loss = linear_cross_entropy(hidden.to(lm_w.dtype), lm_w, labels,
    ignore_index=-100, softcap=30.0, shift=True,
    reduction="sum" if num_items_in_batch is not None else "mean")
```

This never allocates the `S × 262k` fp32 logits tensor that OOM'd at 45.28 GiB after attention was solved.

### 5.3 Unit-test evidence (attention correctness, independent of the training run)
`phaseD_sft/tests/test_chunked_global_attention.py` — **7/7 pass on the box**. Each compares the chunked
op (forward AND all three input grads) against a dense causal reference: exact small block, jagged block
(S not divisible by q_block), single-block degenerate, **head_dim 512**, scale-none default, bf16 loose
tolerance, and a causality test (first tokens ignore future keys). Run:
`cd /home/ironbcc/projects/gemma4-31B-Coder && python -m pytest phaseD_sft/tests/test_chunked_global_attention.py -q`

### 5.4 Live run evidence (the actual Gate B pass, from `/tmp/verify_49k_step8.log`)
```
[patch] xformers attn call #1: q=(1, 46365, 32, 256) sliding_window=1024 scale=1.0
[patch] chunked-global attn call #1: q=(1, 32, 46365, 512) scale=1.0
[patch] fused-CE loss #1: 0.01667 (n_items=402633)
event=optimizer_step optimizer_step=1/1 microstep=16/16 elapsed_seconds=2885
event=optimizer_step_timing optimizer_step=1/1 step_wall_seconds=2885.15
{'loss': '0.02034', 'grad_norm': '0.07587', 'learning_rate': '0', 'epoch': '1'}
[done] adapter saved -> adapters/v6_gateB_49152_verify_claude
```
Both attention layer types fired (256 and 512 head_dim), fused CE produced finite losses, the optimizer
step completed, and the checkpoint was written — `adapters/v6_gateB_49152_verify_claude/checkpoint-1/adapter_model.safetensors`
(979 MB) + optimizer.pt + trainer_state.json. GPU1 peak 82,300 MiB; prod ports green throughout.

### 5.5 Promoted-flag verification (what `--sm120-attn` guarantees)
On host, in the train venv: `sm120_attention` imports; `register_sm120_attention()` puts `"xformers_sm120"`
in `ALL_ATTENTION_FUNCTIONS`; `install_fused_ce_loss(Trainer)` swaps in `_fused_ce_compute_loss`;
xformers + cut_cross_entropy import; and `require_return_hidden_states()` raises when the env flag is unset.
`py_compile` clean on all three files. Step A in §3 is the belt-and-suspenders 1-step run that confirms the
flag path reproduces the /tmp result before the 4 h Step B.

### 5.6 Scripts that may still need updating (audit)
- **`train_rust_lora.py`** — DONE (flag added, synced). No further change for the nohup path in §3.
- **Bounded systemd launcher `recover_swe_edit_v4_smoke_first.sh`** — NOT updated. It does not export
  `UNSLOTH_RETURN_HIDDEN_STATES=1` / `UNSLOTH_COMPILE_DISABLE=1` nor pass `--sm120-attn`. The §3 commands
  deliberately use plain **nohup** (the exact config that passed) and bypass this launcher, so no edit is
  required to execute. Only touch it if you later want the bounded unit for a long multi-day run — then add
  the two env vars + the flag and keep `MemoryMax` ≥ 48G (49k peaks ~43G; never raise it beyond real headroom).
- **Eval/serving scripts (Phase 6)** — unchanged: serving uses vLLM, a different attention stack from
  training; the sm_120 chunked path is a training-only concern. Follow `V6_PIPELINE_RUNBOOK.md` as-is.
- **No other script needs changes.** Do NOT modify `chunked_global_attention.py`, `compact_observations.py`,
  `dedup_and_trim_traces.py`, `prune_loops.py`, or dataset builders — the data lane is frozen and format-gated.
