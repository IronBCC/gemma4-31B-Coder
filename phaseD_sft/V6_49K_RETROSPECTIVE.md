# v6 @ 49k — Retrospective, Findings, and Path Forward (2026-07-11)

Companion to `V6_49K_CODEX_HANDOFF.md` (execution spec) and `phaseH_eval/SESSION_STATE.md` (state record). This document analyzes the day's work: what happened, what broke, what we learned, and the decision tree from here.

---

## 1. What we set out to do

Prove that a clean-data LoRA (v5 dup/trailing-user corruption fixed + compressed openswe Rust augmentation), trained at 49,152 context on frozen Gemma-4-31B, beats the cp20 baseline. Cap was raised 34,816 → 49,152 specifically to recover openswe Rust diversity (yield 3% → 36% pre-compression, 65% post-compression).

## 2. What actually happened (timeline of the 49k blocker)

| # | Event | Outcome |
|---|---|---|
| 1 | Gate B @ 49,152 (Codex) | aborted by launcher swap guard at 16.9 GB system swap, mid model-load |
| 2 | Gate B @ 40,960 (Codex) | same swap-guard abort — **guard, not workload** |
| 3 | User kills swap guard ("using SWAP limit was wrong") | retest reaches the real forward |
| 4 | True failure exposed | SDPA dense wants **256 GiB**; FlexAttention falls back to dense math wanting **128 GiB** |
| 5 | Probe: SDPA-FLASH @ exact Gemma shapes fwd-only | **works, 2.3 GiB** → "platform limitation" claim falsified |
| 6 | Probe: compiled flex fwd-only | works, 24 GiB → "re-enable compile" directive |
| 7 | Codex re-runs with compile restored | still dense → forward-only probes were **insufficient evidence** |
| 8 | Probe: compiled flex **fwd+backward** | **FAIL — backward Triton kernel needs 135,680 B shared memory; sm_120 limit is 101,376 B** ← true root cause |
| 9 | Probes: SDPA-FLASH bwd (11.3 GiB), xformers bwd (8.3 GiB), flex w/ small `kernel_options` (24 GiB), jagged len 46,365 (21.5 GiB) | three viable kernel paths identified |
| 10 | End-to-end verify runs 1-4 (real trainer, monkeypatched) | ALL still dense-OOM — each run eliminated one integration hypothesis |
| 11 | Run 5 (in flight): xformers custom attention interface | bypasses flex/dynamo/unsloth-compile entirely |

### The integration elimination ladder (runs 1-4)

| Run | Hypothesis tested | Diagnostic result |
|---|---|---|
| 1 | kernel_options + flex routing suffice | No — traceback shows flex nested inside `unsloth_compiled_cache` module (nested dynamo can't compile) |
| 2 | `UNSLOTH_COMPILE_DISABLE=1` removes nesting | Nesting gone, still dense → suspected global dynamo kill |
| 3 | re-enable dynamo + compile self-test | **Falsified**: dynamo was never disabled (self-test fired). Still dense. |
| 4 | unsloth's reentrant checkpoint poisons compiled flex → swap to native `use_reentrant=False` | Swap was a **no-op**: unsloth globally replaced `torch.utils.checkpoint.checkpoint` itself, and additionally wraps the whole forward at `compute_loss` level ("smart gradient offload") |

**Where this leaves the flex path**: unsloth's runtime stack (generated compiled modules + global checkpoint replacement + compute-loss-level offload wrapper) creates contexts in which `torch.compile(flex_attention)` silently degrades to eager → dense math. Fixing it inside unsloth is whack-a-mole against generated code.

**Why xformers (run 5) is structurally different**: `memory_efficient_attention` is a native CUTLASS autograd function — no dynamo, no compile, immune to checkpoint wrappers by construction. Both Gemma-4 mask patterns pre-proven fwd+bwd at 8.3 GiB.

## 3. What went wrong (honest list, ours included)

1. **Stale safety guard treated as gospel.** The 16 GB swap-used abort (calibrated for the degraded-RAM era) was carried into the v6 handoff as "non-negotiable" without re-checking after RAM was restored. It masked the real failure for two gate cycles and killed a healthy compile phase at 15 min. *User caught it, not us.*
2. **Forward-only probes over-claimed.** "49k is fixable — re-enable compile" was sent to Codex based on fwd-only evidence; training needs backward, where the real kernel limit lives. One wasted Codex cycle.
3. **multiprocessing.Pool tokenizer incident.** 16 forked workers each loaded a full tokenizer → swap 55→102 GB in ~3 min on a crash-prone box. Caught visually by the user, not by our own post-launch check.
4. **Wrong Orca channel for hours.** `orchestration send/reply` messages queue unread for busy workers (4 directives sat `read=0`, including a user spec change). Mid-task delivery requires `orca terminal send`. Cost: Codex laddered down to 34k against a stale plan before the hold arrived manually.
5. **Cap-ladder fallacy.** Both we and Codex initially treated lowering the cap as a fix for what was actually a kernel-path bug — the ladder can't outrun an O(n²) attention path.
6. **Session-long claim discipline lapses**: an early hard30 eval was invalidated by a LoRA/base name collision; the "92.8 GB VRAM at 24k" figure was repeated from memory without re-verification.

## 4. What went well

1. **Probe-driven elimination.** Every hypothesis got a cheap, isolated, shapes-exact GPU probe before (or after) expensive full runs. The probe matrix (fwd vs fwd+bwd × 4 backends × 2 masks × jagged length) is what actually located the sm_120 shared-memory limit.
2. **Instrumented verify wrapper.** Runs 3-4 carried self-tests (`dynamo disabled?`, `compile fires?`, `checkpoint fn module?`) that falsified hypotheses in seconds instead of hours, and *proved* run-4's swap was a no-op rather than leaving ambiguity.
3. **Safety record**: production stayed green through ~15 GPU1 launches/kills, all kills by exact PID, zero prod incidents all session.
4. **Data work fully landed**: v6 mixture built + format-gated at three caps; observation-only compression raised openswe yield 3%→65% with zero supervised-token edits (5 deterministic raw-vs-sanitized checks); intentional-vs-accidental dup distinction preserved anchor upweighting.
5. **The user's interventions were decisive twice**: killing the swap guard (exposed the real OOM) and adding RAM (removed the false constraint).

## 5. What we learned (durable)

- **sm_120 (RTX PRO 6000 Blackwell WS) has 101,376 B shared memory/SM** — flex_attention's default backward template (135,680 B) cannot run; PyTorch silently lowers to dense math instead of erroring meaningfully. Hopper (228 KB) doesn't have this problem → unsloth's 60k-on-H100 claims don't transfer.
- **Always probe attention backends with backward**, never forward-only, before making training claims.
- Working kernel paths on this GPU @ 49k/hd256 fwd+bwd: **SDPA-FLASH** (11.3 GiB, causal-only), **xformers memeff** (8.3 GiB, causal + local window), **flex w/ kernel_options** `{BLOCK_M:32,BLOCK_N:32,BLOCK_M1:16,BLOCK_N1:32,BLOCK_M2:32,BLOCK_N2:16}` (24 GiB — but unusable until the unsloth-context problem is solved).
- **unsloth internals that bite**: silently resets `attn_implementation` to sdpa at load; globally replaces `torch.utils.checkpoint.checkpoint`; wraps whole-forward in its own offload checkpoint; `UNSLOTH_COMPILE_DISABLE` accepts `1|partial` with different scopes.
- **Host RAM is not the context lever** — attention/activation memory lives on GPU; host RAM buys stability (swap headroom), not longer context.
- Ops: gate on real memory pressure (MemAvailable + reclaimable), not swap-used; `orca terminal send` for mid-task worker directives; never load heavy resources in `Pool` initializers.

## 5.5 FINAL ROOT CAUSE (superseding §2's flex-backward finding): `global_head_dim = 512`

Run 5 (xformers interface) cracked the case fully. The 50 sliding layers (head_dim 256) ran perfectly through xformers — **the dense-math path died**. The failure moved to the first **global** layer, and its rejection message exposed the real geometry: Gemma-4's 10 full-attention layers use **head_dim 512** (config `global_head_dim: 512`, `num_global_key_value_heads: 4`, V sharing K's projection — "alternative attention").

**head_dim 512 exceeds every fused attention kernel in the current ecosystem**: fa2/fa3 cap at 256; xformers cutlassF gated ≤ sm90; SDPA-flash ≤ 256; flex backward exceeds sm_120 shared memory even at 256. Dense math was the only executable path for global layers — on any backend, at any setting we tried. This retro-explains the whole ladder: **24,576 worked because 24k² dense global attention (~36 GB transient) fits; 49k² (~128 GB) doesn't.** The flex-backward shared-memory limit (§2 step 8) was real but secondary — fixing it would still have hit the hd-512 wall.

**Decision (user, 2026-07-11): build chunked global attention** — exact blockwise attention for the 10 global layers (row-block softmax decomposition is *mathematically exact*, not an approximation: softmax is per-query-row, so per-q-block computation with causal key ranges reproduces dense attention bit-for-bit up to float assoc.), custom autograd.Function with FA2-style recomputing backward, xformers keeps the 50 sliding layers. True 49,152 with the full Rust mixture.

## 6. Path forward (decision tree)

**Now (run 5, in flight)**: xformers interface end-to-end gate @ 49,152.

- **Run 5 PASSES** → recipe = xformers interface + registration + config flip (a ~40-line patch, currently in `/tmp/verify_49k_step.py` on host; promote into `phaseD_sft/` as a proper module + tests). Then: Gate B telemetry recorded → hand recipe to Codex → 5-step training from cp20 @ 2e-5 → hard30 vs same-day cp20 (≥ +3) → 30-case SWE-Lite smoke (≥18/30 non-empty, <10% format err) → scale only on pass.
- **Run 5 FAILS with xformers-specific error** (kernel rejects shape/bias on sm_120 inside real model) → try SDPA-FLASH hybrid: flash for the 10 full-attention layers (`is_causal=True`), xformers-local or manual chunked attention for the 50 sliding layers. All components individually proven.
- **Run 5 FAILS with the same dense-math OOM** → something *else* in unsloth's stack overrides `config._attn_implementation` per-forward (e.g. `temporary_patches/gemma4.py`). Next diagnostic: read that patch file's attention dispatch; if hostile, monkeypatch the attention module's forward directly (replace `Gemma4TextAttention.forward` wholesale).
- **If the whole custom-attention lane exhausts**: fall back to training at **24,576** (the only launcher-proven cap; VRAM 53.4 GiB) with the 34,816 dataset filtered to ≤24k, and file the 49k work as blocked-on-upstream (torch flex backward for sm_120 — worth checking newer torch nightlies, which may add sm_120 templates).

**Deliberately NOT pursuing**: flash-attn package builds (nvcc crash on sm_120 backward kernels, upstream issues #1987/#2361); article-drop/obfuscation compression of supervised tokens (corrupts the training signal); GPU0/prod for training.

## 7. Open items beyond the kernel fight

- Codex holds the task (`task_f1b83f0d17c6`) with the hold instruction; re-engage via `orca terminal send` once run 5 verdict is in.
- 30-case SWE-Lite smoke spec change is delivered to Codex and in its plan.
- hard30 gate must be re-anchored vs same-day cp20 (serving-path sensitivity, measured 21/30 vs historical 25-26/30).
- Nebius (3,129 rows) and Open-SWE go/TS/JS remain deliberate v7 candidates, not v6 scope.
