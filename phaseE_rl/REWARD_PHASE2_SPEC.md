# Reward Phase 2: Docker-Verified Tier (spec, 2026-07-14)

Goal: replace the parse-only 1.0 tier with execution truth, attacking stalled
resolve (6-7/30) and fallen patch precision (30%). Round: GRPO from promoted
`swe_grpo_v1_s2/checkpoint-100` on `data/rlvr_swe_decision_v2.jsonl` (1,895
prefixes, verified unique).

## Tier ladder (v2)

| Reward | Condition |
|---|---|
| 0.0 | no tool call / unparseable |
| 0.2 | valid bash call, read-only command |
| 0.5 | edit command, parses, but NOT executable in container (no image / apply fails) |
| 0.8 | edit command **executes successfully** in reconstructed container state |
| 1.0 | edit executes AND instance F2P tests pass afterward |

Rationale: keep shaping gradient (0.2/0.5) so groups aren't all-zero; execution
tiers (0.8/1.0) supply the correctness signal parse-only couldn't.

## Execution architecture (cost-controlled)

1. **Per-prompt state reconstruction, ONCE:** replay the prefix's commands in the
   instance image (`docker run` → replay → `docker commit` as
   `rlvr-state:<prompt_hash>`). Cache on disk keyed by prompt hash; LRU-evict
   (disk watchdog rules apply, ≥40G loopback floor).
2. **Per-sample verification:** spawn container from the committed state image,
   run the completion's command (`bash -c`, never `-lc`), check rc + worktree
   diff for 0.8; then targeted F2P subset (not full suite — instance F2P list
   only, timeout 300s) for 1.0. `docker rm -f` by exact cid.
3. **Batching:** num_generations=4 per prompt → 4 verifications per state image.
   Verification runs on CPU while GPU generates the next group (overlap via
   worker thread + queue; reward fn blocks only on its own batch).

## Image coverage reality (must measure BEFORE training)

v2 sources vs images:
- swe-smith (700) + kwai (500, swe-smith-based): images pullable
  (`jyangballin/swesmith.x86_64.*`) — expected coverage high.
- smoke (139): SWE-bench Lite instances — princeton images local from evals.
- oracle (256): swe-smith-derived — verify instance_id → image mapping.
- open_swe_traces_qwen35 (300): repo/commit provenance unclear — likely NO images.

Preflight task: compute image availability per row → emit
`rlvr_swe_decision_v2_imagecov.json`. Rows without images keep the parse-only
ladder capped at 0.5 (never 0.8/1.0). If coverage <60%, consider training on the
image-backed subset only (rebalance check first).

## Config deltas vs round 1

- Same integration stack (MM→text graft, generation_kwargs stops [1,49,106],
  prompt cap 2560, batch 2 / accum 2, gradient checkpointing).
- `loss_type="grpo"` with default scale_rewards (σ-scaling ON — the DrGRPO
  step-14 blowup showed unscaled discrete rewards act as a removed grad clamp;
  revisit dr_grpo only with constant-baseline rewards).
- Reward timeout budget: hard 420s/group; on timeout → tier 0.5 fallback + log
  (never stall the trainer).
- save_steps 25, eval ladder unchanged: hard30 floor + 30-case temp-0.7 smoke;
  success = resolve >7/30 AND precision recovery (>40%) with patches ≥18/30.

## Split of work (proposal)

- Codex: preflight image-coverage script + state-reconstruction/commit cache
  (CPU/docker, parallel to C++ builds under the same disk guard).
- Claude: reward fn integration into `grpo_swe_edit_decision.py` (worker-thread
  verification queue, tier ladder, timeout fallback) + unit tests mirroring
  `tests/test_grpo_swe_reward.py`.
- Gate to launch training: image coverage measured, 10-prompt dry-run of the
  full verify path (state cache hit + 0.8/1.0 tiers observed), GPU<1GB free
  check, bounded unit.
