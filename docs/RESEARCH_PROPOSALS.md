# Research Proposals: Beyond Incremental RLVR

Four designs to raise Gemma-4-31B solo SWE performance, ordered by the user's
priority (1 → 5 → 4 → 6). All build on the frozen-base + LoRA + execution-verifier
stack already in the repo.

**Baselines to beat (fixed self-retry harness, 30-case Lite, temp 0.7):**
Python cp100 = **21 patches / 10 resolved (33%)**. Rust raw = **11/75 (14.7%)**.

**Measurement discipline (applies to all):** n=30 session noise is ±3–4 resolves.
A result is only *real* if it clears **≥ +4 on the 30-case slice**, OR — preferred
for these experiments — is measured on a **60-case slice** (build once:
`phaseH_eval/build_lite60.py`, seeded, disjoint-from-training) where a +3 gain is
~2σ. Every run uses `swebench_edit_first_selfretry.yaml` at 131,072.

**Compute anchors (measured this program):** GRPO ~45 s/step (num_gen 4) /
~167 s (num_gen 5); one full agent rollout ~5–8 min; 30-case smoke ~35 min @ 8
workers; adapter merge ~20 min; hard30 floor ~35 min. Single training GPU (GPU1);
prod on GPU0 is off-limits.

---

## Proposal 1 — Expert Iteration (ReST/STaR) on self-generated verified traces

**Why first:** highest ROI, lowest risk, self-contained (no teacher, no quota
wall, no GRPO exploration/OOM pain). Literature (ReST^EM, STaR, rejection-sampling
SFT) shows this is the workhorse behind most open coding-model gains.

**Exact approach (one iteration):**
1. **Sample.** Run the current policy (cp100) as an agent over a training pool of
   ~400 SWE-smith Python instances (images already local), **k=8 samples each**,
   temp 1.0, self-retry harness. → ~3,200 trajectories.
2. **Filter (rejection sampling).** Keep only F2P-**resolved** trajectories (the
   verifier is free). Apply the existing edit-first + dedup + compaction +
   format gates. Expect ~20–35% resolve × dedup → ~500–900 kept traces.
3. **SFT.** Fresh r32 LoRA on the frozen base from the kept traces (init from
   cp100 or from base — ablate both), 49k budget, bounded launcher.
4. **Gate → iterate.** hard30 floor, then 60-case smoke. If resolve ↑ beyond
   noise, the new adapter becomes the sampler for iteration N+1. Stop when two
   consecutive iterations fail to clear +3.

**Scripts:** reuse `rust_baseline_driver.py` generalized to Python images (the
sampler, add `--k` and `--temp`); reuse `verified_reward.f2p_invocations` +
in-image scoring; reuse `dedup_and_trim_traces.py`, `compact_observations.py`,
`verify_gemma_format_loss.py`, `train_rust_lora.py` lineage.
**New:** `phaseD_sft/expert_iteration.py` (orchestrates sample→filter→manifest);
`phaseH_eval/build_lite60.py`.

**Gates:** (a) sampler yield ≥ 15% resolved (else pool too hard / policy too
weak); (b) kept-trace format failures = 0; (c) 60-case smoke resolve ≥ base +3;
(d) hard30 ≥ floor.

**Estimated gain:** **+3 to +8 resolved** (33% → ~40–47%) over 2–3 iterations,
diminishing — this is the most evidence-backed of the four. Ceiling set by the
pool's difficulty and the base's reachable capability.

**Time/compute per iteration:** sampling 3,200 rollouts @ ~6 min / 8-way ≈
**~40 h wall** (the dominant cost; can cut to ~12 h with k=4 and a 200-instance
pool for iteration 1) + SFT ~3 h + eval ~2 h. **~2 days/iteration**; budget 2–3
iterations → ~1 week. Sampling is GPU-bound (serving the policy) — competes with
other GPU lanes, so schedule as the primary GPU consumer.

---

## Proposal 5 — Red-Queen Co-Evolution (adversarial auto-curriculum)

**Why:** the one design that can raise the *ceiling*, not just climb faster.
Attacks the real bottleneck — running out of verified tasks at the right
difficulty. Highest novelty, highest variance.

**Exact approach:**
- **Two roles, alternating.** *Generator* G (SWE-smith bug-synthesis, optionally
  a small LoRA that picks/parameterizes mutation operators) and *Fixer* F (our
  policy).
- **Difficulty-banded reward for G:** reward G for bugs the *current* F resolves
  in the **10–50%** band (barely-solvable). Bugs F resolves >80% (too easy) or
  <5% (too hard/unsolvable) get ~0 reward. This keeps the curriculum pinned to
  F's frontier. Solvability is execution-checked (a bug is valid only if the
  gold patch passes F2P and F2P fails pre-patch — SWE-smith already guarantees
  this).
- **Fixer trains** (expert iteration or GRPO) on the freshly generated
  frontier-difficulty bugs each round.
- **Arms race:** as F improves, the 10–50% band shifts to harder bugs → G is
  pushed up → auto-curriculum.

**Scripts:** reuse SWE-smith synthesis harness for G's action space; reuse the
verifier for solvability + difficulty labels; reuse Proposal-1 machinery for F's
training. **New:** `phaseE_rl/red_queen/generator.py` (bug-op selection +
difficulty-band reward), `phaseE_rl/red_queen/loop.py` (round orchestration:
generate → solvability-filter → estimate F's pass-rate via 4-sample probe →
keep-band → train F → repeat), `phaseE_rl/red_queen/tests/`.

**Gates:** (a) G produces ≥ 100 valid banded bugs/round (else band mis-tuned);
(b) generated-bug distribution not collapsed (repo/op diversity metric);
(c) F's 60-case resolve trends up across ≥3 rounds; (d) **decontam**: generated
bugs never overlap eval slices (hard id/repo exclusion) — critical, or the
number is meaningless.

**Estimated gain:** **wide error bars, +0 to +10.** If the arms race sustains, it
could exceed Proposal-1's plateau because the training distribution keeps
adapting; if the band mis-tunes or G collapses, it yields nothing. Treat as a
research bet, not a scheduled win.

**Time/compute:** build ~3–4 days (generator reward + loop + tests are real new
code). Per round: bug-gen + solvability ~4 h (CPU/docker) + F pass-rate probe
~3 h + F train ~3 h + eval ~1 h ≈ **~half a day/round**, ~5 rounds to see signal
→ ~1 week run after the build. Highest total investment.

---

## Proposal 4 — Evolutionary LoRA Population (natural selection + merge-crossover)

**Why:** cheap, parallelizes on CPU scoring, escapes single-lineage local optima.
Merge-as-crossover is a proven operator (evolutionary model merging).

**Exact approach:**
1. **Seed population** = {cp100, v8cp30, the vgrpo adapters, 2–3 short-SFT
   variants on different data subsets}. ~6–8 adapters.
2. **Fitness** = verified resolve on a fixed 60-case fitness slice (disjoint from
   the final eval slice — no selection leakage).
3. **Select** top ~4 by fitness.
4. **Crossover** = weighted merge of adapter pairs (grid or learned merge
   coefficients via `merge_swe_rlvr_for_eval.py` extended to weighted sums);
   children = merged pairs.
5. **Mutate** = a short (5–15 step) SFT or GRPO burst on fresh verified data per
   child.
6. **Re-score, elitism** (keep best-so-far), repeat ~4–6 generations.

**Scripts:** extend `merge_swe_rlvr_for_eval.py` with weighted/lerp merge +
coefficient sweep; reuse the smoke scorer as fitness. **New:**
`phaseE_rl/evo_merge.py` (population state, selection, crossover, elitism,
manifest per generation), tests for merge-coefficient math.

**Gates:** (a) at least one merged child beats both parents on the fitness slice
(the core evolutionary-merge premise — if never true in gen 1, abort);
(b) final champion validated on the **held-out** eval slice, not the fitness
slice; (c) hard30 floor.

**Estimated gain:** **+2 to +5 resolved**, mostly from merging complementary
adapters (each resolves a different subset — we already saw cp100 and cp300
resolve *different* cases). Cheap way to harvest that complementarity. Unlikely
to exceed Proposal 1 alone, but composes with it (evolve the expert-iteration
lineage).

**Time/compute:** build ~1 day. Per generation: fitness scoring is the cost —
8 adapters × 60-case ≈ merge(20min)+serve+smoke(35min) each, but scoring
parallelizes only by serial GPU serving → ~6–8 h/generation; ~4 generations →
**~2–3 days**. Merges/mutations are cheap; eval-serving is the bottleneck.

---

## Proposal 6 — Immune Memory (negative selection from failure trajectories)

**Why:** cheapest — uses data we already have (thousands of verifier-rejected
trajectories). Complements Proposal 1 (successes) with its mirror (avoid
characteristic failures).

**Exact approach:**
1. **Mine failures.** Cluster rejected trajectories by failure signature:
   read-loop (no edit), empty-finalization (patch.txt without source diff),
   unbounded-deliberation (near-cap reasoning, Rust v2p disease),
   edit-that-changed-nothing, test-tampering.
2. **Build matched pairs.** For instances where we have BOTH a resolved and a
   failed trajectory, form contrastive (chosen=resolved, rejected=failed) pairs.
3. **Preference training.** DPO/KTO-style LoRA on the pairs (preference over
   trajectory prefixes at the divergence point), OR a lighter approach: SFT on
   resolved + an auxiliary "abort-and-retry" behavior distilled from the failure
   taxonomy.
4. **Gate** as usual.

**Scripts:** **New:** `phaseD_sft/mine_failure_signatures.py` (taxonomy +
clustering over banked `runs/*/`), `phaseD_sft/build_contrastive_pairs.py`,
`phaseE_rl/dpo_swe.py` (or reuse a TRL DPOTrainer wired like the GRPO one — same
model-graft/merge scaffolding). Reuse all eval gates.

**Gates:** (a) ≥ 500 matched resolved/failed pairs (else too sparse — fall back
to unpaired failure-avoidance SFT); (b) no regression on edit-reach (must not
just teach "never act"); (c) 60-case resolve ≥ base +3; (d) format not worsened.

**Estimated gain:** **+1 to +4 resolved**, plus a *precision/format* improvement
(fewer degenerate submissions) that may exceed the raw resolve gain — recall the
Rust verbosity and empty-finalization failures are exactly what this targets.

**Time/compute:** build ~1–2 days (mostly the taxonomy/mining). Data prep is CPU
(hours). DPO LoRA ~2–4 h. Eval ~2 h. **~2 days total**, minimal GPU.

---

## Recommended sequencing

1. **Now / next GPU window:** Proposal 1 iteration-1 (small: k=4, 200 instances,
   ~12 h) — fastest path to a real number above cp100. Run the **base-Gemma-4 on
   the 30/60-case fixed harness** in the same window to finally get the honest
   "did the adapter beat base" baseline.
2. **Parallel CPU builds:** Proposal 6 mining + Proposal 4 evo scaffolding (both
   CPU-heavy, GPU-light) while Proposal 1 samples.
3. **Research bet:** Proposal 5 build starts once Proposal 1 establishes a
   stronger base policy (a better Fixer makes the Red-Queen band more meaningful).

**Honest caveat on all estimates:** these are hypotheses with wide error bars on
a 31B solo model at 33%. None is projected to reach frontier (60–75%); the
realistic combined target over a few weeks is **~45–55% on the Lite slice**, and
even that assumes the pool has enough reachable-difficulty tasks. Gains are gated
on measurement that clears the ±3–4 noise floor — hence the 60-case slice is a
prerequisite, not optional.
