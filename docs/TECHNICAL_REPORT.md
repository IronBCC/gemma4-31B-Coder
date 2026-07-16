# Frozen-Base Multi-Language Coding Adapters with Verified-Reward RL and Eval-Hygiene Corrections

**A technical report on Gemma-4-31B as a solo SWE-bench coder (Python · Rust · C++)**

*Internal research log, 2026-07. Numbers below are from internal small-slice
evaluations on our own harness, not official SWE-bench Verified leaderboard
submissions. Read §7 (Honest Positioning) before citing any figure.*

---

## Abstract

We document a methodology for improving a single, frozen open-weight model
(Gemma-4-31B) as a solo software-engineering agent across Python, Rust, and C++,
using per-capability LoRA adapters over an unmodified base with hard
no-regression gates. Over the program we (1) established language baselines,
(2) diagnosed and corrected a *data-mirror* failure in supervised fine-tuning
whereby the student imitates the trajectory *shape* of its training traces,
(3) built a verified-reward RLVR loop that scores edit-decision completions by
executing them in reconstructed container state and running the instance's
fail-to-pass tests, and (4) — our most consequential finding — showed that a
bug in the standard agent evaluation harness was silently suppressing ~3–4
resolved instances per policy, materially distorting measured capability. On a
fixed 30-case SWE-bench Lite slice our best Python policy resolves 10/30 (33%)
under the corrected harness. We explicitly **do not** claim parity with or
superiority over frontier closed models; §7 quantifies the real gap. The
contribution is methodological: a reproducible, evidence-gated pipeline and a
cautionary quantification of eval-harness artifacts.

---

## 1. Objective

One solo model, strong across Python + Rust + C++ with improved multi-step
(agentic) reasoning, achieved via **frozen base + LoRA adapters** with **zero
regression** on existing capability. Every experiment is measured against a
fixed baseline; the base weights on disk are never modified.

Rationale for the constraint: a frozen base with swappable adapters preserves a
known-good fallback, isolates each capability's contribution, and makes
"no-regression" a checkable property rather than an aspiration.

## 2. Experimental setup

- **Model:** Gemma-4-31B-it (bf16 / NVFP4 for serving). Thinking-enabled.
- **Serving:** vLLM, sm120 attention stack; eval serving at the model's full
  context window (131,072), *not* the training budget (a lesson — see §6.4).
- **Agent harness:** mini-swe-agent (bash-only tool, one command per turn),
  Docker per instance, resolution scored by the instance's canonical
  fail-to-pass (F2P) test set.
- **Eval slices:** a fixed 30-case SWE-bench Lite subset (Python promotion
  gate), a 30-case "hard" regression floor, a fixed 75-instance image-backed
  Multi-SWE-bench Rust subset, and a staged 257-instance C++ pool.
- **Standing gate policy:** hard-set is a *regression floor only* (session
  noise measured at ±3–4 on n=30); the 30-case Lite smoke at temperature 0.7 is
  the promotion gate (≥18/30 non-empty patches, resolve improvement, <10%
  format errors).

## 3. Methods

### 3.1 Frozen base + per-capability LoRA
Each capability (Python agentic behavior, Rust, C++) is a rank-32 LoRA over the
frozen base. Adapters compose by ordered merge for evaluation. No base-weight
training occurs at any stage; "no regression" is enforced by re-running the
hard-set floor against a same-day anchor before any promotion.

### 3.2 Data curation and the data-mirror effect
Early SFT mixtures (v6, 49k long-context) ingested external agentic traces whose
median *first source edit* occurred at command ~16. The resulting students
imitated that shape: 24/29 evaluated trajectories never edited a file at all.
We term this the **data-mirror effect** — imitation transfers trajectory
*shape*, not just content. Correction: *edit-first* filtering (first edit ≤ cmd
10, read-streak ≤ 5, no repeated identical commands), content-hash dedup, and
observation compaction to a hard char budget. This lifted non-empty-patch rates
but plateaued at ~11–13/30 (v7, v8): **imitation alone cannot force
act-vs-explore decisiveness.**

### 3.3 Verified-reward RLVR on edit-decision points
To break the SFT plateau we train the specific decision at which the model
historically stalls. Prompts are real trajectory prefixes; completions are
scored by a shaped, execution-grounded reward:

| Reward | Condition |
|---|---|
| 0.0 | no parseable tool call |
| 0.2 | valid bash call, read-only |
| 0.6 | edit command (parse ceiling; also the cap for rows lacking a verifiable fixture) |
| 0.8 | edit **executes** in reconstructed prefix-state container (rc 0, non-empty diff) |
| 1.0 | 0.8 **and** the instance F2P tests pass |

The 0.8/1.0 tiers require a per-prompt **state-reconstruction cache**: the
prefix's read-only commands are replayed in the instance image and committed as
a labelled `rlvr-state:<hash>` image (LRU-evicted under a hard disk floor).
Docker failures degrade monotonically to the parse tier so training never
stalls on infrastructure. GRPO (TRL), fresh LoRA on the SFT-warmed policy,
frozen base grafted from the multimodal checkpoint into a text-only shell.

### 3.4 Gemma-4 thinking-channel fix
`enable_thinking=true` alone degenerates Gemma-4 into `<|turn>model` loops: the
stock chat template never opens the `<|channel>thought` channel. A patched
template that opens the channel (including after tool responses in multi-turn
contexts) restores correct reasoning with zero retraining — verified on the
served path.

### 3.5 Eval hygiene (the harness-artifact correction)
See §4.2 — this is the report's central empirical result and is treated as a
method because it changes how every other number must be read.

## 4. Results

### 4.1 Python: SFT plateau → RLVR
SFT mixtures plateaued at ~11–13/30 non-empty patches. An RLVR A/B (GRPO vs
Dr.GRPO, 30-case Lite, temp 0.7) under the *original* harness: SFT baseline
11/30 patches (6 resolved) → Dr.GRPO 17/30 (7) → GRPO 20/30 (6). GRPO nearly
doubled edit behavior; **resolve stalled at 6–7 across all variants**, and
patch precision fell (55% → 30%): parse-only reward bought decisiveness, not
correctness. This motivated both §3.3 (execution-grounded reward) and §4.2.

### 4.2 The submission-harness artifact (central finding)
Investigating "lost" cases, we found completed, correct work scored as empty.
The agent harness accepted a submission only when a completion marker was the
*first* output line and treated an **empty printed patch as a valid
submission**; trajectories that made real tracked edits (verified by the model's
own mid-run `git diff` showing 1.3–3.8 KB of hunks) but fumbled the
print-the-patch protocol scored zero. Context-window and response-truncation
confounds were ruled out with per-case artifacts.

We implemented a self-retry environment (reject empty submissions with a bounded
corrective turn; nudge when the marker is mis-placed) and re-ran both policies
under it, at the full 131,072 window:

| Fixed harness (30-case Lite, temp 0.7) | cp100 | cp300 |
|---|---|---|
| Non-empty patches | **21** | 18 |
| Resolved | **10 (33%)** | 9 |
| Edit-reach | 27 | 23 |
| Format-error rate | 6.7% | 10.0% |
| Patch precision | 47.6% | 50.0% |

The fix lifted **both** policies by ~3–4 resolves. The long-standing
"resolve stalled at 6–7" narrative was **substantially a harness artifact**, not
a model limitation. This is the report's most transferable result: agent
eval-harness submission logic can silently cap measured resolve rates, and
should be validated (empty-submission and marker-placement edge cases) before
capability conclusions are drawn.

### 4.3 Rust
Raw-base baseline on the fixed 75-instance image-backed subset: **11/75 resolved
(14.7%)**, 61/75 non-empty patches — a healthier raw starting point than Python.
A function-level SFT adapter (v2p; 5k execution-verified rows, benchmark
instances held out to prevent leakage) **regressed** for agentic use: 0/15
resolved vs raw 2/15 on identical stratified instances, 4.3× wall time. Cause:
single-turn function-level SFT taught *unbounded deliberation* — the policy
emitted near-cap reasoning every turn and exhausted its step budget before
editing. This is an out-of-distribution emergent behavior (its training data was
not that verbose), and motivates reasoning-length-controlled data + agentic-trace
mixing for the next iteration. Raw base remains the Rust reference.

### 4.4 C++
257 instances staged and loader-validated; 128 image-backed across three build
families (CMake/make), baseline pending a GPU-serving window. No capability
claim yet.

### 4.5 Exploration dynamics in verified-reward GRPO
Round-1 verified-reward RL never fired the 0.8/1.0 tiers: with 4 samples/group
at temperature 0.9, edits were essentially never sampled at stall prefixes, so
the execution reward had no raw material (an exploration-starvation failure,
confirmed by a served-policy replay: 0/20 edits at stall prefixes, inherited
from the initialization). Round-2 fixes (edit-adjacent prefixes cut one command
before the historical first edit, 5 samples/group, temperature 1.0) reactivated
the verified path (verified-calls > 0, reward means trending up) but remained
weak — edits sparse and the state cache too thin to catch them. Documented as an
in-progress negative-to-marginal result; round-3 levers (prompts cut *at* the
edit, denser state cache) are specified.

## 5. What is novel

We claim **modest, honest** novelty:
1. **Quantified eval-harness artifact (§4.2):** a measured demonstration that an
   agent harness's submission protocol suppressed ~3–4 resolves per policy, with
   the before/after correction. To our knowledge this specific failure mode is
   not documented in the SWE-agent eval literature.
2. **The data-mirror diagnosis (§3.2):** an explicit, measured account of
   trajectory-*shape* imitation (median-first-edit index of training data
   predicting student edit behavior) and the edit-first curation that addresses
   it.
3. **Execution-grounded edit-decision reward with a state-reconstruction cache
   (§3.3):** shaped RLVR tiers that require the model's proposed edit to actually
   execute and pass F2P tests in reconstructed prefix state, engineered to
   degrade safely under Docker failure.

The underlying tools (LoRA, GRPO, rejection sampling, execution verification)
are standard; the contribution is their integration under a frozen-base,
zero-regression, evidence-gated discipline, plus the two empirical findings.

## 6. Engineering lessons (reproducibility-relevant)

1. **Verify capability before declaring incapacity.** The "model can't think"
   conclusion was a broken chat template, not a model limit (§3.4).
2. **Check the artifact, not the proxy.** Multiple "regressions" were
   measurement artifacts (submission protocol, PATH reset in login shells
   dropping toolchains, out-of-credit CLI responses reported as `success`).
3. **Separate training budget from eval serving.** Serving at the 49k training
   budget instead of the 131k model window produced 400-errors on long Rust
   prompts and broke baseline parity.
4. **Cost-gate long runs.** GPU runs are staged with memory probes (num_gen 8 →
   OOM, 6 → 0.2 GB headroom, 5 → fits) and >6h authorization gates; three cheap
   probes prevented multi-hour OOM losses.

## 7. Honest positioning and limitations

**We do not beat frontier models.** Frontier closed systems report ~60–75% on
full SWE-bench Verified; our best measured Python figure is 10/30 (33%) on a
30-case Lite slice — a different, smaller, easier benchmark subset, on a 31B
open model run solo. The numbers here are **internal small-n evaluations** with
±3–4 session noise at n=30; they are not leaderboard results and should not be
compared directly to published Verified scores.

What we can defensibly say: on our fixed slices, evidence-gated LoRA training
over a frozen Gemma-4-31B improved agentic Python patch behavior and held a
regression floor, established Rust/C++ baselines, and — most usefully to the
community — showed that eval-harness hygiene alone moved measured resolve rates
by ~40% relative. The frontier gap is real and large; closing it is future work
(the deferred frontier-teacher distillation program, denser verified-reward
coverage, and language-specific agentic data).

## 8. Artifacts

Training/eval code, reward and harness implementations, per-instance result
JSONL, state-cache manifests, and the fixed eval subsets are in the repository
(`phaseD_sft/`, `phaseE_rl/`, `phaseH_eval/`, `phaseA_scaffold/`). Every headline
number above traces to a committed results artifact and a dated `SESSION_STATE`
entry.

---

*Draft — honest internal report. Do not present §4 figures as SWE-bench Verified
results. Frontier-superiority claims are unsupported and are not made here.*
