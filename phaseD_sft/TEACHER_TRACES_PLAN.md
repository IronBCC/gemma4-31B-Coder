# Teacher-Trace Collection Plan (GPT-5.6 + Claude Fable 5 → Gemma-4-31B)

Status: FUTURE PROJECT (deferred 2026-07-14 by user: revisit AFTER current stage
succeeds — Python behavior/resolve lever via verified-reward RL, Rust/C++
baselines + adapters. Design + prereq verification banked here; nothing running.)
Owner: Claude (design) / Codex (execution candidate)
North star: best SOLO SWE coder, frozen base + LoRA, zero regression.

---

## 0. Goal

Collect high-quality coding trajectories from two frontier teachers — **GPT-5.6**
(OpenAI) and **Claude Fable 5** (Anthropic) — verified by execution, converted to
our Gemma-4 training schema, and mixed into the next SFT round (v9). Two tracks:

- **Track A — agentic SWE traces (Python):** multi-turn bash-agent trajectories
  over verifiable SWE task pools. Targets the proven SFT plateau: v7/v8 imitation
  of mid-tier external traces ceilinged at ~11-13/30 patches; RL doubled behavior
  but resolve stalled at 6-7 with precision falling 55%→30%. The missing
  ingredient is *correct* decisive trajectories — frontier teachers supply them.
- **Track B — general-programming reasoning traces:** single-turn (or short)
  problem → thinking → solution → unit tests pass. Cheap per-trace, high volume,
  directly targets the "improved multi-step reasoning" clause of the north star,
  and produces native content for Gemma-4's `<|channel>thought` (task #6).

## 1. Reference: how the Chinese labs do it

| Lab / artifact | Recipe |
|---|---|
| Kwai **Klear-Agent** (66k dataset we already ingested) | mini-swe-agent harness + strong teacher over SWE-smith pool → keep only execution-**resolved** trajectories → SFT on THOUGHT+action. 8B student hit 39% Verified. |
| **Qwen3-Coder** | "agentic data flywheel": ~20k parallel sandboxes, teacher rollouts on synthesized verifiable tasks, execution-verified rejection sampling, hard decontamination vs benchmarks. |
| **DeepSeek R1-Distill** | teacher samples N per problem → rule/test verification keeps winners → student SFT on full reasoning traces verbatim (Track B pattern). |
| **Kimi K2** | large-scale agentic synthesis: task gen → sandbox rollout → rubric + execution filter → staged SFT. |

Shared core = **verifiable task pool + sandbox rollouts + execution-based
rejection sampling + format normalization + decontamination**. Nobody keeps
unverified traces (SWE-ZERO trap, already rejected in Phase 2b).

## 2. Teachers

**Access mode (decided 2026-07-14): SUBSCRIPTIONS, not pay-per-token APIs.**
Teachers are driven through the CLI agents already authenticated in this project:

| | GPT-5.6 | Claude Fable 5 |
|---|---|---|
| Access | **Codex CLI** (`codex exec`, headless) — already running on the box via Orca | **Claude Code** (`claude -p` headless, or Agent-tool subagents in the coordinator session) |
| Raw trace location | `~/.codex/sessions/*.jsonl` — commands + reasoning **summaries** | `~/.claude/projects/<proj>/*.jsonl` — messages + tool calls; **thinking text NOT persisted** (verified 2026-07-15: stored as `"thinking":""` + signature only) |

**RESOLVED BLOCKER (verified live 2026-07-15):** Fable-5/Sonnet-5 thinking text
is redacted at the API layer — stream-json emits the thinking block with ONLY
`signature_delta` (no `thinking_delta`), in every output mode, model override
confirmed working (`--model claude-fable-5`, cost event present). Hidden-channel
capture is DEAD for this generation.
**GPT-5.6 path (researched + LIVE-PROBED 2026-07-15):** `codex exec --json
--skip-git-repo-check "<prompt>"` streams JSONL: `thread.started`/`turn.*` events
and `item.completed` with `item.type` in {`agent_message`, `command_execution`}.
CORRECTION (probe overturned the docs): `reasoning` items are NOT emitted for
subscription auth even at `model_reasoning_effort=high` (0 reasoning items on a
reasoning-heavy prompt; reasoning folds INLINE into `agent_message` text). So
GPT-5.6 is NOT a hidden-thinking source — SAME visible-reasoning pivot as
Fable-5. Still worth collecting as a diversity teacher (different solve style).
Probe facts: `--skip-git-repo-check` required (guards non-git dirs); box has
node v20 + npm so codex installs there (no Mac hop); auth via existing
`~/.codex/auth.json`. Shell access for docker exec needs `--sandbox
danger-full-access`. Credit-guard string for codex UNVERIFIED (didn't hit the
wall) — make it generic + confirm on first real batch.
- Model/effort: `-c model="gpt-5.6-terra" -c model_reasoning_effort="high"`
  (box `~/.codex/config.toml` defaults gpt-5.5/high; orchestration worker runs
  gpt-5.6-terra — confirm exact id at launch).
- Shell access: agent must `docker exec` on the host (outside cwd) → needs
  `--sandbox danger-full-access` (isolated box only). Same bash-only container
  discipline as the Claude driver.
- LOGISTICS: `codex` binary lives on the Mac (v0.144.1), NOT the box (box has
  only `~/.codex/{auth.json,config.toml,sessions/}`). Options: (a) install codex
  on box, (b) run on Mac with an `ssh ironbccllm docker exec …` hop in the
  prompt, (c) reuse the existing Codex orchestration worker (it IS gpt-5.6 with
  fluent box ssh). Shares ONE subscription quota with the active worker → run in
  a fresh quota window alongside the Fable remainder; do both teachers together.
- Driver: add a `codex` backend to teacher_trace_driver.py (parse `reasoning` +
  `command_execution` items; same F2P scoring + credit-guard pattern).

**OPEN-SOURCE Fable-5 traces (searched 2026-07-15) — NOT a shortcut:** several
brand-new HF community dumps exist (suayptalha/fable-5-claude-code = 63 rows, no
license, no execution verification, no thinking, mixed domains; Glint-Research,
cfahlgren1, armand0e similar raw Claude-Code session logs). All fail our three
hard bars (execution-verified / license-clean / thinking-bearing) — audit-only
curiosities, our own verified collection stays the trunk. Only serious external
option remains `nvidia/Open-SWE-Traces` (200k+, but Minimax/Qwen teachers, not
Fable) — already in Phase-2b scouting.

**PIVOT (= what Kwai actually did):** collect VISIBLE reasoning — the teacher is
instructed to write concise reasoning in its response text before each command
(THOUGHT + action). Capturable from message text, trainable as CoT/thought
content, and the reasoning style is PRESCRIBABLE at collection time (terse,
caveman-brief — implements the thinking-compression goal at the source instead
of post-hoc pruning).
| Thought-channel value | thin — summaries usable as short THOUGHT | **high — real thinking → `<|channel>thought` verbatim** |
| Marginal cost | $0 (subscription quota) | $0 (subscription quota) |

Consequences of subscription mode:
- **Throughput bounded by quota windows**, not dollars: parallelism 2-4 agents,
  P2 runs over days, pilot unaffected. Cost section 7 becomes a quota model.
- **Schema constraint**: our training format is bash-only tool calls. CLI agents
  default to rich toolsets (Read/Edit/Write...). Fix at the source, not in
  conversion: Claude Code headless runs with `--allowedTools "Bash"` (bash-only
  forced); Codex is shell-native already. Bash-only teacher traces convert
  losslessly to our schema.
- **Container access pattern**: CLI agent runs on host, instructed to execute
  every command via `docker exec <cid> bash -c '...'` (non-login shell — the
  `-lc` PATH-reset lesson). Patch harvested by `git diff` inside container;
  verification by the image's canonical test run. Same skeleton as
  `rust_baseline_driver.py`, with the CLI agent replacing the OpenAI loop.

Implication: **Fable 5 is the primary Track-B thinking teacher**; GPT-5.6 is a
diversity/coverage teacher (different failure modes, different edit styles). Keep
per-teacher provenance in every row — mixture ablations later.

## 3. Task pools (Python + general; all verifiable, all decontaminated)

Track A (agentic):
1. **SWE-smith Python** (primary): ~12k+ task instances with F2P tests + prebuilt
   docker images. Synthetic bugs in mirror repos → no overlap with SWE-bench by
   construction. Already our anchor source; licensing vetted (MIT).
2. **SWE-Gym**: real-repo Python tasks with tests, images available. Secondary.

Track B (single/short-turn reasoning):
3. **Test-verified problem pools**: KodCode / TACO / LiveCodeBench-style pools
   with executable unit tests (license check per pool at pull time). Teacher
   solves with thinking on; keep only test-passing solutions.
4. Optional: **error→fix pairs** mined from Track A failures (teacher's own
   near-miss + its correction turn) — free byproduct, matches the repair recipe.

Decontamination (hard, blocking):
- Exclude by instance_id AND (repo, PR/commit): the 30-case SWE-Lite smoke slice,
  hard30, SWE-bench Lite/Verified entirely (eval-reserved), and the 75-Rust /
  257-C++ eval subsets (future-proofing).
- Track B: exclude any problem whose canonical source appears in our eval
  benches; dedup content-hash vs existing corpus.

## 4. Harness

**Track A (subscription mode): CLI-agent rollout driver.**
Per instance, a thin host-side driver (new: `phaseD_sft/teacher_trace_driver.py`,
skeleton = `rust_baseline_driver.py` with the OpenAI loop swapped for a CLI
subprocess):
1. `docker run -d <swe-smith image> sleep infinity` → cid.
2. Launch teacher headless with the task prompt:
   - Fable 5: `claude -p "<prompt>" --allowedTools "Bash"` (bash-only forced),
     or Agent-tool subagents when the coordinator session drives collection.
   - GPT-5.6: `codex exec "<prompt>"` (shell-native).
   Prompt = `<pr_description>` + instructions: operate ONLY via
   `docker exec <cid> bash -c '...'`, decisive-edit guidance (same wording family
   as the Rust driver: first edit by ~cmd 15, no repeated reads, finish marker).
3. Harvest: `git diff` in container = patch; canonical in-image test run =
   resolved verdict; session transcript (`~/.claude/projects/.../*.jsonl` or
   `~/.codex/sessions/*.jsonl`) = raw trace, copied into the raw layer keyed by
   instance_id. VERIFY in P1 that thinking blocks actually appear in every
   Claude transcript turn (artifact, not proxy).
4. Config: step cap ~40 (enforce via prompt + `--max-turns` where supported),
   obs truncation guidance in prompt (2,000 chars, `| head -c 2000` idiom).
- Precedent note: Kwai-Klear used mini-swe-agent + teacher API — same recipe,
  different transport. mini-swe path remains the API-mode fallback if quota
  becomes the bottleneck.

**Track B — trivial harness:** one (or few) API call(s) per problem with thinking
enabled → run pool's unit tests in a container → keep pass. N=1 first;
rejection-sample N=4 only on high-value pools where N=1 yield is low.

**Verification:** SWE-smith official grading (F2P/P2P via its harness images);
disk watchdog active (docker loopback fill lesson); exact-PID process hygiene.

## 5. Capture → conversion → gates (the proven chain, reused)

1. **Raw layer (immutable):** per-instance full trace: messages, tool calls,
   observations, raw API JSON (thinking/reasoning), resolved verdict, token/cost
   accounting. Store under `data/teacher_traces_raw/<teacher>/<pool>/`.
2. **Convert** to our schema (extend `convert_external_traces.py` patterns):
   - assistant turns = `tool_calls` bash JSON; env output = user `OBSERVATION:`.
   - **Thinking compression (banked 2026-07-15, user-requested analysis):**
     Fable-5 thinking blocks will be long; compress BEFORE rendering into the
     student channel — TokenSkip-style importance pruning (preferred) or
     Chain-of-Draft-style terse re-drafting; naive filler-dropping measured at
     only ~18% savings on terse traces, so reserve rule-based stripping for a
     first pass only. Evidence: current Python policy thinks at median 23
     tokens (no compression needed); Rust v2p's 4k rambling was OOD-emergent,
     not data-taught — compression is a distillation-phase lever, not a
     current-lane fix.
   - **Thinking rendering — build BOTH variants** from the same raw layer:
     (a) plain-text CoT in assistant content (current v8 convention);
     (b) `<|channel>thought` native rendering via thinkopen-v2 template
     (resolves task #6 empirically — train small ablation on each).
   - Strip teacher-isms: model self-references, API artifacts, refusal
     boilerplate, teacher-specific tool names.
3. **Quality gates (identical to v6/v7/v8 chain):**
   - resolved-only keep (Track A) / tests-pass-only (Track B);
   - edit-decisiveness filters at ingest: first source edit ≤ cmd 10, read-streak
     ≤ 5, no identical repeated commands (frontier teachers ramble too — filter,
     don't assume);
   - compaction hard cap 2,400 chars/observation; trim-to-last-assistant; exact
     content-hash dedup (incl. vs existing corpus hashes);
   - token budget 49,152 (Gate-B verified ceiling);
   - `verify_gemma_format_loss.py` full pass, 0 failures — blocking.
4. **Mixture policy:** teacher traces are ONE source with a cap (≤50% of
   mixture); anchors (oracle edit traces, coder_repair) stay. Per-source counts
   in manifest. Per-teacher tag kept for ablation.

## 6. Phases + gates

**P1 — pilot (gate: numbers before money).**
- Track A: 50 SWE-smith instances × both teachers × 1 rollout.
- Track B: 200 problems × both teachers × 1 sample.
- Measure: resolve/pass rate per teacher, $/resolved-trace, median first-edit
  index, trace token length distribution, thinking-content quality (spot-read 10).
- Budget: ~$100-300. Batch API where available (−50%).
- GO gate: $/resolved-trace and yield support P2 within budget; traces pass
  format gate after conversion.

**P2 — scale (gated on P1).**
- Target ~2-3k resolved Track-A traces + 5-10k verified Track-B traces.
- Rollouts sized by P1 yield (e.g., 40% resolve → ~7k Track-A rollouts).
- Retry policy: N=2 only on near-misses (patch applied, some tests passed).
- Budget: rough $2-6k — REQUIRES explicit user approval with P1 numbers in hand.

**P3 — SFT v9 + standard gate ladder.**
- Build v9 mixture (both thinking-rendering variants as separate small runs
  first; pick winner). Existing pipeline end-to-end: compaction → trim/dedup →
  budget → format gate → bounded launcher → hard30 floor → 30-case temp-0.7
  smoke (≥18/30 non-empty, <10% format) → Verified slice.
- Also: Track-A stall-point prefixes feed RLVR v2+ dataset (teacher-quality
  decision points > mid-tier ones).

## 7. Cost model (subscription-quota, pilot corrects)

- Marginal $ = 0; the resource is **subscription quota windows** (both CLIs
  rate-limit per 5h window) and wall-clock.
- Track A rollout: ~25-40 turns → est. a handful of rollouts per teacher per
  quota window at parallelism 2-4. P1 (50 instances × 2 teachers) ≈ 2-4 days of
  background collection. P2 (~7k rollouts) is weeks-scale on subscription alone
  → P1 also measures rollouts/window so P2 can decide: stay on subscription
  (slow, free) vs top up with API batch mode (fast, $2-6k) vs mix.
- Track B sample: cheap (1-3 calls) — thousands per week feasible on quota.
- Levers: run collection off-hours, interleave teachers (independent quotas),
  keep step cap tight, resume-safe driver (per-instance completion marker) so
  quota exhaustion never loses work.

## 8. Risks

1. **ToS**: OpenAI + Anthropic terms restrict using outputs to train competing
   models. Business decision = user's; flagged once, not re-litigated here.
2. **Format leakage**: teacher-isms in traces → student mimics wrong surface.
   Mitigation: strip pass + format gate + spot-reads in P1.
3. **Cost overrun**: hard-gated by P1 measurement + explicit P2 approval.
4. **Thinking round-trip bugs** (Anthropic multi-turn tool use requires thinking
   blocks preserved): verify in P1 on 3 instances before batch; check raw
   sidecar actually contains thinking on every turn (artifact, not proxy).
5. **Data-mirror repeat**: v6 lesson — model imitates trajectory *shape*. Even
   frontier traces get the edit-decisiveness audit before inclusion.

## 9. Deliverables

- [ ] `phaseD_sft/teacher_trace_driver/` — harness choice (mini-swe patch or
      generalized driver) + raw-capture sidecar
- [ ] `phaseD_sft/convert_teacher_traces.py` — raw → our schema, both thinking
      renderings, teacher-ism strip
- [ ] `data/teacher_traces_raw/` layout + manifests (per teacher/pool/date)
- [ ] P1 pilot report: yield, cost, quality spot-reads → user P2 decision
- [ ] v9 mixture manifest + ablation plan (thinking-render A/B, per-teacher mix)

## 10. Decisions

1. ~~Budget~~ RESOLVED 2026-07-14: subscription mode (Claude Code + Codex CLIs),
   $0 marginal; P1 approved.
2. ~~ToS~~ RESOLVED 2026-07-14: user accepted.
3. ~~Thinking-render~~ RESOLVED 2026-07-14: **A/B ablation** — two small LoRA
   runs (plain-CoT vs native `<|channel>thought`), gate both on hard30 +
   30-case smoke, keep winner for v9.
4. OPEN: Track B pool selection final (license check at pull).
5. **P1 launch: ON HOLD (user, 2026-07-14)** — prereqs verified (CLIs, pool
   schema, disk), driver not yet built; do not start collection until explicit
   go. Lane = task #10.
