# Rust v2 data plan — post-RL GPU window

## Scope and scheduling

This is a **data plan only**.  It does not authorize corpus construction,
tokenization, adapter training, Docker work, or GPU use while the Python
verified-reward GRPO round is active.  Execute only in the post-RL GPU window
after the user has approved task #13.

The goal is a Rust adapter that improves the real Multi-SWE-Rust repair task
without contaminating its benchmark.  The raw-base reference already recorded
11/75 resolved on the fixed image-backed Rust subset; that set must remain a
clean comparator for every adapter.

## Streaming audit: `rlvr_rust_verified.jsonl`

Audit performed by a single streaming JSONL pass; no tokenizer was loaded.

| Property | Result |
| --- | --- |
| Path | `data/rlvr_rust_verified.jsonl` |
| Rows / bytes | 106,454 / 295,583,237 bytes |
| Schema | `id`, `problem`, `solution`, `tests`, `n_asserts` |
| JSON errors | 0 observed in streaming audit |
| Exact duplicate IDs | 0 |
| Tagged task-type distribution | Not available: no `task_type`, `source`, `dataset`, `origin`, or language field is present in the verified rows |
| Verification meaning | Every retained row compiled and passed its supplied `assert_eq!` checks in the pinned Rust verifier; it is not a repository-patch trajectory corpus |
| Assert-count shape | 1–4: 9,575; 5–10: 46,924; 11–20: 47,005; 21+: 2,950.  The bank is dominated by 5–20 assertion examples. |

The parent raw dataset metadata identifies the source only as
`rlvr-code-data-rust` (133,296 raw rows before execution filtering).  Its
metadata has empty `license`, `homepage`, and `citation` fields.  Treat this as
an unresolved provenance/license item: retain the source snapshot and its
dataset-info JSON in the manifest, and do not claim an OSI license without a
primary-source confirmation.

## Relationship to existing Rust SFT inputs

| Artifact | Rows | Relationship |
| --- | ---: | --- |
| `data/rlvr_rust_verified_snap1.jsonl` | 19,885 | All 19,885 unique IDs are contained in the current 106,454-row bank. |
| `data/rust_sft_v1` | 19,529 | Cardinality is consistent with `build_rust_sft.py`'s shortest-correct dedup over `snap1` (356 rows removed). |
| `data/rust_sft_v2` | 102,810 | Cardinality is consistent with the same builder over the full bank (3,644 rows removed, 3.4%). |

Thus the full bank substantially overlaps the v1 inputs.  A v2 selection must
deduplicate against the normalized-problem and normalized-solution keys used
by `build_rust_sft.py`, not merely against `id`; the existing v1 slice must not
be treated as new training material.

## Multi-SWE-Rust and repair-source decision

`data/mswe_rust_native/` and `data/mswe_rust_prs/` each contain 239 rows across
10 repositories.  They are the same benchmark task family (native schema
contains `fix_patch`, `test_patch`, F2P/P2P fields and `instance_id`), not an
independent free training corpus.

**Default decision: do not include any `mswe_rust_native`/`mswe_rust_prs` row
in Rust v2 training.**  In particular, the 75 image-backed rows used by the
raw-base baseline are held out permanently.  Training on the native pool would
turn the Rust comparison into benchmark leakage.  A future user-approved split
could reserve a fixed held-out set and use only disjoint native instances, but
that changes the benchmark claim and needs a new baseline first.

The only known repair-pair candidate is the 768-row
`coder_repair_synthetic` component in `data/v6_build/anchor_v6.jsonl`.  It is
not Rust-tagged.  It is therefore a candidate, not an assumed source: include
only rows that a later CPU scan proves are Rust code-repair pairs, have an
assistant patch/code target, and do not share an `instance_id`, repository, or
patch with the held-out Multi-SWE-Rust pool.

## Proposed v2 mixture

The initial target is deliberately modest enough to inspect and audit before
training.  Counts below are pre-budget targets, not permission to build now.

| Component | Target | Selection rule | Purpose |
| --- | ---: | --- | --- |
| Execution-verified RLVR Rust bank | 5,000 | Deterministic seed; exclude v1 normalized problem/solution keys; stratify across 1–4, 5–10, 11–20, and 21+ asserts; exact-ID dedup | Correct Rust syntax, APIs, and test-oriented reasoning |
| Rust-qualified repair pairs | Up to 768 | Mine only from `coder_repair_synthetic` after Rust/content and held-out-overlap checks | Edit/repair behavior absent from standalone functions |
| Multi-SWE-Rust native / PR rows | 0 | Evaluation-only unless a separate held-out split is user-approved | Prevent benchmark leakage |
| Backfill | RLVR only | If fewer than 768 repair rows qualify, replace the shortfall with more stratified RLVR rows; do not duplicate native rows | Keep the plan safe and reproducible |

This yields 5,000–5,768 rows before dedup and the 49,152-token filter.  The
manifest must record component input, rejected, duplicate, selected, and
post-budget counts, selection seed, raw source snapshots, and the explicit
`mswe_rust_train_rows=0` leakage guard.

## Build and gate chain (when the post-RL window opens)

1. **CPU source gate.**  Build the candidate as Gemma messages, reject missing
   assistant supervision, non-Rust repair pairs, and any overlap with the
   held-out Rust `instance_id`/repo/patch inventory.  Write a source manifest.
2. **Dedup/trim gate.**  Run `dedup_and_trim_traces.py` with passthrough only
   where justified, then re-run the v1 normalized-problem and solution dedup.
   Report intentional versus accidental duplicates separately.
3. **Token gate.**  Run `filter_token_budget.py --max-tokens 49152` using the
   Gemma tokenizer loaded once in the main thread and a `ThreadPoolExecutor`
   only.  Save accepted, rejected, and manifest artifacts.  This step waits
   until the training RAM window is free; no tokenizer sweep during the active
   GRPO run.
4. **Format gate (blocking).**

   ```bash
   PYTHONPATH=$PWD .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py \
     --data data/unsloth_rust_v2_budget49152 --samples 1000 --workers 1
   ```

   Require exit 0 and `failure_count: 0`.  Any failure stops the lane for a
   data/rendering fix; do not train through a format failure.
5. **GPU smoke gate.**  After GPU1 is free and production ports are green, run
   one 49,152-context step from the approved frozen-base/LoRA init with the
   sm120 attention path.  Require checkpoint-1, no OOM, and GPU1-only use.
6. **Small train/eval gate.**  Train a bounded checkpoint increment, then
   evaluate only on the held-out Rust set: smoke first, then the fixed-75
   baseline protocol with the same image/build-test rules.  Compare resolved,
   patch, infra-flake, and F2P/P2P transitions against raw base 11/75.
7. **Promotion gate.**  Scale or change the mixture only after the held-out
   Rust behavior improves without a Python hard30/SWE-Lite regression.  Keep
   all adapter merges evaluation-only and the base frozen.

## Open decisions before execution

- Confirm the authoritative license/provenance for `rlvr-code-data-rust`.
- Approve or reject a disjoint native-Rust train/eval split; absent approval,
  native rows remain evaluation-only.
- Audit how many of the 768 generic repair rows are genuinely Rust and
  leakage-free.  If the result is too small, acquire a disjoint Rust repair
  source rather than duplicating benchmark rows.
