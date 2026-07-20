# Rust v3.1 Gold-Path Decisive-Suffix Dataset Design

## Purpose

Rebuild the Rust agentic SFT corpus so it teaches instruction-following source edits,
bounded verification, and clean submission rather than the over-testing and forbidden
test-edit behavior observed in the Rust W4 baseline.

The existing `data/rust_sft_v3_agentic_49k` remains immutable evidence and must not be
used for training. Its mechanical gates pass, but its retained edit-to-submit suffixes
are behaviorally unsafe: every prompt forbids test changes, while 96/140 rows edit a
builder-classified test, fixture, example, or bench path. The new output is published
under a distinct v3.1 path and only after every gate below passes.

## Scope

### In scope

- The revision-pinned `nvidia/Open-SWE-Traces`, `openhands/qwen35_122b` source used by
  Rust v3.
- The same frozen 239-task exclusion set and leakage checks.
- Resolved Rust trajectories with a non-empty row-local `metadata.model_patch.patch`.
- Mutation allowlisting, behavioral suffix construction, deterministic per-task
  selection, token budgeting, format/loss verification, and audit manifests.

### Out of scope

- Training or serving an adapter.
- Adding new source datasets merely to satisfy a row-count target.
- Fabricating commands, observations, reasoning, return codes, or test outcomes.
- Editing the published Rust-v3 dataset in place.
- Relaxing the 239-task exclusion, tool pairing, grounding, or Gemma-format gates.

## Alternatives considered

1. **Hard-filter the final 140 rows.** This is simple but discards the earlier candidate
   pool and is expected to leave only about 35-45 rows.
2. **Delete unsafe turns from the final 140 rows.** This retains volume but can leave
   later commands dependent on files or state that no longer exist.
3. **Rebuild from the 294 compressed candidates using a gold-path allowlist and a
   decisive suffix.** This is selected. It rejects state-contaminated trajectories
   instead of trying to repair them and reselects the best row per task afterward.

## Authoritative mutation allowlist

For each resolved source row, parse its own unified `model_patch.patch` with the existing
strict patch parser. The normalized file paths in that patch are the only repository
paths that a retained command may mutate.

Additional hard rules apply even when a path appears in the final patch:

- Reject the row if any retained repository mutation targets a test, testing, fixture,
  example, bench, or benchmark path under the existing `_is_test_path` semantics.
- Reject lockfile mutations (`Cargo.lock`, `supply-chain/imports.lock`, and other files
  whose basename ends in `.lock`) because the target behavior is narrow source repair.
- Reject repository-local summary, demo, reproduction, or verification artifacts that
  are absent from the final patch.
- Permit legitimate non-`.rs` repository files only when they appear in the row-local
  final patch and are not excluded by the rules above. This preserves valid changes to
  manifests, configuration, generated interfaces, and C/C++ bridge files.
- Reject a row on any mutation path that cannot be normalized confidently. Do not guess
  whether an ambiguous path is inside or outside the repository.

The allowlist is row-local. A path from another trajectory or another task can never
authorize a mutation.

## Scratch reproduction policy

Scratch mutations outside the declared repository root are not part of the final patch
allowlist. At most one coherent scratch reproduction chain may be retained:

1. one command creates or updates the scratch artifact;
2. a later retained command executes or consumes that exact normalized path; and
3. both commands and their paired observations occur before the final repository edit.

Reject the row if scratch state is ambiguous, repeatedly rewritten, never consumed, or
used after the final repository edit. Scratch files cannot be used to justify modifying
repository tests or examples.

## Decisive suffix construction

Start from the existing compressed trajectory and preserve all retained assistant
reasoning text verbatim. Never rewrite an assistant command or observation.

1. Keep the system and task turns plus the existing grounded pre-edit read pairs.
2. Keep all paired turns from the first allowlisted repository mutation through the
   final allowlisted repository mutation.
3. Reject the row if any command in that interval performs a forbidden, ambiguous, or
   non-allowlisted repository mutation.
4. After the final allowlisted mutation, retain only read-only inspection and Rust
   verification pairs until the first trusted successful verification.
5. A trusted verification must execute `cargo check`, `cargo test`, `cargo build`,
   `cargo clippy`, `cargo nextest`, `rustc`, or an existing project test/build target;
   its paired observation must carry an explicit zero return code. A shell pipeline is
   trusted only when `pipefail` is explicitly enabled. Output text alone is not proof.
6. Reject the row if no trusted successful verification follows the final mutation.
7. After that verification, retain at most one read-only `git diff` or `git status` pair
   and the original terminal submission turn. Drop all later redundant exploration,
   scratch creation, test creation, summaries, and repeated verification.
8. Reject identical edit commands anywhere in the retained trajectory, not only when
   consecutive.

Every retained tool call keeps its original paired observation. The terminal submission
must remain last. Any dependency or pairing ambiguity rejects the row rather than
creating a synthetic repair.

## Selection and publication

- Apply the behavioral gates before deterministic shortest-per-task selection so a safe
  alternate trajectory can replace an unsafe shorter one.
- Deduplicate by canonical rendered-message content hash.
- Select one row per task, preferring fewer rendered tokens, then the existing stable
  identity tie-break.
- Apply the hard 49,152-token ceiling with one main-thread-loaded Gemma tokenizer and
  bounded thread-only token counting.
- Publish atomically to `data/rust_sft_v3p1_agentic_49k` without overwriting any existing
  directory.
- Write a manifest that binds the source revision, builder, tokenizer, template, frozen
  exclusions, input boundaries, every drop reason, per-repository counts, token stats,
  first/final edit positions, verification positions, and artifact hashes.

## Blocking gates

The build must stop without publication unless all invariants pass:

- 239 exclusion IDs: zero overlap.
- Tool-call/observation pairing: balanced for every row.
- Gemma format/loss verifier: `failure_count=0`; no fallback spans.
- Token count: every rendered row is in `1..49,152`.
- Repository test/fixture/example/bench mutations: zero.
- Lockfile mutations: zero.
- Non-allowlisted or ambiguous repository mutations: zero.
- Repeated identical edit commands: zero.
- Grounded first source edit: 100%.
- Trusted successful Rust verification after the final mutation: 100%.
- Terminal submission last: 100%.
- First-edit median: at most command 4.
- Output content and task IDs: unique.

Row count is an evidence gate, not a quota to pad. A yield of at least 60 rows across at
least 35 repositories is sufficient for a training proposal. If fewer than 60 survive,
stop and report the drop table; expanding source partitions requires a separate design.

## Tests

Focused unit tests must prove:

- row-local final-patch allowlisting for Rust and legitimate non-Rust paths;
- hard rejection of repository test/example/fixture/bench and lockfile mutations;
- rejection of ambiguous or non-allowlisted mutation paths;
- coherent one-chain scratch reproduction handling;
- trusted verification parsing, including rejection of masked pipeline return codes;
- final-edit-to-first-success suffix truncation without altering retained messages;
- rejection of nonconsecutive repeated edit commands;
- balanced pairing and terminal-last invariants after truncation;
- gate-before-selection behavior so a safe alternate can replace an unsafe candidate;
- deterministic atomic no-replace publication and complete manifest accounting.

The existing full Phase-D suite must remain green after the focused tests pass.

## Expected cost and decision

The source scan covers 55,488 rows but only 294 current compressed candidates. On the
existing host this is expected to take roughly one to two CPU hours including tokenization
and the 1,000-sample/all-row format gate. No GPU, Docker, service, or production-port
operation is part of the build.

Passing this design produces a training proposal, not training authorization. Rust
adapter training remains sequenced after the honest 239-case raw-base baseline and its
own smoke gate.
