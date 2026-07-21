# Proposal: decouple execution verification from edit-decisive ingestion

Author: teacher-SFT lane, revised against the implemented `teacher_platform` trust boundaries
Date: 2026-07-21
Status: proposal only; no replay or training is authorized by this document
Related: `FABLE5_DATASET_AUDIT.md`, `fable5_import.py`, `fable5_replay.py`

## Decision summary

The original observation is valid: the current importer answers two different questions with one
strict parser gate:

1. Can the published trajectory be converted into attributable, edit-decisive Gemma supervision?
2. Does the final change made by that trajectory actually solve the pinned task?

`fable5_import.py::_audit_read_only_bash` should remain narrow for question 1. It rejects shell
composition, redirection, dynamic execution, in-place mutation, external paths, and commands outside
an exact read-only grammar. That protects the training shape and the declarative reconstruction
contract.

Question 2 can be measured separately, but **raw trajectory commands must not run inside the trusted
verifier worktree**. The safe design is:

> isolated untrusted replay -> quiescence -> canonical final delta -> fresh deterministic
> reconstruction -> existing trusted controls and verifier -> normalized ingestion

There is no training path for an unnormalized `shell-style-verified` row.

## Current infrastructure and trust boundary

The implemented Fable path already provides these components:

- `canonical_mutation_plan` accepts only typed `Write` and `Edit` mutations. Read-only operations and
  one exact seed verifier command may be observed, but do not mutate the reconstructed candidate.
- `reconstruct_candidate` applies the typed plan to a fresh pinned seed and binds the resulting tree,
  diff, changed paths, protected-file hashes, and operation hash.
- `DockerExecutor.execute` accepts an already reconstructed `candidate_root`; it copies that state
  into a restricted container and runs one trusted verifier command. It is not an arbitrary
  trajectory runner.
- `run_control_set` requires a clean baseline failure, a clean reference success, and two candidate
  reconstructions with the same non-empty state and two valid verifier successes.
- The container policy is digest-pinned, networkless, read-only at the root, capability-free,
  non-root, cgroup-bounded, output-bounded, disk-floor guarded, and exact-CID cleaned.

The verifier-owned copy under `/work` is writable. Protected-file hashes are checked around the
trusted verifier, but a final hash is not sufficient protection when arbitrary commands have run in
the same worktree: an untrusted process could alter tests during verification and restore them
before the final hash. Consequently, the existing executor must remain a **trusted-verifier-only**
component.

## Corrected architecture

### Pass 0: pinned environment eligibility

Join each terminal row to an exact public Moonshiner seed under `tasks/seeds/<task>`. The published
Fable row does not embed its verifier, protected paths, reference patch, or image, but the pinned
Moonshiner repository provides those seed artifacts for joinable task IDs.

Before raw replay, require:

- pinned dataset revision, Moonshiner commit, seed tree, task tree, and source-row hash;
- exact task schema, verifier command, timeout, protected paths, and reference-patch hash;
- a locally cached, digest-pinned image with the required language toolchain;
- an unmodified baseline that fails the trusted verifier cleanly; and
- the pinned reference patch that applies and passes the trusted verifier cleanly.

An environment that cannot satisfy both controls is ineligible. The denominator for yield reporting
is therefore **terminal rows joined to a pinned seed with valid baseline/reference controls**, not
all 2,377 terminal trajectories. Moonshiner tasks use their exact `verify_cmd`; they do not
necessarily provide SWE-bench `FAIL_TO_PASS` lists.

### Pass 1: isolated untrusted trajectory replay

Add a separate `TrajectoryReplayExecutor`; do not add a raw-command mode to `DockerExecutor`.

The new executor starts from a fresh seed copy in a dedicated `/testbed` tmpfs because published
commands refer to `/testbed`. It runs each recorded `Bash`, `Read`, `Write`, `Edit`, `Glob`, and
`Grep` action in source order. Bash commands are passed as data to `/bin/bash -c`, never interpolated
into an outer shell and never executed with login-shell semantics.

Raw replay remains untrusted even though its syntax is not restricted by the ingestion grammar. It
must preserve the existing host boundary:

- cached digest-pinned image only; no implicit pull;
- no network, host bind mounts, Docker socket, devices, or host namespaces;
- read-only root filesystem, all capabilities dropped, `no-new-privileges`, non-root UID;
- sanitized environment and fixed `/testbed` working directory;
- bounded memory, CPU, PIDs, tmpfs, command time, total trajectory time, and output bytes;
- exact command bytes, call order, return code, timeout, and bounded output hash recorded in an
  append-only replay ledger; and
- exact-CID cleanup plus the existing Docker free-space floor.

The first policy version may reuse but must not raise the reviewed ceilings: 2 CPUs, 4 GiB memory
with no additional swap, 256 PIDs, 4 GiB tmpfs, 4 MiB total captured output, and a 40 GiB Docker
free-space floor. Default command time is 30 seconds; an explicit source timeout may raise one action
only to 600 seconds, while the complete trajectory is capped at 1,800 seconds. Any higher limit is a
new policy review, not a runtime flag.

Commands may fail because the original agent could continue after a failed tool call; failures are
recorded and replay continues only when a corresponding source observation exists. Missing or
ambiguous tool-call/result pairing fails closed.

No authoritative verifier runs in this container. A recorded test or verifier-shaped Bash command
may execute as an untrusted trajectory action so later steps see the same state, but its result can
never satisfy an environment or candidate control.

After the final action, terminate and quiesce every process owned by the replay UID before inspecting
or exporting state. A separate observer UID must stream a deterministic archive to an
`O_NOFOLLOW`-created mode-0600 host file; do not use `docker cp`. A strict archive reader inventories
entries without extracting them and rejects absolute or escaping paths, duplicate names, links,
special files, unsupported modes, or bytes beyond the bound. Bind the archive hash and full inventory
hash into the ledger before exact-CID cleanup.

Run the entire raw replay twice from independent seed copies and require an identical canonical
source delta. Output text and declared ephemeral build artifacts may vary, but the admitted source
delta may not.

### Pass 2: canonical delta extraction and normalization

Compare each quiescent final `/testbed` state with the immutable seed inventory. Reject a replay if
the delta contains any of the following:

- a protected path or a test, fixture, example, benchmark, generated-report, or lockfile family;
- a path outside the task tree, traversal, symlink, hardlink, device, socket, FIFO, or submodule;
- an executable-bit or other mode-only change;
- a binary mutation, rename/copy ambiguity, case-fold collision, or duplicate target;
- an excessive changed-file count or changed-byte budget; or
- different canonical deltas across the two raw replays.

Inventory the whole worktree. Untracked compiler caches and build outputs are not silently treated as
source changes: they may be excluded only by an exact, language-specific ephemeral-path allowlist
pinned in the policy and recorded in the manifest (`__pycache__`, Rust `target`, and equivalent
known build roots). An untracked path outside that allowlist is a delta and must pass the normal path
and file-type gates. Ephemeral entries never enter normalization or training.

Convert the accepted final delta—not the shell syntax—into a `MutationPlan` of canonical `Write` and
`Edit` operations. Prefer unique byte-exact replacements. A new file may use `Write`; an existing
file that cannot be represented by unambiguous bounded replacements is rejected rather than
approximated. Applying the normalized plan twice to fresh seeds must reproduce the exact raw-replay
tree and diff hashes.

This avoids building parsers for arbitrary `sed`, heredoc, redirect, or Python mutation programs.
Those programs are evidence about how the final state was reached, not the training representation.

### Pass 3: fresh trusted verification

Pass the normalized plan to the existing reconstruction and control-set path. The trusted verifier
therefore sees only a fresh candidate created from the pinned seed plus typed declarative mutations;
it never sees the raw replay container or any of its processes.

Admission requires all existing evidence:

1. baseline clean failure;
2. reference clean success;
3. two independently reconstructed candidates with identical non-empty state;
4. two candidate verifier successes;
5. unchanged protected files; and
6. exact bindings for dataset, seed, image, verifier, operations, trees, diffs, and run contracts.

A single pass/fail bit is not sufficient.

### Pass 4: edit-decisive SFT ingestion

Only normalized rows may enter training. Do not copy unsupported raw Bash mutation turns into the
Gemma conversation.

Build an edit-adjacent native row containing:

- the native system message and original problem statement;
- at most the final few observation-grounded reads that already pass the existing ingestion grammar;
- the normalized explicit edit turn or turns;
- observations regenerated from the clean reconstructed candidate; and
- the trusted verification/diff/submission suffix needed to demonstrate completion.

If source reasoning is retained, it must be verbatim, attached to the equivalent normalized edit,
and grounded by a kept observation of the edited file. Dropped raw-shell steps and their observations
must not remain as hidden dependencies.

The resulting row still passes the existing first-edit/read-streak/repetition filters, content-hash
deduplication, 49,152-token budget, native Gemma format/loss gate, and mixture cap. A raw replay that
cannot be normalized coherently remains audit evidence only.

## Evidence and manifest contract

The lane must publish separate, hash-bound artifacts for:

- environment eligibility and baseline/reference controls;
- raw replay ledgers for both independent runs;
- canonical delta inventories and equality result;
- normalized operation plan and reconstruction-equivalence result;
- trusted candidate/control evidence; and
- final ingestion, format, token, behavior, deduplication, and exclusion manifests.

Report counts and rejection reasons per language at every boundary. In particular, distinguish:
`no_seed`, `invalid_environment_control`, `raw_replay_failed`, `raw_delta_nondeterministic`,
`forbidden_delta`, `normalization_failed`, `trusted_verify_failed`, and `ingestion_failed`.

## Smoke-first execution plan

No execution is authorized by this proposal. If approved later:

1. Unit-test the new executor and normalization layer with hostile commands, temporary protected-file
   tampering, background children, nondeterministic writes, special files, ambiguous replacements,
   tool timeouts, output overflow, and cleanup failures.
2. Re-run the three-language functional container admission probe using cached images only.
3. Run a five-case end-to-end replay smoke drawn from joinable environments. Require every
   baseline/reference control and every cleanup check to pass; do not manufacture a missing language
   by weakening the gate.
4. Only after the smoke, measure the metadata-eligible Fable candidates. Do not start from all 2,377
   terminals and do not pull new images without a separate disk/authority decision.
5. Stop after publishing the yield and language table. Training requires a separate decision and an
   identical-harness A/B gate. Any auxiliary mixture remains capped around one percent.

## Expected effect

This design can measure correct fixes that currently fail the ingestion grammar while preserving the
training-format and verifier trust boundaries. The yield increase is unknown and must not be assumed.
It may recover Python, Rust, or C++ candidates, but a language has training value only if rows survive
environment controls, repeatable raw replay, normalization, trusted verification, and final ingestion.

The existing 46 structural survivors remain a lower bound on directly clean-ingestible rows. They are
not a lower bound on verified-correct rows, and the full 2,377 is not the eligible verification
denominator.
