# Native Rust and C++ Teacher Verification

Date: 2026-07-20

## Decision

Teacher acquisition for Rust and C++ must not reuse the Python verifier. Each
language gets a separate, fail-closed verifier with an explicit parser allowlist:

- `teacher_platform/rust_verifier.py` accepts only `language=rust` and
  `install_config.log_parser=parse_log_cargo`.
- `teacher_platform/cpp_verifier.py` accepts only `language=cpp` and the parser
  names measured in the pinned corpus: `parse_log_cpp`, `parse_log_cpp_v3`, and
  `parse_log_jq`.
- `teacher_platform/verification_common.py` contains only language-neutral
  container lifecycle, patch application, evidence serialization, disk guards,
  and pinned-parser loading. It makes no language or scoring decisions.

The existing Python verifier and Python acquisition paths remain unchanged.
This work does not execute the deferred `teacher_platform/acquisition/`
consolidation and does not move any `phaseD_sft` file.

## Why the split is required

The current collector calls `phaseE_rl.verified_reward.f2p_invocations`, which
translates only Python unittest and pytest identifiers. It never applies the
hidden `test_patch`, ignores `PASS_TO_PASS`, ignores the dataset-native
`install_config.test_cmd` and `log_parser`, and assumes `/testbed`.

Those assumptions are invalid for SWE-rebench-V2. Its images place the checkout
at `/<repo-name>`, and its official evaluator applies the candidate patch and
hidden test patch, runs the native test command, parses the complete output, and
checks both F2P and P2P. Reusing the Python verifier would produce false
negatives, false positives, or both.

## Pinned upstream contracts

Task corpus:

- dataset: `nebius/SWE-rebench-V2`
- revision: `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`
- config/split: `default/train`
- parquet Git/Xet object ID: `0cb80475fca8a3bac5a72846d8f5f91186d55533`
- parquet LFS content SHA-256:
  `0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad`
- parquet size: `428839266` bytes

Evaluator:

- repository: `SWE-rebench/SWE-rebench-V2`
- revision: `c71902a8cf8d2b725f63d51f199f4d3e56f68d2d`
- `scripts/eval.py` SHA-256:
  `4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda`
- `lib/agent/log_parsers.py` SHA-256:
  `a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c`
- `lib/agent/swe_constants.py` SHA-256:
  `823dd1ef512d363ed5d4dce05d70f22d7f93b25722cda5b0971f17010f5168a5`
- `combine.Dockerfile.j2` SHA-256:
  `4c5765b90079b9e4c7f5cfccd427fab37f5a4e5a0217ad18e8b627099c0ed1af`

The verifier executes the pinned upstream log-parser module in an isolated
subprocess from an explicitly provided checkout. It rejects a wrong commit, a
dirty checkout (including untracked files), or any required-file hash mismatch.
It does not add the checkout to the verifier process's `sys.path`, vendor the
3,500-line parser implementation, or silently update it.

Measured parser distribution at the pinned dataset revision:

- Rust: 3,123/3,123 use `parse_log_cargo`.
- C++: 180 use `parse_log_cpp_v3`, one uses `parse_log_cpp`, and one uses
  `parse_log_jq`.
- No Rust or C++ row is missing `install_config.test_cmd`.

## Verification protocol

### Input contract

Every verifier receives one task row, a candidate patch, and the path to the
pinned evaluator checkout. Required task fields are:

- `instance_id`, `repo`, `language`, `image_name`, and `base_commit`;
- nonempty `test_patch`, `FAIL_TO_PASS`, and `install_config.test_cmd`;
- `PASS_TO_PASS`, which may be empty but must be a list;
- allowlisted `install_config.log_parser`.

The verifier API accepts an explicit `candidate_patch`; it never falls back to
the task row's gold `patch` field. The gold patch is never copied into the
teacher prompt, result ledger, or SFT output. A smoke-test caller may explicitly
pass the gold patch as the candidate for a harness control, but that control is
kept outside teacher collection and training artifacts.

### Container and worktree contract

1. Refuse to start if `/var/lib/docker` has at most 40 GiB free, the artifact
   filesystem has at most 20 GiB free, or host `MemAvailable` is below the
   container memory limit plus 8 GiB (24 GiB for the 16 GiB limit below).
2. Resolve the task's registry tag to a platform-specific `linux/amd64` digest
   without pulling. Freeze that digest in the private sidecar and manifest, and
   execute only the immutable `name@sha256:...` reference. Require the registry
   name and digest to match strict syntax and the expected registry namespace.
   Inspect the registry manifest without pulling. Require free space above the
   40 GiB floor plus `max(8 GiB, 4 * compressed manifest bytes)` before an
   uncached pull. Explicitly pull the exact task image with a 30-minute timeout.
   Record free space before and after the pull and abort if the floor is crossed.
   All native image operations hold a process-scoped exclusive `flock` on
   `/tmp/gemma4-native-teacher-image.lock`; Rust and C++ cannot pull or clean up
   images concurrently. If the client-side pull times out, retain the lock and
   poll only the exact digest plus Docker events for up to ten minutes. Do not
   begin another pull or broad-prune partial layers. If the digest does not
   become inspectable or quiescent, stop for operator review.
3. Start one container and retain its exact CID. Do not discover or kill by
   pattern.
4. Derive the official checkout as `/<repo basename>`. Prove it is the expected
   git worktree and that `HEAD == base_commit` before applying anything.
5. Reset tracked files to `base_commit`, require no pre-existing unignored
   untracked files, and retain ignored build caches supplied by the image.
6. For teacher collection only, create `/testbed` as a symlink to that proven
   checkout. Refuse to overwrite an existing nonmatching path. This preserves
   the mini-SWE command shape while using the upstream image layout.
7. Start teacher and verifier containers without host mounts or the Docker
   socket, and with `--network none --cap-drop ALL --security-opt
   no-new-privileges --cpus 8 --memory 16g --memory-swap 16g --pids-limit
   512`. Network-dependent or more-memory-intensive tasks are ineligible rather
   than granted access to production services.
8. Remove the exact CID in `finally`. Remove the exact image only when this run
   pulled it and it was not present before admission. Never use `pkill`, broad
   container deletion, broad image deletion, or broad Docker prune. If exact
   cleanup does not restore the floor, stop and report.

### Patch and test order

Verification occurs in a fresh container, separate from the teacher's working
container:

1. In the teacher container, capture tracked, deleted, renamed, and newly
   created nonignored files with `git add -A` followed by
   `git diff --cached --binary HEAD`. Do not use plain `git diff`, which loses
   untracked files.
2. Validate the candidate patch is nonempty, applies inside the proven
   repository, and records its complete changed-path classification.
3. Parse changed paths NUL-safely from `git diff --name-status -z
   --find-renames HEAD`, including both sides of copies and renames. Reject
   absolute paths, `..` components, case-folded test/fixture/example/bench/
   generated/vendor components, `.gitmodules`, symlinks, submodules, type
   changes, and binary patches. Apply the same checks to every old and new
   path. Do not turn the verifier into the v3.1 curation
   filter: legitimate fixes may include `Cargo.toml`, `build.rs`, CMake files,
   or headers. The downstream Rust/C++ dataset builders decide which verified
   patches satisfy their narrower training policy.
4. Apply the candidate patch with the same `git apply` flags as the pinned
   upstream evaluator.
5. Apply the hidden `test_patch` through stdin. Never expose its contents to the
   teacher or training row.
6. Execute `install_config.test_cmd` in order with shell `set -e`, matching the
   upstream short-circuit behavior. Enforce a 30-minute per-command and task
   timeout. Stream combined stdout/stderr directly to an artifact file rather
   than retaining it in process memory. Cap raw output at 128 MiB per task and
   fail closed before exhausting the artifact filesystem; do not truncate and
   then score partial parser input.
7. Parse output in the isolated pinned-parser subprocess.
8. Normalize test names exactly as the upstream evaluator does.
9. Match the pinned upstream evaluator exactly: normalize the expected set as
   `PASS_TO_PASS + FAIL_TO_PASS`, normalize the parser's passed set, and mark
   `resolved=true` only when the two sets are equal. Extra parsed passes,
   missing IDs, F2P failures, and P2P regressions all fail closed. Record test
   command exit status as evidence, but do not add it to the resolution
   predicate because the pinned evaluator does not. Differential fixtures must
   prove our result matches the pinned evaluator for every pass/fail case.

The Rust and C++ modules classify language-relevant paths separately and include
that classification in evidence. Language-specific curation policy must not
live in the common verifier module.

### Evidence contract

Each result contains:

- schema version and verifier language;
- dataset and evaluator revisions;
- task ID, immutable platform-specific image digest, base commit, and proven
  checkout path;
- candidate-patch and hidden-test-patch SHA-256 values;
- parser name and parser-file SHA-256;
- test command list, return codes, elapsed times, complete-output SHA-256, and
  the path of the retained raw test-log artifact;
- normalized per-ID F2P and P2P statuses;
- `resolved`, `failure_class`, and a bounded diagnostic excerpt.

`failure_class` distinguishes admission, pull, container, checkout, patch,
test-patch, timeout, parser, missing-test-ID, F2P, and P2P failures. Infrastructure
failures never count as teacher failures.

## Teacher-platform integration

`teacher_platform/rust_teacher_pool.py` builds the Rust pool. It is a
migration-ready platform module, not a new `phaseD_sft` script.

Pool invariants:

- pinned dataset revision;
- Rust only;
- repository-disjoint from all ten frozen Multi-SWE-Rust repositories;
- permissive per-row licenses only: Apache-2.0, MIT, BSD-3-Clause, ISC,
  CC0-1.0, and Unlicense;
- deterministic seed `20260720`;
- 120 selected tasks across at least 50 repositories, with no more than three
  per repository;
- no gold solution patch in the emitted task file;
- manifest includes source hashes, exclusion hashes, license counts,
  per-repository counts, selected-ID hash, image inventory, and projected image
  bytes when probed.

The frozen exclusion source is
`data/mswe_rust_prs_full239.jsonl`: 239 rows, file SHA-256
`5893e74d6e45183fc4e922dbe5bbe1169c1c26c8fbfa31129fb84a0e0c43fa8b`,
and sorted-ID SHA-256
`bc0a6b0994d437af5f00323fbe87846e0f424c6a88b484f3f4c5592f1f1dc645`.
The case-folded canonical repository denylist is exactly
`BurntSushi/ripgrep`, `clap-rs/clap`, `nushell/nushell`, `rayon-rs/rayon`,
`serde-rs/serde`, `sharkdp/bat`, `sharkdp/fd`, `tokio-rs/bytes`,
`tokio-rs/tokio`, and `tokio-rs/tracing`. Any hash, row-count, or repository-set
drift fails closed rather than silently rebuilding a different training pool.

The builder emits two files with the same selected IDs:

- a public collection pool containing only `instance_id`, `repo`, `language`,
  `license`, `image_name`, `base_commit`, and `problem_statement`;
- a mode-`0600` verification sidecar containing the immutable platform-specific
  image digest, `test_patch`, F2P/P2P, `install_config`, and source provenance.

Neither file contains the gold solution patch. The manifest records independent
content hashes for both files and proves their ID sets are identical. Teacher
backends receive only the public row; the verifier parent process joins the
hidden row after the teacher exits.

Current measured headroom is 2,670 eligible tasks across 259 repositories. The
deterministic round-robin selection can choose 120 tasks from 120 repositories.

Repository exclusion compares case-folded canonical `owner/repo` names derived
from frozen instance IDs, not the frozen files' short display labels. Selection
does not depend on incoming dataset order or Python RNG implementation: each
candidate receives `SHA256("20260720\0" + canonical_repo + "\0" +
instance_id)`, repositories and candidates are sorted by that score, and
selection advances one round-robin row per repository until the target is met.

The teacher collector receives a task adapter rather than branching inside the
Python verifier. The adapter prepares the `/testbed` symlink, captures the
teacher patch, calls the matching native verifier in a fresh container, and
writes the evidence record. `prepare` accepts a native row only if its evidence
schema is complete, `resolved=true`, its candidate patch is nonempty, and its
assistant/tool/observation pairing is balanced.

Native verification and collection use a new explicit entry point and never
call `teacher_trace_driver.collect_one`, its broad-prune cleanup, or its Python
F2P helper. The existing Python CLI and collection loop remain byte-for-byte
unchanged. Tests assert that a native task cannot enter the Python path and a
Python task cannot enter the native path.

Native collection must not expose unrestricted host Bash to a teacher CLI. The
current Claude/Codex path grants `Bash` or `danger-full-access`, which can touch
other containers, production ports, or hidden verification files. A native
teacher backend is eligible only if commands pass through a broker bound to the
single exact CID and repository checkout. The broker accepts one bounded command
string, executes only via `docker exec <exact CID> bash -c 'cd /testbed && ...'`,
and returns only the bounded observation. It does not claim to secure Bash with
token blacklists; the actual boundary is the unprivileged, network-isolated,
resource-limited container with no host mounts or Docker socket. The existing
OpenRouter function-tool loop is the nearest current shape, but it must use the
broker rather than constructing Docker commands itself.
Subscription CLI backends remain disabled for native collection until they can
use the same broker without unrestricted Bash. Verifier and gold-control work is
not blocked on teacher-backend availability.

## Smoke and scale gates

No 120-task collection begins immediately.

1. Unit tests and golden parser fixtures must pass for common, Rust, and C++
   modules.
2. A five-task Rust smoke runs sequentially, one repository per task.
3. Before and after every image operation, Docker free space must remain above
   40 GiB. The current host has about 61 GiB free.
4. The smoke must prove image pull, checkout identity, teacher `/testbed`
   compatibility, hidden-test application, native parser execution, auditable
   F2P/P2P results, exact-CID cleanup, and no production-port changes.
5. Every one of the five smoke tasks must run an isolated gold-patch control
   through the same verifier before teacher yield is interpreted. The control
   subcommand reads the gold patch from the pinned source only after the public
   teacher run has ended, keeps it in memory, and writes only under
   `runs/native_verifier_controls/<run_id>/` with `control_only=true`. Dataset
   preparation and merge code hard-rejects `control_only` evidence and any
   artifact path under that namespace. This detects per-image harness failures
   independently of teacher quality without exposing gold content.
6. Only then may collection proceed in small sequential image batches. A full
   120-image cache is prohibited; the sampled five images already represent
   4.70 GB of unique compressed layers and the complete pool cannot fit safely.

The C++ verifier receives unit and fixture coverage now but no C++ collection or
training begins before the Rust ladder completes.

## Tests

Tests must cover:

- parser allowlists and wrong-language rejection for each verifier;
- missing or hash-mismatched upstream parser dependency;
- task schema, dataset-content hash, and immutable revision checks;
- mutable-tag rejection, `linux/amd64` digest resolution, and digest mismatch;
- repository-root derivation and `/testbed` collision rejection;
- exact base-commit verification;
- candidate then test-patch ordering;
- NUL-safe rename/copy path parsing; path-escape, case-variant test edit,
  symlink, submodule, `.gitmodules`, type-change, and binary rejection; plus
  evidence-only classification of legitimate build/configuration edits;
- all F2P/P2P pass, F2P fail, P2P regression, missing expected ID, parser error,
  command error, and timeout;
- differential resolution parity against the pinned evaluator, including extra
  parsed passes and nonzero command exits;
- exact-CID cleanup on every failure path;
- network isolation, 24-GiB host-RAM admission, and container CPU/memory/PID
  limits;
- Docker and artifact disk-floor admission before and after pull;
- streaming raw logs, 128-MiB fail-closed output cap, and artifact exhaustion;
- manifest-size reserve admission, global image lock, and pull-timeout
  quiescence handling;
- preexisting-image preservation and exact newly-pulled-image cleanup;
- tracked/new-file patch capture and rejection of a dirty image baseline;
- broker command-size/NUL validation, exact-CID execution, and absence of host
  mounts or Docker socket;
- evidence completeness and stable hashing;
- deterministic 120-task pool selection, repository disjointness, license
  filtering, gold-patch omission, and fail-closed diversity gates;
- five-of-five isolated gold controls plus `control_only` artifact rejection;
- strict separation of native and legacy Python entry points;
- native-evidence enforcement in `prepare` without changing Python behavior.

Integration fixtures compare Rust and C++ parser results against the pinned
upstream parser on representative logs. The five-task smoke is the runtime
contract test; unit tests do not substitute for it.

## Non-goals

- Moving existing acquisition files into `teacher_platform/acquisition/`.
- Changing Python reward or teacher verification.
- Training a Rust or C++ adapter in this change.
- Pulling or retaining all selected images.
- Accepting gold patches, test patches, or benchmark trajectories as training
  content.
- Weakening the existing v3.1 behavior, format, or frozen-evaluation gates.
