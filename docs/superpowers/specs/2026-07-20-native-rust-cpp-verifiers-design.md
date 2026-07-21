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
- parquet OID: `0cb80475fca8a3bac5a72846d8f5f91186d55533`

Evaluator:

- repository: `SWE-rebench/SWE-rebench-V2`
- revision: `c71902a8cf8d2b725f63d51f199f4d3e56f68d2d`
- `scripts/eval.py` SHA-256:
  `4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda`
- `lib/agent/log_parsers.py` SHA-256:
  `a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c`
- `combine.Dockerfile.j2` SHA-256:
  `4c5765b90079b9e4c7f5cfccd427fab37f5a4e5a0217ad18e8b627099c0ed1af`

The verifier imports the pinned upstream log-parser module from an explicitly
provided checkout and rejects any file-hash mismatch. It does not vendor or
silently update the 3,500-line parser implementation.

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

1. Refuse to start if `/var/lib/docker` has at most 40 GiB free.
2. Explicitly pull the exact task image with a bounded long timeout. Record free
   space before and after the pull. Abort if the floor is crossed.
3. Start one container and retain its exact CID. Do not discover or kill by
   pattern.
4. Derive the official checkout as `/<repo basename>`. Prove it is the expected
   git worktree and that `HEAD == base_commit` before applying anything.
5. For teacher collection only, create `/testbed` as a symlink to that proven
   checkout. Refuse to overwrite an existing nonmatching path. This preserves
   the mini-SWE command shape while using the upstream image layout.
6. Remove the exact CID in `finally`. Remove the exact image only when this run
   pulled it and it was not present before admission. Never use `pkill`, broad
   container deletion, or broad image deletion.

### Patch and test order

Verification occurs in a fresh container, separate from the teacher's working
container:

1. Validate the candidate patch is nonempty, applies inside the proven
   repository, and records its complete changed-path classification.
2. Reject path escapes and changes to tests, fixtures, examples, benches, or
   generated/vendor trees. Do not turn the verifier into the v3.1 curation
   filter: legitimate fixes may include `Cargo.toml`, `build.rs`, CMake files,
   or headers. The downstream Rust/C++ dataset builders decide which verified
   patches satisfy their narrower training policy.
3. Apply the candidate patch with the same `git apply` flags as the pinned
   upstream evaluator.
4. Apply the hidden `test_patch` through stdin. Never expose its contents to the
   teacher or training row.
5. Run every command from `install_config.test_cmd` with individual and total
   timeouts. Capture combined stdout/stderr without truncating the parser input.
6. Parse output with the task's allowlisted parser from the hash-verified pinned
   upstream module.
7. Normalize test names exactly as the upstream evaluator does.
8. Mark `resolved=true` only when every F2P test is parsed as passed and every
   P2P test remains passed. Missing expected test IDs fail closed. A zero test
   command exit status is not sufficient.

The Rust and C++ modules classify language-relevant paths separately and include
that classification in evidence. Language-specific curation policy must not
live in the common verifier module.

### Evidence contract

Each result contains:

- schema version and verifier language;
- dataset and evaluator revisions;
- task ID, image digest, base commit, and proven checkout path;
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

Current measured headroom is 2,670 eligible tasks across 259 repositories. The
deterministic round-robin selection can choose 120 tasks from 120 repositories.

The teacher collector receives a task adapter rather than branching inside the
Python verifier. The adapter prepares the `/testbed` symlink, captures the
teacher patch, calls the matching native verifier in a fresh container, and
writes the evidence record. `prepare` accepts a native row only if its evidence
schema is complete, `resolved=true`, its candidate patch is nonempty, and its
assistant/tool/observation pairing is balanced.

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
5. At least one isolated control must explicitly pass the task's gold patch as
   `candidate_patch` and resolve through the same verifier before teacher yield
   is interpreted. The control artifact is never eligible for training. This
   detects harness failures independently of teacher quality.
6. Only then may collection proceed in small sequential image batches. A full
   120-image cache is prohibited; the sampled five images already represent
   4.70 GB of unique compressed layers and the complete pool cannot fit safely.

The C++ verifier receives unit and fixture coverage now but no C++ collection or
training begins before the Rust ladder completes.

## Tests

Tests must cover:

- parser allowlists and wrong-language rejection for each verifier;
- missing or hash-mismatched upstream parser dependency;
- task schema and immutable revision checks;
- repository-root derivation and `/testbed` collision rejection;
- exact base-commit verification;
- candidate then test-patch ordering;
- path-escape and test-edit rejection, plus evidence-only classification of
  legitimate build/configuration edits;
- all F2P/P2P pass, F2P fail, P2P regression, missing expected ID, parser error,
  command error, and timeout;
- exact-CID cleanup on every failure path;
- disk-floor admission before and after pull;
- preexisting-image preservation and exact newly-pulled-image cleanup;
- evidence completeness and stable hashing;
- deterministic 120-task pool selection, repository disjointness, license
  filtering, gold-patch omission, and fail-closed diversity gates;
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
