# Fable 5 Agentic Trace Harvest

Date: 2026-07-20

## Decision

Use `greghavens/fable-5-coding-and-debugging-traces` as a conditional auxiliary
SFT source after deterministic conversion and independent replay. Do not train
on the raw cumulative-prefix rows. This change builds and audits a CPU-only
pilot dataset; it does not launch training or change serving.

The preferred approach is terminal-trajectory reconstruction plus native-tool
normalization. Two rejected alternatives are:

- training every cumulative prefix with the existing all-assistant loss mask,
  which repeatedly overweights early turns; and
- extending the production agent to expose the full Claude/Fable tool suite,
  which changes the evaluation contract instead of improving the current one.

## Pinned sources

Dataset:

- repository: `greghavens/fable-5-coding-and-debugging-traces`
- revision: `aef8506515979988aa5c1a423f5b0fb3cee60382`
- config / physical split: `default/train`
- source file: `traces.jsonl`
- source-file LFS SHA-256:
  `ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3`
- source-file size: `730331947` bytes
- rows at this revision: `12408`
- terminal trajectories at this revision: `2377`
- license: CC BY 4.0; generated manifests and model cards must preserve
  attribution to the dataset and Moonshiner.

Replay fixtures:

- repository: `greghavens/moonshiner`
- revision: `436316e8f86eb136d5ce3ec95a1a6f48c1d7f940`
- task root: `tasks/seeds/<task>/`

Both revisions and the source LFS hash are mandatory CLI defaults and manifest
fields. A different revision requires a new audit version rather than silently
refreshing this dataset.

## Source facts that shape the design

The dataset contains one physical Hugging Face split, while the row-level
`split` field contains `11375` training rows and `1033` validation rows. Each
trajectory with N assistant turns is represented by N cumulative rows. Only the
row where `assistant_step == assistant_steps` is the full trajectory needed by
our ordinary all-assistant-turn loss mask.

The full Rust slice contains 289 prefixes from 49 trajectories. All 49 terminal
trajectories edit by assistant command ten, median command three, and show Cargo
or build/test activity. The C++ slice contains 197 prefixes from 33
trajectories, but 12 have broken tool/result structure; the structurally clean
ceiling is about 20-21. Python has 1227 prefix rows across the `python` and `py`
labels and needs the same task-level audit.

Fable messages use `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, and a broad
set of unrelated tools. They contain little explicit `reasoning` metadata;
short visible planning lives in assistant `content`. Conversion preserves that
content verbatim and never invents hidden chain of thought.

## Architecture

New external-teacher import code belongs under `teacher_platform`, not
`phaseD_sft`. Existing Python acquisition and training files are not moved in
this change.

- `teacher_platform/fable5_import.py` streams the pinned dataset, selects final
  trajectories, validates structure, normalizes supported tool calls, applies
  behavior and decontamination gates, token-filters, and writes the SFT JSONL
  plus manifest.
- `teacher_platform/fable5_replay.py` joins a selected task to the pinned
  Moonshiner seed, replays it in a disposable restricted workspace, protects
  test files, and records baseline/final verification evidence.
- `teacher_platform/tests/test_fable5_import.py` tests cumulative-row collapse,
  conversion, pairing, behavior gates, deduplication, and manifest arithmetic.
- `teacher_platform/tests/test_fable5_replay.py` tests seed identity,
  protected-file enforcement, baseline failure, final pass, timeouts, and exact
  cleanup using local fixtures only.

The importer accepts dependency-injected iterators, tokenizer functions, and a
replay callback so unit tests never use the network or Docker.

## Selection contract

One streaming pass retains a row only when all conditions hold:

1. Dataset revision and source-file hash match the pinned contract.
2. Row-level `split == "train"`.
3. `assistant_step == assistant_steps`; all other cumulative prefixes are
   counted and discarded.
4. Language canonicalizes to `python`, `rust`, or `cpp` from the exact aliases
   `python|py`, `rust`, and `cpp|c++`.
5. Task/category is code work. `seed-authoring`, `en`, `non-code`, generic
   instruction following, research, scheduling, web, and memory tasks are
   rejected.
6. Every message has a supported role, assistant tool calls have unique IDs,
   every tool call has exactly one following result, and the trajectory ends in
   an assistant message rather than a user/tool message.
7. Every used tool is one of `Bash`, `Read`, `Write`, `Edit`, `Glob`, or `Grep`
   after case normalization. Any subagent, web, cron, notebook, messaging,
   workflow, or unknown tool rejects the entire trajectory.
8. The task ID exists in the pinned Moonshiner seed tree and its fixture hashes
   match the replay inventory.
9. Exact task/prompt/content hashes and fuzzy prompt matching prove no overlap
   with hard30, SWE-Lite 0:30, the frozen 239 Multi-SWE-Rust IDs, or the C++
   evaluation pool. Ambiguous matches fail closed.
10. Content-hash deduplication retains one deterministic representative.

The row-level validation split is retained only as an audit count. It is never
written into training output.

## Tool normalization

Every accepted source trajectory becomes one Gemma-native full conversation.
The original system prompt is replaced by the project's canonical mini-SWE
system prompt; the original user task remains verbatim after secret and host
path scanning.

Supported calls map as follows:

- `Bash(command)` -> native `bash` with the command unchanged.
- `Read(file_path, offset, limit)` -> a shell-quoted `sed -n`/`cat` command.
- `Write(file_path, content)` -> a deterministic Python `Path.write_text`
  heredoc whose payload is base64 encoded.
- `Edit(file_path, old_string, new_string, replace_all)` -> a deterministic
  Python replacement script that asserts the required match count before
  writing.
- `Glob(pattern, path)` -> `rg --files` with shell-quoted root and glob.
- `Grep(pattern, path, ...)` -> a bounded `rg -n` form only when every supplied
  option has an exact supported mapping; otherwise reject the trajectory.

Paths must be relative, NUL-free, and remain inside the task workspace. Absolute
paths, `..`, shell newlines in path fields, symlinks escaping the workspace, and
test/protected-file writes fail closed.

Fable can issue parallel calls in one assistant message, while mini-SWE expects
one Bash call per turn. Conversion emits one assistant/tool-result pair per
source call in original order. The source assistant content is preserved only
on the first emitted assistant turn; later calls have empty content, preventing
duplicated planning loss. Original tool observations are preserved verbatim.

The importer takes only the full terminal row, so the ordinary project loss
mask supervises every converted assistant turn exactly once. It does not need a
new final-target-only trainer mode.

## Behavior gates

After normalization, each candidate must satisfy:

- nonempty source mutation;
- first source-edit command index at most 10;
- maximum consecutive read streak at most 5;
- no identical consecutive normalized commands;
- at least one language-appropriate build/test command after the first edit;
- no test, fixture, benchmark, generated, vendor, or protected-seed-file edit;
- balanced assistant call and observation pairs;
- no trailing user/tool turn;
- rendered Gemma prompt at most 49152 tokens;
- `verify_gemma_format_loss.py --samples 1000` reports `failure_count=0` on
  the final output.

The tokenizer loads once in the main process. Parallel token counting may use a
`ThreadPoolExecutor`; it must not load one tokenizer per process.

## Independent replay contract

Publisher verification is useful provenance but is not accepted as our final
execution evidence. The published legacy rows omit the publisher's candidate
diff, seed fingerprint, protected hashes, verifier identity, and acceptance
ledger. Every retained task therefore needs a candidate reconstructed from
declarative source mutations and independently verified from the pinned
`tasks/seeds/<task>` fixture. A trajectory whose final state depends on an
arbitrary Bash mutation is not reconstructable and is rejected.

1. Copy the seed `files/` tree into a fresh disposable workspace.
2. Record hashes for `test_files` and every fixture file before replay.
3. Run the seed's exact `verify_cmd` with `verify_timeout`; require the baseline
   to fail when the task contract declares a failing baseline.
4. Reconstruct the candidate only from exact `Write`, `Edit`, or equivalent
   declarative patch operations. Do not execute transcript Bash, tests, package
   managers, scripts, interpreters, or command substitutions to build the
   candidate. Any unproven Bash source mutation rejects the task.
5. Apply the reconstructed candidate in an isolated, network-disabled,
   resource-bounded environment. Never execute a Fable command on the host.
6. Require all declared protected/test files to remain byte-identical.
7. Run the exact verification command twice in fresh candidate states and
   require identical successful return codes and output hashes.
8. Capture a binary source diff; require it to be nonempty and free of rejected
   paths.
9. Record task/revision/fixture/command/diff hashes, return codes, durations,
   bounded log paths, and cleanup status.

The first runtime gate is five tasks: two Python, two Rust, and one C++ when a
clean C++ candidate exists. Each task runs three controls: the reconstructed
Fable candidate must pass, the pinned `reference_fix.patch` must pass as a
separate harness-positive control, and a deliberately corrupted candidate must
fail. Control artifacts are tagged and cannot enter training output.

Linux `bwrap` with a read-only host root, isolated PID namespace, temporary home
and `/tmp`, clear environment, disabled network, and only the fixture workspace
writable is preferred. No image pull is automatic. If Docker is required, the
gate uses only an already-cached immutable digest and inherits the 40-GiB
free-space floor, non-root execution, CPU/memory/PID limits, exact-CID cleanup,
no host mounts or Docker socket, no broad prune, and no production-port changes.
A missing safe environment stops with an inventory report.

## Outputs and manifest

The pilot output is `data/fable5_agentic_pilot_v1/`:

- `train.jsonl`: Gemma-native SFT rows, one full trajectory per task;
- `manifest.json`: pinned revisions and hashes, row arithmetic, per-language and
  per-category counts, all drop reasons, tool-conversion counts, token and
  first-edit histograms, dedup hashes, decontamination hashes, replay evidence
  summary, and CC BY attribution;
- `replay.jsonl`: one evidence record per attempted task, containing hashes and
  log references but no protected test contents;
- `original_terminal_rows.jsonl`: mode-`0600`, hash-bound replay sidecar for
  structurally selected terminal rows; it is never a training input;
- `rejected.jsonl`: task ID plus bounded reason codes only, without secrets or
  protected test content.

Every source label is
`teacher:fable5:aef8506515979988aa5c1a423f5b0fb3cee60382:<task>`.

Arithmetic must prove:

`streamed = nonterminal + validation + language_drop + category_drop +
structure_drop + unsupported_tool + contamination_drop + duplicate_drop +
behavior_drop + replay_drop + token_drop + output`.

## Scale and stop gates

1. Unit tests pass with only local fixtures.
2. A read-only metadata audit reproduces 12408 rows and 2377 terminal
   trajectories for the pinned revision.
3. A 30-task structural conversion smoke completes with zero pairing or format
   failures.
4. The five-task independent replay smoke passes five of five; any verifier or
   cleanup failure stops the lane.
5. The complete CPU build advances to training consideration only if it yields
   at least 300 independently verified Python/Rust/C++ trajectories, at least
   70% edit by command ten, median first edit at most eight, and zero format
   failures.

If fewer than 300 survive, report the honest per-language yield and bank the
converter. Do not relax tool, verification, contamination, or behavior rules to
hit the target. A future mixture may begin at 5-10% Fable rows; no mixture or
training launch is authorized by this design.

## Non-goals

- Training or serving a model.
- Adding Claude-style tools to the production mini-SWE harness.
- Using Fable validation rows for training.
- Treating publisher acceptance as independent replay evidence.
- Inventing hidden reasoning or filling missing tool observations.
- Moving existing `phaseD_sft` acquisition code into `teacher_platform`.
- Downloading or caching unrelated language tasks.
