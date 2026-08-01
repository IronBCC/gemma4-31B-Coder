# v2.11r4 Weight Interpolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, lineage-bind, gate, and conditionally evaluate one `75% v2.10 + 25% final-r3` model that can beat canonical v2.10 without regressing empty or loop behavior.

**Architecture:** A streaming CPU utility joins parent tensors by key, validates exact compatibility, writes an atomic sharded checkpoint, and publishes a checksum-bound interpolation manifest. A dedicated lineage validator plugs into the existing portability runner. The evaluation chain stops at portability or fixed150 unless the candidate preserves the v2.10 correctness floor and improves at least one behavior metric.

**Tech Stack:** Python 3.11+, PyTorch, safetensors, pytest, Bash, systemd user units, vLLM, Mini-SWE-Agent, SWE-bench.

## Global Constraints

- Candidate equation is exactly `anchor + 0.25 * (r3 - anchor)` with v2.10 as anchor.
- Create exactly one alpha candidate; no grid search.
- Resolve tensors by key through each parent's index independently.
- Floating tensors compute in FP32 and cast to anchor dtype; non-floating tensors must be exactly equal and copy from v2.10.
- Stage, audit, and atomically publish; never overwrite an existing model.
- Bind both parent model contracts, alpha, copied files, and output contract in the manifest.
- GPU1 only, port 8013; never query or touch GPU0.
- Full300 is forbidden unless portability10 and fixed150 pass the documented correctness and behavior floors.
- Scope is only canonical v2.10 and the v2.11 successor.

---

### Task 1: Streaming checkpoint interpolator

**Files:**
- Create: `phaseD_sft/interpolate_checkpoints_streaming.py`
- Create: `phaseD_sft/tests/test_interpolate_checkpoints_streaming.py`
- Reuse: `phaseD_sft/atomic_checkpoint_publish.py`

**Interfaces:**
- Consumes: `--anchor`, `--candidate`, `--out`, `--alpha`, `--group-gb`, `--max-rss-gb`, and `--manifest-name`.
- Produces: an atomically published sharded checkpoint and `interpolation_manifest.json`.

- [ ] Write tiny real safetensors fixture tests for independent key lookup, exact BF16 interpolation, non-floating equality, manifest bindings, refusal to overwrite, cleanup after failure, and fail-closed config/key/shape/dtype mismatches.
- [ ] Run `pytest -q phaseD_sft/tests/test_interpolate_checkpoints_streaming.py` and verify collection fails because the module does not exist.
- [ ] Implement the minimal streaming utility with bounded shards, RSS checks, staged audit, and `publish_after_audit`.
- [ ] Run `pytest -q phaseD_sft/tests/test_interpolate_checkpoints_streaming.py phaseD_sft/tests/test_merge_lora_streaming.py` and require a clean pass.
- [ ] Commit only the Task 1 source and test as `feat: stream v2.11 checkpoint interpolation`.

### Task 2: Interpolation lineage and portability integration

**Files:**
- Create: `phaseH_eval/v2p11_interpolation_lineage.py`
- Create: `phaseH_eval/tests/test_v2p11_interpolation_lineage.py`
- Modify: `phaseH_eval/run_v2p11_portability_gate.sh`
- Modify: `phaseH_eval/tests/test_run_v2p11_portability_gate.py`

**Interfaces:**
- Consumes: interpolation manifest, immutable v2.10/r3 parent paths, candidate model path/name.
- Produces: a validated interpolation lineage contract accepted by `LINEAGE_MODE=interpolation`.

- [ ] Write fixture tests that fail for any parent, alpha, copied-file, output-contract, or candidate-name mutation and prove invalid lineage/GPU modes fail before serving.
- [ ] Run the new tests and verify RED because the lineage module and mode do not exist.
- [ ] Implement exact lineage reconstruction and extend the shell case to `posttrain|direct_lora|interpolation` without weakening existing modes or GPU1 guards.
- [ ] Run the new lineage/runner tests, existing portability tests, and `bash -n phaseH_eval/run_v2p11_portability_gate.sh`.
- [ ] Commit only Task 2 paths as `feat: bind interpolated v2.11 lineage`.

### Task 3: Pre-full300 promotion gate and launch chain

**Files:**
- Create: `phaseH_eval/v2p11_successor_gate.py`
- Create: `phaseH_eval/tests/test_v2p11_successor_gate.py`
- Create: `phaseH_eval/run_v2p11r4_blend25_chain.sh`
- Create: `phaseH_eval/tests/test_run_v2p11r4_blend25_chain.py`
- Modify: `phaseH_eval/SESSION_STATE.md`

**Interfaces:**
- Consumes: portability gate, candidate and v2.10 fixed150 first-pass artifacts, interpolation lineage, and canonical full300 inputs.
- Produces: immutable `runs/v2p11r4_blend25_prefull_gate.json`; conditionally invokes the existing full300 and empty-only correction path.

- [ ] Write literal decision fixtures proving full300 rejection for each independent resolved, paired, wrong-nonempty, empty, loop, and strict-improvement failure.
- [ ] Verify RED because the gate and chain do not exist.
- [ ] Implement no-overwrite checksum-bound decision publication and a GPU1/8013 chain fixed to alpha 0.25.
- [ ] Run all Task 1-3 tests, shell syntax checks, and `git diff --check`.
- [ ] Commit only Task 3 paths and the runtime record as `feat: gate v2.11r4 before full300`.

### Task 4: Remote verification and conditional evaluation

**Files:**
- Sync only committed Task 1-3 paths to `/home/ironbcc/projects/gemma4-31B-Coder`.
- Runtime artifacts remain under remote `runs/`, `data/`, and the model volume.

**Interfaces:**
- Consumes: committed source, immutable parent checkpoints, and GPU1.
- Produces: interpolated model, lineage manifest, portability/fixed150 gate, and only after a pass the matched full300 verdict.

- [ ] Verify local/remote checksum parity and the complete focused tests using remote `.venv-eval/bin/python`.
- [ ] Materialize the candidate under a named CPU systemd unit and record unit identity, manifest hash, peak RSS, duration, and disk headroom.
- [ ] Run portability10 and fixed150 under named GPU1 units, exact PIDs, port 8013, health guards, and one evidence-based PDT completion boundary.
- [ ] Stop if the pre-full300 gate fails; otherwise launch the same full300 and bounded empty correction.
- [ ] Require a complete 300-ID verdict and v2.11 score strictly above 157 before marking the persistent goal complete.

