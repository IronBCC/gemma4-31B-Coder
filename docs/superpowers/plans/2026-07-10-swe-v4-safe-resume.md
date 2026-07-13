# SWE v4 Safe Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the SWE edit-trace v4 lane on GPU1 without interrupting GPU0 production or risking another host-wide OOM.

**Architecture:** Keep the validated 14,336-token dataset and cp20 initialization adapter. Launch training as a transient user systemd service with cgroup memory limits, verify production health before launch, and stop after a one-step checkpoint so measured memory peak gates the ten-step continuation.

**Tech Stack:** Bash, systemd user services/cgroup v2, Unsloth/Transformers, unittest, SSH, NVIDIA tooling.

## Global Constraints

- GPU0 and `vllm.service` are production and must remain running.
- Training uses only GPU1 through `CUDA_VISIBLE_DEVICES=1`.
- Host baseline is 96 GB physical RAM and 256 GB swap; launch requires at least 60,000 MiB `MemAvailable`.
- Trainer cgroup limits are `MemoryHigh=40G`, `MemoryMax=48G`, and `MemorySwapMax=32G`.
- Run one optimizer step and inspect checkpoint, cgroup memory peak, host memory, swap, and production health before any ten-step run.
- Never use `phaseD_sft/vllm_kill_watch.sh` against production.

---

### Task 1: Encode launcher safety invariants

**Files:**
- Create: `phaseD_sft/tests/test_recover_swe_edit_v4_script.py`
- Modify: `phaseD_sft/recover_swe_edit_v4_smoke_first.sh`

**Interfaces:**
- Consumes: v4 dataset and cp20 adapter paths already validated on the remote host.
- Produces: a smoke-only systemd unit named `swe-edit-v4-smoke` with explicit cgroup limits and production health gates.

- [ ] Write a unittest that asserts GPU1 isolation, cgroup limits, 60,000 MiB launch headroom, production health checks, and absence of vLLM termination commands.
- [ ] Run `python3 -m unittest phaseD_sft.tests.test_recover_swe_edit_v4_script` and confirm it fails against the old launcher.
- [ ] Replace the direct `nohup` launch with `systemd-run --user`, preserve vLLM, and remove automatic ten-step launch.
- [ ] Run the focused unittest and `bash -n phaseD_sft/recover_swe_edit_v4_smoke_first.sh`; expect both to pass.

### Task 2: Disable the dangerous production kill helper

**Files:**
- Modify: `phaseD_sft/vllm_kill_watch.sh`
- Test: `phaseD_sft/tests/test_recover_swe_edit_v4_script.py`

**Interfaces:**
- Consumes: explicit emergency opt-in `ALLOW_PROD_VLLM_KILL=I_UNDERSTAND`.
- Produces: a default refusal that cannot kill GPU0 production accidentally.

- [ ] Add a test that requires a default refusal guard.
- [ ] Add the explicit opt-in guard before any process discovery or signal.
- [ ] Run the focused unittest and shell syntax checks; expect success.

### Task 3: Persist the new machine and run state

**Files:**
- Modify: `phaseH_eval/SESSION_STATE.md`

**Interfaces:**
- Consumes: live host baseline measured on 2026-07-10.
- Produces: durable recovery context for later sessions.

- [ ] Record 96 GB online RAM, current 68 GiB available baseline, 256 GiB free swap, production cgroup use, GPU assignments, v4 artifact state, and the cgroup smoke gate.
- [ ] Re-read the new section and verify it does not claim a checkpoint exists.

### Task 4: Sync and execute the protected smoke

**Files:**
- Sync: the two shell scripts, focused test, and session state to `/home/ironbcc/projects/gemma4-31B-Coder`.

**Interfaces:**
- Consumes: the safe launcher from Task 1.
- Produces: `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_smoke1_14336/checkpoint-1` or a bounded training-only failure.

- [ ] Run local and remote focused tests.
- [ ] Launch only the one-step cgroup-bounded smoke.
- [ ] Monitor unit memory, `MemAvailable`, swap, GPU1, and all production health endpoints.
- [ ] If checkpoint-1 appears, capture memory peak and stop for behavior evaluation planning; if the cgroup kills training, lower sequence/data budget without touching production.
