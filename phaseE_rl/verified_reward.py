#!/usr/bin/env python3
"""Reward phase 2: docker-verified tiers on top of the parse ladder.

Tier ladder (REWARD_PHASE2_SPEC.md, gate-resolved hybrid_full_v2):
  0.0  no parseable tool call
  0.2  valid bash call, read-only
  0.6  edit command (parse ceiling; also the CAP for unverified rows)
  0.8  edit command EXECUTES in the reconstructed prefix state (rc==0, diff lands)
  1.0  0.8 AND the instance F2P tests pass afterward

Verified rows resolve through the state cache built by
reconstruct_swe_decision_state.py (`rlvr-state:<prompt_hash>` images). Any
docker failure or timeout degrades to the parse tier — training never stalls
on infrastructure.
"""
from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_UNITTEST_ID = re.compile(r"^(?P<method>[\w\[\]-]+)\s+\((?P<path>[\w.]+)\)$")


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def load_fixture_index(path: str | Path) -> dict[tuple[str, str], dict]:
    """(source, instance_id) -> fixture row; mirrors reconstruct_* keying."""
    index: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                index[(row["source"], row["instance_id"])] = row
    return index


def f2p_invocations(f2p: list[str]) -> list[str]:
    """Convert mixed-format F2P ids into runnable in-container commands.

    unittest style 'method (pkg.module.Class)' -> python -m unittest pkg.module.Class.method
    pytest node ids 'path/test_x.py::Class::method' -> python -m pytest -x -q <ids>
    Unrecognized entries are dropped (verification then can't award 1.0).
    """
    unittest_ids, pytest_ids = [], []
    for entry in f2p:
        m = _UNITTEST_ID.match(entry.strip())
        if m:
            unittest_ids.append(f"{m.group('path')}.{m.group('method')}")
        elif "::" in entry:
            pytest_ids.append(entry.strip())
    cmds = []
    if unittest_ids:
        cmds.append("python -m unittest -q " + " ".join(unittest_ids[:20]))
    if pytest_ids:
        quoted = " ".join(f"'{i}'" for i in pytest_ids[:20])
        cmds.append(f"python -m pytest -x -q {quoted}")
    return cmds


class StateVerifier:
    """Execute a candidate edit command in a cached prefix-state image."""

    # SWE-bench/SWE-smith images keep the instance env in conda 'testbed'; a
    # non-login `bash -c` sees base conda python WITHOUT pytest (same trap
    # family as the mswebench cargo PATH reset). Tolerant no-op elsewhere.
    ENV_BOOTSTRAP = "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true; "

    def __init__(self, exec_timeout: int = 60, test_timeout: int = 300):
        self.exec_timeout = exec_timeout
        self.test_timeout = test_timeout

    def state_exists(self, state_tag: str) -> bool:
        return run(["docker", "image", "inspect", state_tag], timeout=30).returncode == 0

    def verify(self, state_tag: str, command: str, f2p: list[str],
               test_patch: str = "") -> float:
        """Return the verified bonus tier: 0.6 (failed), 0.8 (executed), 1.0 (F2P pass)."""
        created = run(["docker", "run", "-d", "--network", "none", "--pids-limit", "512",
                       state_tag, "sleep", "infinity"], timeout=120)
        if created.returncode:
            return 0.6
        cid = created.stdout.strip()
        try:
            res = run(["docker", "exec", cid, "bash", "-c",
                       f"cd /testbed && {self.ENV_BOOTSTRAP}{command}"],
                      timeout=self.exec_timeout)
            if res.returncode != 0:
                return 0.6
            diff = run(["docker", "exec", cid, "bash", "-c",
                        "cd /testbed && git diff --stat | tail -1"], timeout=30)
            if not diff.stdout.strip():
                return 0.6  # "edit" that changed nothing is not an edit
            invocations = f2p_invocations(f2p)
            if not invocations:
                return 0.8
            if test_patch.strip():
                # SWE-bench-style rows: F2P tests arrive via test_patch, not the
                # repo; apply AFTER the model edit, BEFORE the test run.
                applied = subprocess.run(
                    ["docker", "exec", "-i", cid, "bash", "-c",
                     "cd /testbed && git apply -"],
                    input=test_patch, text=True, capture_output=True,
                    timeout=60, check=False)
                if applied.returncode != 0:
                    return 0.8  # model edit conflicts with test fixture
            for tc in invocations:
                t = run(["docker", "exec", cid, "bash", "-c",
                         f"cd /testbed && {self.ENV_BOOTSTRAP}{tc}"],
                        timeout=self.test_timeout)
                if t.returncode != 0:
                    return 0.8
            return 1.0
        except subprocess.TimeoutExpired:
            return 0.6
        finally:
            run(["docker", "rm", "-f", cid], timeout=60)


class VerifiedRewardComputer:
    """Batch reward: parse ladder everywhere; docker tiers where state exists.

    Degradation ladder is monotone: docker problems can never score below the
    parse tier the completion already earned.
    """

    def __init__(self, decision_reward, extract_command, *, workers: int = 4,
                 group_timeout: int = 420, verifier: StateVerifier | None = None,
                 log=print):
        self.decision_reward = decision_reward
        self.extract_command = extract_command
        self.verifier = verifier or StateVerifier()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.group_timeout = group_timeout
        self.log = log
        self.stats = {"verified_calls": 0, "tier_08": 0, "tier_10": 0,
                      "degraded": 0, "unverified_rows": 0}

    def compute(self, texts: list[str], ctx_files: list[set[str]],
                state_tags: list[str | None], f2ps: list[list[str]],
                test_patches: list[str] | None = None) -> list[float]:
        if test_patches is None:
            test_patches = [""] * len(texts)
        # parse ladder, clamped to 0.6 everywhere: 0.8/1.0 now mean EXECUTION
        # TRUTH only (gate-resolved policy: unverified rows cap at 0.6 too)
        base = [min(self.decision_reward(t, cf), 0.6)
                for t, cf in zip(texts, ctx_files)]
        futures = {}
        for i, (text, tag) in enumerate(zip(texts, state_tags)):
            if tag is None:
                self.stats["unverified_rows"] += 1
                continue
            if base[i] < 0.6:
                continue  # not an edit — parse tier is final
            cmd = self.extract_command(text)
            if not cmd:
                continue
            if not self.verifier.state_exists(tag):
                continue  # cache miss -> row behaves as unverified this step
            futures[i] = self.pool.submit(self.verifier.verify, tag, cmd, f2ps[i],
                                          test_patches[i])
            self.stats["verified_calls"] += 1
        for i, fut in futures.items():
            try:
                tier = fut.result(timeout=self.group_timeout)
            except Exception as e:  # noqa: BLE001 — includes FutTimeout
                self.stats["degraded"] += 1
                self.log(f"[verified-reward] degrade idx={i}: {type(e).__name__}", flush=True)
                continue
            base[i] = max(base[i], tier)
            if tier >= 1.0:
                self.stats["tier_10"] += 1
            elif tier >= 0.8:
                self.stats["tier_08"] += 1
        return base
