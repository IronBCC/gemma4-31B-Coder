#!/usr/bin/env python3
"""Rust raw-base baseline driver over Multi-SWE image-backed instances.

Self-contained agent rollout + scoring, no mini-swe dependency:
  per instance: docker run -d <image> -> agent loop (openai client on the local
  vLLM server, bash tool, docker exec) -> git diff = model patch -> score by the
  image's canonical /home/test-run.sh (applies test.patch, runs cargo test).

resolved = test-run.sh exit 0 after the model's edits (all F2P+P2P green).
Baseline mode serves the RAW base; the same driver evaluates future adapters.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from openai import OpenAI

SYSTEM = ("You are a practical software engineer using a shell to fix one repository bug. "
          "Prefer a small correct source edit over extended inspection.")
INSTRUCTIONS = """<instructions>
You are in the repository at {repo_dir}. Modify only source files needed for the fix.
Do not edit tests, CI, or unrelated code.
- Locate the relevant source, inspect only the needed lines.
- By command 15, make the smallest plausible source edit. Never repeat a read command.
- After editing, optionally run a focused check (cargo check -q).
- Finish with exactly: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
</instructions>"""
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "run bash in the repository container",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]
MAX_OBS = 2000
STEP_CAP = 40


def sh(cmd: list[str], timeout: int = 300) -> tuple[str, int]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr), p.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 124


def dexec(cid: str, command: str, timeout: int = 120) -> tuple[str, int]:
    # NON-login shell: bash -lc sources /etc/profile which RESETS PATH in most
    # mswebench images, dropping /usr/local/cargo/bin -> cargo rc=127 phantoms.
    return sh(["docker", "exec", cid, "bash", "-c", command], timeout)


def compact(text: str) -> str:
    if len(text) <= MAX_OBS:
        return text
    return text[:MAX_OBS // 2] + "\n[... truncated ...]\n" + text[-MAX_OBS // 2:]


def rollout(client: OpenAI, model: str, cid: str, repo_dir: str, problem: str, log) -> int:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<pr_description>\n{problem}\n</pr_description>\n\n"
                                        + INSTRUCTIONS.format(repo_dir=repo_dir)}]
    n_cmds = 0
    for step in range(STEP_CAP):
        r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS,
                                           temperature=0.7, max_tokens=4096)
        m = r.choices[0].message
        entry = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in m.tool_calls]
        msgs.append(entry)
        if not m.tool_calls:
            msgs.append({"role": "user", "content": "Reply with exactly one bash tool call."})
            continue
        tc = m.tool_calls[0]
        try:
            cmd = json.loads(tc.function.arguments).get("command", "")
        except json.JSONDecodeError:
            cmd = ""
        n_cmds += 1
        log.write(json.dumps({"step": step, "cmd": cmd[:300]}) + "\n")
        log.flush()
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
            break
        out, rc = dexec(cid, f"cd {repo_dir} && {cmd}")
        msgs.append({"role": "tool", "tool_call_id": tc.id,
                     "content": compact(f"<returncode>{rc}</returncode>\n{out}")})
    return n_cmds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mswe_rust_prs_imagebacked.jsonl")
    ap.add_argument("--api", default="http://localhost:8012/v1")
    ap.add_argument("--model", default="gemma4-rust-baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--one-per-repo", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    if a.one_per_repo:
        seen, pick = set(), []
        for r in rows:
            key = r.get("repo")
            if key not in seen:
                seen.add(key)
                pick.append(r)
        rows = pick
    if a.limit:
        rows = rows[: a.limit]

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=a.api, api_key="dummy")
    results = []
    for r in rows:
        iid = r.get("instance_id") or f"{r['org']}__{r['repo']}-{r['number']}"
        img = r["image_name"]
        t0 = time.time()
        cid_out, rc = sh(["docker", "run", "-d", img, "sleep", "infinity"])
        cid = cid_out.strip().splitlines()[-1][:12]
        rec = {"instance_id": iid, "image": img}
        try:
            repo_out, _ = dexec(cid, "ls /home | grep -vE '\\.(sh|patch)$' | head -1")
            repo_dir = f"/home/{repo_out.strip()}"
            with open(outdir / f"{iid}.cmds.jsonl", "w") as log:
                n = rollout(client, a.model, cid, repo_dir, r.get("body", ""), log)
            diff, _ = dexec(cid, f"cd {repo_dir} && git diff")
            rec["n_cmds"] = n
            rec["patch_len"] = len(diff.strip())
            (outdir / f"{iid}.patch").write_text(diff)
            score_out, score_rc = dexec(cid, "cd /home && bash test-run.sh", timeout=1200)
            rec["test_rc"] = score_rc
            rec["resolved"] = bool(diff.strip()) and score_rc == 0
            tail = "\n".join(score_out.splitlines()[-15:])
            rec["test_tail"] = tail
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["resolved"] = False
        finally:
            sh(["docker", "rm", "-f", cid], timeout=60)
        rec["wall_s"] = round(time.time() - t0, 1)
        results.append(rec)
        print(json.dumps({k: rec.get(k) for k in
                          ("instance_id", "n_cmds", "patch_len", "test_rc", "resolved", "wall_s", "error")}),
              flush=True)
        with open(outdir / "results.jsonl", "w") as fh:
            for x in results:
                fh.write(json.dumps(x) + "\n")
    n_res = sum(1 for x in results if x.get("resolved"))
    n_patch = sum(1 for x in results if x.get("patch_len"))
    print(f"[baseline] instances={len(results)} non_empty_patches={n_patch} resolved={n_res}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
