#!/usr/bin/env python3
"""Post-run diagnostic: edit rate + verified tiers on state-backed prompts.

Replays N state-backed decision prompts against a SERVED policy (vLLM openai
endpoint, e.g. merged vgrpo cp300 on 8012) at eval temperature, extracts the
command, and scores it with the SAME StateVerifier used in training. Direct
artifact for: does the policy edit on kwai/swe-smith stall prefixes, and do
edits execute (0.8) / pass F2P (1.0)?

Usage (box):
  .venv-eval/bin/python phaseE_rl/diagnose_vgrpo_states.py \
      --api http://localhost:8012/v1 --model <served> --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_swe_decision_state import prompt_hash, state_tag  # noqa: E402
from verified_reward import StateVerifier, load_fixture_index  # noqa: E402
from grpo_swe_edit_decision import _EDIT, extract_command  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8012/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/rlvr_swe_decision_v2_launchmix.jsonl")
    ap.add_argument("--fixtures", default="data/rlvr_swe_decision_v2_fixtures.jsonl")
    ap.add_argument("--index", default="runs/rlvr_state_cache/index.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="runs/vgrpo_diag/state_replay.jsonl")
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=a.api, api_key="dummy")
    tags = set(json.load(open(a.index)))
    fixtures = load_fixture_index(a.fixtures)
    verifier = StateVerifier()
    tools = [{"type": "function", "function": {
        "name": "bash", "description": "run bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]

    rows = []
    for line in open(a.data):
        r = json.loads(line)
        if f"rlvr-state:{prompt_hash(r)}" in tags:
            rows.append(r)
        if len(rows) >= a.n:
            break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    counts = {"no_call": 0, "read": 0, "edit": 0, "tier_08": 0, "tier_10": 0}
    with open(a.out, "w") as fh:
        for r in rows:
            resp = client.chat.completions.create(
                model=a.model, messages=r["messages"], tools=tools,
                temperature=a.temperature, max_tokens=768)
            m = resp.choices[0].message
            # served path returns structured tool_calls; fall back to raw parse
            if m.tool_calls:
                cmd = json.loads(m.tool_calls[0].function.arguments).get("command", "")
            else:
                cmd = extract_command(m.content or "") or ""
            rec = {"instance_id": r["instance_id"], "source": r["source"],
                   "cmd": cmd[:300], "reasoning_len": len(getattr(m, "reasoning", "") or "")}
            if not cmd.strip():
                counts["no_call"] += 1
                rec["klass"] = "no_call"
            elif not _EDIT.search(cmd):
                counts["read"] += 1
                rec["klass"] = "read"
            else:
                counts["edit"] += 1
                rec["klass"] = "edit"
                fx = fixtures.get((r["source"], r["instance_id"])) or {}
                tier = verifier.verify(f"rlvr-state:{prompt_hash(r)}", cmd,
                                       list(fx.get("f2p") or []),
                                       fx.get("test_patch") or "")
                rec["tier"] = tier
                if tier >= 1.0:
                    counts["tier_10"] += 1
                elif tier >= 0.8:
                    counts["tier_08"] += 1
            fh.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec.get(k) for k in ("instance_id", "klass", "tier")}), flush=True)
    print(f"[diag] {counts} of n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
