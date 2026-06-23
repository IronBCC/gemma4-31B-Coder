#!/usr/bin/env python3
"""Build the Rust SFT dataset from execution-VERIFIED rlvr rows.

Input: data/rlvr_rust_verified.jsonl (rows that actually compiled + passed asserts).
Output: a chat-format dataset (HF save_to_disk) ready for Unsloth SFT, where each
example is a 2-turn conversation:
  user:      the problem statement (+ a terse "write idiomatic Rust" instruction)
  assistant: a <thought> reasoning block + the verified solution in a ```rust fence

Design choices (from gemma4-coder-phaseC-design / PLAN §6):
- Train in Gemma-4 thinking format: assistant content leads with a <thought>...</thought>
  block then the answer, to PRESERVE thinking behavior (>=75% reasoning-style target).
  rlvr has no gold reasoning, so we synthesize a short, faithful rationale from the
  problem+solution (NOT a fabricated long chain — a brief plan, kept honest).
- SHORTEST-CORRECT dedup: rlvr can have near-duplicate problems; keep one per
  normalized-problem key, preferring the solution with fewest chars (terse-correct,
  counters the wandering the baseline exposed).
- Dedup exact solution bodies too (MinHash-lite via normalized text hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _synth_thought(problem: str, solution: str) -> str:
    """A short, faithful plan — not a fabricated long CoT. Honest scaffolding only."""
    sig = ""
    m = re.search(r"fn\s+\w+\s*\([^)]*\)\s*(->\s*[^\{]+)?", solution)
    if m:
        sig = m.group(0).strip()
    bits = ["Plan the Rust implementation from the problem statement."]
    if sig:
        bits.append(f"Target signature: `{sig}`.")
    if "Option" in solution:
        bits.append("Handle the empty/none case explicitly via Option.")
    if "Result" in solution:
        bits.append("Propagate errors with Result.")
    if ".iter()" in solution or ".map(" in solution or ".filter(" in solution:
        bits.append("Use iterator combinators for the transform.")
    bits.append("Then verify against the provided assert_eq! cases.")
    return " ".join(bits)


def build(inp: str, out: str, instr: str) -> None:
    seen_problem: dict[str, dict] = {}
    seen_sol_hash: set[str] = set()
    n_in = 0
    with open(inp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            r = json.loads(line)
            sol = (r.get("solution") or "").strip()
            prob = (r.get("problem") or "").strip()
            if not sol or not prob:
                continue
            sh = hashlib.sha256(_norm(sol).encode()).hexdigest()
            if sh in seen_sol_hash:
                continue
            pkey = _norm(prob)[:400]
            prev = seen_problem.get(pkey)
            if prev is None or len(sol) < len(prev["_sol"]):
                seen_problem[pkey] = {"_sol": sol, "_prob": prob, "_tests": r.get("tests", [])}
            seen_sol_hash.add(sh)

    rows = []
    for v in seen_problem.values():
        prob, sol = v["_prob"], v["_sol"]
        thought = _synth_thought(prob, sol)
        assistant = f"<thought>\n{thought}\n</thought>\n\nHere is the Rust implementation:\n\n```rust\n{sol}\n```"
        rows.append({
            "messages": [
                {"role": "user", "content": f"{instr}\n\n{prob}"},
                {"role": "assistant", "content": assistant},
            ],
            "n_asserts": len(v.get("_tests", [])),
        })

    from datasets import Dataset
    ds = Dataset.from_list(rows)
    ds.save_to_disk(out)
    print(f"BUILT sft rows={len(rows)} (from {n_in} verified, deduped) -> {out}", flush=True)
    if rows:
        print("SAMPLE assistant:\n", rows[0]["messages"][1]["content"][:500], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/rlvr_rust_verified.jsonl")
    ap.add_argument("--out", default="data/rust_sft")
    ap.add_argument("--instr", default="Write idiomatic, correct Rust to solve the following problem. Think step by step, then provide the implementation.")
    a = ap.parse_args()
    build(a.inp, a.out, a.instr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
