#!/usr/bin/env python3
"""Convert r1v3r/multi_SWE_Bench_Rust rows -> multi-swe-bench PullRequest JSONL (per repo).

PullRequest schema (multi_swe_bench.harness.pull_request): org, repo, number, state,
title, body, base{label,ref,sha}, resolved_issues[], fix_patch, test_patch.
r1v3r row: repo="org/name", pull_number, patch (=fix_patch), test_patch, base_commit,
problem_statement, FAIL_TO_PASS, PASS_TO_PASS, instance_id.
Writes one JSONL per repo (the harness builds images per repo).
"""
import json, os
from collections import defaultdict
from datasets import load_dataset

OUT = "/home/ironbcc/projects/gemma4-31B-Coder/data/mswe_rust_prs"
os.makedirs(OUT, exist_ok=True)
d = load_dataset("r1v3r/multi_SWE_Bench_Rust", split="train")
byrepo = defaultdict(list)
for r in d:
    org, name = (r["repo"].split("/", 1) + [""])[:2]
    pr = {
        "org": org, "repo": name, "number": int(r["pull_number"]),
        "state": "closed", "title": (r.get("problem_statement") or "")[:200], "body": r.get("problem_statement") or "",
        "base": {"label": f"{org}:main", "ref": "main", "sha": r.get("base_commit") or ""},
        "resolved_issues": [],
        "fix_patch": r.get("patch") or "", "test_patch": r.get("test_patch") or "",
        "lang": "rust",
    }
    byrepo[(org, name)].append(pr)
manifest = {}
for (org, name), prs in byrepo.items():
    fn = f"{OUT}/{org}__{name}.jsonl"
    with open(fn, "w") as f:
        for pr in sorted(prs, key=lambda p: p["number"]):
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    manifest[f"{org}/{name}"] = (len(prs), fn)
print("WROTE per-repo JSONL:")
for k, (n, fn) in sorted(manifest.items()):
    print(f"  {k}: {n} -> {os.path.basename(fn)}")
print("TOTAL", sum(n for n,_ in manifest.values()))
