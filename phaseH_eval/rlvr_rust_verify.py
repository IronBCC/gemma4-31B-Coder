#!/usr/bin/env python3
"""Verify rlvr-code-data-Rust solutions by ACTUAL compile + test execution.

The project invariant: only train on execution-verified data. Each rlvr row ships a
Rust solution + a list of assert_eq! test cases. We assemble {solution + a main()/test
harness of the asserts} into a tiny crate, `cargo build` then run, and keep only rows
that COMPILE and PASS all asserts. cargo isn't on the host, so every build runs inside a
pinned rust:1-slim container (one container reused across many rows via `docker exec`).

Output: a JSONL of verified rows {id, problem, solution, tests, n_asserts} for the SFT
builder. Parallelized with a thread pool; each worker shells `cargo` in the shared
container against a unique temp crate dir.

Usage:
  rlvr_rust_verify.py --in data/rlvr_rust_raw --out data/rlvr_rust_verified.jsonl \
      --container rlvr-rustc --limit 0 --workers 12
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


def _parse_tests(raw) -> list[str]:
    """translated_test_cases is a python-repr list-of-strings (each an assert_eq!...)."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if not raw:
        return []
    s = str(raw).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
        return [str(v)]
    except (ValueError, SyntaxError):
        # fall back: split on the assert boundary
        return [a for a in s.split("\n") if "assert" in a]


def _make_crate_src(solution: str, tests: list[str]) -> str:
    """Wrap solution + asserts into one main.rs whose `cargo run` exits 0 iff all pass."""
    body = "\n    ".join(tests)
    return (
        f"{solution}\n\n"
        "fn main() {\n"
        f"    {body}\n"
        '    println!("RLVR_ALL_ASSERTS_PASSED");\n'
        "}\n"
    )


def _verify_one(container: str, work_root: str, row: dict, timeout_s: int = 90) -> dict | None:
    sol = (row.get("translated_solution") or "").strip()
    tests = _parse_tests(row.get("translated_test_cases"))
    if not sol or not tests:
        return None
    rid = row.get("id") or uuid.uuid4().hex
    crate = f"{work_root}/c_{rid[:16]}"
    src = _make_crate_src(sol, tests)
    cargo_toml = (
        '[package]\nname = "v"\nversion = "0.0.0"\nedition = "2021"\n'
        '[[bin]]\nname = "v"\npath = "src/main.rs"\n'
    )
    # write both files via quoted heredocs (no shell-expansion of contents), then build+run
    script = (
        f"set -e; mkdir -p {crate}/src\n"
        f"cat > {crate}/Cargo.toml <<'RLVRTOML'\n{cargo_toml}\nRLVRTOML\n"
        f"cat > {crate}/src/main.rs <<'RLVREOF'\n{src}\nRLVREOF\n"
        f"cd {crate} && timeout {timeout_s} cargo run --quiet 2>&1"
    )
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "bash", "-lc", script],
            capture_output=True, text=True, timeout=timeout_s + 30,
        )
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return None
    finally:
        subprocess.run(["docker", "exec", container, "bash", "-lc", f"rm -rf {crate}"],
                       capture_output=True, timeout=30)
    if "RLVR_ALL_ASSERTS_PASSED" in out and proc.returncode == 0:
        return {"id": rid, "problem": row.get("translated_problem", ""),
                "solution": sol, "tests": tests, "n_asserts": len(tests)}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--container", default="rlvr-rustc")
    ap.add_argument("--work-root", default="/tmp/rlvr_crates")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(a.inp)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    rows = [dict(r) for r in ds]
    subprocess.run(["docker", "exec", a.container, "bash", "-lc", f"mkdir -p {a.work_root}"],
                   capture_output=True)

    kept = 0
    done = 0
    with open(a.out, "w") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_verify_one, a.container, a.work_root, r): r for r in rows}
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                kept += 1
            if done % 500 == 0:
                print(f"  verified {done}/{len(rows)}  kept {kept} ({100*kept/done:.0f}%)", flush=True)
    print(f"DONE verified={done} kept={kept} ({100*kept/max(1,done):.1f}%) -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
