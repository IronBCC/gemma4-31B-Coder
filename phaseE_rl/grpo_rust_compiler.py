#!/usr/bin/env python3
"""Phase-E RLVR — GRPO on Rust with a COMPILER/TEST verifiable reward (PLAN §7).

From the SFT adapter (rust-lora-v2), on-policy GRPO. Reward is the verifiable kind
PLAN §7 prescribes: 1.0 iff the generated Rust compiles AND passes the problem's
assert_eq! tests, with partial credit for compiling-but-failing (shapes the gradient
better than pure 0/1 on hard items). Reward is computed by assembling the completion
into a crate and running it in the pinned rust container (rlvr-rustc) — the same
execution gate used to verify the SFT data, now as the RL signal.

Single GPU1 (GPU0=prod). Frozen base + LoRA (never merged). Designed to run a small
--max-steps SMOKE first (prove the loop runs end-to-end) before any long run.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseE_rl/grpo_rust_compiler.py \
      --data data/rlvr_rust_verified.jsonl --adapter adapters/rust-lora-v2 \
      --out adapters/rust-grpo-v1 --max-steps 5 --num-gen 4 --container rlvr-rustc
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import uuid

CONTAINER = "rlvr-rustc"
_FENCE = re.compile(r"```rust\s*(.*?)```", re.DOTALL)


def _extract_rust(text: str) -> str | None:
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: a bare fn block
    return text if "fn " in text else None


def _parse_tests(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(t) for t in raw]
    try:
        v = ast.literal_eval(str(raw))
        return [str(t) for t in v] if isinstance(v, (list, tuple)) else [str(v)]
    except (ValueError, SyntaxError):
        return []


def _compile_and_test(code: str, tests: list[str], container: str, timeout_s: int = 60) -> float:
    """Verifiable reward: 1.0 pass-all-asserts, 0.3 compiles-only, 0.0 won't-compile/none."""
    if not code:
        return 0.0
    body = "\n    ".join(tests)
    src = f"{code}\n\nfn main() {{\n    {body}\n    println!(\"RLVR_OK\");\n}}\n"
    crate = f"/tmp/grpo_{uuid.uuid4().hex[:12]}"
    script = (
        f"set -e; mkdir -p {crate}/src\n"
        f"cat > {crate}/Cargo.toml <<'TOML'\n[package]\nname=\"g\"\nversion=\"0.0.0\"\nedition=\"2021\"\n[[bin]]\nname=\"g\"\npath=\"src/main.rs\"\nTOML\n"
        f"cat > {crate}/src/main.rs <<'EOF'\n{src}\nEOF\n"
        f"cd {crate} && (cargo build --quiet 2>/dev/null && echo BUILD_OK && (cargo run --quiet 2>/dev/null | grep -q RLVR_OK && echo RUN_OK || true) || true)"
    )
    try:
        p = subprocess.run(["docker", "exec", container, "sh", "-c", script],
                           capture_output=True, text=True, timeout=timeout_s + 20)
        out = p.stdout
    except subprocess.TimeoutExpired:
        out = ""
    finally:
        subprocess.run(["docker", "exec", container, "sh", "-c", f"rm -rf {crate}"],
                       capture_output=True, timeout=20)
    if "RUN_OK" in out:
        return 1.0
    if "BUILD_OK" in out:
        return 0.3
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rlvr_rust_verified.jsonl")
    ap.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    ap.add_argument("--adapter", default="adapters/rust-lora-v2")  # SFT cold-start
    ap.add_argument("--out", default="adapters/rust-grpo-v1")
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--num-gen", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # load SFT adapter as the policy cold-start (base+LoRA one pass)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.adapter, max_seq_length=a.max_prompt + a.max_completion, dtype=None, load_in_4bit=False,
    )
    FastLanguageModel.for_training(model)

    # build prompt dataset (problem -> chat prompt); keep tests alongside for the reward
    rows = []
    with open(a.data) as fh:
        for line in fh:
            r = json.loads(line)
            tests = r.get("tests") or _parse_tests(r.get("translated_test_cases"))
            prob = r.get("problem") or r.get("translated_problem")
            if prob and tests:
                rows.append({"prompt": f"Write idiomatic, correct Rust to solve:\n\n{prob}\n\nThink step by step, then give the implementation in a ```rust block.",
                             "tests": json.dumps(tests)})
            if a.limit and len(rows) >= a.limit:
                break
    ds = Dataset.from_list(rows)
    print(f"[grpo] {len(ds)} prompts; num_gen={a.num_gen} max_steps={a.max_steps}", flush=True)

    def reward_fn(completions, tests=None, **kw):
        out = []
        for comp, tj in zip(completions, tests):
            code = _extract_rust(comp if isinstance(comp, str) else comp[-1].get("content", ""))
            out.append(_compile_and_test(code, json.loads(tj), a.container))
        return out

    cfg = GRPOConfig(
        output_dir=a.out, per_device_train_batch_size=a.num_gen, num_generations=a.num_gen,
        gradient_accumulation_steps=1, learning_rate=1e-5, max_steps=a.max_steps,
        max_prompt_length=a.max_prompt, max_completion_length=a.max_completion,
        logging_steps=1, save_steps=a.max_steps, bf16=True, report_to="none",
        temperature=1.0, beta=0.04,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, args=cfg,
                          train_dataset=ds, reward_funcs=[reward_fn])
    print("[grpo] starting RLVR (compiler reward)…", flush=True)
    trainer.train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[grpo] done -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
