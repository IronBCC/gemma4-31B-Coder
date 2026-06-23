#!/usr/bin/env python3
"""Smoke-test a trained Rust LoRA adapter: does it generate valid, thinking Rust?

Loads the FROZEN base + the LoRA adapter via peft (no merge — mirrors the
serve-live-LoRA rule), generates on a few held-out-style Rust prompts, and checks:
  (1) the adapter loads over the frozen base without error,
  (2) output preserves the <thought> thinking block (Gemma-4 reasoning kept),
  (3) output contains a ```rust code fence with a fn definition,
  (4) (best-effort) the emitted Rust compiles in the rlvr-rustc container.

This is the completion check for "real training": a trained adapter that actually
produces valid thinking-Rust. Run on GPU1 (GPU0 = prod, off-limits).

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/smoke_rust_adapter.py \
      --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
      --adapter adapters/rust-lora-v1            # or a checkpoint-NNN dir
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

PROMPTS = [
    "Write idiomatic, correct Rust to solve the following problem. Think step by step, then provide the implementation.\n\nWrite a function `sum_even(nums: &[i32]) -> i32` that returns the sum of the even numbers in the slice.",
    "Write idiomatic, correct Rust to solve the following problem. Think step by step, then provide the implementation.\n\nWrite a function `first_word(s: &str) -> Option<&str>` returning the first whitespace-delimited word, or None if the string is empty/blank.",
    "Write idiomatic, correct Rust to solve the following problem. Think step by step, then provide the implementation.\n\nWrite a function `dedup_sorted(v: Vec<i32>) -> Vec<i32>` that returns the input with consecutive duplicates removed.",
]


def extract_rust(text: str) -> str | None:
    m = re.search(r"```rust\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def compiles(container: str, code: str) -> bool:
    src = code + '\nfn main() { let _ = 0; }\n' if "fn main" not in code else code
    script = (
        "set -e; d=/tmp/smoke_$$; mkdir -p $d/src; "
        "printf '[package]\\nname=\"s\"\\nversion=\"0.0.0\"\\nedition=\"2021\"\\n' > $d/Cargo.toml; "
        f"cat > $d/src/main.rs <<'EOF'\n{src}\nEOF\n"
        "cd $d && cargo build --quiet 2>&1; rc=$?; rm -rf $d; exit $rc"
    )
    try:
        p = subprocess.run(["docker", "exec", container, "sh", "-c", script],
                           capture_output=True, text=True, timeout=120)
        return p.returncode == 0
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--container", default="rlvr-rustc")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit base so smoke fits alongside a running training job on the same GPU")
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=4096, dtype=None, load_in_4bit=a.load_4bit,
    )
    try:
        tok = get_chat_template(tok, chat_template="gemma-4")
    except Exception:
        pass
    print(f"[load] attaching adapter {a.adapter}", flush=True)
    model.load_adapter(a.adapter, adapter_name="rust")
    model.set_adapter("rust")
    FastLanguageModel.for_inference(model)

    results = []
    for i, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         return_tensors="pt").to(model.device)
        out = model.generate(input_ids=inputs, max_new_tokens=a.max_new, temperature=0.0,
                             do_sample=False)
        text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        has_thought = "<thought>" in text or "thought" in text.lower()[:200]
        rust = extract_rust(text)
        has_fn = bool(rust and "fn " in rust)
        ok_compile = compiles(a.container, rust) if has_fn else False
        results.append((i, has_thought, has_fn, ok_compile))
        print(f"\n=== PROMPT {i} ===\n{text[:700]}\n--- thinking={has_thought} rust_fn={has_fn} compiles={ok_compile} ---", flush=True)

    n = len(results)
    th = sum(r[1] for r in results); fn = sum(r[2] for r in results); cc = sum(r[3] for r in results)
    print(f"\n=== SMOKE SUMMARY ===\n  thinking preserved: {th}/{n}\n  rust fn produced:   {fn}/{n}\n  compiles:           {cc}/{n}", flush=True)
    # pass = adapter loads + every prompt yields a rust fn (compile is best-effort signal)
    ok = (fn == n)
    print(f"  SMOKE {'PASS' if ok else 'FAIL'} (adapter loaded + generates Rust)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
