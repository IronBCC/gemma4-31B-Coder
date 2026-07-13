#!/usr/bin/env python3
"""Phase-E RLVR — GRPO on SWE EDIT-DECISION points (behavior-gate lever).

Why: SFT plateaued (v7/v8 both ~11-13/30 non-empty patches vs gate 18/30) because
imitation can't force act-vs-explore decisiveness. This trains the exact decision:
given a REAL agent-trajectory prefix at the point where the model historically
stalls (pre-first-edit / read-loop onset), reward completions that emit a valid
bash tool call that EDITS a file seen in context.

Shaped verifiable reward (phase 1, no docker in the loop — pure parse, fast):
  0.0  no parseable tool call
  0.2  valid bash call, read-only
  0.6  edit command
  1.0  edit command touching a file mentioned in the prompt context
Phase 2 (later): docker apply-check for the top tier.

Frozen-base guarantee: v8cp30 SFT adapter is merged as the cold-start POLICY ONLY
(in memory); GRPO trains a FRESH LoRA on top; base weights on disk never change.

Smoke first:
  CUDA_VISIBLE_DEVICES=1 .venv-rl/bin/python phaseE_rl/grpo_swe_edit_decision.py \
      --data data/rlvr_swe_decision_v1.jsonl \
      --adapter adapters/swe_edit_v8_49k_s1/checkpoint-30 \
      --out adapters/swe_grpo_v1 --max-steps 5 --num-gen 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# gemma4 raw tool-call markup as the model emits it (matches training render)
_TOOLCALL = re.compile(r"call:bash\{(.*?)\}<tool_call\|>", re.DOTALL)
_TOOLCALL_LOOSE = re.compile(r"call:bash\{(.*)", re.DOTALL)
_CMD = re.compile(r'"command"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
_EDIT = re.compile(r"sed -i|perl -i|apply_patch|git apply|cat > |cat >> |tee |>>|python3? - <<|open\(.+[\"']w[\"']")
_PATHISH = re.compile(r"[\w./-]+\.(?:py|rs|c|cc|cpp|h|hpp|toml|cfg|txt|rst)")


def extract_command(text: str) -> str | None:
    m = _TOOLCALL.search(text) or _TOOLCALL_LOOSE.search(text)
    if not m:
        return None
    body = m.group(1)
    cm = _CMD.search(body)
    if cm:
        try:
            return json.loads(f'"{cm.group(1)}"')
        except json.JSONDecodeError:
            return cm.group(1)
    # gemma4 compact form: command:<|"|>...<|"|>
    cm2 = re.search(r'command:\s*<\|"\|>(.*?)<\|"\|>', body, re.DOTALL)
    return cm2.group(1) if cm2 else None


def decision_reward(completion_text: str, context_files: set[str]) -> float:
    cmd = extract_command(completion_text)
    if not cmd or not cmd.strip():
        return 0.0
    if not _EDIT.search(cmd):
        return 0.2
    touched = set(_PATHISH.findall(cmd))
    if touched & context_files:
        return 1.0
    return 0.6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rlvr_swe_decision_v1.jsonl")
    ap.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    ap.add_argument("--adapter", default="adapters/swe_edit_v8_49k_s1/checkpoint-30")
    ap.add_argument("--template", default="phaseH_eval/tool_chat_template_gemma4_thinkopen_v2.jinja",
                    help="serve-parity chat template (opens the thought channel)")
    ap.add_argument("--out", default="adapters/swe_grpo_v1")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--num-gen", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(a.base)
    template = open(a.template).read()
    tools = [{"type": "function", "function": {
        "name": "bash", "description": "run bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]

    rows = []
    with open(a.data) as fh:
        for line in fh:
            r = json.loads(line)
            msgs = r["messages"]
            prompt = tok.apply_chat_template(
                msgs, tools=tools, chat_template=template, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            ctx_files = set()
            for m in msgs:
                c = m.get("content") or ""
                if isinstance(c, str):
                    ctx_files.update(_PATHISH.findall(c))
            rows.append({"prompt": prompt, "ctx_files": json.dumps(sorted(ctx_files))})
            if a.limit and len(rows) >= a.limit:
                break
    ds = Dataset.from_list(rows)
    print(f"[grpo-swe] {len(ds)} decision prompts; num_gen={a.num_gen} steps={a.max_steps}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.adapter)
    model = model.merge_and_unload()  # policy cold-start; disk base untouched

    def reward_fn(completions, ctx_files=None, **kw):
        out = []
        for comp, cf in zip(completions, ctx_files):
            text = comp if isinstance(comp, str) else comp[-1].get("content", "")
            out.append(decision_reward(text, set(json.loads(cf))))
        return out

    cfg = GRPOConfig(
        output_dir=a.out, per_device_train_batch_size=a.num_gen, num_generations=a.num_gen,
        gradient_accumulation_steps=1, learning_rate=a.lr, max_steps=a.max_steps,
        max_completion_length=a.max_completion,
        logging_steps=1, save_steps=a.max_steps, save_total_limit=3, bf16=True,
        report_to="none", temperature=0.9, beta=0.04,
    )
    rl_lora = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, args=cfg,
                          train_dataset=ds, reward_funcs=[reward_fn], peft_config=rl_lora)
    print("[grpo-swe] starting RLVR (edit-decision reward)…", flush=True)
    trainer.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"[grpo-swe] done -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
