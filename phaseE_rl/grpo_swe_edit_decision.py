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
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
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
            if len(tok(prompt, add_special_tokens=False)["input_ids"]) > a.max_prompt_tokens:
                continue  # long-prompt batches OOM the backward at num_gen=4
            rows.append({"prompt": prompt, "ctx_files": json.dumps(sorted(ctx_files))})
            if a.limit and len(rows) >= a.limit:
                break
    ds = Dataset.from_list(rows)
    print(f"[grpo-swe] {len(ds)} decision prompts; num_gen={a.num_gen} steps={a.max_steps}", flush=True)

    # The MM wrapper class demands mm_token_type_ids in training and mis-shapes
    # decode logits under GRPO's generate; loading the text class directly leaves
    # weights RANDOMLY INITIALIZED (MM checkpoint keys don't map). Correct path:
    # load the MM class (weights map), then GRAFT its language tower + lm_head into
    # an empty text-class shell and drop the vision parts.
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
    full_cfg = AutoConfig.from_pretrained(a.base)
    mm = Gemma4ForConditionalGeneration.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    with init_empty_weights():
        base = Gemma4ForCausalLM(full_cfg.text_config)
    lang = mm.model.language_model if hasattr(mm.model, "language_model") else mm.language_model
    base.model = lang
    base.lm_head = mm.lm_head
    base.config = full_cfg.text_config
    base.generation_config = mm.generation_config
    del mm
    torch.cuda.empty_cache()
    metas = [n for n, p in base.named_parameters() if p.device.type == "meta"]
    assert not metas, f"FATAL: {len(metas)} params still meta after graft (first: {metas[:3]})"
    print("[grpo-swe] grafted language tower + lm_head from MM checkpoint (no random init)", flush=True)
    # sanity generation before any training spend
    _p = tok("def add(a, b):", return_tensors="pt").to(base.device)
    _g = base.generate(**_p, max_new_tokens=12, do_sample=False)
    print(f"[grpo-swe] sanity gen: {tok.decode(_g[0][_p['input_ids'].shape[1]:])!r}", flush=True)

    # Gemma-4 wraps projections in Gemma4ClippableLinear (plain peft can't wrap it;
    # unsloth patched this in the SFT venv). Manually merge the SFT adapter into the
    # INNER nn.Linear weights: W += (alpha/r) * B @ A. Disk base stays untouched.
    from safetensors.torch import load_file
    peft_cfg = json.load(open(f"{a.adapter}/adapter_config.json"))
    scale = peft_cfg["lora_alpha"] / peft_cfg["r"]
    sd = load_file(f"{a.adapter}/adapter_model.safetensors")
    a_keys = [k for k in sd if k.endswith("lora_A.weight")]
    merged = 0
    with torch.no_grad():
        for ak in a_keys:
            bk = ak.replace("lora_A.weight", "lora_B.weight")
            # key: base_model.model.<module path>.lora_A.weight -> module path
            mpath = ak.replace("base_model.model.", "").replace(".lora_A.weight", "")
            # adapter was saved against the multimodal layout; text tower drops the prefix
            mpath = mpath.replace("model.language_model.", "model.", 1)
            mod = base.get_submodule(mpath)
            lin = getattr(mod, "linear", mod)  # inner Linear for clippable wrappers
            delta = (sd[bk].to(torch.float32) @ sd[ak].to(torch.float32)) * scale
            lin.weight += delta.to(lin.weight.dtype).to(lin.weight.device)
            merged += 1
    assert merged > 100, f"FATAL: only {merged} adapter matrices merged — key remap wrong (sample: {list(sd)[:2]})"
    print(f"[grpo-swe] merged SFT adapter into {merged} inner linears (cold-start policy)", flush=True)
    model = base  # text-only class: no mm_token_type_ids anywhere

    # Terminate generation right after the tool call (49=<tool_call|>) or turn end
    # (106=<turn|>): without these, completions ramble in the thought channel to the
    # length cap, never parse, and every reward is 0 (observed: clipped_ratio 1.0).
    model.generation_config.eos_token_id = [tok.eos_token_id, 49, 106]

    _dbg = {"n": 0}

    def reward_fn(completions, ctx_files=None, **kw):
        out = []
        for comp, cf in zip(completions, ctx_files):
            text = comp if isinstance(comp, str) else comp[-1].get("content", "")
            out.append(decision_reward(text, set(json.loads(cf))))
        if _dbg["n"] < 4:
            _dbg["n"] += 1
            t0 = completions[0] if isinstance(completions[0], str) else completions[0][-1].get("content", "")
            print(f"[dbg] completion head: {t0[:220]!r}", flush=True)
            print(f"[dbg] completion tail: {t0[-120:]!r}  rewards={out}", flush=True)
        return out

    cfg = GRPOConfig(
        output_dir=a.out, per_device_train_batch_size=a.num_gen, num_generations=a.num_gen,
        gradient_accumulation_steps=1, learning_rate=a.lr, max_steps=a.max_steps,
        max_completion_length=a.max_completion,
        logging_steps=1, save_steps=a.max_steps, save_total_limit=3, bf16=True,
        report_to="none", temperature=0.9, beta=0.04,
        gradient_checkpointing=True,
        # trl overrides model.generation_config (grpo_trainer.py:1418) and hardcodes
        # the tokenizer eos (:806); generation_kwargs is the supported stop override.
        generation_kwargs={"eos_token_id": [tok.eos_token_id, 49, 106]},
    )
    # Build exact LoRA targets at runtime: for each projection, use the INNER
    # nn.Linear of Gemma4ClippableLinear wrappers (plain peft can't wrap the
    # wrapper class); use the module itself only when it is already a Linear.
    import torch.nn as nn
    PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = []
    for name, mod in model.named_modules():
        if not name.rsplit(".", 1)[-1] in PROJ:
            continue
        if isinstance(mod, nn.Linear):
            targets.append(name)
        elif isinstance(getattr(mod, "linear", None), nn.Linear):
            targets.append(f"{name}.linear")
    print(f"[grpo-swe] {len(targets)} LoRA target linears (first: {targets[:2]})", flush=True)
    rl_lora = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=targets,
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
