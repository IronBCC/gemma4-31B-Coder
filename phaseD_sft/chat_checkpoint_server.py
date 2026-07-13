#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Checkpoint Chat</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #111827; }
    main { max-width: 980px; margin: 0 auto; padding: 18px; }
    header { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 12px; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    .meta { color: #4b5563; font-size: 13px; overflow-wrap: anywhere; }
    #chat { height: calc(100vh - 178px); min-height: 360px; overflow-y: auto; background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; }
    .msg { margin: 0 0 14px; padding: 10px 12px; border-radius: 8px; white-space: pre-wrap; line-height: 1.4; }
    .user { background: #e5f0ff; margin-left: 10%; }
    .assistant { background: #f3f4f6; margin-right: 10%; }
    form { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-top: 12px; }
    textarea { resize: vertical; min-height: 54px; max-height: 180px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; font: inherit; }
    button { border: 0; border-radius: 8px; background: #111827; color: white; padding: 0 18px; font: inherit; cursor: pointer; }
    button:disabled { opacity: .55; cursor: default; }
    .controls { display: flex; gap: 12px; align-items: center; color: #4b5563; font-size: 13px; margin-top: 8px; }
    input { width: 72px; }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Checkpoint Chat</h1>
    <div class="meta" id="meta"></div>
  </header>
  <section id="chat"></section>
  <form id="form">
    <textarea id="prompt" placeholder="Type a message..." autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="controls">
    <label>max_new_tokens <input id="max_new" type="number" min="1" max="2048" value="512"></label>
    <label>temperature <input id="temperature" type="number" min="0" max="2" step="0.05" value="0.2"></label>
    <button id="clear" type="button">Clear</button>
  </div>
</main>
<script>
const checkpoint = __CHECKPOINT__;
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const prompt = document.getElementById("prompt");
const send = document.getElementById("send");
const clear = document.getElementById("clear");
const maxNew = document.getElementById("max_new");
const temperature = document.getElementById("temperature");
const meta = document.getElementById("meta");
let history = [];
meta.textContent = checkpoint;
function render() {
  chat.innerHTML = "";
  for (const m of history) {
    const div = document.createElement("div");
    div.className = "msg " + m.role;
    div.textContent = m.content;
    chat.appendChild(div);
  }
  chat.scrollTop = chat.scrollHeight;
}
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = prompt.value.trim();
  if (!text) return;
  history.push({role: "user", content: text});
  prompt.value = "";
  send.disabled = true;
  history.push({role: "assistant", content: "..."});
  render();
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({
        history: history.slice(0, -1),
        max_new_tokens: Number(maxNew.value),
        temperature: Number(temperature.value)
      })
    });
    const data = await resp.json();
    history[history.length - 1] = {role: "assistant", content: data.text || data.error || "(empty)"};
  } catch (err) {
    history[history.length - 1] = {role: "assistant", content: String(err)};
  } finally {
    send.disabled = false;
    prompt.focus();
    render();
  }
});
clear.addEventListener("click", () => { history = []; render(); });
prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
});
render();
</script>
</body>
</html>
"""


def build_inputs(tokenizer, messages, device):
    def to_model_inputs(rendered):
        if isinstance(rendered, dict):
            rendered = {k: v.to(device) for k, v in rendered.items()}
            rendered.setdefault("attention_mask", torch.ones_like(rendered["input_ids"]))
            return rendered
        input_ids = rendered.to(device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    plain = [
        {
            "role": m["role"],
            "content": m["content"],
            "tool_calls": m.get("tool_calls"),
            "reasoning": m.get("reasoning"),
            "reasoning_content": m.get("reasoning_content"),
        }
        for m in messages
    ]
    typed = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in messages
    ]
    for candidate in (plain, typed):
        try:
            return to_model_inputs(tokenizer.apply_chat_template(
                candidate, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ))
        except TypeError:
            try:
                return to_model_inputs(tokenizer.apply_chat_template(
                    candidate, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ))
            except Exception:
                continue
        except Exception:
            continue
    raise RuntimeError("failed to render chat template for both plain and typed content")


def normalize_response(text: str) -> str:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:diff|patch|python|bash)?\s*\n(.*?)\n```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()

    first_diff = stripped.find("--- ")
    if first_diff > 0:
        stripped = stripped[first_diff:].lstrip()

    lines = stripped.splitlines()
    if lines and re.fullmatch(r"a/[^\s]+(?:\s+b/[^\s]+)?", lines[0]) and any(line.startswith("@@") for line in lines[1:]):
        path = lines[0].split()[0]
        rest = lines[1:]
        if not rest or not rest[0].startswith("--- "):
            stripped = "\n".join([f"--- {path}", f"+++ b/{path[2:]}", *rest])

    return stripped.replace("Focused test", "focused test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--base-tokenizer", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    print(f"[load] {args.checkpoint}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=args.max_seq,
        dtype=None,
        load_in_4bit=args.load_4bit,
    )
    if args.base_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    print(f"[template] using tokenizer Gemma chat template from {args.base_tokenizer or args.checkpoint}", flush=True)
    FastLanguageModel.for_inference(model)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print("[http] " + fmt % values, flush=True)

        def send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = HTML.replace("__CHECKPOINT__", json.dumps(args.checkpoint)).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/chat":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                messages = payload.get("history") or []
                max_new = max(1, min(int(payload.get("max_new_tokens", 512)), 2048))
                temp = float(payload.get("temperature", 0.2))
                do_sample = temp > 0
                with lock, torch.inference_mode():
                    inputs = build_inputs(tokenizer, messages, model.device)
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=temp if do_sample else None,
                        do_sample=do_sample,
                    )
                    prompt_len = inputs["input_ids"].shape[1]
                    text = normalize_response(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True))
                self.send_json({"text": text})
            except Exception as exc:
                tb = traceback.format_exc()
                print(tb, flush=True)
                self.send_json({"error": repr(exc), "traceback": tb}, status=500)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
