#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Base Gemma Chat</title>
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
    <h1>Base Gemma Chat</h1>
    <div class="meta" id="meta"></div>
  </header>
  <section id="chat"></section>
  <form id="form">
    <textarea id="prompt" placeholder="Type a message..." autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="controls">
    <label>max_tokens <input id="max_tokens" type="number" min="1" max="4096" value="512"></label>
    <label>temperature <input id="temperature" type="number" min="0" max="2" step="0.05" value="0.2"></label>
    <button id="clear" type="button">Clear</button>
  </div>
</main>
<script>
const metaText = __META__;
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const prompt = document.getElementById("prompt");
const send = document.getElementById("send");
const clear = document.getElementById("clear");
const maxTokens = document.getElementById("max_tokens");
const temperature = document.getElementById("temperature");
document.getElementById("meta").textContent = metaText;
let history = [];
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
        max_tokens: Number(maxTokens.value),
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


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8101/v1")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    meta = f"{args.model} via {args.api}"

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
            body = HTML.replace("__META__", json.dumps(meta)).encode("utf-8")
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
                req = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": max(1, min(int(payload.get("max_tokens", 512)), 4096)),
                    "temperature": float(payload.get("temperature", 0.2)),
                }
                data = post_json(f"{args.api}/chat/completions", req)
                text = data["choices"][0]["message"]["content"]
                self.send_json({"text": text})
            except urllib.error.HTTPError as exc:
                self.send_json({"error": exc.read().decode("utf-8", "replace")}, status=500)
            except Exception as exc:
                self.send_json({"error": repr(exc)}, status=500)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port} -> {meta}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
