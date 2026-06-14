#!/usr/bin/env bash
# Latency profile — interactive single-user coding (stack spec decoding + Blackwell).
set -euo pipefail
CKPT="${1:?path to nvfp4 checkpoint}"
HEAD="${2:-./eagle3-head}"
vllm serve "$CKPT" \
  --max-model-len 65536 \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"$HEAD\",\"num_speculative_tokens\":5}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser "${GEMMA4_PARSER:-gemma4}"
# CUDA graphs on (default). Consider thinking-off for the fast local agent.
# Free extra: n-gram / prompt-lookup speculation for copy-heavy patch edits.
