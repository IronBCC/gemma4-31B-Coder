#!/usr/bin/env bash
# Throughput profile — trajectory collection / best-of-n / RL rollouts (usually NO spec decoding).
# NOTE: all vLLM flags illustrative — confirm exact names against your vLLM version.
set -euo pipefail
CKPT="${1:?path to nvfp4 checkpoint}"
vllm serve "$CKPT" \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 256 \
  --enable-auto-tool-choice --tool-call-parser "${GEMMA4_PARSER:-gemma4}"
# Run ONE replica per GPU (data-parallel, no TP) behind a router → ~2x aggregate.
