#!/usr/bin/env bash
# Phase-0 baseline: run Gemma 4 31B and Qwen3.6-27B under the IDENTICAL mini-SWE-agent harness.
# This produces the same-stack Qwen number that is the real target to beat.
# All flags illustrative — confirm against your vLLM / mini-swe-agent versions.
set -euo pipefail

DATASET="${DATASET:-princeton-nlp/SWE-bench_Lite}"   # Lite for fast dev; switch to _Verified for the gate
SPLIT="${SPLIT:-test}"
OUT="${OUT:-runs/baseline_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT"

run_one () {
  local name="$1" model="$2" parser="$3"
  echo "=== serving $name ($model) ==="
  # BF16 for the capability comparison (PLAN §2)
  vllm serve "$model" \
    --max-model-len 65536 --dtype bfloat16 \
    --enable-prefix-caching --enable-chunked-prefill --max-num-seqs 64 \
    --enable-auto-tool-choice --tool-call-parser "$parser" &
  local pid=$!
  sleep 90   # wait for server; replace with a real /health poll

  echo "=== mini-swe-agent over $DATASET on $name ==="
  mini-swe-agent \
    --model "openai/$model" --base-url http://localhost:8000/v1 \
    --dataset "$DATASET" --split "$SPLIT" \
    --output "$OUT/$name.jsonl"          # confirm exact CLI flags

  kill "$pid" 2>/dev/null || true
  sleep 10
}

run_one gemma4-31b google/gemma-4-31B-it          "${GEMMA4_PARSER:-gemma4}"
run_one qwen36-27b Qwen/Qwen3.6-27B               "${QWEN_PARSER:-hermes}"

echo "=== score patches in official Docker harness ==="
for m in gemma4-31b qwen36-27b; do
  python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" --predictions_path "$OUT/$m.jsonl" \
    --run_id "$m" --report_dir "$OUT/report_$m"   # confirm flags for your swebench version
done

echo "Baseline done → $OUT  (compare resolve rates: gemma vs qwen, same stack)"
