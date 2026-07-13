#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/ironbcc/projects/gemma4-31B-Coder}"
cd "$ROOT" || exit 1

OUT="${OUT:-adapters/unsloth_agentic_24k}"
RUN_NAME="${RUN_NAME:-$(basename "$OUT")}"
LOG="${LOG:-logs/${RUN_NAME}_full.log}"
WATCHLOG="${WATCHLOG:-logs/${RUN_NAME}_watchdog.log}"
PIDFILE="${PIDFILE:-logs/${RUN_NAME}_full.pid}"
MAX_SEQ="${MAX_SEQ:-14336}"
DATA="${DATA:-data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair}"
SAVE_STEPS="${SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
INIT_ADAPTER="${INIT_ADAPTER:-}"
mkdir -p logs adapters "$OUT"

stamp() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

is_alive() {
  local pid
  [[ -s "$PIDFILE" ]] || return 1
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o args= | grep -q "train_rust_lora.py.*unsloth_agentic_24k"
}

is_done() {
  [[ -f "$OUT/adapter_model.safetensors" ]] || return 1
  grep -q "\[done\] adapter saved -> $OUT" "$LOG" 2>/dev/null
}

start_train() {
  local args=(
    phaseD_sft/train_rust_lora.py
    --data "$DATA"
    --out "$OUT"
    --max-seq "$MAX_SEQ"
    --epochs 2
    --bsz 1
    --grad-accum 16
    --warmup-steps 20
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --load-4bit
    --resume
  )
  if [[ -n "$INIT_ADAPTER" ]]; then
    args+=(--init-adapter "$INIT_ADAPTER")
  fi

  echo "[$(stamp)] starting/resuming trainer out=$OUT max_seq=$MAX_SEQ data=$DATA init_adapter=${INIT_ADAPTER:-none}" >> "$WATCHLOG"
  nohup env \
    CUDA_VISIBLE_DEVICES=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv-train/bin/python "${args[@]}" \
    >> "$LOG" 2>&1 < /dev/null &
  echo "$!" > "$PIDFILE"
  echo "[$(stamp)] trainer pid $(cat "$PIDFILE")" >> "$WATCHLOG"
}

echo "[$(stamp)] watchdog started" >> "$WATCHLOG"

while true; do
  if is_done; then
    echo "[$(stamp)] training complete" >> "$WATCHLOG"
    exit 0
  fi

  if ! is_alive; then
    echo "[$(stamp)] trainer not alive" >> "$WATCHLOG"
    start_train
  fi

  sleep "${WATCHDOG_INTERVAL_SECONDS:-300}"
done
