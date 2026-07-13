#!/usr/bin/env bash
set -uo pipefail

HOST="${HOST:-ironbccllm}"
ROOT="${ROOT:-/home/ironbcc/projects/gemma4-31B-Coder}"
OLD_PIDFILE="${OLD_PIDFILE:-logs/unsloth_agentic_24k_full.pid}"

DATA="${DATA:-data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair}"
INIT_ADAPTER="${INIT_ADAPTER:-adapters/unsloth_agentic_14336_gpu1_repair_conc_s40}"
OUT="${OUT:-adapters/unsloth_agentic_filtered_repair_s20_14336}"
RUN_NAME="${RUN_NAME:-$(basename "$OUT")}"
LOG="${LOG:-logs/${RUN_NAME}.log}"
PIDFILE="${PIDFILE:-logs/${RUN_NAME}.pid}"

MAX_SEQ="${MAX_SEQ:-14336}"
MAX_STEPS="${MAX_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
GPU_FREE_MB="${GPU_FREE_MB:-5000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

stamp() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

ssh_ok() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" 'true' >/dev/null 2>&1
}

echo "[$(stamp)] recovery watcher host=$HOST out=$OUT init=$INIT_ADAPTER"

while ! ssh_ok; do
  echo "[$(stamp)] ssh unavailable; retrying in ${SLEEP_SECONDS}s"
  sleep "$SLEEP_SECONDS"
done

echo "[$(stamp)] ssh available; cleaning old trainer if present"
ssh "$HOST" "cd '$ROOT' && old_pid=\$(cat '$OLD_PIDFILE' 2>/dev/null || true); if [[ \"\$old_pid\" =~ ^[0-9]+$ ]]; then kill -TERM \"\$old_pid\" 2>/dev/null || true; sleep 5; kill -KILL \"\$old_pid\" 2>/dev/null || true; fi"

while true; do
  used="$(ssh "$HOST" "nvidia-smi --query-gpu=memory.used --id=1 --format=csv,noheader,nounits" 2>/dev/null | tr -dc '0-9' || true)"
  if [[ -n "$used" && "$used" -le "$GPU_FREE_MB" ]]; then
    break
  fi
  echo "[$(stamp)] GPU1 still busy used_mb=${used:-unknown}; retrying in ${SLEEP_SECONDS}s"
  sleep "$SLEEP_SECONDS"
done

echo "[$(stamp)] launching capped continuation"
ssh "$HOST" "cd '$ROOT' && mkdir -p logs adapters '$OUT' && nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-train/bin/python phaseD_sft/train_rust_lora.py --data '$DATA' --out '$OUT' --init-adapter '$INIT_ADAPTER' --max-seq '$MAX_SEQ' --epochs 2 --max-steps '$MAX_STEPS' --bsz 1 --grad-accum 16 --warmup-steps 5 --load-4bit --logging-steps '$LOGGING_STEPS' --save-steps '$SAVE_STEPS' --save-total-limit '$SAVE_TOTAL_LIMIT' > '$LOG' 2>&1 < /dev/null & echo \$! > '$PIDFILE' && cat '$PIDFILE'"
echo "[$(stamp)] launched; log=$LOG pidfile=$PIDFILE"
