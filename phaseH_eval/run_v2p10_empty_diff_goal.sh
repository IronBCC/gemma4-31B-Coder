#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-empty_retry}"
NAME="${NAME:-teacher_sft_v2p10}"
MODEL="${MODEL:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full}"
IDS="${IDS:-data/fixed150_v2p10_empty_retry_diff1_ids.json}"
RUN_ID="${RUN_ID:-fixed150_v2p10_empty_retry_diff1}"
EXPECTED_IDS="${EXPECTED_IDS:-18}"
SEED="${SEED:-1}"
SOURCE_IDS="${SOURCE_IDS:-data/teacher_sft_fixed150_ids.json}"
SOURCE_RUN="${SOURCE_RUN:-runs/fixed150_v2p10}"
PLAN="${PLAN:-runs/fixed150_v2p10_empty_retry_diff1_plan.json}"
COMPOSITE="${COMPOSITE:-runs/fixed150_v2p10_empty_diff1_composite.json}"
PREDICTIONS="${PREDICTIONS:-runs/fixed150_v2p10_empty_diff1_preds.json}"
PORT="${PORT:-8013}"
GPU_INDEX="${GPU_INDEX:-1}"
REUSE_SERVE="${REUSE_SERVE:-0}"
REUSE_SERVE_PID="${REUSE_SERVE_PID:-}"
VLLM="${VLLM:-/home/ironbcc/projects/llm/vllm/vllm_env/bin/vllm}"
EVAL_PY="$ROOT/.venv-eval/bin/python"
LOG="${LOG:-/tmp/v2p10_empty_diff_goal.log}"
SERVE_LOG="${SERVE_LOG:-/tmp/serve_${NAME}_empty_diff.log}"
SERVE_STOP_WAIT_LOOPS="${SERVE_STOP_WAIT_LOOPS:-60}"
SERVE_STOP_WAIT_SECONDS="${SERVE_STOP_WAIT_SECONDS:-1}"
MEMORY_ADMISSION_GIB="${MEMORY_ADMISSION_GIB:-40}"
MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"
RAM_WATCHDOG_PY="${RAM_WATCHDOG_PY:-$ROOT/.venv-train/bin/python}"
serve_pid=""
memory_watchdog_pid=""
export PATH="${VLLM%/*}:$PATH"

log() {
  echo "[$(date '+%H:%M:%S %Z')] $*" | tee -a "$LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

wait_for_memory_admission() {
  local available_kib
  while true; do
    available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available_kib >= MEMORY_ADMISSION_GIB * 1024 * 1024 )); then
      log "memory admission passed available_kib=$available_kib"
      return
    fi
    log "waiting for ${MEMORY_ADMISSION_GIB}GiB MemAvailable current_kib=$available_kib"
    sleep 60
  done
}

start_memory_watchdog() {
  [[ -x "$RAM_WATCHDOG_PY" ]] ||
    halt "RAM watchdog Python is missing"
  [[ "$serve_pid" =~ ^[1-9][0-9]*$ ]] ||
    halt "RAM watchdog requires an exact owned serve PID"
  "$RAM_WATCHDOG_PY" phaseD_sft/ram_watchdog.py \
    --pid "$serve_pid" \
    --min-available-gib "$MEMORY_WATCHDOG_GIB" \
    --interval-seconds 5 \
    --log-path "/tmp/ram_watchdog_${RUN_ID}.log" \
    >> "$LOG" 2>&1 &
  memory_watchdog_pid="$!"
  log "RAM watchdog started pid=$memory_watchdog_pid target=$serve_pid"
}

finish_memory_watchdog() {
  local pid status
  pid="$memory_watchdog_pid"
  [[ -n "$pid" ]] || return 0
  if wait "$pid"; then
    status=0
  else
    status=$?
  fi
  memory_watchdog_pid=""
  (( status == 0 ))
}

stop_owned_serve() {
  local pid watchdog_status=0
  pid="$serve_pid"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    for _ in $(seq 1 "$SERVE_STOP_WAIT_LOOPS"); do
      kill -0 "$pid" 2>/dev/null || break
      sleep "$SERVE_STOP_WAIT_SECONDS"
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  wait "$pid" 2>/dev/null || true
  serve_pid=""
  if [[ -n "${memory_watchdog_pid:-}" ]]; then
    finish_memory_watchdog || watchdog_status=$?
  fi
  (( watchdog_status == 0 ))
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if ! stop_owned_serve; then
    log "HALT: owned serve PID $serve_pid did not stop"
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

ensure_eval_manifest() {
  "$EVAL_PY" - "$NAME" "$MODEL" "$IDS" "$RUN_ID" "$SEED" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

name, model_text, ids_text, run_id, seed_text = sys.argv[1:]
seed = int(seed_text)
model = Path(model_text)
ids = Path(ids_text)
instance_ids = json.loads(ids.read_text())

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

index = model / "model.safetensors.index.json"
index_value = json.loads(index.read_text())
shards = sorted({model / value for value in index_value["weight_map"].values()})
artifacts = [
    {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}
    for path in shards
]
expected = {
    "schema_version": 1,
    "run_id": run_id,
    "served_name": name,
    "model_path": str(model),
    "model_config_sha256": sha(model / "config.json"),
    "model_index_sha256": sha(index),
    "model_artifacts": artifacts,
    "ids_path": str(ids),
    "ids_sha256": sha(ids),
    "instances": len(instance_ids),
    "config": "swebench_edit_first_selfretry_s120.yaml",
    "environment_class": "docker_selfretry.DockerSelfRetryEnv",
    "temperature": 0.7,
    "seed": seed,
    "step_limit": 120,
    "generation_workers": 16,
    "scorer_workers": 16,
    "batch": 20,
    "pull_workers": 5,
}
root = Path("runs") / run_id
root.mkdir(parents=True, exist_ok=True)
path = root / "eval_manifest.json"
payload = json.dumps(expected, indent=2) + "\n"
if path.exists():
    if path.read_text() != payload:
        raise SystemExit(f"{path}: manifest mismatch")
else:
    temporary = root / f".eval_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(payload)
    os.link(temporary, path)
    temporary.unlink()
PY
}

case "$MODE" in
  empty_retry|primary) ;;
  *) log "invalid MODE=$MODE"; exit 2 ;;
esac
actual_ids="$(jq -er 'if type == "array" then length else error("not array") end' "$IDS" 2>/dev/null || true)"
[[ "$actual_ids" == "$EXPECTED_IDS" ]] ||
  halt "ID artifact count mismatch: expected=$EXPECTED_IDS actual=${actual_ids:-invalid} path=$IDS"
[[ "$SEED" =~ ^[1-9][0-9]*$ ]] ||
  halt "SEED must be a positive integer: $SEED"
[[ "$PORT" =~ ^[1-9][0-9]*$ && "$PORT" -le 65535 ]] ||
  halt "PORT must be in 1..65535: $PORT"
[[ "$GPU_INDEX" == "1" ]] ||
  halt "GPU_INDEX is fixed to GPU1; GPU0 is unavailable"
[[ "$MEMORY_ADMISSION_GIB" =~ ^[1-9][0-9]*$ ]] ||
  halt "MEMORY_ADMISSION_GIB must be a positive integer"
[[ "$MEMORY_WATCHDOG_GIB" =~ ^[1-9][0-9]*$ ]] ||
  halt "MEMORY_WATCHDOG_GIB must be a positive integer"
[[ "$REUSE_SERVE" == "0" || "$REUSE_SERVE" == "1" ]] ||
  halt "REUSE_SERVE must be 0 or 1: $REUSE_SERVE"
[[ -z "$REUSE_SERVE_PID" || "$REUSE_SERVE_PID" =~ ^[1-9][0-9]*$ ]] ||
  halt "REUSE_SERVE_PID must be empty or a positive integer: $REUSE_SERVE_PID"
[[ "$REUSE_SERVE" == "1" || -z "$REUSE_SERVE_PID" ]] ||
  halt "REUSE_SERVE_PID requires REUSE_SERVE=1"
[[ -d "$MODEL" ]] || halt "model directory is missing: $MODEL"
[[ -x "$VLLM" ]] || halt "vLLM executable is missing: $VLLM"
[[ -x "$EVAL_PY" ]] || halt "evaluation Python is missing: $EVAL_PY"
if [[ "$MODE" == "empty_retry" ]]; then
  [[ ! -e "$COMPOSITE" ]] ||
    halt "refusing to overwrite composite: $COMPOSITE"
  [[ ! -e "$PREDICTIONS" ]] ||
    halt "refusing to overwrite predictions: $PREDICTIONS"
fi
gpu_uuid="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid \
  --format=csv,noheader | awk '{$1=$1; print}')"
[[ -n "$gpu_uuid" ]] || halt "GPU_INDEX does not resolve to a GPU: $GPU_INDEX"
gpu_pids="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
  --format=csv,noheader,nounits |
  awk '{$1=$1; if ($1 ~ /^[0-9]+$/) print $1}')"
ensure_eval_manifest

if [[ "$REUSE_SERVE" == "0" ]]; then
  [[ -z "$gpu_pids" ]] ||
    halt "GPU${GPU_INDEX} has foreign compute PIDs: $gpu_pids"
  wait_for_memory_admission
  log "starting $NAME serve on GPU${GPU_INDEX}:$PORT"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" env -u HF_TOKEN "$VLLM" serve "$MODEL" \
    --dtype bfloat16 --served-model-name "$NAME" --port "$PORT" \
    --max-model-len 250000 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --max-num-seqs 64 \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template phaseH_eval/tool_chat_template_gemma4_thinkopen_v2.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    > "$SERVE_LOG" 2>&1 &
  serve_pid="$!"
  start_memory_watchdog
fi

for _ in $(seq 1 180); do
  if [[ -n "$serve_pid" ]] && ! kill -0 "$serve_pid" 2>/dev/null; then
    wait "$serve_pid" || status=$?
    log "serve exited before readiness status=${status:-1}"
    exit "${status:-1}"
  fi
  served="$(
    curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
      "$EVAL_PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' \
      2>/dev/null || true
  )"
  [[ "$served" == "$NAME" ]] && break
  sleep 10
done
[[ "$served" == "$NAME" ]] ||
  halt "serve readiness timed out or model identity differed: ${served:-none}"
if [[ "$REUSE_SERVE" == "1" ]]; then
  [[ -n "$gpu_pids" ]] ||
    halt "reused serve has no compute PID on GPU${GPU_INDEX}"
  matched_pid=""
  if [[ -n "$REUSE_SERVE_PID" ]]; then
    args="$(ps -p "$REUSE_SERVE_PID" -o args= 2>/dev/null || true)"
    if [[ "$args" == *"$MODEL"* &&
      "$args" == *"--served-model-name $NAME"* &&
      "$args" == *"--port $PORT"* ]]; then
      matched_pid="$REUSE_SERVE_PID"
    fi
  fi
  while IFS= read -r pid; do
    [[ -z "$matched_pid" ]] || break
    candidate_pid="$pid"
    for _ in 1 2 3 4; do
      [[ "$candidate_pid" =~ ^[1-9][0-9]*$ ]] || break
      args="$(ps -p "$candidate_pid" -o args= 2>/dev/null || true)"
      if [[ "$args" == *"$MODEL"* &&
        "$args" == *"--served-model-name $NAME"* &&
        "$args" == *"--port $PORT"* ]]; then
        matched_pid="$candidate_pid"
        break 2
      fi
      candidate_pid="$(
        ps -p "$candidate_pid" -o ppid= 2>/dev/null |
          tr -d '[:space:]'
      )"
    done
  done <<<"$gpu_pids"
  [[ -n "$matched_pid" ]] ||
    halt "reused serve process identity does not match model/name/port"
  log "reusing $NAME serve on GPU${GPU_INDEX}:$PORT pid=$matched_pid"
else
  log "serve ready pid=$serve_pid"
fi

"$EVAL_PY" phaseH_eval/retest_empties.py \
  --name "$NAME" --port "$PORT" --ids "$IDS" --runid "$RUN_ID" \
  --batch 20 --workers 16 --pull-workers 5 --seed "$SEED" \
  >> "$LOG" 2>&1

"$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py validate-fixed \
  --name "$NAME" --ids "$IDS" --run-root "runs/$RUN_ID" \
  --out "runs/$RUN_ID/acceptance.json"

stop_owned_serve ||
  halt "owned serve PID $serve_pid did not stop"

if [[ "$MODE" == "primary" ]]; then
  log "complete primary_run=runs/$RUN_ID"
  exit 0
fi

"$EVAL_PY" phaseH_eval/empty_retry_composite.py combine \
  --source-ids "$SOURCE_IDS" --source-run-root "$SOURCE_RUN" \
  --retry-ids "$IDS" --retry-run-root "runs/$RUN_ID" \
  --plan "$PLAN" --out "$COMPOSITE" --preds-out "$PREDICTIONS"

source_expected="$(jq -er 'if type == "array" then length else error("not array") end' "$SOURCE_IDS" 2>/dev/null || true)"
prediction_count="$(jq -er 'if type == "object" then length else error("not object") end' "$PREDICTIONS" 2>/dev/null || true)"
[[ -n "$source_expected" && "$prediction_count" == "$source_expected" ]] ||
  halt "composite prediction count mismatch: expected=${source_expected:-invalid} actual=${prediction_count:-invalid}"
log "complete composite=$COMPOSITE predictions=$PREDICTIONS"
