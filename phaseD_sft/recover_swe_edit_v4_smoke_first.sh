#!/usr/bin/env bash
set -uo pipefail

HOST="${HOST:-192.168.50.148}"
HOST_KEY_ALIAS="${HOST_KEY_ALIAS:-ironbccllm.tail0cc1d4.ts.net}"
ROOT="${ROOT:-/home/ironbcc/projects/gemma4-31B-Coder}"
REMOTE_USER="${REMOTE_USER:-ironbcc}"

DATA="${DATA:-data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget14336}"
INIT_ADAPTER="${INIT_ADAPTER:-adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10}"
SMOKE_OUT="${SMOKE_OUT:-adapters/unsloth_agentic_swe_edit_trace_v4_cp20_smoke1_14336}"
RUN_NAME="${RUN_NAME:-swe_edit_trace_v4}"
SMOKE_LOG="${SMOKE_LOG:-logs/${RUN_NAME}_smoke1.log}"
WATCHLOG="${WATCHLOG:-logs/${RUN_NAME}_recover.local.log}"
SMOKE_PIDFILE="${SMOKE_PIDFILE:-logs/${RUN_NAME}_smoke1.pid}"
SMOKE_UNIT="${SMOKE_UNIT:-swe-edit-v4-smoke}"

MAX_SEQ="${MAX_SEQ:-14336}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
TRAIN_LR="${TRAIN_LR:-0.0002}"
TRAIN_ATTN_IMPLEMENTATION="${TRAIN_ATTN_IMPLEMENTATION:-flex_attention}"
TRAIN_DEBUG_HF_FLEX_ROUTING="${TRAIN_DEBUG_HF_FLEX_ROUTING:-}"
TORCH_LOGS="${TORCH_LOGS:-}"
GPU_FREE_MB="${GPU_FREE_MB:-5000}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-7200}"
ALLOW_64G_TRAINING="${ALLOW_64G_TRAINING:-}"
PROD_PORTS="${PROD_PORTS:-8000 8101 8103 8104}"
LAUNCH_EPOCH=0
MODEL_READY_REPORTED=0
MODEL_READY_EPOCH=0
DATA_READY_REPORTED=0
LAST_STEP=0
LAST_STEP_EPOCH=0

if [[ "$ALLOW_64G_TRAINING" == "I_ACCEPT_BOUNDED_SWAP" ]]; then
  SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
  MIN_MEM_AVAILABLE_MB="${MIN_MEM_AVAILABLE_MB:-36000}"
  ABORT_MEM_AVAILABLE_MB="${ABORT_MEM_AVAILABLE_MB:-2048}"
  TRAIN_MEMORY_HIGH="${TRAIN_MEMORY_HIGH:-36G}"
  TRAIN_MEMORY_MAX="${TRAIN_MEMORY_MAX:-45G}"
  TRAIN_SWAP_MAX="${TRAIN_SWAP_MAX:-16G}"
else
  SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
  MIN_MEM_AVAILABLE_MB="${MIN_MEM_AVAILABLE_MB:-60000}"
  ABORT_MEM_AVAILABLE_MB="${ABORT_MEM_AVAILABLE_MB:-2048}"
  TRAIN_MEMORY_HIGH="${TRAIN_MEMORY_HIGH:-40G}"
  TRAIN_MEMORY_MAX="${TRAIN_MEMORY_MAX:-48G}"
  TRAIN_SWAP_MAX="${TRAIN_SWAP_MAX:-32G}"
fi

stamp() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(stamp)] $*" | tee -a "$WATCHLOG"
}

ssh_ok() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o HostKeyAlias="$HOST_KEY_ALIAS" "$HOST" 'true' >/dev/null 2>&1
}

remote() {
  ssh -o HostKeyAlias="$HOST_KEY_ALIAS" "$HOST" "cd '$ROOT' && $*"
}

wait_for_ssh() {
  while ! ssh_ok; do
    log "ssh unavailable; retrying in ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
  done
}

wait_for_gpu1() {
  local used
  while true; do
    used="$(remote "nvidia-smi --query-gpu=memory.used --id=1 --format=csv,noheader,nounits" 2>/dev/null | tr -dc '0-9' || true)"
    if [[ -n "$used" && "$used" -le "$GPU_FREE_MB" ]]; then
      log "GPU1 available used_mb=$used"
      return 0
    fi
    log "GPU1 busy used_mb=${used:-unknown}; retrying in ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
  done
}

check_host_ram_headroom() {
  local available_mb
  available_mb="$(remote "awk '/MemAvailable:/ {print int(\$2/1024)}' /proc/meminfo" 2>/dev/null | tr -dc '0-9' || true)"
  if [[ -z "$available_mb" ]]; then
    log "could not read MemAvailable; refusing to launch"
    return 1
  fi
  if (( available_mb < MIN_MEM_AVAILABLE_MB )); then
    log "insufficient system RAM headroom available_mb=$available_mb required_mb=$MIN_MEM_AVAILABLE_MB"
    return 1
  fi
  log "system RAM headroom ok available_mb=$available_mb"
}

check_low_ram_swap_capacity() {
  local swap_total_mb
  [[ "$ALLOW_64G_TRAINING" == "I_ACCEPT_BOUNDED_SWAP" ]] || return 0
  swap_total_mb="$(remote "awk '/SwapTotal:/ {print int(\$2/1024)}' /proc/meminfo" 2>/dev/null | tr -dc '0-9' || true)"
  if [[ -z "$swap_total_mb" ]] || (( swap_total_mb < 65536 )); then
    log "64G training mode requires at least 65536 MiB swap; found_mb=${swap_total_mb:-unknown}"
    return 1
  fi
  log "64G bounded-swap mode enabled swap_total_mb=$swap_total_mb"
}

check_user_linger() {
  local linger
  linger="$(remote "loginctl show-user '$REMOTE_USER' -p Linger --value" 2>/dev/null | tr -d '\r' || true)"
  if [[ "$linger" != "yes" ]]; then
    log "user lingering is not enabled for $REMOTE_USER; refusing transient training launch"
    return 1
  fi
  log "user lingering enabled for $REMOTE_USER"
}

check_prod_health() {
  local port
  if ! remote "systemctl is-active --quiet vllm.service"; then
    log "production vllm.service is not active"
    return 1
  fi
  for port in $PROD_PORTS; do
    if ! remote "curl -fsS --max-time 5 'http://127.0.0.1:${port}/health' >/dev/null"; then
      log "production health check failed port=$port"
      return 1
    fi
  done
  log "production health ok ports=$PROD_PORTS"
}

stop_smoke_unit() {
  remote "systemctl --user stop '${SMOKE_UNIT}.service' 2>/dev/null || true"
}

launch_smoke() {
  local absolute_log="$ROOT/$SMOKE_LOG"
  remote "mkdir -p logs adapters '$SMOKE_OUT'; \
    systemctl --user stop '${SMOKE_UNIT}.service' 2>/dev/null || true; \
    systemctl --user reset-failed '${SMOKE_UNIT}.service' 2>/dev/null || true; \
    if [[ -f '$SMOKE_LOG' ]]; then mv '$SMOKE_LOG' '${SMOKE_LOG}.previous-$(date -u +%Y%m%dT%H%M%SZ)'; fi; \
    routing_arg=''; if [[ '$TRAIN_DEBUG_HF_FLEX_ROUTING' == '1' ]]; then routing_arg='--debug-hf-flex-routing'; fi; \
    torch_logs_arg=''; if [[ -n '$TORCH_LOGS' ]]; then torch_logs_arg="--setenv=TORCH_LOGS=$TORCH_LOGS"; fi; \
    systemd-run --user --unit='$SMOKE_UNIT' --remain-after-exit \
      --property=MemoryAccounting=yes \
      --property='MemoryHigh=$TRAIN_MEMORY_HIGH' \
      --property='MemoryMax=$TRAIN_MEMORY_MAX' \
      --property='MemorySwapMax=$TRAIN_SWAP_MAX' \
      --property=OOMScoreAdjust=500 \
      --property='StandardOutput=append:$absolute_log' \
      --property='StandardError=append:$absolute_log' \
      --working-directory='$ROOT' \
      --setenv=CUDA_VISIBLE_DEVICES=1 \
      --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      \$torch_logs_arg \
      '$ROOT/.venv-train/bin/python' phaseD_sft/train_rust_lora.py \
        --data '$DATA' \
        --out '$SMOKE_OUT' \
        --init-adapter '$INIT_ADAPTER' \
        --max-seq '$MAX_SEQ' \
        --epochs 2 \
        --max-steps '$TRAIN_STEPS' \
        --bsz 1 \
        --grad-accum 16 \
        --warmup-steps '$WARMUP_STEPS' \
        --lr '$TRAIN_LR' \
        --attn-implementation '$TRAIN_ATTN_IMPLEMENTATION' \
        \$routing_arg \
        --load-4bit \
        --logging-steps 1 \
        --save-steps '$TRAIN_STEPS' \
        --save-total-limit 10; \
    pid=\$(systemctl --user show '${SMOKE_UNIT}.service' -p MainPID --value); \
    echo \"\$pid\" > '$SMOKE_PIDFILE'; \
    systemctl --user show '${SMOKE_UNIT}.service' -p MainPID -p MemoryHigh -p MemoryMax -p MemorySwapMax"
}

read_pressure() {
  remote "available=\$(awk '/MemAvailable:/ {print int(\$2/1024)}' /proc/meminfo); \
    swap_total=\$(awk '/SwapTotal:/ {print int(\$2/1024)}' /proc/meminfo); \
    swap_free=\$(awk '/SwapFree:/ {print int(\$2/1024)}' /proc/meminfo); \
    unit_current=\$(systemctl --user show '${SMOKE_UNIT}.service' -p MemoryCurrent --value 2>/dev/null || echo 0); \
    unit_peak=\$(systemctl --user show '${SMOKE_UNIT}.service' -p MemoryPeak --value 2>/dev/null || echo 0); \
    gpu1=\$(nvidia-smi --query-gpu=memory.used --id=1 --format=csv,noheader,nounits | tr -dc '0-9'); \
    echo \"available_mb=\$available swap_used_mb=\$((swap_total-swap_free)) unit_current_bytes=\$unit_current unit_peak_bytes=\$unit_peak gpu1_used_mb=\$gpu1\""
}

pressure_is_safe() {
  local pressure available
  pressure="$(read_pressure)" || return 1
  log "$pressure"
  available="$(sed -n 's/.*available_mb=\([0-9][0-9]*\).*/\1/p' <<<"$pressure")"
  [[ -n "$available" ]] || return 1
  if (( available < ABORT_MEM_AVAILABLE_MB )); then
    log "aborting smoke: available_mb=$available below $ABORT_MEM_AVAILABLE_MB"
    return 1
  fi
}

report_timing() {
  local now model_ready data_ready step step_delta elapsed
  (( LAUNCH_EPOCH > 0 )) || return 0
  now="$(date +%s)"

  if (( MODEL_READY_REPORTED == 0 )); then
    model_ready="$(remote "tr '\\r' '\\n' < '$SMOKE_LOG' 2>/dev/null | grep -F -m1 '[progress] event=model_ready ' | sed -nE 's/.*epoch_seconds=([0-9]+).*/\\1/p'" | tr -dc '0-9')"
    if [[ "$model_ready" =~ ^[0-9]+$ ]]; then
      MODEL_READY_REPORTED=1
      MODEL_READY_EPOCH="$model_ready"
      log "model_load_seconds=$((MODEL_READY_EPOCH - LAUNCH_EPOCH))"
    fi
  fi

  if (( DATA_READY_REPORTED == 0 )); then
    data_ready="$(remote "tr '\\r' '\\n' < '$SMOKE_LOG' 2>/dev/null | grep -F -m1 '[progress] event=data_ready ' | sed -nE 's/.*epoch_seconds=([0-9]+).*/\\1/p'" | tr -dc '0-9')"
    if [[ "$data_ready" =~ ^[0-9]+$ ]]; then
      DATA_READY_REPORTED=1
      if (( MODEL_READY_EPOCH > 0 )); then
        log "data_prepare_seconds=$((data_ready - MODEL_READY_EPOCH))"
      else
        log "data_prepare_seconds=unknown_model_ready_not_observed"
      fi
    fi
  fi

  step="$(remote "tr '\\r' '\\n' < '$SMOKE_LOG' 2>/dev/null | grep -oE '\\[progress\\] event=optimizer_step optimizer_step=[0-9]+/${TRAIN_STEPS}' | tail -1 | sed -E 's/.*optimizer_step=([0-9]+).*/\\1/'" 2>/dev/null | tr -dc '0-9')"
  if [[ "$step" =~ ^[0-9]+$ ]] && (( step > LAST_STEP )); then
    step_delta=$((step - LAST_STEP))
    if (( LAST_STEP == 0 )); then
      elapsed=$((now - LAUNCH_EPOCH))
      log "first_step_elapsed_seconds=$elapsed completed_step=$step"
    else
      elapsed=$((now - LAST_STEP_EPOCH))
      log "per_step_wall_seconds=$((elapsed / step_delta)) completed_step=$step interval_steps=$step_delta"
    fi
    LAST_STEP="$step"
    LAST_STEP_EPOCH="$now"
  fi
}

report_compile_status() {
  local stats
  stats="$(remote "main=\$(systemctl --user show '${SMOKE_UNIT}.service' -p MainPID --value 2>/dev/null || echo 0); \
    coordinator=\$(ps -eo ppid=,pid=,args= | awk -v main=\"\$main\" '\$1 == main && /torch\\._inductor\\.compile_worker/ {print \$2; exit}'); \
    if [[ -n \"\$coordinator\" ]]; then \
      ps -eo ppid=,pcpu= | awk -v parent=\"\$coordinator\" '\$1 == parent {workers += 1; cpu += \$2} END {printf \"inductor_compile_workers=%d inductor_compile_cpu_percent=%.1f\", workers, cpu}'; \
    else \
      printf 'inductor_compile_workers=0 inductor_compile_cpu_percent=0.0'; \
    fi" 2>/dev/null | tr -d '\r' || true)"
  [[ -n "$stats" ]] && log "$stats"
}

report_completed_timing() {
  local now runtime per_step
  (( LAUNCH_EPOCH > 0 )) || return 0
  now="$(date +%s)"
  log "run_wall_seconds=$((now - LAUNCH_EPOCH)) completed_steps=$TRAIN_STEPS"
  runtime="$(remote "tr '\\r' '\\n' < '$SMOKE_LOG' 2>/dev/null | sed -n \"s/.*'train_runtime': '\\([^']*\\)'.*/\\1/p\" | tail -1" 2>/dev/null | tr -cd '0-9.')"
  if [[ "$runtime" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    per_step="$(awk -v runtime="$runtime" -v steps="$TRAIN_STEPS" 'BEGIN {printf "%.2f", runtime / steps}')"
    log "trainer_runtime_seconds=$runtime per_step_wall_seconds=$per_step completed_steps=$TRAIN_STEPS"
  else
    log "trainer_runtime_seconds=unavailable per_step_wall_seconds=unavailable completed_steps=$TRAIN_STEPS"
  fi
}

wait_for_smoke_checkpoint() {
  local start now elapsed state substate result status
  start="$(date +%s)"
  while true; do
    state="$(remote "systemctl --user show '${SMOKE_UNIT}.service' -p ActiveState --value 2>/dev/null || echo missing" | tr -d '\r')"
    substate="$(remote "systemctl --user show '${SMOKE_UNIT}.service' -p SubState --value 2>/dev/null || echo missing" | tr -d '\r')"
    if ! check_prod_health || ! pressure_is_safe; then
      log "safety gate failed; stopping only ${SMOKE_UNIT}.service"
      stop_smoke_unit
      return 1
    fi
    if [[ "$state" == "active" && "$substate" == "exited" ]]; then
      report_timing
      result="$(remote "systemctl --user show '${SMOKE_UNIT}.service' -p Result --value 2>/dev/null || echo missing" | tr -d '\r')"
      status="$(remote "systemctl --user show '${SMOKE_UNIT}.service' -p ExecMainStatus --value 2>/dev/null || echo missing" | tr -d '\r')"
      if remote "[[ -f '$SMOKE_OUT/checkpoint-${TRAIN_STEPS}/adapter_model.safetensors' ]]" && [[ "$result" == "success" && "$status" == "0" ]]; then
        report_completed_timing
        log "checkpoint exists and retained unit completed successfully: $SMOKE_OUT/checkpoint-${TRAIN_STEPS}"
        return 0
      fi
      log "smoke command exited without a valid checkpoint result=$result status=$status"
      remote "tail -120 '$SMOKE_LOG' 2>/dev/null || true"
      return 1
    fi
    if [[ "$state" != "active" ]]; then
      result="$(remote "systemctl --user show '${SMOKE_UNIT}.service' -p Result --value 2>/dev/null || echo missing" | tr -d '\r')"
      status="$(remote "systemctl --user show '${SMOKE_UNIT}.service' -p ExecMainStatus --value 2>/dev/null || echo missing" | tr -d '\r')"
      if remote "[[ -f '$SMOKE_OUT/checkpoint-${TRAIN_STEPS}/adapter_model.safetensors' ]]" && [[ "$result" == "success" && "$status" == "0" ]]; then
        log "checkpoint exists and unit completed successfully: $SMOKE_OUT/checkpoint-${TRAIN_STEPS}"
        return 0
      fi
      log "smoke unit exited before a valid checkpoint state=$state result=$result status=$status"
      remote "tail -120 '$SMOKE_LOG' 2>/dev/null || true"
      return 1
    fi
    now="$(date +%s)"
    elapsed="$((now - start))"
    if (( elapsed > SMOKE_TIMEOUT_SECONDS )); then
      log "smoke timed out after ${elapsed}s; stopping only ${SMOKE_UNIT}.service"
      stop_smoke_unit
      return 1
    fi
    report_timing
    report_compile_status
    log "smoke still running unit=${SMOKE_UNIT}.service elapsed=${elapsed}s"
    sleep "$SLEEP_SECONDS"
  done
}

mkdir -p "$(dirname "$WATCHLOG")"
log "protected recovery host=$HOST data=$DATA init=$INIT_ADAPTER"
wait_for_ssh
log "ssh available"
wait_for_gpu1
check_host_ram_headroom || exit 1
check_low_ram_swap_capacity || exit 1
check_prod_health || exit 1
check_user_linger || exit 1

if [[ "$TRAIN_STEPS" == "1" ]]; then
  log "launching protected 1-step v4 smoke"
else
  log "launching protected ${TRAIN_STEPS}-step v4 bounded train"
fi
launch_smoke | tee -a "$WATCHLOG"
LAUNCH_EPOCH="$(date +%s)"
LAST_STEP_EPOCH="$LAUNCH_EPOCH"
log "timing watch started launch_epoch=$LAUNCH_EPOCH target_steps=$TRAIN_STEPS"

if ! wait_for_smoke_checkpoint; then
  log "protected smoke failed; no longer training will be launched"
  exit 1
fi

check_prod_health || exit 1
read_pressure | tee -a "$WATCHLOG"
log "smoke gate passed; inspect behavior and memory peak before any 10-step continuation"
