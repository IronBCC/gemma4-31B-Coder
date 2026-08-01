#!/usr/bin/env bash
# Continue completed v2.11r3 reasoning SFT through recovery SFT and behavior KTO.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE="${BASE:-/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it}"
TRAIN_UNIT="${TRAIN_UNIT:-v2p11r3-reasoned-train-gpu1.service}"
STAGE_A_ADAPTER="${STAGE_A_ADAPTER:-adapters/teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16}"
STAGE_A_MARKER="${STAGE_A_MARKER:-runs/v2p11r3_v2p10init_fable_reasoned_training_completion.json}"
RECOVERY_DATA="${RECOVERY_DATA:-data/v2p11_portable_recovery138_targeted}"
BEHAVIOR_DATA="${BEHAVIOR_DATA:-data/v2p11_behavior_kto_v2.jsonl}"
BEHAVIOR_MANIFEST="${BEHAVIOR_MANIFEST:-data/v2p11_behavior_kto_v2_manifest.json}"
EXCLUSIONS="${EXCLUSIONS:-data/swe_all_eval_exclusions_v2.json}"
FULL_IDS="${FULL_IDS:-data/swebench_lite_test_ids.json}"
RECOVERY_ADAPTER="${RECOVERY_ADAPTER:-adapters/teacher_sft_v2p11r3_behavior_recovery_bf16}"
RECOVERY_MERGED="${RECOVERY_MERGED:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_recovery_base}"
KTO_CANARY="${KTO_CANARY:-adapters/teacher_sft_v2p11r3_behavior_kto_canary}"
KTO_ADAPTER="${KTO_ADAPTER:-adapters/teacher_sft_v2p11r3_behavior_kto}"
FINAL_MERGED="${FINAL_MERGED:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full}"
COMPLETION_MARKER="${COMPLETION_MARKER:-runs/v2p11r3_behavior_posttrain_complete.json}"
RECOVERY_MARKER="${RECOVERY_MARKER:-runs/v2p11r3_behavior_recovery_sft_complete.json}"
RECOVERY_INPUT_MARKER="${RECOVERY_INPUT_MARKER:-runs/v2p11r3_behavior_recovery_sft_inputs.json}"
RECOVERY_MERGE_MARKER="${RECOVERY_MERGE_MARKER:-runs/v2p11r3_behavior_recovery_merge_complete.json}"
KTO_CANARY_MARKER="${KTO_CANARY_MARKER:-runs/v2p11r3_behavior_kto_canary_complete.json}"
KTO_CANARY_INPUT_MARKER="${KTO_CANARY_INPUT_MARKER:-runs/v2p11r3_behavior_kto_canary_inputs.json}"
KTO_MARKER="${KTO_MARKER:-runs/v2p11r3_behavior_kto_complete.json}"
KTO_INPUT_MARKER="${KTO_INPUT_MARKER:-runs/v2p11r3_behavior_kto_inputs.json}"
FINAL_MERGE_MARKER="${FINAL_MERGE_MARKER:-runs/v2p11r3_behavior_final_merge_complete.json}"
LOG="${LOG:-/tmp/v2p11r3_behavior_poststage_gpu1.log}"
RECOVERY_LOG="${RECOVERY_LOG:-/tmp/v2p11r3_behavior_recovery_train_gpu1.log}"
KTO_LOG="${KTO_LOG:-/tmp/v2p11r3_behavior_kto_gpu1.log}"
RAM_LOG="${RAM_LOG:-/tmp/v2p11r3_behavior_recovery_ram_watchdog_gpu1.log}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RAM_FLOOR_GIB="${RAM_FLOOR_GIB:-12}"
MAX_TRAINER_RESTARTS="${MAX_TRAINER_RESTARTS:-6}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-86400}"
EXPECTED_RECOVERY_STEPS=105
GPU_INDEX="${GPU_INDEX:-1}"
TRAIN_RESUME_ARGS=()
INTERRUPTED_OUTPUT_STATUS=""

log() {
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $*" |
    tee -a "$LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

[[ "$GPU_INDEX" == "1" ]] || halt "GPU_INDEX must be 1"

gpu_uuid="$(
  nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader |
    awk '{$1=$1; print}'
)"
[[ -n "$gpu_uuid" ]] || halt "could not resolve GPU$GPU_INDEX UUID"

gpu_compute_pids() {
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
    --format=csv,noheader,nounits |
    awk '{$1=$1; if ($1 ~ /^[0-9]+$/) print $1}'
}

wait_for_gpu_idle() {
  local phase="$1"
  local pids
  while true; do
    pids="$(gpu_compute_pids)"
    if [[ -z "$pids" ]]; then
      log "$phase: GPU$GPU_INDEX is idle"
      return
    fi
    log "$phase: waiting for exact GPU PIDs: $(tr '\n' ',' <<<"$pids")"
    sleep "$WAIT_SECONDS"
  done
}

latest_recovery_checkpoint() {
  .venv-train/bin/python - "$RECOVERY_ADAPTER" "$EXPECTED_RECOVERY_STEPS" <<'PY'
import json
import re
import sys
from pathlib import Path

adapter = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
checkpoints = []
for path in adapter.glob("checkpoint-*"):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        assert path.is_dir() and not path.is_symlink(), path
        checkpoints.append((int(match.group(1)), path))
if not checkpoints:
    print("none")
    raise SystemExit(0)
step, checkpoint = max(checkpoints)
state = json.loads((checkpoint / "trainer_state.json").read_text())
assert state["global_step"] == step
assert state["max_steps"] == expected_steps
assert 1 <= step <= expected_steps
required = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
}
assert not (required - {path.name for path in checkpoint.iterdir()})
print(step)
PY
}

archive_interrupted_output() {
  local path="$1" phase="$2" mode="$3" report
  report="$(
    .venv-train/bin/python phaseH_eval/v2p11_poststage_recovery.py \
      --path "$path" --phase "$phase" --mode "$mode"
  )" || halt "$phase interrupted-output recovery failed"
  INTERRUPTED_OUTPUT_STATUS="$(
    .venv-train/bin/python -c \
      'import json,sys; print(json.load(sys.stdin)["status"])' \
      <<<"$report"
  )" || halt "$phase interrupted-output status is invalid"
  case "$INTERRUPTED_OUTPUT_STATUS" in
    absent|archived|checkpointed) ;;
    *) halt "$phase interrupted-output status is unsupported" ;;
  esac
  log "$phase interrupted-output recovery: $report"
}

validate_merge_audit() {
  local audit="$1"
  .venv-train/bin/python - "$audit" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "schema_version": 1,
    "architecture": "Gemma4ForConditionalGeneration",
    "expected_tensors": 1188,
    "actual_tensors": 1188,
    "expected_vision": 356,
    "actual_vision": 356,
    "missing_tensors": [],
    "unexpected_tensors": [],
    "misplaced_tensors": [],
    "nonfinite_tensors": [],
    "complete": True,
}
assert audit == expected, audit
PY
}

publish_or_verify_phase_marker() {
  local marker="$1" phase="$2"
  shift 2
  .venv-train/bin/python phaseH_eval/v2p11_poststage_phase_marker.py \
    --marker "$marker" --phase "$phase" "$@" >/dev/null
}

model_phase_artifacts() {
  local model="$1" audit="$2"
  .venv-train/bin/python - "$model" "$audit" <<'PY'
import json
import sys
from pathlib import Path

model, audit = map(Path, sys.argv[1:])
index = model / "model.safetensors.index.json"
weight_map = json.loads(index.read_text())["weight_map"]
required = [
    audit,
    model / "config.json",
    index,
    model / "tokenizer.json",
    model / "tokenizer_config.json",
    model / "processor_config.json",
    model / "chat_template.jinja",
]
artifacts = required + sorted(
    {model / name for name in weight_map.values()}
)
assert all(path.is_file() for path in artifacts)
for path in artifacts:
    print(path)
PY
}

publish_or_verify_model_phase_marker() {
  local marker="$1" phase="$2" model="$3" audit="$4"
  shift 4
  local artifact artifact_output
  local artifacts=()
  artifact_output="$(model_phase_artifacts "$model" "$audit")" ||
    halt "$phase model artifact enumeration failed"
  while IFS= read -r artifact; do
    [[ -n "$artifact" ]] || continue
    artifacts+=("$artifact")
  done <<<"$artifact_output"
  (( ${#artifacts[@]} > 0 )) ||
    halt "$phase model artifact enumeration was empty"
  publish_or_verify_phase_marker \
    "$marker" "$phase" "${artifacts[@]}" "$@"
}

prepare_bound_phase_inputs() {
  local marker="$1" phase="$2" output="$3"
  shift 3
  if [[ -e "$output" && ! -f "$marker" ]]; then
    halt "$phase output exists without its immutable input marker"
  fi
  publish_or_verify_phase_marker "$marker" "$phase" "$@"
}

validate_recovery_adapter() {
  .venv-train/bin/python - "$RECOVERY_ADAPTER" "$STAGE_A_ADAPTER" <<'PY'
import json
import sys
from pathlib import Path
from safetensors import safe_open

adapter = Path(sys.argv[1])
stage_a = sys.argv[2]
manifest = json.loads((adapter / "run_manifest.json").read_text())
expected = {
    "data": "data/v2p11_portable_recovery138_targeted",
    "data_len": 138,
    "init_adapter": stage_a,
    "rank": 32,
    "alpha": 32,
    "lr": 5e-6,
    "epochs": 3.0,
    "bsz": 1,
    "grad_accum": 4,
    "max_seq": 32768,
    "warmup_steps": 4,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}
for key, value in expected.items():
    assert manifest[key] == value, (key, manifest[key], value)
with safe_open(adapter / "adapter_model.safetensors", framework="pt") as weights:
    keys = list(weights.keys())
assert sum(".lora_A." in key for key in keys) == 410
assert sum(".lora_B." in key for key in keys) == 410
state = json.loads(
    (adapter / "checkpoint-105" / "trainer_state.json").read_text()
)
assert state["global_step"] == 105
assert state["max_steps"] == 105
PY
}

validate_kto_adapter() {
  local adapter="$1" expected_steps="$2" coverage_required="$3"
  .venv-train/bin/python - \
    "$adapter" "$expected_steps" "$coverage_required" \
    "$RECOVERY_MERGED" "$BEHAVIOR_DATA" "$BEHAVIOR_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from safetensors import safe_open

adapter = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
coverage_required = bool(int(sys.argv[3]))
base, data, manifest = map(Path, sys.argv[4:])
config = json.loads((adapter / "adapter_config.json").read_text())
assert config["r"] == 32
assert config["lora_alpha"] == 32
with safe_open(adapter / "adapter_model.safetensors", framework="pt") as weights:
    keys = list(weights.keys())
assert sum(".lora_A." in key for key in keys) == 410
assert sum(".lora_B." in key for key in keys) == 410
evidence = json.loads((adapter / "training_evidence.json").read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert evidence["schema_version"] == 1
assert evidence["artifact_type"] == "v2p11_kto_training_evidence"
assert evidence["status"] == "complete"
assert evidence["base_model_path"] == str(base.resolve())
assert evidence["source_data_path"] == str(data.resolve())
assert evidence["source_data_sha256"] == sha(data)
assert evidence["source_manifest_path"] == str(manifest.resolve())
assert evidence["source_manifest_sha256"] == sha(manifest)
assert evidence["source_rows"] == 606
assert evidence["coverage_required"] is coverage_required
assert evidence["per_device_train_batch_size"] == 2
assert evidence["gradient_accumulation_steps"] == 1
assert evidence["optimizer_steps"] == expected_steps
assert evidence["seed"] == 0
state = json.loads(
    (adapter / f"checkpoint-{expected_steps}" / "trainer_state.json").read_text()
)
assert state["global_step"] == expected_steps
assert state["max_steps"] == expected_steps
if coverage_required:
    assert evidence["training_rows"] == 50
    assert evidence["coverage_counts"] == {
        "desirable_correct_patch": 25,
        "empty_terminal": 8,
        "repeated_read_loop": 8,
        "wrong_nonempty_replay": 9,
    }
    selected = evidence["selected_sample_uids"]
    assert len(selected) == len(set(selected)) == 50
else:
    assert evidence["training_rows"] == 606
    assert evidence["coverage_counts"] is None
    assert evidence["selected_sample_uids"] is None
PY
}

validate_merged_model() {
  local model="$1" audit="$2"
  .venv-train/bin/python - "$model" "$audit" <<'PY'
import sys
from pathlib import Path
from phaseD_sft.merge_lora_streaming import audit_merged_checkpoint

model, audit = map(Path, sys.argv[1:])
audit_merged_checkpoint(
    model,
    audit_path=audit,
    expected_architecture="Gemma4ForConditionalGeneration",
    expected_tensors=1188,
    expected_vision=356,
    max_rss_bytes=12 * (1 << 30),
)
PY
  validate_merge_audit "$audit"
  .venv-train/bin/python - "$model" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
assert (model / "config.json").is_file()
index = model / "model.safetensors.index.json"
weights = json.loads(index.read_text())["weight_map"]
shards = {model / value for value in weights.values()}
assert shards and all(path.is_file() and path.stat().st_size > 0 for path in shards)
PY
}

validate_completion_marker() {
  .venv-train/bin/python - \
    "$COMPLETION_MARKER" \
    "$STAGE_A_ADAPTER/adapter_model.safetensors" \
    "$RECOVERY_ADAPTER/adapter_model.safetensors" \
    "$KTO_ADAPTER/adapter_model.safetensors" \
    "$FINAL_MERGED/v2p11_final_merge_audit.json" \
    "$KTO_ADAPTER/training_evidence.json" \
    "$FINAL_MERGE_MARKER" \
    "$RECOVERY_INPUT_MARKER" "$KTO_INPUT_MARKER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    marker_path,
    stage_path,
    recovery_path,
    kto_path,
    audit_path,
    kto_evidence_path,
    final_merge_marker_path,
    recovery_input_path,
    kto_input_path,
) = map(
    Path, sys.argv[1:]
)
marker = json.loads(marker_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert marker["schema_version"] == 1
assert marker["complete"] is True
assert marker["stage_a_adapter_sha256"] == sha(stage_path)
assert marker["recovery_adapter_sha256"] == sha(recovery_path)
assert marker["kto_adapter_sha256"] == sha(kto_path)
assert marker["final_merge_audit_sha256"] == sha(audit_path)
assert marker["kto_training_evidence_sha256"] == sha(kto_evidence_path)
assert marker["final_merge_marker_sha256"] == sha(final_merge_marker_path)
assert marker["recovery_input_marker_sha256"] == sha(recovery_input_path)
assert marker["kto_input_marker_sha256"] == sha(kto_input_path)
assert marker["recovery_optimizer_steps"] == 105
assert marker["kto_optimizer_steps"] == 25
PY
}

case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *) halt "PREFLIGHT_ONLY must be 0 or 1" ;;
esac
[[ "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
  halt "WAIT_SECONDS must be a positive integer"
[[ "$WAIT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
  halt "WAIT_TIMEOUT_SECONDS must be a positive integer"
[[ "$MAX_TRAINER_RESTARTS" =~ ^[0-9]+$ ]] ||
  halt "MAX_TRAINER_RESTARTS must be a nonnegative integer"

contract_json="$(
  .venv-train/bin/python phaseH_eval/v2p11_posttrain_contract.py \
    --recovery-data "$RECOVERY_DATA" \
    --behavior-data "$BEHAVIOR_DATA" \
    --behavior-manifest "$BEHAVIOR_MANIFEST" \
    --exclusions "$EXCLUSIONS" \
    --full-ids "$FULL_IDS" \
    --require-production-identity
)"
log "posttrain data contract: $contract_json"
if (( PREFLIGHT_ONLY == 1 )); then
  log "PREFLIGHT COMPLETE"
  exit 0
fi

stage_deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while [[ ! -f "$STAGE_A_MARKER" ]]; do
  stage_state="$(
    systemctl --user show "$TRAIN_UNIT" \
      -p LoadState -p ActiveState -p SubState -p Result -p ExecMainStatus \
      --no-pager 2>/dev/null
  )"
  stage_load="$(awk -F= '$1 == "LoadState" {print $2}' <<<"$stage_state")"
  stage_active="$(awk -F= '$1 == "ActiveState" {print $2}' <<<"$stage_state")"
  stage_result="$(awk -F= '$1 == "Result" {print $2}' <<<"$stage_state")"
  stage_status="$(awk -F= '$1 == "ExecMainStatus" {print $2}' <<<"$stage_state")"
  if [[ "$stage_load" != "loaded" ]]; then
    halt "stage A service is unavailable load=${stage_load:-unknown}"
  fi
  if [[ "$stage_active" == "failed" || "$stage_result" == "failed" ]]; then
    halt "stage A service failed before publishing its completion marker"
  fi
  if [[ "$stage_active" == "inactive" ]]; then
    halt "stage A service terminal without completion marker status=${stage_status:-unknown}"
  fi
  if (( $(date +%s) >= stage_deadline )); then
    halt "timed out waiting for stage A completion marker"
  fi
  log "waiting for stage A completion marker; service=$stage_state"
  sleep "$WAIT_SECONDS"
done
[[ -f "$STAGE_A_ADAPTER/adapter_model.safetensors" ]] ||
  halt "stage A marker exists but final adapter is missing"
stage_a_expected_sha="$(
  .venv-train/bin/python - "$STAGE_A_MARKER" <<'PY'
import json
import sys
from pathlib import Path
marker = json.loads(Path(sys.argv[1]).read_text())
assert marker["schema_version"] == 1
assert marker["artifact_type"] == "v2p11r3_training_completion"
assert marker["status"] == "complete"
assert marker["optimizer_steps"] == 79
assert marker["max_steps"] == 79
print(marker["adapter"]["sha256"])
PY
)"
stage_a_actual_sha="$(sha256sum "$STAGE_A_ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
[[ "$stage_a_actual_sha" == "$stage_a_expected_sha" ]] ||
  halt "stage A adapter hash differs from its completion marker"
wait_for_gpu_idle "pre-recovery"
prepare_bound_phase_inputs \
  "$RECOVERY_INPUT_MARKER" recovery_sft_inputs "$RECOVERY_ADAPTER" \
  "$STAGE_A_MARKER" \
  "$STAGE_A_ADAPTER/adapter_model.safetensors" \
  "$STAGE_A_ADAPTER/adapter_config.json" \
  "$STAGE_A_ADAPTER/run_manifest.json" \
  "$RECOVERY_DATA/train.jsonl" \
  "$RECOVERY_DATA/manifest.json"

if [[ -f "$COMPLETION_MARKER" ]]; then
  validate_recovery_adapter
  publish_or_verify_phase_marker \
    "$RECOVERY_MARKER" recovery_sft \
    "$RECOVERY_INPUT_MARKER" \
    "$RECOVERY_ADAPTER/adapter_model.safetensors" \
    "$RECOVERY_ADAPTER/adapter_config.json" \
    "$RECOVERY_ADAPTER/run_manifest.json" \
    "$RECOVERY_ADAPTER/checkpoint-105/trainer_state.json"
  prepare_bound_phase_inputs \
    "$KTO_INPUT_MARKER" kto_full_inputs "$KTO_ADAPTER" \
    "$RECOVERY_MERGE_MARKER" "$BEHAVIOR_DATA" "$BEHAVIOR_MANIFEST"
  validate_kto_adapter "$KTO_ADAPTER" 25 1
  publish_or_verify_phase_marker \
    "$KTO_MARKER" kto_full \
    "$KTO_INPUT_MARKER" \
    "$KTO_ADAPTER/adapter_model.safetensors" \
    "$KTO_ADAPTER/adapter_config.json" \
    "$KTO_ADAPTER/training_evidence.json" \
    "$KTO_ADAPTER/checkpoint-25/trainer_state.json"
  validate_merged_model \
    "$FINAL_MERGED" "$FINAL_MERGED/v2p11_final_merge_audit.json"
  publish_or_verify_model_phase_marker \
    "$FINAL_MERGE_MARKER" final_merge \
    "$FINAL_MERGED" "$FINAL_MERGED/v2p11_final_merge_audit.json" \
    "$RECOVERY_MERGE_MARKER" "$KTO_MARKER"
  validate_completion_marker
  log "reusing verified completed poststage marker=$COMPLETION_MARKER"
  exit 0
fi
command -v choom >/dev/null || halt "choom is required"

available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) ||
  halt "host MemAvailable is below 40 GiB"

restart_count=0
segment_start_step=0
if [[ -f "$RECOVERY_MARKER" ||
      -f "$RECOVERY_ADAPTER/adapter_model.safetensors" ]]; then
  validate_recovery_adapter
  publish_or_verify_phase_marker \
    "$RECOVERY_MARKER" recovery_sft \
    "$RECOVERY_INPUT_MARKER" \
    "$RECOVERY_ADAPTER/adapter_model.safetensors" \
    "$RECOVERY_ADAPTER/adapter_config.json" \
    "$RECOVERY_ADAPTER/run_manifest.json" \
    "$RECOVERY_ADAPTER/checkpoint-105/trainer_state.json"
  log "reusing verified recovery SFT"
elif [[ -e "$RECOVERY_ADAPTER" || -L "$RECOVERY_ADAPTER" ]]; then
  if [[ -L "$RECOVERY_ADAPTER" || ! -d "$RECOVERY_ADAPTER" ]]; then
    halt "recovery output exists but is not a real directory"
  fi
  archive_interrupted_output \
    "$RECOVERY_ADAPTER" recovery-sft checkpointless
  case "$INTERRUPTED_OUTPUT_STATUS" in
    archived|absent)
      log "restarting recovery SFT from a preserved checkpointless output"
      ;;
    checkpointed)
      if ! latest_step="$(latest_recovery_checkpoint)" ||
        [[ "$latest_step" == "none" ]]; then
        halt "recovery output exists without a verified resumable checkpoint"
      fi
      segment_start_step="$latest_step"
      TRAIN_RESUME_ARGS=("--resume")
      log "resuming verified recovery checkpoint step=$segment_start_step"
      ;;
  esac
fi

if [[ ! -f "$RECOVERY_MARKER" ]]; then
  while true; do
    log "recovery SFT start step=$segment_start_step restart=$restart_count"
    set +e
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      UNSLOTH_COMPILE_DISABLE=0 \
      UNSLOTH_DISABLE_DOUBLE_BUFFER=1 \
      TORCHDYNAMO_DISABLE=0 \
      TORCH_COMPILE_DISABLE=0 \
      TORCHINDUCTOR_COMPILE_THREADS=4 \
      .venv-train/bin/python phaseD_sft/train_rust_lora.py \
      --base "$BASE" \
      --init-adapter "$STAGE_A_ADAPTER" \
      --data "$RECOVERY_DATA" \
      --out "$RECOVERY_ADAPTER" \
      --rank 32 --alpha 32 --lr 5e-6 \
      --epochs 3 --bsz 1 --grad-accum 4 \
      --max-seq 32768 --warmup-steps 4 \
      --gradient-checkpointing bounded_unsloth \
      --logging-steps 5 --save-steps 5 --save-total-limit 6 \
      "${TRAIN_RESUME_ARGS[@]}" \
      >> "$RECOVERY_LOG" 2>&1 &
    trainer_pid="$!"
    if ! choom -n 750 -p "$trainer_pid"; then
      kill -TERM "$trainer_pid" 2>/dev/null
      wait "$trainer_pid" 2>/dev/null
      set -e
      halt "failed to set recovery trainer OOM priority for pid=$trainer_pid"
    fi
    .venv-train/bin/python phaseD_sft/ram_watchdog.py \
      --pid "$trainer_pid" \
      --min-available-gib "$RAM_FLOOR_GIB" \
      --interval-seconds 5 \
      --log-path "$RAM_LOG" \
      >> "$LOG" 2>&1 &
    watchdog_pid="$!"
    wait "$trainer_pid"
    train_status="$?"
    wait "$watchdog_pid"
    watchdog_status="$?"
    set -e
    if (( train_status == 0 && watchdog_status == 0 )); then
      break
    fi
    if ! (( train_status == 143 && watchdog_status == 3 )); then
      tail -n 40 "$RECOVERY_LOG"
      halt "recovery SFT failed train=$train_status watchdog=$watchdog_status"
    fi
    (( restart_count < MAX_TRAINER_RESTARTS )) ||
      halt "recovery resume refused because restart cap was reached"
    restart_count=$((restart_count + 1))
    archive_interrupted_output \
      "$RECOVERY_ADAPTER" recovery-sft checkpointless
    case "$INTERRUPTED_OUTPUT_STATUS" in
      archived|absent)
        segment_start_step=0
        TRAIN_RESUME_ARGS=()
        log "restarting checkpointless recovery segment restart=$restart_count"
        ;;
      checkpointed)
        if ! latest_step="$(latest_recovery_checkpoint)" ||
          [[ "$latest_step" == "none" ]]; then
          halt "recovery restart lacks a verified numeric checkpoint"
        fi
        (( latest_step > segment_start_step )) ||
          halt "recovery resume refused because no checkpoint advanced"
        segment_start_step="$latest_step"
        TRAIN_RESUME_ARGS=("--resume")
        ;;
    esac
    wait_for_gpu_idle "pre-recovery-restart-$restart_count"
  done
  validate_recovery_adapter
  publish_or_verify_phase_marker \
    "$RECOVERY_MARKER" recovery_sft \
    "$RECOVERY_INPUT_MARKER" \
    "$RECOVERY_ADAPTER/adapter_model.safetensors" \
    "$RECOVERY_ADAPTER/adapter_config.json" \
    "$RECOVERY_ADAPTER/run_manifest.json" \
    "$RECOVERY_ADAPTER/checkpoint-105/trainer_state.json"
fi
recovery_adapter_sha="$(
  sha256sum "$RECOVERY_ADAPTER/adapter_model.safetensors" | awk '{print $1}'
)"
log "recovery SFT complete adapter_sha256=$recovery_adapter_sha"

if [[ -f "$RECOVERY_MERGE_MARKER" || -d "$RECOVERY_MERGED" ]]; then
  validate_merged_model \
    "$RECOVERY_MERGED" "$RECOVERY_MERGED/v2p11_recovery_merge_audit.json"
  publish_or_verify_model_phase_marker \
    "$RECOVERY_MERGE_MARKER" recovery_merge \
    "$RECOVERY_MERGED" \
    "$RECOVERY_MERGED/v2p11_recovery_merge_audit.json" \
    "$RECOVERY_MARKER"
  log "reusing verified recovery merge"
else
  wait_for_gpu_idle "pre-recovery-merge"
  .venv-train/bin/python phaseD_sft/merge_lora_streaming.py \
    --base "$BASE" --adapter "$RECOVERY_ADAPTER" --out "$RECOVERY_MERGED" \
    --group-gb 3 --max-rss-gb 12 \
    --audit-architecture Gemma4ForConditionalGeneration \
    --audit-tensors 1188 --audit-vision 356 \
    --audit-filename v2p11_recovery_merge_audit.json
  validate_merged_model \
    "$RECOVERY_MERGED" "$RECOVERY_MERGED/v2p11_recovery_merge_audit.json"
  publish_or_verify_model_phase_marker \
    "$RECOVERY_MERGE_MARKER" recovery_merge \
    "$RECOVERY_MERGED" \
    "$RECOVERY_MERGED/v2p11_recovery_merge_audit.json" \
    "$RECOVERY_MARKER"
  log "recovery merge complete"
fi

prepare_bound_phase_inputs \
  "$KTO_CANARY_INPUT_MARKER" kto_canary_inputs "$KTO_CANARY" \
  "$RECOVERY_MERGE_MARKER" "$BEHAVIOR_DATA" "$BEHAVIOR_MANIFEST"
if [[ -f "$KTO_CANARY_MARKER" || -d "$KTO_CANARY" ]]; then
  validate_kto_adapter "$KTO_CANARY" 1 0
  publish_or_verify_phase_marker \
    "$KTO_CANARY_MARKER" kto_canary \
    "$KTO_CANARY_INPUT_MARKER" \
    "$KTO_CANARY/adapter_model.safetensors" \
    "$KTO_CANARY/adapter_config.json" \
    "$KTO_CANARY/training_evidence.json" \
    "$KTO_CANARY/checkpoint-1/trainer_state.json"
  log "reusing verified KTO canary"
else
  if [[ -e "$KTO_CANARY.inprogress" || -L "$KTO_CANARY.inprogress" ]]; then
    archive_interrupted_output \
      "$KTO_CANARY.inprogress" kto-canary always
    [[ "$INTERRUPTED_OUTPUT_STATUS" == "archived" ||
      "$INTERRUPTED_OUTPUT_STATUS" == "absent" ]] ||
      halt "KTO canary interrupted output was not archived"
  fi
  wait_for_gpu_idle "pre-KTO-canary"
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv-train/bin/python phaseE_rl/kto_swe_outcome.py \
    --base "$RECOVERY_MERGED" \
    --data "$BEHAVIOR_DATA" \
    --manifest "$BEHAVIOR_MANIFEST" \
    --out "$KTO_CANARY" \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 1 \
    --max-steps 1 --save-steps 1 \
    >> "$KTO_LOG" 2>&1
  validate_kto_adapter "$KTO_CANARY" 1 0
  publish_or_verify_phase_marker \
    "$KTO_CANARY_MARKER" kto_canary \
    "$KTO_CANARY_INPUT_MARKER" \
    "$KTO_CANARY/adapter_model.safetensors" \
    "$KTO_CANARY/adapter_config.json" \
    "$KTO_CANARY/training_evidence.json" \
    "$KTO_CANARY/checkpoint-1/trainer_state.json"
  log "KTO one-step canary complete"
fi

prepare_bound_phase_inputs \
  "$KTO_INPUT_MARKER" kto_full_inputs "$KTO_ADAPTER" \
  "$RECOVERY_MERGE_MARKER" "$BEHAVIOR_DATA" "$BEHAVIOR_MANIFEST"
if [[ -f "$KTO_MARKER" || -d "$KTO_ADAPTER" ]]; then
  validate_kto_adapter "$KTO_ADAPTER" 25 1
  publish_or_verify_phase_marker \
    "$KTO_MARKER" kto_full \
    "$KTO_INPUT_MARKER" \
    "$KTO_ADAPTER/adapter_model.safetensors" \
    "$KTO_ADAPTER/adapter_config.json" \
    "$KTO_ADAPTER/training_evidence.json" \
    "$KTO_ADAPTER/checkpoint-25/trainer_state.json"
  log "reusing verified full KTO"
else
  if [[ -e "$KTO_ADAPTER.inprogress" || -L "$KTO_ADAPTER.inprogress" ]]; then
    archive_interrupted_output \
      "$KTO_ADAPTER.inprogress" kto-full always
    [[ "$INTERRUPTED_OUTPUT_STATUS" == "archived" ||
      "$INTERRUPTED_OUTPUT_STATUS" == "absent" ]] ||
      halt "full KTO interrupted output was not archived"
  fi
  wait_for_gpu_idle "pre-KTO-full"
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv-train/bin/python phaseE_rl/kto_swe_outcome.py \
    --base "$RECOVERY_MERGED" \
    --data "$BEHAVIOR_DATA" \
    --manifest "$BEHAVIOR_MANIFEST" \
    --out "$KTO_ADAPTER" \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 1 \
    --require-behavior-coverage \
    --max-steps 25 --save-steps 25 \
    >> "$KTO_LOG" 2>&1
  validate_kto_adapter "$KTO_ADAPTER" 25 1
  publish_or_verify_phase_marker \
    "$KTO_MARKER" kto_full \
    "$KTO_INPUT_MARKER" \
    "$KTO_ADAPTER/adapter_model.safetensors" \
    "$KTO_ADAPTER/adapter_config.json" \
    "$KTO_ADAPTER/training_evidence.json" \
    "$KTO_ADAPTER/checkpoint-25/trainer_state.json"
fi
kto_adapter_sha="$(sha256sum "$KTO_ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
log "full KTO complete adapter_sha256=$kto_adapter_sha"

if [[ -f "$FINAL_MERGE_MARKER" || -d "$FINAL_MERGED" ]]; then
  validate_merged_model \
    "$FINAL_MERGED" "$FINAL_MERGED/v2p11_final_merge_audit.json"
  publish_or_verify_model_phase_marker \
    "$FINAL_MERGE_MARKER" final_merge \
    "$FINAL_MERGED" "$FINAL_MERGED/v2p11_final_merge_audit.json" \
    "$RECOVERY_MERGE_MARKER" "$KTO_MARKER"
  log "reusing verified final merge"
else
  wait_for_gpu_idle "pre-final-merge"
  .venv-train/bin/python phaseD_sft/merge_lora_streaming.py \
    --base "$RECOVERY_MERGED" --adapter "$KTO_ADAPTER" --out "$FINAL_MERGED" \
    --group-gb 3 --max-rss-gb 12 \
    --audit-architecture Gemma4ForConditionalGeneration \
    --audit-tensors 1188 --audit-vision 356 \
    --audit-filename v2p11_final_merge_audit.json
  validate_merged_model \
    "$FINAL_MERGED" "$FINAL_MERGED/v2p11_final_merge_audit.json"
  publish_or_verify_model_phase_marker \
    "$FINAL_MERGE_MARKER" final_merge \
    "$FINAL_MERGED" "$FINAL_MERGED/v2p11_final_merge_audit.json" \
    "$RECOVERY_MERGE_MARKER" "$KTO_MARKER"
fi

mkdir -p runs
.venv-train/bin/python - \
  "$COMPLETION_MARKER" "$gpu_uuid" \
  "$stage_a_actual_sha" "$recovery_adapter_sha" "$kto_adapter_sha" \
  "$RECOVERY_DATA/manifest.json" "$BEHAVIOR_MANIFEST" \
  "$FINAL_MERGED/v2p11_final_merge_audit.json" \
  "$KTO_ADAPTER/training_evidence.json" \
  "$FINAL_MERGE_MARKER" \
  "$RECOVERY_INPUT_MARKER" "$KTO_INPUT_MARKER" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

marker = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "complete": True,
    "gpu_uuid": sys.argv[2],
    "stage_a_adapter_sha256": sys.argv[3],
    "recovery_adapter_sha256": sys.argv[4],
    "kto_adapter_sha256": sys.argv[5],
    "recovery_manifest_sha256": hashlib.sha256(
        Path(sys.argv[6]).read_bytes()
    ).hexdigest(),
    "behavior_manifest_sha256": hashlib.sha256(
        Path(sys.argv[7]).read_bytes()
    ).hexdigest(),
    "final_merge_audit_sha256": hashlib.sha256(
        Path(sys.argv[8]).read_bytes()
    ).hexdigest(),
    "kto_training_evidence_sha256": hashlib.sha256(
        Path(sys.argv[9]).read_bytes()
    ).hexdigest(),
    "final_merge_marker_sha256": hashlib.sha256(
        Path(sys.argv[10]).read_bytes()
    ).hexdigest(),
    "recovery_input_marker_sha256": hashlib.sha256(
        Path(sys.argv[11]).read_bytes()
    ).hexdigest(),
    "kto_input_marker_sha256": hashlib.sha256(
        Path(sys.argv[12]).read_bytes()
    ).hexdigest(),
    "recovery_optimizer_steps": 105,
    "kto_optimizer_steps": 25,
}
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{marker.name}.", dir=marker.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, marker)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY
log "V2P11R3 BEHAVIOR POSTTRAIN COMPLETE marker=$COMPLETION_MARKER"
