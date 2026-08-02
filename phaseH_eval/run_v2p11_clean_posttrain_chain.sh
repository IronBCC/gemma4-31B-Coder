#!/usr/bin/env bash
# Wait for clean raw-base v2.11 training, then audit, merge, gate, and evaluate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_UNIT="${TRAIN_UNIT:-v2p11-clean-fable51-train-gpu1-v2.service}"
TRAIN_INVOCATION_ID="${TRAIN_INVOCATION_ID:-bf042e80eda24928a28c6c0784cbc0b2}"
TRAIN_WRAPPER_PID="${TRAIN_WRAPPER_PID:-68011}"
TRAIN_PID="${TRAIN_PID:-68900}"
WATCHDOG_PID="${WATCHDOG_PID:-68902}"
GPU_INDEX="${GPU_INDEX:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MEMORY_FLOOR_GIB="${MEMORY_FLOOR_GIB:-40}"
MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"
BASE="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
DATA="data/teacher_train_mix_v2p11_fable1262"
BASE_DATA="data/teacher_train_mix_v2p10"
ADAPTER="adapters/teacher_sft_v2p11_clean_fable51_bf16"
CANDIDATE_NAME="teacher_sft_v2p11_clean_fable51"
ARTIFACT_TAG="v2p11_clean_fable51"
MERGED="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11_clean_fable51_full"
MERGE_AUDIT="$MERGED/v2p11_clean_merge_audit.json"
CONTEXT_AUDIT="runs/v2p11_clean_fable51_context_audit.json"
TRAIN_LOG_SOURCE="/tmp/train_v2p11_clean_fable51_gpu1.log"
WATCHDOG_LOG_SOURCE="/tmp/ram_watchdog_v2p11_clean_fable51_gpu1.log"
TRAINING_JOURNAL="runs/${ARTIFACT_TAG}_training_journal.log"
WATCHDOG_EVIDENCE="runs/${ARTIFACT_TAG}_watchdog.log"
UNIT_JOURNAL="runs/${ARTIFACT_TAG}_unit_journal.jsonl"
GPU_IDENTITY="runs/${ARTIFACT_TAG}_gpu1_training_identity.json"
TRAINING_COMPLETION="runs/${ARTIFACT_TAG}_training_completion.json"
PORTABILITY_ROOT="runs/${ARTIFACT_TAG}_portability_stock"
PORTABILITY_GATE="runs/${ARTIFACT_TAG}_portability_gate.json"
PROVENANCE="runs/${ARTIFACT_TAG}_completion_provenance.json"
VERDICT="runs/${ARTIFACT_TAG}_vs_v2p10_full300.json"
FULL300_MARKDOWN="runs/${ARTIFACT_TAG}_vs_v2p10_full300.md"
V2P10_COMPOSITE="runs/v2p10_full300_composite.json"
V2P10_LINEAGE="runs/v2p11_v2p10_training_lineage.json"
LOG="${LOG:-/tmp/run_v2p11_clean_posttrain_chain.log}"
EVAL_PY="$ROOT/.venv-eval/bin/python"
TRAIN_PY="$ROOT/.venv-train/bin/python"

exec > >(tee -a "$LOG") 2>&1

log() {
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

halt() {
  log "HALT: $*"
  exit 1
}

exact_pid() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

process_command() {
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null || true
}

validate_live_processes() {
  local wrapper_command trainer_command watchdog_command main_pid invocation
  kill -0 "$TRAIN_WRAPPER_PID" 2>/dev/null || return 0
  wrapper_command="$(process_command "$TRAIN_WRAPPER_PID")"
  if [[ -z "$wrapper_command" ]]; then
    kill -0 "$TRAIN_WRAPPER_PID" 2>/dev/null || return 0
  fi
  [[ "$wrapper_command" == *"phaseH_eval/train_v2p11_clean_gpu1.sh"* ]] ||
    halt "training wrapper PID identity changed"
  main_pid="$(
    systemctl --user show "$TRAIN_UNIT" -p MainPID --value 2>/dev/null || true
  )"
  invocation="$(
    systemctl --user show "$TRAIN_UNIT" -p InvocationID --value 2>/dev/null || true
  )"
  if [[ "$main_pid" != "$TRAIN_WRAPPER_PID" ]]; then
    kill -0 "$TRAIN_WRAPPER_PID" 2>/dev/null || return 0
    halt "training unit MainPID changed"
  fi
  if [[ "$invocation" != "$TRAIN_INVOCATION_ID" ]]; then
    kill -0 "$TRAIN_WRAPPER_PID" 2>/dev/null || return 0
    halt "training unit InvocationID changed"
  fi

  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    trainer_command="$(process_command "$TRAIN_PID")"
    kill -0 "$TRAIN_PID" 2>/dev/null || trainer_command=""
    if [[ -n "$trainer_command" ]]; then
      [[ "$trainer_command" == *"phaseD_sft/train_rust_lora.py"* &&
        "$trainer_command" == *"--data $DATA"* &&
        "$trainer_command" == *"--out $ADAPTER"* ]] ||
        halt "trainer PID identity changed"
    fi
  fi
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    watchdog_command="$(process_command "$WATCHDOG_PID")"
    kill -0 "$WATCHDOG_PID" 2>/dev/null || watchdog_command=""
    if [[ -n "$watchdog_command" ]]; then
      [[ "$watchdog_command" == *"phaseD_sft/ram_watchdog.py"* &&
        "$watchdog_command" == *"--pid $TRAIN_PID"* ]] ||
        halt "watchdog PID identity changed"
    fi
  fi
}

seal_gpu_identity() {
  local gpu_uuid gpu_compute_pids
  [[ ! -e "$GPU_IDENTITY" ]] ||
    halt "GPU training identity already exists"
  kill -0 "$TRAIN_PID" 2>/dev/null ||
    halt "trainer exited before physical GPU1 identity was sealed"
  gpu_uuid="$(
    nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader |
      awk '{$1=$1; print}'
  )"
  gpu_compute_pids="$(
    nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
      --format=csv,noheader,nounits |
      awk '{$1=$1; if ($1 ~ /^[0-9]+$/) print $1}'
  )"
  [[ -n "$gpu_uuid" ]] || halt "physical GPU1 UUID is unavailable"
  grep -qx "$TRAIN_PID" <<< "$gpu_compute_pids" ||
    halt "trainer PID is not executing on physical GPU1"
  "$TRAIN_PY" - \
    "$TRAIN_PID" "$TRAIN_UNIT" "$TRAIN_INVOCATION_ID" \
    "$gpu_uuid" "$gpu_compute_pids" "$GPU_IDENTITY" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

train_pid, train_unit, invocation_id, gpu_uuid, gpu_pids_text, output = sys.argv[1:]
environment = {}
for item in Path(f"/proc/{train_pid}/environ").read_bytes().split(b"\0"):
    if b"=" in item:
        key, value = item.split(b"=", 1)
        environment[key.decode(errors="strict")] = value.decode(errors="strict")
assert environment.get("CUDA_VISIBLE_DEVICES") == "1"
command_line = Path(f"/proc/{train_pid}/cmdline").read_bytes()
assert b"phaseD_sft/train_rust_lora.py" in command_line
gpu_compute_pids = sorted({int(value) for value in gpu_pids_text.split()})
assert int(train_pid) in gpu_compute_pids
report = {
    "schema_version": 1,
    "artifact_type": "v2p11_clean_gpu_training_identity",
    "status": "complete",
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "train_unit": train_unit,
    "train_invocation_id": invocation_id,
    "train_pid": int(train_pid),
    "gpu_index": 1,
    "gpu_uuid": gpu_uuid,
    "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
    "gpu_compute_pids": gpu_compute_pids,
    "train_pid_on_gpu": True,
    "trainer_cmdline_sha256": hashlib.sha256(command_line).hexdigest(),
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(report, sort_keys=True))
PY
}

wait_for_training() {
  local state result status
  exact_pid "$TRAIN_WRAPPER_PID" || halt "TRAIN_WRAPPER_PID is invalid"
  exact_pid "$TRAIN_PID" || halt "TRAIN_PID is invalid"
  exact_pid "$WATCHDOG_PID" || halt "WATCHDOG_PID is invalid"
  [[ "$TRAIN_INVOCATION_ID" =~ ^[0-9a-f]{32}$ ]] ||
    halt "TRAIN_INVOCATION_ID is invalid"

  while kill -0 "$TRAIN_WRAPPER_PID" 2>/dev/null; do
    validate_live_processes
    sleep "$WAIT_SECONDS"
  done
  for _ in $(seq 1 30); do
    state="$(
      systemctl --user show "$TRAIN_UNIT" -p ActiveState --value 2>/dev/null || true
    )"
    [[ "$state" == "inactive" ]] && break
    [[ "$state" == "active" || "$state" == "activating" ||
      "$state" == "deactivating" ]] ||
      halt "training unit state is ${state:-unknown}"
    sleep 2
  done
  [[ "$state" == "inactive" ]] || halt "training unit did not become inactive"
  result="$(
    systemctl --user show "$TRAIN_UNIT" -p Result --value 2>/dev/null || true
  )"
  status="$(
    systemctl --user show "$TRAIN_UNIT" -p ExecMainStatus --value 2>/dev/null || true
  )"
  [[ "$result" == "success" && "$status" == "0" ]] ||
    halt "training failed result=${result:-unknown} status=${status:-unknown}"
  [[ -f "$ADAPTER/adapter_model.safetensors" &&
    -f "$ADAPTER/checkpoint-79/trainer_state.json" ]] ||
    halt "training exited without the final adapter/checkpoint"
  [[ -s "$TRAIN_LOG_SOURCE" && -s "$WATCHDOG_LOG_SOURCE" ]] ||
    halt "training source evidence is missing"
  grep -q "optimizer_step=79/79" "$TRAIN_LOG_SOURCE" ||
    halt "training did not reach optimizer step 79"
  grep -q "event=target_exited pid=$TRAIN_PID" "$WATCHDOG_LOG_SOURCE" ||
    halt "watchdog did not record the exact trainer exit"
  if grep -Eq \
    "event=(below_threshold|meminfo_error|health_failure) pid=$TRAIN_PID" \
    "$WATCHDOG_LOG_SOURCE"; then
    halt "watchdog recorded a training safety violation"
  fi
  log "training complete wrapper=$TRAIN_WRAPPER_PID trainer=$TRAIN_PID watchdog=$WATCHDOG_PID invocation=$TRAIN_INVOCATION_ID"
}

capture_training_evidence() {
  [[ ! -e "$TRAINING_JOURNAL" &&
    ! -e "$WATCHDOG_EVIDENCE" &&
    ! -e "$UNIT_JOURNAL" ]] ||
    halt "refusing stale or partial captured training evidence"
  "$TRAIN_PY" - \
    "$TRAIN_INVOCATION_ID" "$TRAIN_LOG_SOURCE" "$WATCHDOG_LOG_SOURCE" \
    "$TRAINING_JOURNAL" "$WATCHDOG_EVIDENCE" "$UNIT_JOURNAL" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

(
    invocation_id,
    train_source,
    watchdog_source,
    train_target,
    watchdog_target,
    unit_target,
) = sys.argv[1:]
unit_journal = subprocess.run(
    [
        "journalctl",
        "--user",
        f"_SYSTEMD_INVOCATION_ID={invocation_id}",
        "--output=json",
        "--no-pager",
    ],
    check=True,
    capture_output=True,
).stdout
payloads = (
    (Path(train_target), Path(train_source).read_bytes()),
    (Path(watchdog_target), Path(watchdog_source).read_bytes()),
    (Path(unit_target), unit_journal),
)
assert all(payload for _, payload in payloads)
for path, payload in payloads:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
PY
}

wait_for_memory() {
  local available_kib
  while true; do
    available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available_kib >= MEMORY_FLOOR_GIB * 1024 * 1024 )); then
      log "memory admission passed available_kib=$available_kib"
      return
    fi
    log "waiting for ${MEMORY_FLOOR_GIB}GiB MemAvailable current_kib=$available_kib"
    sleep "$WAIT_SECONDS"
  done
}

validate_dataset() {
  "$TRAIN_PY" - "$DATA" "$BASE_DATA" "$CONTEXT_AUDIT" <<'PY'
import json
import sys
from pathlib import Path

from phaseH_eval.v2p11_clean_completion_provenance import _validate_dataset

validated = _validate_dataset(*(Path(value) for value in sys.argv[1:]))
print(json.dumps({
    "rows": validated["rows"],
    "base_rows": validated["base_rows"],
    "new_fable_rows": validated["new_fable_rows"],
}, sort_keys=True))
PY
}

audit_adapter() {
  "$TRAIN_PY" - \
    "$BASE" "$DATA" "$ADAPTER" "$TRAIN_UNIT" "$TRAIN_INVOCATION_ID" \
    "$TRAIN_WRAPPER_PID" "$TRAIN_PID" "$WATCHDOG_PID" \
    "$TRAINING_JOURNAL" "$WATCHDOG_EVIDENCE" "$UNIT_JOURNAL" \
    "$GPU_IDENTITY" "$TRAINING_COMPLETION" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

from safetensors import safe_open

(
    base,
    data,
    adapter,
    train_unit,
    invocation_id,
    wrapper_pid,
    train_pid,
    watchdog_pid,
    journal_value,
    watchdog_value,
    unit_journal_value,
    gpu_identity_value,
    completion_value,
) = sys.argv[1:]
adapter_path = Path(adapter)
weights_path = adapter_path / "adapter_model.safetensors"
config_path = adapter_path / "adapter_config.json"
run_path = adapter_path / "run_manifest.json"
checkpoint_path = adapter_path / "checkpoint-79" / "trainer_state.json"
journal_path = Path(journal_value)
watchdog_path = Path(watchdog_value)
unit_journal_path = Path(unit_journal_value)
gpu_identity_path = Path(gpu_identity_value)
config = json.loads(config_path.read_text())
run = json.loads(run_path.read_text())
checkpoint = json.loads(checkpoint_path.read_text())
expected = {
    "base": base,
    "data": data,
    "out": adapter,
    "init_adapter": None,
    "data_len": 1262,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-5,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "max_steps": 79,
    "warmup_steps": 8,
    "logging_steps": 5,
    "save_steps": 5,
    "save_total_limit": 3,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}
changed = {
    key: (run.get(key), value)
    for key, value in expected.items()
    if run.get(key) != value
}
assert not changed, changed
assert config["r"] == 32 and config["lora_alpha"] == 32
assert checkpoint["global_step"] == checkpoint["max_steps"] == 79

tensor_count = a_count = b_count = nonzero_b_count = nonfinite_count = 0
with safe_open(weights_path, framework="pt", device="cpu") as weights:
    for key in weights.keys():
        tensor = weights.get_tensor(key)
        finite = tensor.isfinite().all().item()
        nonfinite_count += int(not finite)
        tensor_count += 1
        a_count += int(".lora_A." in key)
        is_b = ".lora_B." in key
        b_count += int(is_b)
        nonzero_b_count += int(is_b and bool(tensor.count_nonzero().item()))
assert (tensor_count, a_count, b_count, nonfinite_count) == (820, 410, 410, 0)
assert nonzero_b_count > 0

def finite_metric(item, key):
    value = item.get(key) if isinstance(item, dict) else None
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else None
    )

history = checkpoint.get("log_history")
assert isinstance(history, list)
loss_samples = sum(finite_metric(item, "loss") is not None for item in history)
grad_norm_samples = sum(
    finite_metric(item, "grad_norm") is not None for item in history
)
assert loss_samples > 0 and grad_norm_samples > 0

journal = journal_path.read_text(encoding="utf-8")
watchdog = watchdog_path.read_text(encoding="utf-8")
assert "optimizer_step=79/79" in journal
assert f"[done] adapter saved -> {adapter}" in journal
assert f"event=target_exited pid={train_pid}" in watchdog
assert not re.search(
    rf"event=(below_threshold|meminfo_error|health_failure) pid={train_pid}\b",
    watchdog,
)
unit_rows = [
    json.loads(line)
    for line in unit_journal_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert any(
    row.get("_SYSTEMD_INVOCATION_ID") == invocation_id for row in unit_rows
)

def binding(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }

report = {
    "schema_version": 1,
    "artifact_type": "v2p11_clean_training_completion",
    "status": "complete",
    "base_model": base,
    "init_adapter": None,
    "train_unit": train_unit,
    "train_invocation_id": invocation_id,
    "train_wrapper_pid": int(wrapper_pid),
    "train_pid": int(train_pid),
    "watchdog_pid": int(watchdog_pid),
    "optimizer_steps": 79,
    "max_steps": 79,
    "finite_loss_samples": loss_samples,
    "finite_grad_norm_samples": grad_norm_samples,
    "adapter_tensor_count": tensor_count,
    "lora_a_tensor_count": a_count,
    "lora_b_tensor_count": b_count,
    "nonfinite_tensor_count": nonfinite_count,
    "nonzero_lora_b_tensor_count": nonzero_b_count,
    "adapter": binding(weights_path),
    "adapter_config": binding(config_path),
    "run_manifest": binding(run_path),
    "checkpoint_state": binding(checkpoint_path),
    "training_journal": binding(journal_path),
    "watchdog_log": binding(watchdog_path),
    "unit_journal": binding(unit_journal_path),
    "gpu_identity": binding(gpu_identity_path),
}
completion_path = Path(completion_value)
completion_path.parent.mkdir(parents=True, exist_ok=True)
rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(
    completion_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(rendered)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({
    "adapter_sha256": report["adapter"]["sha256"],
    "tensors": tensor_count,
    "nonzero_lora_b": nonzero_b_count,
}, sort_keys=True))
PY
}

merge_model() {
  "$TRAIN_PY" phaseD_sft/merge_lora_streaming.py \
    --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" \
    --group-gb 3 --max-rss-gb 12 --dry-run
  wait_for_memory
  "$TRAIN_PY" phaseD_sft/merge_lora_streaming.py \
    --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" \
    --group-gb 3 --max-rss-gb 12 \
    --audit-architecture Gemma4ForConditionalGeneration \
    --audit-tensors 1188 --audit-vision 356 \
    --audit-filename v2p11_clean_merge_audit.json
  [[ -f "$MERGE_AUDIT" ]] || halt "merge completed without audit"
  log "merge complete model=$MERGED"
}

run_portability() {
  env \
    LINEAGE_MODE=direct_lora \
    PORTABILITY_MODE=full \
    CANDIDATE_NAME="$CANDIDATE_NAME" \
    CANDIDATE_MODEL="$MERGED" \
    FINAL_AUDIT="$MERGE_AUDIT" \
    ARTIFACT_ROOT="$PORTABILITY_ROOT" \
    CONTROL_ROOT="runs/v2p11_portability_stock/v2p10" \
    CANDIDATE_ROOT="$PORTABILITY_ROOT/candidate" \
    GATE="$PORTABILITY_GATE" \
    MEMORY_ADMISSION_GIB="$MEMORY_FLOOR_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    GPU_INDEX=1 PORT=8013 \
    /usr/bin/bash phaseH_eval/run_v2p11_portability_gate.sh
  [[ -f "$PORTABILITY_GATE" ]] || halt "portability gate did not publish"
}

publish_provenance() {
  "$EVAL_PY" phaseH_eval/v2p11_clean_completion_provenance.py \
    --candidate-name "$CANDIDATE_NAME" \
    --dataset "$DATA" \
    --base-data "$BASE_DATA" \
    --dataset-context-audit "$CONTEXT_AUDIT" \
    --adapter "$ADAPTER" \
    --training-completion "$TRAINING_COMPLETION" \
    --final-model "$MERGED" \
    --merge-audit "$MERGE_AUDIT" \
    --portability-gate "$PORTABILITY_GATE" \
    --full-ids data/swebench_lite_test_ids.json \
    --v2p10 "$V2P10_COMPOSITE" \
    --v2p10-lineage "$V2P10_LINEAGE" \
    --out "$PROVENANCE"
}

run_full300() {
  local eval_status
  set +e
  env \
    NAME="$CANDIDATE_NAME" \
    MODEL="$MERGED" \
    ARTIFACT_TAG="$ARTIFACT_TAG" \
    LINEAGE_MODE=direct_lora \
    FINAL_AUDIT="$MERGE_AUDIT" \
    PORTABILITY_MARKER="$PORTABILITY_GATE" \
    PROVENANCE="$PROVENANCE" \
    MEMORY_ADMISSION_GIB="$MEMORY_FLOOR_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    GPU_INDEX=1 PORT=8013 \
    /usr/bin/bash phaseH_eval/eval_v2p11_full300_after_merge.sh
  eval_status=$?
  set -e
  [[ -f "$VERDICT" ]] ||
    halt "full300 exited status=$eval_status without a verdict"
  jq '{status,population,v2p10,v2p11,verdict}' "$VERDICT"
  (( eval_status == 0 )) ||
    halt "full300 did not beat canonical v2.10 status=$eval_status"
}

main() {
  [[ "$GPU_INDEX" == "1" ]] || halt "GPU_INDEX is fixed to physical GPU1"
  [[ "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    halt "WAIT_SECONDS must be a positive integer"
  [[ "$MEMORY_FLOOR_GIB" =~ ^[1-9][0-9]*$ ]] ||
    halt "MEMORY_FLOOR_GIB must be a positive integer"
  [[ "$MEMORY_WATCHDOG_GIB" =~ ^[1-9][0-9]*$ ]] ||
    halt "MEMORY_WATCHDOG_GIB must be a positive integer"
  [[ -x "$TRAIN_PY" && -x "$EVAL_PY" ]] ||
    halt "required Python environments are missing"
  [[ -d "$BASE" && -d "$DATA" && -d "$BASE_DATA" ]] ||
    halt "model or dataset inputs are missing"
  [[ -f "$CONTEXT_AUDIT" && -f "$V2P10_COMPOSITE" &&
    -f "$V2P10_LINEAGE" ]] || halt "canonical input evidence is missing"
  [[ ! -e "$TRAINING_COMPLETION" &&
    ! -e "$GPU_IDENTITY" &&
    ! -e "$TRAINING_JOURNAL" &&
    ! -e "$WATCHDOG_EVIDENCE" &&
    ! -e "$UNIT_JOURNAL" &&
    ! -e "$MERGED" &&
    ! -e "$PORTABILITY_ROOT" &&
    ! -e "$PORTABILITY_GATE" &&
    ! -e "$PROVENANCE" &&
    ! -e "$VERDICT" &&
    ! -e "$FULL300_MARKDOWN" ]] ||
    halt "nonresumable post-training outputs already exist"
  validate_live_processes
  seal_gpu_identity
  wait_for_training
  capture_training_evidence
  validate_dataset
  audit_adapter
  wait_for_memory
  merge_model
  wait_for_memory
  run_portability
  publish_provenance
  wait_for_memory
  run_full300
  log "CLEAN V2P11 GOAL CHAIN COMPLETE verdict=$VERDICT"
}

main "$@"
