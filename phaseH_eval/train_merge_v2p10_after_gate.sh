#!/usr/bin/env bash
# Guarded GPU1 bf16 LoRA train/merge for verified Fable revision + GPT v2.10.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="data/teacher_train_mix_v2p10"
ADAPTER="adapters/teacher_sft_v2p10_bf16"
MERGED="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full"
MERGE_AUDIT="$MERGED/v2p10_merge_audit.json"
COMPLETION_MARKER="runs/v2p10_train_merge_complete.json"
TRAIN_LOG="${TRAIN_LOG:-/tmp/train_v2p10.log}"
LOG="${LOG:-/tmp/train_merge_v2p10_after_gate.log}"
RAM_LOG="${RAM_LOG:-/tmp/ram_watchdog_v2p10.log}"
RESUME_MODE="${RESUME_MODE:-0}"
POSTTRAIN_ONLY="${POSTTRAIN_ONLY:-0}"
RAM_FLOOR_GIB="${RAM_FLOOR_GIB:-12}"
MAX_TRAINER_RESTARTS="${MAX_TRAINER_RESTARTS:-6}"
TRAIN_SAVE_STEPS="20"
TRAIN_RESUME_ARGS=()
PROD_PORTS=(8000 8101 8103 8104)
WAITER_PID="$$"
WAITER_START_TICKS="$(awk '{print $22}' /proc/$$/stat)"

log() {
  echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

check_prod_health() {
  local port
  for port in "${PROD_PORTS[@]}"; do
    curl -fsS --max-time 10 "http://127.0.0.1:$port/health" >/dev/null ||
      halt "GPU0 production health failed on port $port"
  done
  log "GPU0 production health: ports ${PROD_PORTS[*]} all green"
}

gpu1_compute_pids() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
    awk -F, -v target="$gpu1_uuid" '
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if ($1 == target) print $2
      }
    '
}

check_gpu1_idle_and_eval_port() {
  local phase="$1"
  local pids
  pids="$(gpu1_compute_pids)"
  [[ -z "$pids" ]] || halt "$phase: GPU1 has foreign compute PIDs: $pids"
  if ss -ltnp "sport = :8013" 2>/dev/null | grep -q LISTEN; then
    halt "$phase: port 8013 has a pre-existing listener"
  fi
  log "$phase: GPU1 and eval port 8013 are idle"
}

latest_checkpoint_step() {
  .venv-train/bin/python - "$ADAPTER" <<'PY'
import json
import re
import sys
from pathlib import Path

from safetensors import safe_open

adapter = Path(sys.argv[1])
checkpoints = []
for path in adapter.glob("checkpoint-*"):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        checkpoints.append((int(match.group(1)), path))
assert checkpoints, "resume requires at least one numeric checkpoint"
step, checkpoint = max(checkpoints)
assert 60 <= step <= 76, (step, checkpoint)
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
state = json.loads((checkpoint / "trainer_state.json").read_text())
assert state["global_step"] == step
assert state["max_steps"] == 76
manifest = json.loads((adapter / "run_manifest.json").read_text())
expected = {
    "data": "data/teacher_train_mix_v2p10",
    "data_len": 1211,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-5,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "warmup_steps": 8,
    "load_4bit": False,
    "selective_assistant_loss": True,
}
for key, value in expected.items():
    assert manifest[key] == value, (key, manifest[key], value)
with safe_open(
    checkpoint / "adapter_model.safetensors",
    framework="pt",
) as weights:
    keys = list(weights.keys())
assert sum(".lora_A." in key for key in keys) == 410
assert sum(".lora_B." in key for key in keys) == 410
print(step)
PY
}

gpu1_uuid="$(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    awk -F, '$1 + 0 == 1 {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      print $2
    }'
)"
[[ -n "$gpu1_uuid" ]] || halt "could not resolve GPU1 UUID"
check_gpu1_idle_and_eval_port "pre-train"
check_prod_health

case "$POSTTRAIN_ONLY" in
  0|1) ;;
  *) halt "POSTTRAIN_ONLY must be 0 or 1" ;;
esac
if (( POSTTRAIN_ONLY == 1 )); then
  (( RESUME_MODE == 1 )) ||
    halt "posttrain-only requires RESUME_MODE=1"
  [[ -d "$ADAPTER" ]] ||
    halt "posttrain-only adapter directory is missing: $ADAPTER"
  [[ -f "$ADAPTER/adapter_model.safetensors" ]] ||
    halt "posttrain-only final adapter is missing"
  TRAIN_SAVE_STEPS="1"
  segment_start_step="$(latest_checkpoint_step)"
  (( segment_start_step == 76 )) ||
    halt "posttrain-only requires checkpoint 76, found $segment_start_step"
  log "posttrain-only gate: checkpoint=$ADAPTER/checkpoint-$segment_start_step global_step=$segment_start_step/76"
else
  case "$RESUME_MODE" in
    0)
      [[ ! -e "$ADAPTER" ]] || halt "adapter output already exists: $ADAPTER"
      ;;
    1)
      [[ -d "$ADAPTER" ]] || halt "resume adapter directory is missing: $ADAPTER"
      [[ ! -f "$ADAPTER/adapter_model.safetensors" ]] ||
        halt "resume refused because the final adapter already exists"
      TRAIN_SAVE_STEPS="1"
      TRAIN_RESUME_ARGS=("--resume")
      segment_start_step="$(latest_checkpoint_step)"
      log "resume gate: checkpoint=$ADAPTER/checkpoint-$segment_start_step global_step=$segment_start_step/76"
      ;;
    *)
      halt "RESUME_MODE must be 0 or 1"
      ;;
  esac
fi
[[ "$MAX_TRAINER_RESTARTS" =~ ^[0-9]+$ ]] ||
  halt "MAX_TRAINER_RESTARTS must be a nonnegative integer"
[[ ! -e "$MERGED" ]] || halt "merged output already exists: $MERGED"
[[ ! -e "$COMPLETION_MARKER" ]] ||
  halt "completion marker already exists: $COMPLETION_MARKER"

available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
minimum_available_gib=40
if (( POSTTRAIN_ONLY == 0 && RESUME_MODE == 1 )); then
  minimum_available_gib=32
fi
(( available_kib >= minimum_available_gib * 1024 * 1024 )) ||
  halt "host MemAvailable is below ${minimum_available_gib} GiB"

read -r mix_rows minimum_teacher_rows teacher_rows \
  fable_revision_rows gpt56sol_rows max_rendered_tokens \
  dataset_manifest_sha256 bound_train_sha256 < <(
  .venv-train/bin/python - "$DATA" <<'PY'
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from phaseD_sft.train_rust_lora import assert_training_dataset_admitted
from transformers import AutoTokenizer

data = Path(sys.argv[1])
manifest_path = data / "manifest.json"
train_path = data / "train.jsonl"
manifest = json.loads(manifest_path.read_text())
assert manifest["schema_version"] == 2
assert manifest["complete"] is True
assert manifest["dataset_variant"] == "teacher_train_mix_v2p10"
assert manifest["all_training_gates_complete"] is True
assert manifest["training_admitted"] == manifest["rendered"]
minimum_teacher_rows = manifest["minimum_teacher_rows"]
teacher_rows = manifest["teacher_rows"]
fable_revision_rows = manifest["fable_revision_rows"]
assert minimum_teacher_rows == 30
assert manifest["base_rows_input"] == 1148
assert manifest["base_rows_quarantined"] == 6
assert len(manifest["base_quarantined_instance_ids"]) == 6
assert manifest["base_rows"] == 1142
assert 1 <= fable_revision_rows <= 41
assert manifest["gpt56sol_rows"] == 59
assert teacher_rows == fable_revision_rows + 59
assert manifest["rendered"] == 1142 + teacher_rows
assert manifest["teacher_row_policy"] == "standard_30_to_100"
assert manifest["teacher_delta_policy"] == (
    "verified_fable_revision_plus_gpt56sol"
)
assert manifest["success_path_distilled"] is False
assert manifest["verified_revision_supervision"] is True
assert manifest["functionally_verified_native_replay"] is True
assert len(manifest["delta_instance_ids"]) == teacher_rows
sources = {source["kind"]: source for source in manifest["teacher_sources"]}
assert set(sources) == {"fable_revision", "gpt56sol"}
fable = sources["fable_revision"]
assert fable["manifest"]["schema_version"] == 2
assert fable["manifest"]["complete"] is True
assert (
    fable["manifest"]["dataset_variant"]
    == "fable5_recent_verified_revision_v1"
)
assert fable["manifest"]["source"] == (
    "teacher:claude:claude-fable-5:verified-revision"
)
assert fable["manifest"]["rendered"] == fable_revision_rows
assert fable["manifest"]["minimum_verified_rows"] == 1
assert (
    fable["manifest"]["verified_row_policy"]
    == "explicit_exhausted_collection_override"
)
assert fable["manifest"]["training_admitted"] == fable_revision_rows
assert fable["manifest"]["all_training_gates_complete"] is True
assert fable["manifest"]["legacy_mutations_supervised"] == 0
assert fable["manifest"]["legacy_test_runs_supervised"] == 0
assert fable["manifest"]["repair_targets_exact_replay"] is True
assert fable["manifest"]["oracle_repair_targets"] is True
assert fable["manifest"]["operator_excluded_repositories"] == [
    "swesmith/pydicom__pydicom.7d361b3d"
]
assert (
    fable["manifest"]["standard_native_format_loss_gate"]["status"]
    == "passed"
)
assert (
    fable["manifest"]["standard_native_format_loss_gate"]["failure_count"]
    == 0
)
assert sources["gpt56sol"]["manifest_sha256"] == (
    "fefc7d83b921a258bc36155e082bda0f4bc9ea5d1c0a2db2b77d645455b07497"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
)
max_rendered_tokens = 0
with train_path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        assert row.get("source") != "teacher:claude:claude-fable-5"
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = (
            encoded["input_ids"]
            if isinstance(encoded, Mapping)
            else encoded
        )
        max_rendered_tokens = max(max_rendered_tokens, len(input_ids))
assert max_rendered_tokens <= 32768
train_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
assert manifest["train_jsonl_sha256"] == train_sha256
assert_training_dataset_admitted(data)
print(
    manifest["rendered"],
    minimum_teacher_rows,
    teacher_rows,
    fable_revision_rows,
    manifest["gpt56sol_rows"],
    max_rendered_tokens,
    hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    train_sha256,
)
PY
)
[[ "$mix_rows" =~ ^[0-9]+$ ]] || halt "dataset gate returned invalid row count"
[[ "$minimum_teacher_rows" =~ ^[0-9]+$ ]] ||
  halt "dataset gate returned invalid teacher floor"
[[ "$teacher_rows" =~ ^[0-9]+$ ]] ||
  halt "dataset gate returned invalid teacher row count"
[[ "$fable_revision_rows" =~ ^[0-9]+$ ]] ||
  halt "dataset gate returned invalid Fable revision row count"
[[ "$gpt56sol_rows" =~ ^[0-9]+$ ]] ||
  halt "dataset gate returned invalid GPT-5.6 Sol row count"
[[ "$max_rendered_tokens" =~ ^[0-9]+$ ]] ||
  halt "dataset gate returned invalid rendered token maximum"
(( minimum_teacher_rows == 30 )) ||
  halt "v2.10 teacher floor is not the standard 30: $minimum_teacher_rows"
(( fable_revision_rows >= 1 && fable_revision_rows <= 41 )) ||
  halt "v2.10 verified Fable revision rows are outside 1-41: $fable_revision_rows"
(( gpt56sol_rows == 59 )) ||
  halt "v2.10 GPT-5.6 Sol delta is not the pinned 59 rows: $gpt56sol_rows"
(( teacher_rows == fable_revision_rows + gpt56sol_rows )) ||
  halt "v2.10 teacher row composition is inconsistent: total=$teacher_rows fable=$fable_revision_rows gpt=$gpt56sol_rows"
(( mix_rows == 1142 + teacher_rows )) ||
  halt "v2.10 row count is not 1142 + $teacher_rows: $mix_rows"
(( max_rendered_tokens <= 32768 )) ||
  halt "v2.10 contains a row exceeding the 32,768-token training window: $max_rendered_tokens"
log "dataset gate: rows=$mix_rows teacher_rows=$teacher_rows fable_revision_rows=$fable_revision_rows gpt56sol_rows=$gpt56sol_rows max_rendered_tokens=$max_rendered_tokens minimum_teacher_rows=$minimum_teacher_rows manifest_sha256=$dataset_manifest_sha256 train_sha256=$bound_train_sha256"

if (( POSTTRAIN_ONLY == 0 )); then
  command -v choom >/dev/null || halt "choom is required for trainer OOM priority"
  segment_start_step="${segment_start_step:-0}"
  restart_count=0
  while true; do
    log "training v2.10 bf16 LoRA on GPU1 segment_start_step=$segment_start_step restart_count=$restart_count"
    echo "=== TRAIN SEGMENT START=$segment_start_step RESTART=$restart_count $(date -u +%H:%M:%S) ===" >> "$TRAIN_LOG"
    set +e
    env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      UNSLOTH_COMPILE_DISABLE=0 \
      UNSLOTH_DISABLE_DOUBLE_BUFFER=1 \
      TORCHDYNAMO_DISABLE=0 \
      TORCH_COMPILE_DISABLE=0 \
      TORCHINDUCTOR_COMPILE_THREADS=4 \
      .venv-train/bin/python phaseD_sft/train_rust_lora.py \
      --data data/teacher_train_mix_v2p10 \
      --out adapters/teacher_sft_v2p10_bf16 \
      --rank 32 --alpha 32 --lr 2e-5 \
      --epochs 1 --bsz 1 --grad-accum 16 \
      --max-seq 32768 --warmup-steps 8 \
      --gradient-checkpointing bounded_unsloth \
      --logging-steps 20 --save-steps "$TRAIN_SAVE_STEPS" --save-total-limit 6 \
      "${TRAIN_RESUME_ARGS[@]}" \
      >> "$TRAIN_LOG" 2>&1 &
    trainer_pid="$!"
    if ! choom -n 750 -p "$trainer_pid"; then
      kill "$trainer_pid" 2>/dev/null
      wait "$trainer_pid" 2>/dev/null
      set -e
      halt "failed to raise exact trainer OOM priority for pid=$trainer_pid"
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
    echo "=== TRAIN EXIT=$train_status WATCHDOG_EXIT=$watchdog_status $(date -u +%H:%M:%S) ===" >> "$TRAIN_LOG"

    if (( train_status == 0 && watchdog_status == 0 )); then
      break
    fi
    if ! (( RESUME_MODE == 1 && train_status == 143 && watchdog_status == 3 )); then
      tail -n 40 "$TRAIN_LOG"
      halt "v2.10 training failed with train_exit=$train_status watchdog_exit=$watchdog_status"
    fi

    latest_step="$(latest_checkpoint_step)"
    (( latest_step > segment_start_step )) ||
      halt "watchdog recovery refused: no newly completed checkpoint after step $segment_start_step"
    (( restart_count < MAX_TRAINER_RESTARTS )) ||
      halt "watchdog recovery refused: restart cap $MAX_TRAINER_RESTARTS reached at step $latest_step"
    restart_count=$((restart_count + 1))
    segment_start_step="$latest_step"
    log "watchdog recovery accepted: checkpoint=$latest_step/76 restart=$restart_count/$MAX_TRAINER_RESTARTS"

    for _ in $(seq 1 30); do
      [[ -z "$(gpu1_compute_pids)" ]] && break
      sleep 2
    done
    check_gpu1_idle_and_eval_port "pre-train-restart-$restart_count"
    check_prod_health
    available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    (( available_kib >= 32 * 1024 * 1024 )) ||
      halt "watchdog recovery refused: host MemAvailable is below 32 GiB"
  done
fi

.venv-train/bin/python - \
  "$ADAPTER" "$mix_rows" "$TRAIN_SAVE_STEPS" "$RESUME_MODE" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

adapter = Path(sys.argv[1])
mix_rows = int(sys.argv[2])
save_steps = int(sys.argv[3])
resume = bool(int(sys.argv[4]))
manifest = json.loads((adapter / "run_manifest.json").read_text())
expected = {
    "data": "data/teacher_train_mix_v2p10",
    "data_len": mix_rows,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-5,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "warmup_steps": 8,
    "logging_steps": 20,
    "save_steps": save_steps,
    "save_total_limit": 6,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "hybrid_checkpoint_policy": None,
    "bounded_unsloth_host_buffer_policy": {
        "buffer_count": 200,
        "initial_buffer_elements": 128 * 1024,
        "pageable_buffers": 200,
        "recycle_after_backward": True,
        "cuda_synchronized": True,
        "host_cache_drained": True,
    },
    "selective_assistant_loss": True,
    "unsloth_compile_disabled": False,
    "unsloth_double_buffer_disabled": True,
    "torchdynamo_disabled": False,
    "torch_compile_disabled": False,
    "resume": resume,
}
for key, value in expected.items():
    assert manifest[key] == value, (key, manifest[key], value)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
with safe_open(adapter / "adapter_model.safetensors", framework="pt") as weights:
    keys = list(weights.keys())
lora_a = {key.replace(".lora_A.", ".lora_.") for key in keys if ".lora_A." in key}
lora_b = {key.replace(".lora_B.", ".lora_.") for key in keys if ".lora_B." in key}
assert len(target_modules) == 7
assert lora_a == lora_b
assert len(lora_a) == 410
assert len(lora_a) + len(lora_b) == 820
print(f"adapter gate: pairs={len(lora_a)} tensors={len(lora_a) + len(lora_b)}")
PY

adapter_sha256="$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
log "adapter gate: adapter_sha256=$adapter_sha256"

log "merging v2.10 with the full multimodal base"
.venv-train/bin/python phaseD_sft/merge_lora_streaming.py \
  --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --adapter "$ADAPTER" --out "$MERGED" --group-gb 3 --max-rss-gb 12

.venv-train/bin/python - "$MERGED" "$MERGE_AUDIT" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

merged = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
config = json.loads((merged / "config.json").read_text())
weight_map = json.loads(
    (merged / "model.safetensors.index.json").read_text()
)["weight_map"]
expected_tensors = 1188
expected_vision = 356
vision = sum("vision" in name for name in weight_map)
assert config["architectures"] == ["Gemma4ForConditionalGeneration"]
assert len(weight_map) == expected_tensors
assert vision == expected_vision

actual_tensors = set()
nonfinite_tensors = []
misplaced_tensors = []
for shard_name in sorted(set(weight_map.values())):
    shard_path = merged / shard_name
    assert shard_path.is_file(), shard_path
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            actual_tensors.add(key)
            if weight_map.get(key) != shard_name:
                misplaced_tensors.append(key)
            tensor_slice = handle.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            if not shape:
                chunks = (handle.get_tensor(key),)
            else:
                row_width = math.prod(shape[1:]) if len(shape) > 1 else 1
                rows_per_chunk = max(1, 8_000_000 // max(row_width, 1))
                chunks = (
                    tensor_slice[start : min(start + rows_per_chunk, shape[0])]
                    for start in range(0, shape[0], rows_per_chunk)
                )
            for chunk in chunks:
                if (
                    (torch.is_floating_point(chunk) or torch.is_complex(chunk))
                    and not bool(torch.isfinite(chunk).all())
                ):
                    nonfinite_tensors.append(key)
                    break

missing_tensors = sorted(set(weight_map) - actual_tensors)
unexpected_tensors = sorted(actual_tensors - set(weight_map))
assert not missing_tensors, missing_tensors[:10]
assert not unexpected_tensors, unexpected_tensors[:10]
assert not misplaced_tensors, misplaced_tensors[:10]
assert not nonfinite_tensors, nonfinite_tensors[:10]
audit = {
    "schema_version": 1,
    "architecture": config["architectures"][0],
    "expected_tensors": expected_tensors,
    "actual_tensors": len(actual_tensors),
    "expected_vision": expected_vision,
    "actual_vision": vision,
    "missing_tensors": missing_tensors,
    "unexpected_tensors": unexpected_tensors,
    "misplaced_tensors": misplaced_tensors,
    "nonfinite_tensors": nonfinite_tensors,
    "complete": True,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(
    f"merge gate: architecture={audit['architecture']} "
    f"tensors={audit['actual_tensors']} vision={audit['actual_vision']} "
    "missing=0 unexpected=0 misplaced=0 nonfinite=0"
)
PY

current_dataset_manifest_sha256="$(
  sha256sum "$DATA/manifest.json" | awk '{print $1}'
)"
[[ "$current_dataset_manifest_sha256" == "$dataset_manifest_sha256" ]] ||
  halt "dataset manifest changed during train/merge"
current_train_sha256="$(sha256sum "$DATA/train.jsonl" | awk '{print $1}')"
[[ "$current_train_sha256" == "$bound_train_sha256" ]] ||
  halt "bound trainer JSONL changed during train/merge"
merge_audit_sha256="$(sha256sum "$MERGE_AUDIT" | awk '{print $1}')"

check_gpu1_idle_and_eval_port "post-merge"
check_prod_health

mkdir -p runs
.venv-train/bin/python - \
  "$COMPLETION_MARKER" "$WAITER_PID" "$WAITER_START_TICKS" \
  "$dataset_manifest_sha256" "$adapter_sha256" "$merge_audit_sha256" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

marker = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "complete": True,
    "waiter_pid": int(sys.argv[2]),
    "waiter_start_ticks": sys.argv[3],
    "dataset_manifest_sha256": sys.argv[4],
    "adapter_sha256": sys.argv[5],
    "merge_audit_sha256": sys.argv[6],
}
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{marker.name}.",
    dir=marker.parent,
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, marker)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY

log "=== V2P10 TRAIN+MERGE COMPLETE — GPU1 free; waiter_pid=$WAITER_PID waiter_start_ticks=$WAITER_START_TICKS dataset_manifest_sha256=$dataset_manifest_sha256 adapter_sha256=$adapter_sha256 merge_audit_sha256=$merge_audit_sha256 ==="
