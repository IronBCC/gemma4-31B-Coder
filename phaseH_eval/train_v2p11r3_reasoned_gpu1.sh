#!/usr/bin/env bash
# Clean, data-only v2.11r3 continuation from the immutable v2.10 adapter.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU_INDEX="${GPU_INDEX:-1}"
DATA="${DATA:-data/teacher_train_mix_v2p11_fable1262_reasoned_v1}"
INIT_ADAPTER="${INIT_ADAPTER:-adapters/teacher_sft_v2p10_bf16}"
ADAPTER="${ADAPTER:-adapters/teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16}"
CONTEXT_AUDIT="${CONTEXT_AUDIT:-runs/v2p11r3_reasoned_context_audit.json}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/train_v2p11r3_reasoned_gpu1_rerun1.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/tmp/ram_watchdog_v2p11r3_reasoned_gpu1_rerun1.log}"
RAM_FLOOR_GIB="${RAM_FLOOR_GIB:-12}"
START_FLOOR_GIB="${START_FLOOR_GIB:-24}"
EXPECTED_ROWS=1262
EXPECTED_STEPS=79

log() {
  echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$TRAIN_LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

[[ "$GPU_INDEX" == "1" ]] || halt "GPU_INDEX must be 1"
[[ -d "$DATA" ]] || halt "dataset is missing: $DATA"
[[ -f "$INIT_ADAPTER/adapter_model.safetensors" ]] || halt "init adapter is missing"
[[ ! -e "$ADAPTER" ]] || halt "output adapter already exists: $ADAPTER"
[[ ! -e "$CONTEXT_AUDIT" ]] || halt "context audit already exists: $CONTEXT_AUDIT"
command -v choom >/dev/null || halt "choom is required"

available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= START_FLOOR_GIB * 1024 * 1024 )) ||
  halt "MemAvailable is below ${START_FLOOR_GIB} GiB"

foreign_compute_pids="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | awk 'NF {print $1}')"
[[ -z "$foreign_compute_pids" ]] ||
  halt "GPU${GPU_INDEX} has pre-existing compute PIDs: $foreign_compute_pids"
log "preflight passed gpu=$GPU_INDEX available_kib=$available_kib"

.venv-train/bin/python - "$DATA" "$CONTEXT_AUDIT" <<'PY'
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from phaseD_sft.train_rust_lora import assert_training_dataset_admitted
from transformers import AutoTokenizer

data = Path(sys.argv[1])
out = Path(sys.argv[2])
manifest_path = data / "manifest.json"
train_path = data / "train.jsonl"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "complete": True,
    "all_training_gates_complete": True,
    "dataset_variant": "teacher_train_mix_v2p11_fable_reasoned_v1",
    "source_variant": "teacher_train_mix_v2p11_fable_extension",
    "rendered": 1262,
    "training_admitted": 1262,
    "fable_rows": 92,
    "canonical_bash_tool_turns": 517,
    "reasoned_tool_turns": 468,
    "evaluation_overlap": 0,
}
for key, value in expected.items():
    assert manifest.get(key) == value, (key, manifest.get(key), value)
structural = manifest.get("structural_gates")
assert structural == {
    "all_fable_supervised_tool_calls_are_canonical_bash": True,
    "all_messages_have_boolean_loss": True,
    "evaluation_overlap": 0,
    "reasoned_tool_turns": 468,
}, structural
format_gate = manifest.get("format_loss_gate")
assert isinstance(format_gate, Mapping), format_gate
assert format_gate.get("status") == "passed_full_dataset_verification", format_gate
assert format_gate.get("samples") == 1262, format_gate
assert format_gate.get("failure_count") == 0, format_gate
report = format_gate.get("report")
assert isinstance(report, Mapping) and len(str(report.get("sha256", ""))) == 64, report
assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest["train_jsonl_sha256"]
assert_training_dataset_admitted(str(data))

rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
assert len(rows) == 1262
assert all(isinstance(row.get("instance_id"), str) for row in rows)
tokenizer = AutoTokenizer.from_pretrained("/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
lengths = []
for index, row in enumerate(rows):
    encoded = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    assert isinstance(token_ids, list) and all(type(token) is int for token in token_ids), index
    lengths.append(len(token_ids))
assert max(lengths) <= 32768, max(lengths)
artifact = {
    "schema_version": 1,
    "artifact_type": "v2p11r3_dataset_context_audit",
    "status": "complete",
    "dataset": {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
    },
    "rows": len(rows),
    "max_seq": 32768,
    "max_rendered_tokens": max(lengths),
    "over_limit_rows": sum(length > 32768 for length in lengths),
}
assert artifact["over_limit_rows"] == 0
out.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(artifact, handle, sort_keys=True, indent=2)
    handle.write("\n")
print(json.dumps(artifact, sort_keys=True))
PY

log "sealed dataset and v2p11r3_dataset_context_audit passed"
set +e
env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  UNSLOTH_COMPILE_DISABLE=0 \
  UNSLOTH_DISABLE_DOUBLE_BUFFER=1 \
  TORCHDYNAMO_DISABLE=0 \
  TORCH_COMPILE_DISABLE=0 \
  TORCHINDUCTOR_COMPILE_THREADS=4 \
  .venv-train/bin/python phaseD_sft/train_rust_lora.py \
    --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
    --data "$DATA" \
    --out "$ADAPTER" \
    --init-adapter "$INIT_ADAPTER" \
    --rank 32 --alpha 32 --lr 2e-6 \
    --epochs 1 --bsz 1 --grad-accum 16 \
    --max-seq 32768 --max-steps 79 --warmup-steps 4 \
    --gradient-checkpointing bounded_unsloth \
    --logging-steps 5 --save-steps 10 --save-total-limit 3 \
    >> "$TRAIN_LOG" 2>&1 &
trainer_pid="$!"
if ! choom -n 750 -p "$trainer_pid"; then
  kill "$trainer_pid" 2>/dev/null
  wait "$trainer_pid" 2>/dev/null
  set -e
  halt "failed to apply OOM priority to exact trainer pid=$trainer_pid"
fi
.venv-train/bin/python phaseD_sft/ram_watchdog.py \
  --pid "$trainer_pid" \
  --min-available-gib "$RAM_FLOOR_GIB" \
  --interval-seconds 5 \
  --log-path "$WATCHDOG_LOG" \
  >> "$TRAIN_LOG" 2>&1 &
watchdog_pid="$!"
wait "$trainer_pid"
train_status="$?"
wait "$watchdog_pid"
watchdog_status="$?"
set -e

(( train_status == 0 && watchdog_status == 0 )) ||
  halt "training failed train_exit=$train_status watchdog_exit=$watchdog_status"
.venv-train/bin/python - "$ADAPTER" "$EXPECTED_ROWS" "$EXPECTED_STEPS" <<'PY'
import json
import sys
from pathlib import Path

adapter = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
expected_steps = int(sys.argv[3])
manifest = json.loads((adapter / "run_manifest.json").read_text(encoding="utf-8"))
assert manifest["data_len"] == expected_rows, manifest
assert manifest["max_steps"] == expected_steps, manifest
assert (adapter / "adapter_model.safetensors").is_file()
assert (adapter / "adapter_config.json").is_file()
print(json.dumps({"adapter": str(adapter), "data_len": expected_rows, "max_steps": expected_steps}))
PY
log "training complete adapter=$ADAPTER steps=$EXPECTED_STEPS"
