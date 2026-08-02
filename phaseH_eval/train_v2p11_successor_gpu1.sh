#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU_INDEX="${GPU_INDEX:-1}"
DATA="${DATA:-data/teacher_train_mix_v2p11_submitfix1262}"
BASE_DATA="${BASE_DATA:-data/teacher_train_mix_v2p10}"
SOURCE_DATA="${SOURCE_DATA:-data/teacher_train_mix_v2p11_fable1262}"
ADAPTER="${ADAPTER:-adapters/teacher_sft_v2p11_submitfix1262_bf16}"
CONTEXT_AUDIT="${CONTEXT_AUDIT:-runs/v2p11_submitfix1262_context_audit.json}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/train_v2p11_submitfix1262_gpu1.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/tmp/ram_watchdog_v2p11_submitfix1262_gpu1.log}"
RAM_FLOOR_GIB="${RAM_FLOOR_GIB:-12}"
START_FLOOR_GIB="${START_FLOOR_GIB:-24}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
EXPECTED_MANIFEST_SHA256="36a7f3b13ad1dbd89fa5a02af92cdfe9f8581cedf54ea984b73d5749ec52c223"
EXPECTED_TRAIN_SHA256="04530c4a9237c3af88dcfe1bcce72a3de8ad3982ad2965f96ef436c0c87f254a"
EXPECTED_SOURCE_MANIFEST_SHA256="40531f44c8d5ab1d47a179418aa1adaf1ca31d0f0265f262b15c5fd424b4758a"
EXPECTED_SOURCE_TRAIN_SHA256="3d131c531ca060ad2472304b95b423f292cefde94cf4538788ef52ce36b919c1"
EXPECTED_ROWS=1262
EXPECTED_STEPS=79

log() {
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $*" |
    tee -a "$TRAIN_LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

[[ "$GPU_INDEX" == "1" ]] || halt "GPU_INDEX must be 1"
[[ "$PREFLIGHT_ONLY" == "0" || "$PREFLIGHT_ONLY" == "1" ]] ||
  halt "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "$DATA" && -d "$BASE_DATA" && -d "$SOURCE_DATA" ]] ||
  halt "training dataset is missing"
[[ ! -e "$ADAPTER" ]] || halt "adapter output already exists: $ADAPTER"
[[ ! -e "$CONTEXT_AUDIT" ]] || halt "context audit already exists: $CONTEXT_AUDIT"
command -v choom >/dev/null || halt "choom is required"

available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= START_FLOOR_GIB * 1024 * 1024 )) ||
  halt "MemAvailable is below ${START_FLOOR_GIB} GiB"
foreign_compute_pids="$(
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
    --format=csv,noheader,nounits |
    awk '{$1=$1; if ($1 ~ /^[0-9]+$/) print $1}'
)"
[[ -z "$foreign_compute_pids" ]] ||
  halt "GPU${GPU_INDEX} has pre-existing compute PIDs: $foreign_compute_pids"
log "preflight resources passed gpu=$GPU_INDEX available_kib=$available_kib"

.venv-train/bin/python - \
  "$DATA" "$BASE_DATA" "$SOURCE_DATA" "$CONTEXT_AUDIT" \
  "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_TRAIN_SHA256" \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" "$EXPECTED_SOURCE_TRAIN_SHA256" <<'PY'
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from phaseD_sft.train_rust_lora import assert_training_dataset_admitted
from transformers import AutoTokenizer

data = Path(sys.argv[1])
base_data = Path(sys.argv[2])
source_data = Path(sys.argv[3])
out = Path(sys.argv[4])
expected_manifest_sha = sys.argv[5]
expected_train_sha = sys.argv[6]
expected_source_manifest_sha = sys.argv[7]
expected_source_train_sha = sys.argv[8]
manifest_path = data / "manifest.json"
train_path = data / "train.jsonl"
source_manifest_path = source_data / "manifest.json"
source_train_path = source_data / "train.jsonl"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]

assert sha(manifest_path) == expected_manifest_sha
assert sha(train_path) == expected_train_sha
assert sha(source_manifest_path) == expected_source_manifest_sha
assert sha(source_train_path) == expected_source_train_sha
manifest = json.loads(manifest_path.read_text())
rows = load_rows(train_path)
base_rows = load_rows(base_data / "train.jsonl")
source_rows = load_rows(source_train_path)
assert len(base_rows) == 1211
assert len(rows) == len(source_rows) == 1262
assert rows[:1211] == base_rows
assert rows[:1211] == source_rows[:1211]
assert rows[1247:] == source_rows[1247:]
assert manifest["dataset_variant"] == "teacher_train_mix_v2p11_submitfix_v1"
assert manifest["selection_mode"] == "submit_fix"
assert manifest["base_rows"] == 1211
assert manifest["terminal_repairs"] == 36
assert manifest["unchanged_late_rows"] == 15
assert manifest["rendered"] == manifest["training_admitted"] == 1262
assert manifest["all_training_gates_complete"] is True
assert manifest["evaluation_overlap"] == 0
assert manifest["mutation_sequence_unchanged"] is True
assert manifest["source"]["manifest"]["sha256"] == expected_source_manifest_sha
assert manifest["source"]["train"]["sha256"] == expected_source_train_sha
gate = manifest["standard_native_format_loss_gate"]
assert gate["status"] == "passed"
assert gate["failure_count"] == 0
assert gate["samples"] == 1262

submit_command = (
    "git diff -- . > patch.txt && test -s patch.txt && echo "
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
)
for index in range(1211, 1247):
    assert rows[index]["messages"][:-1] == source_rows[index]["messages"][:-1]
    terminal = rows[index]["messages"][-1]
    assert terminal["role"] == "assistant"
    assert terminal["content"] == ""
    assert terminal["loss"] is True
    assert len(terminal["tool_calls"]) == 1
    function = terminal["tool_calls"][0]["function"]
    assert function.get("name") == "bash"
    arguments = json.loads(function["arguments"])
    assert arguments == {"command": submit_command}

added = rows[1211:]
source_counts = Counter(row.get("source") for row in added)
assert source_counts == {
    "teacher:claude:claude-fable-5": 36,
    "fable5_verified_finalpatch": 15,
}
assert len({row["instance_id"] for row in rows}) == len(rows)
for row in rows:
    assert row["messages"]
    assert all(type(message.get("loss")) is bool for message in row["messages"])
assert_training_dataset_admitted(str(data))

tokenizer = AutoTokenizer.from_pretrained(
    "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it"
)
lengths = []
for index, row in enumerate(rows):
    encoded = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    assert isinstance(token_ids, list) and all(type(token) is int for token in token_ids), index
    lengths.append(len(token_ids))
assert max(lengths) <= 32768

artifact = {
    "schema_version": 1,
    "artifact_type": "v2p11_submitfix1262_context_audit",
    "status": "complete",
    "dataset_manifest_sha256": sha(manifest_path),
    "train_jsonl_sha256": sha(train_path),
    "source_manifest_sha256": sha(source_manifest_path),
    "source_train_jsonl_sha256": sha(source_train_path),
    "base_train_jsonl_sha256": sha(base_data / "train.jsonl"),
    "rows": len(rows),
    "base_rows": len(base_rows),
    "terminal_repairs": 36,
    "unchanged_late_rows": 15,
    "source_counts": dict(sorted(source_counts.items())),
    "max_seq": 32768,
    "max_rendered_tokens": max(lengths),
    "over_limit_rows": sum(length > 32768 for length in lengths),
    "optimizer_steps": 79,
    "init_adapter": None,
}
out.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(artifact, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(artifact, sort_keys=True))
PY
log "dataset/context audit passed rows=$EXPECTED_ROWS optimizer_steps=$EXPECTED_STEPS"

if (( PREFLIGHT_ONLY == 1 )); then
  log "preflight-only complete"
  exit 0
fi

set +e
env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  UNSLOTH_COMPILE_DISABLE=0 \
  UNSLOTH_DISABLE_DOUBLE_BUFFER=1 \
  TORCHDYNAMO_DISABLE=0 \
  TORCH_COMPILE_DISABLE=0 \
  TORCHINDUCTOR_COMPILE_THREADS=4 \
  .venv-train/bin/python phaseD_sft/train_rust_lora.py \
    --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
    --data "$DATA" \
    --out "$ADAPTER" \
    --rank 32 --alpha 32 --lr 2e-5 \
    --epochs 1 --bsz 1 --grad-accum 16 \
    --max-seq 32768 --warmup-steps 8 \
    --max-steps "$EXPECTED_STEPS" \
    --gradient-checkpointing bounded_unsloth \
    --logging-steps 5 --save-steps 5 --save-total-limit 3 \
    >> "$TRAIN_LOG" 2>&1 &
trainer_pid="$!"
if ! choom -n 750 -p "$trainer_pid"; then
  kill -TERM "$trainer_pid" 2>/dev/null
  wait "$trainer_pid" 2>/dev/null
  set -e
  halt "failed to apply OOM priority to trainer pid=$trainer_pid"
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

if ! (( train_status == 0 && watchdog_status == 0 )); then
  halt "training failed train_exit=$train_status watchdog_exit=$watchdog_status trainer_pid=$trainer_pid"
fi

.venv-train/bin/python - "$ADAPTER" "$DATA" "$EXPECTED_ROWS" "$EXPECTED_STEPS" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

adapter = Path(sys.argv[1])
data = sys.argv[2]
expected_rows = int(sys.argv[3])
expected_steps = int(sys.argv[4])
manifest = json.loads((adapter / "run_manifest.json").read_text())
expected = {
    "base": "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it",
    "data": data,
    "data_len": expected_rows,
    "init_adapter": None,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-5,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "max_steps": expected_steps,
    "warmup_steps": 8,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}
for key, value in expected.items():
    assert manifest.get(key) == value, (key, manifest.get(key), value)
with safe_open(adapter / "adapter_model.safetensors", framework="pt") as weights:
    keys = list(weights.keys())
assert sum(".lora_A." in key for key in keys) == 410
assert sum(".lora_B." in key for key in keys) == 410
state = json.loads((adapter / f"checkpoint-{expected_steps}" / "trainer_state.json").read_text())
assert state["global_step"] == state["max_steps"] == expected_steps
print(json.dumps({"adapter": str(adapter), "rows": expected_rows, "steps": expected_steps}))
PY
log "training complete adapter=$ADAPTER steps=$EXPECTED_STEPS trainer_pid=$trainer_pid"
