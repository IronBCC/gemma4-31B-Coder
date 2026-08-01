#!/usr/bin/env bash
# Compare v2.10/v2.11 through unmodified Mini-SWE model/environment classes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EVAL_PY="${EVAL_PY:-$ROOT/.venv-eval/bin/python}"
VLLM="${VLLM:-/home/ironbcc/projects/llm/vllm/vllm_env/bin/vllm}"
PORT="${PORT:-8013}"
GPU_INDEX="${GPU_INDEX:-1}"
IDS="${IDS:-data/v2p11_portability_verified10_ids.json}"
VERIFIED_EXCLUSIONS="${VERIFIED_EXCLUSIONS:-data/swe_verified_eval_exclusions_v1.json}"
LITE_IDS="${LITE_IDS:-data/swebench_lite_test_ids.json}"
POSTTRAIN_MARKER="${POSTTRAIN_MARKER:-runs/v2p11_posttrain_complete.json}"
RECOVERY_MARKER="${RECOVERY_MARKER:-runs/v2p11_recovery_sft_complete.json}"
KTO_MARKER="${KTO_MARKER:-runs/v2p11_kto_complete.json}"
FINAL_MERGE_MARKER="${FINAL_MERGE_MARKER:-runs/v2p11_final_merge_complete.json}"
FINAL_AUDIT="${FINAL_AUDIT:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11_full/v2p11_final_merge_audit.json}"
INTERPOLATION_MANIFEST="${INTERPOLATION_MANIFEST:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r4_blend25/interpolation_manifest.json}"
INTERPOLATION_ANCHOR_MODEL="${INTERPOLATION_ANCHOR_MODEL:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full}"
INTERPOLATION_SOURCE_MODEL="${INTERPOLATION_SOURCE_MODEL:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full}"
CONTROL_NAME="${CONTROL_NAME:-teacher_sft_v2p10}"
CONTROL_MODEL="${CONTROL_MODEL:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full}"
CANDIDATE_NAME="${CANDIDATE_NAME:-teacher_sft_v2p11}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11_full}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-runs/v2p11_portability_stock}"
CONTROL_ROOT="${CONTROL_ROOT:-$ARTIFACT_ROOT/v2p10}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$ARTIFACT_ROOT/v2p11}"
GATE="${GATE:-runs/v2p11_portability_gate.json}"
LOG="${LOG:-/tmp/run_v2p11_portability_gate.log}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
DOCKER_FREE_FLOOR_GIB="${DOCKER_FREE_FLOOR_GIB:-50}"
PORTABILITY_MODE="${PORTABILITY_MODE:-full}"
LINEAGE_MODE="${LINEAGE_MODE:-posttrain}"
STOCK_CONFIG="${STOCK_CONFIG:-}"
SERVE_STOP_WAIT_LOOPS="${SERVE_STOP_WAIT_LOOPS:-90}"
SERVE_STOP_WAIT_SECONDS="${SERVE_STOP_WAIT_SECONDS:-1}"
MEMORY_ADMISSION_GIB="${MEMORY_ADMISSION_GIB:-40}"
MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"
RAM_WATCHDOG_PY="${RAM_WATCHDOG_PY:-$ROOT/.venv-train/bin/python}"
owned_serve_pid=""
memory_watchdog_pid=""

log() {
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $*" |
    tee -a "$LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

gpu1_uuid() {
  nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader |
    awk '{$1=$1; print}'
}

gpu1_compute_pids() {
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
    --format=csv,noheader,nounits |
    awk '{$1=$1; if ($1 ~ /^[0-9]+$/) print $1}'
}

wait_for_gpu1_idle() {
  local uuid pids
  uuid="$(gpu1_uuid)"
  [[ -n "$uuid" ]] || halt "could not resolve GPU1 UUID"
  while true; do
    pids="$(gpu1_compute_pids "$uuid")"
    if [[ -z "$pids" ]] &&
      ! ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
      return
    fi
    log "waiting for GPU1/port $PORT; exact GPU PIDs=$(tr '\n' ',' <<<"$pids")"
    sleep "$WAIT_SECONDS"
  done
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
    sleep "$WAIT_SECONDS"
  done
}

start_memory_watchdog() {
  local label="$1"
  [[ -x "$RAM_WATCHDOG_PY" ]] ||
    halt "RAM watchdog Python is missing"
  [[ "$owned_serve_pid" =~ ^[1-9][0-9]*$ ]] ||
    halt "RAM watchdog requires an exact owned serve PID"
  "$RAM_WATCHDOG_PY" phaseD_sft/ram_watchdog.py \
    --pid "$owned_serve_pid" \
    --min-available-gib "$MEMORY_WATCHDOG_GIB" \
    --interval-seconds 5 \
    --log-path "/tmp/ram_watchdog_${label}_portability.log" \
    >> "$LOG" 2>&1 &
  memory_watchdog_pid="$!"
  log "RAM watchdog started pid=$memory_watchdog_pid target=$owned_serve_pid"
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
  pid="$owned_serve_pid"
  [[ -n "$pid" ]] || return
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
  owned_serve_pid=""
  finish_memory_watchdog || watchdog_status=$?
  wait_for_gpu1_idle
  (( watchdog_status == 0 ))
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if ! stop_owned_serve; then
    log "HALT: owned serve PID $owned_serve_pid did not stop"
    status=1
  fi
  exit "$status"
}

resolve_stock_config() {
  if [[ -n "$STOCK_CONFIG" ]]; then
    return
  fi
  STOCK_CONFIG="$(
    "$EVAL_PY" -c \
      'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("minisweagent").origin).parent)' |
      tail -n 1
  )/config/benchmarks/swebench.yaml"
}

configure_runtime_path() {
  local vllm_bin
  vllm_bin="$(dirname "$VLLM")"
  case ":$PATH:" in
    *":$vllm_bin:"*) ;;
    *) export PATH="$vllm_bin:$PATH" ;;
  esac
}

validate_posttrain_candidate() {
  "$EVAL_PY" phaseH_eval/v2p11_posttrain_lineage.py \
    --posttrain-marker "$POSTTRAIN_MARKER" \
    --recovery-marker "$RECOVERY_MARKER" \
    --kto-marker "$KTO_MARKER" \
    --final-merge-marker "$FINAL_MERGE_MARKER" \
    --final-model "$CANDIDATE_MODEL" \
    --final-audit "$FINAL_AUDIT"
}

validate_direct_lora_candidate() {
  "$EVAL_PY" - "$CANDIDATE_NAME" "$CANDIDATE_MODEL" "$FINAL_AUDIT" <<'PY'
import json
from pathlib import Path
import sys

from phaseH_eval.v2p11_completion_provenance import _model_contract

name, model_value, audit_value = sys.argv[1:]
model = Path(model_value).resolve()
audit_path = Path(audit_value).resolve()
contract = _model_contract(model)
contract["served_name"] = name
assert contract["model_path"] == str(model)
assert audit_path.parent == model
assert json.loads(audit_path.read_text()) == {
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

PY
}

validate_interpolation_candidate() {
  "$EVAL_PY" phaseH_eval/v2p11_interpolation_lineage.py \
    --manifest "$INTERPOLATION_MANIFEST" \
    --anchor-model "$INTERPOLATION_ANCHOR_MODEL" \
    --source-model "$INTERPOLATION_SOURCE_MODEL" \
    --output-model "$CANDIDATE_MODEL" \
    --candidate-name "$CANDIDATE_NAME"
}

start_owned_serve() {
  local name="$1" model="$2" serve_log="$3" served=""
  wait_for_gpu1_idle
  wait_for_memory_admission
  log "starting stock-harness serve name=$name model=$model"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" env -u HF_TOKEN "$VLLM" serve "$model" \
    --dtype bfloat16 --served-model-name "$name" --port "$PORT" \
    --max-model-len 250000 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --max-num-seqs 64 \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template phaseH_eval/tool_chat_template_gemma4_thinkopen_v2.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    > "$serve_log" 2>&1 &
  owned_serve_pid="$!"
  start_memory_watchdog "$name"
  for _ in $(seq 1 180); do
    if ! kill -0 "$owned_serve_pid" 2>/dev/null; then
      wait "$owned_serve_pid" || status=$?
      halt "serve exited before readiness status=${status:-1}: $serve_log"
    fi
    served="$(
      curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
        "$EVAL_PY" -c \
          'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' \
          2>/dev/null || true
    )"
    [[ "$served" == "$name" ]] && break
    sleep 10
  done
  [[ "$served" == "$name" ]] ||
    halt "serve readiness timed out or model identity differed"
  log "serve ready name=$name pid=$owned_serve_pid"
}

ensure_eval_manifest() {
  local name="$1" model="$2" run_root="$3"
  "$EVAL_PY" - "$name" "$model" "$run_root" "$IDS" "$STOCK_CONFIG" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import yaml

name, model_text, run_text, ids_text, config_text = sys.argv[1:]
model = Path(model_text)
run_root = Path(run_text)
ids = Path(ids_text)
stock_config = Path(config_text)
run_root.mkdir(parents=True, exist_ok=True)

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

config = yaml.safe_load(stock_config.read_text())
assert config["environment"]["environment_class"] == "docker"
assert config["agent"]["step_limit"] == 250
index = model / "model.safetensors.index.json"
single = model / "model.safetensors"
if index.is_file():
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted({model / value for value in weight_map.values()})
    weight_contract = {"model_index_sha256": sha(index)}
elif single.is_file():
    shards = [single]
    weight_contract = {"model_safetensors_sha256": sha(single)}
else:
    raise SystemExit("model weights are missing")
artifacts = [
    {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }
    for path in shards
]
payload = {
    "schema_version": 1,
    "run_id": run_root.name,
    "served_name": name,
    "model_path": str(model.resolve()),
    "model_config_sha256": sha(model / "config.json"),
    **weight_contract,
    "model_artifacts": artifacts,
    "ids_sha256": sha(ids),
    "harness_contract": {
        "mini_swe_agent_version": importlib.metadata.version(
            "mini-swe-agent"
        ),
        "stock_config_sha256": sha(stock_config),
        "environment_class": "docker",
        "model_class": (
            "minisweagent.models.litellm_model.LitellmModel"
        ),
        "temperature": 0.7,
        "seed": 1,
        "max_tokens": 4096,
        "workers": 4,
        "step_limit": 250,
        "subset": "verified",
        "ids_sha256": sha(ids),
    },
}
path = run_root / "eval_manifest.json"
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if path.exists():
    assert path.read_text() == rendered, "eval manifest changed"
else:
    descriptor, temporary = tempfile.mkstemp(
        prefix=".eval_manifest.", dir=run_root
    )
    with os.fdopen(descriptor, "w") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
PY
}

run_tool_canary() {
  local name="$1" run_root="$2"
  "$EVAL_PY" - \
    "$name" \
    "$run_root/tool_canary.json" \
    "$run_root/eval_manifest.json" \
    "$PORT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from openai import OpenAI

name = sys.argv[1]
path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
port = int(sys.argv[4])
response = OpenAI(
    base_url=f"http://127.0.0.1:{port}/v1",
    api_key="dummy",
).chat.completions.create(
    model=name,
    messages=[{
        "role": "user",
        "content": "Use the bash tool to run pwd. Do not answer directly.",
    }],
    tools=[BASH_TOOL],
    temperature=0,
    max_tokens=1024,
)
data = response.model_dump(mode="json")
choice = data["choices"][0]
calls = choice["message"].get("tool_calls") or []
assert calls and calls[0]["function"]["name"] == "bash", data
arguments = json.loads(calls[0]["function"]["arguments"])
command = arguments["command"]
assert isinstance(command, str) and command.strip()
payload = {
    "schema_version": 1,
    "model": name,
    "tool_name": "bash",
    "command": command,
    "finish_reason": choice["finish_reason"],
    "system_fingerprint": data.get("system_fingerprint"),
    "eval_manifest_sha256": hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest(),
}
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
descriptor, temporary = tempfile.mkstemp(
    prefix=".tool_canary.", dir=path.parent
)
with os.fdopen(descriptor, "w") as handle:
    handle.write(rendered)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

official_report_binding_current() {
  local run_root="$1"
  "$EVAL_PY" - "$run_root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = root / "eval_manifest.json"
predictions = root / "preds.json"
report = root / "official_report.json"
binding = root / "official_report_binding.json"
if not all(path.is_file() for path in [manifest, predictions, report, binding]):
    raise SystemExit(1)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    value = json.loads(binding.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if (
    value.get("schema_version") != 1
    or value.get("eval_manifest_sha256") != sha(manifest)
    or value.get("predictions_sha256") != sha(predictions)
    or value.get("official_report_sha256") != sha(report)
):
    raise SystemExit(1)
PY
}

write_official_report_binding() {
  local run_root="$1"
  "$EVAL_PY" - "$run_root" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

payload = {
    "schema_version": 1,
    "eval_manifest_sha256": sha(root / "eval_manifest.json"),
    "predictions_sha256": sha(root / "preds.json"),
    "official_report_sha256": sha(root / "official_report.json"),
}
path = root / "official_report_binding.json"
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
descriptor, temporary = tempfile.mkstemp(
    prefix=".official_report_binding.", dir=root
)
with os.fdopen(descriptor, "w") as handle:
    handle.write(rendered)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

score_run() {
  local name="$1" run_root="$2"
  local predictions_sha run_id
  if official_report_binding_current "$run_root"; then
    return
  fi
  predictions_sha="$(
    "$EVAL_PY" -c \
      'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
      "$run_root/preds.json"
  )"
  run_id="portability_${name}_${predictions_sha}"
  local expected="$ROOT/openai__${name}.${run_id}.json"
  (
    cd "$ROOT"
    "$EVAL_PY" -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Verified \
      --predictions_path "$run_root/preds.json" \
      --run_id "$run_id" \
      --report_dir "$run_root/report"
  ) >> "$run_root/score.log" 2>&1
  [[ -f "$expected" ]] ||
    halt "official report is missing after scoring: $expected"
  cp "$expected" "$run_root/official_report.json"
  write_official_report_binding "$run_root"
}

run_stock_harness() {
  local name="$1" model="$2" run_root="$3"
  local filter
  ensure_eval_manifest "$name" "$model" "$run_root"
  start_owned_serve "$name" "$model" "/tmp/serve_${name}_portability.log"
  run_tool_canary "$name" "$run_root"
  if [[ ! -f "$run_root/preds.json" ]] ||
    [[ "$(jq -er length "$run_root/preds.json" 2>/dev/null || true)" != "10" ]]; then
    filter="$(
      "$EVAL_PY" - "$IDS" <<'PY'
import json
import re
import sys
ids = json.load(open(sys.argv[1]))
print("^(?:" + "|".join(re.escape(value) for value in ids) + ")$")
PY
    )"
    log "running stock Mini-SWE holdout name=$name"
    OPENAI_API_KEY=dummy \
    MSWEA_COST_TRACKING=ignore_errors \
      "$EVAL_PY" phaseH_eval/mini_extra_selfretry.py swebench \
        --subset verified --split test --filter "$filter" \
        --workers 4 \
        --model-class minisweagent.models.litellm_model.LitellmModel \
        -m "openai/$name" \
        -c "$STOCK_CONFIG" \
        -c model.model_kwargs.api_base="http://127.0.0.1:$PORT/v1" \
        -c model.model_kwargs.api_key=dummy \
        -c model.model_kwargs.temperature=0.7 \
        -c model.model_kwargs.seed=1 \
        -c model.model_kwargs.max_tokens=4096 \
        -c model.cost_tracking=ignore_errors \
        -o "$run_root" \
        >> "$run_root/generation.log" 2>&1
  fi
  stop_owned_serve ||
    halt "owned serve PID $owned_serve_pid did not stop"
  score_run "$name" "$run_root"
}

validate_stock_run() {
  local run_root="$1" name="$2" model="$3"
  "$EVAL_PY" - "$IDS" "$run_root" "$name" "$model" <<'PY'
import json
from pathlib import Path
import sys

from phaseH_eval.v2p11_portability_gate import _validate_run

ids, run_root, name, model = sys.argv[1:]
result = _validate_run(
    ids_path=Path(ids),
    run_root=Path(run_root),
    expected_name=name,
)
if result["model_contract"]["model_path"] != str(Path(model).resolve()):
    raise ValueError(f"{name} model path differs from requested model")
print(json.dumps({
    "name": result["name"],
    "resolved": result["resolved"],
    "empty": result["empty"],
    "format_errors": result["format_errors"],
}, sort_keys=True))
PY
}

stock_run_complete() {
  local name="$1" model="$2" run_root="$3"
  validate_stock_run "$run_root" "$name" "$model" >/dev/null 2>&1
}

run_or_reuse_stock_harness() {
  local name="$1" model="$2" run_root="$3"
  if stock_run_complete "$name" "$model" "$run_root"; then
    log "reusing verified stock Mini-SWE holdout name=$name root=$run_root"
    return
  fi
  run_stock_harness "$name" "$model" "$run_root"
  validate_stock_run "$run_root" "$name" "$model"
}

prepull_holdout_images() {
  local docker_root free_gib image
  while IFS= read -r image; do
    [[ -n "$image" ]] || continue
    if docker image inspect "$image" >/dev/null 2>&1; then
      continue
    fi
    docker_root="$(docker info --format '{{.DockerRootDir}}')"
    free_gib="$(df -BG --output=avail "$docker_root" | tail -n 1 | tr -dc '0-9')"
    (( free_gib >= DOCKER_FREE_FLOOR_GIB )) ||
      halt "disk admission refused before pull: free=${free_gib}GiB image=$image"
    log "pre-pulling portability image=$image free=${free_gib}GiB"
    docker pull "$image" >> "$LOG" 2>&1
  done < <(
    "$EVAL_PY" - "$IDS" <<'PY'
import json
import sys
from datasets import load_dataset
ids = set(json.load(open(sys.argv[1])))
rows = load_dataset(
    "princeton-nlp/SWE-bench_Verified",
    split="test",
)
found = {
    row["instance_id"]: (
        "docker.io/swebench/sweb.eval.x86_64."
        + row["instance_id"].replace("__", "_1776_")
        + ":latest"
    ).lower()
    for row in rows
    if row["instance_id"] in ids
}
assert set(found) == ids
for instance_id in sorted(ids):
    print(found[instance_id])
PY
  )
}

main() {
  cd "$ROOT"
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  configure_runtime_path
  [[ "$GPU_INDEX" == "1" ]] ||
    halt "GPU_INDEX is fixed to GPU1; GPU0 is unavailable"
  case "$PORTABILITY_MODE" in
    control-only|full) ;;
    *) halt "PORTABILITY_MODE must be control-only or full" ;;
  esac
  case "$LINEAGE_MODE" in
    posttrain|direct_lora|interpolation) ;;
    *) halt "LINEAGE_MODE must be posttrain, direct_lora, or interpolation" ;;
  esac
  [[ "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    halt "WAIT_SECONDS must be a positive integer"
  [[ "$DOCKER_FREE_FLOOR_GIB" =~ ^[1-9][0-9]*$ ]] ||
    halt "DOCKER_FREE_FLOOR_GIB must be a positive integer"
  [[ "$MEMORY_ADMISSION_GIB" =~ ^[1-9][0-9]*$ ]] ||
    halt "MEMORY_ADMISSION_GIB must be a positive integer"
  [[ "$MEMORY_WATCHDOG_GIB" =~ ^[1-9][0-9]*$ ]] ||
    halt "MEMORY_WATCHDOG_GIB must be a positive integer"
  [[ -x "$EVAL_PY" ]] || halt "evaluation Python is missing"
  [[ -x "$VLLM" ]] || halt "vLLM executable is missing"
  resolve_stock_config
  [[ -f "$STOCK_CONFIG" ]] || halt "stock Mini-SWE config is missing"
  grep -q 'environment_class: docker' "$STOCK_CONFIG" ||
    halt "stock Mini-SWE config does not use the docker environment"
  [[ -d "$CONTROL_MODEL" ]] ||
    halt "v2.10 portability control model directory is missing"
  [[ "$(jq -er length "$IDS")" == "10" ]] ||
    halt "portability ID artifact must contain 10 rows"

  if [[ "$PORTABILITY_MODE" == "full" ]]; then
    [[ -d "$CANDIDATE_MODEL" ]] ||
      halt "v2.11 portability candidate model directory is missing"
    case "$LINEAGE_MODE" in
      posttrain)
        [[ -f "$FINAL_AUDIT" ]] ||
          halt "v2.11 final merge audit is missing"
        [[ -f "$POSTTRAIN_MARKER" ]] ||
          halt "v2.11 posttrain marker is missing"
        validate_posttrain_candidate
        ;;
      direct_lora)
        [[ -f "$FINAL_AUDIT" ]] ||
          halt "v2.11 final merge audit is missing"
        validate_direct_lora_candidate
        ;;
      interpolation)
        [[ -f "$INTERPOLATION_MANIFEST" ]] ||
          halt "v2.11 interpolation manifest is missing"
        [[ -d "$INTERPOLATION_ANCHOR_MODEL" ]] ||
          halt "v2.11 interpolation anchor model is missing"
        [[ -d "$INTERPOLATION_SOURCE_MODEL" ]] ||
          halt "v2.11 interpolation source model is missing"
        validate_interpolation_candidate
        ;;
    esac
  fi

  trap cleanup EXIT
  mkdir -p "$CONTROL_ROOT"
  prepull_holdout_images
  run_or_reuse_stock_harness \
    "$CONTROL_NAME" "$CONTROL_MODEL" "$CONTROL_ROOT"
  if [[ "$PORTABILITY_MODE" == "control-only" ]]; then
    validate_stock_run "$CONTROL_ROOT" "$CONTROL_NAME" "$CONTROL_MODEL"
    wait_for_gpu1_idle
    trap - EXIT
    log "V2P10 PORTABILITY CONTROL COMPLETE root=$CONTROL_ROOT"
    return
  fi

  mkdir -p "$CANDIDATE_ROOT"
  run_or_reuse_stock_harness \
    "$CANDIDATE_NAME" "$CANDIDATE_MODEL" "$CANDIDATE_ROOT"
  "$EVAL_PY" phaseH_eval/v2p11_portability_gate.py \
    --ids "$IDS" \
    --verified-exclusions "$VERIFIED_EXCLUSIONS" \
    --lite-ids "$LITE_IDS" \
    --control-root "$CONTROL_ROOT" \
    --candidate-root "$CANDIDATE_ROOT" \
    --candidate-name "$CANDIDATE_NAME" \
    --out "$GATE"
  wait_for_gpu1_idle
  trap - EXIT
  log "$CANDIDATE_NAME PORTABILITY GATE COMPLETE artifact=$GATE"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
