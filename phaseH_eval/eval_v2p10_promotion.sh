#!/usr/bin/env bash
# Matched non-Lite -> fixed150 -> conditional full300 promotion lane for v2.10.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODELS="/media/ironbcc/CrucialX10/models"
VLLM="/home/ironbcc/projects/llm/vllm/vllm_env/bin/vllm"
VLLM_BIN="${VLLM%/*}"
PORT=8013
PROD_PORTS=(8000 8101 8103 8104)
TRAIN_MARKER="runs/v2p10_train_merge_complete.json"
OLD_HARD_TASKS="data/v2p10_nonlite_hard30/tasks.jsonl"
HARD_DIR="data/v2p10_nonlite_hard30_v2"
HARD_TASKS="data/v2p10_nonlite_hard30_v2/tasks.jsonl"
HARD_MANIFEST="data/v2p10_nonlite_hard30_v2/manifest.json"
PANEL_IDS="data/teacher_sft_fixed150_ids.json"
FULL_IDS="data/swebench_lite_test_ids.json"
PANEL_SHA="abc550841d4a64740da2a9f18d717973a501c4bf138755d3d55883c67d9c1390"
FULL_SHA="b98fc2b1054dc8fdfcb94f083f43454fd568961a0b3dbf8c388c210b7b868e14"
PROMOTION_ROOT="runs/v2p10_promotion"
LOG="${LOG:-/tmp/eval_v2p10_promotion.log}"
EVAL_PY="$ROOT/.venv-eval/bin/python"
SELF_PID="$$"
SELF_START_TICKS="$(awk '{print $22}' /proc/$$/stat)"
owned_serve_pid=""

export PATH="$VLLM_BIN:$PATH"

log() {
  echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if [[ -n "$owned_serve_pid" ]] && kill -0 "$owned_serve_pid" 2>/dev/null; then
    log "cleanup: stopping exact owned serve pid=$owned_serve_pid"
    kill "$owned_serve_pid" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$owned_serve_pid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$owned_serve_pid" 2>/dev/null; then
      kill -9 "$owned_serve_pid" 2>/dev/null
    fi
  fi
  exit "$status"
}
trap cleanup EXIT

check_prod_health() {
  local port
  for port in "${PROD_PORTS[@]}"; do
    curl -fsS --max-time 10 "http://127.0.0.1:$port/health" >/dev/null ||
      halt "GPU0 production health failed on port $port"
  done
  log "GPU0 production health: ports ${PROD_PORTS[*]} all green"
}

gpu1_uuid() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    awk -F, '$1 + 0 == 1 {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      print $2
    }'
}

gpu1_compute_pids() {
  local uuid="$1"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
    awk -F, -v target="$uuid" '
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if ($1 == target) print $2
      }
    '
}

assert_gpu1_and_port_free() {
  local uuid pids
  if ss -ltnp "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    halt "port $PORT has a pre-existing listener"
  fi
  uuid="$(gpu1_uuid)"
  [[ -n "$uuid" ]] || halt "could not resolve GPU1 UUID"
  pids="$(gpu1_compute_pids "$uuid")"
  [[ -z "$pids" ]] || halt "GPU1 has foreign compute PIDs: $pids"
}

served_name() {
  curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' \
      2>/dev/null
}

start_owned_serve() {
  local name="$1" model="$2" serve_log args
  [[ -d "$model" ]] || halt "model directory is missing: $model"
  check_prod_health
  assert_gpu1_and_port_free
  serve_log="/tmp/serve_${name}_v2p10_promotion.log"
  log "starting owned GPU1 serve $name <- $model"
  CUDA_VISIBLE_DEVICES=1 env -u HF_TOKEN "$VLLM" serve "$model" \
    --dtype bfloat16 --served-model-name "$name" --port "$PORT" \
    --max-model-len 250000 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --max-num-seqs 64 \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template phaseH_eval/tool_chat_template_gemma4_thinkopen_v2.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    > "$serve_log" 2>&1 &
  owned_serve_pid="$!"
  for _ in $(seq 1 180); do
    kill -0 "$owned_serve_pid" 2>/dev/null ||
      halt "owned serve pid=$owned_serve_pid exited; see $serve_log"
    if [[ "$(served_name)" == "$name" ]]; then
      args="$(ps -p "$owned_serve_pid" -o args= 2>/dev/null || true)"
      [[ "$args" == *"$model"* && "$args" == *"--served-model-name $name"* ]] ||
        halt "owned serve identity mismatch for pid=$owned_serve_pid"
      log "serve READY name=$name pid=$owned_serve_pid model=$model"
      return 0
    fi
    sleep 10
  done
  halt "serve $name failed to become ready; see $serve_log"
}

stop_owned_serve() {
  local pid uuid pids
  pid="$owned_serve_pid"
  [[ -n "$pid" ]] || halt "no owned serve PID is recorded"
  log "stopping owned serve pid=$pid"
  kill "$pid"
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid"
  fi
  owned_serve_pid=""
  for _ in $(seq 1 30); do
    uuid="$(gpu1_uuid)"
    pids="$(gpu1_compute_pids "$uuid")"
    if [[ -z "$pids" ]] &&
      ! ss -ltnp "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
      break
    fi
    sleep 2
  done
  assert_gpu1_and_port_free
  check_prod_health
}

verify_merged() {
  local model="$1"
  .venv-train/bin/python - "$model" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
config = json.loads((model / "config.json").read_text())
weight_map = json.loads(
    (model / "model.safetensors.index.json").read_text()
)["weight_map"]
assert config["architectures"] == ["Gemma4ForConditionalGeneration"]
assert len(weight_map) == 1188
assert sum("vision" in key for key in weight_map) == 356
PY
}

verify_train_marker() {
  .venv-train/bin/python - "$TRAIN_MARKER" \
    "data/teacher_train_mix_v2p10/manifest.json" \
    "adapters/teacher_sft_v2p10_bf16/adapter_model.safetensors" \
    "$MODELS/merged/teacher_sft_v2p10_full/v2p10_merge_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path, dataset_path, adapter_path, audit_path = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

marker = json.loads(marker_path.read_text())
audit = json.loads(audit_path.read_text())
assert marker["schema_version"] == 1
assert marker["complete"] is True
assert audit["complete"] is True
assert marker["dataset_manifest_sha256"] == sha256(dataset_path)
assert marker["adapter_sha256"] == sha256(adapter_path)
assert marker["merge_audit_sha256"] == sha256(audit_path)
PY
}

ensure_eval_manifest() {
  local name="$1" model="$2" ids="$3" runid="$4"
  python3 - "$name" "$model" "$ids" "$runid" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

name, model_text, ids_text, runid = sys.argv[1:]
model = Path(model_text)
ids = Path(ids_text)
instance_ids = json.loads(ids.read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

expected = {
    "schema_version": 1,
    "run_id": runid,
    "served_name": name,
    "model_path": str(model),
    "model_config_sha256": sha(model / "config.json"),
    "model_index_sha256": sha(model / "model.safetensors.index.json"),
    "ids_path": str(ids),
    "ids_sha256": sha(ids),
    "instances": len(instance_ids),
    "config": "swebench_edit_first_selfretry_s120.yaml",
    "environment_class": "docker_selfretry.DockerSelfRetryEnv",
    "temperature": 0.7,
    "seed": 1,
    "step_limit": 120,
    "generation_workers": 16,
    "scorer_workers": 16,
    "batch": 20,
    "pull_workers": 5,
}
root = Path("runs") / runid
root.mkdir(parents=True, exist_ok=True)
path = root / "eval_manifest.json"
if path.exists():
    if json.loads(path.read_text()) != expected:
        raise SystemExit(f"{path}: manifest mismatch")
else:
    temporary = root / f".eval_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(expected, indent=2))
    os.replace(temporary, path)
PY
}

nonlite_generation_complete() {
  local name="$1" runid="$2"
  "$EVAL_PY" - "$HARD_TASKS" "runs/$runid/$name" <<'PY'
import json
import sys
from pathlib import Path

tasks_path, model_root = map(Path, sys.argv[1:])
expected = {
    json.loads(line)["instance_id"]
    for line in tasks_path.read_text().splitlines()
    if line.strip()
}
predictions_path = model_root / "preds.json"
if not predictions_path.exists():
    raise SystemExit(1)
predictions = json.loads(predictions_path.read_text())
trajectories = {
    path.parent.name
    for path in model_root.glob("*/*.traj.json")
}
raise SystemExit(0 if set(predictions) == expected == trajectories else 1)
PY
}

score_nonlite() {
  local name="$1" runid="$2" output_var="$3"
  local attempt score_run acceptance latest=""
  for attempt in 0 1 2; do
    if [[ "$attempt" == "0" ]]; then
      score_run="runs/$runid"
    else
      score_run="runs/${runid}_score_retry${attempt}"
    fi
    acceptance="$score_run/acceptance.json"
    if [[ ! -f "$acceptance" ]]; then
      if "$EVAL_PY" phaseH_eval/score_nonlite_hard30.py \
        --tasks "$HARD_TASKS" --task-manifest "$HARD_MANIFEST" \
        --generation-root "runs/$runid" --run-root "$score_run" \
        --name "$name" --workers 4; then
        :
      else
        log "non-Lite score attempt=$attempt incomplete name=$name runid=$runid"
      fi
    fi
    [[ -f "$acceptance" ]] || continue
    latest="$acceptance"
    if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$acceptance")" == "complete" ]]; then
      break
    fi
  done
  [[ -n "$latest" ]] || halt "non-Lite scoring published no acceptance: $runid"
  printf -v "$output_var" '%s' "$latest"
}

run_nonlite() {
  local name="$1" model="$2" runid="$3" output_var="$4" pass
  if ! nonlite_generation_complete "$name" "$runid"; then
    start_owned_serve "$name" "$model"
    for pass in 1 2; do
      log "non-Lite hard30 pass=$pass name=$name runid=$runid"
      NAME="$name" PORT="$PORT" SUBSET="$HARD_TASKS" SPLIT="train" \
        SLICE="0:30" FILTER="" WORKERS="16" TEMPERATURE="0.7" SEED="1" \
        SCORE="0" CONFIG="swebench_edit_first_selfretry_s120.yaml" \
        OUT="$ROOT/runs/$runid" bash phaseH_eval/smoke_single.sh \
        >> "/tmp/${runid}.log" 2>&1
    done
    stop_owned_serve
  fi
  score_nonlite "$name" "$runid" "$output_var"
}

run_fixed_eval() {
  local name="$1" model="$2" ids="$3" runid="$4" pass
  ensure_eval_manifest "$name" "$model" "$ids" "$runid"
  if [[ ! -f "runs/$runid/acceptance.json" ]]; then
    start_owned_serve "$name" "$model"
    for pass in 1 2; do
      log "fixed eval pass=$pass name=$name ids=$ids runid=$runid"
      python3 phaseH_eval/retest_empties.py \
        --name "$name" --port "$PORT" --ids "$ids" --runid "$runid" \
        --batch 20 --workers 16 --pull-workers 5 \
        >> "/tmp/${runid}.log" 2>&1
    done
    stop_owned_serve
    "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py validate-fixed \
      --name "$name" --ids "$ids" --run-root "runs/$runid" \
      --out "runs/$runid/acceptance.json"
  fi
}

decision_bool() {
  local path="$1" key="$2"
  python3 - "$path" "$key" <<'PY'
import json
import sys
value = json.loads(open(sys.argv[1]).read())
print("1" if value.get(sys.argv[2]) is True else "0")
PY
}

[[ -f "$TRAIN_MARKER" ]] || halt "missing guarded train/merge marker: $TRAIN_MARKER"
verify_train_marker
[[ -x "$EVAL_PY" ]] || halt "evaluation Python is missing: $EVAL_PY"
mkdir -p "$PROMOTION_ROOT"
[[ "$(sha256sum "$PANEL_IDS" | awk '{print $1}')" == "$PANEL_SHA" ]] ||
  halt "fixed150 ID SHA mismatch"
[[ "$(sha256sum "$FULL_IDS" | awk '{print $1}')" == "$FULL_SHA" ]] ||
  halt "full300 ID SHA mismatch"
verify_merged "$MODELS/merged/teacher_sft_v2p8_full"
verify_merged "$MODELS/merged/teacher_sft_v2p10_full"
check_prod_health
assert_gpu1_and_port_free

if [[ ! -d "$HARD_DIR" ]]; then
  .venv-train/bin/python phaseH_eval/freeze_nonlite_hard30.py \
    --exclude data/fable5_batch60.jsonl \
    --exclude data/opus5_batch50.jsonl \
    --exclude data/swe_verified_eval_exclusions_v1.json \
    --out "$HARD_DIR" --preflight-workers 4
fi

if [[ ! -d runs/nonlite_hard30_v2p8_retry1 ]]; then
  "$EVAL_PY" phaseH_eval/reuse_nonlite_generation.py \
    --source-run runs/nonlite_hard30_v2p8 \
    --target-run runs/nonlite_hard30_v2p8_retry1 \
    --name teacher_sft_v2p8 \
    --source-tasks "$OLD_HARD_TASKS" --target-tasks "$HARD_TASKS" \
    --expected-reused 27
fi

v2p8_hard_acceptance=""
v2p10_hard_acceptance=""
run_nonlite \
  "teacher_sft_v2p8" "$MODELS/merged/teacher_sft_v2p8_full" \
  "nonlite_hard30_v2p8_retry1" v2p8_hard_acceptance
run_nonlite \
  "teacher_sft_v2p10" "$MODELS/merged/teacher_sft_v2p10_full" \
  "nonlite_hard30_v2p10_retry1" v2p10_hard_acceptance

hard_decision="$PROMOTION_ROOT/hard30_decision_retry1.json"
if [[ ! -f "$hard_decision" ]]; then
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py decide-hard30 \
    --candidate "$v2p10_hard_acceptance" \
    --v2p8-control "$v2p8_hard_acceptance" \
    --out "$hard_decision"
fi
if [[ "$(decision_bool "$hard_decision" run_fixed150)" != "1" ]]; then
  log "PROMOTION STOPPED at non-Lite hard30; no_adapter_selected=true"
  assert_gpu1_and_port_free
  check_prod_health
  trap - EXIT
  exit 0
fi

run_fixed_eval \
  "teacher_sft_v2p10" "$MODELS/merged/teacher_sft_v2p10_full" \
  "$PANEL_IDS" "fixed150_v2p10"

candidate_metrics="$PROMOTION_ROOT/fixed150_v2p10_metrics.json"
base_metrics="$PROMOTION_ROOT/fixed150_base_metrics.json"
v2_metrics="$PROMOTION_ROOT/fixed150_v2_bf16_metrics.json"
v2p8_metrics="$PROMOTION_ROOT/fixed150_v2p8_metrics.json"
v2p9_metrics="$PROMOTION_ROOT/fixed150_v2p9_metrics.json"
paired_base="$PROMOTION_ROOT/fixed150_v2p10_vs_base.json"
paired_v2="$PROMOTION_ROOT/fixed150_v2p10_vs_v2_bf16.json"
paired_v2p8="$PROMOTION_ROOT/fixed150_v2p10_vs_v2p8.json"
paired_v2p9="$PROMOTION_ROOT/fixed150_v2p10_vs_v2p9.json"
[[ -f "$candidate_metrics" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py summarize \
    --run-root runs/fixed150_v2p10 --out "$candidate_metrics"
[[ -f "$base_metrics" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py summarize \
    --run-root runs/fixed150_base --out "$base_metrics"
[[ -f "$v2_metrics" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py summarize \
    --run-root runs/fixed150_v2_bf16 --out "$v2_metrics"
[[ -f "$v2p8_metrics" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py summarize \
    --run-root runs/fixed150_v2p8 --out "$v2p8_metrics"
[[ -f "$v2p9_metrics" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py summarize \
    --run-root runs/fixed150_v2p9 --out "$v2p9_metrics"
[[ -f "$paired_base" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py pair \
    --candidate runs/fixed150_v2p10/acceptance.json \
    --reference runs/fixed150_base/acceptance.json \
    --out "$paired_base"
[[ -f "$paired_v2" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py pair \
    --candidate runs/fixed150_v2p10/acceptance.json \
    --reference runs/fixed150_v2_bf16/acceptance.json \
    --out "$paired_v2"
[[ -f "$paired_v2p8" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py pair \
    --candidate runs/fixed150_v2p10/acceptance.json \
    --reference runs/fixed150_v2p8/acceptance.json \
    --out "$paired_v2p8"
[[ -f "$paired_v2p9" ]] ||
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py pair \
    --candidate runs/fixed150_v2p10/acceptance.json \
    --reference runs/fixed150_v2p9/acceptance.json \
    --out "$paired_v2p9"

fixed_decision="$PROMOTION_ROOT/fixed150_decision.json"
if [[ ! -f "$fixed_decision" ]]; then
  "$EVAL_PY" phaseH_eval/eval_v2p10_promotion.py decide-fixed150 \
    --candidate "$candidate_metrics" --best-incumbent "$base_metrics" \
    --v2p9 "$v2p9_metrics" --paired "$paired_v2p9" \
    --out "$fixed_decision"
fi
if [[ "$(decision_bool "$fixed_decision" run_full300)" != "1" ]]; then
  log "PROMOTION STOPPED at fixed150; no_adapter_selected=true"
  assert_gpu1_and_port_free
  check_prod_health
  trap - EXIT
  exit 0
fi

run_fixed_eval \
  "teacher_sft_v2p10" "$MODELS/merged/teacher_sft_v2p10_full" \
  "$FULL_IDS" "v2p10_full300"

if [[ ! -d runs/v2p10_full300_wrong_edit_audit ]]; then
  "$EVAL_PY" phaseH_eval/audit_v2p9_failure_modes.py \
    runs/v2p10_full300 runs/v2p10_full300_wrong_edit_audit
fi

assert_gpu1_and_port_free
check_prod_health
trap - EXIT
log "V2P10 EVALUATION COMPLETE; full300 artifacts ready; no_adapter_selected=true; evaluator_pid=$SELF_PID evaluator_start_ticks=$SELF_START_TICKS"
