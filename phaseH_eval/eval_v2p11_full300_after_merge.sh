#!/usr/bin/env bash
# Evaluate v2.11 on the same panels with bounded empty-only correction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NAME="${NAME:-teacher_sft_v2p11}"
MODEL="${MODEL:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11_full}"
ARTIFACT_TAG="${ARTIFACT_TAG:-v2p11}"
LINEAGE_MODE="${LINEAGE_MODE:-posttrain}"
POSTTRAIN_MARKER="${POSTTRAIN_MARKER:-runs/v2p11_posttrain_complete.json}"
FINAL_AUDIT="${FINAL_AUDIT:-$MODEL/v2p11_final_merge_audit.json}"
PORTABILITY_MARKER="${PORTABILITY_MARKER:-runs/${ARTIFACT_TAG}_portability_gate.json}"
PROVENANCE="${PROVENANCE:-runs/${ARTIFACT_TAG}_completion_provenance.json}"
FULL_IDS="data/swebench_lite_test_ids.json"
FULL_IDS_SHA256="b98fc2b1054dc8fdfcb94f083f43454fd568961a0b3dbf8c388c210b7b868e14"
FIXED_IDS="data/teacher_sft_fixed150_ids.json"
COMPLEMENT_IDS="data/swebench_lite_complement150_ids.json"

V2P10_FIXED_SOURCE="runs/fixed150_v2p10"
V2P10_COMPLEMENT_SOURCE="runs/fixed150_v2p10_complement150_v1"
V2P10_COMPOSITE="runs/v2p10_full300_composite.json"
V2P10_LINEAGE="runs/v2p11_v2p10_training_lineage.json"
V2P10_PREDICTIONS="runs/v2p10_full300_preds.json"
V2P10_SCORE_BINDING="runs/v2p10_full300_official_score_binding.json"

FIXED_SOURCE="${FIXED_SOURCE:-runs/fixed150_${ARTIFACT_TAG}_primary}"
FIXED_GEN1_IDS="${FIXED_GEN1_IDS:-data/fixed150_${ARTIFACT_TAG}_empty_retry1_ids.json}"
FIXED_GEN1_PLAN="${FIXED_GEN1_PLAN:-runs/fixed150_${ARTIFACT_TAG}_empty_retry1_plan.json}"
FIXED_GEN1_RUN="${FIXED_GEN1_RUN:-runs/fixed150_${ARTIFACT_TAG}_empty_retry1}"
FIXED_GEN1_COMPOSITE="${FIXED_GEN1_COMPOSITE:-runs/fixed150_${ARTIFACT_TAG}_empty_retry1_composite.json}"
FIXED_GEN1_PREDICTIONS="${FIXED_GEN1_PREDICTIONS:-runs/fixed150_${ARTIFACT_TAG}_empty_retry1_preds.json}"
FIXED_GEN2_IDS="${FIXED_GEN2_IDS:-data/fixed150_${ARTIFACT_TAG}_empty_retry2_ids.json}"
FIXED_GEN2_PLAN="${FIXED_GEN2_PLAN:-runs/fixed150_${ARTIFACT_TAG}_empty_retry2_plan.json}"
FIXED_GEN2_RUN="${FIXED_GEN2_RUN:-runs/fixed150_${ARTIFACT_TAG}_empty_retry2}"
FIXED_GEN2_COMPOSITE="${FIXED_GEN2_COMPOSITE:-runs/fixed150_${ARTIFACT_TAG}_empty_retry2_composite.json}"
FIXED_GEN2_PREDICTIONS="${FIXED_GEN2_PREDICTIONS:-runs/fixed150_${ARTIFACT_TAG}_empty_retry2_preds.json}"

COMPLEMENT_SOURCE="${COMPLEMENT_SOURCE:-runs/complement150_${ARTIFACT_TAG}_primary}"
COMPLEMENT_GEN1_IDS="${COMPLEMENT_GEN1_IDS:-data/complement150_${ARTIFACT_TAG}_empty_retry1_ids.json}"
COMPLEMENT_GEN1_PLAN="${COMPLEMENT_GEN1_PLAN:-runs/complement150_${ARTIFACT_TAG}_empty_retry1_plan.json}"
COMPLEMENT_GEN1_RUN="${COMPLEMENT_GEN1_RUN:-runs/complement150_${ARTIFACT_TAG}_empty_retry1}"
COMPLEMENT_GEN1_COMPOSITE="${COMPLEMENT_GEN1_COMPOSITE:-runs/complement150_${ARTIFACT_TAG}_empty_retry1_composite.json}"
COMPLEMENT_GEN1_PREDICTIONS="${COMPLEMENT_GEN1_PREDICTIONS:-runs/complement150_${ARTIFACT_TAG}_empty_retry1_preds.json}"

COMPOSITE="${COMPOSITE:-runs/${ARTIFACT_TAG}_full300_composite.json}"
PREDICTIONS="${PREDICTIONS:-runs/${ARTIFACT_TAG}_full300_preds.json}"
SCORE_BINDING="${SCORE_BINDING:-runs/${ARTIFACT_TAG}_full300_official_score_binding.json}"
VERDICT="${VERDICT:-runs/${ARTIFACT_TAG}_vs_v2p10_full300.json}"
MARKDOWN="${MARKDOWN:-runs/${ARTIFACT_TAG}_vs_v2p10_full300.md}"
EVAL_PY="$ROOT/.venv-eval/bin/python"
RUNNER="phaseH_eval/run_v2p10_empty_diff_goal.sh"
EMPTY_TOOL="phaseH_eval/empty_retry_composite.py"
PANEL_TOOL="phaseH_eval/full300_panel_composite.py"
COMPARE_TOOL="phaseH_eval/compare_v2p11_full300.py"
SCORE_BINDING_TOOL="phaseH_eval/full300_official_score_binding.py"
LOG="${LOG:-/tmp/eval_${ARTIFACT_TAG}_full300_after_merge.log}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MAX_RUN_ATTEMPTS="${MAX_RUN_ATTEMPTS:-3}"
GPU_INDEX="${GPU_INDEX:-1}"
PORT="${PORT:-8013}"

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
    log "waiting for GPU$GPU_INDEX/port $PORT; exact GPU PIDs=$(tr '\n' ',' <<<"$pids")"
    sleep "$WAIT_SECONDS"
  done
}

require_pair_or_absent() {
  local first="$1" second="$2"
  if [[ -e "$first" && ! -e "$second" ]] ||
    [[ ! -e "$first" && -e "$second" ]]; then
    halt "partial output pair: $first $second"
  fi
}

complete_run_empty_count() {
  local ids="$1" run_root="$2"
  "$EVAL_PY" - "$ids" "$run_root" <<'PY'
import sys
from pathlib import Path
from phaseH_eval.empty_retry_composite import _complete_run
run = _complete_run(
    ids_path=Path(sys.argv[1]),
    run_root=Path(sys.argv[2]),
    label=Path(sys.argv[2]).name,
)
print(len(run["empty_ids"]))
PY
}

validate_composite() {
  local ids="$1" composite="$2" predictions="$3"
  "$EVAL_PY" - "$ids" "$composite" "$predictions" <<'PY'
import sys
from pathlib import Path
from phaseH_eval.empty_retry_composite import _validate_composite_source
_validate_composite_source(
    ids_path=Path(sys.argv[1]),
    composite_path=Path(sys.argv[2]),
    predictions_path=Path(sys.argv[3]),
)
PY
}

archive_incomplete_acceptance() {
  local source="$1" label="$2"
  local acceptance archive index=1
  acceptance="$source/acceptance.json"
  [[ -e "$acceptance" ]] || return 0
  [[ -f "$acceptance" ]] ||
    halt "$label acceptance is not a regular file: $acceptance"
  while :; do
    archive="$source/acceptance.incomplete.attempt-${index}.json"
    [[ ! -e "$archive" ]] && break
    ((index++))
  done
  mv -- "$acceptance" "$archive"
  log "$label archived incomplete acceptance=$archive"
}

run_primary() {
  local ids="$1" source="$2" expected="$3" seed="$4" label="$5"
  local attempt status
  if [[ -e "$source/acceptance.json" ]]; then
    if complete_run_empty_count "$ids" "$source" >/dev/null 2>&1; then
      log "$label reused complete primary run=$source"
      return
    fi
    archive_incomplete_acceptance "$source" "$label"
  fi
  for (( attempt=1; attempt<=MAX_RUN_ATTEMPTS; attempt++ )); do
    wait_for_gpu1_idle
    log "panel launch attempt=$attempt/$MAX_RUN_ATTEMPTS $label expected=$expected seed=$seed"
    set +e
    env \
      MODE=primary \
      NAME="$NAME" \
      MODEL="$MODEL" \
      IDS="$ids" \
      RUN_ID="${source#runs/}" \
      EXPECTED_IDS="$expected" \
      SEED="$seed" \
      GPU_INDEX="$GPU_INDEX" \
      PORT="$PORT" \
      LOG="/tmp/${source#runs/}.log" \
      SERVE_LOG="/tmp/serve_${source#runs/}.log" \
      /usr/bin/bash "$RUNNER"
    status="$?"
    set -e
    if complete_run_empty_count "$ids" "$source" >/dev/null 2>&1; then
      log "$label attempt=$attempt complete"
      return
    fi
    archive_incomplete_acceptance "$source" "$label"
    log "$label attempt=$attempt status=$status did not produce a complete acceptance"
  done
  halt "$label remains incomplete after $MAX_RUN_ATTEMPTS attempts"
}

build_generation_one() {
  local ids="$1" source="$2" expected="$3" retry_ids="$4"
  local plan="$5" retry_run="$6" composite="$7" predictions="$8" label="$9"
  local empty_count
  empty_count="$(complete_run_empty_count "$ids" "$source")"
  require_pair_or_absent "$composite" "$predictions"
  if (( empty_count == 0 )); then
    if [[ ! -e "$composite" ]]; then
      "$EVAL_PY" "$EMPTY_TOOL" promote-no-retry \
        --source-ids "$ids" \
        --source-run-root "$source" \
        --out "$composite" \
        --preds-out "$predictions"
    fi
    validate_composite "$ids" "$composite" "$predictions"
    return
  fi

  "$EVAL_PY" "$EMPTY_TOOL" freeze \
    --source-ids "$ids" \
    --source-run-root "$source" \
    --retry-ids-out "$retry_ids" \
    --plan-out "$plan"
  [[ "$(jq -er length "$retry_ids")" == "$empty_count" ]] ||
    halt "$label retry-1 count differs from frozen source empties"
  run_primary "$retry_ids" "$retry_run" "$empty_count" 1 "$label retry-1"
  if [[ ! -e "$composite" ]]; then
    "$EVAL_PY" "$EMPTY_TOOL" combine \
      --source-ids "$ids" \
      --source-run-root "$source" \
      --retry-ids "$retry_ids" \
      --retry-run-root "$retry_run" \
      --plan "$plan" \
      --out "$composite" \
      --preds-out "$predictions"
  fi
  "$EVAL_PY" "$EMPTY_TOOL" verify \
    --source-ids "$ids" \
    --source-run-root "$source" \
    --retry-ids "$retry_ids" \
    --retry-run-root "$retry_run" \
    --plan "$plan" \
    --out "$composite" \
    --preds-out "$predictions"
}

build_generation_two() {
  local ids="$1" source_composite="$2" source_predictions="$3"
  local retry_ids="$4" plan="$5" retry_run="$6" composite="$7"
  local predictions="$8" label="$9"
  local selected_composite_var="${10}" selected_predictions_var="${11}"
  local empty_count
  empty_count="$(
    "$EVAL_PY" - "$ids" "$source_composite" "$source_predictions" <<'PY'
import sys
from pathlib import Path
from phaseH_eval.empty_retry_composite import _validate_composite_source
source = _validate_composite_source(
    ids_path=Path(sys.argv[1]),
    composite_path=Path(sys.argv[2]),
    predictions_path=Path(sys.argv[3]),
)
print(len(source["empty_ids"]))
PY
  )"
  if (( empty_count == 0 )); then
    printf -v "$selected_composite_var" '%s' "$source_composite"
    printf -v "$selected_predictions_var" '%s' "$source_predictions"
    return
  fi

  "$EVAL_PY" "$EMPTY_TOOL" freeze-composite \
    --source-ids "$ids" \
    --source-composite "$source_composite" \
    --source-preds "$source_predictions" \
    --retry-ids-out "$retry_ids" \
    --plan-out "$plan"
  [[ "$(jq -er length "$retry_ids")" == "$empty_count" ]] ||
    halt "$label retry-2 count differs from frozen generation-1 empties"
  run_primary "$retry_ids" "$retry_run" "$empty_count" 2 "$label retry-2"
  require_pair_or_absent "$composite" "$predictions"
  if [[ ! -e "$composite" ]]; then
    "$EVAL_PY" "$EMPTY_TOOL" combine-composite \
      --source-ids "$ids" \
      --source-composite "$source_composite" \
      --source-preds "$source_predictions" \
      --retry-ids "$retry_ids" \
      --retry-run-root "$retry_run" \
      --plan "$plan" \
      --out "$composite" \
      --preds-out "$predictions"
  fi
  "$EVAL_PY" "$EMPTY_TOOL" verify-composite \
    --source-ids "$ids" \
    --source-composite "$source_composite" \
    --source-preds "$source_predictions" \
    --retry-ids "$retry_ids" \
    --retry-run-root "$retry_run" \
    --plan "$plan" \
    --out "$composite" \
    --preds-out "$predictions"
  printf -v "$selected_composite_var" '%s' "$composite"
  printf -v "$selected_predictions_var" '%s' "$predictions"
}

[[ "$GPU_INDEX" == "1" ]] ||
  halt "GPU_INDEX is fixed to GPU1; GPU0 is unavailable"
case "$LINEAGE_MODE" in
  posttrain|direct_lora) ;;
  *) halt "LINEAGE_MODE must be posttrain or direct_lora" ;;
esac
[[ "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
  halt "WAIT_SECONDS must be a positive integer"
[[ "$MAX_RUN_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] ||
  halt "MAX_RUN_ATTEMPTS must be a positive integer"
[[ -x "$EVAL_PY" ]] || halt "evaluation Python is missing: $EVAL_PY"
[[ -f "$RUNNER" ]] || halt "evaluation runner is missing: $RUNNER"
[[ -d "$MODEL" ]] || halt "v2.11 merged model is missing: $MODEL"
if [[ "$LINEAGE_MODE" == "posttrain" ]]; then
  [[ -f "$POSTTRAIN_MARKER" ]] ||
    halt "v2.11 posttrain marker is missing: $POSTTRAIN_MARKER"
fi
[[ -f "$FINAL_AUDIT" ]] || halt "v2.11 final merge audit is missing"
[[ -f "$PORTABILITY_MARKER" ]] ||
  halt "controller-free portability gate is missing: $PORTABILITY_MARKER"
[[ -f "$PROVENANCE" ]] ||
  halt "v2.11 completion provenance is missing: $PROVENANCE"
[[ -f "$V2P10_COMPOSITE" && -f "$V2P10_PREDICTIONS" &&
  -f "$V2P10_SCORE_BINDING" && -f "$V2P10_LINEAGE" ]] ||
  halt "v2.10 corrected full300 artifacts are missing"
[[ "$(sha256sum "$FULL_IDS" | awk '{print $1}')" == "$FULL_IDS_SHA256" ]] ||
  halt "full300 ID SHA mismatch"

"$EVAL_PY" - \
  "$NAME" "$MODEL" "$LINEAGE_MODE" "$POSTTRAIN_MARKER" \
  "$FINAL_AUDIT" "$PORTABILITY_MARKER" "$PROVENANCE" \
  "$FULL_IDS" "$V2P10_COMPOSITE" "$V2P10_LINEAGE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from phaseH_eval.compare_v2p11_full300 import (
    validate_completion_provenance,
)
from phaseH_eval.v2p11_completion_provenance import _model_contract

(
    name,
    model_value,
    lineage_mode,
    marker_value,
    audit_value,
    portability_value,
    provenance_value,
    full_ids_value,
    v2p10_value,
    v2p10_lineage_value,
) = sys.argv[1:]
model_path = Path(model_value).resolve()
marker_path = Path(marker_value)
audit_path = Path(audit_value)
portability_path = Path(portability_value)
provenance_path = Path(provenance_value)
full_ids_path = Path(full_ids_value)
v2p10_path = Path(v2p10_value)
audit = json.loads(audit_path.read_text())
portability = json.loads(portability_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if lineage_mode == "posttrain":
    marker = json.loads(marker_path.read_text())
    assert marker["schema_version"] == 1 and marker["complete"] is True
    assert marker["recovery_optimizer_steps"] == 105
    assert marker["kto_optimizer_steps"] == 25
    assert marker["final_merge_audit_sha256"] == sha(audit_path)
assert audit == {
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
assert portability["schema_version"] == 1
assert portability["status"] == "complete"
assert portability["passed"] is True
assert portability["candidate_name"] == name
model_contract = _model_contract(model_path)
model_contract["served_name"] = name
validate_completion_provenance(
    provenance_path,
    full_ids_path=full_ids_path,
    v2p10_composite_path=v2p10_path,
    v2p10_lineage_path=Path(v2p10_lineage_value),
    candidate_model_contract=model_contract,
    candidate_name=name,
)
PY

run_primary "$FIXED_IDS" "$FIXED_SOURCE" 150 1 "v2.11 fixed150 primary"
build_generation_one \
  "$FIXED_IDS" "$FIXED_SOURCE" 150 \
  "$FIXED_GEN1_IDS" "$FIXED_GEN1_PLAN" "$FIXED_GEN1_RUN" \
  "$FIXED_GEN1_COMPOSITE" "$FIXED_GEN1_PREDICTIONS" "v2.11 fixed150"
FIXED_SELECTED_COMPOSITE="$FIXED_GEN1_COMPOSITE"
FIXED_SELECTED_PREDICTIONS="$FIXED_GEN1_PREDICTIONS"
build_generation_two \
  "$FIXED_IDS" "$FIXED_GEN1_COMPOSITE" "$FIXED_GEN1_PREDICTIONS" \
  "$FIXED_GEN2_IDS" "$FIXED_GEN2_PLAN" "$FIXED_GEN2_RUN" \
  "$FIXED_GEN2_COMPOSITE" "$FIXED_GEN2_PREDICTIONS" "v2.11 fixed150" \
  FIXED_SELECTED_COMPOSITE FIXED_SELECTED_PREDICTIONS

run_primary \
  "$COMPLEMENT_IDS" "$COMPLEMENT_SOURCE" 150 1 \
  "v2.11 complement150 primary"
build_generation_one \
  "$COMPLEMENT_IDS" "$COMPLEMENT_SOURCE" 150 \
  "$COMPLEMENT_GEN1_IDS" "$COMPLEMENT_GEN1_PLAN" "$COMPLEMENT_GEN1_RUN" \
  "$COMPLEMENT_GEN1_COMPOSITE" "$COMPLEMENT_GEN1_PREDICTIONS" \
  "v2.11 complement150"

require_pair_or_absent "$COMPOSITE" "$PREDICTIONS"
if [[ ! -e "$COMPOSITE" ]]; then
  "$EVAL_PY" "$PANEL_TOOL" join \
    --full-ids "$FULL_IDS" \
    --panel fixed150 "$FIXED_IDS" \
      "$FIXED_SELECTED_COMPOSITE" "$FIXED_SELECTED_PREDICTIONS" \
    --panel complement150 "$COMPLEMENT_IDS" \
      "$COMPLEMENT_GEN1_COMPOSITE" "$COMPLEMENT_GEN1_PREDICTIONS" \
    --out "$COMPOSITE" \
    --preds-out "$PREDICTIONS"
fi
"$EVAL_PY" "$PANEL_TOOL" verify \
  --full-ids "$FULL_IDS" \
  --panel fixed150 "$FIXED_IDS" \
    "$FIXED_SELECTED_COMPOSITE" "$FIXED_SELECTED_PREDICTIONS" \
  --panel complement150 "$COMPLEMENT_IDS" \
    "$COMPLEMENT_GEN1_COMPOSITE" "$COMPLEMENT_GEN1_PREDICTIONS" \
  --out "$COMPOSITE" \
  --preds-out "$PREDICTIONS"
"$EVAL_PY" phaseH_eval/capture_json_contract.py \
  --out "$SCORE_BINDING" -- \
  "$EVAL_PY" "$SCORE_BINDING_TOOL" \
    --full-ids "$FULL_IDS" \
    --composite "$COMPOSITE" \
    --predictions "$PREDICTIONS" \
    --panel fixed150 "$FIXED_IDS" \
      "$FIXED_SELECTED_COMPOSITE" "$FIXED_SELECTED_PREDICTIONS" \
    --panel complement150 "$COMPLEMENT_IDS" \
      "$COMPLEMENT_GEN1_COMPOSITE" "$COMPLEMENT_GEN1_PREDICTIONS"

comparison_args=(
  --full-ids "$FULL_IDS"
  --v2p10 "$V2P10_COMPOSITE"
  --v2p10-lineage "$V2P10_LINEAGE"
  --v2p10-preds "$V2P10_PREDICTIONS"
  --v2p10-score-binding "$V2P10_SCORE_BINDING"
  --v2p11 "$COMPOSITE"
  --v2p11-preds "$PREDICTIONS"
  --v2p11-score-binding "$SCORE_BINDING"
  --candidate-name "$NAME"
  --candidate-model-path "$MODEL"
  --v2p10-first-pass-panel fixed150 "$FIXED_IDS" "$V2P10_FIXED_SOURCE"
  --v2p10-first-pass-panel complement150 "$COMPLEMENT_IDS" "$V2P10_COMPLEMENT_SOURCE"
  --v2p11-first-pass-panel fixed150 "$FIXED_IDS" "$FIXED_SOURCE"
  --v2p11-first-pass-panel complement150 "$COMPLEMENT_IDS" "$COMPLEMENT_SOURCE"
  --provenance "$PROVENANCE"
  --require-trustworthy-win
  --out "$VERDICT"
  --markdown-out "$MARKDOWN"
)
if [[ -e "$VERDICT" || -e "$MARKDOWN" ]]; then
  [[ -f "$VERDICT" && -f "$MARKDOWN" ]] ||
    halt "paired verdict publication is incomplete"
  "$EVAL_PY" "$COMPARE_TOOL" --verify-existing "${comparison_args[@]}"
else
  "$EVAL_PY" "$COMPARE_TOOL" "${comparison_args[@]}"
fi

wait_for_gpu1_idle
log "$NAME FULL300 COMPLETE verdict=$VERDICT markdown=$MARKDOWN"
