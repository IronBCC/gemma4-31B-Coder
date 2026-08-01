#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="$ROOT/.venv-eval/bin/python"
COMPOSITE_PY="phaseH_eval/empty_retry_composite.py"
RUNNER="phaseH_eval/run_v2p10_empty_diff_goal.sh"
FULL300_PY="phaseH_eval/full300_panel_composite.py"
SCORE_BINDING_PY="phaseH_eval/full300_official_score_binding.py"
LOG="${LOG:-/tmp/finish_v2p10_full300_after_complement.log}"
MODEL="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full"
NAME="teacher_sft_v2p10"

FIXED_IDS="data/teacher_sft_fixed150_ids.json"
FIXED_SOURCE="runs/fixed150_v2p10"
FIXED_RETRY1_IDS="data/fixed150_v2p10_empty_retry_diff1_ids.json"
FIXED_RETRY1="runs/fixed150_v2p10_empty_retry_diff1"
FIXED_PLAN1="runs/fixed150_v2p10_empty_retry_diff1_plan.json"
FIXED_BOUND1="runs/fixed150_v2p10_empty_diff1_bound_v2_composite.json"
FIXED_BOUND1_PREDS="runs/fixed150_v2p10_empty_diff1_bound_v2_preds.json"
FIXED_RETRY2_IDS="data/fixed150_v2p10_empty_retry_diff2_bound_v2_ids.json"
FIXED_RETRY2="runs/fixed150_v2p10_empty_retry_diff2_bound_v2"
FIXED_PLAN2="runs/fixed150_v2p10_empty_retry_diff2_bound_v2_plan.json"
FIXED_FINAL="runs/fixed150_v2p10_empty_diff2_bound_v2_composite.json"
FIXED_FINAL_PREDS="runs/fixed150_v2p10_empty_diff2_bound_v2_preds.json"

COMPLEMENT_IDS="data/swebench_lite_complement150_ids.json"
COMPLEMENT_SOURCE="runs/fixed150_v2p10_complement150_v1"
COMPLEMENT_RETRY_IDS="data/fixed150_v2p10_complement150_empty_retry_diff1_ids.json"
COMPLEMENT_RETRY="runs/fixed150_v2p10_complement150_empty_retry_diff1"
COMPLEMENT_PLAN="runs/fixed150_v2p10_complement150_empty_retry_diff1_plan.json"
COMPLEMENT_FINAL="runs/fixed150_v2p10_complement150_empty_diff1_composite.json"
COMPLEMENT_FINAL_PREDS="runs/fixed150_v2p10_complement150_empty_diff1_preds.json"

FULL_IDS="data/swebench_lite_test_ids.json"
FULL_COMPOSITE="runs/v2p10_full300_composite.json"
FULL_PREDS="runs/v2p10_full300_preds.json"
SCORE_BINDING="runs/v2p10_full300_official_score_binding.json"

log() {
  echo "[$(date '+%H:%M:%S %Z')] $*" | tee -a "$LOG"
}

halt() {
  log "HALT: $*"
  exit 1
}

require_pair_or_absent() {
  local first="$1"
  local second="$2"
  if [[ -e "$first" && ! -e "$second" ]] || [[ ! -e "$first" && -e "$second" ]]; then
    halt "partial output pair: $first $second"
  fi
}

[[ -x "$PY" ]] || halt "evaluation Python is missing: $PY"
[[ -f "$RUNNER" ]] || halt "retry runner is missing: $RUNNER"
[[ -d "$MODEL" ]] || halt "model directory is missing: $MODEL"

require_pair_or_absent "$FIXED_BOUND1" "$FIXED_BOUND1_PREDS"
if [[ ! -e "$FIXED_BOUND1" ]]; then
  log "rebuilding checksum-bound fixed150 generation-1 composite"
  "$PY" "$COMPOSITE_PY" combine \
    --source-ids "$FIXED_IDS" \
    --source-run-root "$FIXED_SOURCE" \
    --retry-ids "$FIXED_RETRY1_IDS" \
    --retry-run-root "$FIXED_RETRY1" \
    --plan "$FIXED_PLAN1" \
    --out "$FIXED_BOUND1" \
    --preds-out "$FIXED_BOUND1_PREDS"
fi
"$PY" "$COMPOSITE_PY" verify \
  --source-ids "$FIXED_IDS" \
  --source-run-root "$FIXED_SOURCE" \
  --retry-ids "$FIXED_RETRY1_IDS" \
  --retry-run-root "$FIXED_RETRY1" \
  --plan "$FIXED_PLAN1" \
  --out "$FIXED_BOUND1" \
  --preds-out "$FIXED_BOUND1_PREDS"

"$PY" "$COMPOSITE_PY" freeze-composite \
  --source-ids "$FIXED_IDS" \
  --source-composite "$FIXED_BOUND1" \
  --source-preds "$FIXED_BOUND1_PREDS" \
  --retry-ids-out "$FIXED_RETRY2_IDS" \
  --plan-out "$FIXED_PLAN2"
fixed_retry_count="$(jq -er 'if type == "array" then length else error("not array") end' "$FIXED_RETRY2_IDS")"
[[ "$fixed_retry_count" == "1" ]] ||
  halt "expected one fixed150 generation-2 retry, got $fixed_retry_count"

require_pair_or_absent "$FIXED_FINAL" "$FIXED_FINAL_PREDS"
if [[ ! -e "$FIXED_FINAL" ]]; then
  log "running fixed150 generation-2 empty retry"
  env \
    MODE=primary \
    NAME="$NAME" \
    MODEL="$MODEL" \
    IDS="$FIXED_RETRY2_IDS" \
    RUN_ID="${FIXED_RETRY2#runs/}" \
    EXPECTED_IDS="$fixed_retry_count" \
    SEED=2 \
    REQUIRE_PRODUCTION_HEALTH=0 \
    LOG=/tmp/fixed150_v2p10_empty_retry_diff2_bound_v2.log \
    SERVE_LOG=/tmp/serve_fixed150_v2p10_empty_retry_diff2_bound_v2.log \
    /usr/bin/bash "$RUNNER"
  "$PY" "$COMPOSITE_PY" combine-composite \
    --source-ids "$FIXED_IDS" \
    --source-composite "$FIXED_BOUND1" \
    --source-preds "$FIXED_BOUND1_PREDS" \
    --retry-ids "$FIXED_RETRY2_IDS" \
    --retry-run-root "$FIXED_RETRY2" \
    --plan "$FIXED_PLAN2" \
    --out "$FIXED_FINAL" \
    --preds-out "$FIXED_FINAL_PREDS"
fi
"$PY" "$COMPOSITE_PY" verify-composite \
  --source-ids "$FIXED_IDS" \
  --source-composite "$FIXED_BOUND1" \
  --source-preds "$FIXED_BOUND1_PREDS" \
  --retry-ids "$FIXED_RETRY2_IDS" \
  --retry-run-root "$FIXED_RETRY2" \
  --plan "$FIXED_PLAN2" \
  --out "$FIXED_FINAL" \
  --preds-out "$FIXED_FINAL_PREDS"

"$PY" "$COMPOSITE_PY" freeze \
  --source-ids "$COMPLEMENT_IDS" \
  --source-run-root "$COMPLEMENT_SOURCE" \
  --retry-ids-out "$COMPLEMENT_RETRY_IDS" \
  --plan-out "$COMPLEMENT_PLAN"
complement_retry_count="$(jq -er 'if type == "array" then length else error("not array") end' "$COMPLEMENT_RETRY_IDS")"
[[ "$complement_retry_count" == "10" ]] ||
  halt "expected ten complement empty retries, got $complement_retry_count"

require_pair_or_absent "$COMPLEMENT_FINAL" "$COMPLEMENT_FINAL_PREDS"
if [[ ! -e "$COMPLEMENT_FINAL" ]]; then
  log "running complement150 generation-1 empty retries"
  env \
    MODE=empty_retry \
    NAME="$NAME" \
    MODEL="$MODEL" \
    IDS="$COMPLEMENT_RETRY_IDS" \
    RUN_ID="${COMPLEMENT_RETRY#runs/}" \
    EXPECTED_IDS="$complement_retry_count" \
    SEED=1 \
    REQUIRE_PRODUCTION_HEALTH=0 \
    SOURCE_IDS="$COMPLEMENT_IDS" \
    SOURCE_RUN="$COMPLEMENT_SOURCE" \
    PLAN="$COMPLEMENT_PLAN" \
    COMPOSITE="$COMPLEMENT_FINAL" \
    PREDICTIONS="$COMPLEMENT_FINAL_PREDS" \
    LOG=/tmp/fixed150_v2p10_complement150_empty_retry_diff1.log \
    SERVE_LOG=/tmp/serve_fixed150_v2p10_complement150_empty_retry_diff1.log \
    /usr/bin/bash "$RUNNER"
fi
"$PY" "$COMPOSITE_PY" verify \
  --source-ids "$COMPLEMENT_IDS" \
  --source-run-root "$COMPLEMENT_SOURCE" \
  --retry-ids "$COMPLEMENT_RETRY_IDS" \
  --retry-run-root "$COMPLEMENT_RETRY" \
  --plan "$COMPLEMENT_PLAN" \
  --out "$COMPLEMENT_FINAL" \
  --preds-out "$COMPLEMENT_FINAL_PREDS"

require_pair_or_absent "$FULL_COMPOSITE" "$FULL_PREDS"
if [[ ! -e "$FULL_COMPOSITE" ]]; then
  log "joining disjoint corrected panels into v2.10 full300"
  "$PY" "$FULL300_PY" join \
    --full-ids "$FULL_IDS" \
    --panel fixed150 "$FIXED_IDS" "$FIXED_FINAL" "$FIXED_FINAL_PREDS" \
    --panel complement150 "$COMPLEMENT_IDS" "$COMPLEMENT_FINAL" "$COMPLEMENT_FINAL_PREDS" \
    --out "$FULL_COMPOSITE" \
    --preds-out "$FULL_PREDS"
fi
"$PY" "$FULL300_PY" verify \
  --full-ids "$FULL_IDS" \
  --panel fixed150 "$FIXED_IDS" "$FIXED_FINAL" "$FIXED_FINAL_PREDS" \
  --panel complement150 "$COMPLEMENT_IDS" "$COMPLEMENT_FINAL" "$COMPLEMENT_FINAL_PREDS" \
  --out "$FULL_COMPOSITE" \
  --preds-out "$FULL_PREDS"
"$PY" phaseH_eval/capture_json_contract.py \
  --out "$SCORE_BINDING" -- \
  "$PY" "$SCORE_BINDING_PY" \
    --full-ids "$FULL_IDS" \
    --composite "$FULL_COMPOSITE" \
    --predictions "$FULL_PREDS" \
    --panel fixed150 "$FIXED_IDS" "$FIXED_FINAL" "$FIXED_FINAL_PREDS" \
    --panel complement150 "$COMPLEMENT_IDS" \
      "$COMPLEMENT_FINAL" "$COMPLEMENT_FINAL_PREDS"
sha256sum "$FULL_COMPOSITE" "$FULL_PREDS" "$SCORE_BINDING" |
  tee -a "$LOG"
log "V2P10 FULL300 COMPLETE"
