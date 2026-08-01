#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

ANCHOR="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full"
SOURCE="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full"
OUTPUT="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r4_blend25"
MANIFEST="$OUTPUT/interpolation_manifest.json"
ALPHA="0.25"
CANDIDATE_NAME="teacher_sft_v2p11r4_blend25"
ARTIFACT_TAG="v2p11r4_blend25"
MATERIALIZE_UNIT="v2p11r4-blend25-materialize.service"
MATERIALIZE_LOG="/tmp/v2p11r4_blend25_materialize.log"
FIXED_IDS="data/teacher_sft_fixed150_ids.json"
FULL_IDS="data/swebench_lite_test_ids.json"
CONTROL_FIXED_ROOT="runs/fixed150_v2p10"
CANDIDATE_FIXED_ROOT="runs/fixed150_v2p11r4_blend25_primary"
PORTABILITY_IDS="data/v2p11_portability_verified10_ids.json"
VERIFIED_EXCLUSIONS="data/swe_verified_eval_exclusions_v1.json"
PORTABILITY_CONTROL_ROOT="runs/v2p11_portability_stock/v2p10"
PORTABILITY_ROOT="runs/v2p11r4_blend25_portability_stock"
PORTABILITY_CANDIDATE_ROOT="$PORTABILITY_ROOT/candidate"
PORTABILITY_GATE="runs/v2p11r4_blend25_portability_gate.json"
PREFULL_GATE="runs/v2p11r4_blend25_prefull_gate.json"
V2P10_COMPOSITE="runs/v2p10_full300_composite.json"
V2P10_LINEAGE="runs/v2p11_v2p10_training_lineage.json"
SOURCE_PROVENANCE="runs/v2p11r3_behavior_completion_provenance.json"
EVAL_PY="$ROOT/.venv-eval/bin/python"
TRAIN_PY="$ROOT/.venv-train/bin/python"
GPU_INDEX="${GPU_INDEX:-1}"
PORT="${PORT:-8013}"
MEMORY_ADMISSION_GIB="${MEMORY_ADMISSION_GIB:-24}"
MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"
LOG="${LOG:-/tmp/run_v2p11r4_blend25_chain.log}"

exec > >(tee -a "$LOG") 2>&1

log() {
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

halt() {
  log "HALT: $*"
  exit 1
}

validate_candidate_lineage() {
  "$EVAL_PY" phaseH_eval/v2p11_interpolation_lineage.py \
    --manifest "$MANIFEST" \
    --anchor-model "$ANCHOR" \
    --source-model "$SOURCE" \
    --output-model "$OUTPUT" \
    --candidate-name "$CANDIDATE_NAME"
}

wait_for_materialization() {
  local state result status
  while true; do
    state="$(
      systemctl --user show "$MATERIALIZE_UNIT" -p ActiveState --value \
        2>/dev/null || true
    )"
    case "$state" in
      active|activating|deactivating) sleep 60 ;;
      *) break ;;
    esac
  done
  result="$(
    systemctl --user show "$MATERIALIZE_UNIT" -p Result --value \
      2>/dev/null || true
  )"
  status="$(
    systemctl --user show "$MATERIALIZE_UNIT" -p ExecMainStatus --value \
      2>/dev/null || true
  )"
  [[ "$state" == "inactive" && "$result" == "success" && "$status" == "0" ]] ||
    halt "materialization unit failed state=$state result=$result status=$status"
}

materialize_candidate() {
  local available_gib available_kib main_pid state
  if [[ -d "$OUTPUT" ]]; then
    validate_candidate_lineage
    log "reusing verified interpolated checkpoint output=$OUTPUT"
    return
  fi
  [[ ! -e "$OUTPUT" ]] || halt "candidate output exists but is not a directory"
  [[ -d "$ANCHOR" && -d "$SOURCE" ]] || halt "interpolation parent is missing"
  available_gib="$(df -BG --output=avail "$(dirname "$OUTPUT")" | tail -n 1 | tr -dc '0-9')"
  (( available_gib >= 100 )) ||
    halt "interpolation disk admission refused free=${available_gib}GiB"
  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  (( available_kib >= 20 * 1024 * 1024 )) ||
    halt "interpolation memory admission refused available_kib=$available_kib"
  state="$(
    systemctl --user show "$MATERIALIZE_UNIT" -p ActiveState --value \
      2>/dev/null || true
  )"
  if [[ "$state" != "active" && "$state" != "activating" ]]; then
    systemd-run --user --unit="$MATERIALIZE_UNIT" \
      --property=Type=exec \
      --property=WorkingDirectory="$ROOT" \
      --property=MemoryMax=24G \
      --property=OOMPolicy=stop \
      --property="StandardOutput=append:$MATERIALIZE_LOG" \
      --property="StandardError=append:$MATERIALIZE_LOG" \
      /usr/bin/env PYTHONPATH="$ROOT" \
      "$TRAIN_PY" phaseD_sft/interpolate_checkpoints_streaming.py \
        --anchor "$ANCHOR" --candidate "$SOURCE" --out "$OUTPUT" \
        --alpha "$ALPHA" --group-gb 3 --max-rss-gb 14
  fi
  main_pid="$(
    systemctl --user show "$MATERIALIZE_UNIT" -p MainPID --value \
      2>/dev/null || true
  )"
  [[ "$main_pid" =~ ^[1-9][0-9]*$ ]] ||
    halt "materialization unit has no exact MainPID"
  log "materialization started unit=$MATERIALIZE_UNIT pid=$main_pid"
  wait_for_materialization
  validate_candidate_lineage
  log "materialization complete unit=$MATERIALIZE_UNIT output=$OUTPUT"
}

run_portability() {
  env \
    LINEAGE_MODE=interpolation PORTABILITY_MODE=full \
    INTERPOLATION_MANIFEST="$MANIFEST" \
    INTERPOLATION_ANCHOR_MODEL="$ANCHOR" \
    INTERPOLATION_SOURCE_MODEL="$SOURCE" \
    CANDIDATE_NAME="$CANDIDATE_NAME" CANDIDATE_MODEL="$OUTPUT" \
    ARTIFACT_ROOT="$PORTABILITY_ROOT" \
    CONTROL_ROOT="$PORTABILITY_CONTROL_ROOT" \
    CANDIDATE_ROOT="$PORTABILITY_CANDIDATE_ROOT" \
    GATE="$PORTABILITY_GATE" \
    IDS="$PORTABILITY_IDS" VERIFIED_EXCLUSIONS="$VERIFIED_EXCLUSIONS" \
    LITE_IDS="$FULL_IDS" GPU_INDEX=1 PORT=8013 \
    MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    /usr/bin/bash phaseH_eval/run_v2p11_portability_gate.sh
}

fixed_run_complete() {
  "$EVAL_PY" - "$FIXED_IDS" "$CANDIDATE_FIXED_ROOT" <<'PY'
import sys
from pathlib import Path
from phaseH_eval.empty_retry_composite import _complete_run
_complete_run(
    ids_path=Path(sys.argv[1]),
    run_root=Path(sys.argv[2]),
    label="v2p11r4 fixed150 first pass",
)
PY
}

run_fixed150() {
  if fixed_run_complete >/dev/null 2>&1; then
    log "reusing verified fixed150 run=$CANDIDATE_FIXED_ROOT"
    return
  fi
  env \
    MODE=primary NAME="$CANDIDATE_NAME" MODEL="$OUTPUT" \
    IDS="$FIXED_IDS" RUN_ID="${CANDIDATE_FIXED_ROOT#runs/}" \
    EXPECTED_IDS=150 SEED=1 GPU_INDEX=1 PORT=8013 \
    LOG="/tmp/fixed150_v2p11r4_blend25_primary.log" \
    SERVE_LOG="/tmp/serve_fixed150_v2p11r4_blend25_primary.log" \
    MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    /usr/bin/bash phaseH_eval/run_v2p10_empty_diff_goal.sh
  fixed_run_complete || halt "fixed150 candidate run is incomplete"
}

publish_prefull_gate() {
  "$EVAL_PY" phaseH_eval/v2p11_successor_gate.py \
    --fixed-ids "$FIXED_IDS" \
    --control-run-root "$CONTROL_FIXED_ROOT" \
    --candidate-run-root "$CANDIDATE_FIXED_ROOT" \
    --manifest "$MANIFEST" --anchor-model "$ANCHOR" \
    --source-model "$SOURCE" --output-model "$OUTPUT" \
    --candidate-name "$CANDIDATE_NAME" \
    --portability-gate "$PORTABILITY_GATE" \
    --portability-ids "$PORTABILITY_IDS" \
    --verified-exclusions "$VERIFIED_EXCLUSIONS" \
    --lite-ids "$FULL_IDS" \
    --portability-control-root "$PORTABILITY_CONTROL_ROOT" \
    --portability-candidate-root "$PORTABILITY_CANDIDATE_ROOT" \
    --full-ids "$FULL_IDS" --v2p10-composite "$V2P10_COMPOSITE" \
    --v2p10-lineage "$V2P10_LINEAGE" \
    --source-provenance "$SOURCE_PROVENANCE" \
    --out "$PREFULL_GATE"
}

gate_allows_full300() {
  [[ "$(jq -er '.status == "complete" and .passed == true and .decision.run_full300 == true' "$PREFULL_GATE")" == "true" ]]
}

run_full300() {
  env \
    NAME="$CANDIDATE_NAME" MODEL="$OUTPUT" ARTIFACT_TAG="$ARTIFACT_TAG" \
    LINEAGE_MODE=interpolation \
    INTERPOLATION_MANIFEST="$MANIFEST" \
    INTERPOLATION_ANCHOR_MODEL="$ANCHOR" \
    INTERPOLATION_SOURCE_MODEL="$SOURCE" \
    PREFULL_GATE="$PREFULL_GATE" PROVENANCE="$PREFULL_GATE" \
    PORTABILITY_MARKER="$PORTABILITY_GATE" \
    FIXED_SOURCE="$CANDIDATE_FIXED_ROOT" \
    GPU_INDEX=1 PORT=8013 \
    MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    /usr/bin/bash phaseH_eval/eval_v2p11_full300_after_merge.sh
}

main() {
  [[ "$GPU_INDEX" == "1" ]] || halt "GPU_INDEX is fixed to GPU1"
  [[ "$PORT" == "8013" ]] || halt "PORT is fixed to 8013"
  [[ -x "$EVAL_PY" && -x "$TRAIN_PY" ]] || halt "Python environment is missing"
  materialize_candidate
  run_portability
  run_fixed150
  publish_prefull_gate
  if ! gate_allows_full300; then
    log "pre-full300 gate rejected candidate artifact=$PREFULL_GATE"
    return 0
  fi
  run_full300
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
