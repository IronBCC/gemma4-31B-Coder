#!/usr/bin/env bash
# Audit r3, run recovery/KTO, then evaluate only the final behavior model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_UNIT="${TRAIN_UNIT:-v2p11r3-reasoned-train-gpu1.service}"
TRAIN_LOG_SOURCE="${TRAIN_LOG_SOURCE:-/tmp/train_v2p11r3_reasoned_gpu1_rerun1.log}"
WATCHDOG_LOG_SOURCE="${WATCHDOG_LOG_SOURCE:-/tmp/ram_watchdog_v2p11r3_reasoned_gpu1_rerun1.log}"
DATA="${DATA:-data/teacher_train_mix_v2p11_fable1262_reasoned_v1}"
INIT_ADAPTER="${INIT_ADAPTER:-adapters/teacher_sft_v2p10_bf16}"
ADAPTER="${ADAPTER:-adapters/teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16}"
CANDIDATE_NAME="${CANDIDATE_NAME:-teacher_sft_v2p11r3_behavior}"
ARTIFACT_TAG="${ARTIFACT_TAG:-v2p11r3_behavior}"
MERGED="${MERGED:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full}"
MERGE_AUDIT="$MERGED/v2p11_final_merge_audit.json"
CONTEXT_AUDIT="${CONTEXT_AUDIT:-runs/v2p11r3_reasoned_context_audit.json}"
TRAINING_JOURNAL="runs/v2p11r3_v2p10init_fable_reasoned_training_journal.log"
WATCHDOG_EVIDENCE="runs/v2p11r3_v2p10init_fable_reasoned_watchdog.log"
TRAINING_COMPLETION="runs/v2p11r3_v2p10init_fable_reasoned_training_completion.json"
RECOVERY_DATA="${RECOVERY_DATA:-data/v2p11_portable_recovery138_targeted}"
BEHAVIOR_DATA="${BEHAVIOR_DATA:-data/v2p11_behavior_kto_v2.jsonl}"
BEHAVIOR_MANIFEST="${BEHAVIOR_MANIFEST:-data/v2p11_behavior_kto_v2_manifest.json}"
EXCLUSIONS="${EXCLUSIONS:-data/swe_all_eval_exclusions_v2.json}"
POSTTRAIN_MARKER="${POSTTRAIN_MARKER:-runs/v2p11r3_behavior_posttrain_complete.json}"
RECOVERY_MARKER="${RECOVERY_MARKER:-runs/v2p11r3_behavior_recovery_sft_complete.json}"
KTO_MARKER="${KTO_MARKER:-runs/v2p11r3_behavior_kto_complete.json}"
FINAL_MERGE_MARKER="${FINAL_MERGE_MARKER:-runs/v2p11r3_behavior_final_merge_complete.json}"
PORTABILITY_IDS="${PORTABILITY_IDS:-data/v2p11_portability_verified10_ids.json}"
VERIFIED_EXCLUSIONS="${VERIFIED_EXCLUSIONS:-data/swe_verified_eval_exclusions_v1.json}"
FULL_IDS="${FULL_IDS:-data/swebench_lite_test_ids.json}"
V2P10_COMPOSITE="runs/v2p10_full300_composite.json"
V2P10_LINEAGE="runs/v2p11_v2p10_training_lineage.json"
PORTABILITY_ROOT="runs/${ARTIFACT_TAG}_portability_stock"
CONTROL_ROOT="runs/v2p11_portability_stock/v2p10"
CANDIDATE_ROOT="$PORTABILITY_ROOT/candidate"
PORTABILITY_GATE="runs/${ARTIFACT_TAG}_portability_gate.json"
PROVENANCE="runs/${ARTIFACT_TAG}_completion_provenance.json"
VERDICT="runs/${ARTIFACT_TAG}_vs_v2p10_full300.json"
COMPOSITE="runs/${ARTIFACT_TAG}_full300_composite.json"
PREDICTIONS="runs/${ARTIFACT_TAG}_full300_preds.json"
SCORE_BINDING="runs/${ARTIFACT_TAG}_full300_official_score_binding.json"
GOAL_AUDIT="runs/${ARTIFACT_TAG}_goal_completion_audit.json"
MEMORY_ADMISSION_GIB="${MEMORY_ADMISSION_GIB:-24}"
MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"
TRAIN_PY="$ROOT/.venv-train/bin/python"
EVAL_PY="$ROOT/.venv-eval/bin/python"
LOG="${LOG:-/tmp/run_v2p11r3_posttrain_chain.log}"

exec > >(tee -a "$LOG") 2>&1

log() {
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

halt() {
  log "HALT: $*"
  exit 1
}

ensure_v2p10_lineage() {
  "$EVAL_PY" phaseH_eval/capture_json_contract.py \
    --out "$V2P10_LINEAGE" -- \
    "$EVAL_PY" phaseH_eval/v2p10_training_lineage.py \
      --marker runs/v2p10_train_merge_complete.json \
      --run-manifest adapters/teacher_sft_v2p10_bf16/run_manifest.json \
      --dataset-manifest data/teacher_train_mix_v2p10/manifest.json \
      --train-jsonl data/teacher_train_mix_v2p10/train.jsonl \
      --adapter adapters/teacher_sft_v2p10_bf16/adapter_model.safetensors \
      --merge-audit \
        /media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full/v2p10_merge_audit.json \
      --model \
        /media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full \
      --composite "$V2P10_COMPOSITE" \
      --stage-data-manifest \
        data/teacher_train_mix_v2p11_frozen1247/manifest.json
  "$EVAL_PY" - "$V2P10_LINEAGE" "$V2P10_COMPOSITE" \
    "$INIT_ADAPTER" <<'PY'
import json
import sys
from pathlib import Path

from phaseH_eval.empty_retry_composite import _binding
from phaseH_eval.v2p11r3_completion_provenance import (
    _validate_v2p10_init_lineage,
)

lineage, composite, initial = map(Path, sys.argv[1:])
report = _validate_v2p10_init_lineage(
    lineage_path=lineage,
    v2p10_composite_path=composite,
    training={
        "init_adapter": _binding(
            initial / "adapter_model.safetensors"
        ),
        "init_adapter_config": _binding(
            initial / "adapter_config.json"
        ),
        "init_adapter_path": str(initial.resolve()),
    },
)
print(json.dumps({
    "adapter_sha256": report["adapter"]["sha256"],
    "lineage_sha256": report["contract"]["sha256"],
}, sort_keys=True))
PY
}

wait_for_memory() {
  local available_kib
  while true; do
    available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available_kib >= MEMORY_ADMISSION_GIB * 1024 * 1024 )); then
      log "memory admission passed available_kib=$available_kib"
      return
    fi
    log "waiting for ${MEMORY_ADMISSION_GIB}GiB MemAvailable current_kib=$available_kib"
    sleep 60
  done
}

capture_evidence() {
  "$TRAIN_PY" - "$TRAIN_LOG_SOURCE" "$WATCHDOG_LOG_SOURCE" \
    "$TRAINING_JOURNAL" "$WATCHDOG_EVIDENCE" <<'PY'
import sys
from pathlib import Path

from phaseH_eval.empty_retry_composite import _publish_bytes_noreplace

source_train, source_watchdog, out_train, out_watchdog = map(Path, sys.argv[1:])
pairs = []
for source, output in ((source_train, out_train), (source_watchdog, out_watchdog)):
    payload = source.read_bytes()
    assert payload, source
    pairs.append((output, payload))

for output, payload in pairs:
    if output.exists():
        assert output.read_bytes() == payload, output

for output, payload in pairs:
    if output.exists():
        continue
    _publish_bytes_noreplace(output, payload)
    output.chmod(0o444)
PY
}

validate_training_evidence() {
  "$EVAL_PY" - "$DATA" "$INIT_ADAPTER" "$ADAPTER" \
    "$TRAINING_COMPLETION" <<'PY'
import json
import sys
from pathlib import Path

from phaseH_eval.v2p11r3_completion_provenance import _validate_training

data, initial, adapter, completion = map(Path, sys.argv[1:])
report = _validate_training(
    dataset=data,
    adapter=adapter,
    init_adapter=initial,
    completion_path=completion,
)
print(json.dumps({
    "changed_tensor_count": report["changed_tensor_count"],
    "completion_sha256": report["completion"]["sha256"],
}, sort_keys=True))
PY
}

ensure_captured_evidence() {
  if [[ -e "$TRAINING_JOURNAL" && -e "$WATCHDOG_EVIDENCE" ]]; then
    return
  fi
  [[ -f "$TRAIN_LOG_SOURCE" && -f "$WATCHDOG_LOG_SOURCE" ]] ||
    halt "captured training evidence is incomplete and source logs are missing"
  capture_evidence
}

ensure_training_evidence() {
  ensure_captured_evidence
  if [[ -e "$TRAINING_COMPLETION" ]]; then
    validate_training_evidence
    log "reusing verified r3 training completion artifact=$TRAINING_COMPLETION"
    return
  fi

  audit_training
  validate_training_evidence
}

audit_training() {
  "$TRAIN_PY" - "$DATA" "$INIT_ADAPTER" "$ADAPTER" \
    "$TRAINING_JOURNAL" "$WATCHDOG_EVIDENCE" "$TRAINING_COMPLETION" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from safetensors import safe_open
from phaseH_eval.empty_retry_composite import _publish_json_noreplace

data, initial, adapter, journal_path, watchdog_path, output = map(Path, sys.argv[1:])
manifest = json.loads((adapter / "run_manifest.json").read_text())
expected = {
    "base": "/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it",
    "data": str(data),
    "out": str(adapter),
    "init_adapter": str(initial),
    "data_len": 1262,
    "rank": 32,
    "alpha": 32,
    "lr": 2e-6,
    "epochs": 1.0,
    "bsz": 1,
    "grad_accum": 16,
    "max_seq": 32768,
    "max_steps": 79,
    "warmup_steps": 4,
    "logging_steps": 5,
    "save_steps": 10,
    "save_total_limit": 3,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}
assert {key: manifest.get(key) for key in expected} == expected
weights = adapter / "adapter_model.safetensors"
initial_weights = initial / "adapter_model.safetensors"
config = json.loads((adapter / "adapter_config.json").read_text())
assert config["r"] == 32 and config["lora_alpha"] == 32
total = a_count = b_count = changed = 0
with safe_open(weights, framework="pt", device="cpu") as candidate, safe_open(initial_weights, framework="pt", device="cpu") as baseline:
    keys = list(candidate.keys())
    assert keys == list(baseline.keys())
    for key in keys:
        current = candidate.get_tensor(key)
        prior = baseline.get_tensor(key)
        assert current.shape == prior.shape, key
        assert current.isfinite().all().item(), key
        total += 1
        a_count += int(".lora_A." in key)
        b_count += int(".lora_B." in key)
        changed += int(not current.equal(prior))
assert (total, a_count, b_count) == (820, 410, 410)
assert changed > 0
journal = journal_path.read_text(encoding="utf-8")
watchdog = watchdog_path.read_text(encoding="utf-8")
assert "optimizer_step=79/79" in journal
assert f"[done] adapter saved -> {adapter}" in journal
assert "event=target_exited pid=" in watchdog
assert not re.search(r"event=(below_threshold|meminfo_error|health_failure) pid=", watchdog)
def binding(path):
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
report = {
    "schema_version": 1,
    "artifact_type": "v2p11r3_training_completion",
    "status": "complete",
    "optimizer_steps": 79,
    "max_steps": 79,
    "adapter_changed_from_init": True,
    "changed_tensor_count": changed,
    "adapter": binding(weights),
    "init_adapter": binding(initial_weights),
    "training_journal": binding(journal_path),
    "watchdog_log": binding(watchdog_path),
}
_publish_json_noreplace(output, report)
output.chmod(0o444)
print(json.dumps({"changed_tensor_count": changed, "tensors": total}, sort_keys=True))
PY
}

run_behavior_poststage() {
  env \
    GPU_INDEX=1 TRAIN_UNIT="$TRAIN_UNIT" \
    STAGE_A_ADAPTER="$ADAPTER" STAGE_A_MARKER="$TRAINING_COMPLETION" \
    RECOVERY_DATA="$RECOVERY_DATA" BEHAVIOR_DATA="$BEHAVIOR_DATA" \
    BEHAVIOR_MANIFEST="$BEHAVIOR_MANIFEST" EXCLUSIONS="$EXCLUSIONS" \
    FULL_IDS="$FULL_IDS" \
    FINAL_MERGED="$MERGED" COMPLETION_MARKER="$POSTTRAIN_MARKER" \
    RECOVERY_MARKER="$RECOVERY_MARKER" KTO_MARKER="$KTO_MARKER" \
    FINAL_MERGE_MARKER="$FINAL_MERGE_MARKER" \
    RAM_FLOOR_GIB="$MEMORY_WATCHDOG_GIB" \
    /usr/bin/bash phaseH_eval/train_v2p11r3_behavior_gpu1.sh
  [[ -f "$POSTTRAIN_MARKER" && -f "$MERGE_AUDIT" ]] ||
    halt "behavior poststage ended without completion marker or merge audit"
}

run_portability() {
  if [[ ! -e "$PORTABILITY_GATE" ]]; then
    env \
      LINEAGE_MODE=posttrain \
      POSTTRAIN_MARKER="$POSTTRAIN_MARKER" \
      RECOVERY_MARKER="$RECOVERY_MARKER" \
      KTO_MARKER="$KTO_MARKER" \
      FINAL_MERGE_MARKER="$FINAL_MERGE_MARKER" \
      PORTABILITY_MODE=full \
      CANDIDATE_NAME="$CANDIDATE_NAME" \
      CANDIDATE_MODEL="$MERGED" \
      FINAL_AUDIT="$MERGE_AUDIT" \
      ARTIFACT_ROOT="$PORTABILITY_ROOT" \
      IDS="$PORTABILITY_IDS" VERIFIED_EXCLUSIONS="$VERIFIED_EXCLUSIONS" \
      LITE_IDS="$FULL_IDS" \
      CONTROL_ROOT="$CONTROL_ROOT" CANDIDATE_ROOT="$CANDIDATE_ROOT" \
      GATE="$PORTABILITY_GATE" \
      MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
      MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
      GPU_INDEX=1 PORT=8013 \
      /usr/bin/bash phaseH_eval/run_v2p11_portability_gate.sh
    return
  fi

  "$EVAL_PY" - \
    "$POSTTRAIN_MARKER" "$RECOVERY_MARKER" "$KTO_MARKER" \
    "$FINAL_MERGE_MARKER" "$MERGED" "$MERGE_AUDIT" \
    "$PORTABILITY_GATE" "$CANDIDATE_NAME" "$PORTABILITY_IDS" \
    "$VERIFIED_EXCLUSIONS" "$FULL_IDS" "$CONTROL_ROOT" \
    "$CANDIDATE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from phaseH_eval.empty_retry_composite import _read_object
from phaseH_eval.v2p11_posttrain_lineage import validate_posttrain_lineage
from phaseH_eval.v2p11r3_completion_provenance import _validate_final
from phaseH_eval.v2p11_portability_gate import evaluate_portability

(
    posttrain,
    recovery,
    kto,
    final_merge,
    model,
    audit,
    gate,
    name,
    ids,
    verified_exclusions,
    lite_ids,
    control_root,
    candidate_root,
) = sys.argv[1:]
lineage = validate_posttrain_lineage(
    posttrain_marker_path=Path(posttrain),
    recovery_marker_path=Path(recovery),
    kto_marker_path=Path(kto),
    final_merge_marker_path=Path(final_merge),
    final_model_path=Path(model),
    final_audit_path=Path(audit),
)
current_gate = evaluate_portability(
    ids_path=Path(ids),
    verified_exclusions_path=Path(verified_exclusions),
    lite_ids_path=Path(lite_ids),
    control_root=Path(control_root),
    candidate_root=Path(candidate_root),
    candidate_name=name,
)
published_gate = _read_object(Path(gate))
if current_gate != published_gate:
    raise ValueError(
        "existing portability gate differs from current run artifacts"
    )
if current_gate.get("passed") is not True:
    raise ValueError("existing portability gate did not pass")
final = _validate_final(
    candidate_name=name,
    final_model=Path(model),
    merge_audit=Path(audit),
    portability_gate=Path(gate),
)
print(json.dumps({
    "lineage_artifact_type": lineage["artifact_type"],
    "portability_gate_sha256": final["portability_gate"]["sha256"],
}, sort_keys=True))
PY
  log "reusing verified behavior portability gate artifact=$PORTABILITY_GATE"
}

publish_provenance() {
  if [[ ! -e "$PROVENANCE" ]]; then
    "$EVAL_PY" phaseH_eval/v2p11r3_behavior_completion_provenance.py \
      --candidate-name "$CANDIDATE_NAME" --dataset "$DATA" \
      --dataset-context-audit "$CONTEXT_AUDIT" --adapter "$ADAPTER" \
      --init-adapter "$INIT_ADAPTER" \
      --training-completion "$TRAINING_COMPLETION" \
      --recovery-data "$RECOVERY_DATA" --behavior-data "$BEHAVIOR_DATA" \
      --behavior-manifest "$BEHAVIOR_MANIFEST" --exclusions "$EXCLUSIONS" \
      --posttrain-marker "$POSTTRAIN_MARKER" \
      --recovery-marker "$RECOVERY_MARKER" --kto-marker "$KTO_MARKER" \
      --final-merge-marker "$FINAL_MERGE_MARKER" \
      --final-model "$MERGED" --merge-audit "$MERGE_AUDIT" \
      --portability-gate "$PORTABILITY_GATE" \
      --full-ids "$FULL_IDS" \
      --v2p10 "$V2P10_COMPOSITE" \
      --v2p10-lineage "$V2P10_LINEAGE" --out "$PROVENANCE"
  else
    "$EVAL_PY" - \
      "$PROVENANCE" "$MERGED" "$CANDIDATE_NAME" "$FULL_IDS" \
      "$V2P10_LINEAGE" <<'PY'
import sys
from pathlib import Path
from phaseH_eval.v2p11_completion_provenance import _model_contract
from phaseH_eval.v2p11r3_behavior_completion_provenance import validate_completion_provenance

provenance, model, name, full_ids, v2p10_lineage = sys.argv[1:]
contract = _model_contract(Path(model).resolve())
contract["served_name"] = name
validate_completion_provenance(
    Path(provenance),
    full_ids_path=Path(full_ids),
    v2p10_composite_path=Path("runs/v2p10_full300_composite.json"),
    v2p10_lineage_path=Path(v2p10_lineage),
    candidate_model_contract=contract,
    candidate_name=name,
)
PY
  fi
}

run_full300() {
  env \
    NAME="$CANDIDATE_NAME" MODEL="$MERGED" ARTIFACT_TAG="$ARTIFACT_TAG" \
    LINEAGE_MODE=posttrain POSTTRAIN_MARKER="$POSTTRAIN_MARKER" \
    FINAL_AUDIT="$MERGE_AUDIT" \
    PORTABILITY_MARKER="$PORTABILITY_GATE" PROVENANCE="$PROVENANCE" \
    MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    GPU_INDEX=1 PORT=8013 \
    /usr/bin/bash phaseH_eval/eval_v2p11_full300_after_merge.sh
  [[ -f "$VERDICT" ]] || halt "full300 ended without a verdict"
}

audit_goal() {
  "$EVAL_PY" phaseH_eval/v2p11r3_goal_completion_audit.py \
    --verdict "$VERDICT" --full-ids "$FULL_IDS" \
    --v2p10 "$V2P10_COMPOSITE" \
    --v2p10-lineage "$V2P10_LINEAGE" \
    --v2p10-preds runs/v2p10_full300_preds.json \
    --v2p10-score runs/v2p10_full300_official_score_binding.json \
    --v2p11 "$COMPOSITE" --v2p11-preds "$PREDICTIONS" \
    --v2p11-score "$SCORE_BINDING" --provenance "$PROVENANCE" \
    --candidate-model "$MERGED" --candidate-name "$CANDIDATE_NAME" \
    --out "$GOAL_AUDIT"
  [[ -f "$GOAL_AUDIT" ]] || halt "goal audit did not publish"
}

main() {
  train_state="$(systemctl --user show "$TRAIN_UNIT" -p ActiveState --value 2>/dev/null || true)"
  [[ "$train_state" != "active" && "$train_state" != "activating" &&
    "$train_state" != "deactivating" ]] ||
    halt "training unit is still active: $TRAIN_UNIT"
  [[ -x "$TRAIN_PY" && -x "$EVAL_PY" ]] || halt "required Python environment is missing"
  [[ -d "$DATA" && -d "$INIT_ADAPTER" && -d "$ADAPTER" ]] || halt "training artifacts are missing"
  [[ -f "$CONTEXT_AUDIT" ]] || halt "r3 context audit is missing"
  ensure_v2p10_lineage
  ensure_training_evidence
  wait_for_memory
  run_behavior_poststage
  wait_for_memory
  run_portability
  publish_provenance
  wait_for_memory
  run_full300
  audit_goal
}

main "$@"
