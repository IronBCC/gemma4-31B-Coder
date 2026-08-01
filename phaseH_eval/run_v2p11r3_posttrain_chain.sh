#!/usr/bin/env bash
# Audit, merge, portability-gate, and evaluate the corrected r3 adapter.
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
CANDIDATE_NAME="${CANDIDATE_NAME:-teacher_sft_v2p11r3_v2p10init_fable_reasoned}"
ARTIFACT_TAG="${ARTIFACT_TAG:-v2p11r3_v2p10init_fable_reasoned}"
MERGED="${MERGED:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_v2p10init_fable_reasoned_full}"
MERGE_AUDIT="$MERGED/v2p11r3_merge_audit.json"
CONTEXT_AUDIT="${CONTEXT_AUDIT:-runs/v2p11r3_reasoned_context_audit.json}"
TRAINING_JOURNAL="runs/${ARTIFACT_TAG}_training_journal.log"
WATCHDOG_EVIDENCE="runs/${ARTIFACT_TAG}_watchdog.log"
TRAINING_COMPLETION="runs/${ARTIFACT_TAG}_training_completion.json"
PORTABILITY_ROOT="runs/${ARTIFACT_TAG}_portability_stock"
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
import os
import sys
from pathlib import Path

source_train, source_watchdog, out_train, out_watchdog = map(Path, sys.argv[1:])
for source, output in ((source_train, out_train), (source_watchdog, out_watchdog)):
    payload = source.read_bytes()
    assert payload, source
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
PY
}

audit_training() {
  "$TRAIN_PY" - "$DATA" "$INIT_ADAPTER" "$ADAPTER" \
    "$TRAINING_JOURNAL" "$WATCHDOG_EVIDENCE" "$TRAINING_COMPLETION" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

from safetensors import safe_open

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
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(report, handle, sort_keys=True, indent=2)
    handle.write("\n")
print(json.dumps({"changed_tensor_count": changed, "tensors": total}, sort_keys=True))
PY
}

merge_model() {
  [[ ! -e "$MERGED" ]] || halt "refusing to overwrite merged model: $MERGED"
  "$TRAIN_PY" phaseD_sft/merge_lora_streaming.py \
    --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
    --adapter "$ADAPTER" --out "$MERGED" \
    --group-gb 3 --max-rss-gb 12 --dry-run
  wait_for_memory
  "$TRAIN_PY" phaseD_sft/merge_lora_streaming.py \
    --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
    --adapter "$ADAPTER" --out "$MERGED" \
    --group-gb 3 --max-rss-gb 12 \
    --audit-architecture Gemma4ForConditionalGeneration \
    --audit-tensors 1188 --audit-vision 356 \
    --audit-filename v2p11r3_merge_audit.json
  [[ -f "$MERGE_AUDIT" ]] || halt "merge completed without audit"
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
    MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    GPU_INDEX=1 PORT=8013 \
    /usr/bin/bash phaseH_eval/run_v2p11_portability_gate.sh
}

publish_provenance() {
  "$EVAL_PY" phaseH_eval/v2p11r3_completion_provenance.py \
    --candidate-name "$CANDIDATE_NAME" --dataset "$DATA" \
    --dataset-context-audit "$CONTEXT_AUDIT" --adapter "$ADAPTER" \
    --init-adapter "$INIT_ADAPTER" --training-completion "$TRAINING_COMPLETION" \
    --final-model "$MERGED" --merge-audit "$MERGE_AUDIT" \
    --portability-gate "$PORTABILITY_GATE" \
    --full-ids data/swebench_lite_test_ids.json \
    --v2p10 runs/v2p10_full300_composite.json --out "$PROVENANCE"
}

run_full300() {
  env \
    NAME="$CANDIDATE_NAME" MODEL="$MERGED" ARTIFACT_TAG="$ARTIFACT_TAG" \
    LINEAGE_MODE=direct_lora FINAL_AUDIT="$MERGE_AUDIT" \
    PORTABILITY_MARKER="$PORTABILITY_GATE" PROVENANCE="$PROVENANCE" \
    MEMORY_ADMISSION_GIB="$MEMORY_ADMISSION_GIB" \
    MEMORY_WATCHDOG_GIB="$MEMORY_WATCHDOG_GIB" \
    GPU_INDEX=1 PORT=8013 \
    /usr/bin/bash phaseH_eval/eval_v2p11_full300_after_merge.sh
  [[ -f "$VERDICT" ]] || halt "full300 ended without a verdict"
}

audit_goal() {
  "$EVAL_PY" phaseH_eval/v2p11r3_goal_completion_audit.py \
    --verdict "$VERDICT" --full-ids data/swebench_lite_test_ids.json \
    --v2p10 runs/v2p10_full300_composite.json \
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
  [[ -f "$TRAIN_LOG_SOURCE" && -f "$WATCHDOG_LOG_SOURCE" ]] || halt "training evidence is missing"
  [[ -d "$DATA" && -d "$INIT_ADAPTER" && -d "$ADAPTER" ]] || halt "training artifacts are missing"
  [[ -f "$CONTEXT_AUDIT" ]] || halt "r3 context audit is missing"
  [[ ! -e "$TRAINING_JOURNAL" && ! -e "$WATCHDOG_EVIDENCE" && ! -e "$TRAINING_COMPLETION" ]] ||
    halt "r3 training evidence already exists"
  [[ ! -e "$MERGED" && ! -e "$PORTABILITY_ROOT" && ! -e "$PORTABILITY_GATE" &&
    ! -e "$PROVENANCE" && ! -e "$VERDICT" && ! -e "$GOAL_AUDIT" ]] ||
    halt "r3 posttrain output already exists"
  capture_evidence
  audit_training
  wait_for_memory
  merge_model
  wait_for_memory
  run_portability
  publish_provenance
  wait_for_memory
  run_full300
  audit_goal
}

main "$@"
