#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_UNIT="${TRAIN_UNIT:-v2p11-submitfix1262-train-gpu1-v2.service}"
TRAIN_INVOCATION_ID="${TRAIN_INVOCATION_ID:?set exact training invocation ID}"
TRAIN_WRAPPER_PID="${TRAIN_WRAPPER_PID:?set exact training wrapper PID}"
TRAIN_PID="${TRAIN_PID:?set exact trainer PID}"
WATCHDOG_PID="${WATCHDOG_PID:?set exact watchdog PID}"
GPU_INDEX="${GPU_INDEX:-1}"
DATA="data/teacher_train_mix_v2p11_submitfix1262"
BASE_DATA="data/teacher_train_mix_v2p10"
ADAPTER="adapters/teacher_sft_v2p11_submitfix1262_bf16"
CANDIDATE_NAME="teacher_sft_v2p11_submitfix1262"
ARTIFACT_TAG="v2p11_submitfix1262"
MERGED="/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11_submitfix1262_full"
MERGE_AUDIT_NAME="v2p11_submitfix1262_merge_audit.json"
CONTEXT_AUDIT="runs/v2p11_submitfix1262_context_audit.json"
TRAIN_LOG_SOURCE="/tmp/train_v2p11_submitfix1262_gpu1.log"
WATCHDOG_LOG_SOURCE="/tmp/ram_watchdog_v2p11_submitfix1262_gpu1.log"
TRAIN_SCRIPT_NAME="phaseH_eval/train_v2p11_successor_gpu1.sh"
DATASET_VALIDATOR_MODULE="phaseH_eval.v2p11_submitfix_completion_provenance"
PROVENANCE_TOOL="phaseH_eval/v2p11_submitfix_completion_provenance.py"
GPU_IDENTITY_ARTIFACT_TYPE="v2p11_submitfix1262_gpu_training_identity"
TRAINING_COMPLETION_ARTIFACT_TYPE="v2p11_submitfix1262_training_completion"
LOG="${LOG:-/tmp/run_v2p11_submitfix1262_posttrain_chain.log}"

export \
  TRAIN_UNIT TRAIN_INVOCATION_ID TRAIN_WRAPPER_PID TRAIN_PID WATCHDOG_PID \
  GPU_INDEX DATA BASE_DATA ADAPTER CANDIDATE_NAME ARTIFACT_TAG MERGED \
  MERGE_AUDIT_NAME CONTEXT_AUDIT TRAIN_LOG_SOURCE WATCHDOG_LOG_SOURCE \
  TRAIN_SCRIPT_NAME DATASET_VALIDATOR_MODULE PROVENANCE_TOOL \
  GPU_IDENTITY_ARTIFACT_TYPE TRAINING_COMPLETION_ARTIFACT_TYPE LOG

exec /usr/bin/bash phaseH_eval/run_v2p11_clean_posttrain_chain.sh
