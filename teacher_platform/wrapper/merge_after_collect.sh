#!/usr/bin/env bash
# Wait for one named collection batch to finish, then materialize its positive
# and contrastive-negative banks.  Run from the repository root or set
# TEACHER_PLATFORM_ROOT explicitly.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 PROCESS_PATTERN BATCH_GLOB MERGED_OUT_DIR [POLL_SECONDS]" >&2
  exit 2
fi

process_pattern=$1
batch_glob=$2
merged_out=$3
poll_seconds=${4:-60}
if ! [[ $poll_seconds =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

root=${TEACHER_PLATFORM_ROOT:-$PWD}
cd "$root"
while pgrep -f "$process_pattern" >/dev/null; do
  sleep "$poll_seconds"
done

.venv-train/bin/python teacher_platform/teacher_platform.py merge \
  --glob "$batch_glob" \
  --out-dir "$merged_out"
