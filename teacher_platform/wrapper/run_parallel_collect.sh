#!/usr/bin/env bash
# Launch a bounded number of independent one-pass teacher collectors.
# The caller supplies OPENROUTER_API_KEY through its environment; never store
# credentials in this launcher or the generated task shards.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 TASKS_JSONL OUT_PREFIX [MAX_WORKERS]" >&2
  exit 2
fi

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set by the caller}"

tasks=$1
out_prefix=$2
max_workers=${3:-16}
if ! [[ $max_workers =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_WORKERS must be a positive integer" >&2
  exit 2
fi

root=${TEACHER_PLATFORM_ROOT:-$PWD}
cd "$root"
export PATH="$(cd "$(dirname "$0")" && pwd):$PATH"

python3 -c '
import json
import sys
from pathlib import Path

tasks_path = Path(sys.argv[1])
out_prefix = Path(sys.argv[2])
max_workers = int(sys.argv[3])
rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
workers = min(len(rows), max_workers)
if not workers:
    raise SystemExit("task pool is empty")
for index in range(workers):
    shard = out_prefix.parent / f"{out_prefix.name}_worker_{index + 1}_tasks.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows[index::workers]))
' "$tasks" "$out_prefix" "$max_workers"

workers=$(wc -l < "$tasks")
(( workers < max_workers )) || workers=$max_workers
for index in $(seq 1 "$workers"); do
  shard="${out_prefix}_worker_${index}_tasks.jsonl"
  out_dir="runs/$(basename "$out_prefix")_worker_${index}"
  log="/tmp/$(basename "$out_prefix")_worker_${index}.log"
  nohup .venv-train/bin/python teacher_platform/teacher_platform.py collect \
    --tasks "$shard" \
    --out-dir "$out_dir" \
    --backend openrouter \
    --model poolside/laguna-s-2.1 \
    --max-turns 80 \
    > "$log" 2>&1 &
  echo "worker=$index pid=$! tasks=$shard out=$out_dir log=$log"
done
