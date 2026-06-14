#!/usr/bin/env bash
# Phase-0 box setup. Idempotent-ish; confirm CUDA/vLLM versions for your driver.
set -euo pipefail

echo "[1/4] Python venv"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

echo "[2/4] Python deps"
pip install -r requirements-box.txt

echo "[3/4] Sanity: GPU + vLLM"
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  -", torch.cuda.get_device_name(i))
PY

echo "[4/4] Docker check (needed for SWE-bench scoring + gym envs)"
docker --version || echo "WARN: install Docker before running eval/gym collection"

echo "Done. Next: serve models, then ./phaseH_eval/run_baseline.sh"
