#!/usr/bin/env bash
# Refresh EAGLE-3 draft head on the FINAL fine-tuned model. Tooling: SpecForge / EAGLE-3.
set -euo pipefail
TARGET="${1:?final gemma4-31b checkpoint}"
TRAIN="${2:-./data/swe_trajectories.jsonl}"
# 1) build training cache from the target's own outputs on your SWE data
python build_eagle3_dataset_cache.py \
  --target-model-path "$TARGET" \
  --train-data-path "$TRAIN" \
  --chat-template gemma-4-thinking --max-length 4096   # confirm long-ctx cap
# 2) train the draft head (cheap relative to main training)
torchrun --nproc_per_node 2 train_eagle3.py \
  --target-model-path "$TARGET" --output-dir ./eagle3-head
