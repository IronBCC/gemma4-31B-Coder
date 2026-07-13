#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MINI="${MINI:-$ROOT/.venv-eval/bin/mini-extra}"
EVAL_PY="${EVAL_PY:-$ROOT/.venv-eval/bin/python}"

NAME="${NAME:-gemma4-agentic}"
PORT="${PORT:-8010}"
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-0:5}"
WORKERS="${WORKERS:-1}"
OUT="${OUT:-$ROOT/runs/smoke_single_${NAME}_$(date +%Y%m%d_%H%M%S)}"
SCORE="${SCORE:-1}"
CONFIG="${CONFIG:-swebench_edit_first.yaml}"

# Mini-SWE resolves config specs from its invocation directory.  This wrapper
# is normally launched from the repository root, while its checked-in config
# lives beside the wrapper, so resolve only a missing relative spec there.
if [[ "$CONFIG" != /* && ! -f "$CONFIG" && -f "$HERE/$CONFIG" ]]; then
  CONFIG="$HERE/$CONFIG"
fi

mkdir -p "$OUT"

echo "### mini-SWE smoke name=$NAME port=$PORT subset=$SUBSET split=$SPLIT slice=$SLICE workers=$WORKERS config=$CONFIG -> $OUT ###"
PYTHONPATH="$HERE" MSWEA_SILENT_STARTUP=1 "$MINI" swebench \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --slice "$SLICE" \
  --workers "$WORKERS" \
  --environment-class docker \
  --model-class vllm_direct_model.VllmDirectModel \
  -m "openai/$NAME" \
  -c "$CONFIG" \
  -c model.model_kwargs.api_base="http://localhost:$PORT/v1" \
  -c model.model_kwargs.temperature=0 \
  -o "$OUT/$NAME" \
  > "$OUT/$NAME.gen.log" 2>&1

pred="$OUT/$NAME/preds.json"
if [[ "$SCORE" == "1" && -f "$pred" ]]; then
  dataset="princeton-nlp/SWE-bench_Lite"
  if [[ "$SUBSET" == "verified" ]]; then
    dataset="princeton-nlp/SWE-bench_Verified"
  fi
  echo "### scoring $pred dataset=$dataset ###"
  (
    cd "$ROOT" &&
    "$EVAL_PY" -m swebench.harness.run_evaluation \
      --dataset_name "$dataset" \
      --predictions_path "$pred" \
      --run_id "smoke_$NAME" \
      --report_dir "$OUT/report_$NAME"
  ) > "$OUT/$NAME.score.log" 2>&1 || echo "score failed; see $OUT/$NAME.score.log"
else
  echo "### scoring skipped or missing preds: $pred ###"
fi

echo "### smoke complete -> $OUT ###"
