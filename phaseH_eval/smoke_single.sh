#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
EVAL_PY="${EVAL_PY:-$ROOT/.venv-eval/bin/python}"

NAME="${NAME:-gemma4-agentic}"
PORT="${PORT:-8010}"
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-0:5}"
FILTER="${FILTER:-}"
WORKERS="${WORKERS:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
OUT="${OUT:-$ROOT/runs/smoke_single_${NAME}_$(date +%Y%m%d_%H%M%S)}"
SCORE="${SCORE:-1}"
CONFIG="${CONFIG:-swebench_edit_first.yaml}"
STEP_LIMIT="${STEP_LIMIT:-}"

# Mini-SWE resolves config specs from its invocation directory.  This wrapper
# is normally launched from the repository root, while its checked-in config
# lives beside the wrapper, so resolve only a missing relative spec there.
if [[ "$CONFIG" != /* && ! -f "$CONFIG" && -f "$HERE/$CONFIG" ]]; then
  CONFIG="$HERE/$CONFIG"
fi

config_environment_class="$(
  sed -n 's/^[[:space:]]*environment_class:[[:space:]]*//p' "$CONFIG" | head -1
)"
if [[ -n "${MINI:-}" ]]; then
  if [[ "$config_environment_class" == "docker_selfretry.DockerSelfRetryEnv" ]]; then
    echo "MINI bypass is unsafe for self-retry config; use the checked-in launcher" >&2
    exit 2
  fi
  mini_cmd=("$MINI")
else
  mini_cmd=("$EVAL_PY" "$HERE/mini_extra_selfretry.py")
fi

mkdir -p "$OUT"

# The harness force-diff/force-submit floors derive from the real step limit, so the
# clamp module must see the same number the agent uses.  STEP_LIMIT overrides the
# config; otherwise read step_limit straight out of the resolved config file.
step_limit_args=()
if [[ -n "$STEP_LIMIT" ]]; then
  step_limit_args+=(-c agent.step_limit="$STEP_LIMIT")
  export MSWEA_STEP_LIMIT="$STEP_LIMIT"
else
  cfg_step_limit="$(sed -n 's/^[[:space:]]*step_limit:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$CONFIG" | head -1)"
  if [[ -n "$cfg_step_limit" ]]; then
    export MSWEA_STEP_LIMIT="$cfg_step_limit"
  fi
fi

selection_args=(--subset "$SUBSET" --split "$SPLIT" --slice "$SLICE")
if [[ -n "$FILTER" ]]; then
  selection_args+=(--filter "$FILTER")
fi

seed_args=()
if [[ -n "${SEED:-}" ]]; then
  seed_args+=(-c model.model_kwargs.seed="$SEED")
fi

# optional sampling overrides; unset -> omitted -> previous behaviour
sampling_args=()
if [[ -n "${TOP_P:-}" ]]; then
  sampling_args+=(-c model.model_kwargs.top_p="$TOP_P")
fi
if [[ -n "${TOP_K:-}" ]]; then
  sampling_args+=(-c model.model_kwargs.top_k="$TOP_K")
fi

echo "### mini-SWE smoke name=$NAME port=$PORT subset=$SUBSET split=$SPLIT slice=$SLICE filter=${FILTER:-<none>} workers=$WORKERS temperature=$TEMPERATURE top_p=${TOP_P:-<default>} config=$CONFIG step_limit=${MSWEA_STEP_LIMIT:-<config>} force_submit=${MSWEA_FORCE_SUBMIT_STEP:-<derived>} -> $OUT ###"
PYTHONPATH="$HERE" MSWEA_SILENT_STARTUP=1 "${mini_cmd[@]}" swebench \
  "${selection_args[@]}" \
  --workers "$WORKERS" \
  --model-class vllm_direct_model.VllmDirectModel \
  -m "openai/$NAME" \
  -c "$CONFIG" \
  -c model.model_kwargs.api_base="http://localhost:$PORT/v1" \
  -c model.model_kwargs.temperature="$TEMPERATURE" \
  ${seed_args[@]+"${seed_args[@]}"} \
  ${sampling_args[@]+"${sampling_args[@]}"} \
  ${step_limit_args[@]+"${step_limit_args[@]}"} \
  -o "$OUT/$NAME" \
  > "$OUT/$NAME.gen.log" 2>&1

pred="$OUT/$NAME/preds.json"
if [[ "$SCORE" != "1" ]]; then
  echo "### scoring skipped (SCORE=$SCORE) ###"
elif [[ ! -f "$pred" ]]; then
  echo "### scoring requested but missing preds: $pred ###"
else
  dataset="princeton-nlp/SWE-bench_Lite"
  if [[ "$SUBSET" == "verified" ]]; then
    dataset="princeton-nlp/SWE-bench_Verified"
  fi
  run_id="smoke_$NAME"
  report_dir="$OUT/report_$NAME"
  report_model="openai/$NAME"
  expected_report="$ROOT/${report_model//\//__}.${run_id//\//__}.json"
  echo "### scoring $pred dataset=$dataset ###"
  if (
    cd "$ROOT" &&
    "$EVAL_PY" -m swebench.harness.run_evaluation \
      --dataset_name "$dataset" \
      --predictions_path "$pred" \
      --run_id "$run_id" \
      --report_dir "$report_dir"
  ) > "$OUT/$NAME.score.log" 2>&1; then
    if [[ ! -f "$expected_report" ]]; then
      echo "score succeeded but expected official report is missing: $expected_report" >&2
      exit 1
    fi
    if ! mkdir -p "$report_dir"; then
      echo "failed to create official report directory: $report_dir" >&2
      exit 1
    fi
    if ! report_tmp="$(mktemp "$report_dir/.official_report.json.tmp.XXXXXX")"; then
      echo "failed to create temporary official report in: $report_dir" >&2
      exit 1
    fi
    if ! cp "$expected_report" "$report_tmp"; then
      rm -f "$report_tmp"
      echo "failed to copy official report from: $expected_report" >&2
      exit 1
    fi
    if ! mv -f "$report_tmp" "$report_dir/official_report.json"; then
      rm -f "$report_tmp"
      echo "failed to retain official report at: $report_dir/official_report.json" >&2
      exit 1
    fi
  else
    echo "score failed; see $OUT/$NAME.score.log"
  fi
fi

echo "### smoke complete -> $OUT ###"
