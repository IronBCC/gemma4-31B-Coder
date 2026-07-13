#!/usr/bin/env bash
set -uo pipefail

DURATION_SECONDS="${DURATION_SECONDS:-14400}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"
LOG="${LOG:-logs/vllm_kill_watch.log}"
ALLOW_PROD_VLLM_KILL="${ALLOW_PROD_VLLM_KILL:-}"

if [[ "$ALLOW_PROD_VLLM_KILL" != "I_UNDERSTAND" ]]; then
  echo "Refusing to kill production vLLM. Set ALLOW_PROD_VLLM_KILL=I_UNDERSTAND only for an explicit maintenance action." >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
end_at=$(( $(date +%s) + DURATION_SECONDS ))

stamp() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

while (( $(date +%s) < end_at )); do
  mapfile -t pids < <(
    ps -eo pid=,args= |
      awk '
        /VLLM::EngineCore/ { print $1; next }
        /\/vllm\/vllm_env\/bin\/vllm serve/ { print $1; next }
        /\/vllm\/serve\.sh/ { print $1; next }
        /\/vllm\/router_proxy\.py/ { print $1; next }
      '
  )
  if (( ${#pids[@]} )); then
    echo "[$(stamp)] killing ${pids[*]}" >> "$LOG"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
  sleep "$SLEEP_SECONDS"
done

echo "[$(stamp)] watch expired" >> "$LOG"
