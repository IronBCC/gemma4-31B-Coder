from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "run_v2p10_empty_diff_goal.sh"


def test_empty_diff_runner_accepts_checkpoint_name_and_model_overrides() -> None:
    environment = dict(os.environ)
    environment.update(
        MODE="invalid-preflight-only",
        NAME="teacher_sft_v2p11",
        MODEL="/models/teacher_sft_v2p11_full",
    )

    result = subprocess.run(
        ["bash", "-x", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "+ NAME=teacher_sft_v2p11" in result.stderr
    assert "+ MODEL=/models/teacher_sft_v2p11_full" in result.stderr
    assert "starting v2.10 serve" not in result.stdout


def test_empty_diff_runner_accepts_gpu_port_and_reuse_overrides() -> None:
    environment = dict(os.environ)
    environment.update(
        MODE="invalid-preflight-only",
        PORT="8014",
        GPU_INDEX="0",
        REUSE_SERVE="1",
        REUSE_SERVE_PID="12345",
    )

    result = subprocess.run(
        ["bash", "-x", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "+ PORT=8014" in result.stderr
    assert "+ GPU_INDEX=0" in result.stderr
    assert "+ REUSE_SERVE=1" in result.stderr
    assert "+ REUSE_SERVE_PID=12345" in result.stderr


def test_empty_diff_runner_guards_host_memory_during_owned_serve() -> None:
    script = SCRIPT.read_text()

    assert 'MEMORY_ADMISSION_GIB="${MEMORY_ADMISSION_GIB:-40}"' in script
    assert 'MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"' in script
    assert "wait_for_memory_admission" in script
    assert "phaseD_sft/ram_watchdog.py" in script
    assert '--pid "$serve_pid"' in script
    assert "memory_watchdog_pid" in script


def test_empty_diff_runner_fails_closed_if_serve_or_watchdog_exits() -> None:
    script = SCRIPT.read_text()

    assert "phaseH_eval/run_while_pids_alive.py" in script
    assert 'guard_serve_pid="$serve_pid"' in script
    assert 'guard_watch_args=(--watch-pid "serve=$guard_serve_pid")' in script
    assert 'guard_serve_pid="$matched_pid"' in script
    assert 'guard_watch_args+=(--watch-pid "watchdog=$memory_watchdog_pid")' in script
    assert "watchdog for owned serve PID $pid exited status=$watchdog_status" in script
    assert "watchdog_pid=$owned_watchdog_pid" in script
    assert 'owned serve PID $serve_pid did not stop' not in script


def test_empty_diff_runner_is_hard_bound_to_gpu1() -> None:
    script = SCRIPT.read_text()

    assert 'GPU_INDEX is fixed to GPU1; GPU0 is unavailable' in script
    assert "REQUIRE_PRODUCTION_HEALTH" not in script
    assert "CUDA_VISIBLE_DEVICES=\"$GPU_INDEX\"" in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid' in script
    assert "--query-gpu=index,uuid" not in script
    assert "--query-compute-apps=gpu_uuid,pid" not in script


def test_empty_diff_runner_stops_before_production_checks_for_missing_ids() -> None:
    environment = dict(os.environ)
    environment.update(
        MODE="primary",
        IDS="/definitely/missing/ids.json",
        EXPECTED_IDS="1",
        MODEL="/definitely/missing/model",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ID artifact count mismatch" in result.stdout
    assert "Failed to connect" not in result.stderr


def test_empty_diff_runner_stops_before_production_checks_for_missing_model(
    tmp_path: Path,
) -> None:
    ids = tmp_path / "ids.json"
    ids.write_text('["case-a"]\n')
    environment = dict(os.environ)
    environment.update(
        MODE="primary",
        IDS=str(ids),
        EXPECTED_IDS="1",
        MODEL="/definitely/missing/model",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "model directory is missing" in result.stdout
    assert "Failed to connect" not in result.stderr


def test_empty_diff_runner_rejects_a_nonpositive_seed_before_production_checks(
    tmp_path: Path,
) -> None:
    ids = tmp_path / "ids.json"
    ids.write_text('["case-a"]\n')
    environment = dict(os.environ)
    environment.update(
        MODE="primary",
        IDS=str(ids),
        EXPECTED_IDS="1",
        SEED="0",
        MODEL="/definitely/missing/model",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SEED must be a positive integer" in result.stdout
    assert "model directory is missing" not in result.stdout


def test_empty_diff_runner_rejects_gpu0_before_io(
    tmp_path: Path,
) -> None:
    ids = tmp_path / "ids.json"
    ids.write_text('["case-a"]\n')
    environment = dict(os.environ)
    environment.update(
        MODE="primary",
        IDS=str(ids),
        EXPECTED_IDS="1",
        GPU_INDEX="0",
        MODEL="/definitely/missing/model",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "GPU_INDEX is fixed to GPU1; GPU0 is unavailable" in result.stdout
    assert "model directory is missing" not in result.stdout


def test_owned_serve_pid_is_retained_when_term_does_not_stop_process(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
eval "$(awk '
  /^stop_owned_serve\\(\\) \\{/ { capture=1 }
  capture { print }
  capture && /^}/ { exit }
' "$SCRIPT_UNDER_TEST")"
bash -c 'trap "" TERM; : > "$1"; while :; do sleep 0.05; done' _ "$READY" &
serve_pid=$!
for _ in $(seq 1 100); do
  [[ -f "$READY" ]] && break
  sleep 0.01
done
SERVE_STOP_WAIT_LOOPS=2
SERVE_STOP_WAIT_SECONDS=0.01
if stop_owned_serve; then
  status=0
else
  status=$?
fi
alive=0
kill -0 "$serve_pid" 2>/dev/null && alive=1
printf 'status=%s alive=%s retained=%s\\n' \
  "$status" "$alive" "${serve_pid:+yes}"
kill -KILL "$serve_pid"
wait "$serve_pid" 2>/dev/null || true
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
            "READY": str(ready),
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "status=1 alive=1 retained=yes"


def test_owned_serve_pid_is_reaped_and_cleared_after_term() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
eval "$(awk '
  /^stop_owned_serve\\(\\) \\{/ { capture=1 }
  capture { print }
  capture && /^}/ { exit }
' "$SCRIPT_UNDER_TEST")"
sleep 30 &
serve_pid=$!
owned_pid="$serve_pid"
SERVE_STOP_WAIT_LOOPS=100
SERVE_STOP_WAIT_SECONDS=0.01
stop_owned_serve
alive=0
kill -0 "$owned_pid" 2>/dev/null && alive=1
printf 'alive=%s retained=%s\\n' "$alive" "${serve_pid:+yes}"
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "alive=0 retained="
