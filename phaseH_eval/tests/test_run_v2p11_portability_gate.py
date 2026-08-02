from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "run_v2p11_portability_gate.sh"


def test_portability_launcher_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_interpolation_validator_is_defined_after_sourcing() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$SCRIPT_UNDER_TEST"; '
            "declare -F validate_interpolation_candidate",
        ],
        cwd=ROOT,
        env={**os.environ, "SCRIPT_UNDER_TEST": str(SCRIPT)},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_portability_launcher_uses_stock_controller_free_harness() -> None:
    script = SCRIPT.read_text()

    assert "v2p11_posttrain_complete.json" in script
    assert "v2p11_portability_verified10_ids.json" in script
    assert "teacher_sft_v2p10" in script
    assert "teacher_sft_v2p11" in script
    assert "minisweagent.models.litellm_model.LitellmModel" in script
    assert "config/benchmarks/swebench.yaml" in script
    assert "environment_class: docker" in script
    assert "v2p11_portability_gate.py" in script
    assert "VllmDirectModel" not in script
    assert "DockerSelfRetryEnv" not in script
    assert "pkill" not in script
    assert "pgrep" not in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid' in script
    assert 'nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid' in script
    assert "--query-gpu=index,uuid" not in script
    assert "--query-compute-apps=gpu_uuid,pid" not in script


def test_portability_launcher_preserves_raw_tool_call_evidence() -> None:
    script = SCRIPT.read_text()

    assert "tool_canary.json" in script
    assert "BASH_TOOL" in script
    assert 'calls[0]["function"]["name"] == "bash"' in script
    assert "official_report.json" in script
    assert "eval_manifest.json" in script


def test_portability_launcher_reuses_a_verified_stock_run(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
stock_run_complete() {
  [[ "$1" == control && "$2" == model && "$3" == root ]]
}
run_stock_harness() { echo "unexpected run"; return 1; }
log() { :; }
run_or_reuse_stock_harness control model root
echo reused
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
    assert result.stdout.strip() == "reused"


def test_control_only_mode_does_not_require_candidate_artifacts(
    tmp_path: Path,
) -> None:
    control_model = tmp_path / "control-model"
    control_model.mkdir()
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([f"case-{index}" for index in range(10)]))
    stock_config = tmp_path / "swebench.yaml"
    stock_config.write_text("environment:\n  environment_class: docker\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
prepull_holdout_images() { echo prepull; }
run_or_reuse_stock_harness() { echo "run:$1:$3"; }
validate_stock_run() { echo "validate:$2"; }
check_production() { :; }
wait_for_gpu1_idle() { :; }
main
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
            "PORTABILITY_MODE": "control-only",
            "CONTROL_MODEL": str(control_model),
            "CANDIDATE_MODEL": str(tmp_path / "missing-candidate"),
            "POSTTRAIN_MARKER": str(tmp_path / "missing-posttrain.json"),
            "FINAL_AUDIT": str(tmp_path / "missing-audit.json"),
            "CONTROL_ROOT": str(tmp_path / "control-run"),
            "CANDIDATE_ROOT": str(tmp_path / "candidate-run"),
            "GATE": str(tmp_path / "gate.json"),
            "IDS": str(ids),
            "STOCK_CONFIG": str(stock_config),
            "EVAL_PY": os.environ.get("PYTHON", "/usr/bin/python3"),
            "VLLM": "/usr/bin/true",
            "REQUIRE_PRODUCTION_HEALTH": "0",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "prepull" in result.stdout
    assert "run:teacher_sft_v2p10:" in result.stdout
    assert "validate:teacher_sft_v2p10" in result.stdout
    assert "teacher_sft_v2p11" not in result.stdout


def test_full_mode_validates_posttrain_lineage_before_any_image_pull(
    tmp_path: Path,
) -> None:
    control_model = tmp_path / "control-model"
    candidate_model = tmp_path / "candidate-model"
    control_model.mkdir()
    candidate_model.mkdir()
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([f"case-{index}" for index in range(10)]))
    stock_config = tmp_path / "swebench.yaml"
    stock_config.write_text("environment:\n  environment_class: docker\n")
    posttrain = tmp_path / "posttrain.json"
    posttrain.write_text('{"complete":true}\n')
    final_audit = tmp_path / "audit.json"
    final_audit.write_text("{}\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
validate_posttrain_candidate() { echo validate; }
prepull_holdout_images() { echo prepull; }
run_or_reuse_stock_harness() { echo "run:$1"; }
check_production() { :; }
wait_for_gpu1_idle() { :; }
main
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
            "CONTROL_MODEL": str(control_model),
            "CANDIDATE_MODEL": str(candidate_model),
            "POSTTRAIN_MARKER": str(posttrain),
            "FINAL_AUDIT": str(final_audit),
            "CONTROL_ROOT": str(tmp_path / "control-run"),
            "CANDIDATE_ROOT": str(tmp_path / "candidate-run"),
            "GATE": str(tmp_path / "gate.json"),
            "IDS": str(ids),
            "STOCK_CONFIG": str(stock_config),
            "EVAL_PY": os.environ.get("PYTHON", "/usr/bin/python3"),
            "VLLM": "/usr/bin/true",
            "REQUIRE_PRODUCTION_HEALTH": "0",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert result.stdout.splitlines()[:2] == ["validate", "prepull"]


def test_full_mode_validates_interpolation_before_any_image_pull(
    tmp_path: Path,
) -> None:
    control_model = tmp_path / "control-model"
    candidate_model = tmp_path / "candidate-model"
    anchor_model = tmp_path / "anchor-model"
    source_model = tmp_path / "source-model"
    for model in (control_model, candidate_model, anchor_model, source_model):
        model.mkdir()
    manifest = candidate_model / "interpolation_manifest.json"
    manifest.write_text("{}\n")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([f"case-{index}" for index in range(10)]))
    stock_config = tmp_path / "swebench.yaml"
    stock_config.write_text("environment:\n  environment_class: docker\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
validate_interpolation_candidate() { echo validate; }
prepull_holdout_images() { echo prepull; }
run_or_reuse_stock_harness() { echo "run:$1"; }
check_production() { :; }
wait_for_gpu1_idle() { :; }
main
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
            "LINEAGE_MODE": "interpolation",
            "CONTROL_MODEL": str(control_model),
            "CANDIDATE_MODEL": str(candidate_model),
            "INTERPOLATION_ANCHOR_MODEL": str(anchor_model),
            "INTERPOLATION_SOURCE_MODEL": str(source_model),
            "INTERPOLATION_MANIFEST": str(manifest),
            "FINAL_AUDIT": str(tmp_path / "missing-final-audit.json"),
            "CONTROL_ROOT": str(tmp_path / "control-run"),
            "CANDIDATE_ROOT": str(tmp_path / "candidate-run"),
            "GATE": str(tmp_path / "gate.json"),
            "IDS": str(ids),
            "STOCK_CONFIG": str(stock_config),
            "EVAL_PY": os.environ.get("PYTHON", "/usr/bin/python3"),
            "VLLM": "/usr/bin/true",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert result.stdout.splitlines()[:2] == ["validate", "prepull"]


@pytest.mark.parametrize(
    ("gpu_index", "lineage_mode"),
    [("0", "interpolation"), ("1", "invalid")],
)
def test_invalid_gpu_or_lineage_mode_fails_before_serving(
    tmp_path: Path,
    gpu_index: str,
    lineage_mode: str,
) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
prepull_holdout_images() { echo PREPULL; }
start_owned_serve() { echo SERVE; }
main
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
            "GPU_INDEX": gpu_index,
            "LINEAGE_MODE": lineage_mode,
            "LOG": str(tmp_path / "launcher.log"),
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "PREPULL" not in result.stdout
    assert "SERVE" not in result.stdout


def test_portability_launcher_exposes_vllm_build_tools_on_path() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
VLLM=/opt/vllm/bin/vllm
PATH=/usr/bin
configure_runtime_path
printf '%s\n' "$PATH"
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
    assert result.stdout.strip() == "/opt/vllm/bin:/usr/bin"


def test_portability_launcher_supports_direct_lora_candidate() -> None:
    script = SCRIPT.read_text()

    assert 'LINEAGE_MODE="${LINEAGE_MODE:-posttrain}"' in script
    assert "validate_direct_lora_candidate" in script
    assert "validate_interpolation_candidate" in script
    assert "phaseH_eval/v2p11_interpolation_lineage.py" in script
    assert "posttrain|direct_lora|interpolation" in script
    assert (
        "LINEAGE_MODE must be posttrain, direct_lora, or interpolation"
        in script
    )
    assert "--candidate-name" in script
    assert '"$CANDIDATE_NAME"' in script
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in script


def test_portability_launcher_guards_host_memory_during_owned_serve() -> None:
    script = SCRIPT.read_text()

    assert 'MEMORY_ADMISSION_GIB="${MEMORY_ADMISSION_GIB:-40}"' in script
    assert 'MEMORY_WATCHDOG_GIB="${MEMORY_WATCHDOG_GIB:-12}"' in script
    assert "wait_for_memory_admission" in script
    assert "phaseD_sft/ram_watchdog.py" in script
    assert '--pid "$owned_serve_pid"' in script
    assert "memory_watchdog_pid" in script


def test_stock_harness_is_supervised_by_exact_serve_and_watchdog_pids() -> None:
    script = SCRIPT.read_text()

    assert (
        'EVAL_SUPERVISOR="${EVAL_SUPERVISOR:-'
        '$ROOT/phaseH_eval/run_while_pids_alive.py}"'
    ) in script
    assert 'guard_serve_pid="$owned_serve_pid"' in script
    assert 'guard_watchdog_pid="$memory_watchdog_pid"' in script
    assert '"$EVAL_PY" "$EVAL_SUPERVISOR"' in script
    assert '--watch-pid "serve=$guard_serve_pid"' in script
    assert '--watch-pid "watchdog=$guard_watchdog_pid"' in script
    assert 'eval_status=$?' in script
    assert 'if (( eval_status != 0 )); then' in script
    assert 'stock Mini-SWE failed status=$eval_status' in script


def test_owned_serve_shutdown_logs_original_process_statuses() -> None:
    script = SCRIPT.read_text()

    assert 'serve_pid="$owned_serve_pid"' in script
    assert 'watchdog_pid="$memory_watchdog_pid"' in script
    assert 'serve_status=$serve_status' in script
    assert 'watchdog_status=$watchdog_status' in script


def test_portability_launcher_is_hard_bound_to_gpu1() -> None:
    script = SCRIPT.read_text()

    assert 'GPU_INDEX is fixed to GPU1; GPU0 is unavailable' in script
    assert "REQUIRE_PRODUCTION_HEALTH" not in script
    assert "CUDA_VISIBLE_DEVICES=\"$GPU_INDEX\"" in script


def test_portability_owned_serve_pid_is_retained_when_term_is_ignored(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source "$SCRIPT_UNDER_TEST"
wait_for_gpu1_idle() { :; }
halt() { return 1; }
sleep() { command sleep 0.001; }
bash -c 'trap "" TERM; : > "$1"; while :; do sleep 0.05; done' \
  _ "$READY" >/dev/null 2>&1 &
owned_serve_pid=$!
child_pid=$owned_serve_pid
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
kill -0 "$child_pid" 2>/dev/null && alive=1
printf 'status=%s alive=%s retained=%s\\n' \
  "$status" "$alive" "${owned_serve_pid:+yes}"
kill -KILL "$child_pid"
wait "$child_pid" 2>/dev/null || true
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


def test_portability_owned_serve_pid_is_reaped_and_cleared() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source "$SCRIPT_UNDER_TEST"
wait_for_gpu1_idle() { :; }
sleep 30 &
owned_serve_pid=$!
owned_pid="$owned_serve_pid"
SERVE_STOP_WAIT_LOOPS=100
SERVE_STOP_WAIT_SECONDS=0.01
stop_owned_serve
alive=0
kill -0 "$owned_pid" 2>/dev/null && alive=1
printf 'alive=%s retained=%s\\n' "$alive" "${owned_serve_pid:+yes}"
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
    lines = result.stdout.strip().splitlines()
    assert "serve_status=143" in lines[0]
    assert "watchdog_pid=none watchdog_status=0" in lines[0]
    assert lines[-1] == "alive=0 retained="


def test_portability_cleanup_propagates_bounded_stop_failure() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
stop_owned_serve() { echo bounded-stop; return 1; }
owned_serve_pid=fixture-pid
cleanup
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

    assert result.returncode == 1
    assert result.stdout.splitlines()[0] == "bounded-stop"
    assert "did not stop" in result.stdout


def test_tool_canary_replaces_unbound_resume_artifact(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest_path = run_root / "eval_manifest.json"
    manifest_path.write_text('{"served_name":"candidate"}\n')
    canary_path = run_root / "tool_canary.json"
    canary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "candidate",
                "tool_name": "bash",
                "command": "old",
                "finish_reason": "tool_calls",
                "eval_manifest_sha256": "0" * 64,
            }
        )
        + "\n"
    )
    fake_modules = tmp_path / "fake-modules"
    (fake_modules / "minisweagent/models/utils").mkdir(parents=True)
    for package in [
        "minisweagent",
        "minisweagent/models",
        "minisweagent/models/utils",
    ]:
        (fake_modules / package / "__init__.py").write_text("")
    (fake_modules / "minisweagent/models/utils/actions_toolcall.py").write_text(
        "BASH_TOOL = {'type': 'function'}\n"
    )
    (fake_modules / "openai.py").write_text(
        """
class _Response:
    def model_dump(self, mode):
        return {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "pwd"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "system_fingerprint": "fake",
        }

class _Completions:
    def create(self, **kwargs):
        return _Response()

class _Chat:
    completions = _Completions()

class OpenAI:
    def __init__(self, **kwargs):
        self.chat = _Chat()
"""
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$SCRIPT_UNDER_TEST"
EVAL_PY="$REAL_PY"
run_tool_canary candidate "$RUN_ROOT"
""",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SCRIPT_UNDER_TEST": str(SCRIPT),
            "REAL_PY": sys.executable,
            "RUN_ROOT": str(run_root),
            "PYTHONPATH": str(fake_modules),
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    canary = json.loads(canary_path.read_text())
    assert canary["command"] == "pwd"
    assert canary["eval_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_score_run_rescores_when_prediction_binding_changes(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest_path = run_root / "eval_manifest.json"
    manifest_path.write_text('{"served_name":"candidate"}\n')
    predictions_path = run_root / "preds.json"
    predictions_path.write_text('{"case":{"model_patch":"first"}}\n')
    report_path = run_root / "official_report.json"
    report_path.write_text('{"stale":true}\n')
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" != "-m" ]]; then
  exec "$REAL_PY" "$@"
fi
shift 2
run_id=""
predictions=""
while (( $# )); do
  case "$1" in
    --run_id) run_id="$2"; shift 2 ;;
    --predictions_path) predictions="$2"; shift 2 ;;
    *) shift ;;
  esac
done
name="${run_id#portability_}"
name="${name%_*}"
sha="$("$REAL_PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$predictions")"
printf '{"predictions_sha256":"%s"}\\n' "$sha" > "openai__${name}.${run_id}.json"
printf '%s\\n' "$run_id" >> "$SCORER_CALLS"
"""
    )
    fake_python.chmod(0o755)
    scorer_calls = tmp_path / "scorer-calls"

    def score() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                """
source "$SCRIPT_UNDER_TEST"
ROOT="$TEST_ROOT"
EVAL_PY="$FAKE_PY"
score_run candidate "$RUN_ROOT"
""",
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "SCRIPT_UNDER_TEST": str(SCRIPT),
                "TEST_ROOT": str(tmp_path),
                "RUN_ROOT": str(run_root),
                "FAKE_PY": str(fake_python),
                "REAL_PY": sys.executable,
                "SCORER_CALLS": str(scorer_calls),
            },
            text=True,
            capture_output=True,
        )

    first = score()
    assert first.returncode == 0, first.stderr
    predictions_path.write_text('{"case":{"model_patch":"second"}}\n')
    second = score()
    assert second.returncode == 0, second.stderr

    assert len(scorer_calls.read_text().splitlines()) == 2
    binding = json.loads(
        (run_root / "official_report_binding.json").read_text()
    )
    assert binding["predictions_sha256"] == hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    assert binding["official_report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
