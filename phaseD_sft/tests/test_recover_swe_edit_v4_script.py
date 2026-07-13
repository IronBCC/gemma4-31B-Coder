from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RECOVERY_SCRIPT = ROOT / "phaseD_sft" / "recover_swe_edit_v4_smoke_first.sh"
KILL_WATCH_SCRIPT = ROOT / "phaseD_sft" / "vllm_kill_watch.sh"


class SweEditV4RecoveryScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recovery = RECOVERY_SCRIPT.read_text(encoding="utf-8")
        cls.kill_watch = KILL_WATCH_SCRIPT.read_text(encoding="utf-8")

    def test_recovery_preserves_prod_and_requires_health(self):
        self.assertIn('PROD_PORTS="${PROD_PORTS:-8000 8101 8103 8104}"', self.recovery)
        self.assertIn("check_prod_health", self.recovery)
        self.assertNotIn("check_vllm_not_resident", self.recovery)
        self.assertNotIn("ALLOW_WITH_VLLM", self.recovery)

    def test_recovery_uses_gpu1_and_cgroup_memory_limits(self):
        self.assertIn("CUDA_VISIBLE_DEVICES=1", self.recovery)
        self.assertNotIn("UNSLOTH_COMPILE_DISABLE", self.recovery)
        self.assertIn('TRAIN_ATTN_IMPLEMENTATION="${TRAIN_ATTN_IMPLEMENTATION:-flex_attention}"', self.recovery)
        self.assertIn("--attn-implementation '$TRAIN_ATTN_IMPLEMENTATION'", self.recovery)
        self.assertIn('TORCH_LOGS="${TORCH_LOGS:-}"', self.recovery)
        self.assertIn("if [[ -n '$TORCH_LOGS' ]]; then torch_logs_arg=", self.recovery)
        self.assertIn("\\$torch_logs_arg", self.recovery)
        self.assertNotIn("--setenv=TORCH_LOGS='$TORCH_LOGS'", self.recovery)
        self.assertIn('TRAIN_DEBUG_HF_FLEX_ROUTING="${TRAIN_DEBUG_HF_FLEX_ROUTING:-}"', self.recovery)
        self.assertIn("--debug-hf-flex-routing", self.recovery)
        self.assertIn("systemd-run --user", self.recovery)
        self.assertIn('TRAIN_MEMORY_HIGH="${TRAIN_MEMORY_HIGH:-40G}"', self.recovery)
        self.assertIn('TRAIN_MEMORY_MAX="${TRAIN_MEMORY_MAX:-48G}"', self.recovery)
        self.assertIn('TRAIN_SWAP_MAX="${TRAIN_SWAP_MAX:-32G}"', self.recovery)
        self.assertIn('MIN_MEM_AVAILABLE_MB="${MIN_MEM_AVAILABLE_MB:-60000}"', self.recovery)

    def test_low_ram_mode_aborts_only_below_two_gib_available_ram(self):
        self.assertIn('ALLOW_64G_TRAINING="${ALLOW_64G_TRAINING:-}"', self.recovery)
        self.assertIn('[[ "$ALLOW_64G_TRAINING" == "I_ACCEPT_BOUNDED_SWAP" ]]', self.recovery)
        self.assertIn('MIN_MEM_AVAILABLE_MB="${MIN_MEM_AVAILABLE_MB:-36000}"', self.recovery)
        self.assertIn('ABORT_MEM_AVAILABLE_MB="${ABORT_MEM_AVAILABLE_MB:-2048}"', self.recovery)
        self.assertNotIn('ABORT_SWAP_USED_MB=', self.recovery)
        self.assertNotIn('if (( swap_used > ABORT_SWAP_USED_MB ))', self.recovery)
        self.assertIn('TRAIN_MEMORY_HIGH="${TRAIN_MEMORY_HIGH:-36G}"', self.recovery)
        self.assertIn('TRAIN_MEMORY_MAX="${TRAIN_MEMORY_MAX:-45G}"', self.recovery)
        self.assertIn('TRAIN_SWAP_MAX="${TRAIN_SWAP_MAX:-16G}"', self.recovery)
        self.assertIn('SLEEP_SECONDS="${SLEEP_SECONDS:-10}"', self.recovery)
        self.assertIn("check_low_ram_swap_capacity", self.recovery)

    def test_recovery_stops_at_the_one_step_gate(self):
        self.assertIn("launching protected 1-step v4 smoke", self.recovery)
        self.assertNotIn("launching 10-step", self.recovery)
        self.assertNotIn("FULL_MAX_STEPS", self.recovery)

    def test_recovery_requires_explicit_override_for_more_than_one_step(self):
        self.assertIn('TRAIN_STEPS="${TRAIN_STEPS:-1}"', self.recovery)
        self.assertIn('WARMUP_STEPS="${WARMUP_STEPS:-1}"', self.recovery)
        self.assertIn('TRAIN_LR="${TRAIN_LR:-0.0002}"', self.recovery)
        self.assertIn("--max-steps '$TRAIN_STEPS'", self.recovery)
        self.assertIn("--warmup-steps '$WARMUP_STEPS'", self.recovery)
        self.assertIn("--lr '$TRAIN_LR'", self.recovery)
        self.assertIn("checkpoint-${TRAIN_STEPS}", self.recovery)

    def test_recovery_requires_linger_and_retains_completed_unit_metrics(self):
        self.assertIn("check_user_linger", self.recovery)
        self.assertIn("Linger --value", self.recovery)
        self.assertIn("--remain-after-exit", self.recovery)
        self.assertIn("reset-failed '${SMOKE_UNIT}.service'", self.recovery)
        self.assertIn("SubState", self.recovery)

    def test_recovery_reports_model_load_and_per_step_timing(self):
        self.assertIn("model_load_seconds", self.recovery)
        self.assertIn("data_prepare_seconds", self.recovery)
        self.assertIn("inductor_compile_workers", self.recovery)
        self.assertIn("per_step_wall_seconds", self.recovery)
        self.assertIn("report_timing", self.recovery)

    def test_recovery_never_broad_matches_or_kills_trainers(self):
        self.assertNotIn("pgrep -f", self.recovery)
        self.assertNotIn("pkill", self.recovery)
        self.assertNotIn("kill_matching_v4_trainers", self.recovery)

    def test_kill_watch_requires_explicit_emergency_opt_in(self):
        self.assertIn('ALLOW_PROD_VLLM_KILL="${ALLOW_PROD_VLLM_KILL:-}"', self.kill_watch)
        self.assertIn('[[ "$ALLOW_PROD_VLLM_KILL" != "I_UNDERSTAND" ]]', self.kill_watch)


if __name__ == "__main__":
    unittest.main()
