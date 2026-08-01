from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "smoke_single.sh"
CONFIG = ROOT / "phaseH_eval" / "swebench_edit_first.yaml"
SELFRETRY_CONFIG = ROOT / "phaseH_eval" / "swebench_edit_first_selfretry.yaml"


class SmokeSingleConfigPathTests(unittest.TestCase):
    def _run_smoke_with_fake_mini(
        self,
        config: str | None,
        temperature: str | None = None,
        filter_pattern: str | None = None,
        seed: str | None = None,
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini = tmp_path / "mini-extra"
            captured = tmp_path / "captured-args"
            mini.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURED_ARGS\"\n"
            )
            mini.chmod(0o755)

            env = os.environ | {
                "MINI": str(mini),
                "CAPTURED_ARGS": str(captured),
                "OUT": str(tmp_path / "out"),
                "SCORE": "0",
            }
            if config is not None:
                env["CONFIG"] = config
            else:
                env.pop("CONFIG", None)
            if temperature is not None:
                env["TEMPERATURE"] = temperature
            if filter_pattern is not None:
                env["FILTER"] = filter_pattern
            if seed is not None:
                env["SEED"] = seed
            else:
                env.pop("SEED", None)
            result = subprocess.run(
                ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            return captured.read_text().splitlines()

    @staticmethod
    def _config_arg(args: list[str]) -> str:
        return args[args.index("-c") + 1]

    def test_resolves_relative_config_from_repository_root(self) -> None:
        self.assertEqual(
            self._config_arg(self._run_smoke_with_fake_mini("swebench_edit_first.yaml")), str(CONFIG)
        )

    def test_defaults_to_edit_first_config(self) -> None:
        self.assertEqual(self._config_arg(self._run_smoke_with_fake_mini(None)), str(CONFIG))

    def test_selfretry_config_rejects_mini_shim_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini = tmp_path / "mini-extra"
            mini.write_text("#!/usr/bin/env bash\nexit 0\n")
            mini.chmod(0o755)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=os.environ
                | {
                    "MINI": str(mini),
                    "CONFIG": str(SELFRETRY_CONFIG),
                    "OUT": str(tmp_path / "out"),
                    "SCORE": "0",
                },
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MINI bypass is unsafe for self-retry config", result.stderr)

    def test_temperature_can_be_overridden(self) -> None:
        args = self._run_smoke_with_fake_mini(None, temperature="0.7")
        self.assertIn("model.model_kwargs.temperature=0.7", args)

    def test_seed_is_forwarded_to_model_kwargs_when_set(self) -> None:
        args = self._run_smoke_with_fake_mini(None, seed="1")
        self.assertIn("model.model_kwargs.seed=1", args)

    def test_seed_is_omitted_when_unset(self) -> None:
        args = self._run_smoke_with_fake_mini(None)
        self.assertFalse(any(arg.startswith("model.model_kwargs.seed=") for arg in args))

    def test_filter_is_forwarded_as_one_exact_argument(self) -> None:
        pattern = "^(case-a|case-b)$"
        args = self._run_smoke_with_fake_mini(None, filter_pattern=pattern)
        self.assertEqual(args[args.index("--filter") + 1], pattern)


class SmokeSingleOfficialReportTests(unittest.TestCase):
    def test_score_disabled_is_not_reported_as_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini = tmp_path / "mini-extra"
            mini.write_text("#!/usr/bin/env bash\nexit 0\n")
            mini.chmod(0o755)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=os.environ
                | {
                    "MINI": str(mini),
                    "OUT": str(tmp_path / "run"),
                    "SCORE": "0",
                },
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("scoring skipped (SCORE=0)", result.stdout)
            self.assertNotIn("missing preds", result.stdout)

    def test_scored_runs_retain_their_own_official_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            name = f"report-retention-{tmp_path.name}"
            mini = tmp_path / "mini-extra"
            evaluator = tmp_path / "fake-python"
            mini.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'preds.json').write_text(json.dumps({'marker': os.environ['REPORT_MARKER']}))\n"
            )
            evaluator.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "pred = Path(sys.argv[sys.argv.index('--predictions_path') + 1])\n"
                "run_id = sys.argv[sys.argv.index('--run_id') + 1]\n"
                "report = Path.cwd() / f\"openai__{os.environ['NAME']}.{run_id}.json\"\n"
                "report.write_text(json.dumps(json.loads(pred.read_text())))\n"
            )
            mini.chmod(0o755)
            evaluator.chmod(0o755)

            root_report = ROOT / f"openai__{name}.smoke_{name}.json"
            run_dirs = [tmp_path / "run-one", tmp_path / "run-two"]
            try:
                for marker, out in zip(("first", "second"), run_dirs, strict=True):
                    env = os.environ | {
                        "MINI": str(mini),
                        "EVAL_PY": str(evaluator),
                        "NAME": name,
                        "OUT": str(out),
                        "REPORT_MARKER": marker,
                        "SCORE": "1",
                    }
                    result = subprocess.run(
                        ["bash", str(SCRIPT)],
                        cwd=ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

                retained = [
                    out / f"report_{name}" / "official_report.json" for out in run_dirs
                ]
                self.assertTrue(retained[0].is_file(), "first run did not retain its report")
                self.assertTrue(retained[1].is_file(), "second run did not retain its report")
                self.assertEqual(json.loads(retained[0].read_text()), {"marker": "first"})
                self.assertEqual(json.loads(retained[1].read_text()), {"marker": "second"})
            finally:
                root_report.unlink(missing_ok=True)

    def test_successful_scorer_fails_clearly_when_report_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            name = f"missing-report-{tmp_path.name}"
            mini = tmp_path / "mini-extra"
            evaluator = tmp_path / "fake-python"
            mini.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'preds.json').write_text(json.dumps({}))\n"
            )
            evaluator.write_text("#!/usr/bin/env bash\nexit 0\n")
            mini.chmod(0o755)
            evaluator.chmod(0o755)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=os.environ
                | {
                    "MINI": str(mini),
                    "EVAL_PY": str(evaluator),
                    "NAME": name,
                    "OUT": str(tmp_path / "run"),
                    "SCORE": "1",
                },
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected official report is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
