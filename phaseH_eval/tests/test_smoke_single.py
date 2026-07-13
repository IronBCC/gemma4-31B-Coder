from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseH_eval" / "smoke_single.sh"
CONFIG = ROOT / "phaseH_eval" / "swebench_edit_first.yaml"


class SmokeSingleConfigPathTests(unittest.TestCase):
    def _run_smoke_with_fake_mini(self, config: str | None) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini = tmp_path / "mini-extra"
            captured = tmp_path / "captured-config"
            mini.write_text(
                "#!/usr/bin/env bash\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  if [[ $1 == -c ]]; then shift; printf '%s' \"$1\" > \"$CAPTURED_CONFIG\"; exit 0; fi\n"
                "  shift\n"
                "done\n"
            )
            mini.chmod(0o755)

            env = os.environ | {
                "MINI": str(mini),
                "CAPTURED_CONFIG": str(captured),
                "OUT": str(tmp_path / "out"),
                "SCORE": "0",
            }
            if config is not None:
                env["CONFIG"] = config
            else:
                env.pop("CONFIG", None)
            result = subprocess.run(
                ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            return captured.read_text()

    def test_resolves_relative_config_from_repository_root(self) -> None:
        self.assertEqual(
            self._run_smoke_with_fake_mini("swebench_edit_first.yaml"), str(CONFIG)
        )

    def test_defaults_to_edit_first_config(self) -> None:
        self.assertEqual(self._run_smoke_with_fake_mini(None), str(CONFIG))


if __name__ == "__main__":
    unittest.main()
