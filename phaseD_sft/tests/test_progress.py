from __future__ import annotations

import unittest

from phaseD_sft.progress import EtaProgress


class EtaProgressTests(unittest.TestCase):
    def test_reports_rate_elapsed_eta_and_total_estimate(self) -> None:
        now = [100.0]
        progress = EtaProgress("format", total=10, clock=lambda: now[0])

        now[0] = 102.0
        report = progress.update(2, force=True)

        self.assertIn("format 2/10", report)
        self.assertIn("rate=1.00 it/s", report)
        self.assertIn("elapsed=2s", report)
        self.assertIn("eta=8s", report)
        self.assertIn("total_est=10s", report)

    def test_avoids_division_by_zero_before_progress(self) -> None:
        progress = EtaProgress("scan", total=5, clock=lambda: 10.0)

        report = progress.update(0, force=True)

        self.assertIn("scan 0/5", report)
        self.assertIn("rate=unknown", report)
        self.assertIn("eta=unknown", report)


if __name__ == "__main__":
    unittest.main()
