from __future__ import annotations

import unittest

import numpy as np

from pathrel.metrics import expected_calibration_error, false_safe_rate, summarize


class CalibrationMetricsTest(unittest.TestCase):
    def test_perfect_probability_is_calibrated(self) -> None:
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        targets = np.asarray([0.0, 0.0, 1.0, 1.0])
        report = summarize(probabilities, targets)
        self.assertAlmostEqual(report["brier"], 0.0)
        self.assertAlmostEqual(report["ece"], 0.0)
        self.assertAlmostEqual(report["false_safe_rate@0.8"], 0.0)

    def test_false_safe_rate_ignores_low_confidence_predictions(self) -> None:
        probabilities = np.asarray([0.95, 0.7, 0.99])
        targets = np.asarray([0.0, 0.0, 1.0])
        self.assertAlmostEqual(false_safe_rate(probabilities, targets), 0.5)

    def test_empty_calibration_bin_is_reported(self) -> None:
        ece, diagram = expected_calibration_error(np.asarray([0.5]), np.asarray([1.0]), bins=4)
        self.assertAlmostEqual(ece, 0.5)
        self.assertEqual(sum(int(item["count"]) for item in diagram), 1)
        self.assertTrue(np.isnan(diagram[0]["accuracy"]))


if __name__ == "__main__":
    unittest.main()
