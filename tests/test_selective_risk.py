import unittest

import numpy as np

from pathrel.selective_risk import risk_at_coverage


class SelectiveRiskTest(unittest.TestCase):
    def test_ties_are_not_broken_by_label_or_row_order(self):
        for labels in ([1, 0], [0, 1]):
            result = risk_at_coverage(np.array([0.8, 0.8]), np.array(labels), np.ones(2), 0.5)
            self.assertAlmostEqual(result["false_safe_rate"], 0.5)
            self.assertAlmostEqual(result["boundary_acceptance_fraction"], 0.5)

    def test_weighted_boundary_accepts_fraction_of_entire_tie(self):
        result = risk_at_coverage(np.array([0.9, 0.8, 0.8]), np.array([1, 1, 0]), np.array([0.2, 0.2, 0.6]), 0.6)
        self.assertAlmostEqual(result["false_safe_rate"], 0.5)
        self.assertAlmostEqual(result["boundary_acceptance_fraction"], 0.5)

    def test_full_coverage_is_weighted_negative_rate(self):
        result = risk_at_coverage(np.array([1, 0.5, 0]), np.array([0, 1, 0]), np.array([2, 3, 5]), 1.0)
        self.assertAlmostEqual(result["false_safe_rate"], 0.7)

    def test_zero_weight_events_do_not_change_selection(self):
        result = risk_at_coverage(np.array([1, 0.8]), np.array([0, 1]), np.array([0, 2]), 0.5)
        self.assertEqual(result["false_safe_rate"], 0)
        self.assertEqual(result["boundary_probability"], 0.8)

    def test_invalid_inputs_are_rejected(self):
        for coverage in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                risk_at_coverage(np.array([0.5]), np.array([1]), np.array([1]), coverage)
        with self.assertRaises(ValueError):
            risk_at_coverage(np.array([0.5]), np.array([1]), np.array([0]), 0.5)
