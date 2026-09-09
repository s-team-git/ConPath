"""Synthetic checks for post-hoc subgroup weighting; no data or model access."""
import unittest

import numpy as np

from scripts.diagnose_parent_pilot import summary


class DiagnosticSubgroups(unittest.TestCase):
    def test_equal_parent_weight_and_fractional_risk_ties(self):
        p = np.ones(4)
        y = np.array([0., 1., 1., 1.])
        parents = np.array(['a', 'b', 'b', 'b'])
        result = summary(p, y, parents, np.ones(4, dtype=bool))
        self.assertEqual(result['parents'], 2)
        self.assertEqual(result['events'], 4)
        self.assertAlmostEqual(result['brier'], .5)
        self.assertAlmostEqual(result['risk30'], .5)
        self.assertAlmostEqual(result['positive_fraction'], .5)

    def test_subgroups_renormalize_contributing_parents(self):
        p = np.array([1., 0., 1.])
        y = np.array([0., 0., 1.])
        parents = np.array(['a', 'a', 'b'])
        result = summary(p, y, parents, np.array([False, True, True]))
        self.assertEqual(result['parents'], 2)
        self.assertEqual(result['events'], 2)
        self.assertEqual(result['brier'], 0.)
        self.assertEqual(result['risk30'], 0.)
        self.assertIsNone(summary(p, y, parents, np.zeros(3, dtype=bool)))


if __name__ == '__main__':
    unittest.main()
