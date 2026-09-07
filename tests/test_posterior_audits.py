import unittest

import numpy as np

from pathrel.posterior_audits import shuffle_worlds_by_cell


class PosteriorAuditTest(unittest.TestCase):
    def test_exact_marginals_and_observed_evidence_survive_decorrelation(self):
        worlds = np.zeros((64, 4, 8), dtype=bool)
        worlds[:32] = True
        worlds[:, 0] = True
        worlds[:, -1] = False
        shuffled = shuffle_worlds_by_cell(worlds, np.random.default_rng(13))
        np.testing.assert_array_equal(shuffled.sum(axis=0), worlds.sum(axis=0))
        np.testing.assert_array_equal(shuffled[:, 0], worlds[:, 0])
        np.testing.assert_array_equal(shuffled[:, -1], worlds[:, -1])
        self.assertTrue(np.any(shuffled[:, 1, 0] != shuffled[:, 1, 1]))
        np.testing.assert_array_equal(shuffled, shuffle_worlds_by_cell(worlds, np.random.default_rng(13)))

    def test_nonbinary_or_single_world_controls_are_rejected(self):
        for worlds in [np.zeros((1, 3, 3), dtype=bool), np.zeros((8, 3, 3)), np.zeros((8, 3), dtype=bool)]:
            with self.assertRaises(ValueError):
                shuffle_worlds_by_cell(worlds, np.random.default_rng(3))
