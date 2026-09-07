import unittest

import numpy as np

from pathrel.labels import clearance_radius_map, maximum_clearance_map
from pathrel.posterior_audits import shuffle_worlds_by_cell, threshold_supported_mean_map


class PosteriorAuditTest(unittest.TestCase):
    def test_observed_free_outside_support_cannot_bridge_disconnected_regions(self):
        probability = np.ones((5, 9))
        observation = np.zeros((3, 5, 9))
        observation[0] = 1
        valid = np.ones((5, 9), dtype=bool)
        valid[:, 4] = False
        world = threshold_supported_mean_map(probability, observation, valid)
        clearance = clearance_radius_map(world)
        scores = maximum_clearance_map(world, (2, 1), clearance=clearance)
        self.assertEqual(scores[2, 7], -1)
        self.assertGreaterEqual(scores[2, 3], 0)

    def test_observed_blocked_wins_over_conflicting_free_evidence(self):
        observation = np.zeros((3, 5, 9))
        observation[0] = 1
        observation[1, :, 4] = 1
        world = threshold_supported_mean_map(np.ones((5, 9)), observation, np.ones((5, 9), dtype=bool))
        scores = maximum_clearance_map(world, (2, 1))
        self.assertEqual(scores[2, 7], -1)

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
