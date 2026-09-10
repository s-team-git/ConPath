import unittest

import numpy as np

from scripts.audit_flatlands_formal_stage import (
    compare, explicit_disk_events, independent_event_summary, parent_weights, recipe_outer_seed,
)


class FormalStageAuditTests(unittest.TestCase):
    def test_parent_weights_are_equal_mass_despite_unequal_queries(self):
        weights = parent_weights(["a", "a", "a", "b"])
        np.testing.assert_allclose(weights, [1 / 6, 1 / 6, 1 / 6, 1 / 2])

    def test_false_safe_is_conditional_on_acceptance_and_ties_stay_together(self):
        summary = independent_event_summary([.9, .9, .2], [1, 0, 0], ["a", "b", "c"], "pooled")
        self.assertAlmostEqual(summary["coverage_at_confidence"], 2 / 3)
        self.assertAlmostEqual(summary["false_safe_at_confidence"], .5)
        self.assertAlmostEqual(summary["false_safe_mass"], 1 / 3)
        self.assertEqual(len(summary["false_safe_coverage_curve"]), 3)

    def test_no_acceptance_is_undefined_risk_not_zero(self):
        summary = independent_event_summary([.1, .2], [0, 1], ["a", "b"], "pooled")
        self.assertEqual(summary["coverage_at_confidence"], 0)
        self.assertIsNone(summary["false_safe_at_confidence"])

    def test_ece_is_weighted_global_bin_error(self):
        summary = independent_event_summary([.75, .75, .75], [1, 1, 0], ["a", "a", "b"], "scene_weighted")
        self.assertAlmostEqual(summary["event_ece"], .25)
        self.assertAlmostEqual(summary["event_brier"], .3125)

    def test_integer_receipt_identity_is_not_rounded_to_float(self):
        with self.assertRaises(ValueError):
            compare(2**62 + 1, 2**62, "seed")

    def test_actual_frozen_driver_recipe_uses_seed_without_outer_seed(self):
        driver_recipe = {"method": "flow", "seed": 20260831, "member": 0, "protocol_sha256": "4" * 64}
        self.assertEqual(recipe_outer_seed(driver_recipe), 20260831)
        self.assertEqual(recipe_outer_seed({**driver_recipe, "outer_seed": 20260831}), 20260831)
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            recipe_outer_seed({**driver_recipe, "outer_seed": 20260901})
        with self.assertRaisesRegex(ValueError, "integer training seed"):
            recipe_outer_seed({"method": "flow"})

    def test_explicit_disk_footprint_preserves_negative_goal_and_radius_order(self):
        world = np.ones((9, 9), bool)
        world[4, 7] = False
        events = explicit_disk_events(world[None], np.array([[4, 4], [4, 4]]), np.array([[4, 6], [4, 7]]))
        self.assertEqual(events.tolist(), [[[True, False, False], [False, False, False]]])


if __name__ == "__main__":
    unittest.main()
