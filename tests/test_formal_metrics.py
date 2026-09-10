import unittest

import numpy as np
from scipy import ndimage

from pathrel.formal_metrics import (event_report, exact_world_events, map_report,
                                   scene_weights, score_map_case,
                                   fit_calibration_platt, apply_calibration_platt)


class FormalMetricsTests(unittest.TestCase):
    def test_scene_weighting_is_not_pooled_event_weighting(self):
        report = event_report([0, 0, 0, 0], [1, 0, 0, 0], ["a", "b", "b", "b"], [0] * 4)
        self.assertAlmostEqual(report["pooled"]["overall"]["event_brier"], .25)
        self.assertAlmostEqual(report["scene_weighted"]["overall"]["event_brier"], .5)
        self.assertAlmostEqual(report["scene_weighted"]["overall"]["event_ece"], .5)
        np.testing.assert_allclose(scene_weights(["a", "b", "b", "b"]), [.5, 1/6, 1/6, 1/6])

    def test_confidence_point_and_empty_acceptance(self):
        report = event_report([0, .25, .5, .75, 1], [0, 0, 1, 1, 0], ["a"] * 5, [0] * 5)
        summary = report["pooled"]["overall"]
        self.assertAlmostEqual(summary["coverage_at_confidence"], .2)
        self.assertEqual(summary["false_safe_at_confidence"], 1.)
        self.assertAlmostEqual(summary["false_safe_mass"], .2)
        absent = event_report([.75], [1], ["a"], [0])["pooled"]["overall"]
        self.assertEqual(absent["coverage_at_confidence"], 0.)
        self.assertIsNone(absent["false_safe_at_confidence"])

    def test_nll_clipping_and_ece_boundaries(self):
        report = event_report([0, .1, .8, 1], [1, 0, 1, 0], ["a"] * 4, [0] * 4)
        pooled = report["pooled"]["overall"]
        self.assertTrue(np.isfinite(pooled["event_nll"]))
        self.assertAlmostEqual(pooled["event_nll"], (-2 * np.log(1e-6) - np.log(.9) - np.log(.8)) / 4, places=8)
        self.assertEqual([b["events"] for b in pooled["reliability"]], [1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(pooled["accepted_events"], 2)

    def test_curve_ties_do_not_depend_on_target_order(self):
        a = event_report([.9, .9, .2], [1, 0, 0], ["a"] * 3, [0] * 3)
        b = event_report([.9, .9, .2], [0, 1, 0], ["a"] * 3, [0] * 3)
        curve = a["pooled"]["overall"]["false_safe_coverage_curve"]
        self.assertEqual(curve, b["pooled"]["overall"]["false_safe_coverage_curve"])
        self.assertEqual(len(curve), 3)
        self.assertAlmostEqual(curve[1]["coverage"], 2/3)
        self.assertEqual(curve[1]["false_safe"], .5)

    def test_radius_and_class_strata_reweight_their_own_scenes(self):
        report = event_report([.1, .9, .3], [0, 1, 0], ["a", "a", "b"], [0, 10, 0])
        self.assertEqual(report["scene_weighted"]["by_radius"]["10"]["scenes"], 1)
        self.assertEqual(report["scene_weighted"]["by_truth"]["reachable"]["events"], 1)
        absent = report["scene_weighted"]["by_radius_and_truth"]["10"]["unreachable"]
        self.assertEqual(absent["events"], 0)
        self.assertIsNone(absent["event_brier"])
        self.assertAlmostEqual(report["scene_weighted"]["by_radius"]["0"]["event_brier"], .05)

    def test_invalid_scores_labels_radii_and_shapes_fail_closed(self):
        for scores, targets, radii in (([np.nan], [1], [0]), ([1.1], [1], [0]), ([.1], [.5], [0]),
                                      ([.1], [1], [1.5]), ([.1], [1], [-1])):
            with self.assertRaises(ValueError):
                event_report(scores, targets, ["a"], radii)
        with self.assertRaises(ValueError):
            event_report([.1], [1], [""], [0])
        with self.assertRaises(ValueError):
            event_report([.1, .2], [1], ["a"], [0])

    def test_exact_geometry_matches_independent_disk_erosion(self):
        rng = np.random.default_rng(814)
        radii = [0, 1, 2, 4]
        for h, w in ((1, 1), (9, 13), (17, 12)):
            worlds = rng.random((5, h, w)) > .12
            starts = np.column_stack((rng.integers(h, size=50), rng.integers(w, size=50)))
            goals = np.column_stack((rng.integers(h, size=50), rng.integers(w, size=50)))
            actual = exact_world_events(worlds, starts, goals, radii)
            for k, world in enumerate(worlds):
                for ri, radius in enumerate(radii):
                    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
                    eroded = ndimage.binary_erosion(world, structure=x*x+y*y <= radius*radius, border_value=0)
                    # scipy default 2-D connectivity is four-neighbour.
                    labels, _ = ndimage.label(eroded)
                    a, b = labels[tuple(starts.T)], labels[tuple(goals.T)]
                    np.testing.assert_array_equal(actual[k, :, ri], (a > 0) & (a == b))

    def test_event_frequency_is_not_connectivity_of_mean_cell_map(self):
        worlds = np.zeros((2, 3, 6), dtype=bool)
        worlds[:, 1, 1:5] = True
        worlds[0, 1, 2] = False
        worlds[1, 1, 3] = False
        starts, goals = np.array([[1, 1]]), np.array([[1, 4]])
        events = exact_world_events(worlds, starts, goals, [0])
        self.assertEqual(events.mean(), 0.)
        fabricated = exact_world_events((worlds.mean(0) >= .5)[None], starts, goals, [0])
        self.assertTrue(fabricated.item())

    def test_geometry_does_not_truncate_or_wrap_coordinates(self):
        world = np.ones((1, 4, 4))
        for start, goal, radii in (([[1.2, 1]], [[1, 2]], [0]), ([[1, 1]], [[-1, 2]], [0]),
                                   ([[1, 1]], [[4, 2]], [0]), ([[1, 1]], [[1, 2]], [-1]),
                                   ([[1, 1]], [[1, 2]], [0, 0])):
            with self.assertRaises(ValueError):
                exact_world_events(world, start, goal, radii)
        empty = exact_world_events(world, np.empty((0, 2)), np.empty((0, 2)), [0])
        self.assertEqual(empty.shape, (1, 0, 1))

    def case(self):
        support = np.ones((3, 4), dtype=bool); support[0] = False
        hidden = np.zeros_like(support); hidden[1, 1:3] = True
        observed = np.zeros_like(support); observed[2, 1] = True
        target = observed.copy(); target[1, 1] = True
        worlds = np.repeat(observed[None], 4, axis=0)
        worlds[:2, 1, 1] = True
        worlds[:1, 1, 2] = True
        return worlds, target, hidden, support, observed

    def test_hidden_map_metrics_and_native_scores_have_distinct_names(self):
        args = self.case()
        report = score_map_case(*args, continuous_score=args[1].astype(float),
                                continuous_score_semantics="test native completion score")
        self.assertAlmostEqual(report["sample_vote_cell_brier"], (.5**2 + .25**2)/2)
        self.assertAlmostEqual(report["sample_vote_cell_nll"], (-np.log(.5)-np.log(.75))/2)
        self.assertEqual(report["continuous_completion_score"]["hidden_brier"], 0)
        self.assertEqual(report["observed_evidence_violation_count"], 0)
        self.assertEqual(report["valid_support_violation_count"], 0)
        with self.assertRaises(ValueError):
            score_map_case(*args, continuous_score=args[1].astype(float))

    def test_violation_counts_are_not_repaired_and_k1_is_not_duplicated(self):
        worlds, target, hidden, support, observed = self.case()
        worlds[0, 0, 0] = True
        worlds[0, 2, 1] = False
        report = score_map_case(worlds, target, hidden, support, observed)
        self.assertEqual(report["observed_evidence_violation_count"], 1)
        self.assertEqual(report["valid_support_violation_count"], 1)
        deterministic = score_map_case(worlds[:1], target, hidden, support, observed)
        self.assertEqual(deterministic["actual_world_count"], 1)
        self.assertEqual(deterministic["distinct_world_count"], 1)

    def test_map_aggregation_uses_cell_counts_or_scene_weights(self):
        base = score_map_case(*self.case())
        cases = [{**base, "hidden_cells": 2, "sample_vote_cell_brier": 1., "sample_vote_cell_nll": 2.},
                 {**base, "hidden_cells": 6, "sample_vote_cell_brier": 0., "sample_vote_cell_nll": 0.}]
        report = map_report(cases, ["a", "b"])
        self.assertEqual(report["pooled"]["sample_vote_cell_brier"], .25)
        self.assertEqual(report["scene_weighted"]["sample_vote_cell_brier"], .5)

    def test_inconsistent_reference_or_masks_are_rejected(self):
        worlds, target, hidden, support, observed = self.case()
        target[2, 1] = False
        with self.assertRaises(ValueError):
            score_map_case(worlds, target, hidden, support, observed)
        target[2, 1] = True
        hidden[0, 0] = True
        with self.assertRaises(ValueError):
            score_map_case(worlds, target, hidden, support, observed)

    def test_platt_fit_requires_calibration_and_matching_protocol(self):
        kwargs = dict(query_keys=["a/0", "a/1", "b/0", "b/1"], protocol_sha256="a" * 64)
        args = ([0, 0, 1, 1], [0, 1, 0, 1], ["a", "a", "b", "b"])
        with self.assertRaises(ValueError):
            fit_calibration_platt(*args, split="validation", **kwargs)
        fitted = fit_calibration_platt(*args, split="calibration", **kwargs)
        values = apply_calibration_platt([0, .25, .5, .75, 1], fitted, protocol_sha256="a" * 64)
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertFalse(fitted["guaranteed_calibration"])
        self.assertEqual(fitted, fit_calibration_platt(*args, split="calibration", **kwargs))
        with self.assertRaises(ValueError):
            apply_calibration_platt([.5], fitted, protocol_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
