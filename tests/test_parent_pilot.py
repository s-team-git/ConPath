import copy
from dataclasses import replace
import unittest

import numpy as np
import torch

from pathrel.parent_pilot_data import PilotSample, canonical_d4_digest, choose_parent_subset, validate_partition
from scripts.evaluate_flatlands_support_clamped import _accelerated_events
from scripts.train_parent_group_pilot import tensor_batch
from scripts.evaluate_parent_group_pilot import paired_interval, risk_at_coverage, rule_world


class ParentPilotTests(unittest.TestCase):
    def rows(self):
        return [dict(global_id=f'obs_{i:06d}', parent_group=f'ScanNet:p{i//2}', source_dataset='ScanNet',
                     archive_split='train', packet_directory=f'train/obs_{i:06d}', candidate_split=split)
                for split, first in [('train', 0), ('calibration', 20), ('validation', 40)] for i in range(first, first+12)]

    def test_parent_selection_is_stable_under_row_permutation(self):
        rows = self.rows()
        sizes = {'train': 3, 'calibration': 2, 'validation': 2}
        a = choose_parent_subset(rows, sizes)
        b = choose_parent_subset(list(reversed(rows)), sizes)
        self.assertEqual(a, b)
        self.assertEqual(len({r['parent_group'] for r in a}), 7)

    def test_relabeling_physical_test_cannot_unlock_pilot(self):
        rows = self.rows()
        rows[0]['archive_split'] = 'test'
        with self.assertRaises(ValueError):
            validate_partition(rows)
        rows[0]['archive_split'] = 'train'
        rows[0]['packet_directory'] = 'test/obs_000000'
        with self.assertRaises(ValueError):
            validate_partition(rows)

    def test_rescan_parent_cannot_cross_splits(self):
        rows = self.rows()
        rows[-1]['parent_group'] = rows[0]['parent_group']
        with self.assertRaises(ValueError):
            validate_partition(rows)

    def test_duplicate_geometry_detects_rotated_and_mirrored_maps(self):
        rng = np.random.default_rng(71)
        a = rng.integers(0, 2, (2, 7, 7), dtype=np.uint8)
        self.assertEqual(canonical_d4_digest(a), canonical_d4_digest(np.rot90(a, 3, axes=(-2, -1))))
        self.assertEqual(canonical_d4_digest(a), canonical_d4_digest(a[..., ::-1]))
        b = a.copy(); b[0, 0, 0] ^= 1
        self.assertNotEqual(canonical_d4_digest(a), canonical_d4_digest(b))

    def test_obstacle_goals_are_negative_events_not_missing_labels(self):
        world = np.ones((7, 7), dtype=bool); world[3, 5] = False
        values = _accelerated_events(world[None], np.array([[3, 3], [3, 3]]), np.array([[3, 5], [3, 4]]), (0, 1, 2))
        self.assertEqual(values.shape, (1, 2, 3))
        self.assertEqual(values[0, 0].tolist(), [False, False, False])
        self.assertEqual(values[0, 1].tolist(), [True, False, False])

    def test_padded_queries_preserve_shared_start_and_zero_weight(self):
        a = np.ones((8, 8), dtype=bool)
        sample = PilotSample({}, np.stack((a, ~a, ~a)).astype(np.float32), a, a, ~a,
                             np.array([[3, 4], [3, 4]]), np.array([[4, 4], [5, 4]]), np.ones((2, 3), dtype=bool), np.array([2, 7]))
        short = replace(sample, starts=sample.starts[:1], goals=sample.goals[:1], targets=sample.targets[:1], candidate_indices=np.array([2]))
        batch, tensors = tensor_batch([sample, short], torch.device('cpu'))
        self.assertEqual(batch['starts'][1].tolist(), [[3, 4], [3, 4]])
        self.assertEqual(batch['query_mask'].tolist(), [[True, True], [True, False]])
        self.assertEqual(tensors['query_mask'].device.type, 'cpu')

    def test_risk_ties_are_fractional_and_label_order_invariant(self):
        self.assertAlmostEqual(risk_at_coverage([.7, .7], [1, 0], [.1, .9]), .9)
        self.assertAlmostEqual(risk_at_coverage([.7, .7], [0, 1], [.9, .1]), .9)
        self.assertAlmostEqual(risk_at_coverage([1, .5, 0], [1, 0, 0], [1, 1, 1]), 0.)

    def test_paired_bootstrap_preserves_parent_pairing(self):
        a = np.linspace(0, .7, 40)
        result = paired_interval(a, a+.08, ['a']*8+['b']*8+['c']*8+['d']*8+['e']*8, np.random.default_rng(3), 200)
        self.assertAlmostEqual(result['other_minus_conpath'], .08)
        np.testing.assert_allclose(result['ci95'], [.08, .08])

    def test_rules_preserve_known_evidence_and_invalid_support(self):
        valid = np.ones((8, 8), dtype=bool); valid[0] = False
        free = np.zeros_like(valid); free[2, 2] = True
        blocked = np.zeros_like(valid); blocked[3, 2] = True
        hidden = valid & ~free & ~blocked
        s = PilotSample({}, np.stack((free, blocked, hidden)).astype(np.float32), valid, free, hidden,
                        np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=bool), np.array([], dtype=np.int64))
        for method in ('all_floor', 'all_blocked', 'nearest_observed'):
            w = rule_world(s, method)
            self.assertFalse(w[:, ~valid].any())
            np.testing.assert_array_equal(w[0, ~hidden], free[~hidden])


if __name__ == '__main__':
    unittest.main()
