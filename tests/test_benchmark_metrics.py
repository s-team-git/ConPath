import unittest
import numpy as np
from pathrel.benchmark_metrics import masked_map_metrics, require_comparable_protocols


class BenchmarkMetricsTest(unittest.TestCase):
    def test_energy_score_two_world_exact(self):
        # Perfect vs complementary map: E[d(Y,y)]=1/2, half pair distance=1/2.
        w = np.array([[[1, 0]], [[0, 1]]])
        r = masked_map_metrics(w, w[0], np.ones((1, 2)))
        self.assertEqual(r['masked_energy_score'], 0)
        self.assertEqual(r['mean_iou'], .5)
        self.assertEqual(r['sample_vote_cell_brier'], .25)

    def test_mask_and_empty_conventions(self):
        w = np.array([[[0, 1]]])
        r = masked_map_metrics(w, np.zeros((1, 2)), np.array([[1, 0]]))
        self.assertEqual(r['mean_iou'], 1)
        self.assertIsNone(r['masked_energy_score'])
        self.assertIsNone(masked_map_metrics(w, w[0], np.zeros((1, 2)))['mean_iou'])

    def test_oracle_never_changes_first(self):
        w = np.array([[[0, 0]], [[1, 1]]])
        r = masked_map_metrics(w, w[1], np.ones((1, 2)))
        self.assertEqual(r['first_iou'], 0)
        self.assertEqual(r['oracle_best_iou'], 1)

    def test_reject_nonbinary(self):
        with self.assertRaises(ValueError):
            masked_map_metrics(np.array([[[.5]]]), np.ones((1, 1)), np.ones((1, 1)))

    def test_published_number_cannot_bypass_protocol(self):
        fields = ('task', 'dataset_revision', 'split_manifest_sha256', 'input_information',
                  'evaluation_mask', 'metric', 'aggregation', 'sample_selection', 'samples',
              'postprocessing', 'metric_parameters', 'training_protocol')
        p = dict.fromkeys(fields, 'same')
        require_comparable_protocols(p, dict(p))
        for field in fields:
            q = dict(p); q[field] = 'different'
            with self.assertRaises(ValueError): require_comparable_protocols(p, q)
            q = dict(p); del q[field]
            with self.assertRaises(ValueError): require_comparable_protocols(p, q)


if __name__ == '__main__': unittest.main()
