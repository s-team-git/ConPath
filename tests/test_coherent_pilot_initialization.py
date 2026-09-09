import unittest
import torch

from pathrel.model import PathRelNet
from scripts.train_coherent_parent_pilot import model_for


class MatchedInitialization(unittest.TestCase):
    def test_candidate_keeps_all_initial_parameters_and_global_rng(self):
        for seed in (20260910, 20260911):
            torch.manual_seed(seed)
            baseline = PathRelNet(feature_channels=16, latent_dim=4, local_kernel_size=5)
            expected_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            candidate = model_for('cpu')
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            before, after = dict(baseline.named_parameters()), dict(candidate.named_parameters())
            self.assertEqual(set(before), set(after))
            self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))
            self.assertEqual(sum(p.numel() for p in after.values()), 120108)


if __name__ == '__main__':
    unittest.main()
