from __future__ import annotations

import unittest


try:
    import torch
except (ImportError, OSError) as error:
    torch = None
    TORCH_ERROR = str(error)
else:
    TORCH_ERROR = ""


@unittest.skipIf(torch is None, f"PyTorch unavailable: {TORCH_ERROR}")
class PathRelLossTest(unittest.TestCase):
    def test_posterior_marginal_is_mean_of_probabilities(self) -> None:
        from pathrel.losses import posterior_marginal_nll

        # Two opposite, highly confident latent logits have a 0.5 posterior marginal. Taking
        # softmax of their mean would conceal why the averaging order matters in general.
        logits = torch.tensor(
            [[[[[8.0]], [[-8.0]]], [[[-2.0]], [[2.0]]]]]
        )
        target = torch.tensor([[[0]]])
        probabilities = logits.softmax(dim=2).mean(dim=1)
        expected = -torch.log(probabilities[:, 0, 0, 0])
        torch.testing.assert_close(posterior_marginal_nll(logits, target), expected.mean())

    def test_posterior_marginal_accounts_for_scaled_gumbel_noise(self) -> None:
        from pathrel.losses import posterior_marginal_nll

        logits = torch.tensor([[[[[0.5]], [[-0.5]]]]])
        target = torch.tensor([[[0]]])
        expected_probability = (logits / 0.25).softmax(dim=2)[:, :, 0].mean(dim=1)
        expected = -torch.log(expected_probability).mean()
        torch.testing.assert_close(
            posterior_marginal_nll(logits, target, categorical_noise_scale=0.25),
            expected,
        )

    def test_u_statistic_matches_brier_for_identical_members(self) -> None:
        from pathrel.losses import reachability_brier, reachability_brier_u_statistic

        target = torch.tensor([[[1.0, 0.0]]])
        member = torch.tensor([[[[1.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]]])
        ordinary = reachability_brier(member.mean(dim=1), target)
        unbiased = reachability_brier_u_statistic(member, target)
        torch.testing.assert_close(ordinary, unbiased)

    def test_variogram_is_zero_for_perfect_identical_samples(self) -> None:
        from pathrel.losses import spatial_variogram_score

        target = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        )
        samples = target[:, None].expand(1, 4, 3, 3)
        score = spatial_variogram_score(samples, target, offsets=((0, 1), (1, 0)))
        self.assertAlmostEqual(float(score), 0.0)


if __name__ == "__main__":
    unittest.main()
