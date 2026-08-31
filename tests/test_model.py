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
class PathRelModelTest(unittest.TestCase):
    def test_shapes_probabilities_and_backward(self) -> None:
        from pathrel.model import PathRelNet

        torch.manual_seed(3)
        model = PathRelNet(input_channels=3, feature_channels=8, latent_dim=4)
        observation = torch.randn(2, 3, 12, 12)
        starts = torch.tensor([[[6, 2]], [[6, 2]]])
        goals = torch.tensor([[[6, 9]], [[6, 9]]])
        output = model(
            observation,
            starts=starts,
            goals=goals,
            footprint_radii_cells=[0, 1],
            num_samples=4,
            max_reachability_steps=48,
            generator=torch.Generator().manual_seed(19),
        )

        self.assertEqual(tuple(output.posterior.mean_logits.shape), (2, 2, 12, 12))
        self.assertEqual(tuple(output.posterior.sample_probs.shape), (2, 4, 2, 12, 12))
        self.assertEqual(tuple(output.reachability.shape), (2, 1, 2))
        torch.testing.assert_close(
            output.posterior.sample_probs.sum(dim=2),
            torch.ones(2, 4, 12, 12),
        )
        self.assertTrue(torch.isfinite(output.reachability).all())

        loss = (
            output.posterior.sample_logits.square().mean()
            + output.posterior.local_scale.mean()
            + output.reachability.mean()
        )
        loss.backward()
        for head in (model.decoder.mean_head, model.decoder.factor_head, model.decoder.scale_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

    def test_seeded_sampling_is_reproducible(self) -> None:
        from pathrel.model import PathRelNet

        torch.manual_seed(5)
        model = PathRelNet(input_channels=3, feature_channels=8, latent_dim=3).eval()
        observation = torch.randn(1, 3, 12, 12)
        with torch.no_grad():
            first = model(
                observation,
                num_samples=3,
                generator=torch.Generator().manual_seed(41),
            ).posterior.sample_probs
            second = model(
                observation,
                num_samples=3,
                generator=torch.Generator().manual_seed(41),
            ).posterior.sample_probs
        torch.testing.assert_close(first, second)

    def test_conditional_probabilities_respect_categorical_noise_scale(self) -> None:
        from pathrel.model import PathRelNet

        model = PathRelNet(input_channels=3, feature_channels=8, latent_dim=3).eval()
        posterior = model(
            torch.randn(1, 3, 12, 12),
            num_samples=3,
            categorical_noise_scale=0.25,
            generator=torch.Generator().manual_seed(43),
        ).posterior
        expected = (posterior.sample_logits / 0.25).softmax(dim=2)
        torch.testing.assert_close(posterior.conditional_class_probs, expected)
        torch.testing.assert_close(
            posterior.posterior_marginal_probs,
            posterior.conditional_class_probs.mean(dim=1),
        )

    def test_observed_cells_remain_deterministic_in_posterior_samples(self) -> None:
        from pathrel.model import PathRelNet

        model = PathRelNet(input_channels=3, feature_channels=8, latent_dim=3).eval()
        observation = torch.zeros(1, 3, 12, 12)
        observation[:, 0, :, :6] = 1.0  # observed traversable half
        observation[:, 1, :, 6:] = 1.0  # observed blocked half
        with torch.no_grad():
            posterior = model(
                observation,
                num_samples=4,
                generator=torch.Generator().manual_seed(91),
            ).posterior
        torch.testing.assert_close(posterior.sample_probs[:, :, 0, :, :6], torch.ones(1, 4, 12, 6))
        torch.testing.assert_close(posterior.sample_probs[:, :, 1, :, 6:], torch.ones(1, 4, 12, 6))

    def test_soft_samples_are_rejected_for_probability_queries(self) -> None:
        from pathrel.model import PathRelNet

        model = PathRelNet(input_channels=3, feature_channels=8, latent_dim=3)
        with self.assertRaisesRegex(ValueError, "event probability"):
            model(
                torch.randn(1, 3, 12, 12),
                starts=torch.tensor([[[6, 2]]]),
                goals=torch.tensor([[[6, 9]]]),
                footprint_radii_cells=[0],
                num_samples=3,
                hard_samples=False,
                max_reachability_steps=24,
            )

    def test_reachability_loss_alone_reaches_all_stochastic_heads(self) -> None:
        from pathrel.losses import reachability_brier_u_statistic
        from pathrel.model import PathRelNet

        torch.manual_seed(13)
        model = PathRelNet(input_channels=3, feature_channels=8, latent_dim=3)
        output = model(
            torch.randn(2, 3, 10, 10),
            starts=torch.tensor([[[5, 2]], [[5, 2]]]),
            goals=torch.tensor([[[5, 7]], [[5, 7]]]),
            footprint_radii_cells=[0],
            num_samples=6,
            max_reachability_steps=100,
            generator=torch.Generator().manual_seed(71),
        )
        target = 1.0 - output.reachability.detach().round()
        loss = reachability_brier_u_statistic(output.sample_reachability, target)
        loss.backward()
        for head in (model.decoder.mean_head, model.decoder.factor_head, model.decoder.scale_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
