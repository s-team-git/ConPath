from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import torch
except (ImportError, OSError) as error:
    torch = None
    TORCH_ERROR = str(error)
else:
    TORCH_ERROR = ""


@unittest.skipIf(torch is None, f"PyTorch unavailable: {TORCH_ERROR}")
class CoherentCategoricalTest(unittest.TestCase):
    """CPU-only synthetic checks; every frequency is over independent worlds.

    Spatial cells are tested separately, never pooled as independent replicates.
    The fixed seeds and conservative per-cell sampling tolerances avoid pretending
    that these finite checks prove exact probabilities or downstream improvement.
    """

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def decoder(self, classes=2, kernel_size=9):
        from pathrel.coherent_categorical import CoherentCategoricalDecoder

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3201)
            return CoherentCategoricalDecoder(
                4, num_classes=classes, latent_dim=2, gumbel_kernel_size=kernel_size
            )

    def test_candidate_configuration_and_parent_marginals_are_preserved(self):
        from pathrel.coherent_categorical import CoherentCategoricalDecoder
        from pathrel.stochastic_decoder import CorrelatedCategoricalDecoder

        candidate = self.decoder()
        parent = CorrelatedCategoricalDecoder(4, latent_dim=2)
        parent.load_state_dict({k: v for k, v in candidate.state_dict().items() if k != "gumbel_kernel"})
        self.assertEqual(candidate.gumbel_kernel_size, 9)
        self.assertEqual(
            {name: tuple(p.shape) for name, p in candidate.named_parameters()},
            {name: tuple(p.shape) for name, p in parent.named_parameters()},
        )
        features = torch.linspace(-1, 1, 80).reshape(1, 4, 4, 5)
        with torch.no_grad():
            base = parent(features, num_samples=6, generator=torch.Generator().manual_seed(27))
            changed = candidate(features, num_samples=6, generator=torch.Generator().manual_seed(27))
        for name in ("mean_logits", "factor_maps", "local_scale", "sample_logits", "conditional_class_probs"):
            torch.testing.assert_close(getattr(changed, name), getattr(base, name), atol=0, rtol=0)
        torch.testing.assert_close(changed.conditional_class_probs, changed.sample_logits.softmax(2))
        with self.assertRaisesRegex(ValueError, "synthetic control"):
            CoherentCategoricalDecoder(4, gumbel_kernel_size=5)
        for scale in (0, 0.5, 2, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "scale=1.0"):
                candidate(features, num_samples=2, categorical_noise_scale=scale)

    def test_small_maps_have_unit_variance_at_every_edge_and_interior_cell(self):
        decoder = self.decoder()
        worlds = 6000
        for height, width in ((1, 1), (3, 4), (9, 10)):
            with self.subTest(height=height, width=width):
                reference = torch.empty(1, worlds, 2, height, width)
                field = decoder._sample_standard_normal(reference, generator=torch.Generator().manual_seed(720))
                self.assertTrue(torch.isfinite(field).all())
                self.assertLess(float(field.mean(1).abs().max()), 0.08)
                self.assertLess(float((field.var(1, unbiased=True) - 1).abs().max()), 0.13)
        # Exercise saturated CDF tails, including low-precision references, directly.
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            reference = torch.empty(1, 2, 2, 1, 1, dtype=dtype)
            work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            extremes = torch.tensor([-100.0, 100.0, -9.0, 9.0], dtype=work_dtype).reshape(reference.shape)
            with patch.object(decoder, "_sample_standard_normal", return_value=extremes):
                gumbel = decoder._sample_gumbel(reference)
            self.assertEqual(gumbel.dtype, work_dtype)
            self.assertTrue(torch.isfinite(gumbel).all())
            self.assertLess(float(gumbel.max() - gumbel.min()), 20)

    def test_binary_and_three_class_frequencies_match_each_cells_softmax(self):
        worlds, height, width = 12000, 3, 4
        x = torch.linspace(-2.5, 2.5, height * width).reshape(height, width)
        for classes in (2, 3):
            with self.subTest(classes=classes):
                logits = torch.stack((x, -0.5 * x, torch.full_like(x, 0.6))[:classes])
                reference = logits[None, None].expand(1, worlds, classes, height, width)
                gumbel = self.decoder(classes)._sample_gumbel(reference, generator=torch.Generator().manual_seed(210 + classes))
                indices = (reference + gumbel).argmax(2)
                expected = logits.softmax(0)
                observed = torch.stack([(indices == c).float().mean(1)[0] for c in range(classes)])
                tolerance = 6 * (expected * (1 - expected) / worlds).sqrt() + 2 / worlds
                self.assertTrue(torch.all((observed - expected).abs() < tolerance))

    def test_batch_world_class_and_parent_noise_are_independent(self):
        decoder = self.decoder(classes=3)
        reference = torch.empty(2, 12000, 3, 2, 3)
        field = decoder._sample_standard_normal(reference, generator=torch.Generator().manual_seed(31))
        # Each row is one fixed (batch,class,cell), replicated over independent K.
        series = field[..., 0, 1].permute(0, 2, 1).reshape(6, 12000)
        correlation = torch.corrcoef(series)
        self.assertLess(float((correlation - torch.eye(6)).abs().max()), 0.05)
        for row in series:
            paired_worlds = torch.stack((row[::2], row[1::2]))
            self.assertLess(abs(float(torch.corrcoef(paired_worlds)[0, 1])), 0.08)

        # Capture the new field and compare it against the actual existing local
        # logit noise at one fixed cell. No training or spatial pooling is involved.
        captured = []
        original = decoder._sample_standard_normal

        def capture(*args, **kwargs):
            noise = original(*args, **kwargs)
            captured.append(noise)
            return noise

        with torch.no_grad():
            decoder.factor_head.bias.fill_(0.4)
        for disable_global in (True, False):
            captured.clear()
            with torch.no_grad(), patch.object(decoder, "_sample_standard_normal", side_effect=capture):
                posterior = decoder(
                    torch.ones(1, 4, 1, 1), num_samples=12000,
                    disable_global_factors=disable_global, generator=torch.Generator().manual_seed(83),
                )
            for c in range(3):
                pair = torch.stack((posterior.sample_logits[0, :, c, 0, 0], captured[0][0, :, c, 0, 0]))
                self.assertLess(abs(float(torch.corrcoef(pair)[0, 1])), 0.05)

    def test_known_classes_and_invalid_support_never_flip(self):
        from pathrel.model import PathRelNet

        decoder = self.decoder(classes=3)
        known = torch.tensor([[[0, 1, 2, -1], [2, 0, -1, 1], [1, 2, 0, -1]]])
        with torch.no_grad():
            posterior = decoder(
                torch.ones(1, 4, 3, 4), num_samples=128,
                known_classes=known, generator=torch.Generator().manual_seed(37),
            )
        indices = posterior.sample_probs.argmax(2)
        mask = known >= 0
        expected = known[:, None].expand_as(indices)
        self.assertTrue(torch.equal(indices[mask[:, None].expand_as(indices)], expected[mask[:, None].expand_as(indices)]))

        # Integration uses a new in-memory model only; no frozen source is edited.
        model = PathRelNet(feature_channels=4, latent_dim=2)
        model.decoder = self.decoder()
        observation = torch.zeros(1, 3, 5, 6)
        observation[:, 0, :, 0] = 1
        observation[:, 1, -1] = 1
        observation[:, 0, -1] = 0
        valid = torch.ones(1, 5, 6, dtype=torch.bool)
        valid[:, :, 3] = False
        with torch.no_grad():
            posterior = model(
                observation, valid_support_mask=valid, num_samples=32,
                generator=torch.Generator().manual_seed(42),
            ).posterior
        free = posterior.safe_samples() > 0.5
        self.assertTrue(free[0, :, :-1, 0].all())
        self.assertFalse(free[0, :, -1].any())
        self.assertFalse(free[0, :, :, 3].any())
        self.assertTrue(torch.isfinite(posterior.relaxed_probs).all())

    def test_hard_categories_do_not_depend_on_backward_temperature(self):
        decoder = self.decoder()
        features = torch.linspace(-1, 1, 80).reshape(1, 4, 4, 5)
        samples = []
        with torch.no_grad():
            for temperature in (0.1, 0.7, 5.0, 1e8):
                posterior = decoder(
                    features, num_samples=20, concrete_backward_temperature=temperature,
                    generator=torch.Generator().manual_seed(72),
                )
                self.assertTrue(torch.all((posterior.sample_probs == 0) | (posterior.sample_probs == 1)))
                samples.append(posterior.sample_probs.argmax(2))
            soft = decoder(features, num_samples=2, hard=False, generator=torch.Generator().manual_seed(71))
        self.assertTrue(all(torch.equal(samples[0], sample) for sample in samples[1:]))
        torch.testing.assert_close(soft.sample_probs, soft.relaxed_probs, atol=0, rtol=0)

    def test_straight_through_gradients_are_finite_and_reach_all_heads(self):
        decoder = self.decoder()
        features = torch.linspace(-1, 1, 160).reshape(2, 4, 4, 5).requires_grad_()
        posterior = decoder(features, num_samples=8, generator=torch.Generator().manual_seed(14))
        cell_weights = torch.linspace(-1, 1, 20).reshape(1, 1, 4, 5)
        loss = (posterior.sample_probs[:, :, 0] * cell_weights).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        for head in (decoder.mean_head, decoder.factor_head, decoder.scale_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        self.assertFalse(decoder.gumbel_kernel.requires_grad)

    def test_generator_state_restores_complete_worlds(self):
        decoder = self.decoder()
        features = torch.linspace(-1, 1, 80).reshape(1, 4, 4, 5)
        generator = torch.Generator().manual_seed(151)
        with torch.no_grad():
            decoder(features, num_samples=4, generator=generator)
            state = generator.get_state().clone()
            first = decoder(features, num_samples=4, generator=generator)
            state_after = generator.get_state().clone()
            generator.set_state(state)
            replay = decoder(features, num_samples=4, generator=generator)
        self.assertFalse(torch.equal(state, state_after))
        self.assertTrue(torch.equal(state_after, generator.get_state()))
        for name in ("sample_logits", "conditional_class_probs", "relaxed_probs", "sample_probs"):
            torch.testing.assert_close(getattr(first, name), getattr(replay, name), atol=0, rtol=0)

    def test_correlation_changes_serial_and_parallel_events_in_opposite_directions(self):
        # Two uncertain cells: series requires both open; parallel requires either.
        # This demonstrates a joint-law change, not better real-world predictions.
        reference = torch.zeros(1, 12000, 2, 1, 2)
        results = {}
        for kernel_size in (1, 9):
            gumbel = self.decoder(kernel_size=kernel_size)._sample_gumbel(
                reference, generator=torch.Generator().manual_seed(451)
            )
            free = gumbel.argmax(2)[0, :, 0] == 0
            self.assertTrue(torch.all((free.float().mean(0) - 0.5).abs() < 0.025))
            results[kernel_size] = (
                float(free.all(1).float().mean()), float(free.any(1).float().mean())
            )
        self.assertAlmostEqual(results[1][0], 0.25, delta=0.025)
        self.assertAlmostEqual(results[1][1], 0.75, delta=0.025)
        self.assertGreater(results[9][0], results[1][0] + 0.15)
        self.assertLess(results[9][1], results[1][1] - 0.15)


if __name__ == "__main__":
    unittest.main()
