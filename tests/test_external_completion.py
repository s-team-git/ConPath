import copy
import unittest

import torch

from pathrel.external_completion import clamp_completion, masked_mean, validate_condition
from pathrel.flow_matching import FlowUNet, SpatialCrossAttention, flow_training_loss, sample_heun
from pathrel.lama import LaMaBEV, load_native_lama, make_discriminator, lama_update
from pathrel.external_training import lama_accumulated_update


class ExternalCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def condition(self, size=32):
        c = torch.zeros(1, 3, size, size)
        c[:, 2, 1:-1, 1:-1] = 1
        c[:, 1, 3:-3, 3:-3] = 1
        c[:, 0, 1, 2] = 1
        return c

    def test_equivalent_condition_and_final_projection(self):
        c = self.condition()
        validate_condition(c)
        known_blocked = c[:, 2:3] - c[:, :1] - c[:, 1:2]
        self.assertTrue(((known_blocked == 0) | (known_blocked == 1)).all())
        v = torch.ones(1, 1, 32, 32, requires_grad=True)
        p = clamp_completion(v, c)
        self.assertEqual(p[0, 0, 1, 2], 1)
        self.assertEqual(p[0, 0, 1, 1], 0)
        self.assertEqual(p[0, 0, 0, 0], 0)
        p.sum().backward()
        torch.testing.assert_close(v.grad, c[:, 1:2], rtol=0, atol=0)
        c[:, 0, 4, 4] = 1
        with self.assertRaises(ValueError): validate_condition(c)

    def test_masked_loss_empty_and_observation_equal_weight(self):
        v = torch.tensor([[[[2., 7.]]], [[[4., 4.]]]], requires_grad=True)
        mask = torch.tensor([[[[1., 0.]]], [[[1., 1.]]]])
        self.assertEqual(masked_mean(v, mask), 3.)
        zero = masked_mean(v, torch.zeros_like(mask))
        self.assertEqual(zero, 0)
        zero.backward(); self.assertFalse(v.grad.any())

    def test_native_lama_backbone_and_support(self):
        model = LaMaBEV(ngf=8, n_blocks=1).eval()
        native = load_native_lama()
        self.assertEqual(sum(type(m).__name__ == 'FourierUnit' for m in model.modules()), 2)
        condition = self.condition()
        raw = model.generator(condition)
        expected = clamp_completion(raw, condition)
        torch.testing.assert_close(model(condition), expected, rtol=0, atol=0)
        loss = model(condition)[:, :, 4:28, 4:28].sum()
        loss.backward()
        spectral = [p.grad for n, p in model.named_parameters() if '.fu.conv_layer.' in n]
        self.assertTrue(spectral and all(p is not None and p.abs().sum() > 0 for p in spectral))
        self.assertIn('FFCResNetGenerator', native)

    def test_heun_analytic_solution_projection_and_actual_calls(self):
        condition = self.condition(16)
        class LinearVelocity(torch.nn.Module):
            def __init__(self): super().__init__(); self.calls = 0
            def forward(self, state, t, supplied):
                self.calls += 1
                assert (state[:, :, 0, :] == 0).all()
                assert (state[:, :, 1, 2] == 1).all()
                return state
        model = LinearVelocity()
        noise = torch.full((4, 1, 16, 16), .1)
        values, worlds, cost = sample_heun(model, condition, noise=noise, steps=5, samples=4, guidance=2)
        expected = .1 * (1 + .2 + .5 * .2 ** 2) ** 5
        self.assertAlmostEqual(float(values[0, 0, 5, 5]), expected, places=6)
        self.assertEqual(model.calls, 10)
        self.assertEqual(cost['sample_equivalent_forwards'], 80)
        self.assertEqual(cost['cfg_branches'], 2)
        self.assertFalse(worlds[:, :, 0].any())
        self.assertTrue(worlds[:, :, 1, 2].all())

    def test_flow_objective_is_masked_velocity_mse(self):
        c = self.condition(16)
        target = c[:, 2:3].clone()
        rng = torch.Generator().manual_seed(8)
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(8))
        class Oracle(torch.nn.Module):
            def forward(self, state, t, condition): return target - noise
        loss = flow_training_loss(Oracle(), c, target, rng=rng)
        self.assertEqual(loss, 0)

    def test_full_spatial_cross_attention_uses_condition_and_has_gradients(self):
        torch.manual_seed(11)
        layer = SpatialCrossAttention(8, heads=2)
        x = torch.randn(1, 8, 8, 8)
        c = torch.randn_like(x, requires_grad=True)
        y = layer(x, c)
        y.square().mean().backward()
        self.assertTrue(c.grad.abs().sum() > 0)
        self.assertFalse(torch.equal(y, layer(x, c.detach().flip(-1))))

    def test_checkpoint_recompute_preserves_gradient(self):
        torch.manual_seed(19)
        model = FlowUNet(base_channels=8, heads=2, activation_checkpointing=True).train()
        direct = copy.deepcopy(model); direct.activation_checkpointing = False
        c = self.condition()
        state = torch.randn(1, 1, 32, 32); t = torch.tensor([.45])
        a, b = model(state, t, c), direct(state, t, c)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for p, q in zip(model.parameters(), direct.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_accumulation_single_chunk_matches_existing_gan_update(self):
        torch.manual_seed(23)
        a = LaMaBEV(ngf=8, n_blocks=1)
        da = make_discriminator(ndf=8)
        b, db = copy.deepcopy(a), copy.deepcopy(da)
        oa, oda, ob, odb = [torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=.01) for m in (a, da, b, db)]
        c = self.condition().repeat(2, 1, 1, 1)
        t = c[:, 2:3].clone()
        lama_update(a, da, oa, oda, c, t)
        lama_accumulated_update(b, db, ob, odb, c, t, microbatch=2)
        for original, accumulated in ((a, b), (da, db)):
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, accumulated.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
