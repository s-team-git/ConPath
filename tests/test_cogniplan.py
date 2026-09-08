"""Scientific compatibility checks for the external native module boundary."""
import ast
import copy
import inspect
import unittest

import numpy as np
import torch

from pathrel.cogniplan import (NATIVE_CONDITIONS, SOURCE_ROOT, encode_partial,
                              encode_target, load_native, native_update,
                              postprocess, predict_four)


class CogniPlanCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.native = load_native()

    def test_native_encoding_preserves_edge_and_start_semantics(self):
        partial = np.full((250, 250), 127, np.uint8)
        partial[0, 0] = 1
        partial[-1, -1] = 255
        x, mask = encode_partial(partial)
        self.assertEqual(tuple(x.shape), (1, 1, 256, 256))
        torch.testing.assert_close(x[0, 0, 3:253, 3:253], torch.tensor(partial.astype(np.float32)).div(255).mul(2).sub(1), rtol=0, atol=0)
        self.assertEqual(x[0, 0, 0, 0], x[0, 0, 3, 3])
        self.assertEqual(mask[0, 0, 0, 0], 0)
        self.assertEqual(mask[0, 0, 100, 100], 1)
        target = np.full((250, 250), 127, np.uint8)
        target[2, 3] = 208
        self.assertEqual(encode_target(target)[0, 0, 5, 6], 1)
        with self.assertRaises(ValueError):
            encode_partial(np.zeros((250, 250), np.uint8))

    def test_postprocess_replays_native_with_all_three_classes(self):
        rng = np.random.default_rng(734)
        partial = rng.choice(np.array([1, 127, 255], np.uint8), (250, 250), p=[.1, .7, .2])
        raw = rng.uniform(-1, 1, (250, 250)).astype(np.float32)
        x, _ = encode_partial(partial)
        ref = self.native["Evaluator"].post_process(torch.tensor(raw)[None, None], x[:, :, 3:253, 3:253]) > 0
        np.testing.assert_array_equal(postprocess(raw, partial, np.ones_like(partial, bool)), ref)

    def test_support_projection_is_last_and_missing_classes_are_defined(self):
        partial = np.full((250, 250), 127, np.uint8)
        partial[120, 120] = 255
        support = np.ones_like(partial, bool)
        support[:, 120] = False
        world = postprocess(np.ones_like(partial, np.float32), partial, support)
        self.assertFalse(world[:, 120].any())
        self.assertTrue(world[:, 119].all())
        world = postprocess(np.full_like(partial, -.9, dtype=np.float32), partial, np.ones_like(support))
        self.assertEqual(int(world.sum()), 1)

    def test_inference_uses_four_fixed_conditions_and_no_ground_truth(self):
        class RecordingGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.seen = []

            def forward(self, x, mask, condition):
                self.seen.append(condition.cpu().tolist()[0])
                return torch.zeros_like(x) + self.weight

        model = RecordingGenerator()
        partial = np.full((250, 250), 127, np.uint8)
        raw, worlds = predict_four(model, partial)
        np.testing.assert_allclose(model.seen, NATIVE_CONDITIONS, rtol=1e-7)
        self.assertEqual(raw.shape, (4, 250, 250))
        self.assertTrue(worlds.all())
        self.assertNotIn("target", inspect.signature(predict_four).parameters)
        self.assertNotIn("layout", inspect.signature(predict_four).parameters)

    def test_optimizer_updates_match_unmodified_official_training_loop(self):
        config = {"cuda": False, "netG": {"input_dim": 1, "ngf": 2},
                  "netD": {"input_dim": 1, "ndf": 2}, "lr": .0001,
                  "beta1": .5, "beta2": .9, "spatial_discounting_gamma": .95,
                  "ae_loss_alpha": 20., "l1_loss_alpha": .5, "f1_loss_alpha": .2,
                  "gan_loss_alpha": .0001, "wgan_gp_lambda": 20., "n_critic": 5,
                  "warmup_iter": 1}
        actual = self.native["Trainer"](config)
        expected = copy.deepcopy(actual)
        parsed = ast.parse((SOURCE_ROOT / "mapinpaint/train.py").read_text())
        block = next(n for n in ast.walk(parsed) if isinstance(n, ast.If)
                     and "iteration <= config['warmup_iter']" == ast.unparse(n.test))
        reference = compile(ast.Module(body=[block], type_ignores=[]), "native_train_loop", "exec")
        x = torch.ones((1, 1, 256, 256))
        mask = torch.zeros_like(x); mask[:, :, 30:220, 30:220] = 1
        x[mask.bool()] = -1. / 255.
        target = torch.ones_like(x); target[:, :, 120:130] = -253. / 255.
        onehot = torch.tensor([[1., 0., 0.]])
        for iteration in (1, 2, 5):
            torch.manual_seed(831)
            native_update(actual, (x, mask, target, onehot), adversarial=iteration > 1,
                          generator_update=iteration % 5 == 0)
            torch.manual_seed(831)
            exec(reference, {"trainer": expected, "trainer_module": expected, "config": config,
                             "iteration": iteration, "x": x, "mask": mask, "ground_truth": target,
                             "map_onehot": onehot, "torch": torch})
            for a, b in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
