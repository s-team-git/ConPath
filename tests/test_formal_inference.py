from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pathrel.formal_inference import (direct_forward, make_models, observation_from_condition,
                                     sample_models, sample_seed)


class LabelFreeSample(SimpleNamespace):
    @property
    def target(self):
        raise AssertionError("Inference accessed a reference map")

    @property
    def targets(self):
        raise AssertionError("Inference accessed event labels")


class FixedCompletion(torch.nn.Module):
    def __init__(self, score):
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor(float(score)))

    def forward(self, condition):
        return (self.score * condition[:, 1:2] + condition[:, :1]) * condition[:, 2:3]


class ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, state, time, condition):
        return torch.zeros_like(state) + self.bias


class FormalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self):
        valid = np.ones((32, 32), dtype=bool); valid[:2] = False
        free = np.zeros_like(valid); free[15, 4] = True
        blocked = np.zeros_like(valid); blocked[20, 4] = True
        hidden = valid & ~free & ~blocked
        return LabelFreeSample(row={"global_id": "obs_000001", "parent_group": "synthetic:a"},
                               condition=np.stack([free, hidden, valid]).astype(np.float32),
                               observation=np.stack([free, blocked, hidden]).astype(np.float32),
                               valid=valid, hidden=hidden, starts=np.array([[15, 4]]),
                               goals=np.array([[15, 24]]), candidate_indices=np.array([0]))

    def test_input_encoding_is_lossless(self):
        sample = self.sample()
        got = observation_from_condition(torch.from_numpy(sample.condition[None]))
        np.testing.assert_array_equal(got[0].numpy(), sample.observation)

    def test_lama_requires_actual_four_members_and_counts_real_forwards(self):
        members = [{"model": FixedCompletion(p)} for p in (.1, .3, .7, .9)]
        report = sample_models(members, "lama", self.sample(), 4)
        self.assertEqual(report["worlds"].shape, (4, 32, 32))
        self.assertEqual(report["efficiency"]["actual_batch_forward_calls"], 4)
        self.assertEqual(report["efficiency"]["actual_model_input_examples"], 4)
        self.assertEqual(report["event_scores"][0, 0], .5)
        with self.assertRaises(ValueError):
            sample_models(members[:1], "lama", self.sample(), 4)
        stage = sample_models(members[:1], "lama", self.sample(), 4, one_member=True)
        self.assertEqual(stage["efficiency"]["actual_world_count"], 1)
        self.assertTrue(stage["efficiency"]["stage_one_member_diagnostic"])

    def test_flow_counts_heun_velocity_cfg_and_actual_batch_calls(self):
        report = sample_models({"model": ZeroVelocity()}, "flow", self.sample(), 4, solver_steps=2)
        stats = report["efficiency"]
        self.assertEqual(stats["actual_batch_forward_calls"], 4)
        self.assertEqual(stats["velocity_evaluations_per_sample"], 4)
        self.assertEqual(stats["actual_model_input_examples"], 4 * 4 * 2)
        self.assertEqual(stats["actual_model_input_examples"], stats["sample_equivalent_forwards"])

    def test_conpath_inference_is_reproducible_and_never_reads_labels(self):
        torch.manual_seed(9)
        model = make_models("conpath", "cpu")
        sample = self.sample()
        a = sample_models(model, "conpath", sample, 15)
        torch.manual_seed(500)
        b = sample_models(model, "conpath", sample, 15)
        np.testing.assert_array_equal(a["worlds"], b["worlds"])
        np.testing.assert_array_equal(a["event_scores"], b["event_scores"])
        self.assertFalse(a["worlds"][:, ~sample.valid].any())
        np.testing.assert_array_equal(a["worlds"][:, ~sample.hidden], np.broadcast_to(sample.observation[0, ~sample.hidden], a["worlds"][:, ~sample.hidden].shape))
        self.assertEqual(a["efficiency"]["actual_world_count"], 4)

    def test_direct_control_has_no_fabricated_maps_or_world_budget(self):
        report = sample_models(make_models("direct_query", "cpu"), "direct_query", self.sample(), 4)
        self.assertIsNone(report["worlds"])
        self.assertIsNone(report["efficiency"]["actual_world_count"])
        self.assertEqual(report["event_scores"].shape, (1, 3))

    def test_direct_geometry_uses_grid_cells_only(self):
        class Inspect(torch.nn.Module):
            def forward(self, observation, starts, goals, legacy_distance, angles, radii):
                torch.testing.assert_close(legacy_distance / 1.2, torch.tensor([[5/120]]))
                return torch.zeros((1, 1, 3))
        direct_forward(Inspect(), torch.zeros(1, 3, 8, 8), torch.tensor([[[1, 1]]]), torch.tensor([[[4, 5]]]))


if __name__ == "__main__":
    unittest.main()
