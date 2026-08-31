from __future__ import annotations

import unittest

import numpy as np


try:
    import torch
except (ImportError, OSError) as error:  # Current workspace Python has a broken Torch DLL.
    torch = None
    TORCH_ERROR = str(error)
else:
    TORCH_ERROR = ""


@unittest.skipIf(torch is None, f"PyTorch unavailable: {TORCH_ERROR}")
class ReachabilityLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        from pathrel.reachability import multi_radius_reachability

        self.compute = multi_radius_reachability

    def test_open_blocked_and_radius_monotonicity(self) -> None:
        safe = torch.ones((1, 2, 11, 11), dtype=torch.float32)
        starts = torch.tensor([[[5, 2]]])
        goals = torch.tensor([[[5, 8]]])
        open_q, _ = self.compute(safe, starts, goals, [0, 1, 2], max_steps=121)
        self.assertTrue(torch.all(open_q[..., :-1] >= open_q[..., 1:]))
        self.assertAlmostEqual(float(open_q[0, 0, 0]), 1.0)

        safe[:, :, :, 5] = 0.0
        blocked_q, _ = self.compute(safe, starts, goals, [0], max_steps=121)
        self.assertAlmostEqual(float(blocked_q[0, 0, 0]), 0.0)

    def test_shared_start_propagation_matches_generic_scores(self) -> None:
        from pathrel.reachability import maxmin_path_scores, maxmin_path_scores_shared_start

        torch.manual_seed(19)
        scores = torch.rand(2, 3, 7, 8)
        starts = torch.tensor([[[2, 1], [2, 1], [2, 1]], [[4, 6], [4, 6], [4, 6]]])
        goals = torch.tensor([[[5, 7], [1, 6], [3, 0]], [[0, 0], [6, 2], [2, 4]]])
        generic = maxmin_path_scores(scores, starts, goals, max_steps=15, backward_temperature=0.0)
        shared = maxmin_path_scores_shared_start(scores, starts, goals, max_steps=15, backward_temperature=0.0)
        torch.testing.assert_close(shared, generic, rtol=0.0, atol=0.0)

    def test_start_goal_symmetry(self) -> None:
        safe = torch.ones((1, 2, 9, 13), dtype=torch.float32)
        start = torch.tensor([[[4, 2]]])
        goal = torch.tensor([[[4, 10]]])
        forward, _ = self.compute(safe, start, goal, [0, 1], max_steps=117)
        reverse, _ = self.compute(safe, goal, start, [0, 1], max_steps=117)
        torch.testing.assert_close(forward, reverse)

    def test_bottleneck_receives_gradient(self) -> None:
        safe = torch.ones((1, 2, 7, 9), dtype=torch.float32, requires_grad=True)
        mask = torch.zeros_like(safe)
        mask[:, :, :, 4] = 1.0
        gate = torch.zeros_like(safe)
        gate[:, :, 3, 4] = 1.0
        constrained = safe * (1.0 - mask) + safe * gate * 0.4
        q, _ = self.compute(
            constrained,
            torch.tensor([[[3, 1]]]),
            torch.tensor([[[3, 7]]]),
            [0],
            max_steps=63,
        )
        q.sum().backward()
        self.assertIsNotNone(safe.grad)
        self.assertGreater(float(safe.grad[:, :, 3, 4].abs().sum()), 0.0)

    def test_high_level_surrogate_focuses_open_event_gradient_on_door(self) -> None:
        hard = torch.ones((1, 1, 7, 9), dtype=torch.float32)
        hard[:, :, :, 4] = 0.0
        hard[:, :, 3, 4] = 1.0
        relaxed = torch.full_like(hard, 0.9)
        relaxed[:, :, :, 4] = 0.1
        relaxed[:, :, 3, 4] = 0.55
        relaxed.requires_grad_()

        probability, events = self.compute(
            hard,
            torch.tensor([[[3, 1]]]),
            torch.tensor([[[3, 7]]]),
            [0],
            surrogate_safe_samples=relaxed,
            max_steps=63,
        )
        self.assertEqual(float(events[0, 0, 0, 0].detach()), 1.0)
        probability.square().sum().backward()
        self.assertIsNotNone(relaxed.grad)
        door_gradient = float(relaxed.grad[0, 0, 3, 4].abs())
        other_wall_gradient = float(relaxed.grad[0, 0, :, 4].abs().sum()) - door_gradient
        self.assertGreater(door_gradient, 0.1)
        self.assertLess(other_wall_gradient, door_gradient * 0.01)

    def test_same_marginals_can_have_different_connectivity(self) -> None:
        distribution_a = torch.zeros((1, 2, 3, 6), dtype=torch.float32)
        distribution_b = torch.zeros_like(distribution_a)
        # Start and goal are always valid. The two cells in series have equal 0.5 marginals.
        distribution_a[:, :, 1, 1] = 1.0
        distribution_a[:, :, 1, 4] = 1.0
        distribution_b[:, :, 1, 1] = 1.0
        distribution_b[:, :, 1, 4] = 1.0
        distribution_a[:, 0, 1, 2:4] = 1.0  # both open or both closed
        distribution_b[:, 0, 1, 2] = 1.0    # exactly one open
        distribution_b[:, 1, 1, 3] = 1.0

        torch.testing.assert_close(
            distribution_a.mean(dim=1)[:, 1, 2:4],
            distribution_b.mean(dim=1)[:, 1, 2:4],
        )
        starts = torch.tensor([[[1, 1]]])
        goals = torch.tensor([[[1, 4]]])
        q_a, _ = self.compute(distribution_a, starts, goals, [0], max_steps=18)
        q_b, _ = self.compute(distribution_b, starts, goals, [0], max_steps=18)
        self.assertAlmostEqual(float(q_a[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(q_b[0, 0, 0]), 0.0)

    def test_binary_torch_layer_matches_numpy_oracle(self) -> None:
        from pathrel.labels import reachability_targets

        rng = np.random.default_rng(101)
        for _ in range(8):
            free = rng.random((8, 9)) > 0.24
            starts_np = np.asarray(((3, 1), (5, 1)), dtype=np.int64)
            goals_np = np.asarray(((3, 7), (5, 7)), dtype=np.int64)
            free[starts_np[:, 0], starts_np[:, 1]] = True
            free[goals_np[:, 0], goals_np[:, 1]] = True
            labels, _ = reachability_targets(free, starts_np, goals_np, [0, 1])

            probability, events = self.compute(
                torch.from_numpy(free.astype(np.float32))[None, None],
                torch.from_numpy(starts_np)[None],
                torch.from_numpy(goals_np)[None],
                [0, 1],
                max_steps=free.size,
            )
            torch.testing.assert_close(
                probability[0], torch.from_numpy(labels.astype(np.float32))
            )
            self.assertTrue(torch.all((events == 0.0) | (events == 1.0)))

    def test_finite_propagation_can_miss_a_long_snake_path(self) -> None:
        safe = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
        safe[:, :, 1, 1:8] = 1.0
        safe[:, :, 3, 1:8] = 1.0
        safe[:, :, 5, 1:8] = 1.0
        safe[:, :, 7, 1:8] = 1.0
        safe[:, :, 2, 7] = 1.0
        safe[:, :, 4, 1] = 1.0
        safe[:, :, 6, 7] = 1.0
        starts = torch.tensor([[[1, 1]]])
        goals = torch.tensor([[[7, 1]]])
        short, _ = self.compute(safe, starts, goals, [0], max_steps=10)
        exact, _ = self.compute(safe, starts, goals, [0], max_steps=81)
        self.assertEqual(float(short[0, 0, 0]), 0.0)
        self.assertEqual(float(exact[0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
