from __future__ import annotations

import unittest

import numpy as np

from pathrel.labels import (
    clearance_radius_map,
    merge_tree_bottleneck_scores,
    maximum_clearance_path,
    reachability_targets,
)
from pathrel.synthetic import ambiguous_corridor_scene


class ClearanceOracleTest(unittest.TestCase):
    @staticmethod
    def _brute_reachable(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int], radius: int) -> bool:
        height, width = free.shape
        offsets = [
            (d_row, d_col)
            for d_row in range(-radius, radius + 1)
            for d_col in range(-radius, radius + 1)
            if d_row * d_row + d_col * d_col <= radius * radius
        ]
        center_safe = np.zeros_like(free)
        for row in range(height):
            for col in range(width):
                center_safe[row, col] = all(
                    0 <= row + d_row < height
                    and 0 <= col + d_col < width
                    and free[row + d_row, col + d_col]
                    for d_row, d_col in offsets
                )
        if not center_safe[start] or not center_safe[goal]:
            return False
        frontier = [start]
        visited = {start}
        while frontier:
            row, col = frontier.pop()
            if (row, col) == goal:
                return True
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + d_row, col + d_col)
                if (
                    0 <= neighbor[0] < height
                    and 0 <= neighbor[1] < width
                    and center_safe[neighbor]
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return False

    def test_wall_blocks_and_gap_restores_connectivity(self) -> None:
        free = np.ones((11, 11), dtype=bool)
        free[:, 5] = False
        blocked = maximum_clearance_path(free, (5, 2), (5, 8))
        self.assertFalse(blocked.reachable)
        self.assertEqual(blocked.max_radius_cells, -1)

        free[4:7, 5] = True
        open_result = maximum_clearance_path(free, (5, 2), (5, 8))
        self.assertTrue(open_result.reachable)
        self.assertGreaterEqual(open_result.max_radius_cells, 1)

    def test_clearance_is_symmetric_and_start_equals_goal(self) -> None:
        free = np.ones((9, 13), dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        forward = maximum_clearance_path(free, (4, 2), (4, 10))
        reverse = maximum_clearance_path(free, (4, 10), (4, 2))
        self.assertEqual(forward, reverse)

        clearance = clearance_radius_map(free)
        same = maximum_clearance_path(free, (4, 6), (4, 6), clearance=clearance)
        self.assertTrue(same.reachable)
        self.assertEqual(same.max_radius_cells, int(clearance[4, 6]))

    def test_blocked_endpoint_is_unreachable_even_at_radius_zero(self) -> None:
        free = np.ones((7, 7), dtype=bool)
        free[3, 3] = False
        labels, maximum = reachability_targets(free, [(3, 3)], [(4, 4)], [0, 1])
        np.testing.assert_array_equal(labels, np.asarray([[False, False]]))
        np.testing.assert_array_equal(maximum, np.asarray([-1]))

    def test_radius_targets_are_monotone(self) -> None:
        free = np.ones((15, 15), dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        labels, _ = reachability_targets(free, [(7, 2)], [(7, 12)], [0, 1, 2, 3, 4])
        numeric = labels[0].astype(np.int64)
        self.assertTrue(np.all(np.diff(numeric) <= 0))

    def test_ambiguous_pair_has_same_observation_and_different_topology(self) -> None:
        open_scene = ambiguous_corridor_scene(0)
        closed_scene = ambiguous_corridor_scene(1)
        np.testing.assert_array_equal(open_scene.observation_bev, closed_scene.observation_bev)
        self.assertTrue(np.any(open_scene.reachability_targets != closed_scene.reachability_targets))

    def test_synthetic_footprint_tasks_are_nontrivial(self) -> None:
        labels = np.stack(
            [ambiguous_corridor_scene(index).reachability_targets for index in range(120)]
        )
        positive_rate = labels.reshape(-1, labels.shape[-1]).mean(axis=0)
        self.assertTrue(np.all(positive_rate > 0.1), positive_rate)
        self.assertTrue(np.all(positive_rate < 0.9), positive_rate)
        self.assertTrue(np.all(np.diff(positive_rate) <= 0), positive_rate)

    def test_context_families_change_hidden_prior_without_revealing_door(self) -> None:
        family_zero = [
            ambiguous_corridor_scene(index, context_family=0, template_id=3)
            for index in range(80)
        ]
        family_one = [
            ambiguous_corridor_scene(index, context_family=1, template_id=3)
            for index in range(80)
        ]
        # Repeated worlds for one template have identical visible evidence; only the hidden
        # doorway state changes.
        np.testing.assert_array_equal(family_zero[0].observation_bev, family_zero[1].observation_bev)
        self.assertGreater(np.mean([scene.doorway_is_open for scene in family_one]), 0.65)
        self.assertLess(np.mean([scene.doorway_is_open for scene in family_zero]), 0.35)
        self.assertGreater(
            np.mean([scene.reachability_targets for scene in family_one]),
            np.mean([scene.reachability_targets for scene in family_zero]),
        )

    def test_maximum_clearance_matches_exhaustive_radius_bfs(self) -> None:
        rng = np.random.default_rng(23)
        for _ in range(8):
            free = rng.random((9, 10)) > 0.22
            free[4, 1] = True
            free[4, 8] = True
            start, goal = (4, 1), (4, 8)
            result = maximum_clearance_path(free, start, goal)
            for radius in range(4):
                expected = self._brute_reachable(free, start, goal, radius)
                actual = result.reachable and result.max_radius_cells >= radius
                self.assertEqual(actual, expected)

    def test_merge_tree_matches_exhaustive_threshold_connectivity(self) -> None:
        rng = np.random.default_rng(211)
        for _ in range(6):
            scores = rng.random((7, 8))
            starts = [(1, 1), (3, 1), (5, 2), (2, 6)]
            goals = [(5, 6), (3, 6), (1, 5), (6, 2)]
            actual = merge_tree_bottleneck_scores(scores, starts, goals)
            expected = []
            for start, goal in zip(starts, goals):
                best = 0.0
                for threshold in np.unique(scores)[::-1]:
                    free = scores >= threshold
                    if self._brute_reachable(free, start, goal, radius=0):
                        best = float(threshold)
                        break
                expected.append(best)
            np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1e-12)

    def test_merge_tree_start_equals_goal_returns_node_score(self) -> None:
        scores = np.asarray([[0.2, 0.4], [0.7, 0.1]])
        result = merge_tree_bottleneck_scores(scores, [(1, 0)], [(1, 0)])
        np.testing.assert_allclose(result, np.asarray([0.7]))


if __name__ == "__main__":
    unittest.main()
