import unittest

import numpy as np

from scripts.freeze_flatlands_formal_figures import (bfs_path, clearance, explicit_disk_safe,
                                                    path_is_valid, two_internal_vertex_disjoint_paths)


class FormalFigureGeometryTests(unittest.TestCase):
    def test_two_paths_have_valid_separate_internal_nodes(self):
        world = np.ones((15, 21), bool)
        start, goal = (7, 3), (7, 17)
        paths = two_internal_vertex_disjoint_paths(world, start, goal)
        self.assertIsNotNone(paths)
        a, b = paths
        self.assertTrue(path_is_valid(a, world, start, goal))
        self.assertTrue(path_is_valid(b, world, start, goal))
        self.assertFalse(set(map(tuple, a[1:-1])) & set(map(tuple, b[1:-1])))

    def test_single_cell_bottleneck_is_not_called_two_disjoint_paths(self):
        world = np.ones((15, 21), bool)
        world[:, 10] = False
        world[7, 10] = True
        start, goal = (7, 3), (7, 17)
        self.assertIsNotNone(bfs_path(world, start, goal))
        self.assertIsNone(two_internal_vertex_disjoint_paths(world, start, goal))

    def test_target_free_endpoints_may_still_be_disconnected(self):
        world = np.ones((15, 21), bool)
        world[:, 10] = False
        self.assertIsNone(bfs_path(world, (7, 3), (7, 17)))

    def test_integer_disk_oracle_matches_distance_clearance_including_boundary(self):
        rng = np.random.default_rng(610)
        for n in range(8):
            world = rng.random((21 + n, 25 + n)) > .08
            clear = clearance(world)
            for radius in (0, 1, 2, 4):
                np.testing.assert_array_equal(clear >= radius, explicit_disk_safe(world, radius))

    def test_internal_bottleneck_is_distinguished_from_endpoint_clearance(self):
        world = np.zeros((31, 61), bool)
        world[:, :21] = True
        world[:, 40:] = True
        world[14:17, 21:40] = True
        start, goal = (15, 10), (15, 50)
        safe = explicit_disk_safe(world, 10)
        self.assertTrue(safe[start] and safe[goal])
        self.assertIsNotNone(bfs_path(world, start, goal))
        self.assertIsNone(bfs_path(safe, start, goal))

    def test_path_validator_rejects_jumps_and_repeated_vertices(self):
        world = np.ones((5, 5), bool)
        self.assertFalse(path_is_valid([[2, 1], [2, 3]], world, (2, 1), (2, 3)))
        self.assertFalse(path_is_valid([[2, 1], [2, 2], [2, 1], [2, 2], [2, 3]], world, (2, 1), (2, 3)))


if __name__ == "__main__":
    unittest.main()
