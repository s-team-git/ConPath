import unittest

import numpy as np

from pathrel.unscenes3d import deterministic_queries, lidar_observation, occupancy_support


class UnScenesAdapterTest(unittest.TestCase):
    def test_occupancy_support_is_conservative_on_mixed_xy(self) -> None:
        rows = np.array(
            [
                [10, 20, 2, 11],  # driveable voxel
                [10, 20, 4, 1],   # obstacle at the same support cell wins
                [11, 20, 2, 11],
            ],
            dtype=np.int32,
        )
        support = occupancy_support(rows)
        self.assertTrue(support.valid[10, 20])
        self.assertTrue(support.blocked[10, 20])
        self.assertFalse(support.free[10, 20])
        self.assertTrue(support.free[11, 20])

    def test_occupancy_support_rejects_out_of_range_voxels(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            occupancy_support(np.array([[256, 0, 0, 11]], dtype=np.int32))

    def test_lidar_observation_has_exclusive_channels(self) -> None:
        points = np.array(
            [
                [3.0, 0.0, -1.0, 0.1],
                [3.0, 1.0, -1.0, 0.2],
            ],
            dtype=np.float32,
        )
        observation = lidar_observation(points)
        self.assertEqual(observation.shape, (3, 256, 256))
        np.testing.assert_allclose(observation.sum(axis=0), 1.0)
        self.assertEqual(observation[1].sum(), 2)
        self.assertGreater(observation[0].sum(), 0)

    def test_ground_endpoint_policy_uses_height_without_labels(self) -> None:
        # Two returns share the forward bin but occupy different cells.  The
        # lower return defines the sensor-only support profile; the elevated
        # return is therefore an obstacle endpoint.
        points = np.array(
            [
                [3.0, 0.0, 0.0, 0.1],
                [3.0, 1.0, 0.0, 0.1],
                [6.0, 0.0, 0.0, 0.1],
                [6.0, 1.0, 2.0, 0.1],
            ],
            dtype=np.float32,
        )
        observation = lidar_observation(points, endpoint_policy="ground")
        ground_row = int(3.0 / 0.3)
        obstacle_row = int(6.0 / 0.3)
        center_col = 128
        side_col = int(round((1.0 - (-38.4)) / 0.3))
        self.assertEqual(observation[0, ground_row, center_col], 1.0)
        self.assertEqual(observation[1, obstacle_row, side_col], 1.0)
        np.testing.assert_allclose(observation.sum(axis=0), 1.0)

    def test_queries_use_validity_but_not_free_class(self) -> None:
        valid = np.zeros((256, 256), dtype=bool)
        valid[20, 128] = True
        valid[33, 128] = True
        starts, goals = deterministic_queries(valid, distances_cells=(13,), angles_deg=(0,), anchor_hint=(8, 128))
        np.testing.assert_array_equal(starts, [[20, 128]])
        np.testing.assert_array_equal(goals, [[33, 128]])

    def test_queries_can_select_an_observed_start_without_free_labels(self) -> None:
        valid = np.zeros((256, 256), dtype=bool)
        valid[96, 128] = True
        valid[100, 128] = True
        valid[113, 128] = True
        observed = np.zeros((256, 256), dtype=bool)
        observed[100, 128] = True
        starts, goals = deterministic_queries(
            valid,
            distances_cells=(13,),
            angles_deg=(0,),
            anchor_hint=(96, 128),
            start_mask=observed,
        )
        np.testing.assert_array_equal(starts, [[100, 128]])
        np.testing.assert_array_equal(goals, [[113, 128]])


if __name__ == "__main__":
    unittest.main()
