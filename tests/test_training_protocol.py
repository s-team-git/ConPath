from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_p0 import make_split, stack
from scripts.train_p0_neural import add_visible_context_plane, repeated_observation_groups


class P0TrainingProtocolTest(unittest.TestCase):
    def test_visible_context_plane_contains_only_family_bit(self) -> None:
        observation = np.zeros((2, 3, 5, 6), dtype=np.float32)
        context = np.asarray([0, 1], dtype=np.int64)
        augmented = add_visible_context_plane(observation, context, mode="plane")
        self.assertEqual(tuple(augmented.shape), (2, 4, 5, 6))
        np.testing.assert_array_equal(augmented[:, :3], observation)
        np.testing.assert_array_equal(augmented[0, 3], np.zeros((5, 6)))
        np.testing.assert_array_equal(augmented[1, 3], np.ones((5, 6)))

    def test_repeated_worlds_form_identical_observation_groups(self) -> None:
        arrays = stack(
            make_split(
                range(3),
                worlds_per_template=8,
                height=12,
                width=12,
                radii=(0, 1, 2),
            )
        )
        groups = repeated_observation_groups(arrays)
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(len(groups[1]), 3)
        for family in (0, 1):
            for indices in groups[family]:
                self.assertEqual(len(indices), 8)
                self.assertTrue(np.all(arrays["context"][indices] == family))
                expected = np.broadcast_to(
                    arrays["observation"][indices[0]], arrays["observation"][indices].shape
                )
                np.testing.assert_array_equal(arrays["observation"][indices], expected)
                empirical_events = arrays["events"][indices].mean(axis=0)
                self.assertTrue(np.all((empirical_events >= 0.0) & (empirical_events <= 1.0)))


if __name__ == "__main__":
    unittest.main()
