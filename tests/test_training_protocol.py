from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_p0 import make_split, stack
from scripts.train_p0_neural import repeated_observation_groups


class P0TrainingProtocolTest(unittest.TestCase):
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
