from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.train_flatlands_conpath import _exact_event_probabilities


class FlatLandsConPathHelperTest(unittest.TestCase):
    def test_exact_event_helper_respects_footprint_and_sample_frequency(self) -> None:
        safe = torch.zeros((1, 2, 7, 9), dtype=torch.float32)
        safe[:, :, 3, 2:7] = 1.0
        safe[:, 0, 2:5, 2:7] = 1.0
        starts = np.asarray([[[3, 3]]], dtype=np.int64)
        goals = np.asarray([[[3, 5]]], dtype=np.int64)
        probabilities = _exact_event_probabilities(safe, starts, goals, (0, 1, 2))
        np.testing.assert_allclose(probabilities, np.asarray([[[1.0, 0.5, 0.0]]]))


if __name__ == "__main__":
    unittest.main()
