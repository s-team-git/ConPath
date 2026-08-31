from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from pathrel.flatlands_baselines import (
    DirectQueryBaseline,
    fit_scene_weighted_radius_prior,
    radius_prior_prediction_rows,
)
from pathrel.flatlands_data import load_bounded_query_manifest
from pathrel.flatlands_query import load_provenance_manifest
from tests.test_flatlands_data import _write_fixture


class FlatLandsRadiusPriorTest(unittest.TestCase):
    def test_prior_is_fit_only_from_requested_split_and_predictions_are_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, selection_path, query_path = _write_fixture(Path(directory))
            observations = load_provenance_manifest(selection_path)
            radii, queries, _ = load_bounded_query_manifest(query_path)
            prior = fit_scene_weighted_radius_prior(
                observations,
                queries,
                radii,
                fit_split="test",
            )
            rows = radius_prior_prediction_rows(
                observations,
                queries,
                radii,
                prior,
                prediction_split="test",
            )

        self.assertEqual(prior.fitted_scene_count, 1)
        self.assertGreater(prior.fitted_event_count, 0)
        self.assertEqual(set(prior.probabilities), set(radii))
        self.assertTrue(all(set(row) == {"global_id", "candidate_index", "radius_cells", "probability"} for row in rows))
        self.assertTrue(all(0.0 <= float(row["probability"]) <= 1.0 for row in rows))

    def test_empty_fit_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, selection_path, query_path = _write_fixture(Path(directory))
            observations = load_provenance_manifest(selection_path)
            radii, queries, _ = load_bounded_query_manifest(query_path)
            with self.assertRaisesRegex(ValueError, "no contributing scenes"):
                fit_scene_weighted_radius_prior(
                    observations,
                    queries,
                    radii,
                    fit_split="train",
                )


class FlatLandsDirectQueryBaselineTest(unittest.TestCase):
    def test_shapes_radius_conditioning_and_backward(self) -> None:
        torch.manual_seed(7)
        model = DirectQueryBaseline(feature_channels=8)
        observation = torch.zeros(2, 3, 24, 24)
        observation[:, 0, 10:13, 4:8] = 1.0
        observation[:, 1, :, 12] = 1.0
        observation[:, 2, 8:16, 8:16] = 1.0
        starts = torch.tensor([[[11, 6], [10, 6]], [[11, 6], [12, 6]]])
        goals = torch.tensor([[[11, 18], [10, 18]], [[11, 18], [12, 18]]])
        distances = torch.tensor([[1.2, 1.2], [1.2, 1.2]])
        angles = torch.tensor([[0, 30], [0, 330]])
        radii = torch.tensor([0, 10, 20])

        logits = model(observation, starts, goals, distances, angles, radii)
        self.assertEqual(logits.shape, (2, 2, 3))
        logits.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_zero_query_batch_is_supported(self) -> None:
        model = DirectQueryBaseline(feature_channels=8)
        observation = torch.zeros(1, 3, 16, 16)
        starts = torch.empty(1, 0, 2, dtype=torch.long)
        output = model(
            observation,
            starts,
            starts,
            torch.empty(1, 0),
            torch.empty(1, 0, dtype=torch.long),
            torch.tensor([0, 10, 20]),
        )
        self.assertEqual(output.shape, (1, 0, 3))


if __name__ == "__main__":
    unittest.main()
