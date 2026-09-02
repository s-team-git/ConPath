from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np

from pathrel.flatlands_baselines import (
    DirectQueryBaseline,
    MarginalCompletionBaseline,
    S4CInspiredCoordinateBaseline,
    fit_scene_weighted_radius_prior,
    radius_prior_prediction_rows,
)
from pathrel.flatlands_data import load_bounded_query_manifest
from pathrel.flatlands_query import load_provenance_manifest
from pathrel.flatlands_sampling import (
    CompletionEventTask,
    completion_event_probabilities,
)
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


class FlatLandsS4CInspiredCoordinateBaselineTest(unittest.TestCase):
    def test_shapes_radius_conditioning_and_backward(self) -> None:
        torch.manual_seed(17)
        model = S4CInspiredCoordinateBaseline(feature_channels=8)
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
        model = S4CInspiredCoordinateBaseline(feature_channels=8)
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


class FlatLandsMarginalCompletionBaselineTest(unittest.TestCase):
    def test_observed_evidence_is_clamped_and_hidden_logits_backpropagate(self) -> None:
        torch.manual_seed(11)
        model = MarginalCompletionBaseline(feature_channels=8)
        observation = torch.zeros(2, 3, 20, 20)
        observation[:, 0, 4:7, 4:7] = 1.0
        observation[:, 1, 10:12, 10:12] = 1.0
        observation[:, 2, 7:10, 7:10] = 1.0
        logits = model(observation)
        probability = model.free_probability(observation)

        self.assertEqual(logits.shape, (2, 20, 20))
        self.assertTrue(torch.all(probability[:, 4:7, 4:7] == 1.0))
        self.assertTrue(torch.all(probability[:, 10:12, 10:12] == 0.0))
        self.assertTrue(torch.all(probability[:, 0:2, 0:2] == 0.0))
        self.assertTrue(
            torch.all((probability[:, 7:10, 7:10] > 0.0) & (probability[:, 7:10, 7:10] < 1.0))
        )
        logits[:, 7:10, 7:10].square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_independent_event_sampling_is_reproducible_and_exact_for_binary_probability(self) -> None:
        free = np.ones((16, 16), dtype=bool)
        free[:, 8] = False
        free[8, 8] = True
        task = CompletionEventTask(
            global_id="fixture",
            free_probability=free.astype(np.float64),
            observed_free=np.zeros_like(free),
            unknown=np.ones_like(free),
            starts=np.asarray([[8, 3]]),
            goals=np.asarray([[8, 12]]),
            candidate_indices=np.asarray([7]),
            radii_cells=(0, 1, 2),
            posterior_samples=4,
            seed=31,
        )
        first = completion_event_probabilities(task)
        second = completion_event_probabilities(task)
        self.assertTrue(np.array_equal(first.deterministic, second.deterministic))
        self.assertTrue(np.array_equal(first.independent, second.independent))
        self.assertTrue(np.array_equal(first.deterministic, first.independent))
        self.assertEqual(first.deterministic.tolist(), [[1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
