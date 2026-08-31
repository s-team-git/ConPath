"""Fixed FlatLands baseline components shared by training and prediction scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .flatlands_data import ReplayQuery
from .flatlands_query import ManifestObservation
from .model import TinyBEVUNet


@dataclass(frozen=True)
class RadiusPrior:
    probabilities: Mapping[int, float]
    fitted_scene_count: int
    fitted_event_count: int


class DirectQueryBaseline(nn.Module):
    """Capacity-fixed direct event predictor for the FlatLands baseline contract."""

    def __init__(self, *, input_channels: int = 3, feature_channels: int = 16) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.encoder = TinyBEVUNet(
            input_channels,
            self.feature_channels,
            use_coordinate_channels=True,
            use_global_context=True,
        )
        geometry_channels = 10
        hidden_channels = 64
        self.event_head = nn.Sequential(
            nn.Linear(3 * self.feature_channels + geometry_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def _gather(features: Tensor, points: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        if points.ndim != 3 or points.shape[0] != batch or points.shape[-1] != 2:
            raise ValueError("points must have shape [B,Q,2]")
        rows = points[..., 0].clamp(0, height - 1)
        cols = points[..., 1].clamp(0, width - 1)
        indices = (rows * width + cols).to(dtype=torch.long)
        flattened = features.reshape(batch, channels, height * width)
        return torch.gather(
            flattened,
            2,
            indices[:, None].expand(batch, channels, indices.shape[1]),
        ).transpose(1, 2)

    def forward(
        self,
        observation: Tensor,
        starts: Tensor,
        goals: Tensor,
        distances_m: Tensor,
        angles_deg: Tensor,
        radii_cells: Tensor,
    ) -> Tensor:
        """Return event logits with shape ``[B,Q,R]``."""

        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation must have shape [B,3,H,W]")
        if starts.shape != goals.shape or starts.shape[-1] != 2:
            raise ValueError("starts/goals must share shape [B,Q,2]")
        if distances_m.shape != starts.shape[:2] or angles_deg.shape != starts.shape[:2]:
            raise ValueError("distance/angle tensors must have shape [B,Q]")
        if radii_cells.ndim != 1 or radii_cells.numel() == 0:
            raise ValueError("radii_cells must be a non-empty vector")
        features = self.encoder(observation)
        batch, _, height, width = features.shape
        query_count = starts.shape[1]
        radius_count = radii_cells.numel()
        if query_count == 0:
            return features.new_empty((batch, 0, radius_count))
        start_features = self._gather(features, starts)
        goal_features = self._gather(features, goals)
        valid = observation.sum(dim=1, keepdim=True) > 0.5
        valid_mass = valid.sum(dim=(-2, -1)).clamp_min(1)
        global_features = (features * valid).sum(dim=(-2, -1)) / valid_mass
        global_features = global_features[:, None, :].expand(batch, query_count, -1)

        starts_float = starts.to(dtype=features.dtype)
        goals_float = goals.to(dtype=features.dtype)
        normalizer = features.new_tensor(
            [max(1, height - 1), max(1, width - 1)]
        )
        starts_normalized = 2.0 * starts_float / normalizer - 1.0
        goals_normalized = 2.0 * goals_float / normalizer - 1.0
        delta = goals_normalized - starts_normalized
        angle_radians = torch.deg2rad(angles_deg.to(dtype=features.dtype))
        base_geometry = torch.cat(
            (
                starts_normalized,
                goals_normalized,
                delta,
                (distances_m.to(dtype=features.dtype) / 1.2)[..., None],
                torch.sin(angle_radians)[..., None],
                torch.cos(angle_radians)[..., None],
            ),
            dim=-1,
        )
        shared = torch.cat(
            (start_features, goal_features, global_features, base_geometry), dim=-1
        )
        shared = shared[:, :, None, :].expand(batch, query_count, radius_count, -1)
        radius_feature = (
            radii_cells.to(device=features.device, dtype=features.dtype)
            / radii_cells.max().clamp_min(1).to(dtype=features.dtype)
        )
        radius_feature = radius_feature[None, None, :, None].expand(
            batch, query_count, radius_count, 1
        )
        return self.event_head(torch.cat((shared, radius_feature), dim=-1)).squeeze(-1)


def fit_scene_weighted_radius_prior(
    observations: Sequence[ManifestObservation],
    queries: Mapping[str, Sequence[ReplayQuery]],
    radii_cells: Sequence[int],
    *,
    fit_split: str = "train",
) -> RadiusPrior:
    """Fit a per-radius control prior with equal mass for every contributing scene."""

    radii = tuple(int(radius) for radius in radii_cells)
    if not radii:
        raise ValueError("radii_cells cannot be empty")
    scene_rates: dict[tuple[str, str], list[list[bool]]] = {}
    event_count = 0
    for observation in observations:
        if observation.provenance_split != fit_split:
            continue
        scene_key = observation.source_dataset, observation.scene_id
        by_radius = scene_rates.setdefault(scene_key, [[] for _ in radii])
        for query in queries[observation.global_id]:
            if not query.retained:
                continue
            for radius_index, label in enumerate(query.reachable):
                if label is None:
                    raise ValueError(
                        f"missing retained label for {observation.global_id}"
                    )
                by_radius[radius_index].append(bool(label))
                event_count += 1
    contributing = {
        scene_key: values
        for scene_key, values in scene_rates.items()
        if all(radius_values for radius_values in values)
    }
    if not contributing:
        raise ValueError(f"no contributing scenes in fit split {fit_split!r}")
    probabilities = {}
    for radius_index, radius in enumerate(radii):
        per_scene = [
            float(np.mean(values[radius_index])) for values in contributing.values()
        ]
        probabilities[radius] = float(np.mean(per_scene))
    return RadiusPrior(
        probabilities=probabilities,
        fitted_scene_count=len(contributing),
        fitted_event_count=event_count,
    )


def radius_prior_prediction_rows(
    observations: Sequence[ManifestObservation],
    queries: Mapping[str, Sequence[ReplayQuery]],
    radii_cells: Sequence[int],
    prior: RadiusPrior,
    *,
    prediction_split: str,
) -> list[dict[str, object]]:
    """Emit label-free rows for the exact retained events of one provenance split."""

    radii = tuple(int(radius) for radius in radii_cells)
    if set(radii) != set(prior.probabilities):
        raise ValueError("prior radii do not match the query manifest")
    rows: list[dict[str, object]] = []
    for observation in observations:
        if observation.provenance_split != prediction_split:
            continue
        for query in queries[observation.global_id]:
            if not query.retained:
                continue
            for radius in radii:
                rows.append(
                    {
                        "global_id": observation.global_id,
                        "candidate_index": query.candidate_index,
                        "radius_cells": radius,
                        "probability": prior.probabilities[radius],
                    }
                )
    if not rows:
        raise ValueError(
            f"no retained prediction events in split {prediction_split!r}"
        )
    return rows
