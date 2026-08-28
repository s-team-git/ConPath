"""End-to-end PathRel core model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .reachability import multi_radius_reachability
from .stochastic_decoder import CorrelatedCategoricalDecoder
from .types import PathRelOutput


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class TinyBEVUNet(nn.Module):
    """Small dataset-independent encoder for rasterized BEV observations."""

    def __init__(self, input_channels: int, feature_channels: int = 32) -> None:
        super().__init__()
        self.enc0 = ConvNormAct(input_channels, feature_channels)
        self.enc1 = ConvNormAct(feature_channels, feature_channels * 2, stride=2)
        self.enc2 = ConvNormAct(feature_channels * 2, feature_channels * 4, stride=2)
        self.dec1 = ConvNormAct(feature_channels * 6, feature_channels * 2)
        self.dec0 = ConvNormAct(feature_channels * 3, feature_channels)

    def forward(self, observation_bev: Tensor) -> Tensor:
        if observation_bev.ndim != 4:
            raise ValueError("observation_bev must have shape [B, C, H, W]")
        level0 = self.enc0(observation_bev)
        level1 = self.enc1(level0)
        level2 = self.enc2(level1)

        up1 = F.interpolate(level2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(torch.cat((up1, level1), dim=1))
        up0 = F.interpolate(up1, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec0(torch.cat((up0, level0), dim=1))


class PathRelNet(nn.Module):
    """Correlated stochastic occupancy followed by derived path reliability."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        feature_channels: int = 32,
        num_classes: int = 2,
        latent_dim: int = 8,
        local_kernel_size: int = 5,
        traversable_index: int = 0,
    ) -> None:
        super().__init__()
        self.traversable_index = traversable_index
        self.encoder = TinyBEVUNet(input_channels, feature_channels)
        self.decoder = CorrelatedCategoricalDecoder(
            feature_channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            local_kernel_size=local_kernel_size,
        )

    def forward(
        self,
        observation_bev: Tensor,
        *,
        starts: Tensor | None = None,
        goals: Tensor | None = None,
        footprint_radii_cells: Sequence[int] | Tensor | None = None,
        num_samples: int = 8,
        concrete_backward_temperature: float = 0.7,
        reachability_backward_temperature: float = 0.1,
        hard_samples: bool = True,
        max_reachability_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PathRelOutput:
        if observation_bev.ndim != 4:
            raise ValueError("observation_bev must have shape [B,C,H,W]")
        features = self.encoder(observation_bev)
        known_classes = None
        if observation_bev.shape[1] >= 3:
            known_classes = torch.full(
                observation_bev.shape[:1] + observation_bev.shape[-2:],
                -1,
                dtype=torch.long,
                device=observation_bev.device,
            )
            known_classes = torch.where(observation_bev[:, 0] > 0.5, 0, known_classes)
            known_classes = torch.where(observation_bev[:, 1] > 0.5, 1, known_classes)
        return self.forward_features(
            features,
            starts=starts,
            goals=goals,
            footprint_radii_cells=footprint_radii_cells,
            num_samples=num_samples,
            concrete_backward_temperature=concrete_backward_temperature,
            reachability_backward_temperature=reachability_backward_temperature,
            hard_samples=hard_samples,
            max_reachability_steps=max_reachability_steps,
            generator=generator,
            known_classes=known_classes,
        )

    def forward_features(
        self,
        features: Tensor,
        *,
        starts: Tensor | None = None,
        goals: Tensor | None = None,
        footprint_radii_cells: Sequence[int] | Tensor | None = None,
        num_samples: int = 8,
        concrete_backward_temperature: float = 0.7,
        reachability_backward_temperature: float = 0.1,
        hard_samples: bool = True,
        max_reachability_steps: int | None = None,
        generator: torch.Generator | None = None,
        known_classes: Tensor | None = None,
    ) -> PathRelOutput:
        """Run the novel head on externally produced BEV features.

        Public-dataset adapters can use an official BEVFusion/OccFormer/Co-Occ encoder and call
        this method, keeping the stochastic posterior and reliability code unchanged.
        """

        posterior = self.decoder(
            features,
            num_samples=num_samples,
            concrete_backward_temperature=concrete_backward_temperature,
            hard=hard_samples,
            known_classes=known_classes,
            generator=generator,
        )

        query_values = (starts, goals, footprint_radii_cells)
        if all(value is None for value in query_values):
            return PathRelOutput(posterior=posterior)
        if any(value is None for value in query_values):
            raise ValueError(
                "starts, goals, and footprint_radii_cells must be supplied together"
            )
        if not hard_samples:
            raise ValueError(
                "query reachability requires straight-through hard map samples; "
                "soft maps produce a max-min surrogate, not an event probability"
            )

        reachability, sample_events = multi_radius_reachability(
            posterior.safe_samples(self.traversable_index),
            starts,
            goals,
            footprint_radii_cells,
            max_steps=max_reachability_steps,
            backward_temperature=reachability_backward_temperature,
        )
        return PathRelOutput(
            posterior=posterior,
            reachability=reachability,
            sample_reachability=sample_events,
        )
