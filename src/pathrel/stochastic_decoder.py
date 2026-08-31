"""Correlated stochastic categorical occupancy decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import OccupancyPosterior


class CorrelatedCategoricalDecoder(nn.Module):
    """Decode BEV features into coherent categorical map samples.

    Spatially global dependence is represented by low-rank factor maps. Local dependence is
    represented by smoothed Gaussian noise with a bounded learned scale. A straight-through
    Concrete sample gives an exactly one-hot forward map and relaxed backward gradients.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        num_classes: int = 2,
        latent_dim: int = 8,
        local_kernel_size: int = 5,
        min_local_scale: float = 1e-3,
        max_local_scale: float = 2.0,
        max_factor_magnitude: float = 2.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if local_kernel_size < 1 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd integer")

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.local_kernel_size = local_kernel_size
        self.min_local_scale = min_local_scale
        self.max_local_scale = max_local_scale
        self.max_factor_magnitude = max_factor_magnitude

        coordinates = torch.arange(local_kernel_size, dtype=torch.float32)
        coordinates = coordinates - (local_kernel_size - 1) / 2.0
        standard_deviation = max(float(local_kernel_size) / 3.0, 0.5)
        gaussian_1d = torch.exp(-coordinates.square() / (2.0 * standard_deviation**2))
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        self.register_buffer("local_kernel", gaussian_2d[None, None])

        self.mean_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.factor_head = nn.Conv2d(
            in_channels, num_classes * latent_dim, kernel_size=1
        )
        self.scale_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Start near a deterministic model but keep enough local noise for non-zero stochastic
        # gradients during the first optimization step.
        nn.init.zeros_(self.factor_head.weight)
        nn.init.zeros_(self.factor_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, -2.0)

    @staticmethod
    def _straight_through_categorical(relaxed: Tensor, hard: bool) -> Tensor:
        if not hard:
            return relaxed
        indices = relaxed.argmax(dim=2, keepdim=True)
        one_hot = torch.zeros_like(relaxed).scatter_(2, indices, 1.0)
        return one_hot + relaxed - relaxed.detach()

    def forward(
        self,
        features: Tensor,
        *,
        num_samples: int,
        concrete_backward_temperature: float = 0.7,
        categorical_noise_scale: float = 1.0,
        hard: bool = True,
        known_classes: Tensor | None = None,
        disable_global_factors: bool = False,
        generator: torch.Generator | None = None,
    ) -> OccupancyPosterior:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,C,H,W], got {tuple(features.shape)}")
        if num_samples < 2:
            raise ValueError("num_samples must be at least two for joint scoring")
        if concrete_backward_temperature <= 0:
            raise ValueError("concrete_backward_temperature must be positive")
        if categorical_noise_scale < 0:
            raise ValueError("categorical_noise_scale must be non-negative")

        batch, _, height, width = features.shape
        mean_logits = self.mean_head(features)
        raw_factors = self.factor_head(features).view(
            batch, self.num_classes, self.latent_dim, height, width
        )
        factor_maps = self.max_factor_magnitude * torch.tanh(raw_factors)

        local_scale = self.min_local_scale + F.softplus(self.scale_head(features))
        local_scale = local_scale.clamp(max=self.max_local_scale)

        normal_shape = (batch, num_samples, self.latent_dim)
        latent = torch.randn(
            normal_shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        )
        if disable_global_factors:
            global_delta = torch.zeros(
                (batch, num_samples, self.num_classes, height, width),
                dtype=features.dtype,
                device=features.device,
            )
        else:
            global_delta = torch.einsum("bcdhw,bkd->bkchw", factor_maps, latent)
            global_delta = global_delta / math.sqrt(float(self.latent_dim))

        local_noise = torch.randn(
            (batch * num_samples * self.num_classes, 1, height, width),
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        )
        padding = self.local_kernel_size // 2
        local_noise = F.conv2d(local_noise, self.local_kernel, padding=padding)
        # Normalize by the exact sum of squared in-bounds weights. This keeps unit marginal
        # variance without inventing extra uncertainty at the map boundary.
        normalization = F.conv2d(
            torch.ones_like(local_noise), self.local_kernel.square(), padding=padding
        ).sqrt().clamp_min(1e-6)
        local_noise = local_noise / normalization
        local_noise = local_noise.view(
            batch, num_samples, self.num_classes, height, width
        )
        local_delta = local_noise * local_scale[:, None]

        sample_logits = mean_logits[:, None] + global_delta + local_delta
        if known_classes is not None:
            if known_classes.shape != (batch, height, width):
                raise ValueError("known_classes must have shape [B,H,W]")
            known_classes = known_classes.to(device=features.device, dtype=torch.long)
            known_mask = (known_classes >= 0) & (known_classes < self.num_classes)
            class_indices = torch.arange(
                self.num_classes, device=features.device, dtype=torch.long
            )[None, None, :, None, None]
            observed_logits = torch.where(
                class_indices == known_classes[:, None, None],
                torch.full_like(sample_logits, 10.0),
                torch.full_like(sample_logits, -10.0),
            )
            sample_logits = torch.where(
                known_mask[:, None, None], observed_logits, sample_logits
            )
        uniform = torch.rand(
            sample_logits.shape,
            dtype=sample_logits.dtype,
            device=sample_logits.device,
            generator=generator,
        ).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        if categorical_noise_scale > 0:
            # If G_i are i.i.d. standard Gumbels, argmax(logit_i + scale * G_i)
            # is categorical with probabilities softmax(logits / scale).  Reporting
            # softmax(logits) for a non-unit scale silently miscalibrates the hard-map
            # posterior, which is the distribution consumed by reachability.
            conditional_class_probs = torch.softmax(
                sample_logits / float(categorical_noise_scale), dim=2
            )
        else:
            conditional_class_probs = F.one_hot(
                sample_logits.argmax(dim=2), num_classes=self.num_classes
            ).movedim(-1, 2).to(sample_logits.dtype)
        relaxed = torch.softmax(
            (sample_logits + float(categorical_noise_scale) * gumbel)
            / concrete_backward_temperature,
            dim=2,
        )
        sample_probs = self._straight_through_categorical(relaxed, hard=hard)

        return OccupancyPosterior(
            mean_logits=mean_logits,
            factor_maps=factor_maps,
            local_scale=local_scale,
            sample_logits=sample_logits,
            conditional_class_probs=conditional_class_probs,
            relaxed_probs=relaxed,
            sample_probs=sample_probs,
        )
