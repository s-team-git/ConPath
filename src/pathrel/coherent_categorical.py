"""Untrained candidate: spatially correlated categorical noise with fixed marginals.

This module is deliberately separate from the frozen decoder and training recipe. The
candidate uses a fixed 9 by 9 Gaussian kernel with sigma=3; kernel size 1 is available
only as a synthetic independent-noise control. It introduces no learned parameters.

Independent Gaussian fields for every batch item, world, and class are transformed
through the normal CDF and Gumbel inverse CDF. Before numerical clipping, each cell
has a standard Gumbel marginal, independent across classes and independent of the
parent decoder's latent/local noise. Spatial dependence changes the joint map law.
The inherited conditional class probabilities therefore remain valid marginal
probabilities, up to floating-point arithmetic and the inherited 1e-6 tail clipping.

Calling the parent first intentionally retains its original RNG consumption and
logit computation, including an unused independent categorical draw. The candidate
then consumes fresh Gaussian randomness and performs additional convolution/CDF
work. Equal parameter counts do not imply equal computation or equal RNG states.
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .stochastic_decoder import CorrelatedCategoricalDecoder
from .types import OccupancyPosterior


class CoherentCategoricalDecoder(CorrelatedCategoricalDecoder):
    """Replace only the final categorical draw; the candidate scale is fixed at 1.

    The existing learned global factors and local Gaussian field are unchanged.
    A size-1 Gumbel kernel is for CPU synthetic checks, not a second training choice.
    Low-precision inputs use float32 for the new noise transform and relaxation;
    float64 inputs retain float64. No spatial empirical standardization is used.
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
        gumbel_kernel_size: int = 9,
    ) -> None:
        if gumbel_kernel_size not in (1, 9):
            raise ValueError("gumbel_kernel_size must be 9 (candidate) or 1 (synthetic control)")
        super().__init__(
            in_channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            local_kernel_size=local_kernel_size,
            min_local_scale=min_local_scale,
            max_local_scale=max_local_scale,
            max_factor_magnitude=max_factor_magnitude,
        )
        self.gumbel_kernel_size = gumbel_kernel_size
        coordinates = torch.arange(gumbel_kernel_size, dtype=torch.float32)
        coordinates = coordinates - (gumbel_kernel_size - 1) / 2.0
        gaussian_1d = torch.exp(-coordinates.square() / (2.0 * 3.0**2))
        kernel = gaussian_1d[:, None] * gaussian_1d[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("gumbel_kernel", kernel[None, None])

    def _sample_standard_normal(
        self, reference: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw unit-marginal-variance fields shaped like [B,K,C,H,W] reference.

        Zero padding introduces no additional random variables. The denominator is
        the exact variance from in-bounds weights, computed with the same padding.
        A singleton denominator broadcasts over independent B/K/C fields. In
        particular, smoothing never runs across those axes or reuses parent noise.
        """
        if reference.ndim != 5:
            raise ValueError("reference must have shape [B,K,C,H,W]")
        batch, samples, classes, height, width = reference.shape
        work_dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
        white = torch.randn(
            (batch * samples * classes, 1, height, width),
            dtype=work_dtype,
            device=reference.device,
            generator=generator,
        )
        kernel = self.gumbel_kernel.to(device=reference.device, dtype=work_dtype)
        padding = self.gumbel_kernel_size // 2
        # Disable outer autocast for the convolution, CDF input, and normalization.
        with torch.autocast(device_type=reference.device.type, enabled=False):
            smoothed = F.conv2d(white, kernel, padding=padding)
            variance = F.conv2d(
                torch.ones((1, 1, height, width), dtype=work_dtype, device=reference.device),
                kernel.square(),
                padding=padding,
            )
            field = smoothed / variance.sqrt()
        return field.reshape(batch, samples, classes, height, width)

    def _sample_gumbel(
        self, reference: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        normal = self._sample_standard_normal(reference, generator=generator)
        with torch.autocast(device_type=reference.device.type, enabled=False):
            uniform = torch.special.ndtr(normal).clamp(1e-6, 1.0 - 1e-6)
            return -torch.log(-torch.log(uniform))

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
        if categorical_noise_scale != 1.0:
            raise ValueError("this candidate requires categorical_noise_scale=1.0")
        if not math.isfinite(concrete_backward_temperature) or concrete_backward_temperature <= 0:
            raise ValueError("concrete_backward_temperature must be finite and positive")
        posterior = super().forward(
            features,
            num_samples=num_samples,
            concrete_backward_temperature=concrete_backward_temperature,
            categorical_noise_scale=1.0,
            hard=hard,
            known_classes=known_classes,
            disable_global_factors=disable_global_factors,
            generator=generator,
        )
        gumbel = self._sample_gumbel(posterior.sample_logits, generator=generator)
        with torch.autocast(device_type=features.device.type, enabled=False):
            perturbed_logits = posterior.sample_logits.to(gumbel.dtype) + gumbel
            relaxed = torch.softmax(
                perturbed_logits / concrete_backward_temperature,
                dim=2,
            )
        if hard:
            # Choose before temperature scaling: a very large backward temperature
            # can round unequal relaxed probabilities into ties in finite precision.
            indices = perturbed_logits.argmax(dim=2, keepdim=True)
            one_hot = torch.zeros_like(relaxed).scatter_(2, indices, 1.0)
            sample_probs = one_hot + (relaxed - relaxed.detach())
        else:
            sample_probs = relaxed
        # At unit noise scale, the clipped Gumbel range is less than the inherited
        # known-logit gap of 20, so observed/support-blocked classes cannot flip.
        # The parent conditional probabilities and all pre-categorical tensors are
        # retained; these are cell marginals, not a factorized joint map density.
        return replace(
            posterior,
            relaxed_probs=relaxed,
            sample_probs=sample_probs,
        )
