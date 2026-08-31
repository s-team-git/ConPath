"""Typed containers shared by the PathRel model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class OccupancyPosterior:
    """Outputs of the correlated categorical occupancy decoder.

    Shapes:
        mean_logits: ``[B, C, H, W]`` logistic-normal location logits.
        factor_maps: ``[B, C, D, H, W]`` low-rank spatial factors.
        local_scale: ``[B, C, H, W]`` bounded local stochastic scale.
        sample_logits: ``[B, K, C, H, W]`` correlated logits before Concrete noise.
        conditional_class_probs: ``[B, K, C, H, W]`` exact categorical probabilities
            conditional on each stochastic logit sample.
        relaxed_probs: ``[B, K, C, H, W]`` Concrete samples used only as a backward surrogate.
        sample_probs: ``[B, K, C, H, W]`` straight-through categorical samples.
    """

    mean_logits: Tensor
    factor_maps: Tensor
    local_scale: Tensor
    sample_logits: Tensor
    conditional_class_probs: Tensor
    relaxed_probs: Tensor
    sample_probs: Tensor

    @property
    def posterior_marginal_probs(self) -> Tensor:
        """Monte-Carlo posterior class marginals ``[B,C,H,W]``.

        ``softmax(E[logits])`` is not the posterior marginal. The correct Rao-Blackwellized
        estimate averages the conditional categorical probabilities over stochastic logit samples.
        The decoder accounts for its configured categorical-noise scale when constructing those
        probabilities.
        """

        return self.conditional_class_probs.mean(dim=1)

    @property
    def empirical_class_frequencies(self) -> Tensor:
        return self.sample_probs.mean(dim=1)

    def safe_samples(self, traversable_index: int = 0) -> Tensor:
        """Return support-valid samples as ``[B, K, H, W]``.

        The latent world state is binary traversable/blocked. Observation unknown is an input
        channel and label-validity condition, not a third physical world state.
        """

        return self.sample_probs[:, :, traversable_index]


@dataclass
class PathRelOutput:
    """End-to-end model output."""

    posterior: OccupancyPosterior
    reachability: Optional[Tensor] = None
    sample_reachability: Optional[Tensor] = None
