"""Losses for stochastic maps and task-level connectivity calibration."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F


def map_cross_entropy(
    mean_logits: Tensor,
    target_classes: Tensor,
    *,
    class_weights: Tensor | None = None,
    ignore_index: int = -1,
) -> Tensor:
    return F.cross_entropy(
        mean_logits,
        target_classes.long(),
        weight=class_weights,
        ignore_index=ignore_index,
    )


def posterior_marginal_nll(
    sample_logits: Tensor,
    target_classes: Tensor,
    *,
    categorical_noise_scale: float = 1.0,
    ignore_index: int = -1,
    epsilon: float = 1e-6,
) -> Tensor:
    """Per-cell log score under the Monte-Carlo posterior marginal.

    Averaging conditional softmax probabilities is essential: softmax of the mean logits is not
    the marginal of a logistic-normal posterior. A scaled Gumbel categorical draw has conditional
    probabilities ``softmax(logits / scale)``, so the scale must match the sampler. Ignored cells
    represent unavailable truth, not a latent UNKNOWN terrain class.
    """

    if sample_logits.ndim != 5:
        raise ValueError("sample_logits must have shape [B,K,C,H,W]")
    if categorical_noise_scale <= 0:
        raise ValueError("categorical_noise_scale must be positive for log-score evaluation")
    conditional_class_probs = torch.softmax(
        sample_logits / float(categorical_noise_scale), dim=2
    )
    return posterior_marginal_nll_from_probs(
        conditional_class_probs,
        target_classes,
        ignore_index=ignore_index,
        epsilon=epsilon,
    )


def posterior_marginal_nll_from_probs(
    conditional_class_probs: Tensor,
    target_classes: Tensor,
    *,
    ignore_index: int = -1,
    epsilon: float = 1e-6,
) -> Tensor:
    """Per-cell log score from conditional class probabilities ``[B,K,C,H,W]``."""

    if conditional_class_probs.ndim != 5:
        raise ValueError("conditional_class_probs must have shape [B,K,C,H,W]")
    target = target_classes.long()
    expected_shape = (
        conditional_class_probs.shape[0],
        *conditional_class_probs.shape[-2:],
    )
    if target.shape != expected_shape:
        raise ValueError(f"target_classes must have shape {expected_shape}")

    marginal = conditional_class_probs.mean(dim=1)
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    selected = marginal.gather(1, safe_target[:, None]).squeeze(1)
    if not torch.any(valid):
        return selected.sum() * 0.0
    return -torch.log(selected.clamp_min(epsilon))[valid].mean()


def reachability_brier(
    reachability: Tensor, targets: Tensor, *, weights: Tensor | None = None
) -> Tensor:
    if reachability.shape != targets.shape:
        raise ValueError("reachability and targets must have the same shape")
    error = (reachability - targets.to(reachability.dtype)).square()
    if weights is None:
        return error.mean()
    if weights.shape != error.shape:
        raise ValueError("weights must have the same shape as reachability")
    weights = weights.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1e-12)


def reachability_brier_u_statistic(
    sample_events: Tensor,
    targets: Tensor,
    *,
    weights: Tensor | None = None,
) -> Tensor:
    """Unbiased finite-ensemble estimator of the Brier score.

    Squaring a finite Monte-Carlo mean adds a variance term that can encourage sample collapse.
    This U-statistic estimates ``p^2 - 2py + y^2`` using distinct sample pairs.
    It may be slightly negative for an individual minibatch although its expectation is the
    non-negative population Brier score.
    """

    if sample_events.ndim < 2:
        raise ValueError("sample_events must contain a sample dimension at index 1")
    samples = sample_events.shape[1]
    if samples < 2:
        raise ValueError("at least two samples are required")
    expected_shape = (sample_events.shape[0], *sample_events.shape[2:])
    if targets.shape != expected_shape:
        raise ValueError(
            f"targets must have shape {expected_shape}, got {tuple(targets.shape)}"
        )

    event_sum = sample_events.sum(dim=1)
    pair_product = (event_sum.square() - sample_events.square().sum(dim=1))
    pair_product = pair_product / float(samples * (samples - 1))
    mean_event = event_sum / float(samples)
    target = targets.to(sample_events.dtype)
    score = pair_product - 2.0 * target * mean_event + target.square()
    if weights is None:
        return score.mean()
    if weights.shape != score.shape:
        raise ValueError("weights must have the same shape as targets")
    weights = weights.to(score.dtype)
    return (score * weights).sum() / weights.sum().clamp_min(1e-12)


def spatial_variogram_score(
    safe_samples: Tensor,
    target_safe: Tensor,
    *,
    offsets: Iterable[tuple[int, int]] = ((0, 1), (1, 0), (0, 4), (4, 0)),
    power: float = 1.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Sampled spatial variogram score over local and longer-range cell pairs."""

    if safe_samples.ndim != 4:
        raise ValueError("safe_samples must have shape [B, K, H, W]")
    if target_safe.shape != (safe_samples.shape[0], *safe_samples.shape[-2:]):
        raise ValueError("target_safe must have shape [B, H, W]")
    if valid_mask is None:
        valid_mask = torch.ones_like(target_safe, dtype=torch.bool)
    elif valid_mask.shape != target_safe.shape:
        raise ValueError("valid_mask must have shape [B,H,W]")
    else:
        valid_mask = valid_mask.to(torch.bool)

    height, width = safe_samples.shape[-2:]
    terms: list[Tensor] = []
    for row_offset, col_offset in offsets:
        row_offset, col_offset = int(row_offset), int(col_offset)
        if row_offset < 0 or col_offset < 0:
            raise ValueError("variogram offsets must be non-negative")
        if row_offset >= height or col_offset >= width:
            continue
        row_a = slice(0, height - row_offset or None)
        row_b = slice(row_offset, height)
        col_a = slice(0, width - col_offset or None)
        col_b = slice(col_offset, width)

        forecast_difference = (
            safe_samples[..., row_a, col_a] - safe_samples[..., row_b, col_b]
        ).abs().pow(power).mean(dim=1)
        target_difference = (
            target_safe[..., row_a, col_a] - target_safe[..., row_b, col_b]
        ).abs().pow(power)
        pair_valid = valid_mask[..., row_a, col_a] & valid_mask[..., row_b, col_b]
        if torch.any(pair_valid):
            squared_error = (forecast_difference - target_difference).square()
            terms.append(squared_error[pair_valid].mean())

    if not terms:
        raise ValueError("no variogram offset fits inside the map")
    return torch.stack(terms).mean()
