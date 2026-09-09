"""Shared, lossless observation encoding for external BEV completion methods."""
from __future__ import annotations

import torch


def validate_condition(condition: torch.Tensor):
    """B x 3 x H x W = observed free, unknown, valid support (never ground truth)."""
    if condition.ndim != 4 or condition.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW [observed_free, unknown, support]")
    if not torch.isfinite(condition).all() or not ((condition == 0) | (condition == 1)).all():
        raise ValueError("Condition must contain finite binary masks")
    free, unknown, support = condition.split(1, dim=1)
    if (free * unknown).any() or (free > support).any() or (unknown > support).any():
        raise ValueError("Overlapping states or observations outside support")


def clamp_completion(value: torch.Tensor, condition: torch.Tensor):
    """Preserve observed free/blocked evidence, close invalid support last."""
    free, unknown, support = condition.split(1, dim=1)
    return (value * unknown + free) * support


def masked_mean(value: torch.Tensor, mask: torch.Tensor):
    """Per-observation masked mean, then equal weight to each nonempty observation."""
    mask = mask.expand_as(value)
    count = mask.sum(dim=tuple(range(1, value.ndim)))
    total = (value * mask).sum(dim=tuple(range(1, value.ndim)))
    active = count > 0
    return (total / count.clamp_min(1) * active).sum() / active.sum().clamp_min(1)
