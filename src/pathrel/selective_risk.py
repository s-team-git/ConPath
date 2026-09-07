"""Selective risk with weighted coverage and label-independent handling of ties."""

from __future__ import annotations

import numpy as np


def risk_at_coverage(
    probabilities: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    """Accept high probabilities up to a requested mass, randomizing boundary ties.

    A fraction of every event at the boundary probability is accepted, independently
    of its label. Returned risk is the expected false-safe mass divided by the
    requested accepted mass. This is a descriptive operating curve, not a threshold
    fitted for deployment. Input weights are normalized within each call.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if p.ndim != 1 or y.shape != p.shape or w.shape != p.shape or p.size == 0:
        raise ValueError("probabilities, targets and weights must be matching nonempty vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and have positive mass")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0,1]")
    positive = w > 0
    p, y, w = p[positive], y[positive], w[positive]
    w = w / w.sum()
    levels, bins = np.unique(-p, return_inverse=True)
    mass = np.bincount(bins, weights=w, minlength=len(levels))
    error = np.bincount(bins, weights=w * (1 - y), minlength=len(levels))
    cumulative = np.cumsum(mass)
    index = min(int(np.searchsorted(cumulative, coverage, side="left")), len(levels) - 1)
    before = float(cumulative[index - 1]) if index else 0.0
    fraction = float(np.clip((coverage - before) / mass[index], 0, 1))
    false_mass = float(error[:index].sum() + fraction * error[index])
    return {
        "coverage": float(coverage),
        "false_safe_rate": false_mass / float(coverage),
        "boundary_probability": float(-levels[index]),
        "boundary_acceptance_fraction": fraction,
    }
