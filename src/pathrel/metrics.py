"""Calibration metrics used by the synthetic death test.

The metrics intentionally operate on already materialised event probabilities.  This keeps the
evaluation layer independent of a particular decoder and makes it harder to accidentally report
``soft max-min`` scores as event probabilities.  A prediction is a probability in ``[0, 1]`` and
the target is a realised Bernoulli event.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _arrays(probabilities: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if probabilities.size == 0:
        raise ValueError("probabilities cannot be empty")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(targets)):
        raise ValueError("probabilities and targets must be finite")
    if np.any(probabilities < -1e-8) or np.any(probabilities > 1.0 + 1e-8):
        raise ValueError("probabilities must lie in [0,1]")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("targets must lie in [0,1]")
    return np.clip(probabilities, 0.0, 1.0), targets


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probabilities, targets = _arrays(probabilities, targets)
    return float(np.mean((probabilities - targets) ** 2))


def log_score(probabilities: np.ndarray, targets: np.ndarray, epsilon: float = 1e-6) -> float:
    probabilities, targets = _arrays(probabilities, targets)
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(-np.mean(targets * np.log(probabilities) + (1.0 - targets) * np.log1p(-probabilities)))


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    """Return equal-width ECE and the reliability diagram bins."""

    if bins < 1:
        raise ValueError("bins must be positive")
    probabilities, targets = _arrays(probabilities, targets)
    flat_probabilities = probabilities.reshape(-1)
    flat_targets = targets.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, flat_probabilities, side="right") - 1, bins - 1)
    diagram: list[dict[str, float | int]] = []
    ece = 0.0
    total = float(flat_probabilities.size)
    for index in range(bins):
        selected = assignments == index
        count = int(np.sum(selected))
        if count:
            confidence = float(np.mean(flat_probabilities[selected]))
            accuracy = float(np.mean(flat_targets[selected]))
            weight = count / total
            ece += weight * abs(confidence - accuracy)
        else:
            confidence = float((edges[index] + edges[index + 1]) / 2.0)
            accuracy = float("nan")
        diagram.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), diagram


def false_safe_rate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    confidence_threshold: float = 0.8,
) -> float:
    """Fraction of high-confidence safe predictions that are actually unsafe."""

    probabilities, targets = _arrays(probabilities, targets)
    selected = probabilities.reshape(-1) >= float(confidence_threshold)
    if not np.any(selected):
        return float("nan")
    return float(np.mean(targets.reshape(-1)[selected] < 0.5))


def summarize(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    ece, diagram = expected_calibration_error(probabilities, targets)
    return {
        "brier": brier_score(probabilities, targets),
        "nll": log_score(probabilities, targets),
        "ece": ece,
        "false_safe_rate@0.8": false_safe_rate(probabilities, targets),
        "reliability": diagram,
        "count": int(np.asarray(probabilities).size),
    }
