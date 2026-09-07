"""Posterior controls with explicit empirical-marginal and support contracts."""

from __future__ import annotations

import numpy as np


MEAN_MAP_PROJECTION_VERSION = "v2: observed blocked wins; invalid support blocked last"


def threshold_supported_mean_map(
    probability: np.ndarray, observation: np.ndarray, valid_support: np.ndarray,
    *, threshold: float = 0.5,
) -> np.ndarray:
    """Project a posterior mean onto the same hard evidence as PathRelNet.

    Observed free cells outside valid support must never reopen a path. Blocked
    observations also take precedence when the observation channels overlap.
    """
    probability = np.asarray(probability)
    observation = np.asarray(observation)
    valid_support = np.asarray(valid_support, dtype=bool)
    if probability.ndim != 2 or valid_support.shape != probability.shape:
        raise ValueError("probability and valid_support must have matching [H,W] shapes")
    if observation.ndim != 3 or observation.shape[0] < 3 or observation.shape[1:] != probability.shape:
        raise ValueError("observation must have shape [C,H,W] with C >= 3")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0,1]")
    return ((probability >= threshold) | (observation[0] > 0.5)) & ~(observation[1] > 0.5) & valid_support


def shuffle_worlds_by_cell(worlds: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Independently permute the world index at every cell of a [K,H,W] sample set.

    Each cell retains exactly its original free count. Observed and invalid cells
    that are constant across worlds therefore remain fixed. Labels are not used.
    """
    worlds = np.asarray(worlds)
    if worlds.ndim != 3 or worlds.shape[0] < 2 or worlds.dtype != np.bool_:
        raise ValueError("worlds must be a boolean [K,H,W] array with K >= 2")
    return generator.permuted(worlds, axis=0)
