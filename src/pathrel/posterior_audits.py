"""Controls that alter posterior dependence without changing empirical marginals."""

from __future__ import annotations

import numpy as np


def shuffle_worlds_by_cell(worlds: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Independently permute the world index at every cell of a [K,H,W] sample set.

    Each cell retains exactly its original free count. Observed and invalid cells
    that are constant across worlds therefore remain fixed. Labels are not used.
    """
    worlds = np.asarray(worlds)
    if worlds.ndim != 3 or worlds.shape[0] < 2 or worlds.dtype != np.bool_:
        raise ValueError("worlds must be a boolean [K,H,W] array with K >= 2")
    return generator.permuted(worlds, axis=0)
