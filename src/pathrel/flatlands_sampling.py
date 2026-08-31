"""Pure-NumPy exact event sampling for deterministic and independent-cell baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .labels import reachability_targets


@dataclass(frozen=True)
class CompletionEventTask:
    global_id: str
    free_probability: np.ndarray
    observed_free: np.ndarray
    unknown: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    candidate_indices: np.ndarray
    radii_cells: tuple[int, ...]
    posterior_samples: int
    seed: int


@dataclass(frozen=True)
class CompletionEventResult:
    global_id: str
    candidate_indices: np.ndarray
    deterministic: np.ndarray
    independent: np.ndarray


def stable_observation_seed(seed: int, global_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{global_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def completion_event_probabilities(task: CompletionEventTask) -> CompletionEventResult:
    """Evaluate one observation with exact disk-footprint connectivity.

    The per-observation RNG key makes outputs invariant to worker count and task order.
    """

    probability = np.asarray(task.free_probability, dtype=np.float64)
    observed_free = np.asarray(task.observed_free, dtype=bool)
    unknown = np.asarray(task.unknown, dtype=bool)
    starts = np.asarray(task.starts, dtype=np.int64)
    goals = np.asarray(task.goals, dtype=np.int64)
    candidate_indices = np.asarray(task.candidate_indices, dtype=np.int64)
    if probability.ndim != 2 or probability.shape != observed_free.shape or probability.shape != unknown.shape:
        raise ValueError("probability, observed_free, and unknown maps must share shape [H,W]")
    if starts.ndim != 2 or starts.shape != goals.shape or starts.shape[-1] != 2:
        raise ValueError("starts/goals must share shape [Q,2]")
    if candidate_indices.shape != (starts.shape[0],):
        raise ValueError("candidate_indices must have shape [Q]")
    if task.posterior_samples < 1:
        raise ValueError("posterior_samples must be positive")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("free_probability must be finite and in [0,1]")
    if np.any(observed_free & unknown):
        raise ValueError("observed_free and unknown cannot overlap")

    deterministic_map = observed_free | (unknown & (probability >= 0.5))
    deterministic, _ = reachability_targets(
        deterministic_map, starts, goals, task.radii_cells
    )
    rng = np.random.default_rng(stable_observation_seed(task.seed, task.global_id))
    event_sum = np.zeros_like(deterministic, dtype=np.float64)
    for _ in range(task.posterior_samples):
        sampled = observed_free | (unknown & (rng.random(probability.shape) < probability))
        events, _ = reachability_targets(sampled, starts, goals, task.radii_cells)
        event_sum += events
    return CompletionEventResult(
        global_id=task.global_id,
        candidate_indices=candidate_indices,
        deterministic=deterministic.astype(np.float64),
        independent=event_sum / task.posterior_samples,
    )
