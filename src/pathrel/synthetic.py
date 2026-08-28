"""Deterministic ambiguous-corridor data for contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .labels import reachability_targets


@dataclass(frozen=True)
class SyntheticScene:
    observation_bev: np.ndarray
    target_classes: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    reachability_targets: np.ndarray
    max_clearance_cells: np.ndarray
    # Metadata is intentionally kept outside the observation tensor.  It makes the split and
    # conditional-prior audit reproducible without allowing a baseline to read the hidden world.
    context_family: int = 0
    template_id: int = 0
    doorway_is_open: bool = False


def ambiguous_corridor_scene(
    index: int,
    *,
    height: int = 24,
    width: int = 24,
    footprint_radii_cells: Sequence[int] = (0, 1, 2),
    context_family: int | None = None,
    template_id: int | None = None,
    context_open_probabilities: Sequence[float] = (0.2, 0.8),
) -> SyntheticScene:
    """Create a scene whose critical doorway is hidden in the observation.

    Consecutive indices share almost the same observation. Even indices contain a coherent open
    doorway behind the unknown band; odd indices contain a coherent closed wall. This makes the
    global topology stochastic instead of reducing uncertainty to independent pixels. Passing
    ``context_family=0`` or ``1`` switches to the paper-facing mode: repeated worlds share a
    template and visible context token, while the hidden doorway is drawn from the corresponding
    ``context_open_probabilities`` (default 0.2 and 0.8).
    """

    if height < 12 or width < 12:
        raise ValueError("synthetic maps must be at least 12x12")
    context_mode = context_family is not None
    if context_mode:
        context_family = int(context_family)
        probabilities = tuple(float(value) for value in context_open_probabilities)
        if context_family < 0 or context_family >= len(probabilities):
            raise ValueError("context_family must index context_open_probabilities")
        if not 0.0 <= probabilities[context_family] <= 1.0:
            raise ValueError("context open probabilities must lie in [0,1]")
    else:
        # Preserve the original paired open/closed contract used by the unit tests.  The
        # context-family mode below is the paper-facing generator with non-trivial priors.
        context_family = 0
        probabilities = (0.5,)

    if template_id is None:
        template_id = index // 2
    template_id = int(template_id)
    rng = np.random.default_rng(template_id + 17)
    free = np.ones((height, width), dtype=bool)
    free[[0, -1], :] = False
    free[:, [0, -1]] = False

    wall_col = width // 2
    free[1:-1, wall_col] = False
    if context_mode:
        # In context mode use an independent, deterministic Bernoulli draw.  This branch is
        # deliberately explicit so repeated worlds can share a template while differing only in
        # the hidden doorway state.
        world_rng = np.random.default_rng(1009 * (template_id + 1) + 9176 * (index + 1))
        doorway_is_open = bool(world_rng.random() < probabilities[context_family])
    else:
        doorway_is_open = index % 2 == 0
    doorway_center = height // 2 + int(rng.integers(-2, 3))
    doorway_half_height = int(rng.integers(2 if context_mode else 1, 4))
    if doorway_is_open:
        begin = max(1, doorway_center - doorway_half_height)
        end = min(height - 1, doorway_center + doorway_half_height + 1)
        free[begin:end, wall_col] = True

    # Secondary geometry supplies non-trivial but non-decisive visual structure.
    for _ in range(2):
        row = int(rng.integers(3, height - 3))
        col = int(rng.integers(2, max(3, wall_col - 2)))
        free[row, col] = False

    if len(probabilities) > 1:
        # Encode context with a small, physically consistent landmark outside the occluded strip.
        # The landmark is part of the visible world (and therefore also part of the target map),
        # so it cannot create contradictory free/blocked observation channels.
        marker_col = 1 + 3 * context_family
        free[1:3, marker_col : marker_col + 2] = False

    unknown = np.zeros_like(free)
    unknown[:, max(0, wall_col - 1) : min(width, wall_col + 2)] = True
    known = ~unknown
    observed_free = (free & known).astype(np.float32)
    observed_obstacle = ((~free) & known).astype(np.float32)
    observation = np.stack(
        (observed_free, observed_obstacle, unknown.astype(np.float32)), axis=0
    )

    target_classes = np.where(free, 0, 1).astype(np.int64)
    starts = np.asarray(
        ((height // 2, 4), (height // 3, 4)), dtype=np.int64
    )
    goals = np.asarray(
        ((height // 2, width - 5), (2 * height // 3, width - 5)), dtype=np.int64
    )
    labels, max_clearance = reachability_targets(
        free, starts, goals, footprint_radii_cells
    )
    return SyntheticScene(
        observation_bev=observation,
        target_classes=target_classes,
        starts=starts,
        goals=goals,
        reachability_targets=labels.astype(np.float32),
        max_clearance_cells=max_clearance,
        context_family=int(context_family),
        template_id=template_id,
        doorway_is_open=bool(doorway_is_open),
    )


def stack_scenes(scenes: Sequence[SyntheticScene]) -> dict[str, np.ndarray]:
    if not scenes:
        raise ValueError("at least one scene is required")
    return {
        "observation_bev": np.stack([scene.observation_bev for scene in scenes]),
        "target_classes": np.stack([scene.target_classes for scene in scenes]),
        "starts": np.stack([scene.starts for scene in scenes]),
        "goals": np.stack([scene.goals for scene in scenes]),
        "reachability_targets": np.stack(
            [scene.reachability_targets for scene in scenes]
        ),
        "max_clearance_cells": np.stack(
            [scene.max_clearance_cells for scene in scenes]
        ),
        "context_family": np.asarray([scene.context_family for scene in scenes], dtype=np.int64),
        "template_id": np.asarray([scene.template_id for scene in scenes], dtype=np.int64),
        "doorway_is_open": np.asarray([scene.doorway_is_open for scene in scenes], dtype=bool),
    }
