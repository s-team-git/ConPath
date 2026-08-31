"""Differentiable footprint-conditioned max-min reachability."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def _straight_through_extreme(
    values: Tensor,
    *,
    dim: int,
    take_maximum: bool,
    backward_temperature: float,
) -> Tensor:
    """Exact max/min forward with a smooth log-sum-exp backward surrogate."""

    hard = values.amax(dim=dim) if take_maximum else values.amin(dim=dim)
    if backward_temperature <= 0:
        return hard
    temperature = float(backward_temperature)
    if take_maximum:
        soft = temperature * torch.logsumexp(values / temperature, dim=dim)
    else:
        soft = -temperature * torch.logsumexp(-values / temperature, dim=dim)
    return soft + (hard - soft).detach()


def disk_footprint_min(
    safe: Tensor,
    radius_cells: int,
    *,
    backward_temperature: float = 0.1,
) -> Tensor:
    """Erode a soft/binary safe map by a discrete disk footprint.

    Args:
        safe: tensor with any leading dimensions and shape ``[..., H, W]``.
        radius_cells: non-negative integer disk radius.

    Outside-map values are zero (unsafe). The implementation is intended for the research
    prototype; large maps should use a custom sparse/morphological operator.
    """

    if safe.ndim < 2:
        raise ValueError("safe must end in [H, W]")
    radius = int(radius_cells)
    if radius < 0:
        raise ValueError("radius_cells must be non-negative")
    if radius == 0:
        return safe

    height, width = safe.shape[-2:]
    flat = safe.reshape(-1, 1, height, width)
    kernel_size = 2 * radius + 1
    padded = F.pad(flat, (radius, radius, radius, radius), value=0.0)
    patches = F.unfold(padded, kernel_size=kernel_size)
    patches = patches.view(flat.shape[0], kernel_size * kernel_size, height, width)

    coordinates = torch.arange(-radius, radius + 1, device=safe.device)
    row_offset, col_offset = torch.meshgrid(coordinates, coordinates, indexing="ij")
    disk_mask = (row_offset.square() + col_offset.square()) <= radius * radius
    selected = patches[:, disk_mask.flatten()]
    eroded = _straight_through_extreme(
        selected,
        dim=1,
        take_maximum=False,
        backward_temperature=backward_temperature,
    )
    return eroded.reshape(*safe.shape[:-2], height, width)


def _four_neighbor_max(values: Tensor, *, backward_temperature: float) -> Tensor:
    """Maximum over four-neighbor values, with zero outside the map."""

    height, width = values.shape[-2:]
    flat = values.reshape(-1, 1, height, width)
    padded = F.pad(flat, (1, 1, 1, 1), value=0.0)
    neighbors = torch.stack(
        (
            padded[:, :, 0:height, 1 : width + 1],
            padded[:, :, 2 : height + 2, 1 : width + 1],
            padded[:, :, 1 : height + 1, 0:width],
            padded[:, :, 1 : height + 1, 2 : width + 2],
        ),
        dim=0,
    )
    neighbors = _straight_through_extreme(
        neighbors,
        dim=0,
        take_maximum=True,
        backward_temperature=backward_temperature,
    )
    return neighbors.reshape_as(values)


def _validate_queries(starts: Tensor, goals: Tensor, height: int, width: int) -> None:
    if starts.ndim != 3 or starts.shape[-1] != 2:
        raise ValueError("starts must have shape [B, Q, 2]")
    if goals.shape != starts.shape:
        raise ValueError("goals must have the same shape as starts")
    if torch.any(starts[..., 0] < 0) or torch.any(starts[..., 0] >= height):
        raise ValueError("start row lies outside the map")
    if torch.any(starts[..., 1] < 0) or torch.any(starts[..., 1] >= width):
        raise ValueError("start column lies outside the map")
    if torch.any(goals[..., 0] < 0) or torch.any(goals[..., 0] >= height):
        raise ValueError("goal row lies outside the map")
    if torch.any(goals[..., 1] < 0) or torch.any(goals[..., 1] >= width):
        raise ValueError("goal column lies outside the map")


def maxmin_path_scores(
    node_scores: Tensor,
    starts: Tensor,
    goals: Tensor,
    *,
    max_steps: int | None = None,
    backward_temperature: float = 0.1,
) -> Tensor:
    """Bellman-style max-min path scores for many stochastic maps and queries.

    Args:
        node_scores: ``[B, K, H, W]`` values in ``[0, 1]``.
        starts/goals: integer row-column tensors ``[B, Q, 2]``.
        max_steps: relaxation iterations. ``H*W`` is exact for a finite four-neighbor grid;
            smaller values are a bounded-compute approximation and must be reported as such.

    Returns:
        Path bottleneck score ``[B, K, Q]``.
    """

    if node_scores.ndim != 4:
        raise ValueError("node_scores must have shape [B, K, H, W]")
    batch, samples, height, width = node_scores.shape
    if starts.shape[0] != batch:
        raise ValueError("query batch does not match node_scores")
    _validate_queries(starts, goals, height, width)
    if max_steps is None:
        max_steps = height * width
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    queries = starts.shape[1]
    start_index = starts[..., 0] * width + starts[..., 1]
    goal_index = goals[..., 0] * width + goals[..., 1]

    node_flat = node_scores.flatten(start_dim=2)
    start_values = node_flat.gather(
        2, start_index[:, None].expand(batch, samples, queries)
    )
    reach_flat = torch.zeros(
        (batch, samples, queries, height * width),
        dtype=node_scores.dtype,
        device=node_scores.device,
    )
    scatter_index = start_index[:, None, :, None].expand(batch, samples, queries, 1)
    reach_flat = reach_flat.scatter(3, scatter_index, start_values.unsqueeze(-1))
    reach = reach_flat.view(batch, samples, queries, height, width)
    capacity = node_scores[:, :, None].expand(batch, samples, queries, height, width)

    for _ in range(int(max_steps)):
        neighbor_best = _four_neighbor_max(
            reach, backward_temperature=backward_temperature
        )
        candidate = _straight_through_extreme(
            torch.stack((capacity, neighbor_best), dim=0),
            dim=0,
            take_maximum=False,
            backward_temperature=backward_temperature,
        )
        reach = _straight_through_extreme(
            torch.stack((reach, candidate), dim=0),
            dim=0,
            take_maximum=True,
            backward_temperature=backward_temperature,
        )

    reach_flat = reach.flatten(start_dim=3)
    gather_index = goal_index[:, None, :, None].expand(batch, samples, queries, 1)
    return reach_flat.gather(3, gather_index).squeeze(-1)


def maxmin_path_scores_shared_start(
    node_scores: Tensor,
    starts: Tensor,
    goals: Tensor,
    *,
    max_steps: int | None = None,
    backward_temperature: float = 0.1,
) -> Tensor:
    """Maximum-bottleneck scores when all queries in each batch item share one start.

    FlatLands natural queries use the camera cell as the common start and vary only the goal. This
    implementation propagates one ``[B,K,H,W]`` field per stochastic map, then gathers all goals;
    the generic :func:`maxmin_path_scores` expands that field by ``Q`` and is consequently much
    more expensive on dense public maps. The forward semantics are identical for shared starts.
    """

    if node_scores.ndim != 4:
        raise ValueError("node_scores must have shape [B,K,H,W]")
    batch, samples, height, width = node_scores.shape
    if starts.shape[0] != batch:
        raise ValueError("query batch does not match node_scores")
    _validate_queries(starts, goals, height, width)
    if starts.shape[1] == 0:
        return node_scores.new_empty((batch, samples, 0))
    if not torch.all(starts == starts[:, :1]):
        raise ValueError("maxmin_path_scores_shared_start requires one start per batch item")
    if max_steps is None:
        max_steps = height * width
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    start_index = starts[:, 0, 0] * width + starts[:, 0, 1]
    goal_index = goals[..., 0] * width + goals[..., 1]
    node_flat = node_scores.flatten(start_dim=2)
    start_values = node_flat.gather(2, start_index[:, None, None].expand(batch, samples, 1))
    reach_flat = torch.zeros(
        (batch, samples, height * width), dtype=node_scores.dtype, device=node_scores.device
    )
    reach_flat = reach_flat.scatter(2, start_index[:, None, None].expand(batch, samples, 1), start_values)
    reach = reach_flat.view(batch, samples, height, width)
    capacity = node_scores
    for _ in range(int(max_steps)):
        neighbor_best = _four_neighbor_max(reach, backward_temperature=backward_temperature)
        candidate = _straight_through_extreme(
            torch.stack((capacity, neighbor_best), dim=0),
            dim=0,
            take_maximum=False,
            backward_temperature=backward_temperature,
        )
        reach = _straight_through_extreme(
            torch.stack((reach, candidate), dim=0),
            dim=0,
            take_maximum=True,
            backward_temperature=backward_temperature,
        )
    return reach.flatten(start_dim=2).gather(
        2, goal_index[:, None].expand(batch, samples, goal_index.shape[1])
    )


def multi_radius_reachability(
    safe_samples: Tensor,
    starts: Tensor,
    goals: Tensor,
    footprint_radii_cells: Sequence[int] | Tensor,
    *,
    surrogate_safe_samples: Tensor | None = None,
    max_steps: int | None = None,
    backward_temperature: float = 0.1,
    shared_start: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute a discrete footprint-conditioned reliability curve.

    ``surrogate_safe_samples`` enables a high-level straight-through estimator: the forward event
    is still computed exactly from the hard binary maps, while the backward pass follows the
    continuous maximum-bottleneck score of the relaxed maps. This avoids repeatedly applying soft
    backward extrema to an already-hard state, whose gradients can miss the critical cut.

    Returns:
        reachability: Monte-Carlo mean ``[B, Q, R]``.
        sample_events: straight-through event samples ``[B, K, Q, R]``.
    """

    if safe_samples.ndim != 4:
        raise ValueError("safe_samples must have shape [B, K, H, W]")
    if surrogate_safe_samples is not None:
        if surrogate_safe_samples.shape != safe_samples.shape:
            raise ValueError("surrogate_safe_samples must have the same shape as safe_samples")
        if not surrogate_safe_samples.is_floating_point():
            raise ValueError("surrogate_safe_samples must be floating point")
    if isinstance(footprint_radii_cells, Tensor):
        radii = [int(value) for value in footprint_radii_cells.detach().cpu().tolist()]
    else:
        radii = [int(value) for value in footprint_radii_cells]
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("footprint radii must be non-empty and non-negative")

    events = []
    for radius in radii:
        if surrogate_safe_samples is None:
            center_safe = disk_footprint_min(
                safe_samples, radius, backward_temperature=backward_temperature
            )
            path_fn = maxmin_path_scores_shared_start if shared_start else maxmin_path_scores
            event = path_fn(
                center_safe, starts, goals, max_steps=max_steps, backward_temperature=backward_temperature
            )
        else:
            # Keep the probability semantics in the hard forward pass. The relaxed branch is a
            # bottleneck-score surrogate only; it is never returned or reported as an event.
            hard_center_safe = disk_footprint_min(
                safe_samples.detach(), radius, backward_temperature=0.0
            )
            path_fn = maxmin_path_scores_shared_start if shared_start else maxmin_path_scores
            hard_event = path_fn(
                hard_center_safe,
                starts,
                goals,
                max_steps=max_steps,
                backward_temperature=0.0,
            )
            surrogate_center_safe = disk_footprint_min(
                surrogate_safe_samples, radius, backward_temperature=0.0
            )
            surrogate_score = path_fn(
                surrogate_center_safe,
                starts,
                goals,
                max_steps=max_steps,
                backward_temperature=0.0,
            )
            event = surrogate_score + (hard_event - surrogate_score).detach()
        events.append(event)
    sample_events = torch.stack(events, dim=-1)
    reachability = sample_events.mean(dim=1)
    return reachability, sample_events
