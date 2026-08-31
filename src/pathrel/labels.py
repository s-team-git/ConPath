"""Exact offline labels for discrete footprint-conditioned connectivity.

This module depends only on NumPy. It defines the geometry used by the first prototype:

* cells have binary support-valid free/occupied state;
* outside the grid is occupied;
* a footprint is a disk with an integer radius in grid cells;
* motion uses four-neighbor cell-center transitions.

The returned clearance is a *radius*, not a corridor width.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable, Sequence

import numpy as np


GridPoint = tuple[int, int]


@dataclass(frozen=True)
class ClearanceResult:
    reachable: bool
    max_radius_cells: int


def _validate_free_map(free: np.ndarray) -> np.ndarray:
    array = np.asarray(free, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"free must have shape [H, W], got {array.shape}")
    if array.size == 0:
        raise ValueError("free map cannot be empty")
    return array


def _validate_point(point: Sequence[int], height: int, width: int, name: str) -> GridPoint:
    if len(point) != 2:
        raise ValueError(f"{name} must contain (row, col)")
    row, col = int(point[0]), int(point[1])
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"{name} {(row, col)} lies outside a {height}x{width} map")
    return row, col


def _squared_distance_transform_1d(cost: np.ndarray) -> np.ndarray:
    """Exact lower envelope of parabolas for one finite one-dimensional cost array."""

    cost = np.asarray(cost, dtype=np.float64)
    length = int(cost.size)
    if length == 0 or not np.any(np.isfinite(cost)):
        raise ValueError("distance-transform input must contain a finite value")
    sites = np.empty(length, dtype=np.int64)
    boundaries = np.empty(length + 1, dtype=np.float64)
    envelope_size = 0
    sites[0] = int(np.flatnonzero(np.isfinite(cost))[0])
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for query in range(int(sites[0]) + 1, length):
        if not np.isfinite(cost[query]):
            continue
        while True:
            site = int(sites[envelope_size])
            intersection = (
                (cost[query] + query * query) - (cost[site] + site * site)
            ) / (2.0 * (query - site))
            if intersection > boundaries[envelope_size] or envelope_size == 0:
                break
            envelope_size -= 1
        if envelope_size == 0 and intersection <= boundaries[envelope_size]:
            # This is only reachable for unusual non-binary finite costs.  Replacing the first
            # site keeps the helper mathematically complete without affecting the binary EDT.
            sites[0] = query
            boundaries[0] = -np.inf
            boundaries[1] = np.inf
            continue
        envelope_size += 1
        sites[envelope_size] = query
        boundaries[envelope_size] = intersection
        boundaries[envelope_size + 1] = np.inf

    output = np.empty(length, dtype=np.float64)
    envelope_index = 0
    for query in range(length):
        while boundaries[envelope_index + 1] < query:
            envelope_index += 1
        site = int(sites[envelope_index])
        output[query] = (query - site) ** 2 + cost[site]
    return output


def _squared_distance_to_obstacle(free: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance to occupied support, including one occupied outer border."""

    padded = np.pad(free, 1, mode="constant", constant_values=False)
    initial = np.where(padded, np.inf, 0.0)
    vertical = np.empty_like(initial)
    for column in range(initial.shape[1]):
        vertical[:, column] = _squared_distance_transform_1d(initial[:, column])
    squared = np.empty_like(vertical)
    for row in range(vertical.shape[0]):
        squared[row] = _squared_distance_transform_1d(vertical[row])
    return squared[1:-1, 1:-1]


def clearance_radius_map(free: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """Largest collision-free integer disk radius at every cell.

    The definition exactly matches a discrete disk containing all offsets ``(dy, dx)`` with
    ``dy**2 + dx**2 <= r**2``. A one-cell occupied border makes leaving the map unsafe.

    The implementation is the exact linear-time squared Euclidean distance transform of
    Felzenszwalb and Huttenlocher, applied separably. ``chunk_size`` is retained as a backwards-
    compatible ignored argument from the original pairwise reference implementation.
    """

    free = _validate_free_map(free)
    # An obstacle exactly at distance d is included by a radius-d disk, so the largest safe
    # integer radius is ceil(d)-1. Occupied centers are explicitly marked -1.
    del chunk_size
    squared_distance = _squared_distance_to_obstacle(free).astype(np.int64)
    floor_root = np.sqrt(squared_distance).astype(np.int64)
    ceil_root = floor_root + (floor_root * floor_root < squared_distance)
    radius = ceil_root - 1
    radius[~free] = -1
    return radius


def maximum_clearance_map(
    free: np.ndarray,
    start: Sequence[int],
    *,
    clearance: np.ndarray | None = None,
    stop_points: Iterable[Sequence[int]] | None = None,
) -> np.ndarray:
    """Maximum four-neighbor bottleneck clearance from one start to every reached cell.

    ``stop_points`` permits a bounded query audit to stop once all requested terminals have been
    finalized. Unreached or unprocessed cells retain ``-1``.
    """

    free = _validate_free_map(free)
    height, width = free.shape
    start_rc = _validate_point(start, height, width, "start")
    if clearance is None:
        clearance = clearance_radius_map(free)
    clearance = np.asarray(clearance)
    if clearance.shape != free.shape:
        raise ValueError("clearance must have the same shape as free")

    best = np.full((height, width), -1, dtype=np.int64)
    if not free[start_rc]:
        return best

    remaining: set[GridPoint] | None = None
    if stop_points is not None:
        remaining = {
            _validate_point(point, height, width, "stop_point") for point in stop_points
        }
        if not remaining:
            return best

    best[start_rc] = int(clearance[start_rc])
    queue: list[tuple[int, int, int]] = [(-int(best[start_rc]), start_rc[0], start_rc[1])]
    while queue:
        negative_capacity, row, col = heapq.heappop(queue)
        capacity = -negative_capacity
        if capacity != int(best[row, col]):
            continue
        if remaining is not None:
            remaining.discard((row, col))
            if not remaining:
                break
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            if not free[n_row, n_col]:
                continue
            candidate = min(capacity, int(clearance[n_row, n_col]))
            if candidate > int(best[n_row, n_col]):
                best[n_row, n_col] = candidate
                heapq.heappush(queue, (-candidate, n_row, n_col))
    return best


def maximum_clearance_path(
    free: np.ndarray,
    start: Sequence[int],
    goal: Sequence[int],
    *,
    clearance: np.ndarray | None = None,
) -> ClearanceResult:
    """Return the exact four-neighbor maximum-bottleneck path clearance."""

    free = _validate_free_map(free)
    height, width = free.shape
    start_rc = _validate_point(start, height, width, "start")
    goal_rc = _validate_point(goal, height, width, "goal")

    if not free[start_rc] or not free[goal_rc]:
        return ClearanceResult(False, -1)

    if clearance is None:
        clearance = clearance_radius_map(free)
    clearance = np.asarray(clearance)
    if clearance.shape != free.shape:
        raise ValueError("clearance must have the same shape as free")

    best = maximum_clearance_map(
        free, start_rc, clearance=clearance, stop_points=[goal_rc]
    )
    capacity = int(best[goal_rc])
    return ClearanceResult(capacity >= 0, capacity)


def reachability_targets(
    free: np.ndarray,
    starts: Iterable[Sequence[int]],
    goals: Iterable[Sequence[int]],
    footprint_radii_cells: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate exact query labels and maximum-clearance values.

    Returns:
        labels: boolean array ``[Q, R]``.
        max_clearance: integer array ``[Q]`` with ``-1`` for unreachable queries.
    """

    free = _validate_free_map(free)
    starts = list(starts)
    goals = list(goals)
    radii = np.asarray(list(footprint_radii_cells), dtype=np.int64)
    if len(starts) != len(goals):
        raise ValueError("starts and goals must contain the same number of queries")
    if radii.ndim != 1 or radii.size == 0 or np.any(radii < 0):
        raise ValueError("footprint radii must be a non-empty list of non-negative integers")

    clearance = clearance_radius_map(free)
    labels = np.zeros((len(starts), radii.size), dtype=bool)
    max_clearance = np.full(len(starts), -1, dtype=np.int64)
    for query_index, (start, goal) in enumerate(zip(starts, goals)):
        result = maximum_clearance_path(free, start, goal, clearance=clearance)
        max_clearance[query_index] = result.max_radius_cells
        if result.reachable:
            labels[query_index] = result.max_radius_cells >= radii
    return labels, max_clearance


def merge_tree_bottleneck_scores(
    node_scores: np.ndarray,
    starts: Iterable[Sequence[int]],
    goals: Iterable[Sequence[int]],
) -> np.ndarray:
    """Exact all-query max-min scores using a Kruskal merge tree.

    For a fixed map, every four-neighbor edge receives the minimum of its endpoint scores.  The
    maximum bottleneck value between two terminals is the minimum edge on their maximum spanning
    tree path.  A Kruskal reconstruction tree stores those merge levels, so all subsequent
    terminal queries are answered with an LCA lookup rather than another ``H*W`` relaxation.

    This is an exact-forward NumPy reference for the scalable operator planned in the paper.  It is
    intentionally separate from the differentiable PyTorch prototype: a future CUDA implementation
    can keep this forward contract and provide a soft backward surrogate.
    """

    scores = np.asarray(node_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("node_scores must have non-empty shape [H,W]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("node_scores must be finite")
    height, width = scores.shape
    starts = list(starts)
    goals = list(goals)
    if len(starts) != len(goals):
        raise ValueError("starts and goals must contain the same number of queries")

    def point_index(point: Sequence[int], name: str) -> int:
        row, col = _validate_point(point, height, width, name)
        return row * width + col

    number_cells = height * width
    edge_weights: list[tuple[float, int, int]] = []
    for row in range(height):
        for col in range(width):
            current = row * width + col
            if row + 1 < height:
                edge_weights.append((min(scores[row, col], scores[row + 1, col]), current, current + width))
            if col + 1 < width:
                edge_weights.append((min(scores[row, col], scores[row, col + 1]), current, current + 1))
    edge_weights.sort(key=lambda item: item[0], reverse=True)

    # DSU components and reconstruction-tree nodes have separate representatives.  Each successful
    # Kruskal union creates one parent above the two component trees at exactly the edge weight.
    dsu_parent = np.arange(number_cells, dtype=np.int64)
    dsu_size = np.ones(number_cells, dtype=np.int64)
    component_tree = np.arange(number_cells, dtype=np.int64)
    values = [float(value) for value in scores.reshape(-1)]
    left = [-1] * number_cells
    right = [-1] * number_cells

    def find(value: int) -> int:
        while dsu_parent[value] != value:
            dsu_parent[value] = dsu_parent[dsu_parent[value]]
            value = int(dsu_parent[value])
        return value

    for weight, first, second in edge_weights:
        root_first, root_second = find(first), find(second)
        if root_first == root_second:
            continue
        tree_first, tree_second = int(component_tree[root_first]), int(component_tree[root_second])
        merge_node = len(values)
        values.append(float(weight))
        left.append(tree_first)
        right.append(tree_second)
        if dsu_size[root_first] < dsu_size[root_second]:
            root_first, root_second = root_second, root_first
        dsu_parent[root_second] = root_first
        dsu_size[root_first] += dsu_size[root_second]
        component_tree[root_first] = merge_node

    number_nodes = len(values)
    parent = np.full(number_nodes, -1, dtype=np.int64)
    depth = np.zeros(number_nodes, dtype=np.int64)
    for merge_node, (child_left, child_right) in enumerate(zip(left[number_cells:], right[number_cells:]), start=number_cells):
        if child_left >= 0:
            parent[child_left] = merge_node
            parent[child_right] = merge_node
    roots = np.flatnonzero(parent < 0)
    stack = [(int(root), 0) for root in roots]
    while stack:
        node, node_depth = stack.pop()
        depth[node] = node_depth
        if node >= number_cells:
            stack.append((left[node], node_depth + 1))
            stack.append((right[node], node_depth + 1))

    levels = max(1, int(np.ceil(np.log2(max(2, number_nodes)))) + 1)
    ancestors = np.full((levels, number_nodes), -1, dtype=np.int64)
    ancestors[0] = parent
    for level in range(1, levels):
        previous = ancestors[level - 1]
        valid = previous >= 0
        ancestors[level, valid] = previous[previous[valid]]

    def lca(first: int, second: int) -> int:
        if depth[first] < depth[second]:
            first, second = second, first
        difference = int(depth[first] - depth[second])
        bit = 0
        while difference:
            if difference & 1:
                first = int(ancestors[bit, first])
            difference >>= 1
            bit += 1
        if first == second:
            return first
        for level in range(levels - 1, -1, -1):
            first_parent, second_parent = int(ancestors[level, first]), int(ancestors[level, second])
            if first_parent >= 0 and first_parent != second_parent:
                first, second = first_parent, second_parent
        return int(parent[first])

    output = np.empty(len(starts), dtype=np.float64)
    for query_index, (start, goal) in enumerate(zip(starts, goals)):
        first, second = point_index(start, "start"), point_index(goal, "goal")
        output[query_index] = values[lca(first, second)]
    return output
