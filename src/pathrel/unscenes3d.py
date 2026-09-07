"""UnScenes3D-to-ConPath contract adapter.

The adapter keeps the raw sensor observation separate from the official occupancy
target.  It uses the release's 256x256x32 grid (0.3 m voxels, x forward and y
lateral) and projects the semantic labels to a conservative 2-D support slice:
class 11 (``driveable_surface``) is free; any other labeled class blocks a cell.
Cells without an occupancy label stay outside the target-valid mask and are never
silently treated as free.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


GRID_SHAPE = (256, 256)
GRID_Z = 32
VOXEL_SIZE_M = 0.3
POINT_CLOUD_RANGE_M = (0.0, -38.4, -4.0, 76.8, 38.4, 5.6)
Y_ORIGIN_M = POINT_CLOUD_RANGE_M[1]
FREE_CLASS = 11  # official unstructured-road class: driveable_surface


@dataclass(frozen=True)
class SupportSlice:
    """Conservative 2-D support target and its validity mask."""

    free: np.ndarray
    valid: np.ndarray

    @property
    def blocked(self) -> np.ndarray:
        return self.valid & ~self.free


@dataclass(frozen=True)
class UnScenesFrame:
    timestamp: str
    input_bev: np.ndarray
    target_free: np.ndarray
    target_valid: np.ndarray
    starts: np.ndarray
    goals: np.ndarray


def occupancy_support(occupancy: np.ndarray) -> SupportSlice:
    """Project sparse ``occ/*.npy`` rows to a conservative support slice.

    The release stores rows as ``x_index, y_index, z_index, class``.  A cell is
    valid only if at least one voxel is labeled.  If both a driveable and an
    obstacle voxel occur at one (x, y), the obstacle wins conservatively.
    """

    array = np.asarray(occupancy)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"occupancy must have shape [N,4], got {array.shape}")
    if array.dtype.kind not in "iu":
        raise ValueError("occupancy rows must be integer-valued")
    coords = array[:, :3]
    if np.any(coords < 0) or np.any(coords[:, 0] >= GRID_SHAPE[0]) or np.any(coords[:, 1] >= GRID_SHAPE[1]) or np.any(coords[:, 2] >= 32):
        raise ValueError("occupancy voxel lies outside the official 256x256x32 grid")
    classes = array[:, 3]
    valid = np.zeros(GRID_SHAPE, dtype=bool)
    blocked = np.zeros(GRID_SHAPE, dtype=bool)
    valid[coords[:, 0], coords[:, 1]] = True
    blocked[coords[classes != FREE_CLASS, 0], coords[classes != FREE_CLASS, 1]] = True
    free = valid & ~blocked
    return SupportSlice(free=free, valid=valid)


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> Iterable[tuple[int, int]]:
    """Yield inclusive integer cells on a 2-D Bresenham segment."""

    row0, col0 = start
    row1, col1 = end
    d_row, d_col = abs(row1 - row0), abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    error = d_row - d_col
    row, col = row0, col0
    while True:
        yield row, col
        if (row, col) == (row1, col1):
            return
        doubled = 2 * error
        if doubled > -d_col:
            error -= d_col
            row += step_row
        if doubled < d_row:
            error += d_row
            col += step_col


def _ground_profile(
    points: np.ndarray,
    *,
    bin_size_m: float = 1.2,
    lateral_limit_m: float = 20.0,
    quantile: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a label-free longitudinal ground-height profile.

    UnScenes3D LiDAR returns include road/support returns.  Treating every return
    as an occupied endpoint (the historical adapter behavior) therefore turns a
    largely traversable road into a wall.  This small robust profile uses only
    the raw point cloud: in each forward bin it takes a low quantile of returns
    near the vehicle centreline.  It is deliberately not derived from occupancy
    or elevation labels.  Empty bins are linearly interpolated by the caller.
    """

    if bin_size_m <= 0:
        raise ValueError("ground profile bin size must be positive")
    if lateral_limit_m <= 0:
        raise ValueError("ground profile lateral limit must be positive")
    if not 0.0 < quantile <= 0.5:
        raise ValueError("ground profile quantile must lie in (0, 0.5]")
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {array.shape}")
    xyz = array[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    selected = finite & (xyz[:, 0] >= 0.0) & (xyz[:, 0] < POINT_CLOUD_RANGE_M[3])
    selected &= np.abs(xyz[:, 1]) <= lateral_limit_m
    if not np.any(selected):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    x = xyz[selected, 0]
    z = xyz[selected, 2]
    bin_count = int(np.ceil(POINT_CLOUD_RANGE_M[3] / bin_size_m))
    bin_index = np.floor(x / bin_size_m).astype(np.int64)
    bin_index = np.clip(bin_index, 0, bin_count - 1)
    # Sort once, rather than constructing a full-size boolean mask for every
    # forward bin.  A frame has tens of thousands of returns; the distinction is
    # material when the adapter is replayed over all train/validation frames.
    order = np.argsort(bin_index, kind="stable")
    sorted_bins = bin_index[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1]
    ends = np.r_[starts[1:], len(order)]
    centers: list[float] = []
    heights: list[float] = []
    for begin, end in zip(starts.tolist(), ends.tolist()):
        index = int(sorted_bins[begin])
        values = z[order[begin:end]]
        centers.append((index + 0.5) * bin_size_m)
        heights.append(float(np.quantile(values, quantile)))
    return np.asarray(centers, dtype=np.float32), np.asarray(heights, dtype=np.float32)


def _interpolated_ground_height(
    points: np.ndarray,
    *,
    bin_size_m: float = 1.2,
    lateral_limit_m: float = 20.0,
    quantile: float = 0.15,
) -> np.ndarray:
    """Return a per-point robust ground-height estimate from raw geometry."""

    xyz = np.asarray(points)[:, :3]
    centers, heights = _ground_profile(
        points,
        bin_size_m=bin_size_m,
        lateral_limit_m=lateral_limit_m,
        quantile=quantile,
    )
    if centers.size == 0:
        fallback = float(np.nanmedian(xyz[:, 2])) if xyz.size else 0.0
        return np.full((len(xyz),), fallback, dtype=np.float32)
    return np.interp(
        xyz[:, 0], centers, heights, left=float(heights[0]), right=float(heights[-1])
    ).astype(np.float32, copy=False)


def lidar_observation(
    points: np.ndarray,
    *,
    endpoint_policy: str = "blocked",
    ground_margin_m: float = 0.35,
    ground_bin_size_m: float = 1.2,
    ground_lateral_limit_m: float = 20.0,
    ground_quantile: float = 0.15,
) -> np.ndarray:
    """Rasterize a label-free LiDAR ray observation into ``[3,256,256]``.

    Channel order matches :class:`PathRelNet`: observed-free ray cells, observed
    blocked returns, and unknown cells.  ``endpoint_policy='blocked'`` preserves
    the historical all-returns-as-obstacles adapter.  ``endpoint_policy='ground'``
    uses only point heights to classify a lowest return near a robust support
    profile as free; it never reads occupancy/elevation labels.
    """

    if endpoint_policy not in {"blocked", "ground"}:
        raise ValueError("endpoint_policy must be 'blocked' or 'ground'")
    if ground_margin_m < 0:
        raise ValueError("ground margin must be non-negative")
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {array.shape}")
    finite = np.isfinite(array[:, :3]).all(axis=1)
    x, y = array[:, 0], array[:, 1]
    in_range = finite & (x >= 0.0) & (x < GRID_SHAPE[0] * VOXEL_SIZE_M) & (y >= Y_ORIGIN_M) & (y < -Y_ORIGIN_M)
    selected_xyz = array[in_range, :3]
    rows = np.floor(selected_xyz[:, 0] / VOXEL_SIZE_M).astype(np.int64)
    cols = np.floor((selected_xyz[:, 1] - Y_ORIGIN_M) / VOXEL_SIZE_M).astype(np.int64)
    in_grid = (rows >= 0) & (rows < GRID_SHAPE[0]) & (cols >= 0) & (cols < GRID_SHAPE[1])
    rows, cols = rows[in_grid], cols[in_grid]
    selected_xyz = selected_xyz[in_grid]
    endpoint_status: dict[tuple[int, int], bool] = {}
    if endpoint_policy == "blocked":
        endpoint_status = {(int(row), int(col)): True for row, col in zip(rows, cols)}
    else:
        # A cell is free when its lowest return is compatible with the robust
        # support profile.  An elevated lowest return is evidence of an obstacle.
        # Multiple returns are aggregated conservatively: one elevated return does
        # not override a ground return, but a cell with no ground return is blocked.
        ground = _interpolated_ground_height(
            selected_xyz,
            bin_size_m=ground_bin_size_m,
            lateral_limit_m=ground_lateral_limit_m,
            quantile=ground_quantile,
        )
        keys = rows * GRID_SHAPE[1] + cols
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
        ends = np.r_[starts[1:], len(order)]
        for begin, end in zip(starts.tolist(), ends.tolist()):
            indices = order[begin:end]
            lowest = int(indices[np.argmin(selected_xyz[indices, 2])])
            key = int(sorted_keys[begin])
            endpoint_status[(key // GRID_SHAPE[1], key % GRID_SHAPE[1])] = bool(
                selected_xyz[lowest, 2] - ground[lowest] > ground_margin_m
            )
    endpoints = sorted(endpoint_status)
    observed_free = np.zeros(GRID_SHAPE, dtype=bool)
    observed_blocked = np.zeros(GRID_SHAPE, dtype=bool)
    origin = (0, int(round((0.0 - Y_ORIGIN_M) / VOXEL_SIZE_M)))
    for endpoint in endpoints:
        segment = list(_bresenham(origin, endpoint))
        for cell in segment[:-1]:
            observed_free[cell] = True
        if endpoint_status[endpoint]:
            observed_blocked[endpoint] = True
        else:
            observed_free[endpoint] = True
    observed_free &= ~observed_blocked
    unknown = ~(observed_free | observed_blocked)
    return np.stack((observed_free, observed_blocked, unknown), axis=0).astype(np.float32)


def deterministic_queries(
    target_valid: np.ndarray,
    *,
    distances_cells: Sequence[int] = (13, 27, 40),
    angles_deg: Sequence[int] = tuple(range(0, 360, 30)),
    anchor_hint: tuple[int, int] = (96, 128),
    start_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed polar stencil using bounds/validity and optional input geometry.

    The target-free class is not inspected.  By default the anchor is the nearest
    valid cell to ``anchor_hint``.  When ``start_mask`` is supplied, the anchor is
    chosen from ``target_valid & start_mask``; this is useful for selecting a
    sensor-observed support cell without consulting its target class.  Candidates
    outside the grid or outside the published label-validity mask are omitted.
    Event labels are computed later by the evaluator.
    """

    valid = np.asarray(target_valid, dtype=bool)
    if valid.shape != GRID_SHAPE:
        raise ValueError(f"target_valid must have shape {GRID_SHAPE}, got {valid.shape}")
    if start_mask is None:
        candidate_mask = valid
    else:
        candidate = np.asarray(start_mask, dtype=bool)
        if candidate.shape != GRID_SHAPE:
            raise ValueError(f"start_mask must have shape {GRID_SHAPE}, got {candidate.shape}")
        candidate_mask = valid & candidate
        # Keep the function total for sparse releases.  The fallback is explicitly
        # observable to callers through the selected start and does not inspect the
        # target-free class.
        if not np.any(candidate_mask):
            candidate_mask = valid
    cells = np.argwhere(candidate_mask)
    if cells.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    hint = np.asarray(anchor_hint, dtype=np.int64)
    start = cells[np.argmin(np.sum((cells - hint) ** 2, axis=1))]
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    for distance in distances_cells:
        if int(distance) < 1:
            raise ValueError("query distances must be positive cell counts")
        for angle in angles_deg:
            theta = math.radians(float(angle))
            goal = np.rint(start + np.array((distance * math.cos(theta), distance * math.sin(theta)))).astype(np.int64)
            row, col = int(goal[0]), int(goal[1])
            if 0 <= row < GRID_SHAPE[0] and 0 <= col < GRID_SHAPE[1] and valid[row, col]:
                starts.append((int(start[0]), int(start[1])))
                goals.append((row, col))
    return np.asarray(starts, dtype=np.int64), np.asarray(goals, dtype=np.int64)


def load_frame(
    timestamp: str,
    *,
    raw_root: Path,
    label_root: Path,
    endpoint_policy: str = "blocked",
    start_selection: str = "valid",
    ground_margin_m: float = 0.35,
    ground_bin_size_m: float = 1.2,
    ground_lateral_limit_m: float = 20.0,
    ground_quantile: float = 0.15,
) -> UnScenesFrame:
    """Load one frame without accessing any held-out split metadata."""

    lidar_path = raw_root / "clouds" / f"{timestamp}.bin"
    occ_path = label_root / "occ" / f"{timestamp}.npy"
    if not lidar_path.exists() or not occ_path.exists():
        raise FileNotFoundError(f"missing raw/occupancy pair for timestamp {timestamp}")
    points = np.fromfile(lidar_path, dtype=np.float32)
    if points.size % 4:
        raise ValueError(f"LiDAR file is not Nx4 float32: {lidar_path}")
    support = occupancy_support(np.load(occ_path, allow_pickle=False))
    if start_selection not in {"valid", "observed_free"}:
        raise ValueError("start_selection must be 'valid' or 'observed_free'")
    input_bev = lidar_observation(
        points.reshape(-1, 4),
        endpoint_policy=endpoint_policy,
        ground_margin_m=ground_margin_m,
        ground_bin_size_m=ground_bin_size_m,
        ground_lateral_limit_m=ground_lateral_limit_m,
        ground_quantile=ground_quantile,
    )
    starts, goals = deterministic_queries(
        support.valid,
        start_mask=(input_bev[0] > 0.5) if start_selection == "observed_free" else None,
    )
    return UnScenesFrame(
        timestamp=timestamp,
        input_bev=input_bev,
        target_free=support.free,
        target_valid=support.valid,
        starts=starts,
        goals=goals,
    )
