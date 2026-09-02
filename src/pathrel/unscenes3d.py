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
VOXEL_SIZE_M = 0.3
Y_ORIGIN_M = -38.4
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


def lidar_observation(points: np.ndarray) -> np.ndarray:
    """Rasterize a label-free LiDAR ray observation into ``[3,256,256]``.

    Channel order matches :class:`PathRelNet`: observed-free ray cells, observed
    blocked returns, and unknown cells.  Every return is treated as a blocked
    endpoint because the raw release has no per-point semantic class; this is a
    conservative sensor-only observation and does not read occupancy labels.
    """

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {array.shape}")
    finite = np.isfinite(array[:, :3]).all(axis=1)
    x, y = array[:, 0], array[:, 1]
    in_range = finite & (x >= 0.0) & (x < GRID_SHAPE[0] * VOXEL_SIZE_M) & (y >= Y_ORIGIN_M) & (y < -Y_ORIGIN_M)
    rows = np.floor(x[in_range] / VOXEL_SIZE_M).astype(np.int64)
    cols = np.floor((y[in_range] - Y_ORIGIN_M) / VOXEL_SIZE_M).astype(np.int64)
    in_grid = (rows >= 0) & (rows < GRID_SHAPE[0]) & (cols >= 0) & (cols < GRID_SHAPE[1])
    rows, cols = rows[in_grid], cols[in_grid]
    endpoints = sorted(set(zip(rows.tolist(), cols.tolist())))
    observed_free = np.zeros(GRID_SHAPE, dtype=bool)
    observed_blocked = np.zeros(GRID_SHAPE, dtype=bool)
    origin = (0, int(round((0.0 - Y_ORIGIN_M) / VOXEL_SIZE_M)))
    for endpoint in endpoints:
        segment = list(_bresenham(origin, endpoint))
        for cell in segment[:-1]:
            observed_free[cell] = True
        observed_blocked[endpoint] = True
    observed_free &= ~observed_blocked
    unknown = ~(observed_free | observed_blocked)
    return np.stack((observed_free, observed_blocked, unknown), axis=0).astype(np.float32)


def deterministic_queries(
    target_valid: np.ndarray,
    *,
    distances_cells: Sequence[int] = (13, 27, 40),
    angles_deg: Sequence[int] = tuple(range(0, 360, 30)),
    anchor_hint: tuple[int, int] = (96, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed polar stencil using bounds/validity only.

    The target-free class is not inspected.  The anchor is the nearest valid cell
    to ``anchor_hint``; candidates outside the grid or outside the published label
    validity mask are omitted.  Event labels are computed later by the evaluator.
    """

    valid = np.asarray(target_valid, dtype=bool)
    if valid.shape != GRID_SHAPE:
        raise ValueError(f"target_valid must have shape {GRID_SHAPE}, got {valid.shape}")
    cells = np.argwhere(valid)
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
    starts, goals = deterministic_queries(support.valid)
    return UnScenesFrame(
        timestamp=timestamp,
        input_bev=lidar_observation(points.reshape(-1, 4)),
        target_free=support.free,
        target_valid=support.valid,
        starts=starts,
        goals=goals,
    )
