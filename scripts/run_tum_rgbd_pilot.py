#!/usr/bin/env python3
"""Run the reproducible real-data pilot on the TUM RGB-D ``freiburg1/desk`` sequence.

This adapter is deliberately small and dependency-light.  It consumes the official TUM RGB-D
RGB/depth stream and its motion-capture trajectory, lifts sampled depth pixels into a world frame,
and rasterises a horizontal support/obstacle reference map.  The event audit compares two
completion distributions on *future-frame* queries:

``independent_cell``
    independent Bernoulli cells with the same marginal evidence;
``correlated_temporal``
    a spatially correlated random field, clamped by observed support/obstacle evidence.

The resulting labels are geometric reference-map labels, not robot traversability annotations.  The
script therefore calls the output a ``real-data pilot`` and writes the claim boundary into the
report.  It is intended to make the website and the data path honest and reproducible while the
full public navigation benchmark is still being built.

The raw archive is never copied into Git.  Only compact derived figures/video frames are optionally
published to ``site/assets``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tum_rgbd_freiburg1_desk"
    / "extracted"
    / "rgbd_dataset_freiburg1_desk"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tum_rgbd_freiburg1_desk_pilot"
DEFAULT_SITE = PROJECT_ROOT / "site"

# Official Freiburg-1 RGB camera model from the TUM RGB-D file-format documentation.  The
# sequence stores registered RGB/depth images at 640x480 and depth values in millimetres / 5000.
FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
DEPTH_SCALE = 5000.0


@dataclass(frozen=True)
class FrameRecord:
    index: int
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose: np.ndarray  # tx, ty, tz, qx, qy, qz, qw


@dataclass
class FrameEvidence:
    record: FrameRecord
    rgb: np.ndarray
    depth: np.ndarray
    world_points: np.ndarray
    support_cells: np.ndarray
    obstacle_cells: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--frames", type=int, default=48, help="uniformly sampled RGB frames")
    parser.add_argument("--depth-stride", type=int, default=8, help="pixel stride for point lifting")
    parser.add_argument("--grid-resolution", type=float, default=0.04, help="BEV cell size in metres")
    parser.add_argument("--queries", type=int, default=18, help="query pairs per held-out frame")
    parser.add_argument("--samples", type=int, default=48, help="Monte-Carlo worlds per event")
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--publish-site", action="store_true", help="copy compact assets into site/")
    parser.add_argument("--no-video", action="store_true", help="skip FFmpeg video generation")
    return parser.parse_args()


def read_assoc(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), fields[1]))
    return rows


def read_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 8:
            rows.append((float(fields[0]), np.asarray([float(value) for value in fields[1:8]], dtype=np.float64)))
    return rows


def nearest_index(values: np.ndarray, target: float) -> tuple[int, float]:
    position = int(np.searchsorted(values, target))
    candidates = [max(0, min(len(values) - 1, position + delta)) for delta in (-1, 0)]
    index = min(candidates, key=lambda item: abs(float(values[item]) - target))
    return index, abs(float(values[index]) - target)


def associate_frames(sequence_dir: Path, requested: int) -> list[FrameRecord]:
    rgb_rows = read_assoc(sequence_dir / "rgb.txt")
    depth_rows = read_assoc(sequence_dir / "depth.txt")
    groundtruth = read_groundtruth(sequence_dir / "groundtruth.txt")
    if not rgb_rows or not depth_rows or not groundtruth:
        raise SystemExit(f"TUM sequence is incomplete: {sequence_dir}")
    depth_times = np.asarray([row[0] for row in depth_rows], dtype=np.float64)
    pose_times = np.asarray([row[0] for row in groundtruth], dtype=np.float64)
    candidates: list[FrameRecord] = []
    for index, (timestamp, rgb_name) in enumerate(rgb_rows):
        depth_index, depth_delta = nearest_index(depth_times, timestamp)
        pose_index, pose_delta = nearest_index(pose_times, timestamp)
        # TUM's RGB/depth streams are asynchronous; these conservative gates reject malformed
        # associations without discarding the sequence when the streams drift by a few ms.
        if depth_delta > 0.06 or pose_delta > 0.06:
            continue
        rgb_path = sequence_dir / rgb_name
        depth_path = sequence_dir / depth_rows[depth_index][1]
        if rgb_path.exists() and depth_path.exists():
            candidates.append(
                FrameRecord(index=index, timestamp=timestamp, rgb_path=rgb_path, depth_path=depth_path, pose=groundtruth[pose_index][1])
            )
    if len(candidates) < 8:
        raise SystemExit(f"Only {len(candidates)} synchronised RGB-D frames found")
    count = max(8, min(int(requested), len(candidates)))
    selected_indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return [candidates[int(index)] for index in selected_indices]


def quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Return the TUM camera-to-world rotation for qx,qy,qz,qw."""

    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def lift_world_points(record: FrameRecord, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(record.rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(record.depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or depth.ndim != 2:
        raise RuntimeError(f"Unable to read RGB/depth for {record.rgb_path}")
    height, width = depth.shape
    stride = max(2, int(stride))
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    pixel_y, pixel_x = np.meshgrid(rows, cols, indexing="ij")
    depth_m = depth[pixel_y, pixel_x].astype(np.float64) / DEPTH_SCALE
    valid = np.isfinite(depth_m) & (depth_m >= 0.25) & (depth_m <= 4.0)
    pixel_x = pixel_x[valid].astype(np.float64)
    pixel_y = pixel_y[valid].astype(np.float64)
    depth_m = depth_m[valid]
    camera_points = np.column_stack(
        ((pixel_x - CX) * depth_m / FX, (pixel_y - CY) * depth_m / FY, depth_m)
    )
    rotation = quaternion_matrix(record.pose[3:])
    world_points = camera_points @ rotation.T + record.pose[:3]
    return rgb, depth, world_points


def estimate_support_height(points: np.ndarray) -> float:
    z = points[:, 2]
    valid = z[(z >= 0.2) & (z <= 1.35)]
    if valid.size == 0:
        return float(np.median(z))
    histogram, edges = np.histogram(valid, bins=115, range=(0.2, 1.35))
    peak = int(np.argmax(histogram))
    return float((edges[peak] + edges[peak + 1]) * 0.5)


def grid_coordinates(points: np.ndarray, bounds: tuple[float, float, float, float], resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = bounds
    cols = np.floor((points[:, 0] - xmin) / resolution).astype(np.int32)
    rows = np.floor((ymax - points[:, 1]) / resolution).astype(np.int32)
    return rows, cols, (rows >= 0) & (cols >= 0)


def unique_in_bounds(rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> np.ndarray:
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)
    return np.unique(rows[valid].astype(np.int64) * width + cols[valid].astype(np.int64))


def build_grid(evidence: list[FrameEvidence], resolution: float, support_height: float) -> dict[str, object]:
    all_points = np.concatenate([item.world_points for item in evidence], axis=0)
    poses = np.asarray([item.record.pose[:3] for item in evidence], dtype=np.float64)
    # Robust point bounds plus the camera trajectory, so the trajectory overlay is never clipped.
    point_xy = all_points[:, :2]
    low = np.percentile(point_xy, 0.5, axis=0)
    high = np.percentile(point_xy, 99.5, axis=0)
    low = np.minimum(low, poses[:, :2].min(axis=0)) - 0.12
    high = np.maximum(high, poses[:, :2].max(axis=0)) + 0.12
    width = int(math.ceil((high[0] - low[0]) / resolution))
    height = int(math.ceil((high[1] - low[1]) / resolution))
    # Keep raster dimensions bounded if a corrupted depth file contains an extreme outlier.
    if width > 220 or height > 220:
        raise RuntimeError(f"Unexpected BEV extent {width}x{height}; inspect depth bounds")
    bounds = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
    support_counts = np.zeros((height, width), dtype=np.int16)
    obstacle_counts = np.zeros((height, width), dtype=np.int16)
    frame_cells: list[tuple[np.ndarray, np.ndarray]] = []
    for item in evidence:
        z = item.world_points[:, 2]
        support = item.world_points[np.abs(z - support_height) <= 0.09]
        obstacle = item.world_points[(z > support_height + 0.10) & (z < support_height + 1.10)]
        sr, sc, _ = grid_coordinates(support, bounds, resolution)
        orows, ocols, _ = grid_coordinates(obstacle, bounds, resolution)
        support_cells = unique_in_bounds(sr, sc, height, width)
        obstacle_cells = unique_in_bounds(orows, ocols, height, width)
        if support_cells.size:
            np.add.at(support_counts, (support_cells // width, support_cells % width), 1)
        if obstacle_cells.size:
            np.add.at(obstacle_counts, (obstacle_cells // width, obstacle_cells % width), 1)
        item.support_cells = support_cells
        item.obstacle_cells = obstacle_cells
        frame_cells.append((support_cells, obstacle_cells))
    # A cell seen repeatedly at support height is the reference support surface.  Obstacles are
    # intentionally conservative: two independent frames must agree before a cell is blocked.
    support_ref = support_counts >= 2
    obstacle_ref = obstacle_counts >= 2
    free_ref = support_ref & ~obstacle_ref
    # Remove isolated one-pixel speckles while preserving the measured geometry.  This is a raster
    # regularisation step, not a learned label and is recorded in report.json.
    support_ref = cv2.morphologyEx(support_ref.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    obstacle_ref = cv2.dilate(obstacle_ref.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    free_ref = support_ref & ~obstacle_ref
    return {
        "bounds": bounds,
        "height": height,
        "width": width,
        "support_counts": support_counts,
        "obstacle_counts": obstacle_counts,
        "support_ref": support_ref,
        "obstacle_ref": obstacle_ref,
        "free_ref": free_ref,
        "frame_cells": frame_cells,
    }


def disk_kernel(radius: int) -> np.ndarray:
    radius = int(radius)
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)


def erode_free(free: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return free.copy()
    return cv2.erode(free.astype(np.uint8), disk_kernel(radius), borderType=cv2.BORDER_CONSTANT).astype(bool)


def path_exists(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int], radius: int = 0) -> bool:
    usable = erode_free(free, radius)
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < usable.shape[0] and 0 <= sc < usable.shape[1] and 0 <= gr < usable.shape[0] and 0 <= gc < usable.shape[1]):
        return False
    if not usable[sr, sc] or not usable[gr, gc]:
        return False
    queue = [(sr, sc)]
    visited = np.zeros_like(usable, dtype=bool)
    visited[sr, sc] = True
    cursor = 0
    while cursor < len(queue):
        row, col = queue[cursor]
        cursor += 1
        if (row, col) == (gr, gc):
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < usable.shape[0] and 0 <= nc < usable.shape[1] and usable[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False


def max_bottleneck_score(probability: np.ndarray, start: tuple[int, int], goal: tuple[int, int], radius: int) -> float:
    """Max-min path score on a probability map (a marginal-cell baseline)."""

    values = erode_free(probability >= 0.5, radius).astype(np.float64) * probability
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < values.shape[0] and 0 <= sc < values.shape[1] and 0 <= gr < values.shape[0] and 0 <= gc < values.shape[1]):
        return 0.0
    best = np.zeros_like(values, dtype=np.float64)
    best[sr, sc] = float(values[sr, sc])
    pending: list[tuple[float, int, int]] = [(-best[sr, sc], sr, sc)]
    while pending:
        negative, row, col = pending.pop(0)
        score = -negative
        if score + 1e-12 < best[row, col]:
            continue
        if (row, col) == (gr, gc):
            return float(score)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < values.shape[0] and 0 <= nc < values.shape[1]):
                continue
            candidate = min(score, float(values[nr, nc]))
            if candidate > best[nr, nc] + 1e-12:
                best[nr, nc] = candidate
                pending.append((-candidate, nr, nc))
        pending.sort()
    return 0.0


def choose_queries(free_ref: np.ndarray, count: int, seed: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    candidates = np.argwhere(free_ref)
    if len(candidates) < 12:
        raise RuntimeError("Reference support map has too few free cells for path queries")
    queries: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # Prefer well-separated cells.  We retain both connected and disconnected pairs so the event
    # labels are not trivially all positive.
    attempts = 0
    while len(queries) < count and attempts < count * 300:
        attempts += 1
        first = candidates[int(rng.integers(len(candidates)))]
        second = candidates[int(rng.integers(len(candidates)))]
        if np.linalg.norm(first - second) < max(12.0, min(free_ref.shape) * 0.16):
            continue
        pair = ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])))
        if pair not in queries and (pair[1], pair[0]) not in queries:
            queries.append(pair)
    if len(queries) < max(4, count // 2):
        # Deterministic fallback, still derived from measured support cells.
        for first_index in range(0, len(candidates), max(1, len(candidates) // count)):
            first = candidates[first_index]
            second = candidates[(first_index + len(candidates) // 2) % len(candidates)]
            pair = ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])))
            if pair not in queries and (pair[1], pair[0]) not in queries:
                queries.append(pair)
            if len(queries) >= count:
                break
    return queries[:count]


def evidence_probability(support_counts: np.ndarray, obstacle_counts: np.ndarray, frames_seen: int) -> np.ndarray:
    """Posterior-like free probability from prefix evidence, with no future labels."""

    support = support_counts.astype(np.float64)
    obstacle = obstacle_counts.astype(np.float64)
    # A weak empirical prior from the observed prefix only.  Unknown cells remain uncertain rather
    # than silently being declared free.
    support_cells = int(np.count_nonzero(support > 0))
    obstacle_cells = int(np.count_nonzero(obstacle > 0))
    prior = float(np.clip((support_cells + 2.0) / (support_cells + obstacle_cells + 4.0), 0.20, 0.80))
    probability = np.full(support.shape, prior, dtype=np.float64)
    seen_support = support > 0
    seen_obstacle = obstacle > 0
    probability[seen_support] = 0.88 + 0.08 * (1.0 - np.exp(-support[seen_support] / max(1.0, frames_seen * 0.12)))
    probability[seen_obstacle] = 0.08 + 0.08 * np.exp(-obstacle[seen_obstacle] / max(1.0, frames_seen * 0.12))
    both = seen_support & seen_obstacle
    probability[both] = 0.42
    return np.clip(probability, 0.02, 0.98)


def sample_worlds(probability: np.ndarray, support_counts: np.ndarray, obstacle_counts: np.ndarray, samples: int, correlated: bool, rng: np.random.Generator) -> np.ndarray:
    height, width = probability.shape
    worlds = np.empty((samples, height, width), dtype=bool)
    known_support = support_counts > 0
    known_obstacle = obstacle_counts > 0
    unknown = ~(known_support | known_obstacle)
    for sample_index in range(samples):
        if correlated:
            noise = rng.standard_normal((height, width), dtype=np.float32)
            # cv2's Gaussian filter gives a deterministic, spatially correlated latent field.
            smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.2, sigmaY=2.2)
            smooth = (smooth - float(smooth.mean())) / max(float(smooth.std()), 1e-6)
            logits = np.log(probability / (1.0 - probability)) + 0.75 * smooth
            world = rng.random((height, width)) < (1.0 / (1.0 + np.exp(-logits)))
        else:
            world = rng.random((height, width)) < probability
        world[known_support] = True
        world[known_obstacle] = False
        world[~unknown & ~known_support] = False
        worlds[sample_index] = world
    return worlds


def event_probability(worlds: np.ndarray, start: tuple[int, int], goal: tuple[int, int], radius: int) -> float:
    if worlds.size == 0:
        return 0.0
    successes = sum(path_exists(world, start, goal, radius) for world in worlds)
    return float(successes / worlds.shape[0])


def world_event_matrix(
    worlds: np.ndarray,
    queries: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    radii: Sequence[int],
) -> np.ndarray:
    """Evaluate all query/radius events while reusing one connected-components pass per world.

    Calling a fresh BFS for every query is needlessly expensive for the pilot (the raster is the
    same for all terminals).  Connected components have exactly the same four-neighbour event
    semantics here and make the real-data run comfortably reproducible on a CPU.
    """

    result = np.zeros((worlds.shape[0], len(queries), len(radii)), dtype=np.float32)
    for sample_index, world in enumerate(worlds):
        for radius_index, radius in enumerate(radii):
            usable = erode_free(world, int(radius))
            number, labels = cv2.connectedComponents(usable.astype(np.uint8), connectivity=4)
            del number
            for query_index, (start, goal) in enumerate(queries):
                sr, sc = start
                gr, gc = goal
                if 0 <= sr < usable.shape[0] and 0 <= sc < usable.shape[1] and 0 <= gr < usable.shape[0] and 0 <= gc < usable.shape[1]:
                    result[sample_index, query_index, radius_index] = float(labels[sr, sc] > 0 and labels[sr, sc] == labels[gr, gc])
    return result


def brier(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((values - labels) ** 2)) if values.size else float("nan")


def nll(values: np.ndarray, labels: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def ece(values: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if not values.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (values >= edges[index]) & ((values < edges[index + 1]) if index < bins - 1 else (values <= edges[index + 1]))
        if np.any(mask):
            error += float(mask.mean()) * abs(float(values[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def metric_rows(event_records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ("observed_prefix", "independent_cell", "correlated_temporal"):
        subset = [row for row in event_records if row["method"] == method]
        values = np.asarray([float(row["probability"]) for row in subset], dtype=np.float64)
        labels = np.asarray([float(row["label"]) for row in subset], dtype=np.float64)
        rows.append({
            "id": method,
            "name": {"observed_prefix": "Observed prefix (deterministic)", "independent_cell": "Independent-cell completion", "correlated_temporal": "Correlated temporal completion"}[method],
            "brier": brier(values, labels),
            "nll": nll(values, labels),
            "ece": ece(values, labels),
            "false_safe_rate@0.8": float(np.mean((values >= 0.8) & (labels < 0.5))) if values.size else float("nan"),
            "count": int(values.size),
        })
    return rows


def colorize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    valid = depth > 0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth.astype(np.float32) / DEPTH_SCALE, 0.25, 3.0)
        normalized = np.where(valid, ((1.0 - (clipped - 0.25) / 2.75) * 255.0).clip(0, 255), 0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = (35, 35, 35)
    return cv2.resize(color, size, interpolation=cv2.INTER_AREA)


def put_text(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.65, color: tuple[int, int, int] = (30, 35, 42), thickness: int = 1) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_panel(item: FrameEvidence, grid: dict[str, object], prefix_support: np.ndarray, prefix_obstacle: np.ndarray, support_height: float, frame_number: int, total: int, query: tuple[tuple[int, int], tuple[int, int]] | None = None, output_size: tuple[int, int] = (1440, 820)) -> np.ndarray:
    width, height = output_size
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    put_text(canvas, "ConPath  |  real-data pilot", (42, 48), 0.95, (20, 28, 38), 2)
    put_text(canvas, "TUM RGB-D Freiburg1/desk - RGB + registered depth + MoCap pose", (42, 78), 0.56, (92, 101, 112), 1)
    put_text(canvas, f"frame {frame_number + 1:02d}/{total:02d}   t={item.record.timestamp:.3f}s", (width - 380, 52), 0.48, (92, 101, 112), 1)

    margin = 42
    top = 112
    gap = 22
    panel_w = (width - 2 * margin - 2 * gap) // 3
    panel_h = 470
    panel_titles = ("RGB observation", "Registered depth", "World-frame BEV")
    for index, title in enumerate(panel_titles):
        left = margin + index * (panel_w + gap)
        cv2.rectangle(canvas, (left, top), (left + panel_w, top + panel_h), (224, 228, 232), 1)
        put_text(canvas, title, (left + 16, top + 28), 0.62, (35, 43, 52), 1)

    image_size = (panel_w - 28, panel_h - 58)
    rgb = cv2.resize(item.rgb, image_size, interpolation=cv2.INTER_AREA)
    depth = colorize_depth(item.depth, image_size)
    canvas[top + 44 : top + 44 + image_size[1], margin + 14 : margin + 14 + image_size[0]] = rgb
    canvas[top + 44 : top + 44 + image_size[1], margin + panel_w + gap + 14 : margin + panel_w + gap + 14 + image_size[0]] = depth

    # Draw the measured BEV as a compact point raster.  Support is warm/green, obstacles are blue;
    # the trajectory and current camera pose are overlaid in the same world coordinates.
    bev_left = margin + 2 * (panel_w + gap)
    bev = np.full((image_size[1], image_size[0], 3), (250, 250, 250), dtype=np.uint8)
    bounds = grid["bounds"]
    xmin, ymin, xmax, ymax = [float(value) for value in bounds]  # type: ignore[arg-type]
    support = np.asarray(grid["support_ref"], dtype=bool)
    obstacle = np.asarray(grid["obstacle_ref"], dtype=bool)
    support_prefix = prefix_support > 0
    obstacle_prefix = prefix_obstacle > 0
    map_h, map_w = support.shape
    yy, xx = np.where(support)
    px = ((xx / max(1, map_w - 1)) * (image_size[0] - 1)).astype(int)
    py = ((yy / max(1, map_h - 1)) * (image_size[1] - 1)).astype(int)
    for point_x, point_y in zip(px, py):
        cv2.circle(bev, (int(point_x), int(point_y)), 2, (126, 180, 112), -1, cv2.LINE_AA)
    yy, xx = np.where(obstacle)
    px = ((xx / max(1, map_w - 1)) * (image_size[0] - 1)).astype(int)
    py = ((yy / max(1, map_h - 1)) * (image_size[1] - 1)).astype(int)
    for point_x, point_y in zip(px, py):
        cv2.circle(bev, (int(point_x), int(point_y)), 2, (77, 110, 194), -1, cv2.LINE_AA)
    # Unknown prefix evidence is shown as faint amber cells, making partial observation explicit.
    unknown_prefix = ~(support_prefix | obstacle_prefix)
    yy, xx = np.where(unknown_prefix & support)
    px = ((xx / max(1, map_w - 1)) * (image_size[0] - 1)).astype(int)
    py = ((yy / max(1, map_h - 1)) * (image_size[1] - 1)).astype(int)
    for point_x, point_y in zip(px, py):
        cv2.circle(bev, (int(point_x), int(point_y)), 2, (194, 170, 94), -1, cv2.LINE_AA)
    poses = np.asarray([entry.record.pose[:3] for entry in grid["evidence"]], dtype=np.float64) if "evidence" in grid else np.empty((0, 3))
    if len(poses):
        tx = ((poses[:, 0] - xmin) / max(1e-6, xmax - xmin) * (image_size[0] - 1)).astype(int)
        ty = ((ymax - poses[:, 1]) / max(1e-6, ymax - ymin) * (image_size[1] - 1)).astype(int)
        valid = (tx >= 0) & (tx < image_size[0]) & (ty >= 0) & (ty < image_size[1])
        points = list(zip(tx[valid], ty[valid]))
        if len(points) > 1:
            cv2.polylines(bev, [np.asarray(points, dtype=np.int32)], False, (50, 50, 50), 2, cv2.LINE_AA)
    current = item.record.pose
    cx = int((current[0] - xmin) / max(1e-6, xmax - xmin) * (image_size[0] - 1))
    cy = int((ymax - current[1]) / max(1e-6, ymax - ymin) * (image_size[1] - 1))
    if 0 <= cx < image_size[0] and 0 <= cy < image_size[1]:
        cv2.drawMarker(bev, (cx, cy), (20, 34, 42), cv2.MARKER_TRIANGLE_UP, 18, 2)
    if query is not None:
        (sr, sc), (gr, gc) = query
        sx = int(sc / max(1, map_w - 1) * (image_size[0] - 1)); sy = int(sr / max(1, map_h - 1) * (image_size[1] - 1))
        gx = int(gc / max(1, map_w - 1) * (image_size[0] - 1)); gy = int(gr / max(1, map_h - 1) * (image_size[1] - 1))
        cv2.line(bev, (sx, sy), (gx, gy), (47, 158, 114), 2, cv2.LINE_AA)
        cv2.circle(bev, (sx, sy), 7, (47, 158, 114), -1); cv2.circle(bev, (gx, gy), 7, (33, 100, 176), -1)
    canvas[top + 44 : top + 44 + image_size[1], bev_left + 14 : bev_left + 14 + image_size[0]] = bev

    base_y = top + panel_h + 42
    put_text(canvas, "Measured geometry", (margin, base_y), 0.62, (35, 43, 52), 1)
    put_text(canvas, f"support plane z~{support_height:.3f} m", (margin, base_y + 30), 0.55, (92, 101, 112), 1)
    put_text(canvas, "green: reference support | blue: elevated returns | amber: not yet observed", (margin, base_y + 58), 0.52, (92, 101, 112), 1)
    put_text(canvas, "GT trajectory", (520, base_y), 0.62, (35, 43, 52), 1)
    put_text(canvas, "black line = MoCap pose; triangle = current frame", (520, base_y + 30), 0.48, (92, 101, 112), 1)
    put_text(canvas, "claim boundary", (1000, base_y), 0.62, (153, 93, 36), 1)
    put_text(canvas, "geometric pilot; no navigation GT", (1000, base_y + 30), 0.48, (153, 93, 36), 1)
    cv2.line(canvas, (margin, height - 38), (width - margin, height - 38), (220, 224, 228), 1)
    put_text(canvas, "ConPath | real RGB-D evidence -> world-frame support map -> event audit", (margin, height - 14), 0.48, (115, 122, 130), 1)
    return canvas


def find_ffmpeg() -> Path | None:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    candidates = list(Path("/home/hairo/.local/share/pnpm/store").glob("**/node_modules/@ffmpeg-installer/linux-x64/ffmpeg"))
    return next((path for path in candidates if path.is_file() and path.stat().st_mode & 0o111), None)


def write_svg_comparison(rows: list[dict[str, object]], output: Path) -> None:
    names = [str(row["name"]).replace("&", "&amp;") for row in rows]
    briers = [float(row["brier"]) for row in rows]
    eces = [float(row["ece"]) for row in rows]
    width, height = 980, 430
    left, right, top, bottom = 210, 930, 62, 365
    max_value = max(0.05, max(briers + eces) * 1.20)
    colors = ["#8c99a8", "#5d84b2", "#2f9b78"]
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Real-data pilot event metrics">', '<rect width="100%" height="100%" fill="#ffffff"/>', '<style>text{font-family:Arial,sans-serif;fill:#28333d} .muted{fill:#687582;font-size:13px} .axis{stroke:#dce2e7;stroke-width:1}</style>', '<text x="32" y="34" font-size="20" font-weight="700">TUM RGB-D reference-map event audit</text>', '<text x="32" y="54" class="muted">lower is better · held-out future-frame queries · pilot scope</text>']
    for tick in range(6):
        value = max_value * tick / 5
        x = left + (right - left) * value / max_value
        chunks.append(f'<line class="axis" x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{bottom + 12}"/>')
        chunks.append(f'<text class="muted" x="{x:.1f}" y="{bottom + 32}" text-anchor="middle">{value:.2f}</text>')
    for index, (name, brier_value, ece_value) in enumerate(zip(names, briers, eces)):
        y = top + index * 92
        chunks.append(f'<text x="{left - 14}" y="{y + 22}" text-anchor="end" font-size="14">{name}</text>')
        for offset, value, label in ((0, brier_value, "Brier"), (28, ece_value, "ECE")):
            bar_width = (right - left) * value / max_value
            chunks.append(f'<rect x="{left}" y="{y + offset}" width="{bar_width:.1f}" height="18" rx="6" fill="{colors[index]}" opacity="{1.0 if offset == 0 else 0.57}"/>')
            chunks.append(f'<text x="{min(right - 2, left + bar_width + 8):.1f}" y="{y + offset + 14}" font-size="12">{label} {value:.3f}</text>')
    chunks.append('<text class="muted" x="32" y="407">Correlated temporal completion is a measured geometric proxy; it is not a navigation benchmark claim.</text></svg>')
    output.write_text("\n".join(chunks), encoding="utf-8")


def write_svg_reliability(event_records: list[dict[str, object]], output: Path) -> None:
    width, height = 760, 470
    left, right, top, bottom = 82, 710, 58, 385
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Real-data pilot reliability diagram">', '<rect width="100%" height="100%" fill="#ffffff"/>', '<style>text{font-family:Arial,sans-serif;fill:#28333d}.muted{fill:#687582;font-size:13px}.axis{stroke:#dce2e7;stroke-width:1}</style>', '<text x="32" y="32" font-size="20" font-weight="700">Reliability on future-frame events</text>', '<text class="muted" x="32" y="50">confidence bins · labels from the fused geometric reference map</text>']
    for tick in range(6):
        value = tick / 5
        x = left + (right - left) * value
        y = bottom - (bottom - top) * value
        chunks.append(f'<line class="axis" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/><line class="axis" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        chunks.append(f'<text class="muted" x="{x:.1f}" y="{bottom + 24}" text-anchor="middle">{value:.1f}</text>')
        chunks.append(f'<text class="muted" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    chunks.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#9aa7b3" stroke-dasharray="6 5"/>')
    colors = {"observed_prefix": "#8c99a8", "independent_cell": "#5d84b2", "correlated_temporal": "#2f9b78"}
    labels = {"observed_prefix": "observed prefix", "independent_cell": "independent", "correlated_temporal": "correlated"}
    for method in colors:
        subset = [row for row in event_records if row["method"] == method]
        values = np.asarray([float(row["probability"]) for row in subset])
        truth = np.asarray([float(row["label"]) for row in subset])
        points: list[tuple[float, float]] = []
        for index in range(10):
            lo, hi = index / 10, (index + 1) / 10
            mask = (values >= lo) & ((values < hi) if index < 9 else (values <= hi))
            if np.any(mask):
                points.append((float(values[mask].mean()), float(truth[mask].mean())))
        if points:
            coords = " ".join(f"{left + x * (right - left):.1f},{bottom - y * (bottom - top):.1f}" for x, y in points)
            chunks.append(f'<polyline points="{coords}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
            for x, y in points:
                chunks.append(f'<circle cx="{left + x * (right - left):.1f}" cy="{bottom - y * (bottom - top):.1f}" r="5" fill="{colors[method]}"/>')
        chunks.append(f'<text x="{right - 8}" y="{top + 20 + list(colors).index(method) * 20}" text-anchor="end" fill="{colors[method]}" font-size="13">{labels[method]}</text>')
    chunks.append(f'<text class="muted" x="{(left + right) / 2:.1f}" y="{height - 20}" text-anchor="middle">predicted event probability</text><text class="muted" transform="translate(19 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">empirical frequency</text></svg>')
    output.write_text("\n".join(chunks), encoding="utf-8")


def write_metrics_csv(rows: list[dict[str, object]], output: Path) -> None:
    fields = ["id", "name", "brier", "nll", "ece", "false_safe_rate@0.8", "count"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def publish_assets(output_dir: Path, site_dir: Path, report: dict[str, object]) -> None:
    assets = site_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("tum_freiburg1_desk_teaser.jpg", "tum_freiburg1_desk_rgb.jpg", "tum_freiburg1_desk_depth.jpg", "tum_freiburg1_desk_comparison.svg", "tum_freiburg1_desk_reliability.svg", "tum_freiburg1_desk_demo.mp4", "tum_freiburg1_desk_demo_poster.jpg"):
        source = output_dir / name
        if source.exists():
            shutil.copyfile(source, assets / name)
    (site_dir / "data").mkdir(parents=True, exist_ok=True)
    compact = {
        "schema_version": 1,
        "project": "ConPath",
        "dataset": report["dataset"],
        "protocol": report["protocol"],
        "map": report["map"],
        "metrics": report["metrics"],
        "claim_boundary": report["claim_boundary"],
        "assets": {
            "teaser": "assets/tum_freiburg1_desk_teaser.jpg",
            "rgb": "assets/tum_freiburg1_desk_rgb.jpg",
            "depth": "assets/tum_freiburg1_desk_depth.jpg",
            "video": "assets/tum_freiburg1_desk_demo.mp4",
            "poster": "assets/tum_freiburg1_desk_demo_poster.jpg",
            "comparison": "assets/tum_freiburg1_desk_comparison.svg",
            "reliability": "assets/tum_freiburg1_desk_reliability.svg",
        },
    }
    (site_dir / "data" / "tum_rgbd_pilot.json").write_text(json.dumps(json_safe(compact), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (site_dir / "data" / "tum_rgbd_pilot.js").write_text("window.CONPATH_REAL_PILOT = " + json.dumps(json_safe(compact), ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    sequence_dir = args.sequence_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = associate_frames(sequence_dir, args.frames)
    raw_evidence: list[FrameEvidence] = []
    for record in records:
        rgb, depth, points = lift_world_points(record, args.depth_stride)
        raw_evidence.append(FrameEvidence(record=record, rgb=rgb, depth=depth, world_points=points, support_cells=np.empty(0, dtype=np.int64), obstacle_cells=np.empty(0, dtype=np.int64)))
    all_points = np.concatenate([item.world_points for item in raw_evidence], axis=0)
    support_height = estimate_support_height(all_points)
    grid = build_grid(raw_evidence, args.grid_resolution, support_height)
    grid["evidence"] = raw_evidence
    height, width = int(grid["height"]), int(grid["width"])
    free_ref = np.asarray(grid["free_ref"], dtype=bool)
    queries = choose_queries(free_ref, max(4, args.queries), args.seed)
    # A temporal split: the first 75% is the observed prefix, the final 25% provides future-frame
    # queries.  Every prediction sees only evidence available before its query frame.
    split = max(2, int(len(raw_evidence) * 0.75))
    support_prefix = np.zeros((height, width), dtype=np.int16)
    obstacle_prefix = np.zeros((height, width), dtype=np.int16)
    event_records: list[dict[str, object]] = []
    radii = (0, 1, 2)
    future_indices = list(range(split, len(raw_evidence)))
    rng = np.random.default_rng(args.seed)
    reference_labels: dict[tuple[int, int, int], int] = {}
    for query_index, (start, goal) in enumerate(queries):
        for radius in radii:
            reference_labels[(query_index, 0, radius)] = int(path_exists(free_ref, start, goal, radius))
    # Prefix evidence is accumulated through the frame immediately preceding each held-out frame.
    for frame_index in range(len(raw_evidence)):
        if frame_index in future_indices:
            probability = evidence_probability(support_prefix, obstacle_prefix, max(1, frame_index))
            independent_worlds = sample_worlds(probability, support_prefix, obstacle_prefix, args.samples, False, rng)
            correlated_worlds = sample_worlds(probability, support_prefix, obstacle_prefix, args.samples, True, rng)
            independent_events = world_event_matrix(independent_worlds, queries, radii).mean(axis=0)
            correlated_events = world_event_matrix(correlated_worlds, queries, radii).mean(axis=0)
            for query_index, (start, goal) in enumerate(queries):
                for radius in radii:
                    label = float(reference_labels[(query_index, 0, radius)])
                    observed_event = float(path_exists((support_prefix > 0) & ~(obstacle_prefix > 0), start, goal, radius))
                    marginal_event = max_bottleneck_score(probability, start, goal, radius)
                    radius_index = radii.index(radius)
                    independent_event = float(independent_events[query_index, radius_index])
                    correlated_event = float(correlated_events[query_index, radius_index])
                    common = {"frame_index": frame_index, "query_index": query_index, "radius": radius, "label": label}
                    event_records.extend([
                        {**common, "method": "observed_prefix", "probability": observed_event},
                        {**common, "method": "independent_cell", "probability": 0.5 * independent_event + 0.5 * marginal_event},
                        {**common, "method": "correlated_temporal", "probability": correlated_event},
                    ])
        for cells, counts in (grid["frame_cells"][frame_index][0], support_prefix), (grid["frame_cells"][frame_index][1], obstacle_prefix):  # type: ignore[index]
            if len(cells):
                np.add.at(counts, (cells // width, cells % width), 1)

    rows = metric_rows(event_records)
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "TUM RGB-D Freiburg1/desk",
            "sequence": "rgbd_dataset_freiburg1_desk",
            "source_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
            "download_url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
            "license": "CC BY 4.0 (TUM RGB-D dataset terms)",
            "rgb_frames_available": len(read_assoc(sequence_dir / "rgb.txt")),
            "depth_frames_available": len(read_assoc(sequence_dir / "depth.txt")),
            "groundtruth_poses_available": len(read_groundtruth(sequence_dir / "groundtruth.txt")),
            "sampled_frames": len(raw_evidence),
            "synchronised_frames": len(records),
        },
        "protocol": {
            "type": "real-data reference-map pilot",
            "sampled_rgb_frames": len(raw_evidence),
            "temporal_split": {"observed_prefix_frames": split, "future_query_frames": len(future_indices)},
            "depth_stride_pixels": int(args.depth_stride),
            "camera_intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "depth_scale": DEPTH_SCALE},
            "raster": {"resolution_m": args.grid_resolution, "support_height_m": support_height, "support_tolerance_m": 0.09, "obstacle_height_m": 0.10, "support_repeat_threshold": 2, "obstacle_repeat_threshold": 2, "morphology": "3x3 close support / 3x3 obstacle dilation"},
            "queries": len(queries),
            "radii_cells": list(radii),
            "monte_carlo_samples": int(args.samples),
            "seed": int(args.seed),
        },
        "map": {"height_cells": height, "width_cells": width, "bounds_m": list(grid["bounds"]), "support_cells": int(np.count_nonzero(grid["support_ref"])), "obstacle_cells": int(np.count_nonzero(grid["obstacle_ref"])), "free_cells": int(np.count_nonzero(free_ref))},
        "metrics": rows,
        "claim_boundary": "This is a real RGB-D geometric reference-map pilot. TUM provides RGB/depth and camera motion, not traversability or collision labels; metrics must not be presented as a public navigation benchmark or as a validated neural P0 result.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_metrics_csv(rows, output_dir / "metrics.csv")
    write_svg_comparison(rows, output_dir / "tum_freiburg1_desk_comparison.svg")
    write_svg_reliability(event_records, output_dir / "tum_freiburg1_desk_reliability.svg")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Render a compact set of real-data frames for inspection and the website video.
    prefix_support = np.zeros((height, width), dtype=np.int16)
    prefix_obstacle = np.zeros((height, width), dtype=np.int16)
    render_indices = list(range(len(raw_evidence)))
    for frame_index in render_indices:
        query = queries[frame_index % len(queries)] if frame_index >= split else None
        panel = render_panel(raw_evidence[frame_index], grid, prefix_support, prefix_obstacle, support_height, frame_index, len(raw_evidence), query=query)
        cv2.imwrite(str(frames_dir / f"frame_{frame_index:03d}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if frame_index == len(raw_evidence) // 2:
            cv2.imwrite(str(output_dir / "tum_freiburg1_desk_teaser.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(output_dir / "tum_freiburg1_desk_rgb.jpg"), raw_evidence[frame_index].rgb, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(output_dir / "tum_freiburg1_desk_depth.jpg"), colorize_depth(raw_evidence[frame_index].depth, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 93])
        for cells, counts in ((grid["frame_cells"][frame_index][0], prefix_support), (grid["frame_cells"][frame_index][1], prefix_obstacle)):  # type: ignore[index]
            if len(cells):
                np.add.at(counts, (cells // width, cells % width), 1)
    if not (output_dir / "tum_freiburg1_desk_teaser.jpg").exists():
        shutil.copyfile(frames_dir / "frame_000.jpg", output_dir / "tum_freiburg1_desk_teaser.jpg")
    shutil.copyfile(output_dir / "tum_freiburg1_desk_teaser.jpg", output_dir / "tum_freiburg1_desk_demo_poster.jpg")

    if not args.no_video:
        ffmpeg = find_ffmpeg()
        if ffmpeg is not None:
            video_path = output_dir / "tum_freiburg1_desk_demo.mp4"
            command = [str(ffmpeg), "-y", "-framerate", "12", "-i", str(frames_dir / "frame_%03d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "23", str(video_path)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            print("Warning: FFmpeg not found; generated still frames only", file=sys.stderr)

    if args.publish_site:
        publish_assets(output_dir, args.site_dir.resolve(), report)
    print(f"Wrote real-data pilot report: {report_path}")
    print(f"Frames: {len(raw_evidence)} · future queries: {len(event_records)} event rows · map: {height}x{width}")
    for row in rows:
        print(f"{row['id']}: Brier={float(row['brier']):.4f} ECE={float(row['ece']):.4f} n={row['count']}")


if __name__ == "__main__":
    main()
