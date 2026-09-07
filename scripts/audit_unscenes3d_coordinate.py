#!/usr/bin/env python3
"""Audit the UnScenes3D LiDAR/occupancy coordinate contract.

This is a read-only, train/validation-only audit.  It checks the official voxel
range and calibration matrices, projects LiDAR returns into the camera, compares
same-timestamp raw returns with the two local-map parts, and measures how the
sensor-only endpoint policies interact with the sparse target-validity mask.  The
location-6 files are never opened by this script.

The report is deliberately diagnostic rather than a model result.  In particular,
target classes are used only to audit the adapter and are never passed to the input
observation or query-start selector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from pathrel.labels import clearance_radius_map, maximum_clearance_map
from pathrel.unscenes3d import (
    GRID_SHAPE,
    POINT_CLOUD_RANGE_M,
    VOXEL_SIZE_M,
    deterministic_queries,
    lidar_observation,
    occupancy_support,
)


TRAIN_VALIDATION_SITES = frozenset(
    {"location_1", "location_2", "location_3", "location_4_5"}
)
LOCKED_SITES = frozenset({"location_6"})
RADII_CELLS = (0, 1, 2)
DISTANCES_CELLS = (13, 27, 40)
ANGLES_DEG = tuple(range(0, 360, 30))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_calibration(path: Path) -> dict[str, np.ndarray]:
    """Parse the release's KITTI-style calibration text without assumptions about order."""

    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        tokens = raw.strip().split()
        if not tokens:
            continue
        try:
            values[key.strip()] = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"non-numeric calibration value in {path}: {line!r}") from exc

    required = {
        "P2": (3, 4),
        "R0_rect": (3, 3),
        "Tr_velo_to_cam": (3, 4),
        "Tr_velo_to_imu": (3, 4),
    }
    result: dict[str, np.ndarray] = {}
    for key, shape in required.items():
        if key not in values or len(values[key]) != shape[0] * shape[1]:
            raise ValueError(f"calibration {path} has missing/malformed {key}")
        result[key] = np.asarray(values[key], dtype=np.float64).reshape(shape)
    return result


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[: matrix.shape[0], :] = matrix
    return result


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.concatenate(
        (np.asarray(points, dtype=np.float64)[:, :3], np.ones((len(points), 1))),
        axis=1,
    )
    return (homogeneous @ matrix.T)[:, :3]


def _project(points: np.ndarray, calibration: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points using P2 * R0_rect * Tr_velo_to_cam."""

    lidar_to_camera = _homogeneous(calibration["Tr_velo_to_cam"])
    rectification = np.eye(4, dtype=np.float64)
    rectification[:3, :3] = calibration["R0_rect"]
    camera = _transform(points, rectification @ lidar_to_camera)
    camera_h = np.concatenate(
        (camera, np.ones((len(camera), 1), dtype=np.float64)), axis=1
    )
    projected = camera_h @ calibration["P2"].T
    positive = projected[:, 2] > 1e-6
    uv = np.full((len(camera), 2), np.nan, dtype=np.float64)
    uv[positive] = projected[positive, :2] / projected[positive, 2, None]
    return camera, uv


def _rotation_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    rotation = matrix[:, :3]
    return {
        "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
        "determinant": float(np.linalg.det(rotation)),
        "translation_norm_m": float(np.linalg.norm(matrix[:, 3])) if matrix.shape[1] == 4 else 0.0,
    }


def _sample_rows(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return np.asarray(points)
    # Evenly spaced selection is deterministic and avoids favouring the file's
    # first return ring.  It is used only for a diagnostic nearest-neighbour check.
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
    return np.asarray(points)[indices]


def _nearest_distances(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, str]:
    """Return nearest distances with scipy acceleration and a NumPy fallback."""

    if len(points) == 0 or len(reference) == 0:
        return np.empty((0,), dtype=np.float64), "empty"
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(reference)
        distances, _ = tree.query(points, k=1, workers=1)
        return np.asarray(distances, dtype=np.float64), "scipy.cKDTree"
    except ImportError:
        pass

    # Keep the fallback bounded in memory.  This path is slower but has no extra
    # project dependency and is sufficient for the small sampled audit packet.
    output = np.empty((len(points),), dtype=np.float64)
    for begin in range(0, len(points), 64):
        chunk = points[begin : begin + 64]
        best = np.full((len(chunk),), np.inf, dtype=np.float64)
        for ref_begin in range(0, len(reference), 4096):
            ref = reference[ref_begin : ref_begin + 4096]
            distance = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            best = np.minimum(best, distance.min(axis=1))
        output[begin : begin + len(chunk)] = np.sqrt(best)
    return output, "numpy_chunked"


def _distance_summary(distances: np.ndarray) -> dict[str, float | int | None]:
    if len(distances) == 0:
        return {
            "count": 0,
            "median_m": None,
            "p90_m": None,
            "within_0.3m": None,
            "within_1m": None,
        }
    return {
        "count": int(len(distances)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.quantile(distances, 0.9)),
        "within_0.3m": float(np.mean(distances <= 0.3)),
        "within_1m": float(np.mean(distances <= 1.0)),
    }


def _cell_indices(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(points)[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    rows = np.full((len(xyz),), -1, dtype=np.int64)
    cols = np.full((len(xyz),), -1, dtype=np.int64)
    rows[finite] = np.floor(xyz[finite, 0] / VOXEL_SIZE_M).astype(np.int64)
    cols[finite] = np.floor(
        (xyz[finite, 1] - POINT_CLOUD_RANGE_M[1]) / VOXEL_SIZE_M
    ).astype(np.int64)
    in_grid = finite & (rows >= 0) & (rows < GRID_SHAPE[0]) & (cols >= 0) & (cols < GRID_SHAPE[1])
    return rows, cols, in_grid


def _status_counts(mask: np.ndarray, support: Any, rows: np.ndarray, cols: np.ndarray) -> Counter[str]:
    result: Counter[str] = Counter()
    if not np.any(mask):
        return result
    selected_rows, selected_cols = rows[mask], cols[mask]
    valid = support.valid[selected_rows, selected_cols]
    free = support.free[selected_rows, selected_cols]
    result["invalid"] = int(np.sum(~valid))
    result["free"] = int(np.sum(valid & free))
    result["blocked"] = int(np.sum(valid & ~free))
    return result


def _status_for_start(start: np.ndarray, support: Any) -> str:
    if len(start) == 0:
        return "empty"
    row, col = (int(start[0, 0]), int(start[0, 1]))
    if not support.valid[row, col]:
        return "invalid"
    return "free" if support.free[row, col] else "blocked"


def _event_labels(
    target_free: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    *,
    clearance: np.ndarray | None = None,
) -> np.ndarray:
    if len(starts) == 0:
        return np.empty((0, len(RADII_CELLS)), dtype=bool)
    if clearance is None:
        clearance = clearance_radius_map(target_free)
    best = maximum_clearance_map(
        target_free,
        tuple(int(value) for value in starts[0]),
        clearance=clearance,
        stop_points=[tuple(int(value) for value in goal) for goal in goals],
    )
    max_clearance = best[goals[:, 0], goals[:, 1]]
    return max_clearance[:, None] >= np.asarray(RADII_CELLS)[None, :]


def _render_overlay(
    *,
    output: Path,
    image_path: Path,
    points: np.ndarray,
    calibration: dict[str, np.ndarray],
    support: Any,
    legacy_observation: np.ndarray,
    ground_observation: np.ndarray,
    local_map: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    timestamp: str,
    scene_id: str,
) -> dict[str, object]:
    """Render a compact camera/BEV contract figure using Pillow."""

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent fallback
        raise RuntimeError("Pillow is required for --render-overlay") from exc

    image = Image.open(image_path).convert("RGB")
    camera, uv = _project(points[:, :3], calibration)
    width, height = image.size
    visible = (
        np.isfinite(uv).all(axis=1)
        & (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    projected_count = int(np.sum(visible))

    canvas_width, canvas_height = 1600, 1140
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 247, 249))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
        panel_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:  # pragma: no cover - font availability varies
        title_font = panel_font = small_font = ImageFont.load_default()

    draw.text((32, 18), f"UnScenes3D coordinate audit · {scene_id} · {timestamp}", fill=(25, 38, 50), font=title_font)

    camera_panel = image.copy()
    camera_panel.thumbnail((1000, 535), Image.Resampling.LANCZOS)
    camera_x, camera_y = 32, 64
    canvas.paste(camera_panel, (camera_x, camera_y))
    scale_x = camera_panel.width / width
    scale_y = camera_panel.height / height
    camera_draw = ImageDraw.Draw(canvas)
    visible_indices = np.flatnonzero(visible)
    visible_indices = _sample_rows(visible_indices[:, None], 1800).reshape(-1) if len(visible_indices) > 1800 else visible_indices
    for index in visible_indices.tolist():
        u, v = uv[index]
        px = camera_x + int(round(u * scale_x))
        py = camera_y + int(round(v * scale_y))
        depth = float(camera[index, 2])
        color = (41, 166, 124) if depth < 12 else (235, 155, 55) if depth < 30 else (89, 125, 190)
        camera_draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    draw.text((camera_x, 42), f"Camera projection (visible {projected_count}/{len(points)} returns)", fill=(25, 38, 50), font=panel_font)

    def bev_image(*, mode: str) -> Image.Image:
        base = np.full((*GRID_SHAPE, 3), (33, 43, 52), dtype=np.uint8)
        valid = support.valid
        base[valid & support.free] = (50, 156, 111)
        base[valid & ~support.free] = (194, 79, 72)
        if mode == "legacy":
            observed_free, observed_blocked = legacy_observation[0] > 0.5, legacy_observation[1] > 0.5
        elif mode == "ground":
            observed_free, observed_blocked = ground_observation[0] > 0.5, ground_observation[1] > 0.5
        else:
            observed_free = np.zeros(GRID_SHAPE, dtype=bool)
            observed_blocked = np.zeros(GRID_SHAPE, dtype=bool)
        if mode != "target":
            # Blend sensor evidence over the target so coordinate and endpoint
            # semantics can be inspected in one panel.
            base[observed_free] = (49, 190, 211)
            base[observed_blocked] = (244, 157, 53)
        base[0, 128] = (255, 255, 255)
        if len(starts):
            base[int(starts[0, 0]), int(starts[0, 1])] = (255, 255, 255)
        for goal in goals:
            base[int(goal[0]), int(goal[1])] = (245, 230, 70)
        # local-map points are sparse; draw them after the raster with a dark
        # blue mark to make the direct same-frame agreement visible.
        local_rows, local_cols, local_in = _cell_indices(local_map)
        step = max(1, int(np.sum(local_in) / 5000))
        for row, col in zip(local_rows[local_in][::step], local_cols[local_in][::step]):
            base[int(row), int(col)] = (72, 106, 184)
        return Image.fromarray(base, mode="RGB").resize((480, 480), Image.Resampling.NEAREST)

    legacy_panel = bev_image(mode="legacy")
    ground_panel = bev_image(mode="ground")
    target_panel = bev_image(mode="target")
    panel_y = 632
    panel_positions = (32, 560, 1088)
    panel_titles = (
        "BEV: legacy all-return blocked",
        "BEV: label-free ground endpoints",
        "BEV: target + local-map",
    )
    for x0, panel, title in zip(
        panel_positions,
        (legacy_panel, ground_panel, target_panel),
        panel_titles,
    ):
        canvas.paste(panel, (x0, panel_y))
        draw.text((x0, panel_y - 24), title, fill=(25, 38, 50), font=panel_font)
    draw.text((32, 1118), "row = forward x · col = lateral y · green target support · red target block · cyan observed free · orange observed endpoint · blue local map", fill=(70, 82, 92), font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    temporary.replace(output)
    return {
        "path": str(output),
        "scene_id": scene_id,
        "timestamp": timestamp,
        "image_size": [int(width), int(height)],
        "projected_positive_depth": int(np.sum(camera[:, 2] > 0)),
        "projected_in_image": projected_count,
    }


def _frame_paths(raw_root: Path, label_root: Path, timestamp: str) -> dict[str, Path]:
    return {
        "image": raw_root / "images" / f"{timestamp}.jpg",
        "cloud": raw_root / "clouds" / f"{timestamp}.bin",
        "calib": raw_root / "calibs" / f"{timestamp}.txt",
        "occ": label_root / "occ" / f"{timestamp}.npy",
    }


def audit(
    raw_root: Path,
    label_root: Path,
    map_dirs: Iterable[Path],
    *,
    max_frames_per_scene: int,
    nearest_sample_points: int,
    visual_scene: str,
    visual_index: int,
    output: Path,
    render_overlay: bool,
) -> dict[str, object]:
    if max_frames_per_scene < 1:
        raise ValueError("max_frames_per_scene must be positive")
    if nearest_sample_points < 1:
        raise ValueError("nearest_sample_points must be positive")
    scene_info_path = raw_root / "imagesets" / "scene_info.json"
    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    map_dirs = tuple(Path(path) for path in map_dirs)
    map_lookup: dict[str, Path] = {}
    for directory in map_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.bin"):
            map_lookup.setdefault(path.stem, path)

    aggregate = Counter()
    per_scene: dict[str, dict[str, object]] = {}
    calibration_rows: list[dict[str, object]] = []
    nearest_rows: list[dict[str, object]] = []
    start_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    overlay_payload: dict[str, object] | None = None
    selected_scene_found = False
    started = time.monotonic()

    for scene_id, scene in scene_info.items():
        location = str(scene.get("location", ""))
        # This explicit allow-list is the test lock: no path under location_6 is
        # constructed or opened, even if it appears in scene_info.
        if location not in TRAIN_VALIDATION_SITES:
            if location in LOCKED_SITES:
                aggregate["locked_scene_count"] += 1
            continue
        timestamps = [str(value) for value in scene.get("samples", [])]
        if not timestamps:
            continue
        selected_indices = np.linspace(
            0, len(timestamps) - 1, num=min(max_frames_per_scene, len(timestamps)), dtype=np.int64
        )
        selected_timestamps = {timestamps[int(index)] for index in selected_indices}
        scene_counter = Counter()
        scene_nearest: list[dict[str, object]] = []
        scene_starts: list[dict[str, object]] = []
        scene_endpoint: list[dict[str, object]] = []
        for timestamp in timestamps:
            paths = _frame_paths(raw_root, label_root, timestamp)
            if not paths["cloud"].exists() or not paths["occ"].exists() or not paths["calib"].exists():
                scene_counter["missing_frame_inputs"] += 1
                continue
            points_flat = np.fromfile(paths["cloud"], dtype=np.float32)
            if points_flat.size % 4:
                scene_counter["malformed_cloud"] += 1
                continue
            points = points_flat.reshape(-1, 4)
            support = occupancy_support(np.load(paths["occ"], allow_pickle=False))
            rows, cols, in_grid = _cell_indices(points)
            point_status = _status_counts(in_grid, support, rows, cols)
            for key, value in point_status.items():
                aggregate[f"point_{key}"] += value
                scene_counter[f"point_{key}"] += value
            if np.any(in_grid):
                keys = np.unique(rows[in_grid] * GRID_SHAPE[1] + cols[in_grid])
                unique_rows, unique_cols = keys // GRID_SHAPE[1], keys % GRID_SHAPE[1]
                unique_valid = support.valid[unique_rows, unique_cols]
                unique_free = support.free[unique_rows, unique_cols]
                unique_status = {
                    "invalid": int(np.sum(~unique_valid)),
                    "free": int(np.sum(unique_valid & unique_free)),
                    "blocked": int(np.sum(unique_valid & ~unique_free)),
                }
                for key, value in unique_status.items():
                    aggregate[f"unique_endpoint_cell_{key}"] += value
                    scene_counter[f"unique_endpoint_cell_{key}"] += value
            aggregate["points"] += int(len(points))
            scene_counter["points"] += int(len(points))

            legacy = lidar_observation(points, endpoint_policy="blocked")
            ground = lidar_observation(points, endpoint_policy="ground")
            for policy, observation in (("legacy", legacy), ("ground", ground)):
                blocked_cells = observation[1] > 0.5
                counts: Counter[str] = Counter()
                blocked_positions = np.argwhere(blocked_cells)
                if len(blocked_positions):
                    valid = support.valid[blocked_positions[:, 0], blocked_positions[:, 1]]
                    free = support.free[blocked_positions[:, 0], blocked_positions[:, 1]]
                    counts = Counter({"invalid": int(np.sum(~valid)), "free": int(np.sum(valid & free)), "blocked": int(np.sum(valid & ~free))})
                for key, value in counts.items():
                    aggregate[f"{policy}_blocked_cell_{key}"] += value
                    scene_counter[f"{policy}_blocked_cell_{key}"] += value

            old_starts, old_goals = deterministic_queries(
                support.valid,
                distances_cells=DISTANCES_CELLS,
                angles_deg=ANGLES_DEG,
            )
            candidate_starts, candidate_goals = deterministic_queries(
                support.valid,
                distances_cells=DISTANCES_CELLS,
                angles_deg=ANGLES_DEG,
            )
            old_status = _status_for_start(old_starts, support)
            candidate_status = _status_for_start(candidate_starts, support)
            for label, status in (("legacy", old_status), ("candidate", candidate_status)):
                aggregate[f"start_{label}_{status}"] += 1
                scene_counter[f"start_{label}_{status}"] += 1
            aggregate["old_queries"] += int(len(old_goals))
            aggregate["candidate_queries"] += int(len(candidate_goals))
            scene_counter["old_queries"] += int(len(old_goals))
            scene_counter["candidate_queries"] += int(len(candidate_goals))
            old_patterns: Counter[str] = Counter()
            candidate_patterns: Counter[str] = Counter()
            # Exact bottleneck labels are expensive (the Euclidean distance
            # transform is ~0.1 s/frame), so calculate them on the deterministic
            # per-scene audit sample.  The full frozen manifest performs the same
            # calculation for every retained frame.
            if timestamp in selected_timestamps:
                clearance = clearance_radius_map(support.free)
                old_events = _event_labels(
                    support.free, old_starts, old_goals, clearance=clearance
                )
                candidate_events = _event_labels(
                    support.free, candidate_starts, candidate_goals, clearance=clearance
                )
                old_patterns = Counter(
                    "".join("1" if value else "0" for value in row)
                    for row in old_events
                )
                candidate_patterns = Counter(
                    "".join("1" if value else "0" for value in row)
                    for row in candidate_events
                )
                for pattern, value in old_patterns.items():
                    aggregate[f"old_event_pattern_{pattern}"] += value
                    scene_counter[f"old_event_pattern_{pattern}"] += value
                for pattern, value in candidate_patterns.items():
                    aggregate[f"candidate_event_pattern_{pattern}"] += value
                    scene_counter[f"candidate_event_pattern_{pattern}"] += value
                for radius_index, radius in enumerate(RADII_CELLS):
                    aggregate[f"old_positive_radius_{radius}"] += int(
                        old_events[:, radius_index].sum()
                    )
                    aggregate[f"candidate_positive_radius_{radius}"] += int(
                        candidate_events[:, radius_index].sum()
                    )
                    scene_counter[f"old_positive_radius_{radius}"] += int(
                        old_events[:, radius_index].sum()
                    )
                    scene_counter[f"candidate_positive_radius_{radius}"] += int(
                        candidate_events[:, radius_index].sum()
                    )

            if timestamp in selected_timestamps:
                calibration = _parse_calibration(paths["calib"])
                for key in ("R0_rect", "Tr_velo_to_cam", "Tr_velo_to_imu"):
                    diagnostics = _rotation_diagnostics(calibration[key])
                    calibration_rows.append({"scene_id": scene_id, "timestamp": timestamp, "matrix": key, **diagnostics})
                image_size: tuple[int, int] | None = None
                try:
                    from PIL import Image  # type: ignore

                    if paths["image"].exists():
                        with Image.open(paths["image"]) as image_handle:
                            image_size = image_handle.size
                except ImportError:
                    image_size = None
                if image_size is not None:
                    camera, uv = _project(points[:, :3], calibration)
                    positive = camera[:, 2] > 1e-6
                    in_image = positive & np.isfinite(uv).all(axis=1)
                    in_image &= (uv[:, 0] >= 0) & (uv[:, 0] < image_size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < image_size[1])
                    aggregate["camera_points_checked"] += int(len(points))
                    aggregate["camera_positive_depth"] += int(np.sum(positive))
                    aggregate["camera_in_image"] += int(np.sum(in_image))
                    scene_counter["camera_points_checked"] += int(len(points))
                    scene_counter["camera_positive_depth"] += int(np.sum(positive))
                    scene_counter["camera_in_image"] += int(np.sum(in_image))

                local_path = map_lookup.get(timestamp)
                if local_path is not None:
                    local_flat = np.fromfile(local_path, dtype=np.float32)
                    if local_flat.size % 3 == 0:
                        local = local_flat.reshape(-1, 3)
                        sampled = _sample_rows(points[:, :3], nearest_sample_points)
                        sampled = sampled[np.isfinite(sampled).all(axis=1)]
                        local = local[np.isfinite(local).all(axis=1)]
                        direct, backend = _nearest_distances(sampled, local)
                        transformed = _transform(sampled, _homogeneous(calibration["Tr_velo_to_imu"]))
                        inverse = _transform(sampled, np.linalg.inv(_homogeneous(calibration["Tr_velo_to_imu"])))
                        direct_summary = _distance_summary(direct)
                        transformed_summary = _distance_summary(_nearest_distances(transformed, local)[0])
                        inverse_summary = _distance_summary(_nearest_distances(inverse, local)[0])
                        row = {
                            "scene_id": scene_id,
                            "timestamp": timestamp,
                            "sample_points": int(len(sampled)),
                            "local_points": int(len(local)),
                            "nearest_backend": backend,
                            "direct_lidar_to_localmap": direct_summary,
                            "imu_transform_then_localmap": transformed_summary,
                            "inverse_imu_transform_then_localmap": inverse_summary,
                        }
                        nearest_rows.append(row)
                        scene_nearest.append(row)

            if scene_id == visual_scene and timestamp == timestamps[visual_index % len(timestamps)] and render_overlay:
                selected_scene_found = True
                calibration = _parse_calibration(paths["calib"])
                local_path = map_lookup.get(timestamp)
                local = np.empty((0, 3), dtype=np.float32)
                if local_path is not None:
                    local_values = np.fromfile(local_path, dtype=np.float32)
                    if local_values.size % 3 == 0:
                        local = local_values.reshape(-1, 3)
                overlay_path = output.with_name("coordinate_overlay.png")
                overlay_payload = _render_overlay(
                    output=overlay_path,
                    image_path=paths["image"],
                    points=points,
                    calibration=calibration,
                    support=support,
                    legacy_observation=legacy,
                    ground_observation=ground,
                    local_map=local,
                    starts=candidate_starts,
                    goals=candidate_goals,
                    timestamp=timestamp,
                    scene_id=scene_id,
                )

            endpoint_rows.append(
                {
                    "scene_id": scene_id,
                    "location": location,
                    "timestamp": timestamp,
                    "point_status": dict(point_status),
                    "legacy_blocked_cells": int(np.sum(legacy[1] > 0.5)),
                    "ground_blocked_cells": int(np.sum(ground[1] > 0.5)),
                    "legacy_start": old_starts[0].tolist() if len(old_starts) else None,
                    "legacy_start_status": old_status,
                    "candidate_start": candidate_starts[0].tolist() if len(candidate_starts) else None,
                    "candidate_start_status": candidate_status,
                    "old_query_count": int(len(old_goals)),
                    "candidate_query_count": int(len(candidate_goals)),
                    "old_event_patterns": dict(sorted(old_patterns.items())),
                    "candidate_event_patterns": dict(sorted(candidate_patterns.items())),
                }
            )
            if timestamp in selected_timestamps:
                start_rows.append({"scene_id": scene_id, "timestamp": timestamp, "legacy": old_status, "candidate": candidate_status, "old_query_count": int(len(old_goals)), "candidate_query_count": int(len(candidate_goals))})

        per_scene[scene_id] = {
            "location": location,
            "frames": int(len(timestamps)),
            "audited_frames": int(sum(1 for row in endpoint_rows if row["scene_id"] == scene_id)),
            "sampled_coordinate_frames": len(scene_nearest),
            "counters": dict(sorted(scene_counter.items())),
            "nearest_rows": scene_nearest,
        }

    if render_overlay and not selected_scene_found:
        raise ValueError(f"visual scene/index not found in train/validation samples: {visual_scene}#{visual_index}")

    nearest_direct = [row["direct_lidar_to_localmap"] for row in nearest_rows]
    nearest_transformed = [row["imu_transform_then_localmap"] for row in nearest_rows]
    nearest_inverse = [row["inverse_imu_transform_then_localmap"] for row in nearest_rows]

    def _merge_distance(rows: list[dict[str, object]]) -> dict[str, object]:
        values = [float(row["median_m"]) for row in rows if row.get("median_m") is not None]
        fractions_03 = [float(row["within_0.3m"]) for row in rows if row.get("within_0.3m") is not None]
        fractions_1 = [float(row["within_1m"]) for row in rows if row.get("within_1m") is not None]
        return {
            "frames": len(rows),
            "median_of_frame_medians": float(np.median(values)) if values else None,
            "mean_frame_fraction_within_0.3m": float(np.mean(fractions_03)) if fractions_03 else None,
            "mean_frame_fraction_within_1m": float(np.mean(fractions_1)) if fractions_1 else None,
        }

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "unscenes3d_coordinate_contract_audit",
        "protocol_version": "UNSCENES3D_PROTOCOL.md v0.2 (coordinate audit)",
        "raw_root": str(raw_root),
        "label_root": str(label_root),
        "local_map_dirs": [str(path) for path in map_dirs],
        "grid": {
            "shape_xy": list(GRID_SHAPE),
            "shape_xyz": [GRID_SHAPE[0], GRID_SHAPE[1], 32],
            "voxel_size_m": VOXEL_SIZE_M,
            "point_cloud_range_m": list(POINT_CLOUD_RANGE_M),
            "row_axis": "x forward",
            "column_axis": "y lateral",
        },
        "split": {
            "audited_sites": sorted(TRAIN_VALIDATION_SITES),
            "locked_sites": sorted(LOCKED_SITES),
            "scenes_seen": len(per_scene),
            "location_6_opened": False,
        },
        "aggregate": dict(sorted(aggregate.items())),
        "calibration": {
            "sampled_frames": len({(row["scene_id"], row["timestamp"]) for row in calibration_rows}),
            "sampled_matrices": len(calibration_rows),
            "matrix_diagnostics": calibration_rows,
            "max_orthogonality_error": max((row["orthogonality_error"] for row in calibration_rows), default=None),
            "min_rotation_determinant": min((row["determinant"] for row in calibration_rows), default=None),
        },
        "local_map_nearest": {
            "sampled_frames": len(nearest_rows),
            "direct": _merge_distance(nearest_direct),
            "imu_transform": _merge_distance(nearest_transformed),
            "inverse_imu_transform": _merge_distance(nearest_inverse),
            "rows": nearest_rows,
        },
        "endpoint_and_start_rows": endpoint_rows,
        "sampled_start_rows": start_rows,
        "per_scene": per_scene,
        "overlay": overlay_payload,
        "configuration": {
            "max_frames_per_scene": max_frames_per_scene,
            "nearest_sample_points": nearest_sample_points,
            "endpoint_policies": {"legacy": "blocked", "candidate": "ground"},
            "start_policies": {"legacy": "valid", "candidate": "valid"},
            "query_distances_cells": list(DISTANCES_CELLS),
            "query_angles_deg": list(ANGLES_DEG),
            "radii_cells": list(RADII_CELLS),
        },
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "Read-only train/validation adapter audit; location_6 test labels and images were not read; no model score or cross-domain claim.",
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"))
    parser.add_argument("--localmap-dir", type=Path, action="append", default=[], help="localmap_clouds directory; repeat for both release parts")
    parser.add_argument("--output", type=Path, default=Path("results/unscenes3d_coordinate_audit/report.json"))
    parser.add_argument("--max-frames-per-scene", type=int, default=5)
    parser.add_argument("--nearest-sample-points", type=int, default=2048)
    parser.add_argument("--visual-scene", default="scene_00427")
    parser.add_argument("--visual-index", type=int, default=-1)
    parser.add_argument("--no-overlay", action="store_true", help="skip the Pillow coordinate overlay")
    args = parser.parse_args()
    map_dirs = args.localmap_dir or [
        Path("data/raw/unscenes3d/localmap_package_1/unscenes3d-mini_loaclmap-1/localmap_clouds"),
        Path("data/raw/unscenes3d/localmap_package_2/unscenes3d-mini_loaclmap-2/localmap_clouds"),
    ]
    report = audit(
        args.raw_root,
        args.label_root,
        map_dirs,
        max_frames_per_scene=args.max_frames_per_scene,
        nearest_sample_points=args.nearest_sample_points,
        visual_scene=args.visual_scene,
        visual_index=args.visual_index,
        output=args.output,
        render_overlay=not args.no_overlay,
    )
    print(json.dumps({
        "output": str(args.output),
        "audited_scenes": report["split"]["scenes_seen"],
        "locked_sites": report["split"]["locked_sites"],
        "local_map_frames": report["local_map_nearest"]["sampled_frames"],
        "overlay": report["overlay"],
        "location_6_opened": report["split"]["location_6_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
