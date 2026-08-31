"""Bounded, target-blind natural-query auditing for FlatLands packets.

This module deliberately separates query construction from target scoring.  Observation selection
uses the tracked provenance manifest.  Start/goal construction consumes only the observed floor,
the released unobserved and epistemic masks, camera metadata, raster bounds, and metric resolution.
The complete ``floor_map`` is passed only to :func:`score_natural_queries` afterwards.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping, Sequence
import zlib

import numpy as np

from .labels import clearance_radius_map, maximum_clearance_map


DEFAULT_QUERY_DISTANCES_M = (0.4, 0.8, 1.2)
DEFAULT_QUERY_ANGLES_DEG = tuple(range(0, 360, 30))
DEFAULT_FOOTPRINT_RADII_M = (0.0, 0.1, 0.2)
PROVENANCE_MANIFEST_SHA256 = (
    "a5eb28123f0fa2e38cc8244e6675c1eb76bc9a534ee54f56ef9ed68c4bdbc77b"
)


@dataclass(frozen=True)
class ManifestObservation:
    global_id: str
    provenance_split: str
    archive_split: str
    source_dataset: str
    scene_id: str
    packet_directory: str
    metadata_member: str
    original_observation_id: str
    quality_category: str
    resolution: float
    camera_px: tuple[int, int]


@dataclass(frozen=True)
class NaturalQuery:
    candidate_index: int
    distance_m: float
    angle_deg: int
    start_row: int | None
    start_col: int | None
    goal_row: int | None
    goal_col: int | None
    selection_status: str


def sha256_path(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_provenance_manifest(path: Path) -> list[ManifestObservation]:
    required = {
        "global_id",
        "provenance_split",
        "archive_split",
        "source_dataset",
        "scene_id",
        "packet_directory",
        "metadata_member",
        "original_observation_id",
        "quality_category",
        "resolution",
        "camera_px",
    }
    rows: list[ManifestObservation] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                camera = json.loads(row["camera_px"])
                if not isinstance(camera, list) or len(camera) != 2:
                    raise ValueError("camera_px must be a JSON [x,y] pair")
                resolution = float(row["resolution"])
                if not math.isfinite(resolution) or resolution <= 0:
                    raise ValueError("resolution must be positive and finite")
                rows.append(
                    ManifestObservation(
                        global_id=row["global_id"],
                        provenance_split=row["provenance_split"],
                        archive_split=row["archive_split"],
                        source_dataset=row["source_dataset"],
                        scene_id=row["scene_id"],
                        packet_directory=row["packet_directory"],
                        metadata_member=row["metadata_member"],
                        original_observation_id=row["original_observation_id"],
                        quality_category=row["quality_category"],
                        resolution=resolution,
                        camera_px=(int(camera[0]), int(camera[1])),
                    )
                )
            except Exception as error:
                raise ValueError(f"invalid manifest row {line_number}: {error}") from error
    if not rows:
        raise ValueError("manifest contains no observations")
    return rows


def _stable_digest(seed: int, *parts: object) -> bytes:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def select_scene_observations(
    observations: Sequence[ManifestObservation],
    *,
    scenes_per_stratum: int,
    seed: int,
) -> list[ManifestObservation]:
    """Choose one observation per scene, stratified by provenance split and source.

    Stable SHA-256 ranks make the selection independent of input row order and Python's hash seed.
    No image member or target label is consulted.
    """

    if scenes_per_stratum <= 0:
        raise ValueError("scenes_per_stratum must be positive")
    by_scene: dict[tuple[str, str, str], list[ManifestObservation]] = defaultdict(list)
    for row in observations:
        by_scene[(row.provenance_split, row.source_dataset, row.scene_id)].append(row)

    chosen_by_scene: dict[tuple[str, str, str], ManifestObservation] = {}
    for identity, rows in by_scene.items():
        chosen_by_scene[identity] = min(
            rows,
            key=lambda row: (
                _stable_digest(seed, "observation", *identity, row.global_id),
                row.global_id,
            ),
        )

    by_stratum: dict[tuple[str, str], list[tuple[str, ManifestObservation]]] = defaultdict(list)
    for (split, source, scene_id), row in chosen_by_scene.items():
        by_stratum[(split, source)].append((scene_id, row))

    selected: list[ManifestObservation] = []
    for (split, source), scene_rows in sorted(by_stratum.items()):
        ranked = sorted(
            scene_rows,
            key=lambda item: (
                _stable_digest(seed, "scene", split, source, item[0]),
                item[0],
            ),
        )
        selected.extend(row for _, row in ranked[:scenes_per_stratum])
    return sorted(
        selected,
        key=lambda row: (
            row.provenance_split,
            row.source_dataset,
            row.scene_id,
            row.global_id,
        ),
    )


def decode_binary_grayscale_png(payload: bytes) -> np.ndarray:
    """Decode a non-interlaced 8-bit grayscale PNG and require values to be exactly 0/255."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not payload.startswith(signature):
        raise ValueError("invalid PNG signature")
    offset = len(signature)
    width = height = None
    idat = bytearray()
    saw_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ValueError("truncated PNG chunk")
        chunk = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"PNG CRC mismatch in {chunk_type!r}")
        if chunk_type == b"IHDR":
            if length != 13 or width is not None:
                raise ValueError("invalid or duplicate PNG IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if width <= 0 or height <= 0:
                raise ValueError("PNG dimensions must be positive")
            if (depth, color, compression, filtering, interlace) != (8, 0, 0, 0, 0):
                raise ValueError(
                    "PNG must be non-interlaced 8-bit grayscale with standard compression/filtering"
                )
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
    if width is None or height is None or not idat or not saw_iend:
        raise ValueError("PNG is missing IHDR, IDAT, or IEND")
    if offset != len(payload):
        raise ValueError("unexpected bytes after PNG IEND")

    decompressed = zlib.decompress(bytes(idat))
    stride = width
    expected_bytes = height * (stride + 1)
    if len(decompressed) != expected_bytes:
        raise ValueError(
            f"unexpected PNG raster size {len(decompressed)}; expected {expected_bytes}"
        )
    output = np.empty((height, width), dtype=np.uint8)
    previous = np.zeros(width, dtype=np.uint8)
    cursor = 0
    for row_index in range(height):
        filter_type = decompressed[cursor]
        cursor += 1
        encoded = np.frombuffer(decompressed[cursor : cursor + stride], dtype=np.uint8)
        cursor += stride
        decoded = np.empty(width, dtype=np.uint8)
        for column in range(width):
            left = int(decoded[column - 1]) if column else 0
            above = int(previous[column])
            upper_left = int(previous[column - 1]) if column else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                estimate = left + above - upper_left
                left_distance = abs(estimate - left)
                above_distance = abs(estimate - above)
                diagonal_distance = abs(estimate - upper_left)
                predictor = (
                    left
                    if left_distance <= above_distance and left_distance <= diagonal_distance
                    else above if above_distance <= diagonal_distance else upper_left
                )
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            decoded[column] = (int(encoded[column]) + predictor) & 0xFF
        output[row_index] = decoded
        previous = decoded
    values = np.unique(output)
    if not set(int(value) for value in values).issubset({0, 255}):
        raise ValueError(f"PNG is not binary 0/255; values include {values[:20].tolist()}")
    return output == 255


def camera_xy_to_row_col(camera_px: Sequence[int]) -> tuple[int, int]:
    if len(camera_px) != 2:
        raise ValueError("camera_px must contain [x,y]")
    return int(camera_px[1]), int(camera_px[0])


def _nearest_true(mask: np.ndarray, point: tuple[int, int]) -> tuple[int, int] | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    row, col = point
    squared = (coordinates[:, 0] - row) ** 2 + (coordinates[:, 1] - col) ** 2
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0], squared))
    selected = coordinates[int(order[0])]
    return int(selected[0]), int(selected[1])


def construct_natural_queries(
    observed_floor: np.ndarray,
    unobserved: np.ndarray,
    epistemic_mask: np.ndarray,
    *,
    camera_px: Sequence[int],
    resolution_m: float,
    distances_m: Sequence[float] = DEFAULT_QUERY_DISTANCES_M,
    angles_deg: Sequence[int] = DEFAULT_QUERY_ANGLES_DEG,
) -> list[NaturalQuery]:
    """Create a metric polar stencil without access to the complete floor target."""

    observed = np.asarray(observed_floor, dtype=bool)
    hidden = np.asarray(unobserved, dtype=bool)
    valid = np.asarray(epistemic_mask, dtype=bool)
    if observed.ndim != 2 or hidden.shape != observed.shape or valid.shape != observed.shape:
        raise ValueError("observed_floor, unobserved, and epistemic_mask must share shape [H,W]")
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be positive and finite")
    if not distances_m or any(not math.isfinite(value) or value <= 0 for value in distances_m):
        raise ValueError("distances_m must contain positive finite values")
    if not angles_deg:
        raise ValueError("angles_deg cannot be empty")

    height, width = observed.shape
    camera_row, camera_col = camera_xy_to_row_col(camera_px)
    start: tuple[int, int] | None = None
    if 0 <= camera_row < height and 0 <= camera_col < width:
        if observed[camera_row, camera_col] and valid[camera_row, camera_col]:
            start = (camera_row, camera_col)
        else:
            start = _nearest_true(observed & valid, (camera_row, camera_col))

    output: list[NaturalQuery] = []
    seen_goals: set[tuple[int, int]] = set()
    candidate_index = 0
    for distance_m in distances_m:
        radius_cells = float(distance_m) / resolution_m
        for angle_value in angles_deg:
            angle_deg = int(angle_value)
            if start is None:
                output.append(
                    NaturalQuery(
                        candidate_index,
                        float(distance_m),
                        angle_deg,
                        None,
                        None,
                        None,
                        None,
                        "no_observed_valid_start",
                    )
                )
                candidate_index += 1
                continue
            radians = math.radians(angle_deg)
            goal_row = start[0] + int(round(math.sin(radians) * radius_cells))
            goal_col = start[1] + int(round(math.cos(radians) * radius_cells))
            status = "selected"
            if not (0 <= goal_row < height and 0 <= goal_col < width):
                status = "goal_out_of_bounds"
            elif (goal_row, goal_col) == start:
                status = "goal_equals_start"
            elif not valid[goal_row, goal_col]:
                status = "goal_outside_epistemic_mask"
            elif not hidden[goal_row, goal_col]:
                status = "goal_not_unobserved"
            elif (goal_row, goal_col) in seen_goals:
                status = "duplicate_goal"
            if status == "selected":
                seen_goals.add((goal_row, goal_col))
            output.append(
                NaturalQuery(
                    candidate_index,
                    float(distance_m),
                    angle_deg,
                    start[0],
                    start[1],
                    goal_row,
                    goal_col,
                    status,
                )
            )
            candidate_index += 1
    return output


def radii_m_to_cells(radii_m: Sequence[float], resolution_m: float) -> tuple[int, ...]:
    if not radii_m or any(not math.isfinite(value) or value < 0 for value in radii_m):
        raise ValueError("radii_m must contain non-negative finite values")
    cells = tuple(int(round(value / resolution_m)) for value in radii_m)
    if any(abs(cell * resolution_m - value) > 1e-9 for cell, value in zip(cells, radii_m)):
        raise ValueError("each physical radius must be an integer number of cells")
    if tuple(sorted(set(cells))) != cells:
        raise ValueError("radii must map to unique increasing cell radii")
    return cells


def mask_relation_counts(
    observed_floor: np.ndarray,
    floor_map: np.ndarray,
    unobserved: np.ndarray,
    epistemic_mask: np.ndarray,
) -> dict[str, int]:
    observed = np.asarray(observed_floor, dtype=bool)
    target = np.asarray(floor_map, dtype=bool)
    hidden = np.asarray(unobserved, dtype=bool)
    valid = np.asarray(epistemic_mask, dtype=bool)
    if any(array.shape != observed.shape for array in (target, hidden, valid)):
        raise ValueError("all four FlatLands maps must have the same shape")
    return {
        "pixels": int(observed.size),
        "observed_floor_positive": int(observed.sum()),
        "floor_map_positive": int(target.sum()),
        "unobserved_positive": int(hidden.sum()),
        "epistemic_mask_positive": int(valid.sum()),
        "observed_not_floor": int(np.sum(observed & ~target)),
        "observed_unobserved_overlap": int(np.sum(observed & hidden)),
        "observed_outside_epistemic": int(np.sum(observed & ~valid)),
        "floor_outside_epistemic": int(np.sum(target & ~valid)),
        "unobserved_outside_epistemic": int(np.sum(hidden & ~valid)),
        "valid_unobserved": int(np.sum(hidden & valid)),
        "valid_unobserved_floor": int(np.sum(hidden & valid & target)),
    }


def score_natural_queries(
    floor_map: np.ndarray,
    epistemic_mask: np.ndarray,
    queries: Sequence[NaturalQuery],
    *,
    radii_cells: Sequence[int],
) -> list[dict[str, object]]:
    """Score frozen target-blind queries against the complete target and exact disk oracle."""

    target = np.asarray(floor_map, dtype=bool)
    valid = np.asarray(epistemic_mask, dtype=bool)
    if target.ndim != 2 or valid.shape != target.shape:
        raise ValueError("floor_map and epistemic_mask must share shape [H,W]")
    radii = tuple(int(value) for value in radii_cells)
    if not radii or tuple(sorted(set(radii))) != radii or radii[0] != 0:
        raise ValueError("radii_cells must be unique increasing integers beginning at zero")

    selected = [query for query in queries if query.selection_status == "selected"]
    start_pairs = {(query.start_row, query.start_col) for query in selected}
    if len(start_pairs) > 1:
        raise ValueError("all natural queries for one observation must share a start")
    free = target & valid
    capacity: np.ndarray | None = None
    if selected:
        start = next(iter(start_pairs))
        if start[0] is None or start[1] is None:
            raise ValueError("selected query has no start")
        start_rc = (int(start[0]), int(start[1]))
        if free[start_rc]:
            free_goals = [
                (int(query.goal_row), int(query.goal_col))
                for query in selected
                if query.goal_row is not None
                and query.goal_col is not None
                and free[int(query.goal_row), int(query.goal_col)]
            ]
            clearance = clearance_radius_map(free)
            capacity = maximum_clearance_map(
                free, start_rc, clearance=clearance, stop_points=free_goals
            )

    output: list[dict[str, object]] = []
    for query in queries:
        record: dict[str, object] = {
            "candidate_index": query.candidate_index,
            "distance_m": query.distance_m,
            "angle_deg": query.angle_deg,
            "start_row": query.start_row,
            "start_col": query.start_col,
            "goal_row": query.goal_row,
            "goal_col": query.goal_col,
            "selection_status": query.selection_status,
            "target_status": "not_scored_selection_rejected",
            "max_clearance_cells": None,
        }
        for radius in radii:
            record[f"reachable_r{radius}_cells"] = None
        if query.selection_status != "selected":
            output.append(record)
            continue
        assert query.start_row is not None and query.start_col is not None
        assert query.goal_row is not None and query.goal_col is not None
        start_rc = (query.start_row, query.start_col)
        goal_rc = (query.goal_row, query.goal_col)
        if not free[start_rc]:
            record["target_status"] = "target_invalid_start"
        elif not free[goal_rc]:
            record["target_status"] = "target_invalid_goal"
        else:
            assert capacity is not None
            max_clearance = int(capacity[goal_rc])
            record["max_clearance_cells"] = max_clearance
            for radius in radii:
                record[f"reachable_r{radius}_cells"] = max_clearance >= radius
            if max_clearance < 0:
                record["target_status"] = "disconnected_radius_zero"
            elif max_clearance < radii[-1]:
                record["target_status"] = "footprint_failure"
            else:
                record["target_status"] = "high_clearance_positive"
        output.append(record)
    return output


def aggregate_integer_counts(records: Iterable[Mapping[str, int]]) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for record in records:
        for key, value in record.items():
            output[key] += int(value)
    return dict(sorted(output.items()))

