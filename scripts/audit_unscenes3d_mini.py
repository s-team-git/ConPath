#!/usr/bin/env python3
"""Audit the locally acquired UnScenes3D mini raw/label/map packages.

This is deliberately a read-only contract audit.  It verifies timestamp joins and
basic voxel/map invariants, but it does not construct a train/test split or report
model metrics.  The local-map archive is supplied more than once because the
official mini release is split into ``loaclmap-1`` and ``loaclmap-2`` archives.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


RAW_DIRS = ("images", "clouds", "calibs")
LABEL_DIRS = ("occ", "elevation", "depths", "image_caption", "vehicle_infos", "label_3d")


def _stems(path: Path) -> set[str]:
    return {p.stem for p in path.iterdir() if p.is_file()}


def _png_shape(path: Path) -> tuple[int, int] | None:
    # PNG dimensions are available in the fixed 24-byte header; no image library
    # (and therefore no decoding side effect) is needed for this audit.
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _scene_join(scene_info: dict, available: dict[str, set[str]]) -> dict:
    scenes = {}
    duplicate_samples: list[str] = []
    seen: dict[str, str] = {}
    for scene, value in scene_info.items():
        samples = list(value.get("samples", []))
        for stamp in samples:
            previous = seen.setdefault(stamp, scene)
            if previous != scene:
                duplicate_samples.append(stamp)
        row = {"location": value.get("location"), "samples": len(samples)}
        for name, stems in available.items():
            row[name] = len(set(samples) & stems)
        scenes[scene] = row
    return {"scenes": scenes, "duplicate_samples": sorted(set(duplicate_samples))}


def audit(raw_root: Path, label_root: Path, map_dirs: Iterable[Path]) -> dict:
    info_path = raw_root / "imagesets" / "scene_info.json"
    scene_info = json.loads(info_path.read_text(encoding="utf-8"))
    raw = {name: _stems(raw_root / name) for name in RAW_DIRS}
    labels = {name: _stems(label_root / name) for name in LABEL_DIRS}
    maps = set()
    map_file_count = 0
    map_bad_float_count = 0
    map_point_stats: list[int] = []
    map_coordinate_min: np.ndarray | None = None
    map_coordinate_max: np.ndarray | None = None
    for directory in map_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.bin"):
            map_file_count += 1
            maps.add(path.stem)
            values = np.fromfile(path, dtype=np.float32)
            if values.size % 3:
                map_bad_float_count += 1
                continue
            points = values.reshape(-1, 3)
            map_point_stats.append(int(points.shape[0]))
            if not np.isfinite(points).all():
                map_bad_float_count += 1
            pmin, pmax = points.min(axis=0), points.max(axis=0)
            map_coordinate_min = pmin if map_coordinate_min is None else np.minimum(map_coordinate_min, pmin)
            map_coordinate_max = pmax if map_coordinate_max is None else np.maximum(map_coordinate_max, pmax)

    available = {**raw, **labels, "localmap": maps}
    joins = _scene_join(scene_info, available)
    all_samples = {stamp for value in scene_info.values() for stamp in value.get("samples", [])}
    required_raw = {name: len(all_samples - stems) for name, stems in raw.items()}
    label_intersections = {name: len(stems & all_samples) for name, stems in labels.items()}

    occ_files = sorted((label_root / "occ").glob("*.npy"))
    occ_bad_shape = 0
    occ_bad_bounds = 0
    occ_bad_class = 0
    occ_rows = 0
    occ_classes: set[int] = set()
    for path in occ_files:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != 4 or arr.dtype.kind not in "iu":
            occ_bad_shape += 1
            continue
        occ_rows += int(arr.shape[0])
        coords, classes = arr[:, :3], arr[:, 3]
        occ_classes.update(int(x) for x in np.unique(classes))
        if np.any(coords < 0) or np.any(coords[:, 0] >= 256) or np.any(coords[:, 1] >= 256) or np.any(coords[:, 2] >= 32):
            occ_bad_bounds += 1
        if np.any((classes < 0) | (classes > 11)):
            occ_bad_class += 1

    elevation_files = sorted((label_root / "elevation").glob("*.png"))
    elevation_shapes = sorted({shape for path in elevation_files if (shape := _png_shape(path)) is not None})
    result = {
        "raw_root": str(raw_root),
        "label_root": str(label_root),
        "scene_count": len(scene_info),
        "sample_count": len(all_samples),
        "raw_file_counts": {name: len(stems) for name, stems in raw.items()},
        "label_file_counts": {name: len(stems) for name, stems in labels.items()},
        "required_raw_missing": required_raw,
        "label_intersections_with_scene_samples": label_intersections,
        "localmap_file_count": map_file_count,
        "localmap_intersection_with_scene_samples": len(maps & all_samples),
        "localmap_missing_for_scene_samples": len(all_samples - maps),
        "localmap_bad_float_or_nonfinite_files": map_bad_float_count,
        "localmap_point_count": {
            "min": min(map_point_stats) if map_point_stats else None,
            "median": float(np.median(map_point_stats)) if map_point_stats else None,
            "max": max(map_point_stats) if map_point_stats else None,
        },
        "localmap_coordinate_bounds": {
            "min": map_coordinate_min.tolist() if map_coordinate_min is not None else None,
            "max": map_coordinate_max.tolist() if map_coordinate_max is not None else None,
        },
        "occupancy": {
            "files": len(occ_files),
            "rows": occ_rows,
            "classes": sorted(occ_classes),
            "bad_shape": occ_bad_shape,
            "bad_bounds": occ_bad_bounds,
            "bad_class": occ_bad_class,
        },
        "elevation_png_shapes": [list(shape) for shape in elevation_shapes],
        "scene_join": joins,
    }
    result["passed_basic_integrity"] = bool(
        not joins["duplicate_samples"]
        and not any(required_raw.values())
        and not occ_bad_shape
        and not occ_bad_bounds
        and not occ_bad_class
        and not map_bad_float_count
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"))
    parser.add_argument(
        "--localmap-dir",
        type=Path,
        action="append",
        default=[],
        help="directory containing localmap_clouds/*.bin; repeat for both release archives",
    )
    parser.add_argument("--output", type=Path, default=Path("results/unscenes3d_mini_audit/report.json"))
    args = parser.parse_args()
    result = audit(args.raw_root, args.label_root, args.localmap_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("scene_count", "sample_count", "localmap_file_count", "localmap_intersection_with_scene_samples", "localmap_missing_for_scene_samples", "passed_basic_integrity")}, sort_keys=True))


if __name__ == "__main__":
    main()
