#!/usr/bin/env python3
"""Build a train/validation-only UnScenes3D event manifest.

Query geometry is generated from the frozen adapter using bounds and the published
label-validity mask.  Event labels are then computed with the exact ConPath
clearance/merge geometry.  The location-6 test site is deliberately excluded and
is not opened by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from pathrel.labels import clearance_radius_map, maximum_clearance_map
from pathrel.unscenes3d import deterministic_queries, occupancy_support


PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.1"
SPLIT_SITES = {
    "train": frozenset({"location_1", "location_2", "location_3"}),
    "validation": frozenset({"location_4_5"}),
}
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


def _event_labels(target_free: np.ndarray, starts: np.ndarray, goals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.empty((0, len(RADII_CELLS)), dtype=bool), np.empty((0,), dtype=np.int64)
    clearance = clearance_radius_map(target_free)
    # The frozen stencil has one start per frame, so one bottleneck propagation
    # serves all retained goals instead of repeating a full-map search per query.
    best = maximum_clearance_map(
        target_free,
        tuple(int(value) for value in starts[0]),
        clearance=clearance,
        stop_points=[tuple(int(value) for value in goal) for goal in goals],
    )
    max_clearance = best[goals[:, 0], goals[:, 1]].astype(np.int64, copy=False)
    labels = max_clearance[:, None] >= np.asarray(RADII_CELLS, dtype=np.int64)[None, :]
    return labels, max_clearance


def build_manifest(raw_root: Path, label_root: Path) -> dict[str, object]:
    scene_info = json.loads((raw_root / "imagesets" / "scene_info.json").read_text(encoding="utf-8"))
    records: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    split_summary: dict[str, object] = {}
    started = time.monotonic()
    for split, sites in SPLIT_SITES.items():
        counters = Counter()
        per_scene: dict[str, dict[str, object]] = {}
        for scene_id, scene in scene_info.items():
            location = str(scene.get("location", ""))
            if location not in sites:
                continue
            scene_counter = Counter()
            frame_count = 0
            for timestamp in scene.get("samples", []):
                occupancy_path = label_root / "occ" / f"{timestamp}.npy"
                if not occupancy_path.exists():
                    continue
                occupancy = np.load(occupancy_path, allow_pickle=False)
                support = occupancy_support(occupancy)
                starts, goals = deterministic_queries(
                    support.valid,
                    distances_cells=DISTANCES_CELLS,
                    angles_deg=ANGLES_DEG,
                )
                labels, max_clearance = _event_labels(support.free, starts, goals)
                frame_count += 1
                counters["frames"] += 1
                counters["queries"] += len(starts)
                counters["valid_cells"] += int(support.valid.sum())
                counters["free_cells"] += int(support.free.sum())
                row_queries: list[dict[str, object]] = []
                for index, (start, goal) in enumerate(zip(starts, goals)):
                    event = labels[index]
                    pattern = "".join("1" if value else "0" for value in event)
                    counters[f"event_pattern_{pattern}"] += 1
                    scene_counter[f"event_pattern_{pattern}"] += 1
                    row_queries.append(
                        {
                            "candidate_index": index,
                            "start": [int(start[0]), int(start[1])],
                            "goal": [int(goal[0]), int(goal[1])],
                            "max_clearance_cells": int(max_clearance[index]),
                            "reachable": [bool(value) for value in event],
                        }
                    )
                records[split].append(
                    {
                        "scene_id": scene_id,
                        "location": location,
                        "timestamp": str(timestamp),
                        "queries": row_queries,
                    }
                )
            per_scene[scene_id] = {
                "location": location,
                "frames": frame_count,
                "queries": sum(
                    len(record["queries"])
                    for record in records[split]
                    if record["scene_id"] == scene_id
                ),
                "event_patterns": dict(sorted(scene_counter.items())),
            }
        split_summary[split] = {
            "sites": sorted(sites),
            "scenes": sorted(per_scene),
            "scene_count": len(per_scene),
            "frames": counters["frames"],
            "queries": counters["queries"],
            "valid_cells": counters["valid_cells"],
            "free_cells": counters["free_cells"],
            "event_patterns": dict(
                sorted(
                    (key.removeprefix("event_pattern_"), value)
                    for key, value in counters.items()
                    if key.startswith("event_pattern_")
                )
            ),
            "per_scene": per_scene,
        }
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "grid_shape": [256, 256],
        "voxel_size_m": 0.3,
        "radii_cells": list(RADII_CELLS),
        "query_distances_cells": list(DISTANCES_CELLS),
        "query_angles_deg": list(ANGLES_DEG),
        "anchor_hint": [96, 128],
        "test_locked_sites": ["location_6"],
        "split_summary": split_summary,
        "records": records,
        "generation_seconds": time.monotonic() - started,
        "claim_boundary": "Train/validation event manifest only; location_6 test labels were not read.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"))
    parser.add_argument("--output", type=Path, default=Path("results/unscenes3d_contract_manifest/manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.raw_root, args.label_root)
    _atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train": manifest["split_summary"]["train"],
                "validation": manifest["split_summary"]["validation"],
                "test_locked_sites": manifest["test_locked_sites"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
