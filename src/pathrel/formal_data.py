"""Frozen physical-train-only packets shared by the formal external comparison.

The external condition is [observed free, unknown, supplied valid support].
The legacy ConPath observation [free, blocked, unknown] is a lossless encoding
of exactly the same information. No complete target enters either encoding.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image

from .data_access import require_development_packet


SOURCES = ("3RScan", "ARKitScenes", "Matterport3D", "ScanNet", "ZInD")
SPLITS = ("train", "calibration", "validation")
QUOTAS = {
    "train": dict(zip(SOURCES, (232, 232, 72, 232, 232))),
    "calibration": dict(zip(SOURCES, (23, 23, 8, 23, 23))),
    "validation": dict(zip(SOURCES, (46, 46, 8, 46, 46))),
}
VIEWS_PER_PARENT = {"train": 2, "calibration": 1, "validation": 1}
RADII_CELLS = (0, 10, 20)
DISTANCES_CELLS = (40, 80, 120)
ANGLES_DEG = tuple(range(0, 360, 30))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*values):
    return hashlib.sha256("|".join(map(str, ("flatlands-external-formal-v1", *values))).encode()).hexdigest()


def validate_formal_partition(rows):
    ids, parent_splits = set(), {}
    for row in rows:
        require_development_packet(row)
        gid = row["global_id"]
        if not re.fullmatch(r"obs_[0-9]{6}", gid):
            raise ValueError("Invalid observation identifier")
        if (row["archive_split"] != "train" or row["packet_directory"] != "train/" + gid
                or row["metadata_member"] != "train/" + gid + "/metadata.json"):
            raise ValueError("Formal packets must physically belong to archive train")
        if gid in ids:
            raise ValueError("Duplicate selected observation")
        ids.add(gid)
        parent = row["parent_group"]
        if parent in parent_splits and parent_splits[parent] != row["candidate_split"]:
            raise ValueError("Parent place crosses formal partitions")
        parent_splits[parent] = row["candidate_split"]


def choose_formal_subset(rows, quotas=None, views_per_parent=None):
    """Metadata-only capacity-capped selection; never move an existing partition."""
    quotas = QUOTAS if quotas is None else quotas
    views_per_parent = VIEWS_PER_PARENT if views_per_parent is None else views_per_parent
    validate_formal_partition(rows)
    grouped = {}
    for row in rows:
        key = (row["candidate_split"], row["source_dataset"], row["parent_group"])
        grouped.setdefault(key, []).append(row)
    selected, excluded, counts = [], [], {}
    for split, source_counts in quotas.items():
        counts[split] = {}
        for source, count in source_counts.items():
            candidates = {key[2]: group for key, group in grouped.items() if key[:2] == (split, source)}
            eligible = {}
            for parent, group in candidates.items():
                if len(group) < views_per_parent[split]:
                    excluded.append({"parent_group": parent, "source": source, "split": split,
                                     "observations": len(group), "reason": "insufficient_metadata_views"})
                else:
                    eligible[parent] = group
            counts[split][source] = {"candidate_parents": len(candidates), "eligible_parents": len(eligible),
                                     "selected_parents": count, "views_per_parent": views_per_parent[split]}
            if len(eligible) < count:
                raise ValueError(f"Insufficient independent parents: {source}/{split}")
            parents = sorted(eligible, key=lambda p: (stable_rank("parent", split, source, p), p))[:count]
            for parent in parents:
                views = sorted(eligible[parent], key=lambda r: (stable_rank("view", parent, r["global_id"]), r["global_id"]))
                for index, row in enumerate(views[:views_per_parent[split]]):
                    selected.append({**row, "selection_view_index": str(index)})
    validate_formal_partition(selected)
    return selected, {"counts": counts, "metadata_exclusions": sorted(excluded, key=lambda r: r["parent_group"])}


def construct_cell_queries(observed, hidden, valid, camera_px,
                           distances_cells=DISTANCES_CELLS, angles_deg=ANGLES_DEG):
    """Input-only polar queries, with geometry explicitly expressed in grid cells."""
    observed, hidden, valid = (np.asarray(a, dtype=bool) for a in (observed, hidden, valid))
    if observed.ndim != 2 or hidden.shape != observed.shape or valid.shape != observed.shape:
        raise ValueError("Input maps must share a two-dimensional shape")
    if (len(camera_px) != 2 or any(not math.isfinite(float(v)) for v in camera_px)
            or any(float(v) != int(v) for v in camera_px)):
        raise ValueError("Camera must be an integer [x,y] grid coordinate")
    if not distances_cells or any(not math.isfinite(float(v)) or v <= 0 for v in distances_cells):
        raise ValueError("Query distances must be positive finite grid-cell counts")
    height, width = observed.shape
    camera = (int(camera_px[1]), int(camera_px[0]))
    start = None
    if 0 <= camera[0] < height and 0 <= camera[1] < width:
        candidates = np.argwhere(observed & valid)
        if len(candidates):
            squared_distance = np.sum((candidates - np.asarray(camera)) ** 2, axis=1)
            order = np.lexsort((candidates[:, 1], candidates[:, 0], squared_distance))
            start = tuple(map(int, candidates[int(order[0])]))
    queries, seen = [], set()
    for distance in distances_cells:
        for angle in angles_deg:
            query = {"candidate_index": len(queries), "distance_cells": int(distance), "angle_deg": int(angle),
                     "start_row": None, "start_col": None, "goal_row": None, "goal_col": None}
            status = "no_observed_valid_start"
            if start is not None:
                theta = math.radians(angle)
                goal = (start[0] + int(round(math.sin(theta) * distance)),
                        start[1] + int(round(math.cos(theta) * distance)))
                query.update(start_row=start[0], start_col=start[1], goal_row=goal[0], goal_col=goal[1])
                status = "selected"
                if not (0 <= goal[0] < height and 0 <= goal[1] < width):
                    status = "goal_out_of_bounds"
                elif goal == start:
                    status = "goal_equals_start"
                elif not valid[goal]:
                    status = "goal_outside_epistemic_mask"
                elif not hidden[goal]:
                    status = "goal_not_unobserved"
                elif goal in seen:
                    status = "duplicate_goal"
                if status == "selected":
                    seen.add(goal)
            query["selection_status"] = status
            queries.append(query)
    return queries


def decode_binary_png(blob):
    """Pillow decoding with strict 8-bit grayscale PNG header, CRC and binary values."""
    if len(blob) < 33 or blob[:8] != b"\x89PNG\r\n\x1a\n" or blob[12:16] != b"IHDR":
        raise ValueError("Expected PNG with an IHDR header")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", blob[16:29])
    if (depth, color, compression, filtering, interlace) != (8, 0, 0, 0, 0):
        raise ValueError("Expected non-interlaced 8-bit grayscale PNG")
    with Image.open(io.BytesIO(blob)) as image:
        image.verify()
    with Image.open(io.BytesIO(blob)) as image:
        if image.mode != "L" or image.size != (width, height):
            raise ValueError("PNG decoder disagrees with image header")
        array = np.array(image, dtype=np.uint8)
    if not np.all((array == 0) | (array == 255)):
        raise ValueError("Binary PNG may only contain 0 and 255")
    return array == 255


@dataclass
class FormalSample:
    row: dict
    observation: np.ndarray
    valid: np.ndarray
    target: np.ndarray
    hidden: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    targets: np.ndarray
    candidate_indices: np.ndarray

    @property
    def condition(self):
        return np.stack((self.observation[0], self.hidden, self.valid)).astype(np.float32)

    @property
    def scene_key(self):
        return self.row["parent_group"]


def load_formal_data(root, split):
    if split not in SPLITS:
        raise ValueError("Only train/calibration/validation are allowed; no final-test loader exists")
    root = Path(root)
    seal = json.loads((root / "seal.json").read_text())
    for relative in ("selected.csv", "data_protocol.json", "data_audit.json", "input_queries_before_target.jsonl"):
        if sha256(root / relative) != seal["files"][relative]:
            raise ValueError(f"Frozen formal data changed: {relative}")
    audit = json.loads((root / "data_audit.json").read_text())
    if not audit["passed"] or audit["new_physical_test_images_opened"] != 0:
        raise ValueError("Formal data gate did not pass")
    rows = list(csv.DictReader((root / "selected.csv").open()))
    validate_formal_partition(rows)
    result = []
    fields = ("observation", "valid", "target", "hidden", "starts", "goals", "targets", "candidate_indices")
    for row in rows:
        if row["candidate_split"] != split:
            continue
        relative = "packets/" + row["global_id"] + ".npz"
        if sha256(root / relative) != seal["files"][relative]:
            raise ValueError("Frozen packet checksum changed")
        with np.load(root / relative, allow_pickle=False) as packet:
            result.append(FormalSample(row, **{key: packet[key] for key in fields}))
    if not result:
        raise ValueError("Empty formal development split")
    return result


def collate_formal_data(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    count = max(len(s.starts) for s in samples)
    starts = np.zeros((len(samples), count, 2), dtype=np.int64)
    goals = starts.copy()
    targets = np.zeros((len(samples), count, len(RADII_CELLS)), dtype=bool)
    mask = np.zeros((len(samples), count), dtype=bool)
    candidate_indices = np.full((len(samples), count), -1, dtype=np.int64)
    for i, sample in enumerate(samples):
        n = len(sample.starts)
        if n:
            starts[i] = sample.starts[0]
        starts[i, :n], goals[i, :n], targets[i, :n], mask[i, :n] = sample.starts, sample.goals, sample.targets, True
        candidate_indices[i, :n] = sample.candidate_indices
    return {"observation": np.stack([s.observation for s in samples]), "condition": np.stack([s.condition for s in samples]),
            "valid_support_mask": np.stack([s.valid for s in samples]), "target_free": np.stack([s.target for s in samples]),
            "loss_mask": np.stack([s.hidden for s in samples]), "starts": starts, "goals": goals,
            "reachability_targets": targets, "query_mask": mask, "candidate_indices": candidate_indices,
            "parent_group": [s.scene_key for s in samples], "global_id": [s.row["global_id"] for s in samples]}
