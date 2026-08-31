"""Streaming FlatLands packets and frozen natural-query replay.

The public archive is never extracted.  Dataset splitting is *only* keyed by the audited
``provenance_split`` column; ``archive_split`` merely locates a packet in the release ZIP and must
not be mistaken for the leaking official evaluation split.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Iterable, Sequence
from zipfile import ZipFile

import numpy as np

from .flatlands import (
    FLATLANDS_ARCHIVE_BYTES,
    FLATLANDS_SPLITS,
    canonical_flatlands_split,
)
from .flatlands_query import (
    ManifestObservation,
    NaturalQuery,
    construct_natural_queries,
    decode_binary_grayscale_png,
    load_provenance_manifest,
    sha256_path,
)


BOUNDED_SELECTION_SHA256 = (
    "4e7ae4c992cf943ab81618e3826c4748fcaaa97c3c4d7cb187518ee3fe6a9409"
)
BOUNDED_QUERIES_SHA256 = (
    "33e7f8a0343269b0dde47b428b3be622c80effdb0f80ae34b352ca282018d60d"
)
RETAINED_TARGET_STATUSES = frozenset(
    {"disconnected_radius_zero", "footprint_failure", "high_clearance_positive"}
)
_REACHABLE_FIELD = re.compile(r"^reachable_r([0-9]+)_cells$")


@dataclass(frozen=True)
class ReplayQuery:
    candidate_index: int
    distance_m: float
    angle_deg: int
    start_row: int | None
    start_col: int | None
    goal_row: int | None
    goal_col: int | None
    selection_status: str
    target_status: str
    max_clearance_cells: int | None
    reachable: tuple[bool | None, ...]

    @property
    def retained(self) -> bool:
        return self.target_status in RETAINED_TARGET_STATUSES


@dataclass(frozen=True)
class FlatLandsReplaySample:
    observation: ManifestObservation
    observed_floor: np.ndarray
    floor_map: np.ndarray
    unobserved: np.ndarray
    epistemic_mask: np.ndarray
    radii_cells: tuple[int, ...]
    queries: tuple[ReplayQuery, ...]

    @property
    def input_bev(self) -> np.ndarray:
        """Model input ``[3,H,W]``: observed free, hidden-region mask, valid-support mask."""

        return np.stack(
            (self.observed_floor, self.unobserved, self.epistemic_mask), axis=0
        ).astype(np.float32, copy=False)

    @property
    def target_free(self) -> np.ndarray:
        """Binary support target with invalid epistemic cells explicitly excluded."""

        return self.floor_map & self.epistemic_mask

    @property
    def loss_mask(self) -> np.ndarray:
        """Official completion region restricted to valid epistemic support."""

        return self.unobserved & self.epistemic_mask

    @property
    def retained_queries(self) -> tuple[ReplayQuery, ...]:
        return tuple(query for query in self.queries if query.retained)


def _optional_int(value: str, *, field: str, line_number: int) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"invalid {field} at query row {line_number}: {value!r}") from error


def _optional_bool(value: str, *, field: str, line_number: int) -> bool | None:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid {field} at query row {line_number}: {value!r}")


def load_bounded_query_manifest(
    path: Path,
) -> tuple[tuple[int, ...], dict[str, tuple[ReplayQuery, ...]], dict[str, dict[str, str]]]:
    """Load and structurally validate the frozen query CSV.

    The third return value contains immutable identity fields for cross-checking against the
    selected-observation manifest.
    """

    required = {
        "global_id",
        "provenance_split",
        "source_dataset",
        "scene_id",
        "resolution_m",
        "candidate_index",
        "distance_m",
        "angle_deg",
        "start_row",
        "start_col",
        "goal_row",
        "goal_col",
        "selection_status",
        "target_status",
        "max_clearance_cells",
    }
    query_lists: dict[str, list[ReplayQuery]] = {}
    identities: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = required - set(fields)
        if missing:
            raise ValueError(f"query manifest is missing columns: {sorted(missing)}")
        radius_fields = sorted(
            (
                (int(match.group(1)), field)
                for field in fields
                if (match := _REACHABLE_FIELD.match(field)) is not None
            ),
            key=lambda item: item[0],
        )
        radii = tuple(radius for radius, _ in radius_fields)
        if not radii or radii[0] != 0 or tuple(sorted(set(radii))) != radii:
            raise ValueError("query manifest radii must be unique, increasing, and begin at zero")

        for line_number, row in enumerate(reader, start=2):
            global_id = row["global_id"]
            if not global_id:
                raise ValueError(f"missing global_id at query row {line_number}")
            identity = {
                key: row[key]
                for key in (
                    "provenance_split",
                    "source_dataset",
                    "scene_id",
                    "resolution_m",
                )
            }
            previous_identity = identities.setdefault(global_id, identity)
            if previous_identity != identity:
                raise ValueError(f"inconsistent identity fields for {global_id}")
            try:
                query = ReplayQuery(
                    candidate_index=int(row["candidate_index"]),
                    distance_m=float(row["distance_m"]),
                    angle_deg=int(row["angle_deg"]),
                    start_row=_optional_int(
                        row["start_row"], field="start_row", line_number=line_number
                    ),
                    start_col=_optional_int(
                        row["start_col"], field="start_col", line_number=line_number
                    ),
                    goal_row=_optional_int(
                        row["goal_row"], field="goal_row", line_number=line_number
                    ),
                    goal_col=_optional_int(
                        row["goal_col"], field="goal_col", line_number=line_number
                    ),
                    selection_status=row["selection_status"],
                    target_status=row["target_status"],
                    max_clearance_cells=_optional_int(
                        row["max_clearance_cells"],
                        field="max_clearance_cells",
                        line_number=line_number,
                    ),
                    reachable=tuple(
                        _optional_bool(row[field], field=field, line_number=line_number)
                        for _, field in radius_fields
                    ),
                )
            except ValueError as error:
                if str(error).startswith("invalid"):
                    raise
                raise ValueError(f"invalid query row {line_number}: {error}") from error
            query_lists.setdefault(global_id, []).append(query)

    if not query_lists:
        raise ValueError("query manifest contains no rows")
    output: dict[str, tuple[ReplayQuery, ...]] = {}
    for global_id, rows in query_lists.items():
        ordered = tuple(sorted(rows, key=lambda item: item.candidate_index))
        indices = tuple(query.candidate_index for query in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError(f"query candidate indices are not contiguous for {global_id}")
        for query in ordered:
            if query.retained:
                if any(value is None for value in query.reachable):
                    raise ValueError(f"retained query has missing event labels for {global_id}")
                labels = tuple(bool(value) for value in query.reachable)
                if any(labels[index] and not labels[index - 1] for index in range(1, len(labels))):
                    raise ValueError(f"reachability is not radius-monotone for {global_id}")
            elif any(value is not None for value in query.reachable):
                raise ValueError(f"non-retained query unexpectedly has event labels for {global_id}")
        output[global_id] = ordered
    return radii, output, identities


def _unique_in_order(values: Iterable[object]) -> tuple[object, ...]:
    return tuple(dict.fromkeys(values))


def _assert_query_geometry(
    observation: ManifestObservation,
    observed_floor: np.ndarray,
    unobserved: np.ndarray,
    epistemic_mask: np.ndarray,
    replay: Sequence[ReplayQuery],
) -> None:
    distances = tuple(float(value) for value in _unique_in_order(q.distance_m for q in replay))
    angles = tuple(int(value) for value in _unique_in_order(q.angle_deg for q in replay))
    reconstructed = construct_natural_queries(
        observed_floor,
        unobserved,
        epistemic_mask,
        camera_px=observation.camera_px,
        resolution_m=observation.resolution,
        distances_m=distances,
        angles_deg=angles,
    )
    if len(reconstructed) != len(replay):
        raise ValueError(f"query replay count mismatch for {observation.global_id}")
    for expected, actual in zip(replay, reconstructed):
        comparable = (
            "candidate_index",
            "distance_m",
            "angle_deg",
            "start_row",
            "start_col",
            "goal_row",
            "goal_col",
            "selection_status",
        )
        for field in comparable:
            if getattr(expected, field) != getattr(actual, field):
                raise ValueError(
                    f"query replay mismatch for {observation.global_id} candidate "
                    f"{expected.candidate_index}: {field}"
                )


class FlatLandsReplayDataset:
    """Map-style, process-safe reader for the frozen bounded FlatLands benchmark.

    PyTorch's ``DataLoader`` accepts this object through the ordinary ``__len__``/``__getitem__``
    protocol.  A ZIP handle is opened lazily per process and removed during pickling.
    """

    def __init__(
        self,
        archive_path: Path,
        selection_path: Path,
        query_path: Path,
        *,
        split: str | None = None,
        sources: Iterable[str] | None = None,
        verify_frozen: bool = True,
        verify_query_geometry: bool = True,
    ) -> None:
        self.archive_path = Path(archive_path).resolve()
        self.selection_path = Path(selection_path).resolve()
        self.query_path = Path(query_path).resolve()
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        if not self.selection_path.is_file():
            raise FileNotFoundError(self.selection_path)
        if not self.query_path.is_file():
            raise FileNotFoundError(self.query_path)
        if verify_frozen:
            if self.archive_path.stat().st_size != FLATLANDS_ARCHIVE_BYTES:
                raise ValueError("FlatLands archive byte count does not match the frozen release")
            if sha256_path(self.selection_path) != BOUNDED_SELECTION_SHA256:
                raise ValueError("selected-observation manifest SHA-256 mismatch")
            if sha256_path(self.query_path) != BOUNDED_QUERIES_SHA256:
                raise ValueError("bounded query manifest SHA-256 mismatch")

        canonical_split = None if split is None else canonical_flatlands_split(split)
        if split is not None and canonical_split is None:
            raise ValueError(f"unknown provenance split: {split!r}")
        source_filter = None if sources is None else frozenset(str(value) for value in sources)
        if source_filter is not None and not source_filter:
            raise ValueError("sources cannot be empty")

        observations = load_provenance_manifest(self.selection_path)
        radii, queries, identities = load_bounded_query_manifest(self.query_path)
        observation_ids = {row.global_id for row in observations}
        if observation_ids != set(queries):
            missing_queries = sorted(observation_ids - set(queries))[:20]
            extra_queries = sorted(set(queries) - observation_ids)[:20]
            raise ValueError(
                "selection/query global IDs differ: "
                f"missing_queries={missing_queries}, extra_queries={extra_queries}"
            )
        for row in observations:
            identity = identities[row.global_id]
            if (
                identity["provenance_split"] != row.provenance_split
                or identity["source_dataset"] != row.source_dataset
                or identity["scene_id"] != row.scene_id
                or float(identity["resolution_m"]) != row.resolution
            ):
                raise ValueError(f"selection/query identity mismatch for {row.global_id}")

        self.radii_cells = radii
        self._queries = queries
        self._observations = tuple(
            row
            for row in observations
            if (canonical_split is None or row.provenance_split == canonical_split)
            and (source_filter is None or row.source_dataset in source_filter)
        )
        if not self._observations:
            raise ValueError("no observations match the requested provenance split/source filters")
        self.verify_query_geometry = bool(verify_query_geometry)
        self._archive: ZipFile | None = None
        self._archive_pid: int | None = None

    @property
    def observations(self) -> tuple[ManifestObservation, ...]:
        return self._observations

    def __len__(self) -> int:
        return len(self._observations)

    def _zip(self) -> ZipFile:
        process_id = os.getpid()
        if self._archive is None or self._archive_pid != process_id:
            self.close()
            self._archive = ZipFile(self.archive_path)
            self._archive_pid = process_id
        return self._archive

    def close(self) -> None:
        archive = getattr(self, "_archive", None)
        if archive is not None:
            archive.close()
        self._archive = None
        self._archive_pid = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_archive"] = None
        state["_archive_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> FlatLandsReplaySample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self._observations[index]
        archive = self._zip()
        root = row.packet_directory

        def read_map(filename: str) -> np.ndarray:
            try:
                payload = archive.read(f"{root}/{filename}")
            except KeyError as error:
                raise ValueError(f"missing {filename} for {row.global_id}") from error
            return decode_binary_grayscale_png(payload)

        observed_floor = read_map("observed_floor.png")
        unobserved = read_map("unobserved.png")
        epistemic_mask = read_map("epistemic_mask.png")
        floor_map = read_map("floor_map.png")
        if any(
            array.shape != observed_floor.shape
            for array in (unobserved, epistemic_mask, floor_map)
        ):
            raise ValueError(f"misaligned packet maps for {row.global_id}")
        if np.any(observed_floor & ~floor_map):
            raise ValueError(f"observed floor disagrees with target for {row.global_id}")
        if np.any(observed_floor & unobserved):
            raise ValueError(f"observed/unobserved masks overlap for {row.global_id}")

        try:
            metadata = json.loads(archive.read(row.metadata_member))
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid metadata for {row.global_id}: {error}") from error
        scene = metadata.get("scene", {})
        provenance = metadata.get("provenance", {})
        observation_metadata = metadata.get("observation", {})
        if (
            str(provenance.get("global_id")) != row.global_id
            or canonical_flatlands_split(provenance.get("original_split"))
            != row.provenance_split
            or str(scene.get("dataset")) != row.source_dataset
            or str(scene.get("scene_id")) != row.scene_id
            or float(scene.get("resolution")) != row.resolution
            or tuple(int(value) for value in observation_metadata.get("camera_px", ()))
            != row.camera_px
        ):
            raise ValueError(f"archive metadata disagrees with frozen manifest for {row.global_id}")

        replay = self._queries[row.global_id]
        if self.verify_query_geometry:
            _assert_query_geometry(
                row, observed_floor, unobserved, epistemic_mask, replay
            )
        return FlatLandsReplaySample(
            observation=row,
            observed_floor=observed_floor,
            floor_map=floor_map,
            unobserved=unobserved,
            epistemic_mask=epistemic_mask,
            radii_cells=self.radii_cells,
            queries=replay,
        )


def collate_flatlands_replay(
    samples: Sequence[FlatLandsReplaySample],
) -> dict[str, object]:
    """Pad retained queries while preserving a mask for proper per-query weighting."""

    if not samples:
        raise ValueError("cannot collate an empty FlatLands batch")
    radii = samples[0].radii_cells
    shape = samples[0].observed_floor.shape
    for sample in samples:
        if sample.radii_cells != radii or sample.observed_floor.shape != shape:
            raise ValueError("FlatLands batch samples must share map shape and radii")
    retained = [sample.retained_queries for sample in samples]
    maximum_queries = max((len(rows) for rows in retained), default=0)
    batch_size = len(samples)
    starts = np.zeros((batch_size, maximum_queries, 2), dtype=np.int64)
    goals = np.zeros_like(starts)
    targets = np.zeros((batch_size, maximum_queries, len(radii)), dtype=bool)
    query_mask = np.zeros((batch_size, maximum_queries), dtype=bool)
    distances_m = np.zeros((batch_size, maximum_queries), dtype=np.float32)
    angles_deg = np.zeros((batch_size, maximum_queries), dtype=np.int64)
    for batch_index, rows in enumerate(retained):
        for query_index, query in enumerate(rows):
            assert query.start_row is not None and query.start_col is not None
            assert query.goal_row is not None and query.goal_col is not None
            starts[batch_index, query_index] = (query.start_row, query.start_col)
            goals[batch_index, query_index] = (query.goal_row, query.goal_col)
            targets[batch_index, query_index] = tuple(bool(value) for value in query.reachable)
            query_mask[batch_index, query_index] = True
            distances_m[batch_index, query_index] = query.distance_m
            angles_deg[batch_index, query_index] = query.angle_deg
    return {
        "observation": np.stack([sample.input_bev for sample in samples]),
        "target_free": np.stack([sample.target_free for sample in samples]),
        "loss_mask": np.stack([sample.loss_mask for sample in samples]),
        "starts": starts,
        "goals": goals,
        "reachability_targets": targets,
        "query_mask": query_mask,
        "distances_m": distances_m,
        "angles_deg": angles_deg,
        "radii_cells": np.asarray(radii, dtype=np.int64),
        "global_ids": [sample.observation.global_id for sample in samples],
        "provenance_splits": [sample.observation.provenance_split for sample in samples],
        "source_datasets": [sample.observation.source_dataset for sample in samples],
        "scene_ids": [sample.observation.scene_id for sample in samples],
    }
