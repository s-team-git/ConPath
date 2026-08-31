from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
import struct
import tempfile
import unittest
from zipfile import ZipFile
import zlib

import numpy as np

from pathrel.flatlands_data import FlatLandsReplayDataset, collate_flatlands_replay
from pathrel.flatlands_query import construct_natural_queries, score_natural_queries


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _binary_png(array: np.ndarray) -> bytes:
    image = np.where(np.asarray(array, dtype=bool), 255, 0).astype(np.uint8)
    height, width = image.shape
    raster = b"".join(b"\x00" + row.tobytes() for row in image)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raster)),
            _png_chunk(b"IEND", b""),
        )
    )


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    archive_path = root / "tiny.zip"
    selection_path = root / "selected.csv"
    query_path = root / "queries.csv"
    packet = "train/obs_fixture"
    shape = (32, 32)
    observed = np.zeros(shape, dtype=bool)
    observed[15:18, 15:18] = True
    unobserved = ~observed
    epistemic = np.ones(shape, dtype=bool)
    floor = np.ones(shape, dtype=bool)
    floor[:, 20] = False
    floor[16, 20] = True
    # One valid non-floor cell is observed rather than hidden, exercising the canonical blocked
    # input channel. The remaining wall is left unobserved for the completion/query task.
    unobserved[0, 20] = False
    metadata = {
        "scene": {
            "dataset": "ScanNet",
            "scene_id": "scene-fixture",
            "resolution": 0.1,
        },
        "observation": {"camera_px": [16, 16]},
        "provenance": {
            "global_id": "obs_fixture",
            "original_split": "test",
        },
    }
    with ZipFile(archive_path, "w") as archive:
        for name, array in (
            ("observed_floor.png", observed),
            ("floor_map.png", floor),
            ("unobserved.png", unobserved),
            ("epistemic_mask.png", epistemic),
        ):
            archive.writestr(f"{packet}/{name}", _binary_png(array))
        archive.writestr(f"{packet}/metadata.json", json.dumps(metadata))

    selection_fields = (
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
    )
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields)
        writer.writeheader()
        writer.writerow(
            {
                "global_id": "obs_fixture",
                "provenance_split": "test",
                # This deliberately proves that filtering does not use the physical archive split.
                "archive_split": "train",
                "source_dataset": "ScanNet",
                "scene_id": "scene-fixture",
                "packet_directory": packet,
                "metadata_member": f"{packet}/metadata.json",
                "original_observation_id": "obs_000",
                "quality_category": "LEARNABLE",
                "resolution": "0.1",
                "camera_px": "[16,16]",
            }
        )

    natural = construct_natural_queries(
        observed,
        unobserved,
        epistemic,
        camera_px=(16, 16),
        resolution_m=0.1,
    )
    scored = score_natural_queries(floor, epistemic, natural, radii_cells=(0, 1, 2))
    query_fields = (
        "global_id",
        "provenance_split",
        "source_dataset",
        "scene_id",
        "resolution_m",
        *scored[0].keys(),
    )
    with query_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        writer.writeheader()
        for record in scored:
            writer.writerow(
                {
                    "global_id": "obs_fixture",
                    "provenance_split": "test",
                    "source_dataset": "ScanNet",
                    "scene_id": "scene-fixture",
                    "resolution_m": "0.1",
                    **record,
                }
            )
    return archive_path, selection_path, query_path


class FlatLandsReplayDatasetTest(unittest.TestCase):
    def test_streams_packet_by_provenance_split_and_collates_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, selection, queries = _write_fixture(Path(temporary))
            dataset = FlatLandsReplayDataset(
                archive,
                selection,
                queries,
                split="test",
                verify_frozen=False,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.observations[0].archive_split, "train")
            self.assertEqual(dataset.observations[0].provenance_split, "test")
            sample = dataset[0]
            self.assertEqual(sample.input_bev.shape, (3, 32, 32))
            self.assertTrue(sample.observed_free[16, 16])
            self.assertTrue(sample.observed_blocked[0, 20])
            self.assertFalse(sample.unknown[0, 20])
            self.assertTrue(sample.unknown[1, 20])
            self.assertTrue(np.array_equal(sample.input_bev[0] > 0.5, sample.observed_free))
            self.assertTrue(np.array_equal(sample.input_bev[1] > 0.5, sample.observed_blocked))
            self.assertTrue(np.array_equal(sample.input_bev[2] > 0.5, sample.unknown))
            self.assertTrue(
                np.array_equal(
                    sample.observed_free | sample.observed_blocked | sample.unknown,
                    sample.epistemic_mask,
                )
            )
            self.assertEqual(sample.target_free.shape, (32, 32))
            self.assertTrue(np.all(~sample.loss_mask | sample.epistemic_mask))
            self.assertEqual(sample.radii_cells, (0, 1, 2))
            self.assertGreater(len(sample.retained_queries), 0)

            batch = collate_flatlands_replay([sample, sample])
            self.assertEqual(batch["observation"].shape, (2, 3, 32, 32))
            self.assertEqual(batch["reachability_targets"].shape[0], 2)
            self.assertEqual(batch["reachability_targets"].shape[-1], 3)
            self.assertTrue(np.all(batch["query_mask"].sum(axis=1) > 0))
            self.assertTrue(np.all(batch["candidate_indices"][batch["query_mask"]] >= 0))
            self.assertTrue(np.all(batch["candidate_indices"][~batch["query_mask"]] == -1))

            # A live ZipFile handle never enters a spawned/forked DataLoader payload.
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored[0].observation.global_id, "obs_fixture")
            dataset.close()
            restored.close()

    def test_does_not_fall_back_to_leaking_archive_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, selection, queries = _write_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "no observations"):
                FlatLandsReplayDataset(
                    archive,
                    selection,
                    queries,
                    split="train",
                    verify_frozen=False,
                )

    def test_query_geometry_tampering_is_detected_on_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, selection, queries = _write_fixture(Path(temporary))
            rows: list[dict[str, str]] = []
            with queries.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = tuple(reader.fieldnames or ())
                rows = list(reader)
            selected = next(row for row in rows if row["selection_status"] == "selected")
            selected["goal_col"] = str(int(selected["goal_col"]) + 1)
            with queries.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            dataset = FlatLandsReplayDataset(
                archive,
                selection,
                queries,
                verify_frozen=False,
            )
            with self.assertRaisesRegex(ValueError, "query replay mismatch"):
                _ = dataset[0]
            dataset.close()


if __name__ == "__main__":
    unittest.main()
