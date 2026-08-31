from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo

from pathrel.flatlands import (
    archive_structure_gate,
    audit_metadata,
    build_archive_index,
    integrity_gate,
    metadata_integrity_gate,
    provenance_split_gate,
)


PACKET_FILES = (
    "observed_floor.png",
    "floor_map.png",
    "unobserved.png",
    "epistemic_mask.png",
)


def add_packet(
    archive: ZipFile,
    split: str,
    observation: str,
    source: str,
    scene_id: str,
) -> None:
    root = f"FlatLands/{split}/{observation}"
    for filename in PACKET_FILES:
        archive.writestr(f"{root}/{filename}", b"png-placeholder")
    archive.writestr(
        f"{root}/metadata.json",
        json.dumps(
            {
                "scene": {
                    "dataset": source,
                    "scene_id": scene_id,
                    "metric_resolution": 0.05,
                },
                "observation": {"camera_pixel_position": [10, 20]},
                "provenance": {
                    "global_id": observation,
                    "original_split": split,
                },
            }
        ),
    )


class FlatLandsArchiveAuditTest(unittest.TestCase):
    def test_complete_packets_and_scene_disjointness_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.zip"
            with ZipFile(path, "w") as archive:
                add_packet(archive, "train", "obs_000001", "ScanNet", "train-scene")
                # The published archive uses `val/` even though its dataset card says validation.
                add_packet(archive, "val", "obs_000002", "ScanNet", "val-scene")
                add_packet(archive, "test", "obs_000003", "ScanNet++", "test-scene")
            with ZipFile(path) as archive:
                index = build_archive_index(archive.infolist())
                observations: list[dict[str, object]] = []
                metadata = audit_metadata(
                    archive, index.metadata_members, observation_callback=observations.append
                )

        self.assertEqual(index.report["packet_count"], 3)
        self.assertEqual(index.report["incomplete_packet_count"], 0)
        self.assertEqual(index.report["unsafe_member_count"], 0)
        self.assertTrue(metadata["complete_metadata_scan"])
        self.assertTrue(metadata["scene_disjoint"])
        self.assertTrue(metadata["provenance_scene_disjoint"])
        self.assertEqual(len(observations), 3)
        self.assertEqual(metadata["missing_global_id_count"], 0)
        self.assertEqual(metadata["provenance_missing_split_count"], 0)
        self.assertEqual(metadata["provenance_manifest_record_count"], 3)
        validation = next(row for row in observations if row["global_id"] == "obs_000002")
        self.assertEqual(validation["archive_split"], "validation")
        self.assertEqual(validation["provenance_split"], "validation")
        self.assertTrue(metadata_integrity_gate(metadata))
        self.assertTrue(provenance_split_gate(metadata))
        self.assertFalse(archive_structure_gate(index.report))
        # A tiny fixture intentionally cannot masquerade as the official release.
        self.assertFalse(integrity_gate(index.report, metadata))

    def test_cross_split_scene_leakage_fails_scene_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "leak.zip"
            with ZipFile(path, "w") as archive:
                add_packet(archive, "train", "obs_000001", "ZInD", "shared")
                add_packet(archive, "test", "obs_000002", "ZInD", "shared")
            with ZipFile(path) as archive:
                index = build_archive_index(archive.infolist())
                metadata = audit_metadata(archive, index.metadata_members)

        self.assertFalse(metadata["scene_disjoint"])
        self.assertFalse(metadata["provenance_scene_disjoint"])
        self.assertFalse(provenance_split_gate(metadata))
        self.assertEqual(metadata["scene_overlap"]["train__test"]["count"], 1)

    def test_missing_provenance_fields_fail_replayable_split_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-provenance.zip"
            with ZipFile(path, "w") as archive:
                add_packet(archive, "train", "obs_000001", "ScanNet", "scene")
                metadata_name = "FlatLands/train/obs_000002/metadata.json"
                for filename in PACKET_FILES:
                    archive.writestr(
                        f"FlatLands/train/obs_000002/{filename}", b"png-placeholder"
                    )
                archive.writestr(
                    metadata_name,
                    json.dumps(
                        {
                            "scene": {"dataset": "ScanNet", "scene_id": "other-scene"},
                            "provenance": {},
                        }
                    ),
                )
            with ZipFile(path) as archive:
                index = build_archive_index(archive.infolist())
                metadata = audit_metadata(archive, index.metadata_members)

        self.assertEqual(metadata["missing_global_id_count"], 1)
        self.assertEqual(metadata["provenance_missing_split_count"], 1)
        self.assertFalse(metadata["provenance_scene_disjoint"])
        self.assertFalse(metadata_integrity_gate(metadata))
        self.assertFalse(provenance_split_gate(metadata))

    def test_unsafe_and_incomplete_members_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("../escape.txt", b"unsafe")
                symlink = ZipInfo("FlatLands/link")
                symlink.create_system = 3
                symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(symlink, b"../../target")
                archive.writestr("FlatLands/train/obs_000001/metadata.json", b"{}")
            with ZipFile(path) as archive:
                index = build_archive_index(archive.infolist())

        self.assertEqual(index.report["unsafe_member_count"], 1)
        self.assertEqual(index.report["symlink_member_count"], 1)
        self.assertEqual(index.report["incomplete_packet_count"], 1)


if __name__ == "__main__":
    unittest.main()
