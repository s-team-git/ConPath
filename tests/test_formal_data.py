import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pathrel.formal_data import (FormalSample, choose_formal_subset, collate_formal_data,
                                construct_cell_queries, decode_binary_png, load_formal_data,
                                validate_formal_partition)
from pathrel.flatlands_query import construct_natural_queries, decode_binary_grayscale_png
from scripts.prepare_flatlands_formal_data import prepare


class FormalDataTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(global_id=f"obs_{i:06d}", parent_group=f"ScanNet:p{i//2}", source_dataset="ScanNet",
                     archive_split="train", packet_directory=f"train/obs_{i:06d}",
                     metadata_member=f"train/obs_{i:06d}/metadata.json", candidate_split=split)
                for split, first in (("train", 0), ("calibration", 20), ("validation", 40))
                for i in range(first, first + 12)]

    def test_metadata_selection_keeps_parent_partition_and_fixed_views(self):
        quotas = {"train": {"ScanNet": 3}, "calibration": {"ScanNet": 2}, "validation": {"ScanNet": 2}}
        selected, _ = choose_formal_subset(self.rows(), quotas)
        reversed_selected, _ = choose_formal_subset(list(reversed(self.rows())), quotas)
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({r["parent_group"] for r in selected}), 7)
        for parent in {r["parent_group"] for r in selected if r["candidate_split"] == "train"}:
            self.assertEqual(sum(r["parent_group"] == parent for r in selected), 2)

    def test_single_view_parent_exclusion_is_metadata_only(self):
        rows = self.rows()[1:]
        _, report = choose_formal_subset(rows, {"train": {"ScanNet": 3}})
        self.assertEqual(report["metadata_exclusions"], [{"parent_group": "ScanNet:p0", "source": "ScanNet",
                          "split": "train", "observations": 1, "reason": "insufficient_metadata_views"}])

    def test_reassigned_test_and_metadata_path_escape_fail_closed(self):
        for key, value in (("archive_split", "test"), ("packet_directory", "test/obs_000000"),
                           ("metadata_member", "train/obs_000000/../../test/metadata.json"),
                           ("global_id", "../test")):
            rows = self.rows()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                validate_formal_partition(rows)

    def test_parent_rescan_cannot_cross_splits(self):
        rows = self.rows()
        rows[-1]["parent_group"] = rows[0]["parent_group"]
        with self.assertRaises(ValueError):
            validate_formal_partition(rows)

    def test_cell_queries_reproduce_existing_geometry_without_metric_conversion(self):
        rng = np.random.default_rng(401)
        for _ in range(5):
            valid = rng.random((256, 256)) > .15
            observed = (rng.random((256, 256)) > .99) & valid
            unknown = valid & ~observed
            actual = construct_cell_queries(observed, unknown, valid, [128, 192])
            expected = construct_natural_queries(observed, unknown, valid, camera_px=[128, 192], resolution_m=.01)
            for a, b in zip(actual, expected):
                self.assertEqual(a["distance_cells"], round(b.distance_m / .01))
                for key in ("candidate_index", "start_row", "start_col", "goal_row", "goal_col", "selection_status"):
                    self.assertEqual(a[key], getattr(b, key))

    def test_input_only_query_keeps_goal_without_any_target_argument(self):
        observed = np.zeros((9, 9), bool)
        observed[4, 4] = True
        valid = np.ones_like(observed)
        queries = construct_cell_queries(observed, ~observed, valid, [4, 4], distances_cells=[2], angles_deg=[0, 180])
        self.assertEqual([(q["goal_row"], q["goal_col"]) for q in queries], [(4, 6), (4, 2)])
        self.assertTrue(all(q["selection_status"] == "selected" for q in queries))

    def test_strict_fast_png_matches_existing_decoder_and_rejects_soft_pixels(self):
        rng = np.random.default_rng(412)
        original = rng.integers(0, 2, (31, 29), dtype=np.uint8) * 255
        handle = io.BytesIO()
        Image.fromarray(original).save(handle, format="PNG")
        blob = handle.getvalue()
        np.testing.assert_array_equal(decode_binary_png(blob), decode_binary_grayscale_png(blob))
        original[0, 0] = 128
        handle = io.BytesIO()
        Image.fromarray(original).save(handle, format="PNG")
        with self.assertRaises(ValueError):
            decode_binary_png(handle.getvalue())

    def test_condition_is_lossless_and_padded_query_has_shared_start(self):
        valid = np.ones((8, 8), bool)
        valid[0] = False
        free = np.zeros_like(valid)
        free[2, 2] = True
        blocked = np.zeros_like(valid)
        blocked[3, 3] = True
        hidden = valid & ~free & ~blocked
        sample = FormalSample({"global_id": "obs_000001", "parent_group": "ScanNet:p1"},
                              np.stack((free, blocked, hidden)).astype(np.float32), valid, free, hidden,
                              np.array([[2, 2], [2, 2]]), np.array([[4, 4], [5, 5]]),
                              np.ones((2, 3), bool), np.array([3, 9]))
        short = FormalSample(sample.row, sample.observation, valid, free, hidden,
                             sample.starts[:1], sample.goals[:1], sample.targets[:1], sample.candidate_indices[:1])
        batch = collate_formal_data([sample, short])
        condition_free, condition_unknown, condition_support = sample.condition.astype(bool)
        np.testing.assert_array_equal(condition_support & ~condition_free & ~condition_unknown, blocked)
        self.assertEqual(batch["starts"][1].tolist(), [[2, 2], [2, 2]])
        self.assertEqual(batch["query_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["candidate_indices"][1].tolist(), [3, -1])

    def test_existing_directory_refuses_before_archive_access(self):
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.prepare_flatlands_formal_data.ZipFile") as archive:
            with self.assertRaises(FileExistsError):
                prepare(Path(temporary))
            archive.assert_not_called()

    def test_test_loader_rejected_before_filesystem_access(self):
        with self.assertRaises(ValueError):
            load_formal_data(Path("/a/nonexistent/final/test/path"), "test")


if __name__ == "__main__":
    unittest.main()
