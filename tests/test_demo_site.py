import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_demo_site import (
    build_flatlands_outcomes_svg,
    build_flatlands_reachability_svg,
    build_flatlands_site,
    build_flatlands_snapshot,
)


def _split_row(selected: int, invalid: int, disconnected: int, failure: int, positive: int):
    retained = disconnected + failure + positive
    return {
        "candidate_queries": selected * 2,
        "selected_queries": selected,
        "target_invalid_endpoints": invalid,
        "retained_valid_endpoint_queries": retained,
        "target_status_counts": {
            "disconnected_radius_zero": disconnected,
            "footprint_failure": failure,
            "high_clearance_positive": positive,
        },
        "reachable_by_radius_cells": {
            "0": {"rate": (retained - disconnected) / retained, "reachable": retained - disconnected, "total": retained},
            "10": {"rate": positive / retained, "reachable": positive, "total": retained},
            "20": {"rate": positive / retained, "reachable": positive, "total": retained},
        },
        "scene_weighted_failure_rate": (disconnected + failure) / retained,
    }


def _stratum_row(retained: int, positive: int):
    return {
        "retained_valid_endpoint_queries": retained,
        "scene_count_with_retained_queries": 9,
        "scene_weighted_failure_rate": 1.0 - positive / retained,
        "reachable_by_radius_cells": {
            "0": {"rate": 1.0, "reachable": retained, "total": retained},
            "10": {"rate": positive / retained, "reachable": positive, "total": retained},
            "20": {"rate": positive / retained, "reachable": positive, "total": retained},
        },
    }


class FlatLandsSiteSnapshotTest(unittest.TestCase):
    def _reports(self, root: Path) -> tuple[Path, Path]:
        query = {
            "gates": {
                "p1_bounded_query_gate_passed": True,
                "mask_semantics_passed": True,
                "query_balance_by_gated_stratum": {
                    "validation/SourceA": {"passed": True},
                    "test/SourceB": {"passed": True},
                },
            },
            "sample": {"selected_observations": 12, "split_source_strata": 3},
            "protocol": {
                "footprint_radii_cells": [0, 10, 20],
                "footprint_radii_m": [0.0, 0.1, 0.2],
            },
            "query_summary_by_split": {
                "train": _split_row(10, 2, 1, 4, 3),
                "validation": _split_row(12, 3, 1, 5, 3),
                "test": _split_row(14, 3, 2, 4, 5),
            },
            "query_summary_by_split_source": {
                "train/SourceA": _stratum_row(8, 3),
                "validation/SourceA": _stratum_row(9, 3),
                "test/SourceB": _stratum_row(11, 5),
            },
            "outputs": {
                "queries": {"rows": 72, "sha256": "a" * 64},
                "selected_observations": {"rows": 12, "sha256": "b" * 64},
            },
        }
        provenance = {
            "metadata": {
                "scene_overlap": {
                    "train__validation": {"count": 7},
                    "train__test": {"count": 5},
                    "validation__test": {"count": 3},
                },
                "provenance_scene_overlap": {
                    "train__validation": {"count": 0},
                    "train__test": {"count": 0},
                    "validation__test": {"count": 0},
                },
            }
        }
        query_path = root / "query.json"
        provenance_path = root / "provenance.json"
        query_path.write_text(json.dumps(query), encoding="utf-8")
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return query_path, provenance_path

    def test_snapshot_is_compact_non_model_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            query_path, provenance_path = self._reports(Path(directory))
            snapshot = build_flatlands_snapshot(query_path, provenance_path)

        self.assertFalse(snapshot["paper_result"])
        self.assertFalse(snapshot["model_result"])
        self.assertTrue(snapshot["data_gate_passed"])
        self.assertEqual(snapshot["selected_observations"], 12)
        self.assertEqual(snapshot["gated_strata"], {"passed": 2, "total": 2})
        self.assertEqual(snapshot["totals"]["selected_queries"], 36)
        self.assertEqual(len(snapshot["strata"]), 2)
        self.assertEqual(
            [row["official_scene_overlap"] for row in snapshot["official_split_overlap"]],
            [7, 5, 3],
        )
        self.assertTrue(snapshot["adapter"]["direct_zip_streaming"])
        self.assertFalse(snapshot["adapter"]["physical_archive_split_used_for_evaluation"])

    def test_writes_strict_snapshot_and_data_driven_svgs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path, provenance_path = self._reports(root)
            output = root / "site"
            snapshot = build_flatlands_site(query_path, provenance_path, output)
            parsed = json.loads((output / "data" / "flatlands_audit.json").read_text())
            reachability_svg = (output / "assets" / "flatlands_reachability.svg").read_text()
            outcomes_svg = (output / "assets" / "flatlands_query_outcomes.svg").read_text()

        self.assertEqual(parsed, snapshot)
        self.assertIn("validation · SourceA", reachability_svg)
        self.assertIn("100.0%", build_flatlands_reachability_svg(snapshot))
        self.assertIn("n = 10 selected", outcomes_svg)
        self.assertIn("Target-invalid endpoint", build_flatlands_outcomes_svg(snapshot))


if __name__ == "__main__":
    unittest.main()
