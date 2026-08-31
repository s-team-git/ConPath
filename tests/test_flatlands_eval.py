from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from pathrel.flatlands_data import load_bounded_query_manifest
from pathrel.flatlands_eval import (
    FlatLandsEventRecord,
    evaluate_flatlands_prediction_file,
    join_flatlands_predictions,
    load_prediction_manifest,
    summarize_event_records,
    write_evaluation_report,
    write_prediction_manifest,
)
from pathrel.flatlands_query import load_provenance_manifest
from tests.test_flatlands_data import _write_fixture


def _perfect_rows(selection_path: Path, query_path: Path) -> list[dict[str, object]]:
    observations = load_provenance_manifest(selection_path)
    radii, queries, _ = load_bounded_query_manifest(query_path)
    rows: list[dict[str, object]] = []
    for observation in observations:
        for query in queries[observation.global_id]:
            if not query.retained:
                continue
            for radius, target in zip(radii, query.reachable):
                rows.append(
                    {
                        "global_id": observation.global_id,
                        "candidate_index": query.candidate_index,
                        "radius_cells": radius,
                        "probability": float(bool(target)),
                    }
                )
    return rows


class FlatLandsEvaluationContractTest(unittest.TestCase):
    def test_exact_join_and_perfect_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, selection, queries = _write_fixture(root)
            predictions = root / "predictions.csv"
            row_count = write_prediction_manifest(
                predictions, _perfect_rows(selection, queries)
            )
            report = evaluate_flatlands_prediction_file(
                predictions,
                selection,
                queries,
                method="perfect_fixture",
                split="test",
                bootstrap_samples=20,
                verify_frozen=False,
            )
            write_evaluation_report(root / "evaluation", report)
            persisted = json.loads(
                (root / "evaluation" / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["prediction"]["rows"], row_count)
        self.assertEqual(report["metrics"][0]["scene_weighted"]["brier"], 0.0)
        self.assertEqual(report["metrics"][0]["query_weighted"]["brier"], 0.0)
        self.assertEqual(report["radius_monotonicity"]["violating_queries"], 0)
        self.assertEqual(persisted["method"], "perfect_fixture")
        self.assertFalse(persisted["benchmark"]["frozen_hashes_verified"])

    def test_missing_duplicate_and_label_bearing_predictions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, selection, queries = _write_fixture(root)
            rows = _perfect_rows(selection, queries)
            predictions = root / "predictions.csv"
            write_prediction_manifest(predictions, rows[:-1])
            with self.assertRaisesRegex(ValueError, "do not exactly cover"):
                join_flatlands_predictions(
                    predictions,
                    selection,
                    queries,
                    split="test",
                    verify_frozen=False,
                )

            with self.assertRaisesRegex(ValueError, "duplicate prediction key"):
                write_prediction_manifest(predictions, [rows[0], rows[0]])

            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "global_id",
                        "candidate_index",
                        "radius_cells",
                        "probability",
                        "target",
                    ),
                )
                writer.writeheader()
                writer.writerow({**rows[0], "target": 1})
            with self.assertRaisesRegex(ValueError, "forbidden columns"):
                load_prediction_manifest(predictions)

    def test_primary_scene_weighting_prevents_query_rich_scene_domination(self) -> None:
        records = []
        for candidate in range(10):
            records.append(
                FlatLandsEventRecord(
                    global_id="many",
                    provenance_split="validation",
                    source_dataset="Source",
                    scene_id="scene-many",
                    candidate_index=candidate,
                    radius_cells=0,
                    probability=1.0,
                    target=False,
                )
            )
        records.append(
            FlatLandsEventRecord(
                global_id="one",
                provenance_split="validation",
                source_dataset="Source",
                scene_id="scene-one",
                candidate_index=0,
                radius_cells=0,
                probability=1.0,
                target=True,
            )
        )
        summary = summarize_event_records(records, bootstrap_samples=30)
        overall = summary["metrics"][0]
        self.assertAlmostEqual(overall["query_weighted"]["brier"], 10.0 / 11.0)
        self.assertAlmostEqual(overall["scene_weighted"]["brier"], 0.5)
        self.assertEqual(overall["scene_weighted"]["scene_count"], 2)

    def test_radius_probability_increase_is_reported(self) -> None:
        records = [
            FlatLandsEventRecord(
                global_id="item",
                provenance_split="test",
                source_dataset="Source",
                scene_id="scene",
                candidate_index=0,
                radius_cells=radius,
                probability=probability,
                target=target,
            )
            for radius, probability, target in (
                (0, 0.4, True),
                (10, 0.5, True),
                (20, 0.2, False),
            )
        ]
        summary = summarize_event_records(records, bootstrap_samples=10)
        self.assertEqual(summary["radius_monotonicity"]["violating_queries"], 1)
        self.assertAlmostEqual(
            summary["radius_monotonicity"]["maximum_probability_increase"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
