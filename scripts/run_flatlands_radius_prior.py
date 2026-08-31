#!/usr/bin/env python3
"""Fit the frozen train-only radius-prior control and evaluate it on one locked split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from pathrel.flatlands_baselines import (
    fit_scene_weighted_radius_prior,
    radius_prior_prediction_rows,
)
from pathrel.flatlands_data import load_bounded_query_manifest
from pathrel.flatlands_eval import (
    evaluate_flatlands_prediction_file,
    write_evaluation_report,
    write_prediction_manifest,
)
from pathrel.flatlands_query import load_provenance_manifest, sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="explicit unlock required after every method/config has been frozen",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/p1_flatlands_radius_prior_validation"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "FlatLands test is locked by P1_BASELINE_PROTOCOL.md; pass --allow-test only after "
            "all validation-selected methods/configurations are frozen."
        )
    observations = load_provenance_manifest(args.selection)
    radii, queries, _ = load_bounded_query_manifest(args.queries)
    prior = fit_scene_weighted_radius_prior(observations, queries, radii)
    rows = radius_prior_prediction_rows(
        observations,
        queries,
        radii,
        prior,
        prediction_split=args.split,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    row_count = write_prediction_manifest(prediction_path, rows)
    evaluation = evaluate_flatlands_prediction_file(
        prediction_path,
        args.selection,
        args.queries,
        method="radius_prior_control",
        split=args.split,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    evaluation_dir = args.output_dir / "evaluation"
    write_evaluation_report(evaluation_dir, evaluation)
    run_report = {
        "schema_version": 1,
        "kind": "flatlands_radius_prior_control",
        "paper_result": False,
        "protocol_version": "P1_BASELINE_PROTOCOL.md v1",
        "fit_split": "train",
        "prediction_split": args.split,
        "test_explicitly_unlocked": bool(args.split == "test" and args.allow_test),
        "scene_weighting": "equal contributing scene, then equal retained query",
        "radii_cells": list(radii),
        "probabilities": {str(key): value for key, value in prior.probabilities.items()},
        "fitted_scene_count": prior.fitted_scene_count,
        "fitted_event_count": prior.fitted_event_count,
        "prediction_rows": row_count,
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_path(prediction_path),
            },
            "evaluation_report": str(evaluation_dir / "report.json"),
            "metrics_csv": str(evaluation_dir / "metrics.csv"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "argv": sys.argv,
        },
        "claim_boundary": (
            "Train-only per-radius prevalence control. It uses no image and is not a strong "
            "baseline, learned public-data model result, or paper result."
        ),
    }
    atomic_json(args.output_dir / "run.json", run_report)
    overall = evaluation["metrics"][0]["scene_weighted"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "split": args.split,
                "prior": run_report["probabilities"],
                "fitted_scenes": prior.fitted_scene_count,
                "prediction_rows": row_count,
                "scene_weighted": {
                    key: overall[key]
                    for key in (
                        "brier",
                        "nll",
                        "ece",
                        "false_safe_rate@0.8",
                        "count",
                        "scene_count",
                    )
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
