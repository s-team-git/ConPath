#!/usr/bin/env python3
"""Evaluate one label-free FlatLands prediction manifest on the frozen bounded benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathrel.flatlands_eval import (
    evaluate_flatlands_prediction_file,
    write_evaluation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_flatlands_prediction_file(
        args.predictions,
        args.selection,
        args.queries,
        method=args.method,
        split=args.split,
        bins=args.bins,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_evaluation_report(args.output_dir, report)
    overall = report["metrics"][0]["scene_weighted"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "method": args.method,
                "split": args.split,
                "rows": report["prediction"]["rows"],
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
