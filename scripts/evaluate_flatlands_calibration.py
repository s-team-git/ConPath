#!/usr/bin/env python3
"""Build validation-only calibration and high-confidence-risk diagnostics.

The evaluator consumes the already-exported, label-free prediction manifests from the four
capacity-matched controls.  It joins each manifest to the frozen provenance validation labels,
then aggregates equal-scene metrics across the three seeds.  No checkpoint is selected here and
the FlatLands test split is never opened.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pathrel.flatlands_eval import (
    _equal_scene_weights,
    _metric_summary,
    join_flatlands_predictions,
)


SEEDS = (20260831, 20260901, 20260902)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
COLORS = {
    "conpath": "#168f63",
    "independent_decoder": "#d97706",
    "mean_map": "#7c3aed",
    "s4c_coordinate": "#2563eb",
}


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
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/p1_flatlands_calibration_validation")
    )
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--publish-site", action="store_true")
    return parser.parse_args()


def _method_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "conpath",
            "label": "ConPath stochastic",
            "root": Path("results/p1_flatlands_conpath_matrix_f16"),
            "pattern": "seed{seed}_conpath/predictions_validation.csv",
        },
        {
            "id": "independent_decoder",
            "label": "Independent decoder",
            "root": Path("results/p1_flatlands_conpath_independent_f16"),
            "pattern": "seed{seed}_independent_decoder/predictions_validation.csv",
        },
        {
            "id": "mean_map",
            "label": "Posterior mean-map threshold",
            "root": Path("results/p1_flatlands_conpath_mean_map_f16"),
            "pattern": "seed{seed}/predictions_validation.csv",
        },
        {
            "id": "s4c_coordinate",
            "label": "S4C coordinate control",
            "root": Path("results/p1_flatlands_s4c_coordinate_f16"),
            "pattern": "seed{seed}_s4c_coordinate/predictions_validation.csv",
        },
    )


def _mean_sd(values: Iterable[float | None]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "sd": None, "values": []}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "values": [float(value) for value in finite],
    }


def _risk_curve(records: tuple[object, ...]) -> list[dict[str, float | None]]:
    probabilities = np.asarray([row.probability for row in records], dtype=np.float64)
    targets = np.asarray([row.target for row in records], dtype=np.float64)
    weights = _equal_scene_weights(records)
    curve: list[dict[str, float | None]] = []
    for threshold in THRESHOLDS:
        selected = probabilities >= threshold
        coverage = float(np.sum(weights[selected]))
        false_safe = (
            float(np.sum(weights[selected] * (targets[selected] < 0.5)) / coverage)
            if coverage > 0.0
            else None
        )
        precision = (
            float(np.sum(weights[selected] * targets[selected]) / coverage)
            if coverage > 0.0
            else None
        )
        curve.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "false_safe_rate": false_safe,
                "safe_precision": precision,
            }
        )
    return curve


def _aggregate_reliability(
    summaries: list[dict[str, Any]], bins: int
) -> list[dict[str, float | int | None]]:
    output: list[dict[str, float | int | None]] = []
    for index in range(bins):
        rows = [summary["reliability"][index] for summary in summaries]
        confidence = [row["confidence"] for row in rows if row["accuracy"] is not None]
        accuracy = [row["accuracy"] for row in rows if row["accuracy"] is not None]
        mass = [row["weight"] for row in rows]
        output.append(
            {
                "bin": index,
                "lower": float(rows[0]["lower"]),
                "upper": float(rows[0]["upper"]),
                "confidence": float(np.mean(confidence)) if confidence else None,
                "accuracy": float(np.mean(accuracy)) if accuracy else None,
                "confidence_sd": float(np.std(confidence, ddof=1)) if len(confidence) > 1 else 0.0,
                "accuracy_sd": float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else 0.0,
                "weight": float(np.mean(mass)),
                "available_seeds": len(accuracy),
            }
        )
    return output


def _aggregate_curve(curves: list[list[dict[str, float | None]]]) -> list[dict[str, float | None]]:
    output: list[dict[str, float | None]] = []
    for index, threshold in enumerate(THRESHOLDS):
        fields: dict[str, float | None] = {"coverage": None, "false_safe_rate": None, "safe_precision": None}
        for field in tuple(fields):
            values = [curve[index][field] for curve in curves]
            summary = _mean_sd(values)
            fields[field] = summary["mean"]
            fields[f"{field}_sd"] = summary["sd"]
        output.append({"threshold": threshold, **fields})
    return output


def _radius_summary(records_by_seed: list[tuple[object, ...]], radii: tuple[int, ...]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for radius in radii:
        values = []
        for records in records_by_seed:
            selected = tuple(row for row in records if row.radius_cells == radius)
            values.append(_metric_summary(selected, weighting="scene", bins=10)["brier"])
        output[str(radius)] = _mean_sd(values)
    return output


def _load_methods(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[int, ...]]:
    methods: list[dict[str, Any]] = []
    radii: tuple[int, ...] | None = None
    for spec in _method_specs():
        records_by_seed: list[tuple[object, ...]] = []
        paths: list[str] = []
        for seed in SEEDS:
            path = spec["root"] / spec["pattern"].format(seed=seed)
            if not path.exists():
                raise FileNotFoundError(f"missing validation prediction manifest: {path}")
            records, method_radii = join_flatlands_predictions(
                path,
                args.selection,
                args.queries,
                split=args.split,
                verify_frozen=True,
            )
            records_by_seed.append(records)
            paths.append(str(path))
            if radii is None:
                radii = tuple(method_radii)
            elif tuple(method_radii) != radii:
                raise ValueError("method prediction manifests disagree on radii")
        summaries = [_metric_summary(records, weighting="scene", bins=args.bins) for records in records_by_seed]
        metrics = {
            name: _mean_sd([summary[name] for summary in summaries])
            for name in ("brier", "nll", "ece", "false_safe_rate@0.8", "high_confidence_safe_coverage@0.8")
        }
        methods.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "seed_values": {"seeds": list(SEEDS), "prediction_paths": paths},
                "metrics": metrics,
                "reliability": _aggregate_reliability(summaries, args.bins),
                "false_safe_curve": _aggregate_curve([_risk_curve(records) for records in records_by_seed]),
                "radius_brier": _radius_summary(records_by_seed, radii or ()),
                "event_count": len(records_by_seed[0]),
                "scene_count": len({row.scene_key for row in records_by_seed[0]}),
            }
        )
    if radii is None:
        raise RuntimeError("no calibration methods loaded")
    return {"methods": methods}, radii


def _svg_header(title: str, subtitle: str, width: int = 1000, height: int = 570) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="#fbfcfa"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#26332f} .grid{stroke:#dce6e1;stroke-width:1} .axis{stroke:#6b7c75;stroke-width:1.4} .tick{font-size:13px} .title{font-size:22px;font-weight:700} .subtitle{font-size:13px;fill:#596a63} .legend{font-size:14px;font-weight:600}</style>',
        f'<text x="90" y="36" class="title">{html.escape(title)}</text>',
        f'<text x="90" y="58" class="subtitle">{html.escape(subtitle)}</text>',
    ]


def _svg_footer(lines: list[str]) -> list[str]:
    lines.extend([
        '<text x="90" y="548" class="subtitle">Validation-only · equal-scene weighting · 3 seeds · FlatLands test split locked</text>',
        '</svg>',
    ])
    return lines


def _write_reliability_svg(path: Path, methods: list[dict[str, Any]]) -> None:
    left, top, width, height = 90, 90, 820, 390
    lines = _svg_header("FlatLands event reliability", "Confidence versus observed connectivity frequency; ideal calibration is the diagonal")
    for tick in range(0, 11, 2):
        value = tick / 10
        x = left + value * width
        y = top + (1.0 - value) * height
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" class="grid"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" class="tick">{value:.1f}</text>')
        lines.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top}" stroke="#aab8b2" stroke-width="2" stroke-dasharray="7 6"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/><line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>')
    lines.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 48}" text-anchor="middle" class="subtitle">mean predicted probability</text>')
    lines.append(f'<text x="22" y="{top + height / 2:.1f}" text-anchor="middle" class="subtitle" transform="rotate(-90 22 {top + height / 2:.1f})">observed event frequency</text>')
    for method in methods:
        points = []
        for row in method["reliability"]:
            if row["accuracy"] is not None and row["weight"] > 0:
                points.append((float(row["confidence"]), float(row["accuracy"])))
        if points:
            point_string = " ".join(f"{left + x * width:.1f},{top + (1 - y) * height:.1f}" for x, y in points)
            color = COLORS[method["id"]]
            lines.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
            for x, y in points:
                lines.append(f'<circle cx="{left + x * width:.1f}" cy="{top + (1 - y) * height:.1f}" r="5" fill="{color}"/>')
    for index, method in enumerate(methods):
        x = 100 + index * 215
        color = COLORS[method["id"]]
        lines.append(f'<line x1="{x}" y1="505" x2="{x + 24}" y2="505" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{x + 32}" y="510" class="legend">{html.escape(method["label"])}</text>')
    _svg_footer(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_false_safe_svg(path: Path, methods: list[dict[str, Any]]) -> None:
    left, top, width, height = 90, 90, 820, 390
    lines = _svg_header("FlatLands high-confidence risk", "False-safe rate versus coverage as the confidence threshold increases")
    for tick in range(0, 11, 2):
        value = tick / 10
        x = left + value * width
        y = top + (1.0 - value) * height
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" class="grid"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" class="tick">{value:.1f}</text>')
        lines.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/><line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>')
    lines.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 48}" text-anchor="middle" class="subtitle">coverage of accepted events</text>')
    lines.append(f'<text x="22" y="{top + height / 2:.1f}" text-anchor="middle" class="subtitle" transform="rotate(-90 22 {top + height / 2:.1f})">false-safe rate</text>')
    for method in methods:
        points = [(float(row["coverage"]), float(row["false_safe_rate"])) for row in method["false_safe_curve"] if row["coverage"] is not None and row["false_safe_rate"] is not None]
        if points:
            point_string = " ".join(f"{left + x * width:.1f},{top + (1 - y) * height:.1f}" for x, y in points)
            color = COLORS[method["id"]]
            lines.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
            for x, y in points:
                lines.append(f'<circle cx="{left + x * width:.1f}" cy="{top + (1 - y) * height:.1f}" r="5" fill="{color}"/>')
    for index, method in enumerate(methods):
        x = 100 + index * 215
        color = COLORS[method["id"]]
        lines.append(f'<line x1="{x}" y1="505" x2="{x + 24}" y2="505" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{x + 32}" y="510" class="legend">{html.escape(method["label"])}</text>')
    _svg_footer(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise SystemExit("--bins must be at least 2")
    payload, radii = _load_methods(args)
    payload.update(
        {
            "schema_version": 1,
            "kind": "flatlands_calibration_snapshot",
            "paper_result": False,
            "protocol": {
                "provenance_split": args.split,
                "physical_archive_split_used": False,
                "radii_cells": list(radii),
                "calibration_bins": args.bins,
                "weighting": "equal scene, then equal event within scene",
                "seeds": list(SEEDS),
                "false_safe_thresholds": list(THRESHOLDS),
                "test_split": "locked; not opened",
            },
            "claim_boundary": "Validation diagnostics for uncertainty and selective-risk analysis; not final paper or test results.",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "calibration_snapshot.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.publish_site:
        data_path = args.site_dir / "data" / "flatlands_calibration_validation.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_reliability_svg(args.site_dir / "assets" / "flatlands_calibration_reliability.svg", payload["methods"])
        _write_false_safe_svg(args.site_dir / "assets" / "flatlands_calibration_false_safe.svg", payload["methods"])
    print(json.dumps({"output": str(result_path), "methods": [method["id"] for method in payload["methods"]], "radii": list(radii)}, indent=2))


if __name__ == "__main__":
    main()
