#!/usr/bin/env python3
"""Generate validation-only UnScenes3D reliability and selective-risk snapshots.

The stochastic controls intentionally export only their scalar report, so this diagnostic
plots the two controls that have label-free per-event CSVs (the deterministic mean-map and
the S4C-inspired coordinate query) plus a train-fitted radius prior.  Validation labels are
joined by the frozen manifest key; no test directory is discovered or opened.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260831, 20260901, 20260902)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
RADII = (0, 1, 2)
COLORS = {
    "radius_prior": "#d97706",
    "mean_map": "#7c3aed",
    "s4c_coordinate": "#2563eb",
}


def _portable_path(path: Path) -> str:
    """Store artifact paths relative to the repository in published JSON."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"calibration artifact must live under project root: {resolved}") from exc


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _targets(manifest: dict[str, Any]) -> tuple[list[tuple[str, str, int, int]], np.ndarray, np.ndarray]:
    keys: list[tuple[str, str, int, int]] = []
    labels: list[float] = []
    scenes: list[str] = []
    for record in manifest["records"]["validation"]:
        scene = str(record["scene_id"])
        timestamp = str(record["timestamp"])
        for query in record["queries"]:
            candidate = int(query["candidate_index"])
            for radius, target in zip(RADII, query["reachable"]):
                keys.append((scene, timestamp, candidate, radius))
                labels.append(float(bool(target)))
                scenes.append(scene)
    return keys, np.asarray(labels, dtype=np.float64), np.asarray(scenes, dtype=object)


def _load_csv(path: Path, keys: list[tuple[str, str, int, int]]) -> np.ndarray:
    import csv

    expected = {"scene_id", "timestamp", "candidate_index", "radius_cells", "probability"}
    values: dict[tuple[str, str, int, int], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if fields != expected:
            raise ValueError(f"prediction CSV must be label-free ({path}): {sorted(fields)}")
        for row in reader:
            key = (str(row["scene_id"]), str(row["timestamp"]), int(row["candidate_index"]), int(row["radius_cells"]))
            probability = float(row["probability"])
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability outside [0,1] in {path}: {probability}")
            if key in values:
                raise ValueError(f"duplicate prediction key in {path}: {key}")
            values[key] = probability
    missing = [key for key in keys if key not in values]
    extra = [key for key in values if key not in set(keys)]
    if missing or extra:
        raise ValueError(f"prediction/manifest key mismatch for {path}: missing={len(missing)} extra={len(extra)}")
    return np.asarray([values[key] for key in keys], dtype=np.float64)


def _weights(scenes: np.ndarray) -> np.ndarray:
    unique = sorted(set(str(value) for value in scenes.tolist()))
    weights = np.zeros((len(scenes),), dtype=np.float64)
    for scene in unique:
        selected = np.asarray([str(value) == scene for value in scenes], dtype=bool)
        weights[selected] = 1.0 / (len(unique) * int(selected.sum()))
    return weights


def _reliability(probability: np.ndarray, labels: np.ndarray, weights: np.ndarray, bins: int = 10) -> list[dict[str, float | int | None]]:
    output: list[dict[str, float | int | None]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probability >= lower) & ((probability < upper) if upper < 1.0 else (probability <= upper))
        mass = float(weights[selected].sum())
        output.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "confidence": float(np.sum(weights[selected] * probability[selected]) / mass) if mass else None,
                "accuracy": float(np.sum(weights[selected] * labels[selected]) / mass) if mass else None,
                "weight": mass,
            }
        )
    return output


def _risk_curve(probability: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> list[dict[str, float | None]]:
    curve: list[dict[str, float | None]] = []
    for threshold in THRESHOLDS:
        selected = probability >= threshold
        coverage = float(weights[selected].sum())
        curve.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "false_safe_rate": float(np.sum(weights[selected] * (labels[selected] < 0.5)) / coverage) if coverage else None,
                "safe_precision": float(np.sum(weights[selected] * labels[selected]) / coverage) if coverage else None,
            }
        )
    return curve


def _mean_sd(values: list[float | None]) -> dict[str, float | None | list[float]]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    return {
        "mean": float(finite.mean()) if len(finite) else None,
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else (0.0 if len(finite) else None),
        "values": [float(value) for value in finite],
    }


def _aggregate_reliability(rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index in range(10):
        selected = [row[index] for row in rows]
        for field in ("confidence", "accuracy", "weight"):
            values = [item[field] for item in selected]
            if field == "weight":
                summary = _mean_sd([float(value) for value in values])
            else:
                summary = _mean_sd([None if value is None else float(value) for value in values])
            if field == "confidence":
                confidence = summary
            elif field == "accuracy":
                accuracy = summary
            else:
                weight = summary
        output.append({
            "bin": index,
            "lower": selected[0]["lower"],
            "upper": selected[0]["upper"],
            "confidence": confidence["mean"],
            "accuracy": accuracy["mean"],
            "confidence_sd": confidence["sd"],
            "accuracy_sd": accuracy["sd"],
            "weight": weight["mean"],
        })
    return output


def _aggregate_curve(rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, threshold in enumerate(THRESHOLDS):
        item: dict[str, Any] = {"threshold": threshold}
        for field in ("coverage", "false_safe_rate", "safe_precision"):
            summary = _mean_sd([None if row[index][field] is None else float(row[index][field]) for row in rows])
            item[field] = summary["mean"]
            item[f"{field}_sd"] = summary["sd"]
        output.append(item)
    return output


def _metric_summary(
    probability: np.ndarray,
    labels: np.ndarray,
    scene_weights: np.ndarray,
    query_weights: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    # Match the scalar reports: Brier is equal-scene; NLL/ECE/risk use the
    # validation event distribution.  The snapshot labels both conventions.
    brier = float(np.sum(scene_weights * (probability - labels) ** 2))
    nll = float(-np.sum(query_weights * (labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))))
    reliability = _reliability(probability, labels, query_weights)
    ece = float(sum(float(row["weight"]) * abs(float(row["confidence"]) - float(row["accuracy"])) for row in reliability if row["confidence"] is not None and row["accuracy"] is not None))
    high = probability >= 0.8
    coverage = float(query_weights[high].sum())
    false_safe = float(np.sum(query_weights[high] * (labels[high] < 0.5)) / coverage) if coverage else 0.0
    return {
        "brier": brier,
        "nll": nll,
        "ece": ece,
        "false_safe_rate@0.8": false_safe,
        "coverage@0.8": coverage,
    }


def _svg_header(title: str, subtitle: str) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 630" role="img">',
        '<rect width="1000" height="630" fill="#fbfcfa"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#26332f}.grid{stroke:#dce6e1;stroke-width:1}.axis{stroke:#6b7c75;stroke-width:1.4}.tick{font-size:13px}.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#596a63}.legend{font-size:14px;font-weight:600}</style>',
        f'<text x="90" y="36" class="title">{html.escape(title)}</text>',
        f'<text x="90" y="58" class="subtitle">{html.escape(subtitle)}</text>',
    ]


def _write_svg(path: Path, methods: list[dict[str, Any]], *, risk: bool) -> None:
    left, top, width, height = 90, 90, 820, 390
    lines = _svg_header(
        "UnScenes3D high-confidence risk" if risk else "UnScenes3D event reliability",
        "Validation-only · query-weighted accepted events · three seeds; test site locked" if risk else "Query-weighted predicted probability versus observed validation frequency",
    )
    for tick in range(0, 11, 2):
        value = tick / 10
        x = left + value * width
        y = top + (1.0 - value) * height
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" class="grid"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" class="tick">{value:.1f}</text>')
        lines.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top + height}" x2="{left}" y2="{top}" class="axis"/><line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>')
    if risk:
        lines.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 48}" text-anchor="middle" class="subtitle">accepted-event coverage</text>')
        lines.append(f'<text x="22" y="{top + height / 2:.1f}" text-anchor="middle" class="subtitle" transform="rotate(-90 22 {top + height / 2:.1f})">false-safe rate</text>')
    else:
        lines.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top}" stroke="#aab8b2" stroke-width="2" stroke-dasharray="7 6"/>')
        lines.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 48}" text-anchor="middle" class="subtitle">mean predicted probability</text>')
        lines.append(f'<text x="22" y="{top + height / 2:.1f}" text-anchor="middle" class="subtitle" transform="rotate(-90 22 {top + height / 2:.1f})">observed event frequency</text>')
    for method in methods:
        if risk:
            points = [(float(row["coverage"]), float(row["false_safe_rate"])) for row in method["false_safe_curve"] if row["coverage"] is not None and row["false_safe_rate"] is not None]
        else:
            points = [(float(row["confidence"]), float(row["accuracy"])) for row in method["reliability"] if row["confidence"] is not None and row["accuracy"] is not None and float(row["weight"]) > 0]
        if not points:
            continue
        point_string = " ".join(f"{left + x * width:.1f},{top + (1 - y) * height:.1f}" for x, y in points)
        color = COLORS[method["id"]]
        lines.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for x, y in points:
            lines.append(f'<circle cx="{left + x * width:.1f}" cy="{top + (1 - y) * height:.1f}" r="5" fill="{color}"/>')
    for index, method in enumerate(methods):
        x = 100 + index * 270
        color = COLORS[method["id"]]
        # Keep the legend below the x-axis label so long method names remain
        # readable in a browser screenshot and in the paper's rasterized copy.
        lines.append(f'<line x1="{x}" y1="560" x2="{x + 24}" y2="560" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{x + 32}" y="565" class="legend">{html.escape(method["label"])}</text>')
    lines.extend([
        '<text x="90" y="610" class="subtitle">Validation-only · query-weighted curve · no test labels · predictive objects shown separately</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/unscenes3d_calibration_validation"))
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    keys, labels, scenes = _targets(manifest)
    scene_weights = _weights(scenes)
    query_weights = np.full((len(keys),), 1.0 / len(keys), dtype=np.float64)
    train_frame_rates: dict[str, list[list[float]]] = {}
    for record in manifest["records"]["train"]:
        queries = list(record["queries"])
        rates = [float(np.mean([bool(query["reachable"][radius]) for query in queries])) for radius in RADII]
        train_frame_rates.setdefault(str(record["scene_id"]), []).append(rates)
    prior = {
        str(radius): float(
            np.mean(
                [np.mean([frame[radius] for frame in frames]) for frames in train_frame_rates.values()]
            )
        )
        for radius in RADII
    }
    methods: list[dict[str, Any]] = []
    specs = (
        ("radius_prior", "Train-fitted radius prior", None),
        ("mean_map", "Posterior mean-map threshold", Path("results/unscenes3d_ground_valid_mean_map_k128")),
        ("s4c_coordinate", "S4C-inspired coordinate query", Path("results/unscenes3d_s4c_coordinate_f16_v2")),
    )
    for method_id, label, root in specs:
        seed_metrics: list[dict[str, float]] = []
        reliabilities: list[list[dict[str, Any]]] = []
        curves: list[list[dict[str, Any]]] = []
        paths: list[str] = []
        for seed in SEEDS:
            if method_id == "radius_prior":
                probability = np.asarray([prior[str(key[3])] for key in keys], dtype=np.float64)
            else:
                assert root is not None
                path = ROOT / root / f"seed{seed}" / "predictions_validation.csv"
                probability = _load_csv(path, keys)
                paths.append(_portable_path(path))
            seed_metrics.append(_metric_summary(probability, labels, scene_weights, query_weights))
            reliabilities.append(_reliability(probability, labels, query_weights))
            curves.append(_risk_curve(probability, labels, query_weights))
        methods.append({
            "id": method_id,
            "label": label,
            "prediction_paths": paths,
            "prior_probabilities": prior if method_id == "radius_prior" else None,
            "seed_metrics": seed_metrics,
            "metrics_mean_sd": {key: _mean_sd([metric[key] for metric in seed_metrics]) for key in seed_metrics[0]},
            "reliability": _aggregate_reliability(reliabilities),
            "false_safe_curve": _aggregate_curve(curves),
            "event_row_count": len(keys),
            "scene_count": len(set(str(value) for value in scenes.tolist())),
        })
    payload = {
        "schema_version": 1,
        "kind": "unscenes3d_validation_calibration_snapshot",
        "protocol_version": "UNSCENES3D_PROTOCOL.md v0.2",
        "validation_only": True,
        "test_evaluated": False,
        "manifest": {"path": str(args.manifest), "validation_records": len(manifest["records"]["validation"]), "validation_queries": len(keys) // len(RADII)},
        "thresholds": list(THRESHOLDS),
        "weighting": {
            "brier": "equal scene, then equal validation event",
            "nll_ece_false_safe_coverage_and_curves": "equal validation event (query weighted)",
            "radius_prior_fit": "equal train scene, then equal frame, then equal retained query",
        },
        "methods": methods,
        "claim_boundary": "Validation-only reliability/selective-risk diagnostics for per-event CSV controls; stochastic reports do not export rows here; no location_6 files or test labels are used.",
    }
    output_dir = ROOT / args.output_dir
    _atomic_write(output_dir / "calibration_snapshot.json", payload)
    _write_svg(ROOT / args.site_dir / "assets/unscenes3d_calibration_reliability.svg", methods, risk=False)
    _write_svg(ROOT / args.site_dir / "assets/unscenes3d_calibration_false_safe.svg", methods, risk=True)
    _atomic_write(ROOT / args.site_dir / "data/unscenes3d_calibration_validation.json", payload)
    print(json.dumps({"output_dir": str(args.output_dir), "event_rows": len(keys), "methods": [method["id"] for method in methods], "test_evaluated": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
