"""Leak-resistant evaluation contract for frozen FlatLands event predictions.

Prediction files deliberately contain no labels or physical archive split. The evaluator joins
``(global_id, candidate_index, radius_cells)`` against the frozen target-blind query manifest,
requires exact coverage, and reports both query-weighted and equal-scene-weighted calibration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .flatlands import canonical_flatlands_split
from .flatlands_data import (
    BOUNDED_QUERIES_SHA256,
    BOUNDED_SELECTION_SHA256,
    ReplayQuery,
    load_bounded_query_manifest,
)
from .flatlands_query import load_provenance_manifest, sha256_path


PREDICTION_FIELDS = ("global_id", "candidate_index", "radius_cells", "probability")


@dataclass(frozen=True)
class FlatLandsEventRecord:
    global_id: str
    provenance_split: str
    source_dataset: str
    scene_id: str
    candidate_index: int
    radius_cells: int
    probability: float
    target: bool

    @property
    def key(self) -> tuple[str, int, int]:
        return self.global_id, self.candidate_index, self.radius_cells

    @property
    def scene_key(self) -> tuple[str, str]:
        return self.source_dataset, self.scene_id


def _finite_probability(value: object, *, context: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid probability for {context}: {value!r}") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be finite and in [0,1] for {context}")
    return probability


def load_prediction_manifest(
    path: Path,
) -> dict[tuple[str, int, int], float]:
    """Read a label-free prediction CSV and reject duplicate event keys."""

    output: dict[tuple[str, int, int], float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = set(PREDICTION_FIELDS) - set(fields)
        if missing:
            raise ValueError(f"prediction manifest is missing columns: {sorted(missing)}")
        forbidden = {
            "target",
            "label",
            "reachable",
            "floor_map",
            "archive_split",
        } & set(fields)
        if forbidden:
            raise ValueError(
                "prediction manifest must remain label/archive-split blind; forbidden columns: "
                f"{sorted(forbidden)}"
            )
        for line_number, row in enumerate(reader, start=2):
            global_id = row["global_id"]
            if not global_id:
                raise ValueError(f"missing global_id at prediction row {line_number}")
            try:
                candidate_index = int(row["candidate_index"])
                radius_cells = int(row["radius_cells"])
            except ValueError as error:
                raise ValueError(
                    f"invalid integer event key at prediction row {line_number}"
                ) from error
            if candidate_index < 0 or radius_cells < 0:
                raise ValueError(f"negative event key at prediction row {line_number}")
            key = global_id, candidate_index, radius_cells
            if key in output:
                raise ValueError(f"duplicate prediction key at row {line_number}: {key}")
            output[key] = _finite_probability(
                row["probability"], context=f"prediction row {line_number}"
            )
    if not output:
        raise ValueError("prediction manifest contains no rows")
    return output


def _expected_events(
    selection_path: Path,
    query_path: Path,
    *,
    split: str,
) -> tuple[
    dict[tuple[str, int, int], tuple[object, ReplayQuery, bool]], tuple[int, ...]
]:
    canonical_split = canonical_flatlands_split(split)
    if canonical_split is None:
        raise ValueError(f"unknown provenance split: {split!r}")
    observations = load_provenance_manifest(Path(selection_path))
    radii, query_rows, _ = load_bounded_query_manifest(Path(query_path))
    expected: dict[tuple[str, int, int], tuple[object, ReplayQuery, bool]] = {}
    for observation in observations:
        if observation.provenance_split != canonical_split:
            continue
        for query in query_rows[observation.global_id]:
            if not query.retained:
                continue
            for radius_index, radius in enumerate(radii):
                label = query.reachable[radius_index]
                if label is None:
                    raise ValueError(
                        f"retained query is missing radius label for {observation.global_id}"
                    )
                key = observation.global_id, query.candidate_index, radius
                expected[key] = observation, query, bool(label)
    if not expected:
        raise ValueError(f"no retained events for provenance split {canonical_split}")
    return expected, radii


def join_flatlands_predictions(
    prediction_path: Path,
    selection_path: Path,
    query_path: Path,
    *,
    split: str,
    verify_frozen: bool = True,
) -> tuple[tuple[FlatLandsEventRecord, ...], tuple[int, ...]]:
    """Join predictions to frozen labels after exact split/key coverage checks."""

    if verify_frozen:
        if sha256_path(Path(selection_path)) != BOUNDED_SELECTION_SHA256:
            raise ValueError("selected-observation manifest SHA-256 mismatch")
        if sha256_path(Path(query_path)) != BOUNDED_QUERIES_SHA256:
            raise ValueError("bounded query manifest SHA-256 mismatch")
    predictions = load_prediction_manifest(prediction_path)
    expected, radii = _expected_events(selection_path, query_path, split=split)
    prediction_keys = set(predictions)
    expected_keys = set(expected)
    if prediction_keys != expected_keys:
        missing = sorted(expected_keys - prediction_keys)[:20]
        extra = sorted(prediction_keys - expected_keys)[:20]
        raise ValueError(
            "prediction keys do not exactly cover the frozen split: "
            f"missing={missing}, extra={extra}"
        )
    records: list[FlatLandsEventRecord] = []
    for key in sorted(expected):
        observation, query, target = expected[key]
        records.append(
            FlatLandsEventRecord(
                global_id=observation.global_id,
                provenance_split=observation.provenance_split,
                source_dataset=observation.source_dataset,
                scene_id=observation.scene_id,
                candidate_index=query.candidate_index,
                radius_cells=key[2],
                probability=predictions[key],
                target=target,
            )
        )
    return tuple(records), radii


def _equal_scene_weights(records: Sequence[FlatLandsEventRecord]) -> np.ndarray:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        counts[record.scene_key] = counts.get(record.scene_key, 0) + 1
    scene_count = len(counts)
    return np.asarray(
        [1.0 / (scene_count * counts[record.scene_key]) for record in records],
        dtype=np.float64,
    )


def _metric_summary(
    records: Sequence[FlatLandsEventRecord],
    *,
    weighting: str,
    bins: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize empty event records")
    if bins < 1:
        raise ValueError("bins must be positive")
    probabilities = np.asarray([row.probability for row in records], dtype=np.float64)
    targets = np.asarray([row.target for row in records], dtype=np.float64)
    if weighting == "query":
        weights = np.full(probabilities.size, 1.0 / probabilities.size)
    elif weighting == "scene":
        weights = _equal_scene_weights(records)
    else:
        raise ValueError(f"unknown weighting: {weighting!r}")

    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    brier = float(np.sum(weights * (probabilities - targets) ** 2))
    nll = float(
        -np.sum(weights * (targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped)))
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(
        np.searchsorted(edges, probabilities, side="right") - 1, bins - 1
    )
    reliability: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        mass = float(np.sum(weights[selected]))
        if mass > 0.0:
            confidence = float(np.sum(weights[selected] * probabilities[selected]) / mass)
            accuracy = float(np.sum(weights[selected] * targets[selected]) / mass)
            ece += mass * abs(confidence - accuracy)
        else:
            confidence = float((edges[index] + edges[index + 1]) / 2.0)
            accuracy = None
        reliability.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(np.sum(selected)),
                "weight": mass,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    high_confidence = probabilities >= 0.8
    high_confidence_mass = float(np.sum(weights[high_confidence]))
    false_safe = (
        float(
            np.sum(weights[high_confidence] * (targets[high_confidence] < 0.5))
            / high_confidence_mass
        )
        if high_confidence_mass > 0.0
        else None
    )
    return {
        "weighting": weighting,
        "brier": brier,
        "nll": nll,
        "ece": float(ece),
        "false_safe_rate@0.8": false_safe,
        "high_confidence_safe_coverage@0.8": high_confidence_mass,
        "positive_rate": float(np.sum(weights * targets)),
        "mean_probability": float(np.sum(weights * probabilities)),
        "count": len(records),
        "scene_count": len({row.scene_key for row in records}),
        "reliability": reliability,
    }


def _confidence_interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _scene_bootstrap(
    records: Sequence[FlatLandsEventRecord],
    *,
    bins: int,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Cluster bootstrap with equal contribution from each sampled scene."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    grouped: dict[tuple[str, str], list[FlatLandsEventRecord]] = {}
    for record in records:
        grouped.setdefault(record.scene_key, []).append(record)
    scenes = [grouped[key] for key in sorted(grouped)]
    scene_count = len(scenes)
    scene_brier = np.empty(scene_count, dtype=np.float64)
    scene_nll = np.empty(scene_count, dtype=np.float64)
    bin_mass = np.zeros((scene_count, bins), dtype=np.float64)
    bin_probability_mass = np.zeros_like(bin_mass)
    bin_target_mass = np.zeros_like(bin_mass)
    false_safe_mass = np.zeros(scene_count, dtype=np.float64)
    high_confidence_mass = np.zeros(scene_count, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, rows in enumerate(scenes):
        probabilities = np.asarray([row.probability for row in rows], dtype=np.float64)
        targets = np.asarray([row.target for row in rows], dtype=np.float64)
        per_event_weight = 1.0 / len(rows)
        scene_brier[index] = np.mean((probabilities - targets) ** 2)
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        scene_nll[index] = -np.mean(
            targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped)
        )
        assignments = np.minimum(
            np.searchsorted(edges, probabilities, side="right") - 1, bins - 1
        )
        for bin_index in range(bins):
            selected = assignments == bin_index
            bin_mass[index, bin_index] = np.sum(selected) * per_event_weight
            bin_probability_mass[index, bin_index] = (
                np.sum(probabilities[selected]) * per_event_weight
            )
            bin_target_mass[index, bin_index] = (
                np.sum(targets[selected]) * per_event_weight
            )
        selected = probabilities >= 0.8
        high_confidence_mass[index] = np.sum(selected) * per_event_weight
        false_safe_mass[index] = (
            np.sum(selected & (targets < 0.5)) * per_event_weight
        )

    draws = rng.integers(0, scene_count, size=(samples, scene_count))
    brier = scene_brier[draws].mean(axis=1)
    nll = scene_nll[draws].mean(axis=1)
    sampled_bin_probability = bin_probability_mass[draws].mean(axis=1)
    sampled_bin_target = bin_target_mass[draws].mean(axis=1)
    ece = np.sum(np.abs(sampled_bin_probability - sampled_bin_target), axis=1)
    sampled_false_safe = false_safe_mass[draws].mean(axis=1)
    sampled_high_confidence = high_confidence_mass[draws].mean(axis=1)
    false_safe = np.divide(
        sampled_false_safe,
        sampled_high_confidence,
        out=np.full(samples, np.nan, dtype=np.float64),
        where=sampled_high_confidence > 0.0,
    )
    return {
        "unit": "(source_dataset, scene_id)",
        "samples": samples,
        "brier_95": _confidence_interval(brier),
        "nll_95": _confidence_interval(nll),
        "ece_95": _confidence_interval(ece),
        "false_safe_rate@0.8_95": _confidence_interval(false_safe),
    }


def _monotonicity(records: Sequence[FlatLandsEventRecord]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[FlatLandsEventRecord]] = {}
    for record in records:
        grouped.setdefault((record.global_id, record.candidate_index), []).append(record)
    violating = 0
    maximum_increase = 0.0
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: row.radius_cells)
        increases = [
            right.probability - left.probability
            for left, right in zip(ordered, ordered[1:])
        ]
        if any(value > 1e-8 for value in increases):
            violating += 1
        maximum_increase = max(maximum_increase, max(increases, default=0.0))
    return {
        "query_count": len(grouped),
        "violating_queries": violating,
        "violation_rate": violating / len(grouped),
        "maximum_probability_increase": maximum_increase,
        "rule": "predicted reachability must be non-increasing as footprint radius grows",
    }


def summarize_event_records(
    records: Sequence[FlatLandsEventRecord],
    *,
    bins: int = 10,
    bootstrap_samples: int = 2_000,
    seed: int = 20260831,
) -> dict[str, Any]:
    """Produce the frozen overall/source/radius evaluation scopes."""

    records = tuple(records)
    if not records:
        raise ValueError("event records cannot be empty")
    splits = {record.provenance_split for record in records}
    if len(splits) != 1:
        raise ValueError("one report may contain exactly one provenance split")
    scopes: list[tuple[str, str | None, int | None, tuple[FlatLandsEventRecord, ...]]] = [
        ("overall", None, None, records)
    ]
    sources = sorted({record.source_dataset for record in records})
    radii = sorted({record.radius_cells for record in records})
    for source in sources:
        scopes.append(
            (
                "source",
                source,
                None,
                tuple(row for row in records if row.source_dataset == source),
            )
        )
    for radius in radii:
        scopes.append(
            (
                "radius",
                None,
                radius,
                tuple(row for row in records if row.radius_cells == radius),
            )
        )
    for source in sources:
        for radius in radii:
            selected = tuple(
                row
                for row in records
                if row.source_dataset == source and row.radius_cells == radius
            )
            if selected:
                scopes.append(("source_radius", source, radius, selected))

    rng = np.random.default_rng(seed)
    metrics: list[dict[str, Any]] = []
    for scope, source, radius, selected in scopes:
        metrics.append(
            {
                "scope": scope,
                "source_dataset": source,
                "radius_cells": radius,
                "query_weighted": _metric_summary(
                    selected, weighting="query", bins=bins
                ),
                "scene_weighted": _metric_summary(
                    selected, weighting="scene", bins=bins
                ),
                "scene_bootstrap_95": _scene_bootstrap(
                    selected,
                    bins=bins,
                    samples=bootstrap_samples,
                    rng=rng,
                ),
            }
        )
    return {
        "provenance_split": next(iter(splits)),
        "metrics": metrics,
        "radius_monotonicity": _monotonicity(records),
    }


def evaluate_flatlands_prediction_file(
    prediction_path: Path,
    selection_path: Path,
    query_path: Path,
    *,
    method: str,
    split: str,
    bins: int = 10,
    bootstrap_samples: int = 2_000,
    seed: int = 20260831,
    verify_frozen: bool = True,
) -> dict[str, Any]:
    records, radii = join_flatlands_predictions(
        prediction_path,
        selection_path,
        query_path,
        split=split,
        verify_frozen=verify_frozen,
    )
    summary = summarize_event_records(
        records,
        bins=bins,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "schema_version": 1,
        "kind": "flatlands_event_evaluation",
        "paper_result": False,
        "method": method,
        "protocol": {
            "provenance_split": summary["provenance_split"],
            "physical_archive_split_used": False,
            "radii_cells": list(radii),
            "calibration_bins": bins,
            "primary_weighting": "equal scene, then equal event within scene",
            "bootstrap": (
                f"{bootstrap_samples} deterministic cluster resamples of "
                "(source_dataset, scene_id)"
            ),
            "bootstrap_seed": seed,
            "false_safe_threshold": 0.8,
        },
        "benchmark": {
            "selection_sha256": sha256_path(Path(selection_path)),
            "selection_frozen_sha256": BOUNDED_SELECTION_SHA256,
            "queries_sha256": sha256_path(Path(query_path)),
            "queries_frozen_sha256": BOUNDED_QUERIES_SHA256,
            "frozen_hashes_verified": bool(verify_frozen),
        },
        "prediction": {
            "path": str(Path(prediction_path)),
            "sha256": sha256_path(Path(prediction_path)),
            "rows": len(records),
            "label_columns_allowed": False,
        },
        "metrics": summary["metrics"],
        "radius_monotonicity": summary["radius_monotonicity"],
        "claim_boundary": (
            "This report evaluates one fixed prediction manifest. It becomes a paper result only "
            "after the method/checkpoint selection protocol, multi-seed aggregation, and frozen "
            "test access rules are satisfied."
        ),
    }


def _atomic_write_text(path: Path, payload: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_prediction_manifest(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> int:
    """Atomically write the only prediction schema accepted by the evaluator."""

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["global_id"]),
            int(row["candidate_index"]),
            int(row["radius_cells"]),
        ),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    seen: set[tuple[str, int, int]] = set()
    for row in ordered:
        global_id = str(row["global_id"])
        candidate_index = int(row["candidate_index"])
        radius_cells = int(row["radius_cells"])
        key = global_id, candidate_index, radius_cells
        if key in seen:
            raise ValueError(f"duplicate prediction key: {key}")
        seen.add(key)
        probability = _finite_probability(row["probability"], context=str(key))
        writer.writerow(
            {
                "global_id": global_id,
                "candidate_index": candidate_index,
                "radius_cells": radius_cells,
                "probability": f"{probability:.17g}",
            }
        )
    if not seen:
        raise ValueError("cannot write an empty prediction manifest")
    _atomic_write_text(Path(path), stream.getvalue())
    return len(seen)


def write_evaluation_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    _atomic_write_text(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    fields = (
        "method",
        "split",
        "scope",
        "source_dataset",
        "radius_cells",
        "weighting",
        "brier",
        "nll",
        "ece",
        "false_safe_rate@0.8",
        "high_confidence_safe_coverage@0.8",
        "positive_rate",
        "mean_probability",
        "count",
        "scene_count",
    )
    rows: list[dict[str, object]] = []
    for item in report["metrics"]:
        for weighting_key in ("query_weighted", "scene_weighted"):
            metric = item[weighting_key]
            rows.append(
                {
                    "method": report["method"],
                    "split": report["protocol"]["provenance_split"],
                    "scope": item["scope"],
                    "source_dataset": item["source_dataset"] or "",
                    "radius_cells": (
                        "" if item["radius_cells"] is None else item["radius_cells"]
                    ),
                    "weighting": metric["weighting"],
                    **{key: metric[key] for key in fields if key in metric},
                }
            )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: "" if row.get(field, "") is None else row.get(field, "")
                for field in fields
            }
        )
    _atomic_write_text(output_dir / "metrics.csv", stream.getvalue())
