#!/usr/bin/env python3
"""Audit and compare paired clean-support UnScenes3D K=128 mean-map controls.

The script joins predictions to targets stored in the frozen validation manifest, checks identical
event keys and radius ordering, independently replays the metrics, and bootstraps whole scenes.
It never loads raw frames or label files and refuses reports that are post-hoc or test-evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pathrel.posterior_audits import MEAN_MAP_PROJECTION_VERSION
SEEDS = (20260831, 20260901, 20260902)
RADII = (0, 1, 2)
EXPECTED_ROWS = 4587
EXPECTED_QUERIES = 1529
EXPECTED_MANIFEST_SHA256 = "d866be0b25ba6635d5ad934bac9c822c55be8034fabd8f4627803bb9efa972d5"
EXPECTED_PROTOCOL = "UNSCENES3D_PROTOCOL.md v0.2"
EXPECTED_SUPPORT_POLICY = (
    "UnScenes3D target_valid complement is deterministically blocked before posterior sampling"
)
METRICS = (
    "scene_weighted_brier",
    "query_weighted_nll",
    "query_weighted_ece",
    "false_safe_rate@0.8",
    "high_confidence_coverage@0.8",
)


EventKey = tuple[str, str, int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _close(left: object, right: object, tolerance: float = 1e-10) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    )


def _load_targets(manifest_path: Path) -> tuple[dict[EventKey, float], tuple[str, ...]]:
    manifest = _read_json(manifest_path)
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("frozen manifest SHA-256 mismatch")
    if manifest.get("test_locked_sites") != ["location_6"]:
        raise ValueError("manifest does not lock exactly location_6")
    records = manifest.get("records", {}).get("validation", [])
    targets: dict[EventKey, float] = {}
    scenes: set[str] = set()
    for record in records:
        scene_id = str(record["scene_id"])
        timestamp = str(record["timestamp"])
        scenes.add(scene_id)
        for query in record["queries"]:
            candidate = int(query["candidate_index"])
            reachable = list(query["reachable"])
            if len(reachable) != len(RADII):
                raise ValueError(f"manifest radius target count mismatch: {scene_id}/{timestamp}/{candidate}")
            for radius, target in zip(RADII, reachable, strict=True):
                key = (scene_id, timestamp, candidate, radius)
                if key in targets:
                    raise ValueError(f"duplicate manifest event key: {key}")
                targets[key] = float(bool(target))
    if len(targets) != EXPECTED_ROWS or len(targets) // len(RADII) != EXPECTED_QUERIES:
        raise ValueError(f"unexpected manifest coverage: {len(targets)} rows")
    if scenes != {"scene_00401", "scene_00427"}:
        raise ValueError(f"unexpected validation scenes: {sorted(scenes)}")
    return targets, tuple(sorted(scenes))


def _read_predictions(path: Path, targets: dict[EventKey, float]) -> dict[EventKey, float]:
    expected_header = {"scene_id", "timestamp", "candidate_index", "radius_cells", "probability"}
    predictions: dict[EventKey, float] = {}
    groups: dict[tuple[str, str, int], dict[int, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"prediction header mismatch: {reader.fieldnames!r}")
        for row_number, row in enumerate(reader, 2):
            key = (
                str(row["scene_id"]),
                str(row["timestamp"]),
                int(row["candidate_index"]),
                int(row["radius_cells"]),
            )
            probability = float(row["probability"])
            if key in predictions:
                raise ValueError(f"duplicate prediction key at row {row_number}: {key}")
            if key[3] not in RADII or key not in targets:
                raise ValueError(f"out-of-contract prediction key at row {row_number}: {key}")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid probability at row {row_number}: {probability!r}")
            predictions[key] = probability
            groups.setdefault(key[:3], {})[key[3]] = probability
    if set(predictions) != set(targets):
        missing = len(set(targets) - set(predictions))
        extra = len(set(predictions) - set(targets))
        raise ValueError(f"prediction key coverage mismatch: missing={missing} extra={extra}")
    if len(groups) != EXPECTED_QUERIES:
        raise ValueError(f"endpoint group count {len(groups)} != {EXPECTED_QUERIES}")
    for key, values in groups.items():
        if set(values) != set(RADII):
            raise ValueError(f"incomplete radius group: {key}")
        if values[0] < values[1] - 1e-12 or values[1] < values[2] - 1e-12:
            raise ValueError(f"radius monotonicity violation: {key} {values}")
    return predictions


def _metric_arrays(
    predictions: dict[EventKey, float],
    targets: dict[EventKey, float],
    scenes: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scene in scenes:
        keys = sorted(key for key in targets if key[0] == scene)
        result[scene] = (
            np.asarray([predictions[key] for key in keys], dtype=np.float64),
            np.asarray([targets[key] for key in keys], dtype=np.float64),
        )
    return result


def _metrics(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    scene_draw: tuple[str, ...] | None = None,
) -> dict[str, float]:
    selected = tuple(arrays) if scene_draw is None else scene_draw
    predictions = np.concatenate([arrays[scene][0] for scene in selected])
    targets = np.concatenate([arrays[scene][1] for scene in selected])
    scene_briers = [float(np.mean((arrays[scene][0] - arrays[scene][1]) ** 2)) for scene in selected]
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    nll = float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped)))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        mask = (predictions >= lower) & (
            (predictions < upper) if upper < 1.0 else (predictions <= upper)
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(predictions[mask].mean()) - float(targets[mask].mean()))
    high = predictions >= 0.8
    return {
        "scene_weighted_brier": float(np.mean(scene_briers)),
        "query_weighted_nll": nll,
        "query_weighted_ece": ece,
        "false_safe_rate@0.8": float(np.mean(targets[high] < 0.5)) if np.any(high) else 0.0,
        "high_confidence_coverage@0.8": float(high.mean()),
    }


def _validate_report(
    report_path: Path,
    prediction_path: Path,
    checkpoint_path: Path,
    *,
    decoder_variant: str,
) -> dict[str, Any]:
    report = _read_json(report_path)
    if (
        report.get("kind") != "unscenes3d_conpath_deterministic_mean_map"
        or report.get("paper_result") is not False
        or report.get("validation_result") is not True
        or report.get("test_evaluated") is not False
        or report.get("posthoc_checkpoint_evaluation") is not False
        or report.get("retraining_required") is not False
    ):
        raise ValueError(f"evaluation report is not a clean validation-only result: {report_path}")
    if report.get("protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError(f"protocol version mismatch: {report_path}")
    if report.get("decoder", {}).get("variant") != decoder_variant:
        raise ValueError(f"decoder variant mismatch: {report_path}")
    forward = report.get("forward", {})
    if (
        forward.get("invalid_support_clamped") is not True
        or forward.get("valid_support_policy") != EXPECTED_SUPPORT_POLICY
        or forward.get("checkpoint_trained_with_same_support_policy") is not True
        or forward.get("mean_map_projection_version") != MEAN_MAP_PROJECTION_VERSION
    ):
        raise ValueError(f"support-forward contract mismatch: {report_path}")
    manifest = report.get("manifest", {})
    if (
        manifest.get("sha256") != EXPECTED_MANIFEST_SHA256
        or manifest.get("validation_queries") != EXPECTED_QUERIES
        or manifest.get("test_locked_sites") != ["location_6"]
    ):
        raise ValueError(f"manifest envelope mismatch: {report_path}")
    mean_map = report.get("mean_map", {})
    if mean_map.get("posterior_samples") != 128 or mean_map.get("sample_chunk") != 16:
        raise ValueError(f"mean-map sampling contract mismatch: {report_path}")
    prediction = report.get("prediction", {})
    if prediction.get("rows") != EXPECTED_ROWS or prediction.get("sha256") != _sha256(prediction_path):
        raise ValueError(f"prediction envelope mismatch: {report_path}")
    checkpoint = report.get("checkpoint", {})
    if checkpoint.get("path") != _relative(checkpoint_path) or checkpoint.get("sha256") != _sha256(checkpoint_path):
        raise ValueError(f"checkpoint envelope mismatch: {report_path}")
    return report


def _validate_training_audit(path: Path, decoder_variant: str) -> dict[str, str]:
    audit = _read_json(path)
    checks = audit.get("checks", {})
    if (
        audit.get("passed") is not True
        or audit.get("test_evaluated") is not False
        or audit.get("decoder_variant") != decoder_variant
        or checks.get("manifest_hash") is not True
        or checks.get("seed_count") is not True
        or checks.get("all_invalid_support_clamped") is not True
        or checks.get("all_implementation_hashes_match") is not True
        or checks.get("all_checkpoints_restore_strictly") is not True
        or checks.get("all_checkpoint_tensors_finite") is not True
    ):
        raise ValueError(f"clean training audit failed: {path}")
    return {"path": _relative(path), "sha256": _sha256(path)}


def _bootstrap_delta(
    correlated: dict[str, tuple[np.ndarray, np.ndarray]],
    independent: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    scenes = tuple(sorted(correlated))
    if scenes != tuple(sorted(independent)):
        raise ValueError("paired scene sets differ")
    corr_point = _metrics(correlated)
    indep_point = _metrics(independent)
    point = {metric: indep_point[metric] - corr_point[metric] for metric in METRICS}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), size=(samples, len(scenes)))
    values = {metric: np.empty(samples, dtype=np.float64) for metric in METRICS}
    for index, draw in enumerate(draws):
        scene_draw = tuple(scenes[item] for item in draw)
        corr = _metrics(correlated, scene_draw)
        indep = _metrics(independent, scene_draw)
        for metric in METRICS:
            values[metric][index] = indep[metric] - corr[metric]
    intervals = {
        metric: [float(np.quantile(values[metric], 0.025)), float(np.quantile(values[metric], 0.975))]
        for metric in METRICS
    }
    return point, intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlated-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--correlated-training-root", type=Path, required=True)
    parser.add_argument("--independent-training-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap-samples must be positive")

    correlated_root = (PROJECT_ROOT / args.correlated_root).resolve()
    independent_root = (PROJECT_ROOT / args.independent_root).resolve()
    correlated_training = (PROJECT_ROOT / args.correlated_training_root).resolve()
    independent_training = (PROJECT_ROOT / args.independent_training_root).resolve()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    targets, scenes = _load_targets(manifest_path)
    training_audits = {
        "correlated": _validate_training_audit(correlated_training / "audit.json", "correlated"),
        "independent": _validate_training_audit(independent_training / "audit.json", "independent"),
    }

    seed_records: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        inputs: dict[str, dict[str, Any]] = {}
        arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        replay_metrics: dict[str, dict[str, float]] = {}
        for variant, evaluation_root, training_root in (
            ("correlated", correlated_root, correlated_training),
            ("independent", independent_root, independent_training),
        ):
            run_dir = evaluation_root / f"seed{seed}"
            prediction_path = run_dir / "predictions_validation.csv"
            checkpoint_path = training_root / f"seed{seed}" / "best.pt"
            report_path = run_dir / "run.json"
            report = _validate_report(
                report_path,
                prediction_path,
                checkpoint_path,
                decoder_variant=variant,
            )
            predictions = _read_predictions(prediction_path, targets)
            arrays[variant] = _metric_arrays(predictions, targets, scenes)
            replay = _metrics(arrays[variant])
            recorded = report.get("event_metrics", {})
            for metric in METRICS:
                if not _close(replay[metric], recorded.get(metric)):
                    raise ValueError(f"seed {seed} {variant}: replay mismatch for {metric}")
            replay_metrics[variant] = replay
            inputs[variant] = {
                "report": {"path": _relative(report_path), "sha256": _sha256(report_path)},
                "prediction": {"path": _relative(prediction_path), "sha256": _sha256(prediction_path)},
                "checkpoint": {"path": _relative(checkpoint_path), "sha256": _sha256(checkpoint_path)},
                "mean_map_metrics": report.get("mean_map", {}).get("metrics"),
            }
        delta, intervals = _bootstrap_delta(
            arrays["correlated"],
            arrays["independent"],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 101 * seed_index,
        )
        seed_records.append(
            {
                "seed": seed,
                "scene_count": len(scenes),
                "event_row_count": EXPECTED_ROWS,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed + 101 * seed_index,
                "correlated": replay_metrics["correlated"],
                "independent": replay_metrics["independent"],
                "delta_independent_minus_correlated": {
                    metric: {
                        "point": delta[metric],
                        "scene_bootstrap_95": intervals[metric],
                    }
                    for metric in METRICS
                },
                "inputs": inputs,
            }
        )

    aggregate: dict[str, Any] = {}
    for method in ("correlated", "independent"):
        aggregate[method] = {}
        for metric in METRICS:
            values = [float(record[method][metric]) for record in seed_records]
            aggregate[method][metric] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "seed_values": values,
            }
    aggregate["delta_independent_minus_correlated"] = {}
    for metric in METRICS:
        values = [
            float(record["delta_independent_minus_correlated"][metric]["point"])
            for record in seed_records
        ]
        aggregate["delta_independent_minus_correlated"][metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "seed_values": values,
        }

    result = {
        "schema_version": 1,
        "kind": "unscenes3d_clean_support_k128_paired_mean_map_comparison",
        "validation_only": True,
        "paper_result": False,
        "test_evaluated": False,
        "posthoc_checkpoint_evaluation": False,
        "retraining_required": False,
        "protocol_version": EXPECTED_PROTOCOL,
        "support_policy": EXPECTED_SUPPORT_POLICY,
        "mean_map_projection_version": MEAN_MAP_PROJECTION_VERSION,
        "manifest": {
            "path": _relative(manifest_path),
            "sha256": _sha256(manifest_path),
            "validation_scenes": list(scenes),
            "validation_queries": EXPECTED_QUERIES,
            "event_rows": EXPECTED_ROWS,
            "test_locked_sites": ["location_6"],
        },
        "training_audits": training_audits,
        "seeds": seed_records,
        "aggregate": aggregate,
        "claim_boundary": (
            "Clean-support validation-only deterministic mean-map diagnostic. Only two validation "
            "scenes are available, so scene-bootstrap intervals are descriptive; location_6 remains "
            "locked and this report is not a final paper result."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": _relative(output),
                "seed_count": len(seed_records),
                "event_rows_per_seed": EXPECTED_ROWS,
                "test_evaluated": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
