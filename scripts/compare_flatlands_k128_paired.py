#!/usr/bin/env python3
"""Compare the correlated and independent FlatLands K=128 controls on paired scenes.

Both prediction manifests are label-free.  Labels are joined only through the frozen validation
evaluator contract, and the bootstrap resamples whole ``(source_dataset, scene_id)`` clusters so
that the comparison does not pretend 4,224 event rows are independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.flatlands_data import BOUNDED_QUERIES_SHA256, BOUNDED_SELECTION_SHA256
from pathrel.flatlands_eval import FlatLandsEventRecord, join_flatlands_predictions


SEEDS = (20260831, 20260901, 20260902)
RADII = (0, 10, 20)
METRICS = ("brier", "nll", "ece", "false_safe_rate@0.8", "high_confidence_safe_coverage@0.8")
BIN_COUNT = 10
VALID_SUPPORT_PROTOCOL = "P1_BASELINE_PROTOCOL.md v1 + ConPath valid-support v2"
VALID_SUPPORT_POLICY = (
    "FlatLands epistemic_mask complement is deterministically blocked before posterior sampling"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _validate_source_run(path: Path, variant: str, support_policy: str) -> dict[str, Any] | None:
    if support_policy == "legacy-unmasked":
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or run.get("test_evaluated") is not False:
        raise ValueError(f"source run is not explicitly test-locked: {path}")
    forward = run.get("forward")
    if not isinstance(forward, dict) or forward.get("decoder_variant") != variant:
        raise ValueError(f"source run decoder contract mismatch: {path}")
    if support_policy == "invalid-clamped-posthoc":
        if run.get("posthoc_checkpoint_evaluation") is not True or run.get("retraining_required") is not True:
            raise ValueError(f"source run is not a post-hoc support-clamp evaluation: {path}")
        if forward.get("invalid_support_policy") != "epistemic_mask complement clamped to blocked before sampling":
            raise ValueError(f"source run post-hoc support policy mismatch: {path}")
    elif support_policy == "valid-support-clean-training":
        if run.get("kind") != "flatlands_conpath_training" or run.get("validation_result") is not True:
            raise ValueError(f"source run is not a completed clean training run: {path}")
        if run.get("protocol_version") != VALID_SUPPORT_PROTOCOL:
            raise ValueError(f"source run protocol mismatch: {path}")
        if forward.get("invalid_support_clamped") is not True or forward.get("valid_support_policy") != VALID_SUPPORT_POLICY:
            raise ValueError(f"source run valid-support contract mismatch: {path}")
    else:  # argparse constrains this, but keep build() safe for programmatic callers.
        raise ValueError(f"unsupported support policy: {support_policy}")
    return {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}


def _scene_stats(records: Iterable[FlatLandsEventRecord]) -> tuple[list[tuple[str, str]], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, str], list[FlatLandsEventRecord]] = {}
    for record in records:
        grouped.setdefault(record.scene_key, []).append(record)
    keys = sorted(grouped)
    if not keys:
        raise ValueError("empty paired stratum")
    edges = np.linspace(0.0, 1.0, BIN_COUNT + 1)
    brier = np.empty(len(keys), dtype=np.float64)
    nll = np.empty(len(keys), dtype=np.float64)
    bin_probability = np.zeros((len(keys), BIN_COUNT), dtype=np.float64)
    bin_target = np.zeros_like(bin_probability)
    false_safe = np.zeros(len(keys), dtype=np.float64)
    high_confidence = np.zeros(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        rows = grouped[key]
        probabilities = np.asarray([row.probability for row in rows], dtype=np.float64)
        targets = np.asarray([row.target for row in rows], dtype=np.float64)
        if not np.isfinite(probabilities).all() or ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError(f"non-finite/out-of-range probability in scene {key}")
        brier[index] = np.mean((probabilities - targets) ** 2)
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        nll[index] = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
        assignments = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, BIN_COUNT - 1)
        per_event = 1.0 / len(rows)
        for bin_index in range(BIN_COUNT):
            selected = assignments == bin_index
            bin_probability[index, bin_index] = np.sum(probabilities[selected]) * per_event
            bin_target[index, bin_index] = np.sum(targets[selected]) * per_event
        selected = probabilities >= 0.8
        high_confidence[index] = np.sum(selected) * per_event
        false_safe[index] = np.sum(selected & (targets < 0.5)) * per_event
    return keys, {
        "brier": brier,
        "nll": nll,
        "bin_probability": bin_probability,
        "bin_target": bin_target,
        "false_safe": false_safe,
        "high_confidence": high_confidence,
    }


def _summarize(stats: dict[str, np.ndarray], draws: np.ndarray | None = None) -> dict[str, np.ndarray | float | None]:
    if draws is None:
        brier = stats["brier"].mean()
        nll = stats["nll"].mean()
        bin_probability = stats["bin_probability"].mean(axis=0)
        bin_target = stats["bin_target"].mean(axis=0)
        false_safe_mass = stats["false_safe"].mean()
        high_confidence = stats["high_confidence"].mean()
        ece = np.abs(bin_probability - bin_target).sum()
        false_safe = false_safe_mass / high_confidence if high_confidence > 0.0 else None
        return {
            "brier": float(brier),
            "nll": float(nll),
            "ece": float(ece),
            "false_safe_rate@0.8": None if false_safe is None else float(false_safe),
            "high_confidence_safe_coverage@0.8": float(high_confidence),
        }
    brier = stats["brier"][draws].mean(axis=1)
    nll = stats["nll"][draws].mean(axis=1)
    bin_probability = stats["bin_probability"][draws].mean(axis=1)
    bin_target = stats["bin_target"][draws].mean(axis=1)
    false_safe_mass = stats["false_safe"][draws].mean(axis=1)
    high_confidence = stats["high_confidence"][draws].mean(axis=1)
    false_safe = np.divide(
        false_safe_mass,
        high_confidence,
        out=np.full(high_confidence.shape, np.nan, dtype=np.float64),
        where=high_confidence > 0.0,
    )
    return {
        "brier": brier,
        "nll": nll,
        "ece": np.abs(bin_probability - bin_target).sum(axis=1),
        "false_safe_rate@0.8": false_safe,
        "high_confidence_safe_coverage@0.8": high_confidence,
    }


def _ci(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _paired_stratum(
    correlated: tuple[FlatLandsEventRecord, ...],
    independent: tuple[FlatLandsEventRecord, ...],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if {row.key for row in correlated} != {row.key for row in independent}:
        raise ValueError("paired manifests do not have identical event keys")
    correlated_keys, correlated_stats = _scene_stats(correlated)
    independent_keys, independent_stats = _scene_stats(independent)
    if correlated_keys != independent_keys:
        raise ValueError("paired manifests do not have identical scene keys")
    scene_count = len(correlated_keys)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, scene_count, size=(bootstrap_samples, scene_count))
    point_correlated = _summarize(correlated_stats)
    point_independent = _summarize(independent_stats)
    boot_correlated = _summarize(correlated_stats, draws)
    boot_independent = _summarize(independent_stats, draws)
    delta: dict[str, Any] = {}
    for key in METRICS:
        left = point_correlated[key]
        right = point_independent[key]
        if left is None or right is None:
            point = None
            interval = None
            probability_positive = None
        else:
            point = float(right) - float(left)
            samples = np.asarray(boot_independent[key]) - np.asarray(boot_correlated[key])
            point = float(point)
            interval = _ci(samples)
            probability_positive = float(np.mean(samples > 0.0))
        delta[key] = {
            "independent_minus_correlated": point,
            "bootstrap_95": interval,
            "bootstrap_probability_positive": probability_positive,
        }
    return {
        "scene_count": scene_count,
        "event_count": len(correlated),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "correlated_scene_weighted": point_correlated,
        "independent_scene_weighted": point_independent,
        "delta": delta,
    }


def _subset(records: tuple[FlatLandsEventRecord, ...], radius: int | None = None, source: str | None = None) -> tuple[FlatLandsEventRecord, ...]:
    return tuple(
        row for row in records
        if (radius is None or row.radius_cells == radius)
        and (source is None or row.source_dataset == source)
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    selection = PROJECT_ROOT / args.selection
    queries = PROJECT_ROOT / args.queries
    if _sha256(selection) != BOUNDED_SELECTION_SHA256 or _sha256(queries) != BOUNDED_QUERIES_SHA256:
        raise ValueError("frozen selection/query hash mismatch")
    source_audits: dict[str, dict[str, str]] | None = None
    if args.support_policy == "valid-support-clean-training":
        source_audits = {}
        for variant, root in (
            ("correlated", args.correlated_root),
            ("independent", args.independent_root),
        ):
            audit_path = PROJECT_ROOT / root / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            checks = audit.get("checks")
            if (
                audit.get("passed") is not True
                or audit.get("test_evaluated") is not False
                or audit.get("decoder_variant") != variant
                or not isinstance(checks, dict)
                or checks.get("all_replays_match") is not True
                or checks.get("all_invalid_support_clamped") is not True
                or checks.get("all_checkpoints_restore_strictly") is not True
                or checks.get("all_checkpoint_tensors_finite") is not True
            ):
                raise ValueError(f"clean source audit failed contract checks: {audit_path}")
            source_audits[variant] = {
                "path": str(audit_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(audit_path),
            }
    paired: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        correlated_dir = PROJECT_ROOT / args.correlated_root / f"seed{seed}_{args.correlated_suffix}"
        independent_dir = PROJECT_ROOT / args.independent_root / f"seed{seed}_{args.independent_suffix}"
        correlated_path = correlated_dir / "predictions_validation.csv"
        independent_path = independent_dir / "predictions_validation.csv"
        correlated_run = _validate_source_run(correlated_dir / "run.json", "correlated", args.support_policy)
        independent_run = _validate_source_run(independent_dir / "run.json", "independent", args.support_policy)
        correlated, radii = join_flatlands_predictions(correlated_path, selection, queries, split="validation")
        independent, independent_radii = join_flatlands_predictions(independent_path, selection, queries, split="validation")
        if radii != independent_radii or len(correlated) != 4224 or len(independent) != 4224:
            raise ValueError(f"seed {seed}: unexpected paired coverage")
        correlated = tuple(correlated)
        independent = tuple(independent)
        sources = sorted({row.source_dataset for row in correlated})
        strata: dict[str, Any] = {}
        strata["overall"] = _paired_stratum(
            correlated,
            independent,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + seed_index * 101,
        )
        for radius in RADII:
            strata[f"radius_{radius}"] = _paired_stratum(
                _subset(correlated, radius=radius),
                _subset(independent, radius=radius),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + seed_index * 101 + radius + 1,
            )
        for source in sources:
            strata[f"source_{source}"] = _paired_stratum(
                _subset(correlated, source=source),
                _subset(independent, source=source),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + seed_index * 101 + 10 + sources.index(source),
            )
        paired.append({
            "seed": seed,
            "correlated_prediction_sha256": _sha256(correlated_path),
            "independent_prediction_sha256": _sha256(independent_path),
            "source_runs": {
                "correlated": correlated_run,
                "independent": independent_run,
            },
            "strata": strata,
        })

    aggregate: dict[str, Any] = {}
    for stratum in ("overall", "radius_0", "radius_10", "radius_20"):
        aggregate[stratum] = {}
        for metric in METRICS:
            values = [record["strata"][stratum]["delta"][metric]["independent_minus_correlated"] for record in paired]
            finite = [float(value) for value in values if value is not None and _finite(float(value))]
            aggregate[stratum][metric] = {
                "mean": statistics.mean(finite) if finite else None,
                "sample_sd": statistics.stdev(finite) if len(finite) > 1 else None,
                "seed_values": values,
            }
    posthoc_support_clamp = args.support_policy == "invalid-clamped-posthoc"
    clean_support_training = args.support_policy == "valid-support-clean-training"
    return {
        "schema_version": 1,
        "kind": "flatlands_k128_paired_validation_comparison",
        "validation_only": True,
        "paper_result": False,
        "test_evaluated": False,
        "posthoc_checkpoint_evaluation": posthoc_support_clamp,
        "retraining_required": posthoc_support_clamp,
        "clean_support_training": clean_support_training,
        "protocol": {
            "split": "FlatLands provenance.original_split",
            "official_archive_split_used": False,
            "radii_cells": list(RADII),
            "weighting": "equal scene, then equal event within scene",
            "bootstrap": "paired cluster resampling of (source_dataset, scene_id)",
            "bootstrap_samples": args.bootstrap_samples,
            "test_lock": "No test labels read; physical test remains locked",
            "delta_sign": "independent minus correlated (positive means independent is worse for lower-is-better metrics)",
            "support_policy": args.support_policy,
        },
        "benchmark": {
            "selection_sha256": BOUNDED_SELECTION_SHA256,
            "queries_sha256": BOUNDED_QUERIES_SHA256,
            "prediction_rows_per_seed": 4224,
            "endpoint_groups_per_seed": 1408,
            "seed_count": len(SEEDS),
        },
        "source_audits": source_audits,
        "seeds": paired,
        "aggregate_delta": aggregate,
        "claim_boundary": (
            "Matched validation-only post-hoc support-clamp diagnostic; the checkpoints predate "
            "the clamp and clean retraining is required. No final paper, leaderboard, test, or "
            "cross-domain claim."
            if posthoc_support_clamp
            else (
                "Matched clean-support training validation diagnostic; test labels remain locked. "
                "Paper release still requires the remaining baseline and cross-domain gates."
                if clean_support_training
                else "Matched validation diagnostic only; no final paper, leaderboard, test, or cross-domain claim."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlated-root", type=Path, default=Path("results/p1_flatlands_conpath_k128_validation_v2"))
    parser.add_argument("--independent-root", type=Path, default=Path("results/p1_flatlands_independent_k128_validation_v2"))
    parser.add_argument("--correlated-suffix", default="conpath")
    parser.add_argument("--independent-suffix", default="independent")
    parser.add_argument(
        "--support-policy",
        choices=("legacy-unmasked", "invalid-clamped-posthoc", "valid-support-clean-training"),
        default="legacy-unmasked",
    )
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    result = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(args.output), "kind": result["kind"], "overall_delta": result["aggregate_delta"]["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()
