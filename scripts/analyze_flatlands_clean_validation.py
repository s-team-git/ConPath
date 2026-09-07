#!/usr/bin/env python3
"""Rebuild the clean FlatLands paper table and risk curves from frozen validation CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pathrel.flatlands_eval import join_flatlands_predictions, _equal_scene_weights, _metric_summary
from pathrel.selective_risk import risk_at_coverage
from scripts.compare_flatlands_k128_paired import _paired_stratum, _validate_source_run
from scripts.evaluate_flatlands_calibration import _radius_monotonicity

SEEDS = (20260831, 20260901, 20260902)
METRICS = ("brier", "nll", "ece", "false_safe_rate@0.8", "high_confidence_safe_coverage@0.8")
COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5)


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None, "values": values.tolist()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications(controls_root, shuffle_root):
    output = [
        ("conpath", "ConPath K=128", "results/p1_flatlands_conpath_k128_support_clamped_v1/seed{seed}_conpath/predictions_validation.csv", SEEDS),
        ("independent", "Independent K=128", "results/p1_flatlands_independent_k128_support_clamped_v1/seed{seed}_independent/predictions_validation.csv", SEEDS),
        ("completion", "Deterministic completion", "results/p1_flatlands_completion_seed{seed}/predictions_deterministic_validation.csv", SEEDS),
        ("coordinate", "Coordinate-query control", "results/p1_flatlands_s4c_coordinate_f16/seed{seed}_s4c_coordinate/predictions_validation.csv", SEEDS),
        ("direct_query", "Direct query (one seed)", "results/p1_flatlands_direct_query_seed{seed}/predictions_validation.csv", SEEDS[:1]),
        ("radius_prior", "Train-fitted radius prior", "results/p1_flatlands_radius_prior_validation/predictions.csv", SEEDS[:1]),
    ]
    if controls_root is not None:
        output.append(("mean_map", "ConPath mean-map K=128", str(controls_root / "correlated/seed{seed}/predictions_mean_map.csv"), SEEDS))
        output.append(("independent_mean_map", "Independent mean-map K=128", str(controls_root / "independent/seed{seed}/predictions_mean_map.csv"), SEEDS))
    if shuffle_root is not None:
        output.append(("marginal_shuffle", "ConPath shuffled worlds (same marginals)", str(shuffle_root / "seed{seed}/predictions_validation.csv"), SEEDS))
    return output


def arrays(records):
    return np.array([r.probability for r in records]), np.array([r.target for r in records]), _equal_scene_weights(records)


def paired_risk(left, right, bootstrap_samples, seed):
    """Resample whole scenes and reselect equal coverage inside every bootstrap draw."""
    if [r.key for r in left] != [r.key for r in right]:
        raise ValueError("equal-coverage comparison requires identical ordered event keys")
    p, y, w = arrays(left)
    q, other_y, other_w = arrays(right)
    if not np.array_equal(y, other_y) or not np.allclose(w, other_w):
        raise ValueError("paired label/weight mismatch")
    scenes = sorted({r.scene_key for r in left})
    indices = {key: index for index, key in enumerate(scenes)}
    scene_index = np.array([indices[r.scene_key] for r in left])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), (bootstrap_samples, len(scenes)))
    values = {c: [] for c in (0.2, 0.3, 0.4)}
    for draw in draws:
        counts = np.bincount(draw, minlength=len(scenes))
        boot_weights = w * counts[scene_index]
        for c in values:
            values[c].append(risk_at_coverage(q, y, boot_weights, c)["false_safe_rate"] - risk_at_coverage(p, y, boot_weights, c)["false_safe_rate"])
    return {
        str(c): {
            "comparator_minus_conpath": risk_at_coverage(q, y, w, c)["false_safe_rate"] - risk_at_coverage(p, y, w, c)["false_safe_rate"],
            "bootstrap_95": np.quantile(values[c], [0.025, 0.975]).tolist(),
        }
        for c in values
    }


def build(args):
    methods = {}
    records_by_method = {}
    canonical_keys = None
    for method, label, pattern, seeds in specifications(args.controls_root, args.shuffle_root):
        seed_rows = []
        records_by_method[method] = {}
        for seed in seeds:
            path = Path(pattern.format(seed=seed))
            if method in ("conpath", "independent"):
                _validate_source_run(ROOT / path.parent / "run.json", "correlated" if method == "conpath" else "independent", "valid-support-clean-training")
                audit = json.loads((path.parent.parent / "audit.json").read_text())
                if audit.get("passed") is not True:
                    raise ValueError(f"clean training audit failed: {path}")
            if method in ("mean_map", "independent_mean_map"):
                checkpoint_report = json.loads((path.parent / "run.json").read_text())
                if checkpoint_report.get("canonical_k128_max_drift") != 0 or not checkpoint_report.get("clean_support_training"):
                    raise ValueError("mean-map control must share an exactly replayed clean checkpoint")
            if method == "marginal_shuffle":
                audit = json.loads((path.parent / "run.json").read_text())
                if not all(audit["checks"].get(k) is True for k in ("canonical_k128_replay_exact", "empirical_cell_counts_preserved", "invalid_support_blocked")):
                    raise ValueError("fixed-marginal shuffle contract failed")
            records, radii = join_flatlands_predictions(path, args.selection, args.queries, split="validation")
            records = tuple(sorted(records, key=lambda r: r.key))
            keys = tuple((r.key, r.scene_key, r.target) for r in records)
            if canonical_keys is None:
                canonical_keys = keys
            if keys != canonical_keys or len(records) != 4224:
                raise ValueError(f"frozen same-query contract mismatch: {path}")
            records_by_method[method][seed] = records
            metrics = _metric_summary(records, weighting="scene", bins=10)
            p, y, w = arrays(records)
            risk = {str(c): risk_at_coverage(p, y, w, c) for c in COVERAGES}
            strata = {}
            for kind, values in [("radius", radii), ("source", sorted({r.source_dataset for r in records}))]:
                for value in values:
                    subset = tuple(r for r in records if (r.radius_cells if kind == "radius" else r.source_dataset) == value)
                    m = _metric_summary(subset, weighting="scene", bins=10)
                    strata[f"{kind}/{value}"] = {k: m[k] for k in (*METRICS, "count", "scene_count", "positive_rate")}
            seed_rows.append({"seed": seed, "prediction": str(path), "prediction_sha256": sha(path), "metrics": metrics, "equal_coverage": risk, "strata": strata, "radius_monotonicity": _radius_monotonicity(records, radii)})
        methods[method] = {
            "label": label,
            "seed_count": len(seeds),
            "aggregate": {k: summary([r["metrics"][k] for r in seed_rows]) for k in METRICS},
            "equal_coverage": {str(c): summary([r["equal_coverage"][str(c)]["false_safe_rate"] for r in seed_rows]) for c in COVERAGES},
            "seeds": seed_rows,
        }
        print(json.dumps({"method": method, "brier": methods[method]["aggregate"]["brier"], "risk_at_30_percent": methods[method]["equal_coverage"]["0.3"]}), flush=True)
    paired = {}
    for method, seed_records in records_by_method.items():
        if method == "conpath":
            continue
        comparisons = []
        for seed, records in seed_records.items():
            base = records_by_method["conpath"][seed]
            pair = _paired_stratum(base, records, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=20260906)
            comparison = {"seed": seed, "delta": {k: {"comparator_minus_conpath": v["independent_minus_correlated"], "bootstrap_95": v["bootstrap_95"]} for k, v in pair["delta"].items()}}
            if method in ("independent", "completion", "coordinate", "mean_map", "independent_mean_map", "marginal_shuffle"):
                comparison["equal_coverage"] = paired_risk(base, records, args.bootstrap_samples, 20260906)
            comparisons.append(comparison)
        paired[method] = {"seeds": comparisons, "brier_delta": summary([r["delta"]["brier"]["comparator_minus_conpath"] for r in comparisons])}
        print(json.dumps({"paired": method, "brier_delta": paired[method]["brier_delta"]}), flush=True)
    return {
        "schema_version": 1,
        "kind": "flatlands_clean_support_paper_analysis",
        "validation_only": True, "test_evaluated": False, "paper_result": False,
        "protocol": {
            "split": "non-official scene-disjoint provenance validation", "events_per_seed": 4224,
            "scene_count": len({r[1] for r in canonical_keys}),
            "weighting": "equal scene, then equal event within scene",
            "coverage_ties": "label-independent fractional acceptance of the complete boundary-probability group",
            "bootstrap": "paired whole scenes; equal-coverage thresholds recomputed within every resample",
            "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": 20260906,
            "inference_scope": "within-seed descriptive validation intervals; multiple comparisons are exploratory, not confirmatory significance tests",
            "selection_sha256": sha(args.selection), "queries_sha256": sha(args.queries),
        },
        "software": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "script_sha256": sha(__file__),
            "selective_risk_sha256": sha(ROOT / "src/pathrel/selective_risk.py"),
            "evaluator_sha256": sha(ROOT / "src/pathrel/flatlands_eval.py"),
            "paired_comparison_sha256": sha(ROOT / "scripts/compare_flatlands_k128_paired.py"),
        },
        "methods": methods, "paired": paired,
        "claim_boundary": "Checkpoints were selected on validation. This analysis does not unlock test data, certify safety, or establish cross-domain generalization. Equal-coverage curves are descriptive ranking diagnostics, not deployable fitted thresholds.",
    }


def write_figures(report, assets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "conpath-clean-paper-v1"})
    assets.mkdir(parents=True, exist_ok=True)
    selected = [m for m in ("conpath", "independent", "completion", "coordinate", "mean_map") if m in report["methods"]]
    colors = {"conpath": "#17866c", "independent": "#c87329", "completion": "#7757a6", "coordinate": "#3268b2", "mean_map": "#9b474d"}
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for method in selected:
        m = report["methods"][method]
        means = np.array([m["equal_coverage"][str(c)]["mean"] for c in COVERAGES])
        sd = np.array([m["equal_coverage"][str(c)]["sample_sd"] for c in COVERAGES])
        ax.plot(COVERAGES, means, "o-", label=m["label"], color=colors[method])
        ax.fill_between(COVERAGES, np.maximum(0, means - sd), np.minimum(1, means + sd), color=colors[method], alpha=0.10)
    ax.set(xlabel="Accepted event coverage (equal-scene weighting)", ylabel="False-safe rate among accepted events", title="Clean FlatLands validation: compare risk at equal coverage", xlim=(0.08, 0.52), ylim=(0, None))
    ax.grid(alpha=0.2); ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.015, "Three training seeds; bands show seed SD. Boundary ties use label-independent fractional acceptance. Test locked.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    for suffix in ("svg", "pdf"):
        fig.savefig(assets / f"flatlands_clean_equal_coverage.{suffix}", metadata={"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    ax.plot([0, 1], [0, 1], "--", color="#8b949e", linewidth=1, label="Perfect calibration")
    for method in selected:
        m = report["methods"][method]; x=[]; y=[]
        for i in range(10):
            bins=[s["metrics"]["reliability"][i] for s in m["seeds"]]
            bins=[b for b in bins if b["weight"] > 0 and b["accuracy"] is not None]
            if bins:
                mass=sum(b["weight"] for b in bins)
                x.append(sum(b["weight"]*b["confidence"] for b in bins)/mass)
                y.append(sum(b["weight"]*b["accuracy"] for b in bins)/mass)
        ax.plot(x, y, "o-", label=m["label"], color=colors[method])
    ax.set(xlabel="Predicted probability", ylabel="Observed reachable fraction", title="Clean FlatLands validation reliability", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2); ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.015, "Equal-scene weighting; bin moments pooled across three training seeds. Validation only; test locked.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    for suffix in ("svg", "pdf"):
        fig.savefig(assets / f"flatlands_clean_reliability.{suffix}", metadata={"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_clean_analysis_v1/report.json"))
    parser.add_argument("--controls-root", type=Path)
    parser.add_argument("--shuffle-root", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--publish-site", action="store_true")
    args = parser.parse_args()
    report = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.write_text(content)
    write_figures(report, args.output.parent / "figures")
    if args.publish_site:
        (ROOT / "site/data/flatlands_clean_paper_analysis.json").write_text(content)
        write_figures(report, ROOT / "site/assets")
    print(json.dumps({"output": str(args.output), "methods": list(report["methods"])}))


if __name__ == "__main__":
    main()
