"""Spreadsheet views of a frozen snapshot; no new scores or inference."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
METRICS = ["brier", "nll", "ece", "false_safe_at_0_8", "coverage_at_0_8", "risk_at_30_percent"]


def main():
    source = ROOT / "results/paper_validation_snapshot.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == "8a3b81ef1f01dec72e57f92d7c02d39ed0e3697e12389143a755a836cc4bbc32"
    snapshot = json.loads(source.read_text())
    rows = []
    for index, run in enumerate(snapshot["development_run_metrics"]):
        strata = [("overall", "all", "equal_parent", run["equal_parent"], "equal_parent"),
                  ("overall", "all", "pooled", run["pooled"], "pooled")]
        for dimension in ("source", "radius", "truth"):
            for value, data in run["by_" + dimension].items():
                strata.append((dimension, value, "equal_parent_within_stratum", data,
                               "by_" + dimension + "/" + value))
        for dimension, value, weighting, data, pointer in strata:
            row = dict(cohort=run["cohort"], method=run["method"], seed=run["seed"],
                       K=run["K"], actual_map_count=run["actual_map_count"], K_role=run["K_role"],
                       stratum_dimension=dimension, stratum_value=value, weighting=weighting,
                       event_count=data["event_count"], parent_count=data["parent_count"],
                       positive_rate=data["positive_rate"], mean_score=data["mean_score"])
            row.update({m: data[m] for m in METRICS})
            row.update(snapshot_json_pointer=f"/development_run_metrics/{index}/{pointer}",
                       source_prediction=run["prediction"]["path"],
                       source_prediction_sha256=run["prediction"]["sha256"],
                       validation_only=True)
            rows.append(row)
    assert len(rows) == 23 * 12
    dest = OUT / "development_metrics_by_seed_and_stratum.csv"
    with dest.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = []
    groups = defaultdict(list)
    for row in rows:
        if row["cohort"] == "parent_isolated_three_seed" and (
            (row["method"] in ("correlated", "independent") and row["K"] == 32)
            or (row["method"] == "deterministic" and row["K"] == 1)
        ):
            groups[(row["method"], row["K"], row["stratum_dimension"], row["stratum_value"], row["weighting"])].append(row)
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        assert {int(r["seed"]) for r in group} == {20260910, 20260911, 20260912}
        row = dict(zip(("method", "K", "stratum_dimension", "stratum_value", "weighting"), key))
        row.update(seeds="20260910;20260911;20260912", repeats=3,
                   events_per_seed=group[0]["event_count"], parents_per_seed=group[0]["parent_count"])
        for metric in METRICS:
            values = [r[metric] for r in group]
            valid = [v for v in values if v is not None]
            row[metric + "_available_repeats"] = len(valid)
            # Keep an aggregate missing unless all registered repeats define it.
            row[metric + "_mean"] = statistics.mean(valid) if len(valid) == 3 else None
            row[metric + "_sample_sd"] = statistics.stdev(valid) if len(valid) == 3 else None
        row["positive_rate"] = group[0]["positive_rate"]
        row["snapshot_sha256"] = digest
        summary.append(row)
    aggregate = OUT / "primary_three_seed_strata.csv"
    with aggregate.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    receipt = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                   source=str(source.relative_to(ROOT)), source_sha256=digest,
                   purpose="Flatten already saved metrics; arithmetic three-seed means and sample SD only",
                   seed_stratum_rows=len(rows), primary_three_seed_stratum_rows=len(summary),
                   outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (dest, aggregate)},
                   null_rule="Empty CSV cell means undefined; primary summary requires all three defined repeats",
                   primary_seed_ids=[20260910, 20260911, 20260912],
                   strata=["overall", "source", "radius", "truth"],
                   warning="Truth-conditioned scores are error diagnostics, not separate calibration proof. No new CI or test.",
                   snapshot_modified=False, predictions_modified=False,
                   new_inference=0, new_training=0, test_asset_reads=0)
    with (OUT / "metric_view_receipt.json").open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"seed_stratum_rows": len(rows), "primary_rows": len(summary)}))
    for row in summary:
        if row["stratum_dimension"] in ("radius", "source"):
            print(row["method"], row["stratum_dimension"], row["stratum_value"],
                  "Brier", row["brier_mean"], "risk@.8", row["false_safe_at_0_8_mean"],
                  "coverage@.8", row["coverage_at_0_8_mean"])


if __name__ == "__main__":
    main()
