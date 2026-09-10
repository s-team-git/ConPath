"""Check the newly assembled manuscript data views and artifact links once."""
from pathlib import Path
from urllib.parse import unquote
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METRICS = ["brier", "nll", "ece", "false_safe_at_0_8", "coverage_at_0_8", "risk_at_30_percent"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve(data, pointer):
    for part in pointer.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def near(text, value):
    if value is None:
        assert text == "", (text, value)
    else:
        assert math.isclose(float(text), float(value), rel_tol=0, abs_tol=2e-12), (text, value)


def main():
    report_path = OUT / "submission_verification.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    snapshot_path = ROOT / "results/paper_validation_snapshot.json"
    assert sha(snapshot_path) == "8a3b81ef1f01dec72e57f92d7c02d39ed0e3697e12389143a755a836cc4bbc32"
    snapshot = json.loads(snapshot_path.read_text())
    with (OUT / "analysis/development_metrics_by_seed_and_stratum.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 276
    groups = defaultdict(list)
    metric_checks = 0
    for row in rows:
        obj = resolve(snapshot, row["snapshot_json_pointer"])
        for m in METRICS + ["positive_rate", "mean_score", "parent_count", "event_count"]:
            near(row[m], obj[m])
            metric_checks += 1
        index = int(row["snapshot_json_pointer"].split("/")[2])
        run = snapshot["development_run_metrics"][index]
        assert row["method"] == run["method"] and row["seed"] == str(run["seed"])
        assert int(row["K"]) == run["K"] and row["cohort"] == run["cohort"]
        assert row["source_prediction_sha256"] == run["prediction"]["sha256"]
        if row["cohort"] == "parent_isolated_three_seed" and (
            row["method"] in ("correlated", "independent") and row["K"] == "32"
            or row["method"] == "deterministic" and row["K"] == "1"
        ):
            key = tuple(row[k] for k in ["method", "K", "stratum_dimension", "stratum_value", "weighting"])
            groups[key].append(row)
    with (OUT / "analysis/primary_three_seed_strata.csv").open() as f:
        summaries = list(csv.DictReader(f))
    assert len(summaries) == len(groups) == 36
    for row in summaries:
        key = tuple(row[k] for k in ["method", "K", "stratum_dimension", "stratum_value", "weighting"])
        members = groups[key]
        assert sorted(r["seed"] for r in members) == ["20260910", "20260911", "20260912"]
        for m in METRICS:
            vals = [float(r[m]) for r in members if r[m] != ""]
            assert int(row[m + "_available_repeats"]) == len(vals)
            near(row[m + "_mean"], statistics.mean(vals) if len(vals) == 3 else None)
            near(row[m + "_sample_sd"], statistics.stdev(vals) if len(vals) == 3 else None)
            metric_checks += 2
    documents = ["PAPER_DRAFT.md", "PAPER_EVIDENCE.md", "PAPER_TABLES.md", "PAPER_FIGURES.md",
                 "PAPER_READING_ZH.md", "PAPER_DATA_PACKAGE.md", "PAPER_CLAIM_AUDIT.md", "paper/README.md"]
    link_count = 0
    for rel in documents:
        p = ROOT / rel
        for url in re.findall(r"\]\(([^)\n]+)\)", p.read_text()):
            url = url.strip().strip("<>").split("#", 1)[0]
            if not url or re.match(r"[a-zA-Z]+://", url) or url.startswith("mailto:"):
                continue
            q = Path(unquote(url))
            if not q.is_absolute():
                q = p.parent / q
            assert q.exists(), (rel, url)
            link_count += 1
    pdf_info = {}
    for name in ("main", "supplement"):
        p = ROOT / "paper/build" / (name + ".pdf")
        assert p.read_bytes().startswith(b"%PDF-")
        info = subprocess.check_output(["pdfinfo", str(p)], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        assert pages > 0
        pdf_info[name] = dict(path=str(p.relative_to(ROOT)), pages=pages, sha256=sha(p))
    assert (ROOT / "results/flatlands_external_formal_v1/STOP").is_file()
    assert not subprocess.check_output(["git", "diff", "--name-only", "--", "src", "configs", "scripts"], cwd=ROOT).strip()
    portability = json.loads((OUT / "portability_verification.json").read_text())
    assert portability["pass"] and portability["scalar_metric_comparisons"] == 368
    holdout = json.loads((OUT / "review/holdout_eligibility.json").read_text())
    assert holdout["eligible_holdout_proven"] is False
    assert holdout["eligible_holdout_count"] is None
    record = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  view_rows_checked=len(rows), primary_stratum_rows_checked=len(summaries),
                  data_view_numeric_checks=metric_checks, manuscript_links_checked=link_count,
                  pdfs=pdf_info, documents={p: sha(ROOT / p) for p in documents},
                  unchanged_snapshot_sha256=sha(snapshot_path),
                  frozen_training_and_evaluator_sources_unchanged=True,
                  external_stop_marker_present=True, final_holdout_still_locked=True,
                  scalar_portability_receipt_sha256=sha(OUT / "portability_verification.json"),
                  new_training_runs=0, new_inference_runs=0, final_test_asset_reads=0,
                  scope="New data-view arithmetic, source pointers, document links and PDF existence; layout/science reviews have separate receipts.")
    with report_path.open("x") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ["passed", "view_rows_checked", "primary_stratum_rows_checked", "data_view_numeric_checks", "manuscript_links_checked", "pdfs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
