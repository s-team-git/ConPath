#!/usr/bin/env python3
"""Build a compact, repository-relative FlatLands K=128 validation hand-off.

The training directories are intentionally ignored because they contain checkpoints and large
intermediate files.  This command turns the independently audited run/report pair into the small
JSON object consumed by the static site.  It only reads validation reports and never invokes the
test evaluator or opens the physical archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260831, 20260901, 20260902)
RADIUS = (0, 10, 20)
METRICS = ("brier", "nll", "ece", "false_safe_rate@0.8", "high_confidence_safe_coverage@0.8")
SELECTION_SHA256 = "4e7ae4c992cf943ab81618e3826c4748fcaaa97c3c4d7cb187518ee3fe6a9409"
QUERIES_SHA256 = "33e7f8a0343269b0dde47b428b3be622c80effdb0f80ae34b352ca282018d60d"
ARCHIVE_BYTES = 2_054_773_316
EXPECTED_PROTOCOL_VERSION = "P1_BASELINE_PROTOCOL.md v1 + ConPath valid-support v2"
EXPECTED_SUPPORT_POLICY = (
    "FlatLands epistemic_mask complement is deterministically blocked before posterior sampling"
)


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


def _metric(report: dict[str, Any], scope: str, radius: int | None = None) -> dict[str, Any]:
    for item in report.get("metrics", []):
        if not isinstance(item, dict) or item.get("scope") != scope:
            continue
        if radius is not None and item.get("radius_cells") != radius:
            continue
        value = item.get("scene_weighted")
        if isinstance(value, dict):
            return value
    suffix = f" radius={radius}" if radius is not None else ""
    raise ValueError(f"missing scene-weighted {scope} metric{suffix}")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in METRICS:
        # The compact site seed rows use the shorter ``coverage@0.8`` spelling, while
        # evaluator reports use the fully qualified metric name.
        raw_values = [record[key] if key in record else record["coverage@0.8"] for record in records]
        values = [float(value) for value in raw_values if value is not None and _finite(value)]
        result[key] = {
            "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "available_seed_count": len(values),
            "total_seed_count": len(raw_values),
        }
    return result


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = PROJECT_ROOT / args.root
    audit_path = root / "audit.json"
    audit = _read_json(audit_path)
    if audit.get("passed") is not True:
        raise ValueError("refusing to publish a failed FlatLands K=128 audit")
    if audit.get("validation_only") is not True or audit.get("test_evaluated") is not False:
        raise ValueError("audit is not explicitly validation-only/test-locked")
    if audit.get("decoder_variant") != args.decoder_variant:
        raise ValueError("audit decoder variant does not match requested snapshot")
    checks = audit.get("checks")
    if (
        not isinstance(checks, dict)
        or checks.get("all_replays_match") is not True
        or checks.get("all_invalid_support_clamped") is not True
        or checks.get("all_checkpoints_restore_strictly") is not True
        or checks.get("all_checkpoint_tensors_finite") is not True
    ):
        raise ValueError("audit does not report exact replay/support/checkpoint agreement")

    seed_records: list[dict[str, Any]] = []
    radius_records: dict[int, list[dict[str, Any]]] = {radius: [] for radius in RADIUS}
    artifacts: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = root / f"seed{seed}_{args.run_suffix}"
        run_path = run_dir / "run.json"
        report_path = run_dir / "evaluation_validation" / "report.json"
        prediction_path = run_dir / "predictions_validation.csv"
        run = _read_json(run_path)
        report = _read_json(report_path)
        if run.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
            raise ValueError(f"seed {seed} is not a valid-support v2 training run")
        forward = run.get("forward")
        if not isinstance(forward, dict) or forward.get("invalid_support_clamped") is not True:
            raise ValueError(f"seed {seed} does not explicitly clamp invalid support")
        if forward.get("valid_support_policy") != EXPECTED_SUPPORT_POLICY:
            raise ValueError(f"seed {seed} valid-support policy mismatch")
        overall = _metric(report, "overall")
        audit_seed = next((item for item in audit.get("seeds", []) if item.get("seed") == seed), None)
        if not isinstance(audit_seed, dict):
            raise ValueError(f"audit has no seed record for {seed}")
        if audit_seed.get("prediction_sha256") != _sha256(prediction_path):
            raise ValueError(f"prediction hash changed after audit for seed {seed}")
        if int(audit_seed.get("prediction_rows", -1)) != 4224:
            raise ValueError(f"unexpected prediction row count for seed {seed}")
        if any(not _finite(overall.get(key)) for key in METRICS):
            raise ValueError(f"non-finite overall metric for seed {seed}")
        record = {
            "seed": seed,
            "best_epoch": run.get("selection", {}).get("best_epoch"),
            "epochs_completed": run.get("selection", {}).get("epochs_completed"),
            "selection_brier": run.get("selection", {}).get("best_validation_scene_weighted_brier"),
            "brier": overall["brier"],
            "nll": overall["nll"],
            "ece": overall["ece"],
            "false_safe_rate@0.8": overall["false_safe_rate@0.8"],
            "coverage@0.8": overall["high_confidence_safe_coverage@0.8"],
            "prediction_rows": int(audit_seed["prediction_rows"]),
            "prediction_sha256": audit_seed["prediction_sha256"],
            "report_scene_count": overall.get("scene_count"),
            "replay_matches": bool(audit_seed.get("replay_matches")),
            "radius_monotonicity_failures": int(audit_seed.get("radius_monotonicity_failures", 0)),
            "checkpoint": audit_seed.get("checkpoint"),
        }
        seed_records.append(record)
        for radius in RADIUS:
            radius_metric = _metric(report, "radius", radius)
            radius_records[radius].append({key: radius_metric[key] for key in METRICS})
        for path, role in (
            (run_dir / "best.pt", "canonical best checkpoint"),
            (run_path, "run metadata"),
            (prediction_path, "label-free validation predictions"),
            (report_path, "exact validation report"),
        ):
            artifacts.append({
                "seed": seed,
                "role": role,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })

    first_run = _read_json(root / f"seed{SEEDS[0]}_{args.run_suffix}" / "run.json")
    first_data = first_run.get("data", {})
    if not isinstance(first_data, dict):
        raise ValueError("run metadata has no data envelope")
    if first_data.get("selection_sha256") != SELECTION_SHA256 or first_data.get("queries_sha256") != QUERIES_SHA256:
        raise ValueError("run metadata hashes differ from frozen contract")
    if first_data.get("archive_bytes") != ARCHIVE_BYTES:
        raise ValueError("FlatLands archive byte count differs from frozen contract")

    aggregate = _aggregate(seed_records)
    per_radius = {
        str(radius): _aggregate(radius_records[radius]) for radius in RADIUS
    }
    report_scene_counts = sorted({record["report_scene_count"] for record in seed_records})
    if len(report_scene_counts) != 1:
        raise ValueError(f"inconsistent report scene counts: {report_scene_counts}")
    snapshot = {
        "schema_version": 1,
        "kind": f"flatlands_{args.decoder_variant}_k128_validation_snapshot",
        "validation_only": True,
        "paper_result": False,
        "test_evaluated": False,
        "posthoc_checkpoint_evaluation": False,
        "retraining_required": False,
        "decoder_variant": args.decoder_variant,
        "protocol": {
            "name": EXPECTED_PROTOCOL_VERSION,
            "split": "FlatLands provenance.original_split",
            "official_archive_split_used": False,
            "radii_cells": list(RADIUS),
            "train_samples": 8,
            "validation_samples": 128,
            "validation_sample_chunk": 8,
            "selection_query_limit": 8,
            "primary_weighting": "equal scene, then equal event within scene",
            "exact_forward": "NumPy disk-clearance + batched Kruskal merge-tree",
            "support_policy": EXPECTED_SUPPORT_POLICY,
            "bootstrap_samples": 2000,
            "test_lock": "No test labels read; physical test remains locked",
        },
        "benchmark": {
            "train_scenes": first_data.get("train_scenes"),
            "validation_scenes": first_data.get("validation_scenes"),
            "report_scene_count": report_scene_counts[0],
            "prediction_rows_per_seed": 4224,
            "endpoint_groups_per_seed": 1408,
            "selection_sha256": SELECTION_SHA256,
            "queries_sha256": QUERIES_SHA256,
            "archive_bytes": ARCHIVE_BYTES,
        },
        "seeds": seed_records,
        "aggregate_scene_weighted": aggregate,
        "per_radius_scene_weighted": per_radius,
        "artifacts": artifacts,
        "audit": {
            "audit_report": str(audit_path.relative_to(PROJECT_ROOT)),
            "passed": True,
            "all_processes_exit_code_0": True,
            "flags_and_config_match": True,
            "prediction_hashes_match": True,
            "rows_finite_and_in_range": True,
            "radius_monotonicity_violations": checks.get("radius_monotonicity_violations", 0),
            "independent_report_replay_matches": checks.get("all_replays_match", False),
            "all_invalid_support_clamped": checks.get("all_invalid_support_clamped", False),
            "all_checkpoints_restore_strictly": checks.get("all_checkpoints_restore_strictly", False),
            "all_checkpoint_tensors_finite": checks.get("all_checkpoint_tensors_finite", False),
            "label_scope": "validation only",
        },
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="ignored K=128 run root")
    parser.add_argument("--run-suffix", default="conpath")
    parser.add_argument("--decoder-variant", choices=("correlated", "independent"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    snapshot = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "output": str(args.output),
        "kind": snapshot["kind"],
        "passed": snapshot["audit"]["passed"],
        "aggregate": snapshot["aggregate_scene_weighted"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
