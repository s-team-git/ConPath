#!/usr/bin/env python3
"""Audit the frozen UnScenes3D validation control package.

This command is intentionally read-only with respect to the data and checkpoints.  It
opens only the explicit train/validation manifest and the canonical validation
artifacts; ``location_6`` is represented as a lock assertion and is never traversed.
The audit checks three independent things that are easy to accidentally conflate:

* the adapter/query contract (manifest hash, counts, geometry and split lock),
* the metric reports and label-free prediction CSVs (recomputed event metrics), and
* the reproducibility/site layer (referenced artifact hashes, PNG dimensions and local
  HTML links).

The report exits non-zero on a failed invariant, making it suitable for the overnight
handoff loop.  Validation labels in the frozen manifest are used only to recompute the
validation diagnostics; no test labels are loaded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import math
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.2"
MANIFEST_PATH = Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json")
SNAPSHOT_PATH = Path("site/data/unscenes3d_controls_validation.json")
SITE_PATH = Path("site/index.html")
MANIFEST_REPLAY_PATH = Path("results/unscenes3d_manifest_replay_audit/report.json")
LOCKED_SITE = "location_6"
RADII = (0, 1, 2)
SEEDS = (20260831, 20260901, 20260902)
EXPECTED_ADAPTER = {
    "endpoint_policy": "ground",
    "start_selection": "valid",
    "ground_margin_m": 0.35,
    "ground_bin_size_m": 1.2,
    "ground_lateral_limit_m": 20.0,
    "ground_quantile": 0.15,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_sha256(manifest: dict[str, Any]) -> str:
    """Hash only deterministic manifest contract fields, excluding runtime timing."""

    normalized = dict(manifest)
    normalized.pop("generation_seconds", None)
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _portable_strings(value: Any) -> Any:
    """Replace local project prefixes in the published audit with relative paths."""

    prefix = str(PROJECT_ROOT).rstrip("/") + "/"
    if isinstance(value, str):
        return value.replace(prefix, "")
    if isinstance(value, list):
        return [_portable_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable_strings(item) for key, item in value.items()}
    return value


def _close(actual: object, expected: object, tolerance: float = 1e-8) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    return actual == expected


class _Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, passed: bool, detail: object = "") -> None:
        record: dict[str, Any] = {"name": name, "passed": bool(passed)}
        if detail != "":
            record["detail"] = detail
        self.checks.append(record)
        if not passed:
            self.failures.append(name if detail == "" else f"{name}: {detail}")

    def warn(self, name: str, detail: object) -> None:
        self.warnings.append(name if detail == "" else f"{name}: {detail}")


def _expected_targets(manifest: dict[str, Any]) -> tuple[dict[tuple[str, str, int, int], bool], dict[str, int], int]:
    validation = manifest.get("records", {}).get("validation", [])
    if not isinstance(validation, list):
        raise ValueError("manifest records.validation must be a list")
    targets: dict[tuple[str, str, int, int], bool] = {}
    scene_queries: dict[str, int] = defaultdict(int)
    for record in validation:
        if not isinstance(record, dict):
            raise ValueError("validation record is not an object")
        scene = str(record["scene_id"])
        timestamp = str(record["timestamp"])
        queries = record.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError(f"queries is not a list for {timestamp}")
        for query in queries:
            candidate = int(query["candidate_index"])
            reachable = list(query["reachable"])
            if len(reachable) != len(RADII):
                raise ValueError(f"unexpected radius count for {timestamp}/{candidate}")
            scene_queries[scene] += 1
            for radius, value in zip(RADII, reachable):
                key = (scene, timestamp, candidate, radius)
                if key in targets:
                    raise ValueError(f"duplicate validation target key: {key}")
                targets[key] = bool(value)
    return targets, dict(scene_queries), len(validation)


def _ece(predictions: list[float], labels: list[float]) -> float:
    import numpy as np

    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    value = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        selected = (p >= lower) & ((p < upper) if upper < 1.0 else (p <= upper))
        if np.any(selected):
            value += float(selected.mean()) * abs(float(p[selected].mean()) - float(y[selected].mean()))
    return value


def _read_prediction_csv(
    path: Path,
    targets: dict[tuple[str, str, int, int], bool],
    expected_query_count: int,
    audit: _Audit,
) -> dict[str, Any]:
    """Read a label-free prediction CSV and recompute event metrics."""

    required = {"scene_id", "timestamp", "candidate_index", "radius_cells", "probability"}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    parse_errors: list[str] = []
    invalid_probabilities: list[float] = []
    duplicate_keys: list[tuple[str, str, int, int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        audit.check(f"prediction header label-free: {path}", fields == required, sorted(fields))
        for row in reader:
            try:
                key = (
                    str(row["scene_id"]),
                    str(row["timestamp"]),
                    int(row["candidate_index"]),
                    int(row["radius_cells"]),
                )
                probability = float(row["probability"])
            except (KeyError, TypeError, ValueError) as exc:
                parse_errors.append(repr(exc))
                continue
            finite = math.isfinite(probability) and 0.0 <= probability <= 1.0
            if not finite:
                invalid_probabilities.append(probability)
            if key in seen:
                duplicate_keys.append(key)
            seen.add(key)
            if key not in targets:
                continue
            rows.append({"key": key, "probability": probability, "target": targets[key]})

    audit.check(f"prediction rows parseable: {path}", not parse_errors, parse_errors[:3])
    audit.check(f"prediction probabilities bounded: {path}", not invalid_probabilities, {"invalid_count": len(invalid_probabilities), "examples": invalid_probabilities[:3]})
    audit.check(f"prediction keys unique: {path}", not duplicate_keys, {"duplicate_count": len(duplicate_keys), "examples": duplicate_keys[:3]})
    unknown_keys = sorted(seen.difference(targets))
    audit.check(f"prediction keys in frozen validation manifest: {path}", not unknown_keys, unknown_keys[:3])

    expected_rows = expected_query_count * len(RADII)
    audit.check(f"prediction row count: {path}", len(rows) == expected_rows, {"actual": len(rows), "expected": expected_rows})
    audit.check(f"prediction key set complete: {path}", seen == set(targets), {"actual": len(seen), "expected": len(targets)})
    if not rows:
        return {"row_count": 0, "query_count": 0, "radius_monotonicity_violations": None}

    import numpy as np

    predictions = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    labels = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    per_scene: dict[str, list[float]] = defaultdict(list)
    grouped: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        scene, timestamp, candidate, radius = row["key"]
        per_scene[scene].append((float(row["probability"]) - float(row["target"])) ** 2)
        grouped[(scene, timestamp, candidate)][radius] = float(row["probability"])
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    nll = float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)))
    high = predictions >= 0.8
    monotonicity = sum(
        int(any(values.get(radius + 1, 0.0) > values.get(radius, 0.0) + 1e-12 for radius in (0, 1)))
        for values in grouped.values()
    )
    return {
        "row_count": len(rows),
        "query_count": len(rows) // len(RADII),
        "scene_count": len(per_scene),
        "scene_weighted_brier": float(np.mean([np.mean(values) for values in per_scene.values()])),
        "query_weighted_nll": nll,
        "query_weighted_ece": _ece(predictions.tolist(), labels.tolist()),
        "false_safe_rate@0.8": float(np.mean(labels[high] < 0.5)) if np.any(high) else 0.0,
        "high_confidence_coverage@0.8": float(high.mean()),
        "positive_prevalence": float(labels.mean()),
        "mean_prediction": float(predictions.mean()),
        "radius_monotonicity_violations": monotonicity,
    }


def _check_common_report(
    report: dict[str, Any],
    *,
    path: Path,
    manifest_sha: str,
    validation_records: int,
    validation_queries: int,
    audit: _Audit,
    allow_legacy_protocol: bool = False,
) -> None:
    protocol = report.get("protocol_version")
    protocol_ok = protocol == PROTOCOL_VERSION or (allow_legacy_protocol and protocol == "UNSCENES3D_PROTOCOL.md v0.1")
    audit.check(f"{path} protocol", protocol_ok, protocol)
    if protocol != PROTOCOL_VERSION:
        audit.warn(f"legacy protocol metadata in {path}", protocol)
    for key, expected in (("schema_version", 1), ("test_evaluated", False), ("validation_result", True), ("paper_result", False)):
        audit.check(f"{path} {key}", report.get(key) == expected, report.get(key))
    boundary = str(report.get("claim_boundary", ""))
    audit.check(f"{path} locks location_6", LOCKED_SITE in boundary, boundary)
    manifest = report.get("manifest", {})
    # The first map-only reports predate the richer v0.2 report envelope and put
    # split counts under ``data``.  Treat that as a legacy schema only when the
    # adapter and lock assertions below still pass.
    if not isinstance(manifest, dict) or not manifest:
        legacy = report.get("data", {})
        manifest = legacy if isinstance(legacy, dict) else {}
    if isinstance(manifest, dict):
        if "sha256" in manifest:
            audit.check(f"{path} manifest hash", manifest.get("sha256") == manifest_sha, manifest.get("sha256"))
        else:
            config = report.get("config", {})
            configured_manifest = config.get("manifest") if isinstance(config, dict) else None
            if configured_manifest is not None:
                configured_path = Path(str(configured_manifest))
                if configured_path.is_absolute():
                    configured_path = configured_path.relative_to(PROJECT_ROOT)
                configured_sha = _sha256(PROJECT_ROOT / configured_path) if (PROJECT_ROOT / configured_path).is_file() else None
                audit.check(f"{path} manifest hash via config", configured_sha == manifest_sha, {"actual": configured_sha, "expected": manifest_sha})
            else:
                audit.warn(f"{path} missing manifest sha256", "legacy report; checkpoint/config still checked")
        if "validation_records" in manifest:
            audit.check(f"{path} validation records", int(manifest.get("validation_records", -1)) == validation_records, manifest.get("validation_records"))
        else:
            audit.warn(f"{path} missing validation_records", "legacy report")
        if "validation_queries" in manifest:
            audit.check(f"{path} validation queries", int(manifest.get("validation_queries", -1)) == validation_queries, manifest.get("validation_queries"))
        else:
            event_metrics = report.get("validation_event_metrics", {})
            derived_queries = event_metrics.get("query_count") if isinstance(event_metrics, dict) else None
            if derived_queries is not None:
                audit.check(f"{path} validation queries via event report", int(derived_queries) == validation_queries, derived_queries)
            else:
                audit.warn(f"{path} missing validation_queries", "legacy/qualitative report")
        if "test_locked_sites" in manifest:
            audit.check(f"{path} test lock metadata", manifest.get("test_locked_sites") == [LOCKED_SITE], manifest.get("test_locked_sites"))
        else:
            audit.warn(f"{path} missing test lock metadata", "legacy report")
    else:
        audit.check(f"{path} manifest object", False, manifest)


def _check_adapter(value: object, name: str, audit: _Audit) -> None:
    if not isinstance(value, dict):
        audit.check(name, False, value)
        return
    for key, expected in EXPECTED_ADAPTER.items():
        actual = value.get(key)
        audit.check(f"{name}.{key}", _close(actual, expected), {"actual": actual, "expected": expected})


class _LocalLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _check_site(root: Path, site_path: Path, audit: _Audit) -> None:
    parser = _LocalLinkParser()
    parser.feed((root / site_path).read_text(encoding="utf-8"))
    local: list[str] = []
    for raw in parser.links:
        if raw.startswith(("#", "http://", "https://", "mailto:", "data:", "javascript:")):
            continue
        clean = raw.split("#", 1)[0].split("?", 1)[0]
        if clean:
            local.append(clean)
    missing: list[str] = []
    for value in sorted(set(local)):
        candidate = (root / site_path.parent / value).resolve()
        if not candidate.is_relative_to((root / site_path.parent).resolve()) or not candidate.is_file():
            missing.append(value)
    audit.check("site local href/src links", not missing, {"checked": len(set(local)), "missing": missing})
    html = (root / site_path).read_text(encoding="utf-8")
    audit.check("site contains UnScenes3D snapshot link", "unscenes3d_controls_validation.json" in html, "snapshot link")
    audit.check("site explicitly labels rejected observed-free result", "rejected observed-free" in html, "claim-boundary wording")
    audit.check("site contains UnScenes3D calibration assets", "unscenes3d_calibration_reliability.svg" in html and "unscenes3d_calibration_false_safe.svg" in html, "calibration figures")
    audit.check(
        "site exposes replay/audit evidence",
        all(name in html for name in (
            "unscenes3d_controls_audit.json",
            "unscenes3d_manifest_replay_audit.json",
            "unscenes3d_replay_audit.json",
        )),
        "audit/replay links",
    )


def _artifact_hash_check(path: Path, expected_sha: str | None, expected_bytes: int | None, audit: _Audit, name: str) -> None:
    audit.check(f"{name} exists", path.is_file(), str(path))
    if not path.is_file():
        return
    if expected_bytes is not None:
        audit.check(f"{name} byte size", path.stat().st_size == int(expected_bytes), {"actual": path.stat().st_size, "expected": expected_bytes})
    if expected_sha is not None:
        actual = _sha256(path)
        audit.check(f"{name} sha256", actual == expected_sha, {"actual": actual, "expected": expected_sha})


def _compare_metrics(report_metrics: dict[str, Any], expected: dict[str, Any], prefix: str, audit: _Audit) -> None:
    aliases = {
        "brier": "scene_weighted_brier",
        "nll": "query_weighted_nll",
        "ece": "query_weighted_ece",
        "false_safe_rate_at_0_8": "false_safe_rate@0.8",
        "coverage_at_0_8": "high_confidence_coverage@0.8",
    }
    for snapshot_key, report_key in aliases.items():
        if snapshot_key not in expected:
            continue
        if report_key not in report_metrics:
            audit.check(f"{prefix} metric present: {report_key}", False, report_metrics)
            continue
        audit.check(f"{prefix} metric {report_key}", _close(report_metrics[report_key], expected[snapshot_key], 2e-8), {"actual": report_metrics[report_key], "expected": expected[snapshot_key]})
    for key in ("query_count", "event_row_count", "radius_monotonicity_violations"):
        if key in expected and key in report_metrics:
            audit.check(f"{prefix} metric {key}", _close(report_metrics[key], expected[key]), {"actual": report_metrics[key], "expected": expected[key]})


def _run_event_control(
    *,
    root: Path,
    run_dir: Path,
    snapshot_entry: dict[str, Any],
    targets: dict[tuple[str, str, int, int], bool],
    validation_records: int,
    validation_queries: int,
    manifest_sha: str,
    audit: _Audit,
    csv_required: bool,
    allow_legacy_protocol: bool = False,
) -> None:
    run_path = root / run_dir / "run.json"
    audit.check(f"run report exists: {run_path}", run_path.is_file(), str(run_path))
    if not run_path.is_file():
        return
    report = _read_json(run_path)
    _check_common_report(report, path=run_path, manifest_sha=manifest_sha, validation_records=validation_records, validation_queries=validation_queries, audit=audit, allow_legacy_protocol=allow_legacy_protocol)
    _check_adapter(report.get("adapter", report.get("config", {}).get("adapter") if isinstance(report.get("config"), dict) else None), str(run_path) + " adapter", audit)
    metrics = report.get("event_metrics", report.get("validation_event_metrics", {}))
    if not isinstance(metrics, dict):
        audit.check(f"{run_path} event metrics object", False, metrics)
        return
    seed = int(run_dir.name.replace("seed", ""))
    expected_seed = next((x for x in snapshot_entry.get("seeds", []) if int(x.get("seed", -1)) == seed), None)
    if expected_seed is None:
        audit.check(f"{run_path} seed in snapshot", False, seed)
    else:
        _compare_metrics(metrics, expected_seed, str(run_path), audit)

    if csv_required:
        prediction = report.get("prediction", {})
        if not isinstance(prediction, dict) or not prediction:
            artifacts = report.get("artifacts", {})
            if isinstance(artifacts, dict):
                prediction = artifacts.get("predictions_validation", {})
        if not isinstance(prediction, dict):
            audit.check(f"{run_path} prediction object", False, prediction)
            return
        prediction_path = root / str(prediction.get("path", ""))
        _artifact_hash_check(prediction_path, str(prediction.get("sha256")) if prediction.get("sha256") else None, None, audit, f"{run_path} prediction")
        if prediction_path.is_file():
            recomputed = _read_prediction_csv(prediction_path, targets, validation_queries, audit)
            for key, value in recomputed.items():
                if key in {"row_count", "query_count"}:
                    continue
                if key in metrics:
                    audit.check(f"{run_path} recomputed {key}", _close(metrics[key], value, 2e-8), {"reported": metrics[key], "recomputed": value})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--site", type=Path, default=SITE_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/unscenes3d_controls_audit/run.json"))
    args = parser.parse_args()
    root = PROJECT_ROOT
    audit = _Audit()

    manifest_path = root / args.manifest
    snapshot_path = root / args.snapshot
    site_path = root / args.site
    audit.check("canonical manifest exists", manifest_path.is_file(), str(manifest_path))
    audit.check("compact snapshot exists", snapshot_path.is_file(), str(snapshot_path))
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise SystemExit("required manifest/snapshot is missing")
    manifest = _read_json(manifest_path)
    snapshot = _read_json(snapshot_path)
    manifest_sha = _sha256(manifest_path)
    audit.check("manifest protocol", manifest.get("protocol_version") == PROTOCOL_VERSION, manifest.get("protocol_version"))
    audit.check("manifest sha matches snapshot", manifest_sha == snapshot.get("manifest", {}).get("sha256"), {"actual": manifest_sha, "snapshot": snapshot.get("manifest", {}).get("sha256")})
    audit.check("manifest test lock", manifest.get("test_locked_sites") == [LOCKED_SITE], manifest.get("test_locked_sites"))
    _check_adapter(manifest.get("adapter"), "manifest adapter", audit)
    targets, scene_queries, validation_records = _expected_targets(manifest)
    validation_queries = len(targets) // len(RADII)
    train_records = manifest.get("records", {}).get("train", [])
    audit.check("validation query count", validation_queries == int(snapshot.get("manifest", {}).get("validation_queries", -1)), validation_queries)
    audit.check("train query count", sum(len(r.get("queries", [])) for r in train_records) == int(snapshot.get("manifest", {}).get("train_queries", -1)), sum(len(r.get("queries", [])) for r in train_records))
    audit.check("validation record count", validation_records == int(snapshot.get("manifest", {}).get("validation_records", -1)), validation_records)
    audit.check("train record count", len(train_records) == int(snapshot.get("manifest", {}).get("train_records", -1)), len(train_records))
    all_records = list(train_records) + list(manifest.get("records", {}).get("validation", []))
    locked_records = [r for r in all_records if str(r.get("location")) == LOCKED_SITE or LOCKED_SITE in str(r)]
    audit.check("train/validation contain no locked-site records", not locked_records, len(locked_records))
    audit.check("validation scene counts", scene_queries == {"scene_00401": 1065, "scene_00427": 464}, scene_queries)

    # A rebuild of the manifest has a different wall-clock field but must have
    # exactly the same deterministic contract.  The dedicated replay command
    # performs the expensive rebuild; this check validates its persisted result.
    manifest_replay_path = root / MANIFEST_REPLAY_PATH
    audit.check("manifest replay report exists", manifest_replay_path.is_file(), str(manifest_replay_path))
    if manifest_replay_path.is_file():
        manifest_replay = _read_json(manifest_replay_path)
        audit.check(
            "manifest replay report passed",
            manifest_replay.get("passed") is True
            and manifest_replay.get("contract_equal") is True
            and manifest_replay.get("test_evaluated") is False,
            manifest_replay.get("claim_boundary"),
        )
        expected_contract_sha = _contract_sha256(manifest)
        audit.check(
            "manifest replay contract hash",
            manifest_replay.get("canonical_contract_sha256") == expected_contract_sha
            and manifest_replay.get("rebuilt_contract_sha256") == expected_contract_sha,
            {
                "actual": [
                    manifest_replay.get("canonical_contract_sha256"),
                    manifest_replay.get("rebuilt_contract_sha256"),
                ],
                "expected": expected_contract_sha,
            },
        )
        audit.check(
            "manifest replay split counts",
            manifest_replay.get("canonical_train_records") == len(train_records)
            and manifest_replay.get("canonical_validation_records") == validation_records
            and manifest_replay.get("canonical_train_queries")
            == sum(len(r.get("queries", [])) for r in train_records)
            and manifest_replay.get("canonical_validation_queries") == validation_queries,
            manifest_replay,
        )
        audit.check("manifest replay has no locked records", manifest_replay.get("locked_record_count") == 0, manifest_replay.get("locked_record_count"))

    controls = {str(entry.get("id")): entry for entry in snapshot.get("controls", []) if isinstance(entry, dict)}
    specs = [
        ("conpath_stochastic", Path("results/unscenes3d_ground_valid_f16"), False, True),
        ("independent_stochastic", Path("results/unscenes3d_ground_valid_independent_f16"), False, False),
        ("conpath_mean_map_k128", Path("results/unscenes3d_ground_valid_mean_map_k128"), True, False),
        ("independent_mean_map_k128", Path("results/unscenes3d_ground_valid_independent_mean_map_k128"), True, False),
        ("s4c_inspired_coordinate", Path("results/unscenes3d_s4c_coordinate_f16_v2"), True, False),
    ]
    for control_id, directory, csv_required, allow_legacy in specs:
        entry = controls.get(control_id)
        audit.check(f"snapshot control present: {control_id}", entry is not None, control_id)
        if entry is None:
            continue
        for seed in SEEDS:
            _run_event_control(root=root, run_dir=directory / f"seed{seed}", snapshot_entry=entry, targets=targets, validation_records=validation_records, validation_queries=validation_queries, manifest_sha=manifest_sha, audit=audit, csv_required=csv_required, allow_legacy_protocol=allow_legacy)

    # Validate the deterministic K curve against the three generated reports.
    k_entry = snapshot.get("k_sensitivity", {})
    for k in (32, 64, 128):
        report_path = root / Path(f"results/unscenes3d_ground_valid_mean_map_k{k}/seed20260831/run.json")
        audit.check(f"K={k} report exists", report_path.is_file(), str(report_path))
        if not report_path.is_file():
            continue
        report = _read_json(report_path)
        metrics = report.get("mean_map", {}).get("metrics", {})
        event = report.get("event_metrics", {})
        expected_map = k_entry.get("posterior_mean_hidden_map_brier", {}).get(str(k))
        expected_event = k_entry.get("thresholded_event_brier", {}).get(str(k))
        audit.check(f"K={k} hidden-map Brier", _close(metrics.get("scene_weighted_brier"), expected_map, 2e-8), {"actual": metrics.get("scene_weighted_brier"), "expected": expected_map})
        audit.check(f"K={k} event Brier", _close(event.get("scene_weighted_brier"), expected_event, 2e-8), {"actual": event.get("scene_weighted_brier"), "expected": expected_event})
        audit.check(f"K={k} monotonicity", event.get("radius_monotonicity_violations") == 0, event.get("radius_monotonicity_violations"))

    # Radius-prior is a deliberately simple, train-fitted comparator.
    prior = snapshot.get("radius_prior_control", {})
    probabilities = prior.get("probabilities", {})
    audit.check("radius-prior probabilities nested", float(probabilities.get("0", 1.0)) >= float(probabilities.get("1", 0.0)) >= float(probabilities.get("2", 0.0)), probabilities)
    audit.check("radius-prior is validation-only", snapshot.get("test_evaluated") is False, snapshot.get("test_evaluated"))

    qualitative = snapshot.get("qualitative", {})
    q_report = root / Path(str(qualitative.get("report", "")))
    audit.check("qualitative report exists", q_report.is_file(), str(q_report))
    if q_report.is_file():
        qr = _read_json(q_report)
        _check_common_report(qr, path=q_report, manifest_sha=manifest_sha, validation_records=validation_records, validation_queries=validation_queries, audit=audit)
        _check_adapter(qr.get("adapter"), "qualitative adapter", audit)
        q_checkpoint = qr.get("checkpoint", {})
        if isinstance(q_checkpoint, dict):
            _artifact_hash_check(root / str(q_checkpoint.get("path", "")), str(q_checkpoint.get("sha256")), None, audit, "qualitative checkpoint")
        for panel in ("positive", "failure"):
            item = qr.get("panels", {}).get(panel, {})
            panel_path = root / Path(str(item.get("path", "")))
            _artifact_hash_check(panel_path, None, None, audit, f"qualitative {panel} image")
            if panel_path.is_file():
                audit.check(f"qualitative {panel} PNG dimensions", _png_dimensions(panel_path) == (1800, 1280), _png_dimensions(panel_path))
            target = item.get("target")
            prediction = item.get("prediction")
            if panel == "failure":
                audit.check("qualitative failure is actually false-safe", target == [False, False, False] and prediction == [True, True, True], {"target": target, "prediction": prediction})
            else:
                audit.check("qualitative positive is actually true-positive", target == [True, True, True] and prediction == [True, True, True], {"target": target, "prediction": prediction})
    audit.check("snapshot qualitative paths", qualitative.get("positive_asset") == "assets/unscenes3d_qualitative_positive.png" and qualitative.get("failure_asset") == "assets/unscenes3d_qualitative_failure.png", qualitative)

    # Compact snapshot fields are checked against the current generated image/report,
    # while the site check catches stale/missing relative links.
    _check_site(root, args.site, audit)
    for source, published in (
        (Path("results/unscenes3d_controls_audit/run.json"), Path("site/data/unscenes3d_controls_audit.json")),
        (Path("results/unscenes3d_manifest_replay_audit/report.json"), Path("site/data/unscenes3d_manifest_replay_audit.json")),
        (Path("results/unscenes3d_replay_audit/report.json"), Path("site/data/unscenes3d_replay_audit.json")),
    ):
        source_path, published_path = root / source, root / published
        audit.check(f"published evidence exists: {published}", published_path.is_file(), str(published_path))
        if source_path.is_file() and published_path.is_file():
            audit.check(
                f"published evidence matches result: {published}",
                _sha256(source_path) == _sha256(published_path),
                {"source": str(source), "published": str(published)},
            )
    replay_report_path = root / Path("results/unscenes3d_replay_audit/report.json")
    audit.check("GPU replay audit exists", replay_report_path.is_file(), str(replay_report_path))
    if replay_report_path.is_file():
        replay_report = _read_json(replay_report_path)
        audit.check("GPU replay audit passed", replay_report.get("passed") is True and replay_report.get("test_evaluated") is False, replay_report.get("failures"))
    for asset in (
        "site/assets/unscenes3d_qualitative_positive.png",
        "site/assets/unscenes3d_qualitative_failure.png",
        "site/assets/unscenes3d_calibration_reliability.svg",
        "site/assets/unscenes3d_calibration_false_safe.svg",
        "site/data/unscenes3d_qualitative_validation.json",
        "site/data/unscenes3d_calibration_validation.json",
        "site/data/unscenes3d_controls_audit.json",
        "site/data/unscenes3d_manifest_replay_audit.json",
        "site/data/unscenes3d_replay_audit.json",
    ):
        audit.check(f"tracked site artifact exists: {asset}", (root / asset).is_file(), asset)
    calibration_path = root / Path("site/data/unscenes3d_calibration_validation.json")
    if calibration_path.is_file():
        calibration = _read_json(calibration_path)
        audit.check("calibration snapshot validation-only", calibration.get("protocol_version") == PROTOCOL_VERSION and calibration.get("test_evaluated") is False, calibration.get("claim_boundary"))
        audit.check("calibration snapshot row count", all(int(method.get("event_row_count", -1)) == validation_queries * len(RADII) for method in calibration.get("methods", []) if isinstance(method, dict)), calibration.get("methods"))
        prediction_paths = [
            path
            for method in calibration.get("methods", [])
            if isinstance(method, dict)
            for path in method.get("prediction_paths", [])
        ]
        audit.check(
            "calibration published paths are portable",
            all(isinstance(path, str) and not Path(path).is_absolute() and str(PROJECT_ROOT) not in path for path in prediction_paths),
            prediction_paths,
        )

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "unscenes3d_controls_audit",
        "protocol_version": PROTOCOL_VERSION,
        "validation_only": True,
        "test_evaluated": False,
        "manifest": {"path": str(args.manifest), "sha256": manifest_sha, "validation_records": validation_records, "validation_queries": validation_queries},
        "checks": _portable_strings(audit.checks),
        "warnings": _portable_strings(audit.warnings),
        "failures": _portable_strings(audit.failures),
        "passed": not audit.failures,
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "passed": not audit.failures, "check_count": len(audit.checks), "warning_count": len(audit.warnings), "failure_count": len(audit.failures)}, indent=2, sort_keys=True))
    if audit.failures:
        for failure in audit.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
