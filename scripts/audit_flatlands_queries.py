#!/usr/bin/env python3
"""Run a bounded target-blind FlatLands mask and natural-query audit directly from ZIP."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping
from zipfile import ZipFile

import numpy as np

from pathrel.flatlands import (
    FLATLANDS_ARCHIVE_BYTES,
    FLATLANDS_ARCHIVE_SHA256,
    sha256_file,
)
from pathrel.flatlands_query import (
    DEFAULT_FOOTPRINT_RADII_M,
    DEFAULT_QUERY_ANGLES_DEG,
    DEFAULT_QUERY_DISTANCES_M,
    PROVENANCE_MANIFEST_SHA256,
    ManifestObservation,
    construct_natural_queries,
    decode_binary_grayscale_png,
    load_provenance_manifest,
    mask_relation_counts,
    radii_m_to_cells,
    score_natural_queries,
    select_scene_observations,
    sha256_path,
)


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/p1_flatlands_provenance_manifest/provenance_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/p1_flatlands_query_audit_bounded")
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--scenes-per-stratum", type=int, default=32)
    parser.add_argument(
        "--query-distances-m",
        type=parse_float_list,
        default=DEFAULT_QUERY_DISTANCES_M,
        help="Comma-separated metric polar-stencil distances.",
    )
    parser.add_argument(
        "--query-angles-deg",
        type=parse_int_list,
        default=DEFAULT_QUERY_ANGLES_DEG,
        help="Comma-separated polar-stencil angles.",
    )
    parser.add_argument(
        "--footprint-radii-m",
        type=parse_float_list,
        default=DEFAULT_FOOTPRINT_RADII_M,
        help="Comma-separated physical disk radii; must map to integer cells.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--minimum-failure-rate", type=float, default=0.10)
    parser.add_argument("--minimum-retained-queries", type=int, default=50)
    parser.add_argument("--minimum-retained-scenes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_csv_atomic(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return count


def git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, check=False, text=True
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else None,
    }


def stable_group_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\x1f".join([str(seed), *parts]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def summarize_queries(
    records: list[dict[str, object]],
    *,
    radii_cells: tuple[int, ...],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    selection = Counter(str(row["selection_status"]) for row in records)
    target = Counter(str(row["target_status"]) for row in records)
    retained_statuses = {
        "disconnected_radius_zero",
        "footprint_failure",
        "high_clearance_positive",
    }
    retained = [row for row in records if row["target_status"] in retained_statuses]
    failures = [
        row
        for row in retained
        if row["target_status"] in {"disconnected_radius_zero", "footprint_failure"}
    ]
    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in retained:
        by_scene[str(row["global_id"])].append(row)
    scene_rates = np.asarray(
        [
            np.mean(
                [
                    item["target_status"]
                    in {"disconnected_radius_zero", "footprint_failure"}
                    for item in scene_rows
                ]
            )
            for scene_rows in by_scene.values()
        ],
        dtype=np.float64,
    )
    if scene_rates.size:
        point = float(scene_rates.mean())
        generator = np.random.default_rng(seed)
        draws = generator.integers(
            0, scene_rates.size, size=(bootstrap_samples, scene_rates.size)
        )
        bootstrap = scene_rates[draws].mean(axis=1)
        interval = [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ]
    else:
        point = None
        interval = [None, None]

    reachable_by_radius: dict[str, dict[str, object]] = {}
    for radius in radii_cells:
        field = f"reachable_r{radius}_cells"
        values = [bool(row[field]) for row in retained]
        reachable_by_radius[str(radius)] = {
            "reachable": int(sum(values)),
            "total": len(values),
            "rate": float(np.mean(values)) if values else None,
        }
    selected_count = selection["selected"]
    invalid_endpoints = target["target_invalid_start"] + target["target_invalid_goal"]
    return {
        "candidate_queries": len(records),
        "selection_status_counts": dict(sorted(selection.items())),
        "selected_queries": selected_count,
        "target_status_counts": dict(sorted(target.items())),
        "target_invalid_endpoints": invalid_endpoints,
        "target_invalid_endpoint_rate_among_selected": (
            invalid_endpoints / selected_count if selected_count else None
        ),
        "retained_valid_endpoint_queries": len(retained),
        "failure_queries": len(failures),
        "query_weighted_failure_rate": len(failures) / len(retained) if retained else None,
        "scene_count_with_retained_queries": len(scene_rates),
        "scene_weighted_failure_rate": point,
        "scene_weighted_failure_rate_bootstrap_95": interval,
        "reachable_by_radius_cells": reachable_by_radius,
    }


def aggregate_mask_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    total: dict[str, int] = defaultdict(int)
    for row in rows:
        counts = row["mask_counts"]
        assert isinstance(counts, dict)
        for key, value in counts.items():
            total[str(key)] += int(value)
    return dict(sorted(total.items()))


def main() -> None:
    args = parse_args()
    if args.scenes_per_stratum <= 0:
        raise SystemExit("--scenes-per-stratum must be positive")
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    if not (0 <= args.minimum_failure_rate <= 1):
        raise SystemExit("--minimum-failure-rate must lie in [0,1]")
    if args.minimum_retained_queries <= 0:
        raise SystemExit("--minimum-retained-queries must be positive")
    if args.minimum_retained_scenes <= 0:
        raise SystemExit("--minimum-retained-scenes must be positive")
    if args.minimum_retained_scenes > args.scenes_per_stratum:
        raise SystemExit("--minimum-retained-scenes cannot exceed --scenes-per-stratum")

    archive_path = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing report: {report_path}")
    if not archive_path.is_file() or not manifest_path.is_file():
        raise SystemExit("archive and provenance manifest must both exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    if args.overwrite:
        for name in (
            "progress.jsonl",
            "failure.json",
            "selected_observations.csv",
            "queries.csv",
            "report.json.tmp",
            "selected_observations.csv.tmp",
            "queries.csv.tmp",
        ):
            (output_dir / name).unlink(missing_ok=True)

    def progress(record: Mapping[str, object]) -> None:
        payload = {"time_utc": datetime.now(timezone.utc).isoformat(), **record}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    progress({"event": "start", "archive": str(archive_path), "manifest": str(manifest_path)})
    try:
        archive_bytes = archive_path.stat().st_size
        archive_sha = sha256_file(archive_path)
        archive_matches = bool(
            archive_bytes == FLATLANDS_ARCHIVE_BYTES
            and archive_sha == FLATLANDS_ARCHIVE_SHA256
        )
        manifest_sha = sha256_path(manifest_path)
        manifest_matches = manifest_sha == PROVENANCE_MANIFEST_SHA256
        progress(
            {
                "event": "input_verification",
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
                "archive_matches": archive_matches,
                "manifest_sha256": manifest_sha,
                "manifest_matches": manifest_matches,
            }
        )
        if not archive_matches or not manifest_matches:
            raise ValueError("archive or provenance manifest does not match the frozen input")

        manifest = load_provenance_manifest(manifest_path)
        selected = select_scene_observations(
            manifest, scenes_per_stratum=args.scenes_per_stratum, seed=args.seed
        )
        selected_path = output_dir / "selected_observations.csv"
        selected_fields = tuple(asdict(selected[0]).keys())
        selected_rows = []
        for row in selected:
            serialized = asdict(row)
            serialized["camera_px"] = json.dumps(row.camera_px, separators=(",", ":"))
            selected_rows.append(serialized)
        selected_count = write_csv_atomic(selected_path, selected_fields, selected_rows)
        progress({"event": "selection_complete", "selected_observations": selected_count})

        all_query_rows: list[dict[str, object]] = []
        observation_reports: list[dict[str, object]] = []
        radius_cell_sets: set[tuple[int, ...]] = set()
        with ZipFile(archive_path) as archive:
            for observation_index, row in enumerate(selected, start=1):
                root = row.packet_directory
                member_names = {
                    name: f"{root}/{name}"
                    for name in (
                        "observed_floor.png",
                        "floor_map.png",
                        "unobserved.png",
                        "epistemic_mask.png",
                    )
                }
                missing: list[str] = []
                for member_name in member_names.values():
                    try:
                        archive.getinfo(member_name)
                    except KeyError:
                        missing.append(member_name)
                if missing:
                    raise ValueError(
                        f"selected packet {row.global_id} is incomplete: {sorted(missing)}"
                    )

                # Freeze all queries before reading the target member.  This ordering is a
                # deliberate guard against accidental target-dependent endpoint selection.
                observed = decode_binary_grayscale_png(
                    archive.read(member_names["observed_floor.png"])
                )
                unobserved = decode_binary_grayscale_png(
                    archive.read(member_names["unobserved.png"])
                )
                epistemic = decode_binary_grayscale_png(
                    archive.read(member_names["epistemic_mask.png"])
                )
                natural_queries = construct_natural_queries(
                    observed,
                    unobserved,
                    epistemic,
                    camera_px=row.camera_px,
                    resolution_m=row.resolution,
                    distances_m=args.query_distances_m,
                    angles_deg=args.query_angles_deg,
                )

                floor = decode_binary_grayscale_png(archive.read(member_names["floor_map.png"]))
                if any(array.shape != floor.shape for array in (observed, unobserved, epistemic)):
                    raise ValueError(f"packet {row.global_id} has misaligned image shapes")
                radii_cells = radii_m_to_cells(args.footprint_radii_m, row.resolution)
                radius_cell_sets.add(radii_cells)
                scored = score_natural_queries(
                    floor, epistemic, natural_queries, radii_cells=radii_cells
                )
                counts = mask_relation_counts(observed, floor, unobserved, epistemic)
                observation_reports.append(
                    {
                        "global_id": row.global_id,
                        "provenance_split": row.provenance_split,
                        "source_dataset": row.source_dataset,
                        "scene_id": row.scene_id,
                        "resolution": row.resolution,
                        "radii_cells": list(radii_cells),
                        "mask_counts": counts,
                    }
                )
                for query_record in scored:
                    all_query_rows.append(
                        {
                            "global_id": row.global_id,
                            "provenance_split": row.provenance_split,
                            "source_dataset": row.source_dataset,
                            "scene_id": row.scene_id,
                            "resolution_m": row.resolution,
                            **query_record,
                        }
                    )
                progress(
                    {
                        "event": "observation_complete",
                        "index": observation_index,
                        "total": len(selected),
                        "global_id": row.global_id,
                        "queries": len(scored),
                    }
                )
        if len(radius_cell_sets) != 1:
            raise ValueError(
                f"bounded sample maps physical radii to inconsistent cell sets: {radius_cell_sets}"
            )
        radii_cells = next(iter(radius_cell_sets))

        query_path = output_dir / "queries.csv"
        query_fields = tuple(all_query_rows[0].keys())
        query_rows = write_csv_atomic(query_path, query_fields, all_query_rows)

        groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        split_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in all_query_rows:
            split = str(record["provenance_split"])
            source = str(record["source_dataset"])
            groups[(split, source)].append(record)
            split_groups[split].append(record)
        summaries_by_stratum: dict[str, dict[str, object]] = {}
        for (split, source), records in sorted(groups.items()):
            key = f"{split}/{source}"
            summaries_by_stratum[key] = summarize_queries(
                records,
                radii_cells=radii_cells,
                bootstrap_samples=args.bootstrap_samples,
                seed=stable_group_seed(args.seed, split, source),
            )
        summaries_by_split = {
            split: summarize_queries(
                records,
                radii_cells=radii_cells,
                bootstrap_samples=args.bootstrap_samples,
                seed=stable_group_seed(args.seed, split),
            )
            for split, records in sorted(split_groups.items())
        }

        mask_totals = aggregate_mask_counts(observation_reports)
        png_and_alignment_passed = len(observation_reports) == len(selected)
        observed_target_agreement_passed = mask_totals.get("observed_not_floor", 0) == 0
        unobserved_polarity_passed = mask_totals.get("observed_unobserved_overlap", 0) == 0
        target_invalid_start_count = sum(
            row["target_status"] == "target_invalid_start" for row in all_query_rows
        )
        mask_semantics_passed = bool(
            png_and_alignment_passed
            and observed_target_agreement_passed
            and unobserved_polarity_passed
            and target_invalid_start_count == 0
        )

        gated_strata: dict[str, dict[str, object]] = {}
        for key, summary in summaries_by_stratum.items():
            split, _ = key.split("/", 1)
            if split not in {"validation", "test"}:
                continue
            failure_rate = summary["scene_weighted_failure_rate"]
            retained = int(summary["retained_valid_endpoint_queries"])
            scenes = int(summary["scene_count_with_retained_queries"])
            passed = bool(
                failure_rate is not None
                and float(failure_rate) >= args.minimum_failure_rate
                and retained >= args.minimum_retained_queries
                and scenes >= args.minimum_retained_scenes
            )
            gated_strata[key] = {
                "passed": passed,
                "scene_weighted_failure_rate": failure_rate,
                "minimum_failure_rate": args.minimum_failure_rate,
                "retained_valid_endpoint_queries": retained,
                "minimum_retained_queries": args.minimum_retained_queries,
                "scenes_with_retained_queries": scenes,
                "minimum_retained_scenes": args.minimum_retained_scenes,
            }
        query_balance_passed = bool(gated_strata and all(row["passed"] for row in gated_strata.values()))
        p1_bounded_gate_passed = bool(mask_semantics_passed and query_balance_passed)

        selected_sha = sha256_path(selected_path)
        query_sha = sha256_path(query_path)
        report: dict[str, object] = {
            "protocol": {
                "command": sys.argv,
                "git": git_state(),
                "seed": args.seed,
                "scenes_per_split_source_stratum": args.scenes_per_stratum,
                "selection": (
                    "Stable SHA-256 rank of provenance scenes, then stable SHA-256 rank of "
                    "observations within each scene; one observation per scene."
                ),
                "target_blind_query_construction": True,
                "query_construction_order": (
                    "Read observed_floor, unobserved, and epistemic_mask; freeze camera-centered "
                    "metric polar queries; only then read floor_map for scoring."
                ),
                "camera_px_convention": "metadata [x,y] converted to array [row=y,col=x]",
                "start_rule": (
                    "camera cell when observed-floor and epistemic-valid, otherwise nearest "
                    "observed-floor epistemic-valid cell with deterministic row/column tie-break"
                ),
                "goal_rule": (
                    "metric polar stencil retained only inside raster, epistemic mask, and "
                    "unobserved mask; floor_map never selects or rejects goals"
                ),
                "query_distances_m": list(args.query_distances_m),
                "query_angles_deg": list(args.query_angles_deg),
                "footprint_radii_m": list(args.footprint_radii_m),
                "footprint_radii_cells": list(radii_cells),
                "connectivity": "exact binary four-neighbor paths with integer disk footprints",
                "outside_epistemic_policy": (
                    "always invalid/blocked for target scoring; never coerced to free"
                ),
                "bootstrap": (
                    f"{args.bootstrap_samples} deterministic scene resamples; one observation "
                    "per sampled scene"
                ),
                "gate_scope": (
                    "every provenance validation/test source stratum, including ScanNet++ OOD"
                ),
                "minimum_scene_weighted_failure_rate": args.minimum_failure_rate,
                "minimum_retained_queries_per_gated_stratum": args.minimum_retained_queries,
                "minimum_retained_scenes_per_gated_stratum": args.minimum_retained_scenes,
            },
            "inputs": {
                "archive": {
                    "path": str(archive_path),
                    "bytes": archive_bytes,
                    "sha256": archive_sha,
                    "matches_frozen_release": archive_matches,
                },
                "provenance_manifest": {
                    "path": str(manifest_path),
                    "rows": len(manifest),
                    "sha256": manifest_sha,
                    "expected_sha256": PROVENANCE_MANIFEST_SHA256,
                    "matches_frozen_manifest": manifest_matches,
                },
            },
            "outputs": {
                "selected_observations": {
                    "path": str(selected_path),
                    "rows": selected_count,
                    "sha256": selected_sha,
                },
                "queries": {
                    "path": str(query_path),
                    "rows": query_rows,
                    "sha256": query_sha,
                },
            },
            "sample": {
                "selected_observations": selected_count,
                "split_source_strata": len(groups),
                "observations_by_stratum": dict(
                    sorted(
                        Counter(
                            f"{row.provenance_split}/{row.source_dataset}" for row in selected
                        ).items()
                    )
                ),
            },
            "mask_semantics": {
                "aggregate_counts": mask_totals,
                "observed_target_agreement_passed": observed_target_agreement_passed,
                "observed_unobserved_disjoint_passed": unobserved_polarity_passed,
                "target_invalid_start_count": target_invalid_start_count,
                "epistemic_boundary_disagreements_are_excluded": True,
                "passed": mask_semantics_passed,
            },
            "query_summary_by_split": summaries_by_split,
            "query_summary_by_split_source": summaries_by_stratum,
            "gates": {
                "mask_semantics_passed": mask_semantics_passed,
                "query_balance_by_gated_stratum": gated_strata,
                "query_balance_passed": query_balance_passed,
                "p1_bounded_query_gate_passed": p1_bounded_gate_passed,
                "training_authorized": False,
                "paper_result": False,
            },
            "interpretation": (
                "The bounded, non-official provenance-split audit passes mask semantics and "
                "natural-query balance. Training remains separately prohibited until this "
                "bounded result is reviewed and a full baseline protocol is frozen."
                if p1_bounded_gate_passed
                else (
                    "Mask polarity is usable only with epistemic masking, but at least one "
                    "provenance validation/test source stratum fails the frozen natural-query "
                    "balance/sample gate. FlatLands remains NO-GO as the main event benchmark."
                    if mask_semantics_passed
                    else (
                        "The bounded sample fails the mask-semantics gate. FlatLands remains "
                        "NO-GO for query training or event claims."
                    )
                )
            ),
        }
        atomic_json(report_path, report)
        progress(
            {
                "event": "complete",
                "mask_semantics_passed": mask_semantics_passed,
                "query_balance_passed": query_balance_passed,
                "p1_bounded_query_gate_passed": p1_bounded_gate_passed,
            }
        )
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "selected_observations": selected_count,
                    "queries": query_rows,
                    **report["gates"],
                },
                indent=2,
            )
        )
    except BaseException as error:
        failure = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "archive": str(archive_path),
            "manifest": str(manifest_path),
            "command": sys.argv,
        }
        atomic_json(output_dir / "failure.json", failure)
        progress({"event": "failure", **failure})
        raise


if __name__ == "__main__":
    main()
