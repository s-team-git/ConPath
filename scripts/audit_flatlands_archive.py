#!/usr/bin/env python3
"""Audit the official FlatLands ZIP without extracting or training on it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from pathrel.flatlands import (
    FLATLANDS_ARCHIVE_BYTES,
    FLATLANDS_ARCHIVE_SHA256,
    archive_structure_gate,
    audit_metadata,
    build_archive_index,
    integrity_gate,
    metadata_integrity_gate,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/p1_flatlands_archive_audit")
    )
    parser.add_argument(
        "--metadata-limit",
        type=int,
        default=1_000,
        help="Metadata packets to inspect; 0 scans all and is required to pass the integrity gate.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing report: {report_path}")
    if not archive_path.is_file():
        raise SystemExit(f"archive not found: {archive_path}")
    if args.metadata_limit < 0:
        raise SystemExit("--metadata-limit must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    if args.overwrite:
        progress_path.unlink(missing_ok=True)
        (output_dir / "failure.json").unlink(missing_ok=True)

    def progress(record: dict[str, object]) -> None:
        payload = {"time_utc": datetime.now(timezone.utc).isoformat(), **record}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    progress({"event": "start", "archive": str(archive_path)})
    try:
        archive_bytes = archive_path.stat().st_size
        archive_sha256 = sha256_file(archive_path)
        archive_matches = (
            archive_bytes == FLATLANDS_ARCHIVE_BYTES
            and archive_sha256 == FLATLANDS_ARCHIVE_SHA256
        )
        progress(
            {
                "event": "archive_hash",
                "bytes": archive_bytes,
                "sha256": archive_sha256,
                "matches_release": archive_matches,
            }
        )
        if not archive_matches:
            raise ValueError("archive size or SHA-256 does not match the frozen release")

        with ZipFile(archive_path) as archive:
            index = build_archive_index(archive.infolist(), progress=progress)
            progress({"event": "index_complete", "packets": index.report["packet_count"]})
            metadata = audit_metadata(
                archive,
                index.metadata_members,
                limit=args.metadata_limit,
                seed=args.seed,
                progress=progress,
            )

        archive_structure_passed = archive_matches and archive_structure_gate(index.report)
        metadata_integrity_passed = metadata_integrity_gate(metadata)
        scene_split_passed = bool(metadata["scene_disjoint"])
        passed = archive_matches and integrity_gate(index.report, metadata)
        report: dict[str, object] = {
            "protocol": {
                "command": sys.argv,
                "git_head": git_head(),
                "archive": str(archive_path),
                "metadata_limit": args.metadata_limit,
                "seed": args.seed,
            },
            "archive": {
                "bytes": archive_bytes,
                "expected_bytes": FLATLANDS_ARCHIVE_BYTES,
                "sha256": archive_sha256,
                "expected_sha256": FLATLANDS_ARCHIVE_SHA256,
                "matches_release": archive_matches,
            },
            "zip_index": index.report,
            "metadata": metadata,
            "gates": {
                "archive_structure_passed": archive_structure_passed,
                "metadata_integrity_passed": metadata_integrity_passed,
                "scene_split_passed": scene_split_passed,
                "p1_prequery_gate_passed": passed,
            },
            "query_balance_gate_passed": False,
            "paper_result": False,
            "interpretation": (
                "Archive structure, metadata, and scene split passed; pixel semantics and "
                "natural-query balance "
                "remain required before extraction or training."
                if passed
                else (
                    "Archive bytes/packets/metadata are valid, but the official observation split "
                    "is not scene-disjoint; do not train on it for a cross-scene claim."
                    if archive_structure_passed and metadata_integrity_passed
                    else "Archive audit is incomplete or failed; do not extract or train."
                )
            ),
        }
        atomic_json(report_path, report)
        progress({"event": "complete", **report["gates"]})
        print(json.dumps({"report": str(report_path), **report["archive"], **report["gates"]}, indent=2))
    except BaseException as error:
        failure = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "archive": str(archive_path),
            "command": sys.argv,
        }
        atomic_json(output_dir / "failure.json", failure)
        progress({"event": "failure", **failure})
        raise


if __name__ == "__main__":
    main()
