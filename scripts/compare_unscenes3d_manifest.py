#!/usr/bin/env python3
"""Rebuild and compare the frozen UnScenes3D train/validation manifest.

The manifest records a small wall-clock ``generation_seconds`` field, so a byte
hash of a newly rebuilt file is expected to differ from the frozen artifact.  This
read-only check removes that runtime field, hashes the canonical JSON contract,
and verifies exact equality of every split, query, adapter, and claim-boundary
field.  The builder has an explicit train/validation site allow-list and never
opens the locked ``location_6`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json")
PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.2"
LOCKED_SITE = "location_6"


def _canonical_payload(value: dict[str, Any]) -> bytes:
    # ``generation_seconds`` is intentionally runtime metadata rather than part
    # of the query contract.  Keep the normalization explicit and narrow so a
    # changed query, adapter, or split cannot be hidden by this comparison.
    normalized = dict(value)
    normalized.pop("generation_seconds", None)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"),
    )
    parser.add_argument("--endpoint-policy", choices=("blocked", "ground"), default="ground")
    parser.add_argument("--start-selection", choices=("valid", "observed_free"), default="valid")
    parser.add_argument("--ground-margin-m", type=float, default=0.35)
    parser.add_argument("--ground-bin-size-m", type=float, default=1.2)
    parser.add_argument("--ground-lateral-limit-m", type=float, default=20.0)
    parser.add_argument("--ground-quantile", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/unscenes3d_manifest_replay_audit/report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts.build_unscenes3d_manifest import build_manifest  # noqa: E402

    canonical_path = (ROOT / args.canonical).resolve()
    raw_root = (ROOT / args.raw_root).resolve()
    label_root = (ROOT / args.label_root).resolve()
    if not canonical_path.is_file():
        raise SystemExit(f"canonical manifest is missing: {canonical_path}")

    canonical = _read_object(canonical_path)
    rebuilt = build_manifest(
        raw_root,
        label_root,
        endpoint_policy=args.endpoint_policy,
        start_selection=args.start_selection,
        ground_margin_m=args.ground_margin_m,
        ground_bin_size_m=args.ground_bin_size_m,
        ground_lateral_limit_m=args.ground_lateral_limit_m,
        ground_quantile=args.ground_quantile,
    )
    canonical_contract = _canonical_payload(canonical)
    rebuilt_contract = _canonical_payload(rebuilt)
    canonical_sha = hashlib.sha256(canonical_contract).hexdigest()
    rebuilt_sha = hashlib.sha256(rebuilt_contract).hexdigest()

    locked_records: list[object] = []
    for split in ("train", "validation"):
        for record in canonical.get("records", {}).get(split, []):
            if str(record.get("location")) == LOCKED_SITE or LOCKED_SITE in str(record):
                locked_records.append(record)
    contract_equal = canonical_contract == rebuilt_contract
    report = {
        "schema_version": 1,
        "kind": "unscenes3d_manifest_replay_audit",
        "protocol_version": PROTOCOL_VERSION,
        "validation_only": True,
        "test_evaluated": False,
        "canonical_path": str(args.canonical),
        "ignored_runtime_fields": ["generation_seconds"],
        "canonical_contract_sha256": canonical_sha,
        "rebuilt_contract_sha256": rebuilt_sha,
        "contract_equal": contract_equal,
        "canonical_protocol": canonical.get("protocol_version"),
        "canonical_adapter": canonical.get("adapter"),
        "canonical_train_records": len(canonical.get("records", {}).get("train", [])),
        "canonical_validation_records": len(canonical.get("records", {}).get("validation", [])),
        "canonical_train_queries": sum(
            len(record.get("queries", []))
            for record in canonical.get("records", {}).get("train", [])
        ),
        "canonical_validation_queries": sum(
            len(record.get("queries", []))
            for record in canonical.get("records", {}).get("validation", [])
        ),
        "locked_record_count": len(locked_records),
        "claim_boundary": "Train/validation manifest replay only; location_6 files and test labels were not read.",
        "passed": bool(
            contract_equal
            and canonical.get("protocol_version") == PROTOCOL_VERSION
            and canonical.get("test_locked_sites") == [LOCKED_SITE]
            and not locked_records
        ),
    }
    output = (ROOT / args.output).resolve()
    _atomic_json(output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": report["passed"],
                "contract_equal": contract_equal,
                "contract_sha256": canonical_sha,
                "locked_record_count": len(locked_records),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
