#!/usr/bin/env python3
"""Audit pinned CogniPlan training archives without extracting or reading test data.

This is a data/interface audit, not an evaluation of the public checkpoint.
Pixel conversion follows mapinpaint/tools.py at the pinned official revision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
from PIL import Image

COMMIT = "444fab8d5d3b8d2b004d83865088a9d707d2e3c8"
HASHES = {
    "maps_train_inpaint.tar.gz": "cd4d05dbcce60c1412cca2af4de6d910754f8b8e8704e3d34f998e2f44045589",
    "maps_train.tar.gz": "28b7227320cfd7794bfe29d9be75a022e39f33ba98283aaac2f877fc5c1b49c5",
    "checkpoints_wgan_inpainting.tar.gz": "64f3d64cd2b524e756fe0c80fae6fe7b253ec3e26c4ad0abe3b1495ff5112e69",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def images(path):
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isfile():
                assert member.name.endswith(".png"), member.name
                raw = archive.extractfile(member).read()
                image = Image.open(io.BytesIO(raw))
                assert image.mode == "L", (member.name, image.mode)
                pixels = np.asarray(image)
                assert pixels.shape == (250, 250), (member.name, pixels.shape)
                yield Path(member.name), raw, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("data/raw/cogniplan/paper_model_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cogniplan_training_asset_audit_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, expected in HASHES.items():
        path = args.archive_root / name
        assert sha(path.read_bytes()) == expected, name
        assets.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})

    full, full_records, partial_records = {}, [], []
    seen_partial = set()
    partial_values, full_values = Counter(), Counter()
    partial_types, full_types = Counter(), Counter()
    with tarfile.open(args.archive_root / "checkpoints_wgan_inpainting.tar.gz") as archive:
        config = archive.extractfile("wgan_inpainting/config.yaml").read()
        checkpoint = archive.extractfile("wgan_inpainting/gen_00500000.pt").read()
    (args.output_dir / "released_config.yaml").write_bytes(config)
    batch_size = re.findall(r"^batch_size:\s*(\d+)\s*$", config.decode(), flags=re.MULTILINE)
    assert len(batch_size) == 1

    # First pass reads only native full maps. Test archives are never opened.
    for name, raw, pixels in images(args.archive_root / "maps_train_inpaint.tar.gz"):
        if name.parent.name != "full":
            continue
        values = set(np.unique(pixels).tolist())
        assert values <= {127, 208, 255}, (str(name), values)
        full_values.update(values)
        mother = name.stem
        kind = mother.split("_")[0]
        assert kind in {"room", "tunnel", "outdoor"} and mother not in full
        # Official target conversion: 208 (start) -> 255 (free), 127 -> 1 (blocked).
        target = np.where(pixels == 127, 1, 255).astype(np.uint8)
        full[mother] = target
        full_types[kind] += 1
        full_records.append({"mother_id": mother, "layout": kind, "member": str(name),
                             "png_sha256": sha(raw), "target_sha256": sha(target.tobytes())})

    observed_cells, conflicting_cells = 0, 0
    for name, raw, pixels in images(args.archive_root / "maps_train_inpaint.tar.gz"):
        if name.parent.name != "part":
            continue
        mother, frame = name.stem.rsplit("_", 1)
        assert mother in full and frame.isdigit() and str(name) not in seen_partial, str(name)
        seen_partial.add(str(name))
        values = set(np.unique(pixels).tolist())
        assert values <= {1, 127, 255}, (str(name), values)
        partial_values.update(values)
        known = pixels != 127
        conflicts = int(np.count_nonzero(known & (pixels != full[mother])))
        observed_cells += int(known.sum())
        conflicting_cells += conflicts
        partial_types[mother.split("_")[0]] += 1
        partial_records.append({"member": str(name), "mother_id": mother,
                                "png_sha256": sha(raw), "observed_fraction": float(known.mean()),
                                "known_target_conflicts": conflicts})

    planner_records = []
    for name, raw, pixels in images(args.archive_root / "maps_train.tar.gz"):
        values = set(np.unique(pixels).tolist())
        assert values <= {127, 208, 255}, (str(name), values)
        target = np.where(pixels == 127, 1, 255).astype(np.uint8)
        planner_records.append({"mother_id": name.stem, "layout": name.parent.name,
                                "member": str(name), "png_sha256": sha(raw),
                                "target_sha256": sha(target.tobytes())})
    by_target = defaultdict(list)
    for row in full_records:
        by_target[row["target_sha256"]].append(row["mother_id"])
    duplicate_groups = [sorted(ids) for ids in by_target.values() if len(ids) > 1]
    planner_by_name = {r["mother_id"]: r for r in planner_records}
    planner_hashes = {r["target_sha256"] for r in planner_records}
    same_names = [r for r in full_records if r["mother_id"] in planner_by_name]
    differing_names = [r["mother_id"] for r in same_names
                       if r["target_sha256"] != planner_by_name[r["mother_id"]]["target_sha256"]]
    for filename, records in (("full_maps.csv", full_records), ("partial_maps.csv", partial_records),
                              ("planner_train_maps.csv", planner_records)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda r: r["member"]))
    (args.output_dir / "duplicate_targets.json").write_text(json.dumps(duplicate_groups, indent=2) + "\n")
    report = {
        "schema_version": 1, "kind": "cogniplan_training_data_interface_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": COMMIT, "script_sha256": sha(Path(__file__).read_bytes()),
        "passed": conflicting_cells == 0, "test_assets_opened": False,
        "model_inference_run": False, "formal_comparison_ready": False, "assets": assets,
        "native_full_maps": len(full_records), "native_partial_maps": len(partial_records),
        "native_full_by_layout": dict(full_types), "native_partial_by_layout": dict(partial_types),
        "full_pixel_values": sorted(full_values), "partial_pixel_values": sorted(partial_values),
        "observed_cells_checked": observed_cells, "known_target_conflicts": conflicting_cells,
        "native_unique_target_geometries": len(by_target), "native_duplicate_target_groups": len(duplicate_groups),
        "native_mothers_without_partial": sorted(set(full) - {r["mother_id"] for r in partial_records}),
        "planner_training_maps": len(planner_records), "shared_mother_names": len(same_names),
        "shared_names_with_different_targets": differing_names,
        "native_targets_also_in_planner_training": sum(r["target_sha256"] in planner_hashes for r in full_records),
        "released_generator": {"checkpoint_sha256": sha(checkpoint), "iteration_from_filename": 500000,
                               "batch_size": int(batch_size[0]), "configuration_sha256": sha(config),
                               "configuration_file": str(args.output_dir / "released_config.yaml")},
        "next_gates": [
            "Group identical target geometries as well as mother IDs before a train/calibration/validation split.",
            "Public generator used original training maps; its outputs on a new subset are train-only sanity checks, not held-out validation.",
            "Lock query construction, radii, native four inference conditions and postprocessing before comparative evaluation.",
            "Complete native output quality checks and resource profiling; released checkpoint batch size differs from repository default.",
            "Test provenance and overlap remain unaudited; no test archive was downloaded or opened by this audit.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"assets", "released_generator", "native_mothers_without_partial", "shared_names_with_different_targets"}}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
