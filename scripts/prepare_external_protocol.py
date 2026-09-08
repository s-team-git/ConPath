#!/usr/bin/env python3
"""Freeze train-only engineering inputs and audit FlatLands crop metadata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from PIL import Image


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rank(text):
    return sha(("external-protocol-v1/20260908/" + text).encode())


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_protocol_v1"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    audit = Path("results/cogniplan_training_asset_audit_v1")
    full = list(csv.DictReader((audit / "full_maps.csv").open()))
    partial = list(csv.DictReader((audit / "partial_maps.csv").open()))
    archive = Path("data/raw/cogniplan/paper_model_exploration/maps_train_inpaint.tar.gz")
    if sha(archive.read_bytes()) != "cd4d05dbcce60c1412cca2af4de6d910754f8b8e8704e3d34f998e2f44045589":
        raise ValueError("Unexpected CogniPlan training archive")
    records = {r["member"]: r for r in full}
    groups = defaultdict(list)
    # Read full training maps only; D4 grouping guards against rotated/flipped copies.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name not in records:
                continue
            raw = tar.extractfile(member).read()
            row = records[member.name]
            assert sha(raw) == row["png_sha256"]
            target = np.asarray(Image.open(io.BytesIO(raw))) != 127
            assert sha(np.where(target, 255, 1).astype(np.uint8).tobytes()) == row["target_sha256"]
            variants = [sha(np.ascontiguousarray(np.rot90(t, k)).tobytes())
                        for t in (target, np.fliplr(target)) for k in range(4)]
            row["d4_group_sha256"] = min(variants)
            groups[min(variants)].append(row)
    assert sum(map(len, groups.values())) == len(full)
    strata = defaultdict(list)
    for group, rows in groups.items():
        strata["+".join(sorted({r["layout"] for r in rows}))].append(group)
    for stratum, group_ids in strata.items():
        group_ids.sort(key=rank)
        n = len(group_ids)
        for i, group in enumerate(group_ids):
            split = "train" if i < int(.8 * n) else "calibration" if i < int(.9 * n) else "validation"
            for row in groups[group]:
                row["split"] = split
                row["namespace"] = "cogniplan/maps_train_inpaint"
    mothers = {r["mother_id"]: r for r in full}
    for row in partial:
        mother = mothers[row["mother_id"]]
        row.update({k: mother[k] for k in ("layout", "split", "d4_group_sha256", "namespace")})
        row["target_member"] = mother["member"]
    # One target-blind hash-ranked view from each of 32 distinct training mothers.
    fixed = []
    for layout, count in (("room", 11), ("tunnel", 11), ("outdoor", 10)):
        candidates = sorted((r for r in full if r["split"] == "train" and r["layout"] == layout),
                            key=lambda r: rank(r["mother_id"]))[:count]
        for mother in candidates:
            views = [r for r in partial if r["mother_id"] == mother["mother_id"]]
            fixed.append(min(views, key=lambda r: rank(r["member"])))
    assert len(fixed) == 32 and len({r["mother_id"] for r in fixed}) == 32
    write_csv(out / "cogniplan_mothers.csv", sorted(full, key=lambda r: r["mother_id"]))
    write_csv(out / "cogniplan_observations.csv", sorted(partial, key=lambda r: r["member"]))
    write_csv(out / "cogniplan_profile32.csv", fixed)

    selection = Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv")
    train = [r for r in csv.DictReader(selection.open()) if r["provenance_split"] == "train"]
    crop_rows = []
    with zipfile.ZipFile("data/raw/flatlands/FlatLands_final_dataset.zip") as z:
        for row in train:
            raw = z.read(row["metadata_member"])
            meta = json.loads(raw)
            assert meta["provenance"]["original_split"] == "train"
            crop = meta["crop"]
            size = crop["cropped_size"]
            consistent = (crop["original_size"] == 512 and size == 256
                          and crop["crop_y"] == [192, 448] and crop["crop_x"] == [128, 384]
                          and meta["observation"]["camera_px"] == [128, 192]
                          and meta["scene"]["resolution"] == .01)
            crop_rows.append({"global_id": row["global_id"], "source": row["source_dataset"],
                              "metadata_sha256": sha(raw), "crop_consistent": consistent,
                              "resolution_metadata_m_per_cell": meta["scene"]["resolution"],
                              "crop_size_cells": size})
    assert len(train) == 160 and all(r["crop_consistent"] for r in crop_rows)
    write_csv(out / "flatlands_train_crop_audit.csv", crop_rows)
    fixed_flat = sorted(train, key=lambda r: rank(r["global_id"]))[:32]
    write_csv(out / "flatlands_profile32.csv", fixed_flat)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "test_assets_opened": False,
        "split_kind": "non-official original-training-only mother/D4-group split",
        "selection_uses_prediction_or_quality_scores": False,
        "cogniplan_mothers": dict(Counter(r["split"] for r in full)),
        "cogniplan_observations": dict(Counter(r["split"] for r in partial)),
        "cogniplan_mothers_by_layout_split": dict(Counter(r["layout"] + "/" + r["split"] for r in full)),
        "d4_groups": len(groups), "d4_duplicate_groups": sum(len(v) > 1 for v in groups.values()),
        "d4_cross_split_groups": 0, "profile_samples": 32,
        "public_checkpoint_overlap": "All original train maps were available to public weights; sanity only, never held-out claims.",
        "flatlands_train_metadata_checked": len(crop_rows),
        "flatlands_metadata_consistent_with_crop": True,
        "flatlands_physical_scale_verified": False,
        "scale_issue": "Local 512->256 crop metadata retains 0.01 m/cell and camera (128,192). Paper S2.3 instead says 512 at .01 becomes 256 at .039 via average pooling, while S2.4 describes the same crop as local metadata. Upstream physical transform is not independently verified.",
        "scale_policy": "Keep existing cell-radius results; no physical-size claims or unverified rescaling. Cell-domain interface/profiling can proceed.",
        "split_frozen_for_new_training": True, "formal_queries_and_training_recipes_frozen": False,
        "inputs": {str(p): sha(p.read_bytes()) for p in (selection, audit / "full_maps.csv", audit / "partial_maps.csv", Path(__file__))},
        "outputs": {p.name: sha(p.read_bytes()) for p in sorted(out.glob("*.csv"))},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
