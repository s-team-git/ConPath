"""Export existing development event labels; never open raw maps or test assets.

This is a one-shot archive export, not an evaluator or inference entry point.
Only the 40 previously registered physical-train validation caches are allowed.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target_path = OUT / "targets.csv"
    receipt_path = OUT / "target_export_receipt.json"
    if target_path.exists() or receipt_path.exists():
        raise FileExistsError("Refuse to overwrite an existing label export")
    snapshot_path = ROOT / "results/paper_validation_snapshot.json"
    assert sha(snapshot_path) == "8a3b81ef1f01dec72e57f92d7c02d39ed0e3697e12389143a755a836cc4bbc32"
    snapshot = json.loads(snapshot_path.read_text())
    registry = {r["path"]: r for r in snapshot["source_registry"]}
    data = ROOT / "results/parent_group_pilot_v1/data"
    metadata = {}
    for name in ("protocol.json", "selected.csv", "seal.json"):
        p = data / name
        expected = registry[str(p.relative_to(ROOT))]["sha256"]
        assert sha(p) == expected
        metadata[str(p.relative_to(ROOT))] = expected
    protocol = json.loads((data / "protocol.json").read_text())
    assert protocol["physical_archive_split"] == "train"
    assert protocol["query"]["radii_cells"] == [0, 10, 20]
    seal = json.loads((data / "seal.json").read_text())["files"]
    with (data / "selected.csv").open(newline="") as f:
        selected = list(csv.DictReader(f))
    assert len(selected) == 165
    assert all(r["archive_split"] == "train" for r in selected)
    partitions = {s: {r["parent_group"] for r in selected if r["candidate_split"] == s}
                  for s in ("train", "calibration", "validation")}
    assert [len(partitions[s]) for s in partitions] == [100, 25, 40]
    assert not (partitions["train"] & partitions["validation"])
    assert not (partitions["calibration"] & partitions["validation"])
    validation = [r for r in selected if r["candidate_split"] == "validation"]
    assert len(validation) == 40
    rows, packets = [], []
    allowed_role = "sealed physical-train development-validation packet"
    for item in validation:
        global_id = item["global_id"]
        assert global_id.startswith("obs_") and global_id[4:].isdigit()
        p = data / "packets" / (global_id + ".npz")
        rel = str(p.relative_to(ROOT))
        registered = registry[rel]
        assert registered["role"] == allowed_role
        digest = sha(p)
        assert digest == registered["sha256"] == seal["packets/" + p.name]
        # NPZ arrays are loaded lazily. Do not request target/observation maps.
        with np.load(p, allow_pickle=False) as cache:
            candidates = cache["candidate_indices"]
            targets = cache["targets"]
        assert targets.shape == (len(candidates), 3)
        assert np.all((targets == 0) | (targets == 1))
        assert len(set(int(c) for c in candidates)) == len(candidates)
        for q, candidate in enumerate(candidates):
            for j, radius in enumerate((0, 10, 20)):
                rows.append(dict(global_id=global_id, parent_group=item["parent_group"],
                                 source_dataset=item["source_dataset"],
                                 candidate_index=int(candidate), radius_cells=radius,
                                 target=int(targets[q, j])))
        packets.append(dict(path=rel, sha256=digest, archive_split="train",
                            candidate_split="validation", arrays_decoded=["candidate_indices", "targets"],
                            events=int(targets.size)))
    keys = {(r["global_id"], r["candidate_index"], r["radius_cells"]) for r in rows}
    assert len(rows) == len(keys) == 1545
    assert len({r["parent_group"] for r in rows}) == 40
    tmp = target_path.with_suffix(".csv.tmp")
    with tmp.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target_path)
    receipt = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                   purpose="Portable export of existing frozen development labels, no new evaluation",
                   snapshot_sha256=sha(snapshot_path), metadata=metadata, packets=packets,
                   events=len(rows), parents=40, queries=515, radii_cells=[0, 10, 20],
                   targets_sha256=sha(target_path), export_script_sha256=sha(Path(__file__)),
                   map_arrays_decoded=0, raw_archive_assets_read=0, final_test_assets_read=0,
                   physical_test_assets_read=0, location_6_assets_read=0,
                   model_inference_runs=0, training_runs=0,
                   caveat="Development validation was already used for model development; not a final test.")
    with receipt_path.open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"targets": str(target_path), "events": len(rows), "parents": 40,
                      "sha256": sha(target_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
