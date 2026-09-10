#!/usr/bin/env python3
"""Freeze the common external-comparison development data before model training.

Only explicitly selected physical ``train/obs_XXXXXX`` members can be opened.
Selection and input-only query lists precede target access. Existing output
directories are never overwritten, including partially completed preparations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pathrel.formal_data import (ANGLES_DEG, DISTANCES_CELLS, QUOTAS, RADII_CELLS, SOURCES,
                                SPLITS, VIEWS_PER_PARENT, choose_formal_subset,
                                construct_cell_queries, decode_binary_png, sha256,
                                stable_rank, validate_formal_partition)
from pathrel.flatlands_query import mask_relation_counts
from pathrel.parent_pilot_data import canonical_d4_digest
from scripts.evaluate_flatlands_support_clamped import _accelerated_events, _verify_accelerator

DEFAULT_OUT = ROOT / "results/flatlands_external_formal_protocol_v1/data"
CANDIDATE = ROOT / "results/flatlands_parent_groups_v2/development_candidate.csv"
ARCHIVE = ROOT / "data/raw/flatlands/FlatLands_final_dataset.zip"


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_record(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def development_exposure_metadata():
    """Known development use only; this does not claim exhaustive final-test history."""
    histories, sources = {}, {}
    for relative, name in [
        ("results/flatlands_parent_groups_v2/old_train_validation_parents.csv", "old_train_or_checkpoint_validation"),
        ("results/parent_group_pilot_v1/data/selected.csv", "parent_pilot_development"),
    ]:
        path = ROOT / relative
        if path.exists():
            histories[name] = {row["parent_group"] for row in csv.DictReader(path.open())}
            sources[relative] = sha256(path)
    relative = "results/model_gallery_20260909_v1/selection.json"
    path = ROOT / relative
    if path.exists():
        histories["published_development_gallery"] = {row["parent_group"] for row in json.loads(path.read_text())["indoor"]}
        sources[relative] = sha256(path)
    return histories, sources


def prepare(out, candidate=CANDIDATE, archive_path=ARCHIVE):
    out = Path(out)
    # Fail before opening even development image members if a directory exists.
    out.mkdir(parents=True, exist_ok=False)
    (out / "packets").mkdir()
    candidates = list(csv.DictReader(Path(candidate).open()))
    rows, capacity = choose_formal_subset(candidates)
    histories, history_hashes = development_exposure_metadata()
    for row in rows:
        row["source"] = row["source_dataset"]
        row["known_development_exposure"] = "|".join(name for name, parents in histories.items() if row["parent_group"] in parents)
    with (out / "selected.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    source_paths = [Path(__file__), ROOT / "src/pathrel/formal_data.py", ROOT / "src/pathrel/data_access.py",
                    ROOT / "src/pathrel/parent_pilot_data.py", ROOT / "src/pathrel/flatlands_query.py",
                    ROOT / "scripts/evaluate_flatlands_support_clamped.py"]
    protocol = {
        "id": "flatlands_external_formal_data_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Shared validation-only development comparison; neither fresh final test nor published benchmark replication",
        "physical_archive_split": "train", "provenance_split_is_metadata_not_partition": True,
        "candidate_sha256": sha256(candidate), "selected_sha256_before_image_access": sha256(out / "selected.csv"),
        "selection": "Existing candidate split; source capacity-capped parent SHA256 rank; independently hash-ranked views per parent; no outcome-based replacement",
        "parent_quotas": QUOTAS, "views_per_parent": VIEWS_PER_PARENT, "capacity_metadata_audit": capacity,
        "input": {"external_channels": ["observed_free", "unknown", "valid_support"],
                  "conpath_lossless_channels": ["observed_free", "observed_blocked", "unknown"],
                  "support": "supplied epistemic_mask", "hidden": "unobserved AND valid_support",
                  "observed_free": "observed_floor AND valid_support",
                  "observed_blocked": "valid_support AND NOT hidden AND NOT observed_free",
                  "partial_observation_generation": "Use the released observed_floor/unobserved/epistemic_mask PNGs and camera metadata unchanged; no resimulation, target-based masking, augmentation or added input channel",
                  "projection": "All models preserve observed free/blocked evidence and close every cell outside supplied support"},
        "query": {"distances_cells": DISTANCES_CELLS, "angles_deg": ANGLES_DEG, "radii_cells": RADII_CELLS,
                  "units": "grid cells only; no physical metre conversion",
                  "start": "nearest observed-free valid cell to released camera [x,y], squared-distance then row/column tie break",
                  "goals": "polar stencil about start; retain all in-bounds distinct unknown-valid goals; freeze before target read",
                  "target_blocked_goal": "retain as a negative event", "query_order": "SHA256(global_id,candidate_index)",
                  "evaluation": "all input-eligible query-radius events; no target-based removal",
                  "oracle": "exact integer disk clearance, closed image boundary, four-neighbour connectivity"},
        "weighting": {"primary": "equal independent parent place; equal observation within parent; equal eligible query within observation",
                      "pooled": "all query-radius events equally weighted", "secondary": "equal source macro and per-source metrics",
                      "zero_queries": "retain map training/scoring; exclude only from event denominator and report counts"},
        "test_lock": {"new_physical_test_images_allowed": False, "location_6_access_allowed": False,
                      "final_test_access_allowed": False, "only_archive_member_prefix": "train/obs_XXXXXX/",
                      "historical_read_scope_warning": "Old code had physical-test access; no new access is allowed here. Historical development exposure is disclosed, not erased.",
                      "physical_test_parent_presence": "metadata flag only; never open the referenced physical test maps"},
        "known_development_history_sources": history_hashes,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "git_commit_at_data_selection": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "png_decoder": {"implementation": "strict PNG header + Pillow verify + binary 0/255 validation", "pillow": PIL.__version__},
        "failure_policy": "Keep frozen selection and failed audit; do not replace cases after viewing targets or overwrite this directory",
    }
    atomic_json(out / "data_protocol.json", protocol)
    # Fix the primary gallery by metadata alone. Individual query lists are frozen below.
    figures = []
    for source in SOURCES:
        pool = [row for row in rows if row["candidate_split"] == "validation" and row["source_dataset"] == source]
        pool.sort(key=lambda row: (stable_rank("primary_figure", row["global_id"]), row["global_id"]))
        figures.extend({"global_id": row["global_id"], "parent_group": row["parent_group"], "source": source} for row in pool[:2])
    atomic_json(out / "primary_figure_selection_before_images.json", {
        "selection": "two hash-ranked validation observations per source before image or model access", "cases": figures,
        "fixed_query_rule": "first input-hash-ranked eligible query, all three radii; zero-query cases are retained and labelled",
        "validation_only": True,
    })
    accelerator_audit = _verify_accelerator()
    reports, problems, hashes = [], [], defaultdict(list)
    read_count = 0
    query_fields = ["global_id", "parent_group", "source_dataset", "scene_id", "candidate_split", "query_id", "event_id",
                    "candidate_index", "start_row", "start_col", "goal_row", "goal_col", "distance_cells", "angle_deg", "radius_cells"]
    with ZipFile(archive_path) as archive, (out / "image_access.jsonl").open("x") as access_log, \
            (out / "input_queries_before_target.jsonl").open("x") as query_log, \
            (out / "queries.csv").open("x", newline="") as query_csv:
        writer = csv.DictWriter(query_csv, fieldnames=query_fields)
        writer.writeheader()
        for position, row in enumerate(rows):
            validate_formal_partition([row])
            blobs = {}

            def read(name):
                nonlocal read_count
                if name not in {"observed_floor.png", "unobserved.png", "epistemic_mask.png", "floor_map.png", "metadata.json"}:
                    raise ValueError("Unexpected archive member requested")
                member = row["packet_directory"] + "/" + name
                if not member.startswith("train/" + row["global_id"] + "/"):
                    raise ValueError("Archive member escaped physical train packet")
                blob = archive.read(member)
                digest = hashlib.sha256(blob).hexdigest()
                blobs[name] = digest
                append_record(access_log, {"sequence": read_count, "global_id": row["global_id"], "member": member,
                                           "physical_split": "train", "candidate_split": row["candidate_split"],
                                           "bytes": len(blob), "sha256": digest})
                read_count += 1
                return blob

            metadata = json.loads(read("metadata.json"))
            if (metadata["provenance"]["global_id"] != row["global_id"] or metadata["scene"]["scene_id"] != row["scene_id"]
                    or metadata["scene"]["dataset"] != row["source_dataset"]
                    or float(metadata["scene"]["resolution"]) != float(row["resolution"])
                    or metadata["observation"]["camera_px"] != json.loads(row["camera_px"])):
                raise ValueError("Packet metadata identity mismatch")
            observed, unobserved, valid = [decode_binary_png(read(name)) for name in (
                "observed_floor.png", "unobserved.png", "epistemic_mask.png")]
            if any(array.shape != (256, 256) for array in (observed, unobserved, valid)):
                raise ValueError("Expected aligned 256x256 maps")
            queries = construct_cell_queries(observed, unobserved, valid, json.loads(row["camera_px"]))
            eligible = sorted((q for q in queries if q["selection_status"] == "selected"),
                              key=lambda q: stable_rank("query", row["global_id"], q["candidate_index"]))
            append_record(query_log, {"global_id": row["global_id"], "input_access_count": read_count,
                                      "queries": queries, "eligible_candidate_order": [q["candidate_index"] for q in eligible],
                                      "target_member_not_yet_opened": True})
            for query in eligible:
                query_id = f"{row['global_id']}:q{query['candidate_index']:03d}"
                for radius in RADII_CELLS:
                    writer.writerow({**{key: row[key] for key in ("global_id", "parent_group", "source_dataset", "scene_id", "candidate_split")},
                                     **{key: query[key] for key in query if key != "selection_status"},
                                     "query_id": query_id, "event_id": query_id + f":r{radius:02d}", "radius_cells": radius})
            query_csv.flush()
            os.fsync(query_csv.fileno())
            target_raw = decode_binary_png(read("floor_map.png"))
            if target_raw.shape != (256, 256):
                raise ValueError("Expected aligned 256x256 target")
            free, hidden = observed & valid, unobserved & valid
            blocked = valid & ~hidden & ~free
            conflicts = int(np.count_nonzero(free & ~target_raw) + np.count_nonzero(blocked & target_raw)
                            + np.count_nonzero(observed & unobserved))
            if conflicts:
                problems.append({"global_id": row["global_id"], "observed_target_or_input_conflicts": conflicts})
            target = target_raw & valid
            observation = np.stack((free, blocked, hidden)).astype(np.float32)
            starts = np.array([[q["start_row"], q["start_col"]] for q in eligible], dtype=np.int64).reshape(-1, 2)
            goals = np.array([[q["goal_row"], q["goal_col"]] for q in eligible], dtype=np.int64).reshape(-1, 2)
            targets = _accelerated_events(target[None], starts, goals, RADII_CELLS)[0]
            packet = out / "packets" / (row["global_id"] + ".npz")
            temporary = packet.with_name("." + packet.name + ".tmp")
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, observation=observation, valid=valid, hidden=hidden, target=target,
                                    starts=starts, goals=goals, targets=targets,
                                    candidate_indices=np.array([q["candidate_index"] for q in eligible], dtype=np.int64))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, packet)
            for kind, array in (("target_support", np.stack((target, valid))), ("input", observation.astype(bool))):
                hashes[(kind, canonical_d4_digest(array))].append(row)
            reports.append({"global_id": row["global_id"], "parent_group": row["parent_group"], "source": row["source_dataset"],
                            "split": row["candidate_split"], "queries": len(eligible), "positive_events": targets.sum(axis=0).tolist(),
                            "blocked_target_goals_kept": int((~target[goals[:, 0], goals[:, 1]]).sum()),
                            "hidden_cells": int(hidden.sum()), "source_sha256": blobs,
                            "known_development_exposure": row["known_development_exposure"],
                            "mask_counts": mask_relation_counts(observed, target_raw, unobserved, valid)})
            if (position + 1) % 50 == 0 or position + 1 == len(rows):
                print(json.dumps({"cached": position + 1, "total": len(rows), "split": row["candidate_split"]}), flush=True)
    duplicates = [{"kind": key[0], "global_ids": [r["global_id"] for r in group],
                   "partitions": sorted({r["candidate_split"] for r in group})}
                  for key, group in hashes.items() if len({r["candidate_split"] for r in group}) > 1]
    within_parent_repeats = [{"kind": key[0], "global_ids": [r["global_id"] for r in group]}
                            for key, group in hashes.items() if len(group) > 1 and len({r["parent_group"] for r in group}) == 1]
    summaries = {}
    for split in SPLITS:
        subset = [r for r in reports if r["split"] == split]
        summaries[split] = {
            "parents": len({r["parent_group"] for r in subset}), "observations": len(subset),
            "queries": sum(r["queries"] for r in subset), "events": len(RADII_CELLS) * sum(r["queries"] for r in subset),
            "parents_with_queries": len({r["parent_group"] for r in subset if r["queries"] > 0}),
            "observations_without_queries": [r["global_id"] for r in subset if r["queries"] == 0],
            "observations_without_hidden_cells": [r["global_id"] for r in subset if r["hidden_cells"] == 0],
            "positive_events_per_radius": np.sum([r["positive_events"] for r in subset], axis=0).tolist(),
            "blocked_target_goals_kept": sum(r["blocked_target_goals_kept"] for r in subset),
            "source_parents": {source: len({r["parent_group"] for r in subset if r["source"] == source}) for source in SOURCES},
            "known_prior_development_parents": len({r["parent_group"] for r in subset if r["known_development_exposure"]}),
        }
    passed = not problems and not duplicates and all(item["queries"] > 0 for item in summaries.values())
    audit = {"passed": passed, "new_physical_test_images_opened": 0, "new_final_test_images_opened": 0,
             "location_6_opened": False, "physical_train_member_reads": read_count, "physical_train_png_reads": len(rows) * 4,
             "parent_overlap": 0, "d4_identical_input_or_target_support_cross_split": duplicates,
             "within_parent_d4_repeats_retained": within_parent_repeats,
             "quality_problems": problems, "summary": summaries, "observations": reports,
             "heldout_is_development": True, "validation_only": True,
             "validation_targets_read_for_fixed_data_quality_and_query_oracle": True,
             "no_target_based_resampling": True, "no_label_based_query_removal": True,
             "independent_oracle_preflight": accelerator_audit,
             "all_zero_query_and_zero_hidden_observations_retained": True}
    atomic_json(out / "data_audit.json", audit)
    files = {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}
    atomic_json(out / "seal.json", {"files": files, "frozen_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"passed": passed, "summary": summaries, "quality_problems": problems,
                      "cross_split_duplicates": duplicates, "seal_sha256": sha256(out / "seal.json")}), flush=True)
    if not passed:
        raise SystemExit("Formal data gate failed. Frozen cases remain intact; no automatic replacement is allowed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
