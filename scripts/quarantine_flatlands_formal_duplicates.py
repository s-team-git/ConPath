#!/usr/bin/env python3
"""Apply a pretraining, input-only duplicate-parent rule without new PNG access.

The initial frozen data directory is retained byte-for-byte. Every parent with
an observation in a cross-partition identical-input component is quarantined
from all partitions, without replacement. This is a conservative distribution
restriction, not a claim that identical empty-space inputs prove place leakage.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pathrel.formal_data import SOURCES, SPLITS, sha256, validate_formal_partition
from pathrel.parent_pilot_data import canonical_d4_digest
from scripts.prepare_flatlands_formal_data import atomic_json

INITIAL = ROOT / "results/flatlands_external_formal_protocol_v1/data"
OUT = INITIAL.parent / "data_eligible_v1"


def summarize(rows, observation_reports):
    result = {}
    for split in SPLITS:
        selected = [row for row in rows if row["candidate_split"] == split]
        ids = {row["global_id"] for row in selected}
        subset = [row for row in observation_reports if row["global_id"] in ids]
        result[split] = {
            "parents": len({r["parent_group"] for r in subset}), "observations": len(subset),
            "queries": sum(r["queries"] for r in subset), "events": 3 * sum(r["queries"] for r in subset),
            "parents_with_queries": len({r["parent_group"] for r in subset if r["queries"] > 0}),
            "observations_without_queries": [r["global_id"] for r in subset if r["queries"] == 0],
            "observations_without_hidden_cells": [r["global_id"] for r in subset if r["hidden_cells"] == 0],
            "positive_events_per_radius": np.sum([r["positive_events"] for r in subset], axis=0).tolist(),
            "blocked_target_goals_kept": sum(r["blocked_target_goals_kept"] for r in subset),
            "source_parents": {source: len({r["parent_group"] for r in subset if r["source"] == source}) for source in SOURCES},
            "known_prior_development_parents": len({r["parent_group"] for r in subset if r["known_development_exposure"]}),
        }
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "packets").mkdir()
    initial_seal = json.loads((INITIAL / "seal.json").read_text())
    for name, digest in initial_seal["files"].items():
        if sha256(INITIAL / name) != digest:
            raise ValueError("Initial frozen data changed: " + name)
    initial_audit = json.loads((INITIAL / "data_audit.json").read_text())
    initial_protocol = json.loads((INITIAL / "data_protocol.json").read_text())
    if initial_audit["new_physical_test_images_opened"] or initial_audit["quality_problems"] or initial_audit["parent_overlap"]:
        raise ValueError("This rule cannot repair a test read, evidence conflict or parent-partition failure")
    rows = list(csv.DictReader((INITIAL / "selected.csv").open()))
    by_id = {row["global_id"]: row for row in rows}
    duplicate_input_groups = [g for g in initial_audit["d4_identical_input_or_target_support_cross_split"] if g["kind"] == "input"]
    duplicate_input_ids = {gid for group in duplicate_input_groups for gid in group["global_ids"]}
    duplicate_target_ids = {gid for group in initial_audit["d4_identical_input_or_target_support_cross_split"]
                            if group["kind"] == "target_support" for gid in group["global_ids"]}
    parents = {by_id[gid]["parent_group"] for gid in duplicate_input_ids}
    if not parents or not duplicate_target_ids.issubset(duplicate_input_ids):
        raise ValueError("Expected input duplicate parent rule to cover the flagged target templates")
    removed = [row for row in rows if row["parent_group"] in parents]
    selected = [row for row in rows if row["parent_group"] not in parents]
    selected_ids = {row["global_id"] for row in selected}
    validate_formal_partition(selected)
    observation_reports = [r for r in initial_audit["observations"] if r["global_id"] in selected_ids]
    after = summarize(selected, observation_reports)
    decision = {
        "id": "input_duplicate_parent_quarantine_v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "before any formal model training or prediction scoring",
        "rule": "Find exact D4-equivalent observation inputs shared across candidate partitions; quarantine every affected parent and all its selected views from every partition, without replacement",
        "selection_uses_targets": False, "selection_uses_model_outputs": False,
        "initial_data_seal_sha256": sha256(INITIAL / "seal.json"),
        "initial_data_directory": str(INITIAL.relative_to(ROOT)),
        "input_duplicate_groups": duplicate_input_groups,
        "duplicate_input_observations": len(duplicate_input_ids), "duplicate_target_support_observations": len(duplicate_target_ids),
        "target_duplicates_are_subset_of_input_duplicates": True,
        "quarantined_parents": sorted(parents), "quarantined_observations": removed,
        "before": initial_audit["summary"], "after": after,
        "interpretation": "These inputs have full support, no observed barriers and the same released field of view. Fifteen targets are uniformly free; four identical inputs have differing hidden targets. This is a generic open-space/ambiguous observation template, not proof that the source parent places are identical.",
        "distribution_limit": "Conservative duplicate isolation removes some open-space and observationally ambiguous cases, primarily ZInD. Results apply to this restricted development distribution. No claim of universal indoor superiority follows.",
        "future_use": "Quarantined maps remain in the initial frozen directory for a separately declared diagnostic only; they are excluded from this training/calibration/validation matrix",
        "new_raw_png_reads": 0, "new_test_or_final_map_reads": 0,
    }
    atomic_json(OUT / "quarantine_decision.json", decision)
    with (OUT / "selected.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    # Copy the already input-frozen query records; counters refer to the original read ledger.
    with (OUT / "input_queries_before_target.jsonl").open("x") as handle:
        for line in (INITIAL / "input_queries_before_target.jsonl").read_text().splitlines():
            if json.loads(line)["global_id"] in selected_ids:
                handle.write(line + "\n")
    with (INITIAL / "queries.csv").open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        query_rows = [row for row in reader if row["global_id"] in selected_ids]
    with (OUT / "queries.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(query_rows)
    primary_figures = json.loads((INITIAL / "primary_figure_selection_before_images.json").read_text())
    primary_figures["quarantined_original_cases"] = [case for case in primary_figures["cases"] if case["global_id"] not in selected_ids]
    primary_figures["cases"] = [case for case in primary_figures["cases"] if case["global_id"] in selected_ids]
    primary_figures["original_selection_sha256"] = sha256(INITIAL / "primary_figure_selection_before_images.json")
    primary_figures["replacement_after_quarantine"] = False
    atomic_json(OUT / "primary_figure_selection_before_images.json", primary_figures)
    protocol = {**initial_protocol, "id": "flatlands_external_formal_data_eligible_v1",
                "initial_selection_protocol_sha256": sha256(INITIAL / "data_protocol.json"),
                "initial_parent_quotas": initial_protocol["parent_quotas"],
                "parent_quotas": {split: item["source_parents"] for split, item in after.items()},
                "selected_sha256_before_image_access": None,
                "selected_sha256": sha256(OUT / "selected.csv"),
                "selection_timing": "Initial candidate/parent/view list frozen before image access; pretraining input-only parent quarantine applied to cached packets without replacement or new PNG reads",
                "quarantine_decision_sha256": sha256(OUT / "quarantine_decision.json"),
                "source_image_access_ledger": {"path": str((INITIAL / "image_access.jsonl").relative_to(ROOT)),
                                                "sha256": sha256(INITIAL / "image_access.jsonl")},
                "input_query_counter_interpretation": "input_access_count records refer to the initial source_image_access_ledger; the eligible copy performs no archive reads"}
    protocol["source_hashes"] = {**initial_protocol["source_hashes"], str(Path(__file__).relative_to(ROOT)): sha256(__file__)}
    atomic_json(OUT / "data_protocol.json", protocol)
    hashes = defaultdict(list)
    for row in selected:
        relative = "packets/" + row["global_id"] + ".npz"
        shutil.copy2(INITIAL / relative, OUT / relative)
        if sha256(OUT / relative) != initial_seal["files"][relative]:
            raise ValueError("Copy does not match its source packet")
        with np.load(OUT / relative, allow_pickle=False) as packet:
            for kind, array in (("target_support", np.stack((packet["target"], packet["valid"]))),
                                ("input", packet["observation"].astype(bool))):
                hashes[(kind, canonical_d4_digest(array))].append(row)
    duplicates = [{"kind": key[0], "global_ids": [row["global_id"] for row in group]}
                  for key, group in hashes.items() if len({r["candidate_split"] for r in group}) > 1]
    repeated_within_parent = [{"kind": key[0], "global_ids": [row["global_id"] for row in group]}
                             for key, group in hashes.items() if len(group) > 1 and len({r["parent_group"] for r in group}) == 1]
    passed = not duplicates and all(item["queries"] > 0 for item in after.values())
    audit = {**initial_audit, "passed": passed, "summary": after, "observations": observation_reports,
             "physical_train_member_reads": 0, "physical_train_png_reads": 0,
             "reused_cached_packets": len(selected), "initial_raw_png_reads": initial_audit["physical_train_png_reads"],
             "initial_frozen_data_retained": True, "quarantine_decision_sha256": sha256(OUT / "quarantine_decision.json"),
             "d4_identical_input_or_target_support_cross_split": duplicates,
             "within_parent_d4_repeats_retained": repeated_within_parent,
             "query_rows": len(query_rows), "new_raw_png_reads": 0}
    atomic_json(OUT / "data_audit.json", audit)
    atomic_json(OUT / "seal.json", {"files": {str(path.relative_to(OUT)): sha256(path)
                                               for path in sorted(OUT.rglob("*")) if path.is_file()},
                                      "frozen_utc": datetime.now(timezone.utc).isoformat()})
    for name, digest in initial_seal["files"].items():
        if sha256(INITIAL / name) != digest:
            raise ValueError("Initial directory changed during quarantine")
    print(json.dumps({"passed": passed, "data_root": str(OUT), "summary": after,
                      "seal_sha256": sha256(OUT / "seal.json"), "quarantined_parents": len(parents),
                      "quarantined_observations": len(removed), "new_raw_png_reads": 0}), flush=True)
    if not passed:
        raise SystemExit("Eligible data audit failed; no output directory may be overwritten")


if __name__ == "__main__":
    main()
