#!/usr/bin/env python3
"""Bound event accuracy attainable by worlds clamped to the frozen LiDAR observation.

The pessimistic world blocks every unknown valid cell; the optimistic world frees
every unknown valid cell. Every sample honoring the observation lies between these
worlds. Monotonicity of disk erosion and connectivity therefore bounds every event.
Targets outside that interval incur squared error one for every such posterior.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pathrel.flatlands_query import sha256_path
from pathrel.unscenes3d import load_frame
from pathrel.posterior_audits import MEAN_MAP_PROJECTION_VERSION
from scripts.evaluate_flatlands_support_clamped import _accelerated_events, _verify_accelerator, _atomic_json

RADII = (0, 1, 2)
SITES = {"train": {"location_1", "location_2", "location_3"}, "validation": {"location_4_5"}}


def audit_prediction_bounds(rows):
    targets = {(r['scene_id'], r['timestamp'], r['candidate_index'], r['radius_cells']): r for r in rows}
    inputs = []
    for variant in ('', 'independent_'):
        root = ROOT / f'results/unscenes3d_ground_valid_{variant}support_clamped_mean_map_k128_v2'
        for seed in (20260831, 20260901, 20260902):
            path = root / f'seed{seed}/predictions_validation.csv'
            run = json.loads((path.parent / 'run.json').read_text())
            if run['forward'].get('mean_map_projection_version') != MEAN_MAP_PROJECTION_VERSION:
                raise ValueError(f'missing final support projection: {path}')
            if run['prediction']['sha256'] != sha256_path(path):
                raise ValueError(f'prediction hash mismatch: {path}')
            seen = set(); predictions = {}
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    key = (row['scene_id'], row['timestamp'], int(row['candidate_index']), int(row['radius_cells']))
                    if key in seen or key not in targets:
                        raise ValueError(f'duplicate or unexpected event: {path}/{key}')
                    seen.add(key)
                    event = targets[key]; probability = float(row['probability'])
                    predictions[key] = probability
                    if not event['lower'] <= probability <= event['upper']:
                        raise ValueError(f'prediction violates observation bound: {path}/{key}')
                    if (probability - float(event['target'])) ** 2 < int(event['forced_error']):
                        raise ValueError(f'prediction violates squared-error bound: {path}/{key}')
            if seen != set(targets):
                raise ValueError(f'missing validation events: {path}')
            previous = Path(str(path).replace('mean_map_k128_v2', 'mean_map_k128_v1'))
            previous_run = json.loads((previous.parent / 'run.json').read_text())
            if run['checkpoint'] != previous_run['checkpoint'] or run['mean_map'] != previous_run['mean_map']:
                raise ValueError(f'model or hidden-map replay drift beyond the projection correction: {path}')
            old_violations = 0; changed = 0
            with previous.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    key = (row['scene_id'], row['timestamp'], int(row['candidate_index']), int(row['radius_cells']))
                    probability = float(row['probability']); event = targets[key]
                    old_violations += int(not event['lower'] <= probability <= event['upper'])
                    changed += int(probability != predictions[key])
            inputs.append({'path':str(path.relative_to(ROOT)), 'sha256':sha256_path(path),
                           'events':len(seen), 'bound_violations':0,
                           'checkpoint_and_hidden_map_replay_exact':True,
                           'superseded_v1':{'path':str(previous.relative_to(ROOT)), 'sha256':sha256_path(previous),
                                            'bound_violations':old_violations, 'changed_predictions':changed}})
    return {'passed':True, 'event_comparisons':sum(r['events'] for r in inputs), 'inputs':inputs}


def summarize(rows):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_id"]].append(row)
    def mean(key):
        return float(np.mean([np.mean([r[key] for r in values]) for values in scenes.values()]))
    return {
        "events": len(rows), "scene_count": len(scenes),
        "scene_weighted_positive_rate": mean("target"),
        "scene_weighted_unavoidable_false_negative": mean("forced_false_negative"),
        "scene_weighted_unavoidable_false_positive": mean("forced_false_positive"),
        "scene_weighted_brier_lower_bound": mean("forced_error"),
        "query_weighted_brier_lower_bound": float(np.mean([r["forced_error"] for r in rows])),
        "scene_weighted_mutable_event_fraction": mean("mutable"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/unscenes3d_observation_ceiling_v1/report.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("test_locked_sites") != ["location_6"]:
        raise ValueError("unexpected test lock")
    if sha256_path(args.manifest) != "d866be0b25ba6635d5ad934bac9c822c55be8034fabd8f4627803bb9efa972d5":
        raise ValueError("canonical ground-valid manifest hash changed")
    audit = _verify_accelerator()
    started = time.monotonic(); splits = {}; frame_audits = []
    for split in ("train", "validation"):
        rows = []
        records = manifest["records"][split]
        for index, record in enumerate(records):
            if record["location"] not in SITES[split]:
                raise ValueError("frame is outside the explicit train/validation allow-list")
            frame = load_frame(record["timestamp"],
                               raw_root=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"),
                               label_root=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"),
                               **manifest["adapter"])
            starts = np.array([q["start"] for q in record["queries"]], dtype=np.int64)
            goals = np.array([q["goal"] for q in record["queries"]], dtype=np.int64)
            if not np.array_equal(starts, frame.starts) or not np.array_equal(goals, frame.goals):
                raise ValueError("runtime query geometry does not match the frozen manifest")
            blocked = frame.input_bev[1] > 0.5
            known_free = (frame.input_bev[0] > 0.5) & ~blocked & frame.target_valid
            optimistic = ~blocked & frame.target_valid
            pessimistic = known_free
            lower, upper, truth = _accelerated_events(np.stack([pessimistic, optimistic, frame.target_free]), starts, goals, RADII)
            targets = np.array([q["reachable"] for q in record["queries"]], dtype=bool)
            if not np.array_equal(truth, targets) or np.any(lower & ~upper):
                raise ValueError("exact target or monotonic bound contract failed")
            for i, query in enumerate(record["queries"]):
                for j, radius in enumerate(RADII):
                    fn = bool(targets[i,j] and not upper[i,j])
                    fp = bool(not targets[i,j] and lower[i,j])
                    rows.append({"scene_id":record["scene_id"], "timestamp":record["timestamp"],
                                 "candidate_index":query["candidate_index"], "radius_cells":radius,
                                 "target":bool(targets[i,j]), "lower":bool(lower[i,j]), "upper":bool(upper[i,j]),
                                 "forced_false_negative":fn, "forced_false_positive":fp,
                                 "forced_error":fn or fp, "mutable":bool(upper[i,j] and not lower[i,j])})
            observed = ((frame.input_bev[0] > 0.5) | blocked) & frame.target_valid
            contradictions = (known_free & ~frame.target_free) | (blocked & frame.target_valid & frame.target_free)
            frame_audits.append({"split":split, "scene_id":record["scene_id"], "timestamp":record["timestamp"],
                                 "observed_valid_cells":int(observed.sum()), "observation_target_conflicts":int(contradictions.sum())})
            if (index+1) % 40 == 0 or index+1 == len(records):
                print(json.dumps({"split":split,"frames":index+1,"total":len(records),"seconds":round(time.monotonic()-started,2)}),flush=True)
        splits[split] = {"overall":summarize(rows),
                         "by_radius":{str(r):summarize([x for x in rows if x['radius_cells']==r]) for r in RADII},
                         "by_scene":{s:summarize([x for x in rows if x['scene_id']==s]) for s in sorted({x['scene_id'] for x in rows})},
                         "events":rows}
    prediction_audit = audit_prediction_bounds(splits['validation']['events'])
    report = {"kind":"unscenes3d_observation_event_bounds", "schema_version":1,
              "test_evaluated":False,"paper_result":False,"validation_only":True,
              "manifest":{"path":str(args.manifest),"sha256":sha256_path(args.manifest)},
              "software":{"script_sha256":sha256_path(Path(__file__)),"adapter_sha256":sha256_path(ROOT/'src/pathrel/unscenes3d.py'),
                          "model_sha256":sha256_path(ROOT/'src/pathrel/model.py'),
                          "oracle_sha256":sha256_path(ROOT/'scripts/evaluate_flatlands_support_clamped.py')},
              "prediction_bound_audit":prediction_audit,
              "accelerator_audit":audit,"radii_cells":RADII,"splits":splits,"frame_conflicts":frame_audits,
              "runtime_seconds":time.monotonic()-started,
              "interpretation":"For any posterior world respecting the hard observation and target-valid support, event probability lies between pessimistic and optimistic connectivity. A positive target with upper=0, or negative target with lower=1, forces Brier error one. This is an adapter-conditioned lower bound, not a universal impossibility result.",
              "claim_boundary":"Frozen train/validation diagnostic only. Target labels are used after observation construction to audit conflicts; no sensor policy is fitted, no manifest changed and location_6 remains unopened."}
    _atomic_json(args.output,report)
    compact={**report,"splits":{s:{k:v for k,v in value.items() if k!='events'} for s,value in splits.items()}}
    compact.pop('frame_conflicts')
    _atomic_json(ROOT/'site/data/unscenes3d_observation_ceiling.json',compact)
    print(json.dumps({s:v['overall'] for s,v in splits.items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
