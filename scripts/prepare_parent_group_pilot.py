#!/usr/bin/env python3
"""Freeze a small from-scratch pilot on physically train-only independent places."""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.data_access import require_development_packet
from pathrel.flatlands_query import construct_natural_queries, decode_binary_grayscale_png, mask_relation_counts
from pathrel.parent_pilot_data import canonical_d4_digest, choose_parent_subset, sha, stable_rank
from scripts.evaluate_flatlands_support_clamped import _accelerated_events, _verify_accelerator

OUT = ROOT / 'results/parent_group_pilot_v1/data'
CANDIDATE = ROOT / 'results/flatlands_parent_groups_v2/development_candidate.csv'
ARCHIVE = ROOT / 'data/raw/flatlands/FlatLands_final_dataset.zip'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'packets').mkdir()
    rows = choose_parent_subset(list(csv.DictReader(CANDIDATE.open())))
    with (OUT / 'selected.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    protocol = {
        'id': 'parent_group_pilot_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Small development pilot, not a final test or a replication of published leaderboard values',
        'physical_archive_split': 'train', 'provenance_split_is_metadata_not_partition': True,
        'partition': {'train': 100, 'calibration': 25, 'validation': 40, 'observations_per_parent': 1, 'source_balanced': True},
        'selection': 'Source and candidate-partition stratified parent hash; one hash-ranked view/parent; no outcome-based replacements',
        'candidate_sha256': sha(CANDIDATE), 'selected_sha256_before_image_access': sha(OUT / 'selected.csv'),
        'query': {'distances_cells': [40, 80, 120], 'angles_deg': list(range(0, 360, 30)), 'radii_cells': [0, 10, 20],
                  'construction': 'observed free start; unknown and supplied-support valid goals; frozen before target PNG',
                  'blocked_target_goal': 'keep as negative, never drop based on target endpoint class',
                  'training_queries_per_observation': 8, 'training_query_order': 'hash of global_id and candidate index',
                  'evaluation_queries': 'all input-eligible queries', 'connectivity': 'four-neighbor with exact integer disk clearance'},
        'scale': 'Raster-cell geometry only; release resolution=0.01 is used to reproduce the cell stencil, not verified physical-metre claims',
        'input': '3 observed free/blocked/unknown channels and supplied epistemic valid support; target map used only for loss/scoring',
        'methods': ['correlated', 'independent', 'deterministic', 'all_floor', 'all_blocked', 'nearest_observed', 'train_radius_prior'],
        'seeds': [20260910, 20260911, 20260912],
        'training': {'from_scratch': True, 'feature_channels': 16, 'latent_dim': 4, 'batch_size': 4, 'train_samples': 4,
                     'max_epochs': 24, 'minimum_epochs': 12, 'patience': 6, 'learning_rate': 0.0003, 'weight_decay': 0.0001,
                     'gradient_clip': 5., 'map_weight': 1., 'variogram_weight': 0.1, 'event_weight': 2.,
                     'max_reachability_steps': 256, 'deterministic_loss': 'unknown-valid binary map NLL',
                     'selection': 'lowest equal-parent calibration event Brier, tie within 1e-5 broken by map NLL',
                     'calibration_K': 16, 'calibration_rng': 'fixed per-case stream reset each epoch, independent of training stream',
                     'validation_access': 'only after all nine baseline runs have selected checkpoints',
                     'precision': 'float32, no AMP; shared GPU timing is not isolated latency'},
        'evaluation': {'K_primary': 32, 'K_common_small': 4, 'deterministic_K': 1, 'no_oracle_selection': True,
                       'aggregation': 'equal parent; one view per parent and equal source counts',
                       'metrics': ['event Brier', 'risk at 30% coverage with fractional probability ties', 'unknown-valid map IoU', 'MES for K>1'],
                       'confidence': 'source-stratified paired bootstrap over parent places; seed means shown separately'},
        'decision': {'implementation_pass': 'checksums, splits, source identities, exact query oracle, finite learning and all 9 complete runs',
                     'research_screen': 'mean K32 Brier lower than independent and deterministic, lower than all-floor; risk30 not worse than deterministic',
                     'after_screen_pass': 'freeze one bounded model improvement using training/calibration diagnostics, compare with original under the same budget; validation becomes iterated development',
                     'after_screen_fail': 'retain negative result; analyze protocol/input/underfitting before extending architecture; no automatic long Transformer run',
                     'strong_claim': 'pilot screen is not statistical proof, final-test result, or external-method superiority'},
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/pathrel/parent_pilot_data.py', ROOT / 'src/pathrel/data_access.py']}}
    write(OUT / 'protocol.json', protocol)
    _verify_accelerator()
    reports, query_log, problems = [], [], []
    hashes = defaultdict(list)
    with ZipFile(ARCHIVE) as archive:
        for row in rows:
            require_development_packet(row)
            assert row['archive_split'] == 'train'
            blobs = {}
            def read(name):
                blob = archive.read(row['packet_directory'] + '/' + name)
                blobs[name] = hashlib.sha256(blob).hexdigest()
                return decode_binary_grayscale_png(blob)
            observed, unobserved, valid = [read(n) for n in ('observed_floor.png', 'unobserved.png', 'epistemic_mask.png')]
            queries = construct_natural_queries(observed, unobserved, valid, camera_px=json.loads(row['camera_px']), resolution_m=float(row['resolution']))
            eligible = sorted((q for q in queries if q.selection_status == 'selected'), key=lambda q: stable_rank(row['global_id'], q.candidate_index))
            # This input-only list is serialized before reading the target.
            with (OUT / 'input_queries_before_target.jsonl').open('a') as f:
                f.write(json.dumps({'global_id': row['global_id'], 'queries': [asdict(q) for q in queries]}) + '\n'); f.flush()
            target = read('floor_map.png')
            metadata_blob = archive.read(row['metadata_member']); blobs['metadata.json'] = hashlib.sha256(metadata_blob).hexdigest()
            meta = json.loads(metadata_blob)
            if (meta['provenance']['global_id'] != row['global_id'] or meta['scene']['scene_id'] != row['scene_id']
                    or meta['scene']['dataset'] != row['source_dataset'] or float(meta['scene']['resolution']) != float(row['resolution'])
                    or meta['observation']['camera_px'] != json.loads(row['camera_px'])):
                raise ValueError('Packet metadata identity mismatch')
            if any(a.shape != (256, 256) for a in (observed, unobserved, valid, target)):
                raise ValueError('Wrong aligned map shape')
            free, hidden = observed & valid, unobserved & valid
            blocked = valid & ~hidden & ~free
            conflicts = int(np.count_nonzero(free & ~target) + np.count_nonzero(blocked & target) + np.count_nonzero(observed & unobserved))
            if conflicts or not hidden.any():
                problems.append({'global_id': row['global_id'], 'conflicts': conflicts, 'hidden_cells': int(hidden.sum())})
            target = target & valid
            observation = np.stack((free, blocked, hidden)).astype(np.float32)
            starts = np.array([[q.start_row, q.start_col] for q in eligible], dtype=np.int64).reshape(-1, 2)
            goals = np.array([[q.goal_row, q.goal_col] for q in eligible], dtype=np.int64).reshape(-1, 2)
            targets = _accelerated_events(target[None], starts, goals, (0, 10, 20))[0]
            np.savez_compressed(OUT / 'packets' / (row['global_id'] + '.npz'), observation=observation, valid=valid,
                                hidden=hidden, target=target, starts=starts, goals=goals, targets=targets,
                                candidate_indices=np.array([q.candidate_index for q in eligible], dtype=np.int64))
            for kind, array in [('target_support', np.stack((target, valid))), ('input', observation.astype(bool))]:
                hashes[(kind, canonical_d4_digest(array))].append(row)
            blocked_goals = int((~target[goals[:, 0], goals[:, 1]]).sum())
            reports.append({'global_id': row['global_id'], 'parent_group': row['parent_group'], 'source': row['source_dataset'],
                            'split': row['candidate_split'], 'queries': len(eligible), 'positive_events': targets.sum(axis=0).tolist(),
                            'blocked_target_goals_kept': blocked_goals, 'hidden_cells': int(hidden.sum()), 'source_sha256': blobs,
                            'mask_counts': mask_relation_counts(observed, target, unobserved, valid)})
            print(json.dumps({'cached': row['global_id'], 'split': row['candidate_split'], 'queries': len(eligible)}), flush=True)
    duplicates = [{'kind': key[0], 'global_ids': [r['global_id'] for r in group]} for key, group in hashes.items()
                  if len({r['candidate_split'] for r in group}) > 1]
    summaries = {}
    for split in ('train', 'calibration', 'validation'):
        sub = [r for r in reports if r['split'] == split]
        summaries[split] = {'parents': len(sub), 'queries': sum(r['queries'] for r in sub),
                            'parents_with_queries': sum(r['queries'] > 0 for r in sub),
                            'positive_events_per_radius': np.sum([r['positive_events'] for r in sub], axis=0).tolist(),
                            'blocked_target_goals_kept': sum(r['blocked_target_goals_kept'] for r in sub)}
    passed = not problems and not duplicates and all(s['queries'] > 0 for s in summaries.values())
    audit = {'passed': passed, 'new_physical_test_images_opened': 0, 'parent_overlap': 0,
             'd4_identical_input_or_target_support_cross_split': duplicates, 'quality_problems': problems,
             'summary': summaries, 'observations': reports, 'heldout_is_development': True,
             'validation_labels_opened_for_frozen_data_quality_audit_only': True,
             'no_label_based_query_removal': True, 'no_target_based_resampling': True}
    write(OUT / 'data_audit.json', audit)
    seal = {'files': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}}
    write(OUT / 'seal.json', seal)
    print(json.dumps({'passed': passed, 'summary': summaries, 'quality_problems': problems, 'duplicates': duplicates}), flush=True)
    if not passed:
        raise SystemExit('Pilot data gate failed; selection remains frozen, no automatic replacement.')


if __name__ == '__main__':
    main()
