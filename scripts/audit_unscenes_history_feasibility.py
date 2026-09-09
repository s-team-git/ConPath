#!/usr/bin/env python3
"""Train-only sensor/pose feasibility for causal history, without fitting a model."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from pathrel.unscenes3d import load_frame
from pathrel.flatlands_query import sha256_path
from scripts.audit_unscenes3d_coordinate import _parse_calibration, _homogeneous
from scripts.evaluate_flatlands_support_clamped import _accelerated_events, _atomic_json

OUT = Path('results/unscenes_history_feasibility_v1')
RAW = Path('data/raw/unscenes3d/raw_package/unscenes3d-mini_raw')
LABEL = Path('data/raw/unscenes3d/label_package/unscenes3d-mini_label')


def reduce_points(points):
    p = points[np.isfinite(points).all(1)]
    p = p[np.linalg.norm(p[:, :2], axis=1) <= 40]
    _, ids = np.unique(np.floor(p/.3).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(ids)][::2]


def pair_overlap(previous, current, matrix):
    transformed = previous @ matrix[:3, :3].T + matrix[:3, 3]
    distances = cKDTree(current).query(transformed, workers=2)[0]
    return {'median_distance_m': float(np.median(distances)),
            'fraction_within_0_3m': float(np.mean(distances <= .3)),
            'fraction_within_1m': float(np.mean(distances <= 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    path = Path('results/unscenes3d_contract_manifest_ground_valid/manifest.json')
    assert sha256_path(path) == 'd866be0b25ba6635d5ad934bac9c822c55be8034fabd8f4627803bb9efa972d5'
    manifest = json.loads(path.read_text())
    groups = defaultdict(list)
    for r in manifest['records']['train']:
        assert r['location'] in {'location_1', 'location_2', 'location_3'}
        groups[r['scene_id']].append(r)
    # Literal earliest four released frames from every training scene; no
    # filtering by target, overlap, confidence, or downstream event outcome.
    selected = {scene: sorted(rows, key=lambda r: float(r['timestamp']))[:4] for scene, rows in groups.items()}
    source_hashes = {str(path): sha256_path(path), str(Path(__file__)): sha256_path(Path(__file__))}
    frames, pairs, pose_checks = [], [], []
    for scene, rows in sorted(selected.items()):
        previous = None
        for record in rows:
            timestamp = record['timestamp']
            cp, cloud_path = RAW/'calibs'/f'{timestamp}.txt', RAW/'clouds'/f'{timestamp}.bin'
            occ_path = LABEL/'occ'/f'{timestamp}.npy'
            for f in [cp, cloud_path, occ_path]: source_hashes[str(f)] = sha256_path(f)
            transform = _homogeneous(_parse_calibration(cp)['Tr_velo_to_imu'])
            points = reduce_points(np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)[:, :3])
            frame = load_frame(timestamp, raw_root=RAW, label_root=LABEL, **manifest['adapter'])
            known_free, known_blocked = frame.input_bev[0].astype(bool) & frame.target_valid, frame.input_bev[1].astype(bool) & frame.target_valid
            free_conflict = known_free & ~frame.target_free
            blocked_conflict = known_blocked & frame.target_free
            observed = known_free | known_blocked
            starts = np.array([q['start'] for q in record['queries']]); goals = np.array([q['goal'] for q in record['queries']])
            if len(starts):
                lo, hi, truth = _accelerated_events(np.stack([known_free, ~known_blocked & frame.target_valid, frame.target_free]), starts, goals, (0, 1, 2))
                assert np.array_equal(truth, np.array([q['reachable'] for q in record['queries']]))
                fn, fp = truth & ~hi, ~truth & lo
                endpoint_blocked = known_blocked[starts[:, 0], starts[:, 1]] | known_blocked[goals[:, 0], goals[:, 1]]
                endpoint_fn = fn & endpoint_blocked[:, None]
            else: fn = fp = endpoint_fn = np.zeros((0, 3), dtype=bool)
            frames.append({'scene': scene, 'timestamp': timestamp, 'location': record['location'],
                           'observed_valid_cells': int(observed.sum()), 'observed_fraction': float(observed.sum()/max(1, frame.target_valid.sum())),
                           'free_observation_target_blocked_cells': int(free_conflict.sum()),
                           'blocked_observation_target_free_cells': int(blocked_conflict.sum()),
                           'observed_conflict_fraction': float((free_conflict.sum()+blocked_conflict.sum())/max(1, observed.sum())),
                           'event_count': int(fn.size), 'forced_fn': int(fn.sum()), 'forced_fp': int(fp.sum()),
                           'forced_fn_with_directly_blocked_endpoint': int(endpoint_fn.sum()),
                           'calibration_rotation_orthogonality_error': float(np.max(np.abs(transform[:3,:3].T@transform[:3,:3]-np.eye(3))))})
            ego = Path('data/raw/unscenes3d/raw_data')/scene/'ego_pose'/f'{timestamp}.txt'
            if ego.exists():
                source_hashes[str(ego)] = sha256_path(ego)
                lines = [np.fromstring(line, sep=' ') for line in ego.read_text().splitlines()]
                candidates = []
                for i, line in enumerate(lines):
                    if len(line) != 8: continue
                    rot = Rotation.from_quat(line[4:8]).as_matrix()
                    candidates.append({'line_1_based': i+1, 'translation_difference_m': float(np.linalg.norm(line[1:4]-transform[:3,3])),
                                       'rotation_max_abs_difference': float(np.max(np.abs(rot-transform[:3,:3])))})
                pose_checks.append({'scene':scene, 'timestamp':timestamp, 'ego_pose':str(ego), 'candidates':candidates})
            if previous is not None:
                ts, prev_points, prev_transform = previous
                assert float(ts) < float(timestamp)
                relative = np.linalg.inv(transform) @ prev_transform
                pairs.append({'scene':scene, 'history_timestamp':ts, 'current_timestamp':timestamp,
                              'gap_seconds':float(timestamp)-float(ts), 'relative_translation_m':float(np.linalg.norm(relative[:3,3])),
                              'unaligned':pair_overlap(prev_points,points,np.eye(4)),
                              'pose_candidate':pair_overlap(prev_points,points,relative),
                              'inverse_direction_control':pair_overlap(prev_points,points,np.linalg.inv(relative))})
            previous = timestamp, points, transform
        print(json.dumps({'scene':scene,'frames':len(rows)},ensure_ascii=False),flush=True)
    def equal_scene_mean(key):
        return float(np.mean([np.mean([r[key] for r in frames if r['scene']==s]) for s in groups]))
    summary = {'training_scenes_available':len(groups), 'training_frames_available':sum(map(len,groups.values())),
               'audited_frames':len(frames), 'audited_history_pairs':len(pairs), 'raw_ego_pose_matches':len(pose_checks),
               'scene_weighted_observation_conflict_fraction':equal_scene_mean('observed_conflict_fraction'),
               'scene_weighted_observed_fraction':equal_scene_mean('observed_fraction'),
               'forced_fn':sum(r['forced_fn'] for r in frames), 'forced_fp':sum(r['forced_fp'] for r in frames),
               'forced_fn_with_directly_blocked_endpoint':sum(r['forced_fn_with_directly_blocked_endpoint'] for r in frames),
               'pose_improves_pair_median_count':sum(p['pose_candidate']['median_distance_m']<p['unaligned']['median_distance_m'] for p in pairs),
               'mean_pair_overlap_0_3m':{k:float(np.mean([p[k]['fraction_within_0_3m'] for p in pairs])) for k in ['unaligned','pose_candidate','inverse_direction_control']}}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'train_only':True,'validation_labels_opened':False,'test_assets_opened':False,
            'model_trained':False,'temporal_prediction_accuracy_established':False,'summary':summary,'frames':frames,'pairs':pairs,'pose_checks':pose_checks,
            'source_hashes':source_hashes,'selection':'First four timestamp-sorted frames in each frozen training scene, selected before reading any labels.',
            'interpretation':'Relative transforms are a tested candidate derived from calibration, with available separate ego-pose checks; static raw geometry overlap does not prove causal traversability prediction quality. No dense accumulated local map or future frame is an input. Every pair uses history strictly before current.'}
    _atomic_json(OUT/'report.json',report);print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':main()
