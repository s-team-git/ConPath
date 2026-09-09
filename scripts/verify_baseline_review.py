#!/usr/bin/env python3
"""Replay saved map scores independently and audit evidence/parent partitions."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.flatlands_data import FlatLandsReplayDataset
from scripts.evaluate_flatlands_clean_controls import SELECTION, QUERIES


def main():
    checked = 0
    def hashes(mapping):
        nonlocal checked
        for name, digest in mapping.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
            checked += 1
    raw = json.loads(Path('results/current_baseline_k4_v1/report.json').read_text())
    hashes(raw['source_hashes'])
    # Added after the original numerical audit: provenance validation includes
    # physical-test packets. Preserve the existing metrics receipt, but do not
    # reopen those images under the reinstated physical-test lock.
    with SELECTION.open() as handle:
        if any(r['provenance_split'] == 'validation' and r['archive_split'] == 'test'
               for r in csv.DictReader(handle)):
            raise RuntimeError('Old validation contains physical-test images. Read flatlands_read_scope_erratum.json; image replay is locked.')
    data = FlatLandsReplayDataset(Path('data/raw/flatlands/FlatLands_final_dataset.zip'), SELECTION, QUERIES,
                                 split='validation', verify_frozen=True, verify_query_geometry=True)
    try:
        samples = {s.observation.global_id: s for s in data}
    finally:
        data.close()
    map_count = metric_count = canonical_count = 0
    def jaccard(a, b):
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(intersection / union) if union else 1.
    for run in raw['runs']:
        folder = Path('results/current_baseline_k4_v1') / run['method'] / str(run['seed'])
        hashes({str(folder / f): v['sha256'] for f, v in run['outputs'].items()})
        if run['canonical_event_replay_max_drift'] is not None:
            assert run['canonical_event_replay_max_drift'] == 0
            canonical_count += 1
        records = {r['global_id']: r for r in json.loads((folder / 'map_metrics.json').read_text())}
        with np.load(folder / 'worlds.npz', allow_pickle=False) as saved:
            assert set(saved['global_ids']) == set(samples)
            for gid, worlds in zip(saved['global_ids'], saved['worlds']):
                s = samples[str(gid)]; r = records[str(gid)]; k = len(worlds)
                assert not np.any(worlds[:, ~s.epistemic_mask])
                assert np.array_equal(worlds[:, ~s.unknown], np.broadcast_to(s.observed_free[~s.unknown], (k, (~s.unknown).sum())))
                map_count += k
                mask = s.loss_mask.astype(bool)
                if not mask.any():
                    assert r['mean_iou'] is None
                    continue
                pred, y = worlds[:, mask], s.target_free[mask].astype(bool)
                values = [jaccard(p, y) for p in pred]
                pair = sum(1-jaccard(a,b) for i,a in enumerate(pred) for j,b in enumerate(pred) if i != j)
                expected = {'first_umr': np.mean(pred[0] != y), 'first_iou': values[0],
                            'mean_iou': np.mean(values), 'oracle_best_iou': max(values),
                            'mean_blocked_iou': np.mean([jaccard(~p, ~y) for p in pred]),
                            'sample_vote_cell_brier': np.mean((pred.mean(0)-y)**2),
                            'masked_energy_score': 1-np.mean(values)-pair/(2*k*(k-1)) if k>1 else None}
                for key, value in expected.items():
                    assert r[key] is None if value is None else abs(r[key]-value)<1e-12, (gid,key)
                    metric_count += 1
    assert map_count == 4800 and canonical_count == 9
    parent = json.loads(Path('results/flatlands_parent_groups_v2/report.json').read_text())
    history = json.loads(Path('results/unscenes_history_feasibility_v1/report.json').read_text())
    hashes(parent['source_hashes']); hashes(history['source_hashes'])
    rows = list(csv.DictReader(open('results/flatlands_parent_groups_v2/development_candidate.csv')))
    group_split = {}
    for r in rows:
        assert r['archive_split'] == 'train' and r['parent_group']
        assert group_split.setdefault(r['parent_group'], r['candidate_split']) == r['candidate_split']
    assert len(rows) == 215289 and len(group_split) == 4479
    assert parent['old_validation_observations_with_training_parent'] == 27
    assert not history['validation_labels_opened'] and not history['test_assets_opened']
    result = {'passed': True, 'saved_maps_checked': map_count, 'independent_map_metric_checks': metric_count,
              'canonical_event_replays_zero_drift': canonical_count, 'file_hashes_checked': checked,
              'candidate_observations': len(rows), 'candidate_parent_groups': len(group_split),
              'candidate_group_split_conflicts': 0, 'test_images_opened': False,
              'limits': 'Numerical replay does not cure the old physical-place leakage or establish external-paper superiority.'}
    Path('results/baseline_review_20260909_v1/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
