#!/usr/bin/env python3
"""Verify saved worlds/CSV joins and independently test disk-eroded connectivity."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_erosion, label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.parent_pilot_data import load_pilot, sha
from scripts.evaluate_flatlands_support_clamped import _accelerated_events
from scripts.train_flatlands_conpath import _atomic_json

RUN = ROOT / 'results/parent_group_pilot_v1'


def direct_events(world, starts, goals):
    result = []
    for radius in (0, 10, 20):
        yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
        centers = binary_erosion(world, structure=xx*xx+yy*yy <= radius*radius, border_value=0)
        components, _ = label(centers, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        a, b = components[tuple(starts.T)], components[tuple(goals.T)]
        result.append((a != 0) & (a == b))
    return np.stack(result, axis=-1)


def direct_map_metrics(worlds, sample):
    w, y = worlds[:, sample.hidden], sample.target[sample.hidden]
    def overlap(a, b):
        union = (a | b).sum(-1)
        return np.divide((a & b).sum(-1), union, out=np.ones(union.shape, dtype=float), where=union != 0)
    iou = overlap(w, y)
    k = len(w)
    diversity = 0.
    for i in range(k):
        for j in range(k):
            if i != j:
                diversity += float(1-overlap(w[i], w[j]))
    return {'mean_iou': float(iou.mean()), 'mean_blocked_iou': float(overlap(~w, ~y).mean()),
            'sample_vote_cell_brier': float(np.mean((w.mean(0)-y)**2)),
            'masked_energy_score': float(np.mean(1-iou)-diversity/(2*k*(k-1))) if k>1 else None}


def direct_risk(rows):
    # Sort score groups, then integrate a fractional top-30% prefix.
    per_parent = defaultdict(list)
    for parent, p, y in rows:
        per_parent[parent].append((p, y))
    groups = defaultdict(lambda: [0., 0.])
    for events in per_parent.values():
        weight = 1 / (len(per_parent)*len(events))
        for p, y in events:
            groups[p][0] += weight
            groups[p][1] += weight * (1-y)
    remainder, unsafe = .3, 0.
    for p in sorted(groups, reverse=True):
        mass, errors = groups[p]
        used = min(remainder, mass)
        unsafe += used * errors / mass
        remainder -= used
        if remainder < 1e-12:
            break
    return unsafe / .3


def main():
    analysis = json.loads((RUN / 'analysis.json').read_text())
    samples = load_pilot(RUN / 'data', 'validation')
    parent_by_id = {s.row['global_id']: s.row['parent_group'] for s in samples}
    for s in samples:
        assert np.array_equal(direct_events(s.target, s.starts, s.goals), s.targets)
    expected = {(s.row['global_id'], int(c), radius): bool(s.targets[q, r]) for s in samples
                for q, c in enumerate(s.candidate_indices) for r, radius in enumerate((0, 10, 20))}
    geometry_checks = worlds_checked = prediction_checks = metric_checks = 0
    reports = []
    for path in sorted((RUN / 'evaluation').glob('*/*/report.json')):
        report = json.loads(path.read_text()); folder = path.parent
        for name, checksum in report['files'].items():
            assert sha(folder/name) == checksum
        all_worlds = None
        if (folder / 'worlds.npz').exists():
            with np.load(folder/'worlds.npz', allow_pickle=False) as z:
                assert z['global_ids'].tolist() == [s.row['global_id'] for s in samples]
                all_worlds = z['worlds'].copy()
            for sample, worlds in zip(samples, all_worlds):
                assert worlds.dtype == bool
                assert not (worlds & ~sample.valid).any()
                assert not (worlds[:, ~sample.hidden] != sample.observation[0, ~sample.hidden]).any()
                fast = _accelerated_events(worlds, sample.starts, sample.goals, (0, 10, 20))
                # Different implementation: explicit disk erosion and component labels.
                for wi in sorted({0, len(worlds)-1}):
                    assert np.array_equal(direct_events(worlds[wi], sample.starts, sample.goals), fast[wi])
                    geometry_checks += 1
                worlds_checked += len(worlds)
        for k, summary in report['budgets'].items():
            rows = list(csv.DictReader((folder/f'predictions_k{k}.csv').open()))
            actual = {(r['global_id'], int(r['candidate_index']), int(r['radius_cells'])): float(r['probability']) for r in rows}
            assert len(actual) == len(rows) == 1545 and set(actual) == set(expected)
            per_parent = defaultdict(list); risk_rows = []
            for row in rows:
                assert row['parent_group'] == parent_by_id[row['global_id']]
                key = row['global_id'], int(row['candidate_index']), int(row['radius_cells'])
                y, p = expected[key], float(row['probability'])
                per_parent[row['parent_group']].append((p-y)**2)
                risk_rows.append((row['parent_group'], p, y))
            assert abs(float(np.mean([np.mean(v) for v in per_parent.values()]))-summary['brier']) < 1e-12
            assert abs(direct_risk(risk_rows)-summary['risk30']) < 1e-12
            if all_worlds is not None:
                map_rows = []
                for sample, worlds in zip(samples, all_worlds):
                    events = _accelerated_events(worlds[:int(k)], sample.starts, sample.goals, (0, 10, 20)).mean(0)
                    for q, candidate in enumerate(sample.candidate_indices):
                        for ri, radius in enumerate((0, 10, 20)):
                            assert events[q, ri] == actual[(sample.row['global_id'], int(candidate), radius)]
                            prediction_checks += 1
                    map_rows.append(direct_map_metrics(worlds[:int(k)], sample))
                for metric in map_rows[0]:
                    estimate = float(np.mean([r[metric] for r in map_rows])) if map_rows[0][metric] is not None else None
                    if estimate is None:
                        assert summary['map'][metric] is None
                    else:
                        assert abs(estimate-summary['map'][metric]) < 1e-10
                    metric_checks += 1
            reports.append({'method': report['method'], 'seed': report['seed'], 'K': int(k), 'passed': True})
    assert len(reports) == 19
    result = {'passed': True, 'worlds_checked': worlds_checked, 'explicit_erosion_connectivity_worlds': geometry_checks, 'independent_target_query_labels_checked': len(samples),
              'prediction_vote_checks': prediction_checks, 'map_aggregate_checks': metric_checks,
              'method_seed_budget_reports': reports, 'analysis_sha256': sha(RUN/'analysis.json'),
              'validation_events_per_report': len(expected), 'cached_development_packets_only': True,
              'new_archive_images_read_by_this_audit': 0}
    _atomic_json(RUN / 'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'method_seed_budget_reports'}), flush=True)


if __name__ == '__main__':
    main()
