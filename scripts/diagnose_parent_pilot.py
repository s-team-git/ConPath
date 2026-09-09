#!/usr/bin/env python3
"""Post-evaluation diagnostics from saved predictions; no new model inference."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.parent_pilot_data import load_pilot, sha
from scripts.evaluate_flatlands_support_clamped import _accelerated_events
from scripts.evaluate_parent_group_pilot import risk_at_coverage
from scripts.train_flatlands_conpath import _atomic_json

RUN = ROOT / 'results/parent_group_pilot_v1'
SEEDS = (20260910, 20260911, 20260912)
RADII = (0, 10, 20)


def summary(probabilities, targets, parents, keep):
    p, y, parent = probabilities[keep], targets[keep], parents[keep]
    if not len(p):
        return None
    names, counts = np.unique(parent, return_counts=True)
    sizes = dict(zip(names, counts))
    weights = np.array([1. / sizes[name] for name in parent])
    return {'parents': len(names), 'events': len(p),
            'brier': float(np.average((p-y)**2, weights=weights)),
            'positive_fraction': float(np.average(y, weights=weights)),
            'risk30': float(risk_at_coverage(p, y, weights))}


def main():
    analysis = json.loads((RUN / 'analysis.json').read_text())
    verification = json.loads((RUN / 'verification.json').read_text())
    assert verification['passed'] is True
    assert verification['analysis_sha256'] == sha(RUN / 'analysis.json')
    samples = load_pilot(RUN / 'data', 'validation')
    metadata = []
    for sample in samples:
        for q, candidate in enumerate(sample.candidate_indices):
            gy, gx = sample.goals[q]
            for r, radius in enumerate(RADII):
                metadata.append({'key': (sample.row['global_id'], int(candidate), radius),
                                 'parent': sample.row['parent_group'], 'source': sample.row['source_dataset'],
                                 'radius': radius, 'goal_blocked': not bool(sample.target[gy, gx]),
                                 'target': bool(sample.targets[q, r])})
    keys = [r['key'] for r in metadata]
    assert len(set(keys)) == len(keys) == analysis['validation_events']
    y = np.array([r['target'] for r in metadata], dtype=float)
    parents = np.array([r['parent'] for r in metadata])
    masks = {'all': np.ones(len(y), dtype=bool)}
    for source in sorted({r['source'] for r in metadata}):
        masks['source/' + source] = np.array([r['source'] == source for r in metadata])
    for radius in RADII:
        masks[f'radius/{radius}'] = np.array([r['radius'] == radius for r in metadata])
    for blocked in (True, False):
        masks['goal/' + ('blocked' if blocked else 'free')] = np.array([r['goal_blocked'] == blocked for r in metadata])

    predictions, hashes = {}, {}
    for method in analysis['methods']:
        seeds = SEEDS if method in ('correlated', 'independent', 'deterministic') else ('rule',)
        budget = 32 if method in ('correlated', 'independent') else 1
        for seed in seeds:
            path = RUN / 'evaluation' / method / str(seed) / f'predictions_k{budget}.csv'
            with path.open() as f:
                rows = list(csv.DictReader(f))
            indexed = {(r['global_id'], int(r['candidate_index']), int(r['radius_cells'])): float(r['probability']) for r in rows}
            assert set(indexed) == set(keys) and len(rows) == len(keys)
            predictions[method, seed] = np.array([indexed[k] for k in keys])
            hashes[str(path.relative_to(ROOT))] = sha(path)
    lower, upper = predictions['all_blocked', 'rule'], predictions['all_floor', 'rule']
    assert (lower <= upper).all()
    immutable = lower == upper
    assert not (immutable & (y != lower)).any(), 'Observed-consistent pilot has an unexpected unavoidable label conflict'
    masks['observations/immutable'] = immutable
    masks['observations/mutable'] = ~immutable

    for method in ('correlated', 'independent'):
        for seed in SEEDS:
            path = RUN / 'evaluation' / method / str(seed) / 'conditional_mean_maps.npz'
            with np.load(path, allow_pickle=False) as z:
                assert z['global_ids'].tolist() == [s.row['global_id'] for s in samples]
                maps = z['probabilities'].copy()
            votes = []
            for sample, probability in zip(samples, maps):
                world = (probability >= .5) & sample.valid
                world[~sample.hidden] = (sample.observation[0] > .5)[~sample.hidden]
                votes.extend(_accelerated_events(world[None], sample.starts, sample.goals, RADII)[0].ravel())
            predictions[method + '_conditional_mean_threshold', seed] = np.asarray(votes, dtype=float)
            hashes[str(path.relative_to(ROOT))] = sha(path)

    by_run = []
    for (method, seed), p in predictions.items():
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        if method != 'train_radius_prior':
            assert ((p >= lower) & (p <= upper)).all()
        by_run.append({'method': method, 'seed': seed,
                       'groups': {name: summary(p, y, parents, keep) for name, keep in masks.items()}})
    combined = {}
    for method in dict.fromkeys(row['method'] for row in by_run):
        runs = [r for r in by_run if r['method'] == method]
        combined[method] = {}
        for name in masks:
            values = [r['groups'][name] for r in runs]
            if values[0] is None:
                combined[method][name] = None
                continue
            combined[method][name] = {k: values[0][k] for k in ('parents', 'events', 'positive_fraction')}
            for metric in ('brier', 'risk30'):
                v = [r[metric] for r in values]
                combined[method][name][metric] = {'mean': float(np.mean(v)), 'values': v,
                                                'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None}
    histories = []
    for path in sorted((RUN / 'runs').glob('*/*/history.json')):
        history = json.loads(path.read_text())
        last = history[-1]
        histories.append({'method': last['method'], 'seed': last['seed'], 'epochs': len(history),
                          'best_epoch': last['best_epoch'], 'best_calibration_brier': last['best_brier'],
                          'first': history[0], 'last': last,
                          'best_in_last_four_epochs': last['best_epoch'] > len(history)-4})
        hashes[str(path.relative_to(ROOT))] = sha(path)
    report = {'id': 'parent_pilot_posthoc_diagnostics_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'posthoc': True, 'changes_prespecified_screen': False,
              'source_analysis_sha256': sha(RUN / 'analysis.json'), 'script_sha256': sha(Path(__file__)),
              'new_model_inference': False, 'new_physical_test_reads': 0,
              'immutable_event_fraction_unweighted': float(immutable.mean()),
              'unavoidable_brier_from_observed_constraints': 0.,
              'scope': 'Each subgroup reweights its contributing parents equally; subgroup risk uses its own top 30%. Conditional-mean threshold controls were added after the original evaluation, with no retraining. Threshold is fixed at 0.5, not tuned.',
              'groups': combined, 'runs': by_run, 'training_trends': histories, 'source_hashes': hashes}
    _atomic_json(RUN / 'diagnostics/report.json', report)
    print(json.dumps({'diagnostics_saved': True, 'immutable_fraction': float(immutable.mean()),
                      'overall': {m: g['all'] for m, g in combined.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
