#!/usr/bin/env python3
"""One held-out development comparison, after every baseline has selected its checkpoint."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.benchmark_metrics import masked_map_metrics
from pathrel.parent_pilot_data import load_pilot, sha
from scripts.evaluate_flatlands_support_clamped import _accelerated_events, _verify_accelerator
from scripts.train_flatlands_conpath import _atomic_json
from scripts.train_parent_group_pilot import model_for, predict, RADII

OUT = ROOT / 'results/parent_group_pilot_v1'
DATA = OUT / 'data'
SEEDS = (20260910, 20260911, 20260912)
METHODS = ('correlated', 'independent', 'deterministic')


def risk_at_coverage(probabilities, targets, weights, coverage=.3):
    """Fractional inclusion of a tied score, never label-based tie ordering."""
    p, y, w = map(np.asarray, (probabilities, targets, weights))
    if p.shape != y.shape or p.shape != w.shape or p.ndim != 1 or not 0 < coverage <= 1 or w.sum() <= 0:
        raise ValueError('Invalid weighted coverage inputs')
    w = w / w.sum()
    accepted = errors = 0.
    for value in np.unique(p)[::-1]:
        subset = p == value
        mass = float(w[subset].sum())
        take = min(mass, max(0., coverage-accepted))
        if mass > 0:
            errors += take * float(np.sum(w[subset] * (1-y[subset]))) / mass
        accepted += take
        if accepted >= coverage - 1e-12:
            break
    return errors / accepted


def event_summary(records, samples):
    brier, values, labels, weights = [], [], [], []
    parents, sources = [], []
    for sample, record in zip(samples, records):
        p = np.asarray(record['probabilities'], dtype=float)
        y = sample.targets
        assert p.shape == y.shape and np.isfinite(p).all() and np.all((0 <= p) & (p <= 1))
        if not len(p):
            continue
        brier.append(float(np.mean((p-y)**2)))
        values.extend(p.ravel()); labels.extend(y.ravel()); weights.extend(np.ones(p.size) / p.size)
        parents.append(sample.row['parent_group']); sources.append(sample.row['source_dataset'])
    return {'brier': float(np.mean(brier)), 'parent_brier': brier,
            'risk30': risk_at_coverage(values, labels, weights), 'parents': parents, 'sources': sources,
            'events': len(values), 'by_source': {s: float(np.mean([v for v, src in zip(brier, sources) if src == s])) for s in sorted(set(sources))}}


def rule_world(sample, method):
    if method == 'all_floor':
        world = (sample.observation[0] > .5) | sample.hidden
    elif method == 'all_blocked':
        world = sample.observation[0] > .5
    elif method == 'nearest_observed':
        known = sample.valid & ~sample.hidden
        if not known.any():
            raise ValueError('No observed evidence for nearest rule')
        indices = distance_transform_edt(~known, return_distances=False, return_indices=True)
        nearest_free = sample.observation[0][tuple(indices)] > .5
        world = np.where(sample.hidden, nearest_free, sample.observation[0] > .5)
    else:
        raise ValueError(method)
    return (world & sample.valid)[None]


def save_method(method, seed, samples, worlds, events, source, prior=None, mean_maps=None):
    folder = OUT / 'evaluation' / method / str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    budgets = (4, 32) if method in ('correlated', 'independent') else (1,)
    report = {'method': method, 'seed': seed, 'source': source, 'budgets': {}}
    if mean_maps is not None:
        np.savez_compressed(folder / 'conditional_mean_maps.npz', probabilities=np.stack(mean_maps), global_ids=np.array([s.row['global_id'] for s in samples]))
    if worlds:
        np.savez_compressed(folder / 'worlds.npz', worlds=np.stack(worlds), global_ids=np.array([s.row['global_id'] for s in samples]))
    for k in budgets:
        records, maps, flat_rows = [], [], []
        for i, sample in enumerate(samples):
            p = events[i][:k].mean(0) if prior is None else np.broadcast_to(prior, sample.targets.shape)
            records.append({'global_id': sample.row['global_id'], 'probabilities': p.tolist()})
            if worlds:
                maps.append({'global_id': sample.row['global_id'], **masked_map_metrics(worlds[i][:k], sample.target, sample.hidden)})
            for qi, candidate in enumerate(sample.candidate_indices):
                for ri, radius in enumerate(RADII):
                    flat_rows.append({'global_id': sample.row['global_id'], 'parent_group': sample.row['parent_group'],
                                      'candidate_index': int(candidate), 'radius_cells': radius, 'probability': float(p[qi, ri])})
        with (folder / f'predictions_k{k}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0])); writer.writeheader(); writer.writerows(flat_rows)
        summary = event_summary(records, samples)
        summary['map'] = {key: float(np.mean([r[key] for r in maps if r[key] is not None]))
                          if any(r[key] is not None for r in maps) else None
                          for key in ('mean_iou', 'mean_blocked_iou', 'sample_vote_cell_brier', 'masked_energy_score')} if maps else None
        report['budgets'][str(k)] = summary
        _atomic_json(folder / f'maps_k{k}.json', maps)
    report['files'] = {str(p.name): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    _atomic_json(folder / 'report.json', report)
    print(json.dumps({'evaluated': method, 'seed': seed, 'scores': {k: {'brier': v['brier'], 'risk30': v['risk30']} for k, v in report['budgets'].items()}}), flush=True)
    return report


def paired_interval(a, b, source_names, rng, repetitions=4000):
    """Resample the same parents for both methods, separately within each source."""
    d = np.asarray(b) - np.asarray(a)
    draws = []
    for source in sorted(set(source_names)):
        indices = np.flatnonzero(np.array(source_names) == source)
        draws.append(d[rng.choice(indices, (repetitions, len(indices)), replace=True)])
    boot = np.concatenate(draws, axis=1).mean(1)
    return {'other_minus_conpath': float(d.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'unit': 'parent place, source stratified; conditional on the displayed seed average'}


def main():
    if (OUT / 'evaluation').exists():
        raise ValueError('Refusing to overwrite evaluation; inspect any partial execution')
    checks = []
    for method in METHODS:
        for seed in SEEDS:
            folder = OUT / 'runs' / method / str(seed)
            complete = json.loads((folder / 'complete.json').read_text())
            recipe = json.loads((folder / 'recipe.json').read_text())
            assert sha(folder / 'best.pt') == complete['checkpoint_sha256']
            assert sha(folder / 'recipe.json') == complete['recipe_sha256']
            assert recipe['data_seal_sha256'] == sha(DATA / 'seal.json')
            for name, checksum in recipe['source_sha256'].items():
                assert sha(ROOT / name) == checksum, name
            checks.append(complete)
    assert len(checks) == 9
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    _verify_accelerator()
    (OUT / 'evaluation').mkdir()
    _atomic_json(OUT / 'evaluation/evaluation_frozen_before_scoring.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'nine_completed_runs': checks,
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/pathrel/benchmark_metrics.py',
                         ROOT / 'scripts/train_parent_group_pilot.py', ROOT / 'src/pathrel/parent_pilot_data.py']},
        'data_seal_sha256': sha(DATA / 'seal.json'), 'validation_model_scoring_previously_started': False})
    samples = load_pilot(DATA, 'validation')
    assert len(samples) == 40
    reports = []
    for method in METHODS:
        for seed in SEEDS:
            folder = OUT / 'runs' / method / str(seed)
            state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            model = model_for(method); model.load_state_dict(state['model']); model.eval()
            worlds, events, mean_maps = [], [], []
            for sample in samples:
                w, e, p, _ = predict(model, method, sample, seed + 4000000, 32)
                worlds.append(w); events.append(e); mean_maps.append(p)
            reports.append(save_method(method, seed, samples, worlds, events,
                           {'checkpoint': str((folder / 'best.pt').relative_to(ROOT)), 'sha256': sha(folder / 'best.pt'), 'best_epoch': state['best_epoch']}, mean_maps=mean_maps))
            del model, state
    for method in ('all_floor', 'all_blocked', 'nearest_observed'):
        worlds = [rule_world(s, method) for s in samples]
        events = [_accelerated_events(w, s.starts, s.goals, RADII) for w, s in zip(worlds, samples)]
        reports.append(save_method(method, 'rule', samples, worlds, events, {'trained': False}))
    train = load_pilot(DATA, 'train')
    prior = np.mean([s.targets.mean(0) for s in train if len(s.targets)], axis=0)
    reports.append(save_method('train_radius_prior', 'rule', samples, [], [],
                              {'trained': False, 'fitted_train_parents': sum(bool(len(s.targets)) for s in train), 'probabilities': prior.tolist()}, prior))
    combined, parent_means = {}, {}
    for method in (*METHODS, 'all_floor', 'all_blocked', 'nearest_observed', 'train_radius_prior'):
        group = [r for r in reports if r['method'] == method]
        budgets = group[0]['budgets']
        combined[method] = {'repeats': len(group), 'budgets': {}}
        for k in budgets:
            keys = ('brier', 'risk30')
            values = {key: [r['budgets'][k][key] for r in group] for key in keys}
            combined[method]['budgets'][k] = {key: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v)>1 else None, 'values': v} for key, v in values.items()}
            maps = [r['budgets'][k]['map'] for r in group]
            combined[method]['budgets'][k]['map'] = {key: float(np.mean([m[key] for m in maps])) if maps[0][key] is not None else None for key in maps[0]} if maps[0] else None
        primary = '32' if method in ('correlated', 'independent') else '1'
        parent_means[method] = np.mean([r['budgets'][primary]['parent_brier'] for r in group], axis=0)
    sources = [s.row['source_dataset'] for s in samples]
    rng = np.random.default_rng(20260909)
    paired = {m: paired_interval(parent_means['correlated'], p, sources, rng) for m, p in parent_means.items() if m != 'correlated'}
    def value(method, metric):
        k = '32' if method in ('correlated', 'independent') else '1'
        return combined[method]['budgets'][k][metric]['mean']
    conditions = {f'brier_better_than_{m}': value('correlated', 'brier') < value(m, 'brier') for m in ('independent', 'deterministic', 'all_floor')}
    conditions['risk30_not_worse_than_deterministic'] = value('correlated', 'risk30') <= value('deterministic', 'risk30')
    analysis = {'id': 'parent_group_pilot_v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'methods': combined,
                'paired_brier': paired, 'screen_conditions': conditions, 'research_screen_passed': all(conditions.values()),
                'implementation_checks_passed': True, 'training_runs_completed': 9, 'validation_parents': 40, 'validation_events': 1545,
                'protocol': json.loads((DATA / 'protocol.json').read_text()),
                'training': checks, 'data_audit_summary': json.loads((DATA / 'data_audit.json').read_text())['summary'],
                'new_physical_test_images_opened': 0, 'final_test': False, 'direct_comparison_to_published_scores': False,
                'claim': 'Small-budget independent-place development comparison; CIs condition on 3 seed means. External common-protocol baselines and fresh final holdout remain pending.'}
    _atomic_json(OUT / 'analysis.json', analysis)
    _atomic_json(ROOT / 'site/data/parent_group_pilot_zh.json', analysis)
    print(json.dumps({'baseline_complete': True, 'research_screen_passed': analysis['research_screen_passed'], 'conditions': conditions}), flush=True)


if __name__ == '__main__':
    main()
