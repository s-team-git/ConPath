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
SEEDS = (20260910, 20260911, 20260912)
TRAINED_METHODS = ('correlated', 'independent', 'deterministic')
RULE_METHODS = ('all_floor', 'all_blocked', 'nearest_observed', 'train_radius_prior')
MAP_METRICS = ('mean_iou', 'mean_blocked_iou', 'sample_vote_cell_brier', 'masked_energy_score')


def check_numbers(actual, expected, context, tolerance=1e-12):
    """Reject shape changes and nonfinite values as well as numerical drift."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f'{context}: shape mismatch'
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), f'{context}: nonfinite value'
    assert np.allclose(actual, expected, rtol=0., atol=tolerance), f'{context}: value mismatch'


def parent_order(samples):
    # Pilot v1 has exactly one observation and at least one query per validation
    # parent. Fail explicitly if that changes: the frozen evaluator assumes it.
    parents = [s.row['parent_group'] for s in samples]
    assert len(parents) == len(set(parents)), 'Validation parents must be unique'
    assert all(len(s.targets) for s in samples), 'Validation parent has no queries'
    return parents, [s.row['source_dataset'] for s in samples]


def audit_parent_summary(summary, per_parent, samples):
    """Join CSV-derived errors by identity before checking the stored vector."""
    parents, sources = parent_order(samples)
    assert set(per_parent) == set(parents), 'CSV parent identities differ'
    assert summary['parents'] == parents, 'Report parent order differs'
    assert summary['sources'] == sources, 'Report source order differs'
    values = [float(np.mean(per_parent[parent])) for parent in parents]
    check_numbers(summary['parent_brier'], values, 'CSV parent Brier')
    check_numbers(summary['brier'], np.mean(values), 'CSV aggregate Brier')
    assert summary['events'] == sum(len(v) for v in per_parent.values()), 'Report event count differs'
    expected_sources = sorted(set(sources))
    assert set(summary['by_source']) == set(expected_sources), 'Report source identities differ'
    for source in expected_sources:
        check_numbers(summary['by_source'][source],
                      np.mean([v for v, name in zip(values, sources) if name == source]),
                      f'CSV source Brier: {source}')


def audit_analysis(analysis, method_reports, samples):
    """Recompute the final table and screen from independently audited reports.

    No model, checkpoint, or packet access occurs here. The report roster and
    seed order are explicit rather than inferred from whichever files exist.
    """
    parents, sources = parent_order(samples)
    event_count = sum(s.targets.size for s in samples)
    methods = (*TRAINED_METHODS, *RULE_METHODS)
    expected_runs = {(m, seed) for m in TRAINED_METHODS for seed in SEEDS}
    expected_runs.update((m, 'rule') for m in RULE_METHODS)
    by_run = {}
    for report in method_reports:
        key = report['method'], report['seed']
        assert key not in by_run, f'Duplicate report: {key}'
        by_run[key] = report
    assert set(by_run) == expected_runs, 'Incomplete or unexpected method/seed reports'
    assert set(analysis['methods']) == set(methods), 'Analysis method roster differs'
    parent_means, primary_scores = {}, {}
    budget_checks = 0
    for method in methods:
        seeds = SEEDS if method in TRAINED_METHODS else ('rule',)
        group = [by_run[method, seed] for seed in seeds]
        budgets = ('4', '32') if method in ('correlated', 'independent') else ('1',)
        combined = analysis['methods'][method]
        assert combined['repeats'] == len(seeds), f'{method}: repeat count differs'
        assert set(combined['budgets']) == set(budgets), f'{method}: analysis budgets differ'
        for report in group:
            assert set(report['budgets']) == set(budgets), f'{method}: report budgets differ'
        for k in budgets:
            summaries = [r['budgets'][k] for r in group]
            aggregate = combined['budgets'][k]
            assert set(aggregate) == {'brier', 'risk30', 'map'}, f'{method}/{k}: table fields differ'
            for summary in summaries:
                assert summary['parents'] == parents, f'{method}/{k}: parent order differs'
                assert summary['sources'] == sources, f'{method}/{k}: source order differs'
                assert summary['events'] == event_count, f'{method}/{k}: event count differs'
                vector = np.asarray(summary['parent_brier'], dtype=float)
                assert vector.shape == (len(parents),), f'{method}/{k}: parent vector length differs'
                assert np.isfinite(vector).all() and ((0 <= vector) & (vector <= 1)).all(), f'{method}/{k}: invalid parent Brier'
                check_numbers(summary['brier'], vector.mean(), f'{method}/{k}: parent mean')
            for metric in ('brier', 'risk30'):
                values = [s[metric] for s in summaries]
                recorded = aggregate[metric]
                assert set(recorded) == {'mean', 'sd', 'values'}, f'{method}/{k}/{metric}: fields differ'
                check_numbers(recorded['values'], values, f'{method}/{k}/{metric}: seed order')
                check_numbers(recorded['mean'], np.mean(values), f'{method}/{k}/{metric}: seed mean')
                if len(values) == 1:
                    assert recorded['sd'] is None, f'{method}/{k}/{metric}: single-run SD must be null'
                else:
                    check_numbers(recorded['sd'], np.std(values, ddof=1), f'{method}/{k}/{metric}: sample SD')
            if method == 'train_radius_prior':
                assert aggregate['map'] is None and all(s['map'] is None for s in summaries), 'Prior must not have map metrics'
            else:
                assert set(aggregate['map']) == set(MAP_METRICS), f'{method}/{k}: map metric fields differ'
                assert all(set(s['map']) == set(MAP_METRICS) for s in summaries), f'{method}/{k}: report map fields differ'
                for metric in MAP_METRICS:
                    values = [s['map'][metric] for s in summaries]
                    if metric == 'masked_energy_score' and k == '1':
                        assert aggregate['map'][metric] is None and all(v is None for v in values), 'K1 MES must be null'
                    else:
                        check_numbers(aggregate['map'][metric], np.mean(values), f'{method}/{k}/{metric}: map seed mean')
            budget_checks += 1
        primary = '32' if method in ('correlated', 'independent') else '1'
        parent_means[method] = np.mean([r['budgets'][primary]['parent_brier'] for r in group], axis=0)
        primary_scores[method] = {metric: float(np.mean([r['budgets'][primary][metric] for r in group]))
                                  for metric in ('brier', 'risk30')}

    # Reproduce the frozen source-stratified parent bootstrap without importing
    # the evaluator's aggregation or paired-interval helper.
    assert set(analysis['paired_brier']) == set(methods) - {'correlated'}, 'Paired comparison roster differs'
    rng = np.random.default_rng(20260909)
    for method in methods[1:]:
        delta = parent_means[method] - parent_means['correlated']
        strata = []
        for source in sorted(set(sources)):
            indices = np.flatnonzero(np.asarray(sources) == source)
            strata.append(delta[rng.choice(indices, size=(4000, len(indices)), replace=True)])
        interval = np.quantile(np.concatenate(strata, axis=1).mean(axis=1), [.025, .975])
        recorded = analysis['paired_brier'][method]
        check_numbers(recorded['other_minus_conpath'], delta.mean(), f'{method}: paired Brier difference')
        check_numbers(recorded['ci95'], interval, f'{method}: paired parent interval')
        assert recorded['unit'] == 'parent place, source stratified; conditional on the displayed seed average', 'Paired interval unit differs'

    conditions = {f'brier_better_than_{method}': primary_scores['correlated']['brier'] < primary_scores[method]['brier']
                  for method in ('independent', 'deterministic', 'all_floor')}
    conditions['risk30_not_worse_than_deterministic'] = primary_scores['correlated']['risk30'] <= primary_scores['deterministic']['risk30']
    assert set(analysis['screen_conditions']) == set(conditions), 'Screen condition roster differs'
    for name, expected in conditions.items():
        assert analysis['screen_conditions'][name] is expected, f'Screen condition differs: {name}'
    assert analysis['research_screen_passed'] is all(conditions.values()), 'Final screen decision differs'
    assert analysis['training_runs_completed'] == len(TRAINED_METHODS) * len(SEEDS), 'Training run count differs'
    assert analysis['validation_parents'] == len(parents), 'Validation parent count differs'
    assert analysis['validation_events'] == event_count, 'Validation event count differs'
    return {'passed': True, 'method_seed_reports': len(by_run), 'method_budget_aggregates': budget_checks,
            'parent_order_checked_per_report_budget': True, 'paired_parent_intervals': len(methods)-1,
            'bootstrap_seed': 20260909, 'bootstrap_repetitions': 4000,
            'recomputed_screen_conditions': conditions, 'recomputed_research_screen_passed': all(conditions.values())}


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
    reports, method_reports = [], []
    for path in sorted((RUN / 'evaluation').glob('*/*/report.json')):
        report = json.loads(path.read_text()); folder = path.parent
        assert folder.parent.name == report['method'] and folder.name == str(report['seed']), 'Report path identity differs'
        method_reports.append(report)
        for name, checksum in report['files'].items():
            assert sha(folder/name) == checksum
        all_worlds = None
        if (folder / 'worlds.npz').exists():
            with np.load(folder/'worlds.npz', allow_pickle=False) as z:
                assert z['global_ids'].tolist() == [s.row['global_id'] for s in samples]
                all_worlds = z['worlds'].copy()
            expected_k = 32 if report['method'] in ('correlated', 'independent') else 1
            assert all_worlds.shape == (len(samples), expected_k, *samples[0].target.shape), 'Saved world count/shape differs'
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
                assert np.isfinite(p) and 0 <= p <= 1, 'Invalid CSV event probability'
                per_parent[row['parent_group']].append((p-y)**2)
                risk_rows.append((row['parent_group'], p, y))
            audit_parent_summary(summary, per_parent, samples)
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
    analysis_audit = audit_analysis(analysis, method_reports, samples)
    result = {'passed': True, 'worlds_checked': worlds_checked, 'explicit_erosion_connectivity_worlds': geometry_checks, 'independent_target_query_labels_checked': len(samples),
              'prediction_vote_checks': prediction_checks, 'map_aggregate_checks': metric_checks,
              'method_seed_budget_reports': reports, 'analysis_sha256': sha(RUN/'analysis.json'),
              'validation_events_per_report': len(expected), 'cached_development_packets_only': True,
              'new_archive_images_read_by_this_audit': 0, 'analysis_aggregate_audit': analysis_audit,
              'audit_source_sha256': sha(Path(__file__))}
    _atomic_json(RUN / 'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'method_seed_budget_reports'}), flush=True)


if __name__ == '__main__':
    main()
