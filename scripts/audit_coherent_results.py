#!/usr/bin/env python3
"""CPU-only, independently rerunnable audit of the fixed two-seed follow-up.

Only cached development packets and persisted result artifacts are accessed.
The original evaluator is never run and no model inference is performed.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.parent_pilot_data import load_pilot, sha
from scripts.audit_parent_pilot_results import (
    MAP_METRICS, audit_parent_summary, check_numbers, direct_events,
    direct_map_metrics, direct_risk, parent_order,
)
from scripts.evaluate_flatlands_support_clamped import _accelerated_events

BASE = ROOT / 'results/parent_group_pilot_v1'
OUT = ROOT / 'results/coherent_parent_pilot_v1'
SEEDS = (20260910, 20260911)
METHODS = ('coherent_categorical', 'correlated', 'independent', 'deterministic')
RADII = (0, 10, 20)


def recompute_selection(history, config):
    """Replay the declared tolerance, tie-break, and stopping rule independently."""
    best, best_epoch, patience = (float('inf'), float('inf')), 0, 0
    for epoch, row in enumerate(history, 1):
        assert epoch <= config['max_epochs']
        assert not (epoch > config['minimum_epochs'] and patience >= config['patience']), 'Training exceeded stopping rule'
        assert row['epoch'] == epoch
        current = (row['calibration']['event_brier'], row['calibration']['map_nll'])
        assert np.isfinite(current).all()
        improved = bool(current[0] < best[0]-1e-5 or (abs(current[0]-best[0]) <= 1e-5 and current[1] < best[1]-1e-5))
        if improved:
            best, best_epoch, patience = current, epoch, 0
        else:
            patience += 1
        assert row['improved'] is improved
        assert row['best_epoch'] == best_epoch and row['patience'] == patience
        check_numbers(row['best_brier'], best[0], 'History selected Brier')
    assert len(history) == config['max_epochs'] or (len(history) >= config['minimum_epochs'] and patience >= config['patience']), 'Incomplete training'
    return best, best_epoch, patience


def audit_aggregate(analysis, reports, samples):
    """Rebuild all four methods and bootstrap without evaluator aggregation code."""
    parents, sources = parent_order(samples)
    assert set(reports) == {(method, seed) for method in METHODS for seed in SEEDS}
    assert set(analysis['methods']) == set(METHODS)
    scores, parent_means = {}, {}
    for method in METHODS:
        k = '1' if method == 'deterministic' else '32'
        summaries = [reports[method, seed]['budgets'][k] for seed in SEEDS]
        result = analysis['methods'][method]
        assert result['seeds'] == list(SEEDS) and result['samples'] == int(k)
        for summary in summaries:
            assert summary['parents'] == parents and summary['sources'] == sources
            assert summary['events'] == sum(s.targets.size for s in samples)
        scores[method] = {}
        for metric in ('brier', 'risk30'):
            values = [s[metric] for s in summaries]
            assert set(result[metric]) == {'values', 'mean', 'sd'}
            check_numbers(result[metric]['values'], values, f'{method}/{metric} seed ordering')
            check_numbers(result[metric]['mean'], np.mean(values), f'{method}/{metric} two-seed mean')
            check_numbers(result[metric]['sd'], np.std(values, ddof=1), f'{method}/{metric} sample SD')
            scores[method][metric] = float(np.mean(values))
        assert set(result['map']) == set(MAP_METRICS)
        for metric in MAP_METRICS:
            values = [s['map'][metric] for s in summaries]
            if metric == 'masked_energy_score' and k == '1':
                assert result['map'][metric] is None and values == [None, None]
            else:
                check_numbers(result['map'][metric], np.mean(values), f'{method}/{metric} two-seed map mean')
        parent_means[method] = np.mean([s['parent_brier'] for s in summaries], axis=0)
    assert set(analysis['paired_brier']) == set(METHODS[1:])
    rng = np.random.default_rng(20260909)
    for method in METHODS[1:]:
        delta = parent_means[method] - parent_means['coherent_categorical']
        strata = []
        for source in sorted(set(sources)):
            indices = np.flatnonzero(np.asarray(sources) == source)
            strata.append(delta[rng.choice(indices, size=(4000, len(indices)), replace=True)])
        interval = np.quantile(np.concatenate(strata, axis=1).mean(axis=1), [.025, .975])
        result = analysis['paired_brier'][method]
        check_numbers(result['other_minus_conpath'], delta.mean(), f'{method} paired difference')
        check_numbers(result['ci95'], interval, f'{method} paired source-stratified interval')
        assert result['unit'] == 'parent place, source stratified; conditional on the displayed two-seed average'
    candidate, baseline = scores['coherent_categorical'], scores['correlated']
    conditions = {
        'brier_better_than_matched_conpath': candidate['brier'] < baseline['brier'],
        'risk30_not_worse_than_matched_conpath': candidate['risk30'] <= baseline['risk30'],
    }
    assert set(analysis['screen_conditions']) == set(conditions)
    for name, expected in conditions.items():
        assert analysis['screen_conditions'][name] is expected, f'{name}: strict boolean mismatch'
    assert analysis['development_screen_passed'] is all(conditions.values())
    assert analysis['training_seeds'] == list(SEEDS)
    assert analysis['validation_parents'] == len(parents) == 40
    assert analysis['validation_events'] == sum(s.targets.size for s in samples) == 1545
    assert analysis['validation_reused_for_model_development'] is True
    assert analysis['final_test'] is False and analysis['no_direct_published_paper_ranking'] is True
    assert type(analysis['new_physical_test_reads']) is int and analysis['new_physical_test_reads'] == 0
    return {'method_seed_reports': len(reports), 'paired_parent_intervals': 3,
            'bootstrap_seed': 20260909, 'bootstrap_repetitions': 4000,
            'baseline_seeds_matched_exactly': list(SEEDS), 'parent_order_checked': True,
            'screen_conditions': conditions, 'development_screen_passed': all(conditions.values()),
            'original_conpath_ci_crosses_zero': bool(analysis['paired_brier']['correlated']['ci95'][0] < 0 < analysis['paired_brier']['correlated']['ci95'][1])}


def main():
    torch.set_num_threads(2)
    files = {}
    def digest(path, expected=None):
        path = Path(path)
        value = sha(path)
        if expected is not None:
            assert value == expected, f'Checksum differs: {path}'
        files[str(path.relative_to(ROOT))] = value
        return value
    def read(path):
        digest(path)
        return json.loads(Path(path).read_text())
    analysis, protocol = read(OUT / 'analysis.json'), read(OUT / 'protocol.json')
    baseline_analysis, baseline_verification = read(BASE / 'analysis.json'), read(BASE / 'verification.json')
    original_verification = read(OUT / 'verification.json')
    protocol_hash, analysis_hash = digest(OUT / 'protocol.json'), digest(OUT / 'analysis.json')
    assert analysis['protocol_sha256'] == protocol_hash
    assert original_verification['passed'] is True and original_verification['analysis_sha256'] == analysis_hash
    assert protocol['seeds'] == list(SEEDS)
    assert protocol['final_test'] is False and protocol['validation_reused_for_model_development'] is True
    assert protocol['automatic_parameter_search'] is False
    assert protocol['screen'] == ['mean K32 Brier below matched original ConPath', 'risk30 no worse than matched original ConPath']
    assert protocol['baseline_screen_passed'] is True and baseline_analysis['research_screen_passed'] is True
    assert baseline_verification['passed'] is True
    assert baseline_verification['audit_source_sha256'] == protocol['source_hashes']['scripts/audit_parent_pilot_results.py']
    assert baseline_verification['analysis_sha256'] == digest(BASE / 'analysis.json', protocol['baseline_analysis_sha256'])
    assert analysis['baseline_analysis_sha256'] == protocol['baseline_analysis_sha256']
    digest(BASE / 'verification.json', protocol['baseline_verification_sha256'])
    assert original_verification['source_hashes'] == protocol['source_hashes']
    for name, expected in protocol['source_hashes'].items():
        digest(ROOT / name, expected)
    digest(ROOT / 'src/pathrel/data_access.py')
    digest(BASE / 'data/seal.json', protocol['data_seal_sha256'])
    seal = read(BASE / 'data/seal.json')
    for name, expected in seal['files'].items():
        digest(BASE / 'data' / name, expected)
    data_protocol = read(BASE / 'data/protocol.json')
    assert protocol['training'] == data_protocol['training']
    frozen = read(OUT / 'evaluation/frozen_before_scoring.json')
    assert frozen['protocol_sha256'] == protocol_hash
    assert frozen['script_sha256'] == protocol['source_hashes']['scripts/evaluate_coherent_parent_pilot.py']
    assert protocol['created_before_training_utc'] < frozen['timestamp_utc'] < analysis['created_utc']
    samples = load_pilot(BASE / 'data', 'validation')
    parents, sources = parent_order(samples)
    assert len(parents) == 40 and sorted(Counter(sources).values()) == [8] * 5
    expected = {(s.row['global_id'], int(c), r): bool(s.targets[q, ri]) for s in samples
                for q, c in enumerate(s.candidate_indices) for ri, r in enumerate(RADII)}
    assert len(expected) == 1545
    parent_by_id = {s.row['global_id']: s.row['parent_group'] for s in samples}
    counts = Counter()
    checkpoints, reports = [], {}
    for sample in samples:
        assert np.array_equal(direct_events(sample.target, sample.starts, sample.goals), sample.targets)
        counts['independent_target_maps'] += 1
    for method in METHODS:
        for seed in SEEDS:
            candidate = method == 'coherent_categorical'
            run = OUT / 'runs' / str(seed) if candidate else BASE / 'runs' / method / str(seed)
            folder = OUT / 'evaluation' / str(seed) if candidate else BASE / 'evaluation' / method / str(seed)
            recipe, complete, history = (read(run / name) for name in ('recipe.json', 'complete.json', 'history.json'))
            assert recipe['method'] == complete['method'] == method and recipe['seed'] == complete['seed'] == seed
            assert recipe['protocol_sha256'] == (protocol_hash if candidate else digest(BASE / 'data/protocol.json'))
            assert recipe['data_seal_sha256'] == protocol['data_seal_sha256']
            assert recipe['validation_packets_loaded_by_trainer'] is False and complete['validation_evaluated'] is False
            assert recipe['effective_batch_size'] == 4 and recipe['micro_batch_size'] == 2
            if candidate:
                assert recipe['source_sha256'] == protocol['source_hashes']
                assert protocol['created_before_training_utc'] < history[0]['timestamp_utc']
                assert history[-1]['timestamp_utc'] < frozen['timestamp_utc']
            for name, expected_hash in recipe['source_sha256'].items():
                digest(ROOT / name, expected_hash)
            digest(run / 'recipe.json', complete['recipe_sha256'])
            digest(run / 'best.pt', complete['checkpoint_sha256'])
            digest(run / 'latest.pt')
            best, best_epoch, patience = recompute_selection(history, protocol['training'])
            assert complete['epochs'] == len(history) and complete['best_epoch'] == best_epoch
            check_numbers(complete['best_calibration'], best, 'Completion calibration selection')
            latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
            selected = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            assert latest['history'] == history and latest['recipe'] == selected['recipe'] == recipe
            assert latest['best_epoch'] == selected['best_epoch'] == best_epoch and latest['patience'] == patience
            check_numbers(latest['best'], best, 'Latest calibration')
            check_numbers(selected['best'], best, 'Selected checkpoint calibration')
            if candidate:
                assert set(selected['model']) == set(latest['best_model'])
                for name, tensor in selected['model'].items():
                    assert torch.isfinite(tensor).all() and torch.equal(tensor, latest['best_model'][name]), name
            else:
                assert selected['history'] == history[:best_epoch]
                assert selected['history'][-1]['improved'] is True
            checkpoints.append({'method': method, 'seed': seed, 'epochs': len(history), 'best_epoch': best_epoch,
                                'best_checkpoint_sha256': complete['checkpoint_sha256'], 'passed': True})
            del latest, selected
            report = read(folder / 'report.json')
            assert report['method'] == method and report['seed'] == seed
            assert set(report['budgets']) == ({'1'} if method == 'deterministic' else {'4', '32'})
            if candidate:
                assert report['checkpoint_sha256'] == complete['checkpoint_sha256'] and report['best_epoch'] == best_epoch
                assert report['parameter_count'] == protocol['single_model_change']['parameters']
            else:
                assert report['source']['sha256'] == complete['checkpoint_sha256'] and report['source']['best_epoch'] == best_epoch
                assert report['source']['checkpoint'] == str((run / 'best.pt').relative_to(ROOT))
            required_files = {'worlds.npz'}
            for k in report['budgets']:
                required_files.update({f'predictions_k{k}.csv', f'maps_k{k}.json'})
            if candidate:
                required_files.add('conditional_mean_maps.npz')
            assert required_files <= set(report['files']), 'Report omits hashes for scored artifacts'
            for name, expected_hash in report['files'].items():
                digest(folder / name, expected_hash)
            reports[method, seed] = report
            with np.load(folder / 'worlds.npz', allow_pickle=False) as z:
                assert z['global_ids'].tolist() == [s.row['global_id'] for s in samples]
                if candidate:
                    worlds = z['worlds'].copy()
            events = []
            if candidate:
                assert worlds.dtype == bool and worlds.shape == (40, 32, *samples[0].target.shape)
                for sample, w in zip(samples, worlds):
                    assert not (w & ~sample.valid).any()
                    assert np.all(w[:, ~sample.hidden] == sample.observation[0, ~sample.hidden])
                    e = _accelerated_events(w, sample.starts, sample.goals, RADII)
                    events.append(e)
                    for index in (0, 31):
                        assert np.array_equal(direct_events(w[index], sample.starts, sample.goals), e[index])
                        counts['independent_geometry_worlds'] += 1
                    counts['worlds_checked'] += len(w)
                with np.load(folder / 'conditional_mean_maps.npz', allow_pickle=False) as z:
                    assert z['global_ids'].tolist() == [s.row['global_id'] for s in samples]
                    p = z['probabilities']
                    assert p.shape == (40, *samples[0].target.shape) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            for k, summary in report['budgets'].items():
                with (folder / f'predictions_k{k}.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                actual = {(r['global_id'], int(r['candidate_index']), int(r['radius_cells'])): float(r['probability']) for r in rows}
                assert len(actual) == len(rows) == len(expected) and set(actual) == set(expected)
                per_parent, risk_rows = defaultdict(list), []
                for row in rows:
                    assert row['parent_group'] == parent_by_id[row['global_id']]
                    key = (row['global_id'], int(row['candidate_index']), int(row['radius_cells']))
                    p, y = actual[key], expected[key]
                    assert np.isfinite(p) and 0 <= p <= 1 and p * int(k) == int(p * int(k))
                    per_parent[row['parent_group']].append((p-y)**2)
                    risk_rows.append((row['parent_group'], p, y))
                audit_parent_summary(summary, per_parent, samples)
                check_numbers(summary['risk30'], direct_risk(risk_rows), 'Fractional probability-tied equal-parent risk')
                map_rows = read(folder / f'maps_k{k}.json')
                assert len(map_rows) == 40
                if candidate:
                    for index, (sample, w, e) in enumerate(zip(samples, worlds, events)):
                        p = e[:int(k)].mean(0)
                        for q, ci in enumerate(sample.candidate_indices):
                            for ri, radius in enumerate(RADII):
                                assert p[q, ri] == actual[sample.row['global_id'], int(ci), radius]
                                counts['event_vote_checks'] += 1
                        independent = direct_map_metrics(w[:int(k)], sample)
                        for metric, value in independent.items():
                            check_numbers(map_rows[index][metric], value, f'Saved per-scene map metric {metric}')
                            counts['per_scene_map_metric_checks'] += 1
                        assert map_rows[index]['hidden_cells'] == int(sample.hidden.sum()) and map_rows[index]['samples'] == int(k)
                for metric in MAP_METRICS:
                    values = [r[metric] for r in map_rows]
                    if values[0] is None:
                        assert metric == 'masked_energy_score' and k == '1'
                        assert all(v is None for v in values) and summary['map'][metric] is None
                    else:
                        check_numbers(summary['map'][metric], np.mean(values), f'{method}/{k} map average')
                    counts['map_aggregate_checks'] += 1
                counts['method_seed_budget_reports'] += 1
                counts['csv_parent_label_checks'] += len(rows)
            print(json.dumps({'audited_method': method, 'seed': seed, 'passed': True}), flush=True)
    aggregate = audit_aggregate(analysis, reports, samples)
    for name in ('worlds_checked', 'independent_geometry_worlds', 'event_vote_checks', 'per_scene_map_metric_checks', 'independent_target_maps'):
        assert counts[name] == analysis['independent_world_checks'][name] == original_verification['checks'][name]
    digest(Path(__file__))
    digest(ROOT / 'tests/test_coherent_results_audit.py')
    receipt = {
        'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_sha256': analysis_hash, 'protocol_sha256': protocol_hash,
        'baseline_analysis_sha256': protocol['baseline_analysis_sha256'],
        'baseline_verification_sha256': protocol['baseline_verification_sha256'],
        'checks': dict(counts), 'checkpoint_checks': checkpoints, 'aggregate_checks': aggregate,
        'validation_parents': 40, 'validation_events_per_report': 1545,
        'training_seeds': list(SEEDS), 'final_test': False,
        'read_scope': {'cached_development_validation_packets_decoded': 40,
                       'sealed_development_files_byte_hashed': len(seal['files']),
                       'new_archive_images_read': 0, 'model_inference_runs': 0, 'gpu_work': False},
        'independence_scope': 'Separate CPU audit execution. Reuses the frozen independent explicit-disk erosion/four-neighbor, off-diagonal map-score, equal-parent tied-risk helpers; does not invoke either evaluator or its aggregation/bootstrap helpers. All candidate saved worlds and votes are freshly checked. Baseline worlds are file-hash checked and their earlier independent geometry proof is reused; all six matched baseline reports/CSVs/map averages and training selections are freshly checked.',
        'limitations': ['Development validation was reused to choose the model change; this is not a final-test result.',
                        'Bootstrap intervals condition on the same two training-seed averages and do not estimate full training-seed uncertainty.',
                        'The original-ConPath paired Brier interval crosses zero; passing the development screen does not establish stable superiority.',
                        'The two zero-query training parents share the documented microbatch event-weighting limitation with the original baseline.'],
        'input_file_sha256': files,
    }
    target = OUT / 'independent_verification.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({'passed': True, 'checks': dict(counts), 'aggregate_checks': aggregate}), flush=True)


if __name__ == '__main__':
    main()
