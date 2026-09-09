#!/usr/bin/env python3
"""Score the fixed two-seed sampler follow-up and independently check its worlds."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.benchmark_metrics import masked_map_metrics
from pathrel.parent_pilot_data import load_pilot, sha
from scripts.audit_parent_pilot_results import direct_events, direct_map_metrics, direct_risk
from scripts.evaluate_flatlands_support_clamped import _accelerated_events
from scripts.evaluate_parent_group_pilot import event_summary, paired_interval
from scripts.train_coherent_parent_pilot import model_for
from scripts.train_parent_group_pilot import predict, RADII
from scripts.train_flatlands_conpath import _atomic_json

BASE = ROOT / 'results/parent_group_pilot_v1'
OUT = ROOT / 'results/coherent_parent_pilot_v1'
SEEDS = (20260910, 20260911)


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['seeds'] == list(SEEDS)
    assert protocol['baseline_analysis_sha256'] == sha(BASE / 'analysis.json')
    assert protocol['data_seal_sha256'] == sha(BASE / 'data/seal.json')
    for name, checksum in protocol['source_hashes'].items():
        assert sha(ROOT / name) == checksum, name
    for seed in SEEDS:
        folder = OUT / 'runs' / str(seed)
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete['checkpoint_sha256'] == sha(folder / 'best.pt')
        assert complete['recipe_sha256'] == sha(folder / 'recipe.json')
        assert json.loads((folder / 'recipe.json').read_text())['protocol_sha256'] == sha(OUT / 'protocol.json')
    evaluation = OUT / 'evaluation'
    if evaluation.exists():
        raise ValueError('Evaluation already exists; inspect any partial output instead of overwriting')
    evaluation.mkdir()
    _atomic_json(evaluation / 'frozen_before_scoring.json', {'protocol_sha256': sha(OUT / 'protocol.json'),
                 'script_sha256': sha(Path(__file__)), 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
    samples = load_pilot(BASE / 'data', 'validation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    reports, comparisons = [], {}
    audit = {'worlds_checked': 0, 'independent_geometry_worlds': 0, 'event_vote_checks': 0,
             'per_scene_map_metric_checks': 0, 'independent_target_maps': 0}
    for sample in samples:
        assert np.array_equal(direct_events(sample.target, sample.starts, sample.goals), sample.targets)
        audit['independent_target_maps'] += 1
    for seed in SEEDS:
        folder = evaluation / str(seed); folder.mkdir()
        state = torch.load(OUT / 'runs' / str(seed) / 'best.pt', map_location='cpu', weights_only=False)
        model = model_for(); model.load_state_dict(state['model']); model.eval()
        worlds, events, probabilities, timings = [], [], [], []
        for sample in samples:
            torch.cuda.synchronize(); start = time.perf_counter()
            w, e, p, _ = predict(model, 'correlated', sample, seed + 4000000, 32)
            torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
            worlds.append(w); events.append(e); probabilities.append(p)
        np.savez_compressed(folder / 'worlds.npz', global_ids=np.array([s.row['global_id'] for s in samples]), worlds=np.stack(worlds))
        np.savez_compressed(folder / 'conditional_mean_maps.npz', global_ids=np.array([s.row['global_id'] for s in samples]), probabilities=np.stack(probabilities))
        report = {'method': 'coherent_categorical', 'seed': seed, 'best_epoch': state['best_epoch'],
                  'checkpoint_sha256': sha(OUT / 'runs' / str(seed) / 'best.pt'), 'budgets': {},
                  'predict_including_exact_events_seconds': timings,
                  'timing_is_shared_gpu_no_warmup_diagnostic': True,
                  'parameter_count': sum(p.numel() for p in model.parameters())}
        del model, state
        # Reopen saved worlds so audits verify the persisted artifact, not only RAM.
        with np.load(folder / 'worlds.npz', allow_pickle=False) as z:
            saved = z['worlds'].copy()
            assert z['global_ids'].tolist() == [s.row['global_id'] for s in samples]
        assert saved.shape == (40, 32, *samples[0].target.shape) and saved.dtype == bool
        for sample, w, e in zip(samples, saved, events):
            assert not (w & ~sample.valid).any()
            assert np.all(w[:, ~sample.hidden] == (sample.observation[0] > .5)[~sample.hidden])
            assert np.array_equal(_accelerated_events(w, sample.starts, sample.goals, RADII), e)
            for index in (0, 31):
                assert np.array_equal(direct_events(w[index], sample.starts, sample.goals), e[index])
                audit['independent_geometry_worlds'] += 1
            audit['worlds_checked'] += len(w)
        for k in (4, 32):
            records, rows, maps, risk_rows, parent_scores = [], [], [], [], []
            for sample, w, e in zip(samples, saved, events):
                p = e[:k].mean(0)
                records.append({'probabilities': p.tolist()})
                parent_scores.append(float(np.mean((p-sample.targets)**2)))
                metric = masked_map_metrics(w[:k], sample.target, sample.hidden)
                independent = direct_map_metrics(w[:k], sample)
                for name, value in independent.items():
                    assert abs(value-metric[name]) < 1e-12
                    audit['per_scene_map_metric_checks'] += 1
                maps.append(metric)
                for q, candidate in enumerate(sample.candidate_indices):
                    for r, radius in enumerate(RADII):
                        rows.append({'global_id': sample.row['global_id'], 'parent_group': sample.row['parent_group'],
                                     'candidate_index': int(candidate), 'radius_cells': radius, 'probability': float(p[q, r])})
                        risk_rows.append((sample.row['parent_group'], float(p[q, r]), bool(sample.targets[q, r])))
                        audit['event_vote_checks'] += 1
            summary = event_summary(records, samples)
            assert np.allclose(summary['parent_brier'], parent_scores, rtol=0, atol=1e-12)
            assert abs(summary['brier'] - np.mean(parent_scores)) < 1e-12
            assert abs(summary['risk30'] - direct_risk(risk_rows)) < 1e-12
            summary['map'] = {name: float(np.mean([m[name] for m in maps])) for name in independent}
            report['budgets'][str(k)] = summary
            with (folder / f'predictions_k{k}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            _atomic_json(folder / f'maps_k{k}.json', maps)
        report['files'] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        _atomic_json(folder / 'report.json', report)
        reports.append(report)
        print(json.dumps({'evaluated_seed': seed, 'brier': report['budgets']['32']['brier'],
                          'risk30': report['budgets']['32']['risk30']}), flush=True)
    methods = {'coherent_categorical': reports}
    for method in ('correlated', 'independent', 'deterministic'):
        methods[method] = [json.loads((BASE / 'evaluation' / method / str(seed) / 'report.json').read_text()) for seed in SEEDS]
    combined, parent_means = {}, {}
    for method, group in methods.items():
        budget = '1' if method == 'deterministic' else '32'
        values = [r['budgets'][budget] for r in group]
        combined[method] = {'seeds': list(SEEDS), 'samples': int(budget)}
        for metric in ('brier', 'risk30'):
            v = [r[metric] for r in values]
            combined[method][metric] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'values': v}
        combined[method]['map'] = {name: float(np.mean([r['map'][name] for r in values])) if values[0]['map'][name] is not None else None for name in values[0]['map']}
        parent_means[method] = np.mean([r['parent_brier'] for r in values], axis=0)
    rng = np.random.default_rng(20260909)
    for method in ('correlated', 'independent', 'deterministic'):
        comparisons[method] = paired_interval(parent_means['coherent_categorical'], parent_means[method],
                                              [s.row['source_dataset'] for s in samples], rng)
        comparisons[method]['unit'] = 'parent place, source stratified; conditional on the displayed two-seed average'
    candidate, baseline = combined['coherent_categorical'], combined['correlated']
    conditions = {'brier_better_than_matched_conpath': candidate['brier']['mean'] < baseline['brier']['mean'],
                  'risk30_not_worse_than_matched_conpath': candidate['risk30']['mean'] <= baseline['risk30']['mean']}
    analysis = {'id': 'coherent_parent_pilot_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'methods': combined, 'paired_brier': comparisons, 'screen_conditions': conditions,
                'development_screen_passed': all(conditions.values()), 'independent_world_checks': audit,
                'baseline_analysis_sha256': sha(BASE / 'analysis.json'), 'protocol_sha256': sha(OUT / 'protocol.json'),
                'validation_parents': 40, 'validation_events': 1545, 'training_seeds': list(SEEDS),
                'validation_reused_for_model_development': True, 'final_test': False,
                'new_physical_test_reads': 0, 'no_direct_published_paper_ranking': True,
                'scope': 'Exactly two candidate seeds compared with the same two baseline seeds; not the earlier three-seed aggregate. One fixed 9x9 sigma-3 sampler change, no hyperparameter search. Development feedback reused; uncertainty is conditional on two training seeds.'}
    _atomic_json(OUT / 'analysis.json', analysis)
    _atomic_json(OUT / 'verification.json', {'passed': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
                 'checks': audit, 'source_hashes': protocol['source_hashes']})
    print(json.dumps({'candidate_complete': True, 'conditions': conditions, 'development_screen_passed': all(conditions.values())}), flush=True)


if __name__ == '__main__':
    main()
