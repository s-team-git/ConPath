#!/usr/bin/env python3
"""Add K4 and paper-defined map metrics to the existing bounded validation cohort.

No training or external-model reproduction. K4 uses the first four actual worlds
of the frozen K128 replay; the remaining draws verify canonical event CSVs and
advance the original RNG. This execution time is NOT K4 deployment latency.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.benchmark_metrics import masked_map_metrics
from pathrel.flatlands_baselines import MarginalCompletionBaseline
from pathrel.flatlands_data import FlatLandsReplayDataset
from pathrel.flatlands_eval import load_prediction_manifest, write_prediction_manifest, join_flatlands_predictions, _metric_summary
from pathrel.flatlands_query import sha256_path
from scripts.evaluate_flatlands_clean_controls import rows_for, SEEDS, SELECTION, QUERIES
from scripts.evaluate_flatlands_support_clamped import _load_model, _accelerated_events, _verify_accelerator, _atomic_json
from scripts.compare_flatlands_k128_paired import _validate_source_run
from scripts.analyze_flatlands_clean_validation import arrays, risk_at_coverage

OUT = Path('results/current_baseline_k4_v1')


def map_summary(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row['source'], row['scene'])].append(row)
    keys = [k for k in rows[0] if k not in {'source', 'scene', 'global_id', 'hidden_cells', 'samples'}]
    result = {}
    for key in keys:
        values = [float(np.mean([r[key] for r in group if r[key] is not None]))
                  for group in groups.values() if any(r[key] is not None for r in group)]
        result[key] = float(np.mean(values)) if values else None
    return {**result, 'observation_count': len(rows), 'scene_count': len(groups)}


def save_run(method, seed, samples, worlds, source, canonical=None, replay_rows=None):
    path = OUT / method / str(seed); path.mkdir(parents=True, exist_ok=False)
    prediction_rows, metrics = [], []
    for sample, w in zip(samples, worlds):
        if np.any(w & ~sample.epistemic_mask[None]): raise ValueError('Invalid support opened')
        known = ~sample.unknown
        if np.any(w[:, known] != sample.observed_free[known]): raise ValueError('Known evidence changed')
        metrics.append({'global_id': sample.observation.global_id, 'source': sample.observation.source_dataset,
                        'scene': sample.observation.scene_id, **masked_map_metrics(w, sample.target_free, sample.loss_mask)})
        if sample.retained_queries:
            starts = np.array([(q.start_row, q.start_col) for q in sample.retained_queries])
            goals = np.array([(q.goal_row, q.goal_col) for q in sample.retained_queries])
            events = _accelerated_events(w, starts, goals, sample.radii_cells).mean(0)
            prediction_rows.extend(rows_for(sample, events))
    write_prediction_manifest(path/'predictions.csv', prediction_rows)
    records, _ = join_flatlands_predictions(path/'predictions.csv', SELECTION, QUERIES, split='validation')
    assert len(records) == 4224
    p, y, weights = arrays(records)
    event_metrics = _metric_summary(records, weighting='scene', bins=10)
    risk = {str(c): risk_at_coverage(p, y, weights, c) for c in (.1, .3, .5)}
    np.savez_compressed(path/'worlds.npz', worlds=np.stack(worlds), global_ids=np.array([s.observation.global_id for s in samples]))
    _atomic_json(path/'map_metrics.json', metrics)
    drift = None
    if canonical is not None:
        old = load_prediction_manifest(canonical)
        replay = {(r['global_id'], r['candidate_index'], r['radius_cells']): r['probability'] for r in (replay_rows if replay_rows is not None else prediction_rows)}
        assert old.keys() == replay.keys()
        drift = max(abs(old[k]-replay[k]) for k in old)
        if drift != 0: raise ValueError(f'Canonical replay changed: {method}/{seed}, {drift}')
    report = {'method': method, 'seed': seed, 'samples': int(worlds[0].shape[0]), 'source': source,
              'validation_only': True, 'test_evaluated': False, 'trained_this_run': False,
              'canonical_event_replay_max_drift': drift, 'event_metrics': event_metrics,
              'equal_coverage': risk, 'map_metrics': map_summary(metrics),
              'map_by_source': {s: map_summary([r for r in metrics if r['source'] == s]) for s in sorted({r['source'] for r in metrics})},
              'outputs': {f.name: {'sha256': sha256_path(f), 'bytes': f.stat().st_size} for f in path.iterdir() if f.is_file()}}
    _atomic_json(path/'report.json', report)
    print(json.dumps({'completed': method, 'seed': seed, 'event_brier': event_metrics['brier'], 'maps': report['map_metrics']}, ensure_ascii=False), flush=True)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    _verify_accelerator()
    started = time.monotonic()
    dataset = FlatLandsReplayDataset(Path('data/raw/flatlands/FlatLands_final_dataset.zip'), SELECTION, QUERIES,
                                    split='validation', verify_frozen=True, verify_query_geometry=True)
    try: samples = [dataset[i] for i in range(len(dataset))]
    finally: dataset.close()
    assert len(samples) == 160
    reports = []
    for method in ('correlated', 'independent'):
        for seed in SEEDS:
            old_root = Path('results') / ('p1_flatlands_conpath_k128_support_clamped_v1' if method == 'correlated' else 'p1_flatlands_independent_k128_support_clamped_v1')
            source = old_root/f"seed{seed}_{'conpath' if method == 'correlated' else 'independent'}"
            _validate_source_run(ROOT/source/'run.json', method, 'valid-support-clean-training')
            frozen = json.loads(Path(f'results/paper_clean_checkpoint_controls_v1/{method}/seed{seed}/run.json').read_text())
            for name, field in [('best.pt', 'checkpoint'), ('latest.pt', 'latest_rng_checkpoint')]:
                assert sha256_path(source/name) == frozen[field]['sha256']
            model, config, _ = _load_model(source/'best.pt', method, torch.device('cuda'))
            latest = torch.load(source/'latest.pt', map_location='cpu', weights_only=False)
            generator = torch.Generator(device='cuda'); generator.set_state(latest['sample_generator_state'].cpu()); del latest
            assert config['validation_sample_chunk'] == 8 and config['batch_size'] == 1
            selected_worlds, replay_rows = [], []
            with torch.inference_mode():
                for i, sample in enumerate(samples):
                    has_queries = bool(sample.retained_queries)
                    # Query-less observations did not consume the canonical RNG. Include
                    # them in map metrics using a separate fixed, identity-keyed stream.
                    aux_seed = int(hashlib.sha256(f'k4-map-only/{seed}/{sample.observation.global_id}'.encode()).hexdigest()[:15], 16)
                    gen = generator if has_queries else torch.Generator(device='cuda').manual_seed(aux_seed)
                    obs = torch.from_numpy(sample.input_bev[None]).cuda()
                    support = torch.from_numpy(sample.epistemic_mask[None]).cuda()
                    event_sum = None
                    for chunk in range(16 if has_queries else 1):
                        posterior = model(obs, valid_support_mask=support, num_samples=8, disable_global_factors=method=='independent', generator=gen).posterior
                        w = posterior.safe_samples()[0].cpu().numpy() > .5
                        if chunk == 0: selected_worlds.append(w[:4].copy())
                        if has_queries:
                            starts = np.array([(q.start_row, q.start_col) for q in sample.retained_queries])
                            goals = np.array([(q.goal_row, q.goal_col) for q in sample.retained_queries])
                            e = _accelerated_events(w, starts, goals, sample.radii_cells).sum(0)
                            event_sum = e if event_sum is None else event_sum + e
                    if has_queries: replay_rows.extend(rows_for(sample, event_sum/128))
                    if (i+1)%40 == 0: print(f'{method}/{seed}: {i+1}/160', flush=True)
            reports.append(save_run(method, seed, samples, selected_worlds, frozen['checkpoint'], source/'predictions_validation.csv', replay_rows))
            del model; torch.cuda.empty_cache()
    for seed in SEEDS:
        source = Path(f'results/p1_flatlands_completion_seed{seed}')
        old = json.loads((source/'run.json').read_text()); record = old['artifacts']['best_checkpoint']
        assert sha256_path(source/'best.pt') == record['sha256']
        model = MarginalCompletionBaseline(feature_channels=old['config']['feature_channels']).cuda()
        model.load_state_dict(torch.load(source/'best.pt', map_location='cpu', weights_only=False)['model']); model.eval()
        with torch.inference_mode():
            w = [(model.free_probability(torch.from_numpy(s.input_bev[None]).cuda())[0].cpu().numpy() >= .5)[None] & s.epistemic_mask[None] for s in samples]
        reports.append(save_run('tiny_deterministic', seed, samples, w, record, source/'predictions_deterministic_validation.csv'))
        del model
    for method in ('all_floor', 'all_blocked', 'nearest_observed'):
        worlds = []
        for s in samples:
            if method == 'nearest_observed':
                observed = s.epistemic_mask & ~s.unknown
                if not observed.any(): raise ValueError('Nearest observed baseline needs an observed cell')
                indices = distance_transform_edt(~observed, return_distances=False, return_indices=True)
                hidden = s.observed_free[tuple(indices)]
            else: hidden = np.full(s.target_free.shape, method == 'all_floor', dtype=bool)
            worlds.append(((hidden & s.unknown) | s.observed_free)[None] & s.epistemic_mask[None])
        reports.append(save_run(method, 'rule', samples, worlds, {'kind': 'parameter-free rule, no learned paper model'}))
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'validation_only': True, 'test_evaluated': False,
              'formal_external_comparison': False, 'training_performed': False, 'runs': reports,
              'map_observations': 160, 'event_scenes': 142, 'events_per_run': 4224,
              'protocol': {'selection_sha256': sha256_path(SELECTION), 'queries_sha256': sha256_path(QUERIES),
                           'k4': 'First4 actual worlds of the frozen128 replay, with full128 exact event check; no oracle selection.',
                           'map_only_cases': '18 query-less cases use a separate identity-keyed RNG, retaining all160 map observations.',
                           'aggregation': 'Per-observation masked metrics, then equal scenes; repeat SD kept separate.',
                           'empty_mask': 'Excluded from map metric means and counted; empty foreground union IoU=1; K1 stochastic MES=None.',
                           'timing_scope': 'Entire audit includes extra128-world replay and disk writes, NOT K4 deployment latency.'},
              'source_hashes': {str(p): sha256_path(p) for p in [Path(__file__), Path('src/pathrel/benchmark_metrics.py'), Path('src/pathrel/model.py'), Path('src/pathrel/flatlands_data.py'), SELECTION, QUERIES]},
              'runtime_seconds': time.monotonic()-started}
    _atomic_json(OUT/'report.json', report)


if __name__ == '__main__': main()
