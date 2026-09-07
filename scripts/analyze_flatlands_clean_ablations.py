#!/usr/bin/env python3
"""Build paired validation evidence after all six clean training ablations pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.flatlands_eval import join_flatlands_predictions, _metric_summary
from pathrel.flatlands_query import sha256_path
from pathrel.selective_risk import risk_at_coverage
from scripts.analyze_flatlands_clean_validation import arrays, paired_risk, summary
from scripts.compare_flatlands_k128_paired import _paired_stratum
from scripts.evaluate_flatlands_support_clamped import _atomic_json
from scripts.run_flatlands_clean_ablations import SEEDS, audit_run, prepare

LABELS = {'conpath': 'Full ConPath', 'no_event': 'Without event training loss',
          'no_global': 'Without global decoder factors'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.matrix_root.resolve().relative_to(ROOT)
    matrix = prepare(root)
    contract = matrix['contract']
    audits = [audit_run(entry, contract['training_source_sha256']) for entry in contract['runs']]
    methods = {}; records_by_method = {}; canonical = None
    for variant in LABELS:
        runs = []; records_by_method[variant] = {}
        for seed in SEEDS:
            run_dir = (Path('results/p1_flatlands_conpath_k128_support_clamped_v1') / f'seed{seed}_conpath'
                       if variant == 'conpath' else root / f'seed{seed}_{variant}')
            prediction = run_dir / 'predictions_validation.csv'
            config = json.loads((run_dir / 'run.json').read_text())['config']
            records, radii = join_flatlands_predictions(prediction, Path(config['selection']),
                                                       Path(config['queries']), split='validation')
            records = tuple(sorted(records, key=lambda r: r.key))
            keys = tuple((r.key, r.scene_key, r.target) for r in records)
            if canonical is None:
                canonical = keys
            if keys != canonical or len(records) != 4224:
                raise ValueError(f'event keys/labels differ from reference: {prediction}')
            records_by_method[variant][seed] = records
            p, y, w = arrays(records)
            metrics = _metric_summary(records, weighting='scene', bins=10)
            strata = {}
            for radius in radii:
                strata[f'radius/{radius}'] = _metric_summary(tuple(r for r in records if r.radius_cells == radius), weighting='scene', bins=10)
            for source in sorted({r.source_dataset for r in records}):
                strata[f'source/{source}'] = _metric_summary(tuple(r for r in records if r.source_dataset == source), weighting='scene', bins=10)
            runs.append({'seed': seed, 'prediction': str(prediction), 'prediction_sha256': sha256_path(prediction),
                         'run': str(run_dir / 'run.json'), 'run_sha256': sha256_path(run_dir / 'run.json'),
                         'metrics': metrics, 'strata': strata,
                         'risk_at_30_percent': risk_at_coverage(p, y, w, .3)})
        methods[variant] = {'label': LABELS[variant], 'seeds': runs,
            'aggregate': {key: summary([run['metrics'][key] for run in runs]) for key in ['brier', 'nll', 'ece']},
            'risk_at_30_percent': summary([run['risk_at_30_percent']['false_safe_rate'] for run in runs])}
    paired = {}
    for variant in ('no_event', 'no_global'):
        runs = []
        for seed in SEEDS:
            left = records_by_method['conpath'][seed]; right = records_by_method[variant][seed]
            delta = _paired_stratum(left, right, bootstrap_samples=2000, bootstrap_seed=20260907)
            risks = paired_risk(left, right, 2000, 20260907)
            runs.append({'seed': seed, 'event_metrics': delta['delta'], 'equal_coverage_risk': risks})
        paired[variant] = {'seeds': runs,
            'brier_delta': summary([r['event_metrics']['brier']['independent_minus_correlated'] for r in runs]),
            'all_seed_brier_intervals_positive': all(r['event_metrics']['brier']['bootstrap_95'][0] > 0 for r in runs)}
    output = root / 'analysis'; output.mkdir(exist_ok=True)
    report = {'schema_version': 1, 'kind': 'flatlands_clean_training_ablation_analysis',
              'validation_only': True, 'test_evaluated': False, 'paper_result': False,
              'matrix_sha256': sha256_path(root / 'matrix.json'), 'script_sha256': sha256_path(Path(__file__)),
              'protocol': {'events_per_seed': 4224, 'contributing_scenes': 142, 'seeds': SEEDS,
                           'weighting': 'equal scene, then equal event within scene',
                           'bootstrap_samples': 2000, 'bootstrap_seed': 20260907,
                           'delta_direction': 'ablation minus full model; positive Brier means removing the factor hurts',
                           'no_event_scope': 'event loss weight zero; fixed validation event-based checkpoint selection is retained',
                           'no_global_scope': 'decoder global factors disabled; encoder global context and local correlation are retained'},
              'audits': audits, 'methods': methods, 'paired': paired,
              'claim_boundary': 'All seeds are reported on reused validation; intervals are descriptive within-seed scene bootstraps, not multiplicity-adjusted confirmation or final test claims.'}
    _atomic_json(output / 'report.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.hashsalt': 'conpath-clean-training-ablations-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1, 1.15]})
    names = list(LABELS); colors = ['#17866c', '#c87329', '#7757a6']
    metrics = [methods[name]['aggregate']['brier'] for name in names]
    axes[0].bar(['Full model', 'No event loss', 'No global factors'], [m['mean'] for m in metrics],
                yerr=[m['sample_sd'] for m in metrics], capsize=4, color=colors)
    axes[0].set(ylabel='Scene-weighted event Brier', title='Matched training; mean ± seed SD')
    labels = []; index = 0
    for variant, color in zip(('no_event', 'no_global'), colors[1:]):
        for record in paired[variant]['seeds']:
            delta = record['event_metrics']['brier']
            value = delta['independent_minus_correlated']; low, high = delta['bootstrap_95']
            axes[1].plot([low, high], [index, index], color=color)
            axes[1].plot(value, index, 'o', color=color)
            labels.append(f"{variant} · {record['seed']}"); index += 1
    axes[1].axvline(0, color='#888888', linestyle='--')
    axes[1].set(yticks=range(len(labels)), yticklabels=labels,
                xlabel='Ablation minus full-model Brier', title='Paired 95% whole-scene intervals')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.5, .015, '4,224 identical validation events per seed; K=128; both physical test sets remain locked.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    figures = output / 'figures'; figures.mkdir(exist_ok=True)
    for suffix in ('svg', 'pdf'):
        path = figures / f'flatlands_clean_training_ablations.{suffix}'
        fig.savefig(path, metadata={'Date': None} if suffix == 'svg' else {'CreationDate': None, 'ModDate': None})
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    def fmt(item):
        return f"{item['mean']:.5f} ± {item['sample_sd']:.5f}"
    lines = ['# Clean FlatLands training ablations', '',
             'Audited candidate evidence; validation only. Generated after all six runs complete.', '',
             '| Model | Event Brier ↓ | NLL ↓ | ECE ↓ | False-safe at 30% coverage ↓ |',
             '|---|---:|---:|---:|---:|']
    for method in methods.values():
        lines.append('| ' + method['label'] + ' | ' + ' | '.join([fmt(method['aggregate'][k]) for k in ('brier', 'nll', 'ece')] + [fmt(method['risk_at_30_percent'])]) + ' |')
    lines += ['', 'Removing event loss retains the same validation event-based checkpoint selection. Removing global factors retains encoder context and local decoder correlation. Only the named training flag differs from each full-model reference.', '',
              '| Ablation | Mean Brier delta (ablation − full) | Per-seed 95% scene intervals |', '|---|---:|---|']
    for variant, comparison in paired.items():
        intervals = '; '.join('[' + ', '.join(f'{v:.5f}' for v in r['event_metrics']['brier']['bootstrap_95']) + ']' for r in comparison['seeds'])
        lines.append(f"| {LABELS[variant]} | {fmt(comparison['brier_delta'])} | {intervals} |")
    lines += ['', 'All seeds, reversals and null intervals are retained. These validation comparisons do not establish final test performance or successful transfer.', '',
              '![Clean training ablations](figures/flatlands_clean_training_ablations.svg)', '']
    (output / 'PAPER_ABLATIONS.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(output / 'report.json'), 'brier': {k: v['aggregate']['brier'] for k, v in methods.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
