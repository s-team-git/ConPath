#!/usr/bin/env python3
"""Build manuscript tables and standalone figures from the clean validation reports."""

from __future__ import annotations

import json
import html
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mean_sd(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": list(values)}


def fmt(value):
    if value["sample_sd"] is None:
        return f"{value['mean']:.5f}"
    return f"{value['mean']:.5f} ± {value['sample_sd']:.5f}"


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'conpath-paper-evidence-v1'})
    analysis=json.loads((ROOT/'results/paper_clean_analysis_v1/report.json').read_text())
    bounds=json.loads((ROOT/'site/data/unscenes3d_observation_ceiling.json').read_text())
    second_domain=json.loads((ROOT/'site/data/unscenes3d_clean_support_k128_candidate.json').read_text())
    if not bounds['prediction_bound_audit']['passed'] or not second_domain.get('mean_map_projection_version','').startswith('v2:'):
        raise ValueError('support-projected v2 predictions and their event-bound audit are required')
    second_brier=second_domain['aggregate']['correlated']['scene_weighted_brier']['mean']
    bound_fraction=bounds['splits']['validation']['overall']['scene_weighted_brier_lower_bound']/second_brier
    controls={}
    for variant in ['correlated','independent']:
        runs=[json.loads(p.read_text()) for p in sorted((ROOT/'results/paper_clean_checkpoint_controls_v1'/variant).glob('seed*/run.json'))]
        if len(runs)!=3 or any(r['canonical_k128_max_drift'] != 0 for r in runs):
            raise ValueError('six exact clean replay reports are required')
        controls[variant]={
            'event_brier':{k:mean_sd([r['event_metrics'][k]['brier'] for r in runs]) for k in ['32','64','128','mean_map']},
            'hidden_map_brier':mean_sd([r['hidden_map']['brier'] for r in runs]),
            'runtime_seconds':mean_sd([r['runtime']['total_seconds'] for r in runs]),
            'sampling_seconds':mean_sd([r['runtime']['sampling_seconds'] for r in runs]),
            'connectivity_seconds':mean_sd([r['runtime']['connectivity_seconds'] for r in runs]),
            'peak_gpu_bytes':mean_sd([r['runtime']['peak_gpu_bytes'] for r in runs]),
            'reports':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'results/paper_clean_checkpoint_controls_v1'/variant).glob('seed*/run.json'))],
        }
    payload={'kind':'flatlands_clean_nested_k_controls','validation_only':True,'paper_result':False,'test_evaluated':False,
             'canonical_k128_replay_max_drift':0,'controls':controls,
             'claim_boundary':'Fixed checkpoints and saved RNG, nested K prefixes. Timings describe this accelerated validation diagnostic only, not end-to-end robot latency or a new CUDA training operator.'}
    (ROOT/'site/data/flatlands_clean_checkpoint_controls.json').write_text(json.dumps(payload,indent=2)+'\n')
    assets=ROOT/'site/assets'
    def save(fig,name):
        for suffix in ['svg','pdf']:
            meta={'Date':None} if suffix=='svg' else {'CreationDate':None,'ModDate':None}
            fig.savefig(assets/f'{name}.{suffix}',metadata=meta)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for variant,label,color in [('correlated','ConPath','#17866c'),('independent','Independent decoder','#c87329')]:
        series=controls[variant]['event_brier']
        means=[series[str(k)]['mean'] for k in [32,64,128]]
        ax.errorbar([32,64,128],means,yerr=[series[str(k)]['sample_sd'] for k in [32,64,128]],fmt='o-',capsize=4,label=label,color=color)
    ax.set(xlabel='Posterior sample count K (nested prefixes)',ylabel='Scene-weighted event Brier',title='Clean checkpoints: posterior sample-budget sensitivity',xticks=[32,64,128])
    ax.legend();ax.grid(alpha=.2)
    fig.text(.5,.02,'Mean ± seed SD; 4,224 events per seed. Every K=128 probability exactly replays the original CSV. Test locked.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'flatlands_clean_k_convergence')
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    names=['conpath','marginal_shuffle']
    values=[analysis['methods'][m]['aggregate']['brier'] for m in names]
    bars=ax.bar(['Original joint worlds','Cellwise shuffled worlds'],[r['mean'] for r in values],yerr=[r['sample_sd'] for r in values],capsize=5,color=['#17866c','#c87329'],width=.55)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value['mean']+value['sample_sd']+.008,f"{value['mean']:.5f}",ha='center',weight='bold')
    ax.set(ylabel='Scene-weighted event Brier',title='Same empirical cell marginals, different spatial dependence',ylim=(0,max(v['mean']+v['sample_sd'] for v in values)*1.30))
    ax.grid(axis='y',alpha=.2)
    fig.text(.5,.02,'Every per-cell free count is preserved exactly. Labels do not select the permutations. Three seeds; test locked.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'flatlands_clean_marginal_shuffle')
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    names=['train','validation']; b=[bounds['splits'][s]['overall'] for s in names]
    fn=[r['scene_weighted_unavoidable_false_negative'] for r in b]
    fp=[r['scene_weighted_unavoidable_false_positive'] for r in b]
    ax.bar(['Train · 9 scenes','Validation · 2 scenes'],fn,label='Forced false negative',color='#c87329',width=.5)
    ax.bar(['Train · 9 scenes','Validation · 2 scenes'],fp,bottom=fn,label='Forced false positive',color='#9b474d',width=.5)
    for i,r in enumerate(b):
        ax.text(i,fn[i]+fp[i]+.012,f"Brier ≥ {r['scene_weighted_brier_lower_bound']:.5f}",ha='center',weight='bold')
    ax.set(ylabel='Unavoidable squared event error',title='UnScenes3D: hard observations constrain attainable accuracy',ylim=(0,.65))
    ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2)
    fig.text(.5,.02,'Exact optimistic/pessimistic path bounds under the frozen adapter. This is not a universal model lower bound.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'unscenes3d_observation_ceiling')
    lines=['# Clean validation evidence for the working paper','',
           'Generated by `scripts/build_paper_evidence.py` from the frozen reports. All model results are validation-only; physical tests remain locked. Values are mean ± sample SD across training seeds unless only one run is available.','',
           '## Same-query FlatLands table','',
           'Each method covers the same 4,224 events from 142 contributing validation scenes. Metrics give equal weight to each scene and then each event within that scene.','',
           '| Method | Seeds | Brier ↓ | NLL ↓ | ECE ↓ | False-safe at 30% coverage ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    for method,m in analysis['methods'].items():
        lines.append(f"| {m['label']} | {m['seed_count']} | {fmt(m['aggregate']['brier'])} | {fmt(m['aggregate']['nll'])} | {fmt(m['aggregate']['ece'])} | {fmt(m['equal_coverage']['0.3'])} |")
    lines += ['', 'The radius prior is one train-fitted control, not an optimization-seed experiment. Direct query currently has one training seed. Binary mean-map/completion controls require fractional acceptance of boundary ties to describe exact coverage; ties never use labels.','',
              '## Paired scene evidence','',
              'Positive deltas mean the comparator has higher Brier than ConPath. Intervals resample whole scenes 2,000 times per seed (bootstrap seed 20260906). These are exploratory validation intervals, not adjusted multi-comparison significance claims.','',
              '| Comparator | Mean Brier delta | Per-seed 95% intervals |','|---|---:|---|']
    for method,p in analysis['paired'].items():
        cis='; '.join('['+', '.join(f'{x:.5f}' for x in row['delta']['brier']['bootstrap_95'])+']' for row in p['seeds'])
        lines.append(f"| {analysis['methods'][method]['label']} | {fmt(p['brier_delta'])} | {cis} |")
    lines += ['', 'At equal 30% coverage, every ConPath-versus-independent per-seed risk interval includes zero. Lower fixed-threshold false-safe rate is therefore insufficient for a robust equal-coverage safety claim. The same-checkpoint deterministic mean-map comparison is materially stronger than the independent-sampling baseline and must remain in the main table.','',
              '![Risk at equal coverage](site/assets/flatlands_clean_equal_coverage.svg)','',
              '![Clean reliability](site/assets/flatlands_clean_reliability.svg)','',
              '## Sample count and marginal controls','',
              '| Decoder | K=32 Brier | K=64 Brier | K=128 Brier | Deterministic mean-map Brier | Hidden-map Brier |','|---|---:|---:|---:|---:|---:|']
    for variant,r in controls.items():
        lines.append('| '+variant+' | '+' | '.join([fmt(r['event_brier'][k]) for k in ['32','64','128','mean_map']]+[fmt(r['hidden_map_brier'])])+' |')
    lines += ['', 'K=32/64 are nested prefixes of the original K=128 worlds, using the saved final CUDA generator state. Every original K=128 event probability reproduced exactly across all six checkpoints. This checks the evaluated budgets, not arbitrary-K asymptotic convergence.','',
              'The shuffle control independently permutes world indices at each cell. It preserves every empirical cell free count exactly and keeps observed/invalid cells fixed. It isolates sample dependence at evaluation time; it does not replace clean no-event/no-global retraining.','',
              '![Nested sample budgets](site/assets/flatlands_clean_k_convergence.svg)','',
              '![Fixed-marginal dependence control](site/assets/flatlands_clean_marginal_shuffle.svg)','',
              '## UnScenes3D observation-conditioned bound','',
              '| Split | Frames | Scenes | Event rows | Brier lower bound | Events mutable by unknown completion |','|---|---:|---:|---:|---:|---:|']
    for split,frames in [('train',478),('validation',62)]:
        r=bounds['splits'][split]['overall']
        lines.append(f"| {split} | {frames} | {r['scene_count']} | {r['events']} | {r['scene_weighted_brier_lower_bound']:.5f} | {r['scene_weighted_mutable_event_fraction']:.2%} |")
    lines += ['', 'Under the frozen adapter, every sampled free set lies between the observed-free-only world and the world with every unknown valid cell free. Disk erosion and connectivity are monotone under free-set inclusion. A positive target unreachable even in the optimistic world, or a negative target reachable even in the pessimistic world, forces squared error one for every posterior honoring those observations. This proves an adapter-conditioned lower bound; it does not imply that better sensor models or softer observation likelihoods cannot help.','',
              f'The validation lower bound 0.47574 accounts for about {bound_fraction:.0%} of the support-projected v2 mean-map Brier {second_brier:.5f}. Only 15.66% of events can change through unknown completion. More training under the same hard observation cannot remove the forced error; the next second-domain experiment must first audit a separate observation model using training data.','',
              'The bound audit exposed a remaining mean-map postprocessing error: restoring observed-free cells after sampling reopened invalid support. The evaluator and qualitative renderer now apply blocked evidence before a final validity projection. Six original clean checkpoints were replayed at K=128 with the same seeds and chunk size; all hidden-map metrics reproduce exactly and all 27,522 event predictions satisfy the bounds. The superseded v1 CSVs are retained. Corrected correlated/independent Brier is 0.51142 ± 0.00198 / 0.51130 ± 0.00218, still a null transfer diagnostic.','',
              '![Observation-conditioned error floor](site/assets/unscenes3d_observation_ceiling.svg)','',
              '## Reproduce','', '```bash',
              'PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/evaluate_flatlands_clean_controls.py',
              'PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/evaluate_flatlands_marginal_shuffle.py',
              '# Reproduce six UnScenes3D mean-map v2 evaluations in fresh output directories:',
              'for prefix in unscenes3d_ground_valid_ unscenes3d_ground_valid_independent_; do',
              '  for seed in 20260831 20260901 20260902; do',
              '    PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/evaluate_unscenes3d_conpath_mean_map.py \\',
              '      --checkpoint "results/${prefix}support_clamped_f16_v1/seed${seed}/best.pt" \\',
              '      --output-dir "results/${prefix}support_clamped_mean_map_k128_v2/seed${seed}" \\',
              '      --seed "$seed" --validation-samples 128 --sample-chunk 16',
              '  done',
              'done',
              'PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/audit_unscenes3d_observation_ceiling.py',
              'PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/analyze_flatlands_clean_validation.py --controls-root results/paper_clean_checkpoint_controls_v1 --shuffle-root results/paper_clean_marginal_shuffle_v1 --publish-site',
              '/home/hairo/miniconda3/bin/python3.13 scripts/build_paper_evidence.py', '```','',
              'Completed checkpoint controls verify their manifests when resumed. The shuffle command refuses to overwrite completed outputs; reproduce it in a fresh checkout/result directory. Raw data and checkpoints remain ignored by Git. SVG and PDF figures are included under `site/assets/`.','']
    (ROOT/'PAPER_EVIDENCE.md').write_text('\n'.join(lines))
    page=ROOT/'site/index.html'
    source=page.read_text()
    start='<!-- CLEAN_PAPER_ROWS_START -->'
    end='<!-- CLEAN_PAPER_ROWS_END -->'
    table_rows=[]
    for method,m in analysis['methods'].items():
        table_rows.append('<tr><td>'+html.escape(m['label'])+'</td><td>'+str(m['seed_count'])+'</td><td>'+fmt(m['aggregate']['brier'])+'</td><td>'+fmt(m['equal_coverage']['0.3'])+'</td></tr>')
    if source.count(start)!=1 or source.count(end)!=1:
        raise ValueError('expected exactly one generated paper-table region')
    before,remainder=source.split(start)
    _,after=remainder.split(end)
    page.write_text(before+start+'\n                '+'\n                '.join(table_rows)+'\n                '+end+after)
    print(json.dumps({'evidence':'PAPER_EVIDENCE.md','controls':list(controls),'figures':5}))


if __name__=='__main__':main()
