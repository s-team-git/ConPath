#!/usr/bin/env python3
"""Whole-scene paired analysis for new K4 controls and fixed optimistic rule."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from pathrel.flatlands_eval import join_flatlands_predictions
from pathrel.flatlands_query import sha256_path
from scripts.evaluate_flatlands_clean_controls import SEEDS, SELECTION, QUERIES
from scripts.analyze_flatlands_clean_validation import arrays, risk_at_coverage


def scene_bootstrap(records, scenes, counts):
    lookup = {s:i for i,s in enumerate(scenes)}
    scene_ids = np.array([lookup[r.scene_key] for r in records])
    p, y, w = arrays(records)
    errors = np.bincount(scene_ids, weights=(p-y)**2*w*len(scenes), minlength=len(scenes))
    # Pre-aggregate probability tie groups by scene. This computes the same
    # label-blind fractional boundary acceptance as risk_at_coverage.
    levels, assignments = np.unique(-p, return_inverse=True)
    mass = np.zeros((len(scenes),len(levels))); error = np.zeros_like(mass)
    np.add.at(mass,(scene_ids,assignments),w*len(scenes))
    np.add.at(error,(scene_ids,assignments),w*(1-y)*len(scenes))
    m, e = counts@mass/len(scenes), counts@error/len(scenes)
    cumulative = np.cumsum(m,axis=1)
    idx = np.minimum((cumulative < .3).sum(1), len(levels)-1)
    index = np.arange(len(counts));before = cumulative[index,idx]-m[index,idx]
    fraction = np.clip((.3-before)/m[index,idx],0,1)
    ec = np.cumsum(e,axis=1)
    risk = (ec[index,idx]-e[index,idx]+fraction*e[index,idx])/.3
    # First bootstrap draw is the unresampled cohort, independently checked.
    assert abs(risk[0]-risk_at_coverage(p,y,w,.3)['false_safe_rate']) < 1e-12
    return counts@errors/len(scenes), risk


def main():
    out=Path('results/baseline_review_20260909_v1')
    path=Path('results/current_baseline_k4_v1/report.json'); raw=json.loads(path.read_text())
    for name,expected in raw['source_hashes'].items(): assert sha256_path(Path(name))==expected,name
    groups=defaultdict(list)
    for run in raw['runs']:
        root=Path('results/current_baseline_k4_v1')/run['method']/str(run['seed'])
        for name,r in run['outputs'].items(): assert sha256_path(root/name)==r['sha256']
        groups[run['method']].append(run)
    methods={}
    def stat(values):
        a=np.array(values,dtype=float)
        return {'mean':float(a.mean()),'seed_sd':float(a.std(ddof=1)) if len(a)>1 else None,'values':values}
    for method,runs in groups.items():
        methods[method]={'repeat_count':len(runs),'samples':runs[0]['samples'],
                         'brier':stat([r['event_metrics']['brier'] for r in runs]),
                         'risk30':stat([r['equal_coverage']['0.3']['false_safe_rate'] for r in runs]),
                         'mean_iou':stat([r['map_metrics']['mean_iou'] for r in runs]),
                         'mes':stat([r['map_metrics']['masked_energy_score'] for r in runs]) if runs[0]['map_metrics']['masked_energy_score'] is not None else None}
    record_cache={}; source_hashes={str(path):sha256_path(path),str(Path(__file__)):sha256_path(Path(__file__))}
    for method,runs in groups.items():
        for run in runs:
            p=Path('results/current_baseline_k4_v1')/method/str(run['seed'])/'predictions.csv'
            records,_=join_flatlands_predictions(p,SELECTION,QUERIES,split='validation')
            record_cache[(method,run['seed'])]=tuple(sorted(records,key=lambda r:r.key));source_hashes[str(p)]=sha256_path(p)
    canonical=record_cache[('correlated',SEEDS[0])]
    scenes=sorted({r.scene_key for r in canonical});n=len(scenes);assert n==142
    expected=[(r.key,r.scene_key,r.target) for r in canonical]
    for records in record_cache.values(): assert [(r.key,r.scene_key,r.target) for r in records]==expected
    rng=np.random.default_rng(20260909);draws=rng.integers(n,size=(2000,n))
    counts=np.vstack([np.ones(n),np.stack([np.bincount(d,minlength=n) for d in draws])])
    boot={key:scene_bootstrap(records,scenes,counts) for key,records in record_cache.items()}
    comparisons=[]
    for reference in ('correlated','correlated_k128'):
        for comparator in (('independent','tiny_deterministic','all_floor','nearest_observed') if reference=='correlated' else ('all_floor',)):
            per_seed=[];deltas=[]
            for seed in SEEDS:
                if reference=='correlated_k128':
                    p=Path(f'results/p1_flatlands_conpath_k128_support_clamped_v1/seed{seed}_conpath/predictions_validation.csv')
                    records,_=join_flatlands_predictions(p,SELECTION,QUERIES,split='validation');records=tuple(sorted(records,key=lambda r:r.key))
                    assert [(r.key,r.scene_key,r.target) for r in records]==expected
                    left=scene_bootstrap(records,scenes,counts);source_hashes[str(p)]=sha256_path(p)
                else:left=boot[(reference,seed)]
                right=boot[(comparator,seed if comparator in ('independent','tiny_deterministic') else 'rule')]
                delta=np.stack(right)-np.stack(left);deltas.append(delta)
                per_seed.append({'seed':seed,'brier_comparator_minus_conpath':float(delta[0,0]),'brier_scene_ci95':np.quantile(delta[0,1:],[.025,.975]).tolist(),
                                 'risk30_comparator_minus_conpath':float(delta[1,0]),'risk30_scene_ci95':np.quantile(delta[1,1:],[.025,.975]).tolist()})
            pooled=np.mean(deltas,axis=0)
            comparisons.append({'reference':reference,'comparator':comparator,'seeds':per_seed,
                                'fixed_three_repeat_average':{'brier_delta':float(pooled[0,0]),'brier_scene_ci95':np.quantile(pooled[0,1:],[.025,.975]).tolist(),
                                     'risk30_delta':float(pooled[1,0]),'risk30_scene_ci95':np.quantile(pooled[1,1:],[.025,.975]).tolist()}})
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'validation_only':True,'test_evaluated':False,'external_paper_superiority_established':False,
             'methods':methods,'paired':comparisons,'scene_count':n,'map_scene_count':160,'event_count':4224,'bootstrap_draws':2000,
             'bootstrap_scope':'Whole scenes paired, conditional on three fixed checkpoints; not a population CI over training seeds. Positive delta favors ConPath. No multiple-comparison adjustment; descriptive diagnostics.',
             'source_hashes':source_hashes,'current_run_report_sha256':sha256_path(path)}
    (out/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    Path('site/data/current_baseline_k4_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'methods':methods,'paired':[{'reference':p['reference'],'comparator':p['comparator'],**p['fixed_three_repeat_average']} for p in comparisons]},ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
