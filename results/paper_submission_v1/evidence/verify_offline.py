#!/usr/bin/env python3
"""Offline scalar-data audit. Python standard library only; no model/data imports.
Run from any working directory. With --report, refuse to overwrite an old receipt.
"""
import argparse, bisect, csv, datetime, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
BASE=Path(__file__).resolve().parent
SNAP='sources/results/paper_validation_snapshot.json'
TARGET='sources/results/paper_submission_v1/analysis/targets.csv'
FIELDS=['global_id','parent_group','source_dataset','candidate_index','radius_cells','target']
METRICS=['brier','nll','ece','false_safe_at_0_8','coverage_at_0_8','risk_at_30_percent','positive_rate','mean_score']

def path(p):
    q=Path(p)
    assert not q.is_absolute() and '..' not in q.parts,p
    r=BASE/q
    assert not r.is_symlink() and r.resolve().is_relative_to(BASE),p
    return r

def load(p):return json.loads(path(p).read_text())
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def rows(p):
    with path(p).open(newline='') as f:return list(csv.DictReader(f))
def key(r):return (r['global_id'],int(r['candidate_index']),int(r['radius_cells']))
def pointer(d,p):
    for k in p.strip('/').split('/'):
        k=k.replace('~1','/').replace('~0','~')
        d=d[int(k)] if isinstance(d,list) else d[k]
    return d

def summarize(rs, weight_mode):
    counts=Counter(r['parent_group'] for r in rs)
    ww=[1/(len(counts)*counts[r['parent_group']]) if weight_mode=='equal_parent' else 1/len(rs) for r in rs]
    total=math.fsum(ww);ww=[w/total for w in ww]
    pp=[r['probability'] for r in rs];yy=[r['target'] for r in rs]
    brier=math.fsum(w*(p-y)**2 for w,p,y in zip(ww,pp,yy))
    nll=math.fsum(w*(-y*math.log(max(1e-6,min(1-1e-6,p)))-(1-y)*math.log(1-max(1e-6,min(1-1e-6,p)))) for w,p,y in zip(ww,pp,yy))
    bin_rows=defaultdict(list)
    edges=[i*0.1 for i in range(11)]
    for w,p,y in zip(ww,pp,yy):bin_rows[min(9,bisect.bisect_right(edges,p)-1)].append((w,p,y))
    ece=math.fsum(abs(math.fsum(w*(y-p) for w,p,y in b)) for b in bin_rows.values())
    coverage=math.fsum(w for w,p in zip(ww,pp) if p>=.8)
    unsafe=math.fsum(w*(1-y) for w,p,y in zip(ww,pp,yy) if p>=.8)
    ties=defaultdict(list)
    for w,p,y in zip(ww,pp,yy):ties[p].append((w,y))
    remaining=.3;accepted=[];errors=[]
    for score in sorted(ties,reverse=True):
        t=ties[score];mass=math.fsum(w for w,y in t)
        fraction=min(1.,max(0.,remaining)/mass)
        accepted.append(fraction*mass)
        errors.append(fraction*math.fsum(w*(1-y) for w,y in t))
        remaining-=fraction*mass
        if remaining<=1e-15:break
    return {'brier':brier,'nll':nll,'ece':ece,'false_safe_at_0_8':unsafe/coverage if coverage else None,'coverage_at_0_8':coverage,'risk_at_30_percent':math.fsum(errors)/math.fsum(accepted),'positive_rate':math.fsum(w*y for w,y in zip(ww,yy)),'mean_score':math.fsum(w*p for w,p in zip(ww,pp))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',help='New JSON receipt path; refuses overwrite');args=ap.parse_args()
    inv=load('inventory.json')
    assert sha('inventory.json')==path('inventory.sha256').read_text().split()[0]
    for f in inv['files']:
        assert sha(f['path'])==f['sha256'],f['path']
        assert path(f['path']).stat().st_size==f['bytes'],f['path']
    snapshot=load(SNAP);mapping=load('path_mapping.json')['mapping'];mapped={r['source_path']:r for r in mapping}
    assert sha(SNAP)==inv['snapshot_sha256']
    for r in mapping:
        if r['status']=='included_byte_exact':assert sha(r['package_path'])==r['sha256']
        else:assert r['package_path'] is None
    for p in BASE.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():assert p.suffix.lower() not in {'.pt','.pth','.npz','.npy','.png','.jpg','.jpeg','.zip','.tar'},str(p)
    selected=rows('sources/results/parent_group_pilot_v1/data/selected.csv')
    assert len(selected)==165 and all(r['archive_split']=='train' for r in selected)
    splits=Counter(r['candidate_split'] for r in selected)
    assert splits=={'train':100,'calibration':25,'validation':40},splits
    ids={r['global_id']:r for r in selected if r['candidate_split']=='validation'}
    parents={k:{r['parent_group'] for r in selected if r['candidate_split']==k} for k in splits}
    assert not (parents['train']&parents['calibration'] or parents['train']&parents['validation'] or parents['calibration']&parents['validation'])
    targets=rows(TARGET);assert len(targets)==1545 and set(targets[0])==set(FIELDS)
    labels={key(r):r for r in targets};assert len(labels)==1545
    assert len({(k[0],k[1]) for k in labels})==515
    assert set(k[2] for k in labels)=={0,10,20}
    for k,r in labels.items():
        assert r['target'] in {'0','1'}
        assert r['global_id'] in ids
        assert all(r[f]==ids[r['global_id']][f] for f in ['parent_group','source_dataset'])
    assert len({r['parent_group'] for r in targets})==40
    rec=load('sources/results/paper_submission_v1/analysis/target_export_receipt.json')
    assert rec['targets_sha256']==sha(TARGET)
    assert rec['snapshot_sha256']==sha(SNAP)
    registry={r['path']:r for r in snapshot['source_registry']}
    for p in rec['packets']:
        assert registry[p['path']]['sha256']==p['sha256']
        assert p['archive_split']=='train' and p['candidate_split']=='validation'
        assert set(p['arrays_decoded'])=={'candidate_indices','targets'}
    assert len(rec['packets'])==40
    assert all(rec[k]==0 for k in ['map_arrays_decoded','raw_archive_assets_read','final_test_assets_read','physical_test_assets_read','location_6_assets_read','model_inference_runs','training_runs'])
    catalog=load('row_catalog.json')['rows'];assert len(catalog)==len(snapshot['evidence_rows'])==29
    pointer_count=0
    for i,r in enumerate(snapshot['evidence_rows']):
        assert catalog[i]['snapshot_pointer']==f'/evidence_rows/{i}'
        assert catalog[i]['method']==r['method'] and catalog[i]['cohort']==r['cohort']
        for source in r['sources']:
            f=mapped[source['path']];assert f['status']=='included_byte_exact'
            if source.get('json_pointer'):pointer(load(f['package_path']),source['json_pointer']);pointer_count+=1
    numrows=rows('numeric_index.csv')
    for r in numrows:assert pointer(snapshot,r['snapshot_pointer'])==json.loads(r['value_json']),r['snapshot_pointer']
    results=[];maximum=0.;comparisons=0;monotonicity_checks=0
    for run in snapshot['development_run_metrics']:
        f=mapped[run['prediction']['path']];scores=rows(f['package_path'])
        assert len(scores)==1545 and set(scores[0])=={'global_id','parent_group','candidate_index','radius_cells','probability'}
        predictions={key(r):r for r in scores};assert predictions.keys()==labels.keys()
        joined=[]
        for k,r in predictions.items():
            t=labels[k];p=float(r['probability']);assert 0<=p<=1 and math.isfinite(p)
            assert r['parent_group']==t['parent_group']
            joined.append({**t,'target':int(t['target']),'probability':p})
        per={}
        for mode in ['equal_parent','pooled']:
            actual=summarize(joined,mode);differences={}
            for metric in METRICS:
                expected=run[mode][metric];got=actual[metric]
                if expected is None:assert got is None;delta=0.
                else:
                    delta=abs(got-expected);assert delta<1e-11,(run['method'],run['seed'],run['K'],mode,metric,got,expected)
                maximum=max(maximum,delta);differences[metric]=delta;comparisons+=1
            per[mode]={'recomputed':actual,'absolute_differences':differences}
        byquery=defaultdict(dict)
        for k,r in predictions.items():byquery[k[:2]][k[2]]=float(r['probability'])
        for k,rad in byquery.items():
            assert rad[0]>=rad[10]>=rad[20],(run['method'],k)
            assert int(labels[(k[0],k[1],0)]['target'])>=int(labels[(k[0],k[1],10)]['target'])>=int(labels[(k[0],k[1],20)]['target'])
            monotonicity_checks+=1
        results.append({'method':run['method'],'seed':run['seed'],'K':run['K'],'actual_map_count':run['actual_map_count'],'source_prediction':run['prediction']['path'],'events':len(joined),'checks':per})
    report={'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pass':True,'inventory_sha256':sha('inventory.json'),'snapshot_sha256':sha(SNAP),'copied_source_files':sum(r['status']=='included_byte_exact' for r in mapping),'excluded_source_files_not_opened':sum(r['status']!='included_byte_exact' for r in mapping),'inventory_files_verified':len(inv['files']),'numeric_index_pointers_checked':len(numrows),'source_json_pointers_checked':pointer_count,'saved_score_files_checked':len(results),'events_per_score_file':1545,'scalar_metric_comparisons':comparisons,'maximum_absolute_metric_difference':maximum,'query_radius_triples_checked':monotonicity_checks,'target_export_receipt_verified_without_opening_packets':True,'scope':'Offline arithmetic verification of already saved development predictions; no new metric definition, model inference, training, hyperparameter search or statistical interval calculation. This verifier only opens files inside this portable directory. It does not verify raw geometry or continuous-map arrays.','historical_aggregate_only':True,'final_test_assets_opened':0,'raw_map_arrays_opened':0,'checkpoint_bytes_opened':0,'array_files_opened':0,'runs':results}
    if args.report:
        with Path(args.report).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
