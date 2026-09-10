#!/usr/bin/env python3
"""Freeze only explicit existing sources. Never recurse into experimental data."""
import argparse, csv, datetime, hashlib, io, json, shutil, subprocess
from collections import Counter
from pathlib import Path

ROOT = Path('/home/hairo/pathrel_transfer/pathrel_pro6000')
DEST = ROOT / 'results/paper_submission_v1/evidence'
SNAP = 'results/paper_validation_snapshot.json'

def digest(b): return hashlib.sha256(b).hexdigest()
def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')
def write_csv(p, rows, fields):
    with p.open('x', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def classify(p):
    if 'coherent_parent' in p:return 'D2_reused_two_seed_development'
    if 'parent_group_pilot' in p:return 'D1_parent_isolated_development'
    if 'p0_neural' in p:return 'S1_synthetic_mechanism'
    if 'marginal_shuffle' in p:return 'M1_historical_fixed_marginal_mechanism'
    if p.startswith('site/data/') or 'paper_clean' in p or 'p1_flatlands' in p:return 'H1_historical_overlap_diagnostic'
    if p.endswith('.py'):return 'frozen_implementation_reference_only'
    return 'cross_cohort_paper_document'

def main():
    if DEST.exists(): raise SystemExit('Refusing to overwrite existing evidence directory')
    s=json.loads((ROOT/SNAP).read_text())
    assert s['physical_archive_split']=='train'
    DEST.mkdir(parents=True, exist_ok=False)
    mappings=[]; included=[]; excluded=[]
    extra=[SNAP,'results/paper_validation_snapshot.csv','PAPER_EVIDENCE.md','PAPER_TABLES.md','PAPER_FIGURES.md','results/FINAL_MODEL_SELECTION.md','results/paper_clean_ablation_matrix_v1/analysis/report.json']
    entries=list(s['source_registry'])+[{ 'path':p,'sha256':None,'role':'frozen paper document or scalar table source'} for p in extra]
    seen=set()
    for e in entries:
        p=e['path']
        if p in seen:continue
        seen.add(p)
        q=Path(p)
        assert not q.is_absolute() and '..' not in q.parts
        item={'source_path':p,'source_registry_sha256':e['sha256'],'purpose':e['role'],'cohort':classify(p)}
        if q.suffix in {'.npz','.pt'}:
            reason='Excluded saved raster/target packet or probability arrays; this pack carries scalar development targets separately.' if q.suffix=='.npz' else 'Excluded model weights; original recorded hash retained without reading checkpoint bytes.'
            item.update({'status':'excluded_without_opening','package_path':None,'sha256':e['sha256'],'reason':reason})
            mappings.append(item);excluded.append(item);continue
        assert q.suffix in {'.json','.csv','.py','.md'},p
        assert 'location_6' not in q.parts and 'test' not in q.parts and 'final_test' not in q.parts,p
        src=ROOT/p
        assert not src.is_symlink(),p
        b=src.read_bytes();h=digest(b)
        if e['sha256']:assert h==e['sha256'],p
        package_path='sources/'+p
        out=DEST/package_path;out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as f:f.write(b)
        item.update({'status':'included_byte_exact','package_path':package_path,'sha256':h,'bytes':len(b),'content_scope':('inert source text; never imported or executed' if q.suffix=='.py' else 'saved metadata/aggregate/scalar predictions; no raster or checkpoint bytes')})
        mappings.append(item);included.append(item)
    write_json(DEST/'path_mapping.json',{'schema_version':1,'mapping':mappings,'paths_are_package_relative':True,'references_in_original_reports_are_not_dereferenced':True})
    write_csv(DEST/'inventory_sources.csv',[{k:x.get(k) for k in ['source_path','package_path','status','cohort','purpose','sha256','bytes','reason']} for x in mappings],['source_path','package_path','status','cohort','purpose','sha256','bytes','reason'])
    write_json(DEST/'excluded_sources.json',{'files':excluded,'count':len(excluded),'not_opened_by_archive_builder':True,'also_not_included':['raw archive images','posterior world arrays','historical label/prediction rows possibly involving archive test','final-test or location_6 assets','untrained/missing current direct-query/no-event/no-global outputs'],'historical_reports':'Existing aggregate reports and read-scope corrections remain included. They mention prior test identifiers as metadata, without distributing test labels or reading referenced paths.'})
    catalog=[]
    for i,r in enumerate(s['evidence_rows']):
        catalog.append({'row_index':i,'snapshot_pointer':f'/evidence_rows/{i}','cohort':r['cohort'],'evidence_level':r['evidence_level'],'method':r['method'],'K':r['K'],'actual_map_count':r.get('actual_map_count'),'K_role':r.get('K_role'),'seeds':r['seeds'],'repeats':r['repeats'],'weighting':r.get('weighting'),'claim_scope':r['claim_scope'],'source_pointers':r.get('sources',[])})
    write_json(DEST/'row_catalog.json',{'rows':catalog,'missing_current_methods':['direct_query','no_event','no_global'],'zero_means_measured_zero_only':True})
    numeric=[]
    def visit(x,p,section):
        if isinstance(x,bool):return
        if isinstance(x,(int,float)):
            numeric.append({'snapshot_pointer':p,'value_json':json.dumps(x,allow_nan=False),'section':section})
        elif isinstance(x,dict):
            for k,v in x.items():visit(v,p+'/'+str(k).replace('~','~0').replace('/','~1'),section)
        elif isinstance(x,list):
            for i,v in enumerate(x):visit(v,p+'/'+str(i),section)
    for k in ['evidence_rows','development_run_metrics','paired_intervals','development_posthoc_diagnostics','synthetic_runs','fixed_marginal_mechanism','parent_protocol','coherent_protocol','parent_manifest_counts']:
        visit(s[k],'/'+k,k)
    write_csv(DEST/'numeric_index.csv',numeric,['snapshot_pointer','value_json','section'])
    write_json(DEST/'scope.json',{'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'snapshot_sha256':digest((ROOT/SNAP).read_bytes()),'source_registry_entries':len(s['source_registry']),'included_original_files':len(included),'excluded_registry_files_without_opening':len(excluded),'included_original_bytes':sum(x['bytes'] for x in included),'evidence_rows':len(s['evidence_rows']),'development_run_metrics':len(s['development_run_metrics']),'numeric_index_entries':len(numeric),'training_runs':0,'model_inference_runs':0,'final_test_assets_opened':0,'location_6_assets_opened':0,'raw_images_opened':0,'checkpoint_bytes_opened':0,'array_files_opened_by_archive_builder':0,'external_training_cancelled':True,'scope':'Existing-evidence portability and exact saved-score arithmetic only. Original historical aggregate scope flags do not override the included later erratum. Current development split is reused and not final test.'})
    shutil.copyfile(__file__,DEST/'build_archive.py')
    print(json.dumps({'files':len(included),'excluded':len(excluded),'bytes':sum(x['bytes'] for x in included),'numeric_entries':len(numeric)},indent=2))
if __name__=='__main__':main()
