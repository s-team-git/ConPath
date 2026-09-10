#!/usr/bin/env python3
"""Assembly receipt: add already exported scalar tables and final paper documents.
Run only during initial construction; refuses a previously sealed inventory.
"""
import csv,hashlib,json,datetime
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
assert not (BASE/'inventory.json').exists(),'Refusing to change sealed package'
def digest(b):return hashlib.sha256(b).hexdigest()
def write(p,d):
 with p.open('w',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
mapping=json.loads((BASE/'path_mapping.json').read_text())
records={r['source_path']:r for r in mapping['mapping']}
paths=['results/paper_submission_v1/analysis/'+n for n in ['targets.csv','export_saved_targets.py','target_export_receipt.json','development_metrics_by_seed_and_stratum.csv','primary_three_seed_strata.csv','export_metric_views.py','metric_view_receipt.json']]+['PAPER_DRAFT.md','PAPER_READING_ZH.md','PAPER_TABLES.md']
changes=[]
for p in paths:
 src=ROOT/p
 assert src.is_file() and not src.is_symlink(),src
 b=src.read_bytes();h=digest(b);prior=records.get(p)
 out=BASE/'sources'/p;out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():
  assert p=='PAPER_TABLES.md','Only assembly-stage table document refresh allowed'
  changes.append({'path':p,'initial_document_sha256':digest(out.read_bytes()),'final_document_sha256':h,'reason':'Root finished new R20 counterexample table before evidence sealing; immutable numeric snapshot did not change.'})
 out.write_bytes(b)
 records[p]={'source_path':p,'source_registry_sha256':None,'purpose':'existing scalar development export and bound receipt/source' if '/analysis/' in p else 'paper document frozen after current author revision','cohort':'D1_D2_physical_train_development_scalar_export' if '/analysis/' in p else 'cross_cohort_paper_document','status':'included_byte_exact','package_path':'sources/'+p,'sha256':h,'bytes':len(b),'content_scope':'existing saved scalar data and metadata only; no raw maps, checkpoints or final-test labels' if not p.endswith('.py') else 'inert implementation archive; not executed by offline verifier'}
mapping['mapping']=list(records.values());mapping['final_document_freeze_note']='Original paper files are byte-frozen at this assembly stage. Subsequent navigation-only edits to root documents do not rewrite these archived documents.'
write(BASE/'path_mapping.json',mapping)
fields=['source_path','package_path','status','cohort','purpose','sha256','bytes','reason']
with (BASE/'inventory_sources.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in records.values())
inc=[r for r in records.values() if r['status']=='included_byte_exact']
write(BASE/'assembly_completion.json',{'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'included_original_files':len(inc),'included_original_bytes':sum(r['bytes'] for r in inc),'excluded_registry_files':sum(r['status']!='included_byte_exact' for r in records.values()),'new_scalar_target_rows':1545,'scalar_target_scope':'existing physical-train development-validation labels, exported by root from two named cache arrays only; no map arrays or final-test data','array_files_opened_by_archive_builder':0,'model_inference_runs':0,'training_runs':0,'final_test_assets_opened':0,'doc_updates_during_unsealed_assembly':changes,'initial_copy_scope_receipt':'scope.json','derived_views':'four existing-prediction export files from analysis; no new model experiment or statistical interval','source_documents_may_have_later_navigation_changes':True})
files=[]
for p in sorted(BASE.rglob('*')):
 if p.is_file():
  assert not p.is_symlink()
  b=p.read_bytes();files.append({'path':p.relative_to(BASE).as_posix(),'sha256':digest(b),'bytes':len(b),'role':'byte-exact archived source' if p.is_relative_to(BASE/'sources') else 'portable lookup/dictionary/verification tool or assembly metadata'})
inv={'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_sha256':digest((BASE/'sources/results/paper_validation_snapshot.json').read_bytes()),'files':files,'files_count':len(files),'files_bytes':sum(r['bytes'] for r in files),'included_original_files':len(inc),'excluded_registry_files':65,'excludes_from_self_inventory':['inventory.json','inventory.sha256','verification.json'],'no_followup_access_to_sources_required':True}
write(BASE/'inventory.json',inv)
(BASE/'inventory.sha256').write_text(digest((BASE/'inventory.json').read_bytes())+'  inventory.json\n')
print(json.dumps({k:v for k,v in inv.items() if k!='files'},indent=2))
