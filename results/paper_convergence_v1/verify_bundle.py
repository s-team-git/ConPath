"""Validate assembled paper artifacts and existing source bindings; no model/data evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import csv,hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads((ROOT/p).read_text())
def near(a,b):assert abs(float(a)-float(b))<2e-12,(a,b)
docs=['PAPER_DRAFT.md','PAPER_EVIDENCE.md','PAPER_TABLES.md','PAPER_FIGURES.md','results/FINAL_MODEL_SELECTION.md']
s=load('results/paper_validation_snapshot.json')
rows=list(csv.DictReader((ROOT/'results/paper_validation_snapshot.csv').open()))
assert len(rows)==len(s['evidence_rows'])==29
for a,b in zip(rows,s['evidence_rows']):
 for k in ['cohort','method']:assert a[k]==b[k]
 assert a['seeds']==';'.join(map(str,b['seeds']))
 for key,m in b['metrics'].items():
  for suffix in ['mean','sample_sd']:
   col=key+'_'+suffix
   if col not in a:continue
   if m.get(suffix) is None:assert a[col]==''
   else:near(a[col],m[suffix])
assert s['all_current_prediction_numeric_checks_passed']
for k in ['raw_archive_images_opened','physical_test_assets_opened','final_test_assets_opened','location_6_assets_opened','gpu_used','model_inference_runs','new_training_runs','posthoc_model_or_epoch_selection']:assert not s[k]
# Verify existing byte identities, without decoding datasets or loading a model.
source_count=0
for r in s['source_registry']:
 p=ROOT/r['path']
 if p.suffix=='.npz':assert str(p.relative_to(ROOT)).startswith(('results/parent_group_pilot_v1/','results/coherent_parent_pilot_v1/'))
 assert p.is_file() and sha(p)==r['sha256'],r['path']
 source_count+=1
embedded=s['software']['snapshot_builder_source'].encode()
assert hashlib.sha256(embedded).hexdigest()==s['software']['snapshot_builder_sha256']
# Figures and their archived inputs. The inventory chooses the final timestamp explicitly.
fdoc=(ROOT/'PAPER_FIGURES.md').read_text()
mp=re.search(r'\]\((results/paper_figures/[^)]+/manifest\.json)\)',fdoc).group(1)
manifest=load(mp);asset_records=[]
for figure in manifest['figures']:
 for r in figure['files']:
  assert sha(ROOT/r['path'])==r['sha256'],r['path'];asset_records.append(r)
for path,h in manifest['source_hashes'].items():assert sha(ROOT/path)==h,path
# Independent numeric implementations used by evidence and plot agents must agree.
fmetrics=load(str(Path(mp).parent/'metrics_from_saved_predictions.json'))
matched=0
for r in fmetrics['runs']:
 other=next(x for x in s['development_run_metrics'] if (x['method'],x['seed'],x['K'])==(r['method'],r['seed'],r['K']))
 for key in ['brier','nll','ece','false_safe_at_0_8','coverage_at_0_8']:
  a=r['metrics'][key];b=other['equal_parent'][key]
  if a is None:assert b is None
  else:near(a,b)
 near(r['metrics']['risk30'],other['equal_parent']['risk_at_30_percent']);matched+=1
# Local manuscript assets, excluding remote citations and anchors.
links=[]
for name in docs:
 p=ROOT/name;assert p.is_file() and p.stat().st_size>100
 for url in re.findall(r'!?\[[^\]\n]+\]\(([^)\n]+)\)',p.read_text()):
  if '://' in url or url.startswith('#'):continue
  target=url.split('#')[0];q=p.parent/target
  assert q.exists(),(name,target);links.append((name,target))
# The external experiment source lock remains exactly intact.
protocol=load('results/flatlands_external_formal_protocol_v1/protocol.json')
for path,h in protocol['source_hashes'].items():assert sha(ROOT/path)==h,path
assert (ROOT/'results/flatlands_external_formal_v1/STOP').is_file()
active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if re.search(r'(?:^|[ /])(?:run_flatlands_formal_(?:stages|full)|train_flatlands_formal|evaluate_flatlands_formal|watch_flatlands_formal_stages)\.py(?: |$)',cmd):active.append({'pid':int(proc.name),'cmd':cmd})
 except (OSError,ProcessLookupError):pass
assert not active,active
result={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'scope':'assembled manuscript, saved numeric consistency, byte provenance, cancelled queue; not scientific submission approval','snapshot':{'path':'results/paper_validation_snapshot.json','sha256':sha(ROOT/'results/paper_validation_snapshot.json')},'csv_sha256':sha(ROOT/'results/paper_validation_snapshot.csv'),'aggregate_rows':29,'source_files_verified_without_model_execution':source_count,'plot_run_metric_matches':matched,'figure_manifest':{'path':mp,'sha256':sha(ROOT/mp)},'figure_groups':len(manifest['figures']),'figure_export_hashes_verified':len(asset_records),'local_links_verified':len(links),'frozen_external_source_hashes_verified':len(protocol['source_hashes']),'project_external_gpu_workers':0,'new_training':False,'new_inference':False,'test_assets_decoded':False,'documents':{p:sha(ROOT/p) for p in docs},'validator_sha256':sha(Path(__file__))}
out=ROOT/'results/paper_convergence_v1/bundle_verification.json'
assert not out.exists(),'Refuse overwrite completed verification'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['documents','validator_sha256']},ensure_ascii=False))
