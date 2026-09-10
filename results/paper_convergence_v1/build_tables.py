"""Render tables from saved reports; no datasets, model loading, or inference."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def fmt(v):
    if v is None:return '—'
    if isinstance(v,dict):
        m=v.get('mean');s=v.get('sample_sd',v.get('sd'))
        return '—' if m is None else f'{m:.5f}'+(f' ± {s:.5f}' if s is not None else '')
    return f'{v:.5f}'
oldpath='results/paper_clean_analysis_v1/report.json'
abpath='results/paper_clean_ablation_matrix_v1/analysis/report.json'
parentpath='results/parent_group_pilot_v1/analysis.json'
old,ab,parent=map(read,[oldpath,abpath,parentpath])
lines=['# Paper tables / 论文数字表','',
'2026-09-10 · 所有数字为已有验证/历史诊断；无最终测试成绩。不同cohort、K和seed集合不混排。±为训练seed样本标准差，不是置信区间。原报告SHA见文末，详细预测来源与补算指标见 [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md) 及 [snapshot](results/paper_validation_snapshot.json)。','',
'## T1. Current parent-isolated development comparison','',
'100 training / 25 selection / 40 validation parent places; same 515 queries × radii 0/10/20 cells, 1545 events. Seeds 20260910/20260911/20260912. Parent-weighted, validation-only, bounded training. Missing methods were not run on this cohort.','',
'| Method | Actual K | Seeds | Event Brier ↓ | Risk at 30% coverage ↓ | Hidden-map Brier ↓ |',
'|---|---:|---:|---:|---:|---:|']
labels={'deterministic':'Deterministic occupancy','independent':'Independent Bernoulli','direct_query':'Direct-query','no_global':'No-global','no_event':'No-event','correlated':'ConPath'}
for method,label in labels.items():
    if method not in parent['methods']:lines.append(f'| {label} | — | 0 | Not evaluated | — | — |');continue
    r=parent['methods'][method];k=1 if method=='deterministic' else 32
    b=r['budgets'][str(k)]
    lines.append(f'| {label} | {k} | {r["repeats"]} | {fmt(b["brier"])} | {fmt(b["risk30"])} | {fmt(b["map"]["sample_vote_cell_brier"])} |')
lines+=['','Hidden-map Brier above uses the stored **hard-world free-frequency** estimator on unknown valid cells; deterministic K1 is its binary-map squared error, not continuous marginal Brier. Risk30 is not risk at confidence 0.8. The continuous native marginal metric, if saved, is separately named in the snapshot.','',
'The source-stratified paired parent interval for independent minus ConPath is [0.00439, 0.01251]; deterministic minus ConPath is [−0.01616, 0.03377]. Intervals condition on the three recorded seed means. The 40 validation places were later reused for development.','',
'<!-- RETROSPECTIVE_METRICS -->','',
'## T2. Historical six-method diagnostic (not unseen-place main evidence)','',
'160 train / 160 validation packets; 4224 common event rows from 142 contributing subscenes. The later audit found train-place overlap in 27/160 validation observations and historical archive-test access. Common input/labels/query/radii/support/evaluation contract within this table does not repair that flaw. Do not copy these numbers into T1.','',
'| Method | Actual K | Training seeds | Event Brier ↓ | Event NLL ↓ | Event ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for method,label,k in [('completion','Deterministic occupancy',1),('independent','Independent Bernoulli',128),('direct_query','Direct-query','N/A'),('no_global','No-global',128),('no_event','No-event',128),('conpath','ConPath',128)]:
    r=(ab if method in ['no_global','no_event'] else old)['methods'][method]
    seeds=[s['seed'] for s in r['seeds']];a=dict(r['aggregate'])
    for key in ['false_safe_rate@0.8','high_confidence_safe_coverage@0.8']:
        if key not in a:
            vals=[s['metrics'][key] for s in r['seeds']]
            a[key]={'mean':statistics.mean(vals) if all(v is not None for v in vals) else None,'sample_sd':statistics.stdev(vals) if len(vals)>1 and all(v is not None for v in vals) else None}
    lines.append('| '+ ' | '.join([label,str(k),','.join(map(str,seeds))]+[fmt(a[key]) for key in ['brier','nll','ece','false_safe_rate@0.8','high_confidence_safe_coverage@0.8']])+' |')
lines+=['','The direct-query row has only one training seed; it is not the separately trained three-seed coordinate-query control. A dash/NA is never zero. All historical full/no-event/no-global within-seed Brier intervals favor full, but the subscene bootstrap does not establish independent-building generalization. No-event still uses event-based checkpoint selection. No-global retains encoder context and local decoder correlation.','',
'## T3. Historical dependence and strong deterministic checks','',
'Same historical cohort as T2; three seeds 20260831/20260901/20260902. A derived mean map is one actual binary output, computed from the recorded K128 posterior estimate.','',
'| Control | Event Brier ↓ | Event NLL ↓ | Event ECE ↓ | Risk at 30% coverage ↓ |',
'|---|---:|---:|---:|---:|']
for method,label in [('conpath','ConPath, K128'),('marginal_shuffle','Cellwise world-index permutation, K128'),('mean_map','ConPath same-checkpoint mean map'),('independent_mean_map','Independent same-checkpoint mean map')]:
    r=old['methods'][method];a=r['aggregate'];lines.append('| '+' | '.join([label]+[fmt(a[m]) for m in ['brier','nll','ece']]+[fmt(r['equal_coverage']['0.3'])])+' |')
lines+=['','The permutation preserves every **empirical hard-world cell marginal** and changes dependence. This is distinct from separately trained independent sampling. Mean-map and equal-coverage comparisons are strong counterevidence to broad stochastic/safety superiority. Every historical independent-versus-full equal-30%-coverage risk interval includes zero.','',
'## T4. Two-seed sampling variant (supplementary only)','',
'Same 100/25/40 parent-place development cohort, restricted to seeds 20260910 and 20260911 in every row. Do not compare this two-seed mean with T1’s three-seed mean.','',
'| Method | K | Event Brier ↓ | Risk at 30% coverage ↓ | Free-class sample IoU ↑ |',
'|---|---:|---:|---:|---:|']
cp='results/coherent_parent_pilot_v1/analysis.json';co=read(cp)
for method,r in co['methods'].items():
    if method not in ['coherent_categorical','correlated','independent','deterministic']:continue
    k=r['samples'];b=r
    label={'coherent_categorical':'Correlated categorical-noise variant (candidate)','correlated':'Original ConPath','independent':'Independent Bernoulli','deterministic':'Deterministic occupancy'}[method]
    lines.append('| '+' | '.join([label,str(k),fmt(b['brier']),fmt(b['risk30']),fmt(b['map']['mean_iou'])])+' |')
lines+=['','Original minus candidate Brier: 0.01135, recorded paired parent interval [−0.00117, 0.02437]. Candidate is not selected as the paper method; no third seed is added. Its K4 risk30 is 23.84% versus original 23.73%, despite lower Brier.','',
'## Traceability and exclusions','',
'All tables are computed from the following existing reports. No new network evaluation is performed by this renderer. Original prediction hashes, evaluator identity, query agreement, threshold summaries and new/old metric status are retained in the snapshot. External under-convergence numbers, P0 synthetic scores and UnScenes3D results are not inserted into these FlatLands comparisons.','',
'| Source | SHA-256 |','|---|---|']
for p in [parentpath,oldpath,abpath,cp]:lines.append(f'| [{p}]({p}) | `{sha(p)}` |')
snapshot_path='results/paper_validation_snapshot.json'
snapshot=read(snapshot_path)
def row(method,k):
    return next(r for r in snapshot['evidence_rows'] if r['cohort']=='parent_isolated_three_seed' and r['method']==method and r['K']==k)
extra=['## T1b. Retrospective event and marginal-probability diagnostics','',
'Same fixed predictions, same three training seeds as T1. NLL/ECE/confidence-0.8 are newly summarized descriptive metrics, not original selection criteria. NLL clips at 1e-6, ECE uses ten equal-width bins; no calibration is fitted.','',
'| Method | Actual K | Event NLL ↓ | Event ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 | Continuous hidden-map Brier ↓ |',
'|---|---:|---:|---:|---:|---:|---:|']
for method,k in [('deterministic',1),('independent',32),('correlated',32)]:
    r=row(method,k)
    assert abs(r['metrics']['brier']['mean']-parent['methods'][method]['budgets'][str(k)]['brier']['mean'])<1e-12
    extra.append('| '+' | '.join([labels[method],str(k)]+[fmt(r['metrics'][x]) for x in ['nll','ece','false_safe_at_0_8','coverage_at_0_8']]+[fmt(r['continuous_map_brier'])])+' |')
extra+=['','Continuous hidden-map Brier uses the saved original sigmoid probabilities for deterministic and the saved K32 conditional-probability average for the stochastic models. **Deterministic continuous Brier is better here**; its thresholded binary Brier in T1 must not be substituted to claim better probabilistic map quality. No continuous K4 re-inference is performed.','',
'## T1c. Frozen sample budget and constraint checks','',
'K4 is the saved nested prefix of K32, with identical queries and checkpoints. All 23 current method/seed/budget score files have zero radius-monotonicity violations across their 515 query triples (including rules); this is a structural check, not calibration. The existing saved-world audits report zero observed-evidence and valid-support violations for evaluated worlds; source/extent are detailed in PAPER_EVIDENCE.md.','',
'| Method | K | Brier ↓ | ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 | Radius monotonicity violations |',
'|---|---:|---:|---:|---:|---:|---:|']
for method in ['correlated','independent']:
    for k in [4,32]:
        r=row(method,k);extra.append('| '+' | '.join([labels[method],str(k)]+[fmt(r['metrics'][x]) for x in ['brier','ece','false_safe_at_0_8','coverage_at_0_8']]+[str(r['radius_monotonicity_violating_queries'])])+' |')
extra+=['','## T1d. Existing training-free/radius-prior controls (supplement)','',
'The single train-fitted radius prior has lower NLL/ECE than ConPath but accepts no event at confidence 0.8. Its risk is undefined, not zero. Rules are one fixed evaluation each, not optimization-seed repeats. These controls remain visible without expanding the requested six-method main table.','',
'| Control | Actual maps | Brier ↓ | NLL ↓ | ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 |',
'|---|---:|---:|---:|---:|---:|---:|']
for method,label in [('all_floor','All unknown free'),('all_blocked','All unknown blocked'),('nearest_observed','Nearest observed'),('train_radius_prior','Train-fitted radius prior')]:
    r=row(method,1);extra.append('| '+' | '.join([label,'N/A' if method=='train_radius_prior' else '1']+[fmt(r['metrics'][x]) for x in ['brier','nll','ece','false_safe_at_0_8','coverage_at_0_8']])+' |')
extra+=['',f'Source for T1b–T1d: [{snapshot_path}]({snapshot_path}), SHA-256 `{sha(snapshot_path)}`. Per-radius, source, reachable/unreachable, pooled and parent-weighted summaries are preserved without changing the original query set.','']
diagnostic_path='results/parent_group_pilot_v1/diagnostics/report.json'
diag=read(diagnostic_path)
assert diag['source_analysis_sha256']==sha(parentpath)
extra+=['## T1e. Previously saved mean-map and observation-constraint diagnostics','',
'This post-hoc report predates manuscript consolidation. It uses the same current three checkpoints, references, query keys and fixed 0.5 threshold. It was not an original registered screening row, and is not a newly run evaluation. NLL/ECE were not stored for these derived controls and are not fabricated.','',
'| Derived output | Actual maps | Event Brier ↓ | Risk at 30% coverage ↓ |','|---|---:|---:|---:|']
for m,label in [('correlated_conditional_mean_threshold','ConPath same-checkpoint conditional-mean threshold'),('independent_conditional_mean_threshold','Independent same-checkpoint conditional-mean threshold')]:
    g=diag['groups'][m]['all'];extra.append('| '+' | '.join([label,'1',fmt(g['brier']),fmt(g['risk30'])])+' |')
g=diag['groups']['correlated']['observations/mutable']
extra+=['',f'ConPath Brier on observation-mutable events: {fmt(g["brier"])}. The known-immutable subset is 750/1545 events (unweighted), with zero ConPath error. Full-cohort and subset metrics have their own parent normalization.',f'Source: [{diagnostic_path}]({diagnostic_path}), SHA-256 `{sha(diagnostic_path)}`.','']
text='\n'.join(lines).replace('<!-- RETROSPECTIVE_METRICS -->','\n'.join(extra))+'\n'
(ROOT/'PAPER_TABLES.md').write_text(text)
print('Rendered PAPER_TABLES.md from four saved JSON reports')
