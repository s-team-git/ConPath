#!/usr/bin/env python3
"""Metadata/saved-summary claim audit. Never opens image, label, model or test assets."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/paper_submission_v1/review'
READS = {}


def read(path):
    assert Path(path).suffix in {'.md', '.json', '.csv', '.jsonl'}
    assert '/packets/' not in path and '/location_6/' not in path
    b = (ROOT / path).read_bytes()
    READS[path] = {'path': path, 'sha256': hashlib.sha256(b).hexdigest(),
                   'bytes': len(b), 'scope': 'existing manuscript, aggregate result, or access/split metadata'}
    return b.decode('utf-8')


def js(path):
    return json.loads(read(path))


def rows(path):
    return list(csv.DictReader(io.StringIO(read(path))))


def write_json(name, obj):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


docs = {p: read(p) for p in ['PAPER_DRAFT.md', 'PAPER_EVIDENCE.md', 'PAPER_TABLES.md',
                            'results/FINAL_MODEL_SELECTION.md']}
snapshot = js('results/paper_validation_snapshot.json')
erratum = js('site/data/flatlands_read_scope_erratum.json')
parent_audit = js('site/data/flatlands_parent_group_audit.json')
protocol = js('results/parent_group_pilot_v1/data/protocol.json')
current_rows = rows('results/parent_group_pilot_v1/data/selected.csv')
old_rows = rows('results/flatlands_parent_groups_v2/old_train_validation_parents.csv')
query_audit = rows('results/p1_flatlands_query_audit_bounded/selected_observations.csv')
external_initial = rows('results/flatlands_external_formal_protocol_v1/data/selected.csv')
external_eligible = rows('results/flatlands_external_formal_protocol_v1/data_eligible_v1/selected.csv')
access = [json.loads(line) for line in read(
    'results/flatlands_external_formal_protocol_v1/data/image_access.jsonl').splitlines() if line]

created = datetime.now(timezone.utc).isoformat()
scope = dict(new_training_runs=0, new_model_inference_runs=0, prediction_rows_re_evaluated=0,
             raw_or_saved_map_assets_opened=0, checkpoint_files_opened=0,
             final_test_assets_opened=0, location_6_assets_opened=0,
             new_holdout_created=False, test_unlocked=False,
             complete_189_source_audit_repeated=False)

# Check only values used in principal claims; no rereading all source files.
numeric_checks = []
for i in [1, 3, 4, 8, 10, 12, 16, 17, 20, 22, 24, 25, 26]:
    r = snapshot['evidence_rows'][i]
    for metric, entry in r['metrics'].items():
        finite = [v for v in entry['values'] if v is not None]
        mean = statistics.mean(finite) if finite else None
        sd = statistics.stdev(finite) if len(finite) > 1 else None
        assert entry['available_repeats'] == len(finite)
        assert mean is None and entry['mean'] is None or abs(mean - entry['mean']) < 2e-12
        assert sd is None and entry['sample_sd'] is None or abs(sd - entry['sample_sd']) < 2e-12
        numeric_checks.append({'source_pointer': f'/evidence_rows/{i}/metrics/{metric}',
                               'cohort': r['cohort'], 'method': r['method'], 'K': r['K'],
                               'mean': mean, 'sample_sd': sd, 'n_training_seeds_or_fixed_rules': len(finite),
                               'mean_and_sample_sd_recomputed_from_saved_seed_values': True})

shown = [(1, 'brier', '0.10380'), (3, 'brier', '0.11201'), (4, 'brier', '0.11152'),
         (1, 'nll', '0.79330'), (1, 'ece', '0.08622'), (3, 'nll', '0.94364'),
         (3, 'ece', '0.10131'), (1, 'false_safe_at_0_8', '0.18649'),
         (1, 'coverage_at_0_8', '0.21340'), (3, 'false_safe_at_0_8', '0.20847'),
         (3, 'coverage_at_0_8', '0.22453'), (8, 'nll', '0.48499'), (8, 'ece', '0.05653'),
         (16, 'brier', '0.06749'), (24, 'brier', '0.16139'), (22, 'brier', '0.06957'),
         (25, 'brier', '0.20425'), (26, 'brier', '0.09499'), (20, 'ece', '0.04076')]
display_checks = []
for i, metric, literal in shown:
    value = snapshot['evidence_rows'][i]['metrics'][metric]['mean']
    assert f'{value:.5f}' == literal
    assert literal in docs['PAPER_DRAFT.md']
    display_checks.append({'source_pointer': f'/evidence_rows/{i}/metrics/{metric}/mean',
                           'exact': value, 'manuscript_literal': literal, 'passed': True})

full, ind, det, prior = [snapshot['evidence_rows'][i] for i in [1, 3, 4, 8]]
assert all(a < b for a, b in zip(full['metrics']['brier']['values'], ind['metrics']['brier']['values']))
assert det['continuous_map_brier']['mean'] < full['continuous_map_brier']['mean']
assert prior['metrics']['nll']['mean'] < full['metrics']['nll']['mean']
assert prior['metrics']['ece']['mean'] < full['metrics']['ece']['mean']
assert prior['metrics']['false_safe_at_0_8']['mean'] is None
assert prior['metrics']['coverage_at_0_8']['mean'] == 0
for missing in ['direct_query', 'no_event', 'no_global']:
    assert not any(r['cohort'] == 'parent_isolated_three_seed' and r['method'] == missing
                   for r in snapshot['evidence_rows'])

interval = snapshot['paired_intervals']['parent_three_seed']
assert abs(interval['independent']['other_minus_conpath'] -
           (ind['metrics']['brier']['mean'] - full['metrics']['brier']['mean'])) < 2e-12
assert interval['independent']['ci95'][0] > 0
assert interval['deterministic']['ci95'][0] < 0 < interval['deterministic']['ci95'][1]
radius_checks = []
for method, budget in [('correlated', 32), ('deterministic', 1)]:
    rr = [(i, r) for i, r in enumerate(snapshot['development_run_metrics'])
          if r['cohort'] == 'parent_isolated_three_seed' and r['method'] == method and r['K'] == budget]
    assert len(rr) == 3
    for radius in ['0', '10', '20']:
        result = {'method': method, 'K': budget, 'radius_cells': int(radius), 'seeds': [r['seed'] for _, r in rr],
                  'source_pointers': [f'/development_run_metrics/{i}/by_radius/{radius}' for i, _ in rr]}
        for metric in ['brier', 'false_safe_at_0_8', 'coverage_at_0_8']:
            vv = [r['by_radius'][radius][metric] for _, r in rr]
            finite = [v for v in vv if v is not None]
            result[metric] = {'mean': statistics.mean(finite) if finite else None, 'values': vv,
                              'available_repeats': len(finite)}
        radius_checks.append(result)
full_r20 = next(r for r in radius_checks if r['method'] == 'correlated' and r['radius_cells'] == 20)
det_r20 = next(r for r in radius_checks if r['method'] == 'deterministic' and r['radius_cells'] == 20)
assert full_r20['coverage_at_0_8']['values'] == [0.0] * 3
assert full_r20['false_safe_at_0_8']['values'] == [None] * 3
assert full_r20['brier']['mean'] > det_r20['brier']['mean']

# Access histories are evidence of access, never proof of absence of unlogged access.
known_test = set(erratum['known_image_audited_physical_test_global_ids'])
old_test_train = {r['global_id'] for r in erratum['prior_development_training_physical_test_observations']}
old_test_val = {r['global_id'] for r in erratum['current_and_prior_validation_physical_test_observations']}
assert old_test_train <= known_test and old_test_val <= known_test
assert {r['global_id'] for r in query_audit if r['archive_split'] == 'test'} == known_test
assert len(known_test) == 53
assert {x['physical_split'] for x in access} == {'train'}
assert {r['global_id'] for r in external_initial} == {e['global_id'] for e in access}
assert all(r['archive_split'] == 'train' for r in current_rows + external_initial + external_eligible)

exposures = defaultdict(lambda: {'observations': set(), 'sources': set()})
exposure_inputs = [('historical_320_selected', old_rows), ('current_165_selected', current_rows),
                   ('external_2292_access_logged', external_initial)]
for label, rr in exposure_inputs:
    for r in rr:
        exposures[r['parent_group']]['observations'].add(r['global_id'])
        exposures[r['parent_group']]['sources'].add(label)
parent_output = OUT / 'known_development_exposure_parents.csv'
if parent_output.exists():
    raise FileExistsError(parent_output)
with parent_output.open('w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['parent_group', 'known_observation_count_in_listed_manifests', 'metadata_sources', 'eligible_holdout'])
    for parent, e in sorted(exposures.items()):
        w.writerow([parent, len(e['observations']), ';'.join(sorted(e['sources'])), 'not_assessed'])

current_splits = {k: {r['parent_group'] for r in current_rows if r['candidate_split'] == k}
                  for k in ['train', 'calibration', 'validation']}
assert [len(current_splits[k]) for k in ['train', 'calibration', 'validation']] == [100, 25, 40]
assert not (current_splits['train'] & current_splits['calibration'] or
            current_splits['train'] & current_splits['validation'] or
            current_splits['calibration'] & current_splits['validation'])
initial_ids = {r['global_id'] for r in external_initial}
eligible_ids = {r['global_id'] for r in external_eligible}
initial_parents = {r['parent_group'] for r in external_initial}
eligible_parents = {r['parent_group'] for r in external_eligible}

missing_metadata = [
    {'id': 'H1', 'item': 'Complete session-spanning access inventory',
     'why': 'Old errata and one newer append-only ledger establish positive access, but no complete proof covers every earlier script, preview, cached target, website image and agent session.'},
    {'id': 'H2', 'item': 'Canonical parent identity for every accessed observation and every proposed holdout observation',
     'why': 'The older 512-observation query-audit manifest has no parent_group column; known corrected groups do not complete all access mappings. No candidate holdout manifest is certified here.'},
    {'id': 'H3', 'item': 'Cross-source identity / duplicate exclusion metadata with unresolved cases quarantined',
     'why': 'Current exact D4 checks certify only the selected development splits, not all unseen candidate assets; aggregate parent audit records unresolved identities.'},
    {'id': 'H4', 'item': 'Parent-level closure over train, selection, validation, model-development and quality-screen accesses',
     'why': 'Physical-test directory membership is insufficient; excluded quality-screened observations remain inspected and cannot be restored to untouched status.'},
    {'id': 'H5', 'item': 'A separately authorized, frozen final-evaluation manifest and evaluator decision',
     'why': 'No new final test, threshold, checkpoint selection, query or holdout is authorized by this writing step.'},
]
holdout = {
    'created_utc': created, 'scope': scope, 'audit_type': 'metadata-only eligibility gap analysis',
    'eligible_holdout_proven': False, 'eligible_holdout_count': None,
    'meaning': 'Not proven does not mean no eligible holdout exists. No candidate set is selected or opened.',
    'current_selected_split': {'counts': {k: len(v) for k, v in current_splits.items()},
                               'cross_split_parent_overlap': 0, 'physical_archive': 'train',
                               'final_test_eligible': False, 'reason': 'All selected parents are existing development data.'},
    'known_historical_physical_test_access': {'unique_observations': len(known_test),
        'old_training_observations': len(old_test_train), 'old_validation_observations': len(old_test_val),
        'eight_and_five_are_subsets_of_fifty_three': True,
        'known_scannetpp_scenes_inspected': len(erratum['known_image_audited_scannetpp_scene_ids'])},
    'archive_parent_identity_overlap': {
        source: data['physical_train_test'] for source, data in parent_audit['per_source'].items()},
    'archive_parent_overlap_interpretation': 'Across the five indoor sources, all physical-test parent IDs also occur in the physical-train archive. This is archive identity overlap, not a claim that every such parent was actually used for model training.',
    'newer_external_preparation_access': {'ledger_entries': len(access),
        'unique_observations': len(initial_ids), 'initial_parents': len(initial_parents),
        'eligible_observations_after_quality_quarantine': len(eligible_ids),
        'eligible_parents_after_quality_quarantine': len(eligible_parents),
        'removed_but_already_inspected_observations': len(initial_ids - eligible_ids),
        'removed_but_already_inspected_parents': len(initial_parents - eligible_parents),
        'all_ledger_members_physical_train': True,
        'cancellation_restores_untouched_status': False},
    'partial_parent_exposure_inventory': {'parents_in_union': len(exposures),
        'source_manifests': [x[0] for x in exposure_inputs], 'complete': False,
        'file': str(parent_output.relative_to(ROOT)),
        'scope': 'Union of three identified development exposure histories only. Not a final-test manifest or full exclusion certificate.'},
    'missing_metadata': missing_metadata,
    'decision': 'Keep test lock. Do not infer eligibility from old test_evaluated=false or from absent rows in this partial inventory.'}

uncertainty = {
    'seed_summary': 'Arithmetic mean and sample SD (ddof=1), not a confidence interval or SEM.',
    'current_parent_interval': {'resampling_unit': '40 parent places, stratified within five sources (eight validation parents each)',
                              'conditional_on': 'Displayed three-seed average; not a confidence statement over arbitrary future training seeds',
                              'registered_source': '/paired_intervals/parent_three_seed'},
    'historical_interval': 'Within-seed subscene bootstrap over 142 event-contributing subscenes; not independent-parent inference; 27/160 validation observations overlap training parents.',
    'candidate_interval': '40 paired parent places, conditional on exactly the matched two seeds; no substitution of three-seed original mean.',
    'event_units': '1545 correlated radius-conditioned events from 515 terminal queries over 40 validation parents; not 1545 independent scenes. K worlds are Monte Carlo draws, not experimental seeds.',
    'radius_units': 'Grid cells (0, 10, 20), never meters.',
    'false_safe_units': 'Conditional unreachable fraction among accepted event forecasts; distinct from FPR among all negatives, probability threshold, and risk at fixed coverage.',
    'risk_aggregation': 'Ratios formed under the existing equal-parent event weights per seed, then seed means; not a pooled ratio over all seeds.',
    'empty_acceptance': 'Undefined risk is JSON null / table dash, never zero.',
    'map_estimators': 'K-world cell vote Brier and continuous conditional-probability Brier are different estimators; current deterministic sigmoid Brier is the fair continuous map comparison.',
    'calibration_interpretation': 'Lower Brier is improved probabilistic event accuracy, not an isolated proof of better calibration. Ten-bin ECE and reliability are descriptive, and not conditional calibration guarantees.'}

claims = [
    {'rq': 'RQ1', 'claim': 'Spatial dependence carries path-event information beyond cell marginals',
     'verdict': 'Supported conditionally by historical finite-ensemble intervention; complementary current development comparison',
     'same_protocol_evidence': ['/evidence_rows/16', '/evidence_rows/24', '/fixed_marginal_mechanism',
                                '/evidence_rows/1', '/evidence_rows/3', '/paired_intervals/parent_three_seed/independent'],
     'evidence': 'Historical K128 cellwise world-index permutation exactly preserves empirical cell counts and raises Brier 0.06749 to 0.16139. Separately, current K32 ConPath 0.10380 versus independent 0.11201 in three matched seeds, paired parent difference 0.00821 [0.00439, 0.01251].',
     'counterevidence': ['Historical mean-map Brier 0.06957 is near ConPath 0.06749 and wins in one seed.',
                         'Current deterministic Brier interval versus ConPath crosses zero and its continuous map Brier is lower.',
                         'Current continuous hidden-map Brier 0.15147 vs 0.15158 is numerically close, but does not establish equal marginal distributions or statistical equivalence.'],
     'allowed_english': 'A fixed-marginal intervention on an archived development cohort shows that the alignment of sampled worlds contains useful path-event information beyond their empirical cell marginals. A separate parent-isolated development comparison favors ConPath over independently sampled completion.',
     'forbidden_extension': 'The current unseen-place experiment causally proves a dependence benefit at identical map marginals, or stochastic inference uniformly dominates deterministic maps.',
     'missing': ['A current parent-isolated fixed-marginal intervention is not archived.',
                 'An untouched final holdout is not certified; no new intervention or final testing is performed.']},
    {'rq': 'RQ2', 'claim': 'Reachability supervision and decoder global factors improve task-level scores',
     'verdict': 'Historical exploratory ablation support for Brier/NLL/ECE; robust false-safe claim not supported',
     'same_protocol_evidence': ['/evidence_rows/16', '/evidence_rows/25', '/evidence_rows/26', '/paired_intervals/historical'],
     'evidence': 'Historical full/no-event/no-global Brier 0.06749/0.20425/0.09499 with three seeds and common K128 evaluator. Historical full/no-event/no-global NLL 0.28854/2.53067/0.78486 and ECE 0.05715/0.21690/0.08592.',
     'counterevidence': ['No-event false-safe@0.8 is lower (0.02561 vs 0.04552), at lower coverage (0.19316 vs 0.33644).',
                         'Historical equal-30%-coverage paired risk intervals include zero.',
                         'Current radius prior has lower NLL/ECE than ConPath but zero coverage at confidence 0.8.',
                         'Historical direct-query ECE is lower, but only one seed is available.'],
     'allowed_english': 'Within the archived historical protocol, event supervision and low-rank decoder factors improve Event Brier, NLL, and ECE. The overlap-affected cohort and incomplete current ablation matrix limit causal and generalization claims; uniformly improved selective risk is not established.',
     'forbidden_extension': 'The current parent-isolated study proves every component necessary or proves better calibration and safety against direct-query and all map-only methods.',
     'missing': ['Current direct-query, no-event and no-global rows are absent.',
                 'No-event retains event-based checkpoint selection; no-global retains local noise and encoder context.',
                 'Three current repeats or adequate external convergence are not fabricated or newly trained.']},
    {'rq': 'RQ3', 'claim': 'Footprint-conditioned event forecasts remain reliable over difficult query settings',
     'verdict': 'Structural radius monotonicity established; conditional reliability only partially investigated',
     'same_protocol_evidence': ['/development_run_metrics', '/metric_contract', '/development_posthoc_diagnostics'],
     'evidence': 'The same saved worlds and exact nested disk erosion produce non-increasing reachability with radius 0/10/20 cells. Existing radius, source, truth-stratum, reliability and fixed-case figures remain descriptive.',
     'counterevidence': ['750/1545 current events are already forced unreachable; full ConPath Brier 0.10380 contrasts with mutable-subset Brier 0.19912 under separately normalized parent weights.',
                         'At radius 20 cells, ConPath has zero confidence-0.8 coverage in every seed, so false-safe is undefined. Its Brier 0.06142 is worse than deterministic 0.04025. Radius-10 false-safe/coverage means are 0.30851/0.03547.',
                         'No saved strict interior-bottleneck case exists among the ten fixed queries after requiring both endpoints to fit.',
                         'No archived registered occlusion sweep or current full/no-event/no-global qualitative worlds exists.'],
     'allowed_english': 'Shared-world footprint evaluation guarantees monotonic reachability forecasts as the disk radius increases. Existing development diagnostics assess reliability across recorded radii and query classes, but do not establish uniform calibration in occluded or narrow-bottleneck environments.',
     'forbidden_extension': 'Zero radius-monotonicity violations prove calibration, safe control, robust indoor navigation or outside-domain generalization.',
     'missing': ['Untouched holdout qualification.', 'Registered occlusion-stratified and strict bottleneck evidence.',
                 'No real robot, outdoor or control-trajectory claim is supported.']}
]

write_json('numeric_claim_checks.json', {'created_utc': created, 'scope': scope,
    'snapshot_sha256': READS['results/paper_validation_snapshot.json']['sha256'],
    'selected_summary_checks': numeric_checks, 'manuscript_literal_checks': display_checks,
    'all_passed': True, 'uncertainty_units': uncertainty, 'radius_counterexample_checks': radius_checks,
    'not_claimed': 'No re-audit of all 189 source files or rerun of saved-prediction scoring; assertions bind the snapshot and manuscript hashes recorded in read_manifest.json.'})
write_json('holdout_eligibility.json', holdout)
write_json('claim_audit.json', {'created_utc': created, 'scope': scope,
    'manuscript_sha256_at_review': READS['PAPER_DRAFT.md']['sha256'],
    'snapshot_sha256': READS['results/paper_validation_snapshot.json']['sha256'],
    'claims': claims, 'uncertainty_units': uncertainty,
    'no_critical_numeric_conflict_found_in_selected_claims': True,
    'required_editorial_constraints': ['Keep every numerical cohort separate.',
        'Use development validation rather than final-test language.',
        'Retain deterministic continuous-map, prior, historical mean-map and selective-risk counterexamples.',
        'Do not equate lower Brier with proof of calibrated probabilities.',
        'Do not label only the post-quarantine external subset as all prior development access.'],
    'holdout_eligibility_proven': False,
    'complete_submission_evidence': False})
write_json('read_manifest.json', {'created_utc': created, 'scope': scope,
    'explicit_input_files': list(READS.values()),
    'input_file_count': len(READS), 'script': str(Path(__file__).relative_to(ROOT)),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'no_model_or_geometry_function_executed': True,
    'authority': 'Existing evidence collection and manuscript claim review only; no dataset or training authorization.'})
print(json.dumps({'selected_summary_checks': len(numeric_checks), 'display_checks': len(display_checks),
                  'known_parent_exposure_lower_bound': len(exposures), 'holdout_eligible_proven': False,
                  'external_removed_but_inspected_parents': len(initial_parents - eligible_parents),
                  'outputs': 5, 'all_numeric_checks_passed': True}))
