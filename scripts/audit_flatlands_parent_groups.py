#!/usr/bin/env python3
"""Metadata-only physical-place leakage audit and new development candidate."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pathrel.parent_groups import parent_group, grouped_development_assignment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'results/flatlands_parent_groups_v2'
    if out.exists():
        raise FileExistsError('Versioned audit is immutable; choose a new output version')
    sources = ROOT / 'results/baseline_review_20260909_v1/parent_sources'
    manifest = ROOT / 'results/p1_flatlands_provenance_manifest/provenance_manifest.csv'
    selected = ROOT / 'results/p1_flatlands_query_audit_bounded/selected_observations.csv'
    arkit = {}
    for row in csv.DictReader((sources / 'arkit_splits.csv').open()):
        if row['visit_id'] != 'NA':
            assert row['video_id'] not in arkit or arkit[row['video_id']] == row['visit_id']
            arkit[row['video_id']] = row['visit_id']
    rscan = {}
    for row in json.loads((sources / '3RScan.json').read_text()):
        for sid in [row['reference']] + [s['reference'] for s in row['scans']]:
            assert sid not in rscan or rscan[sid] == row['reference']
            rscan[sid] = row['reference']
    def resolve(row):
        return parent_group(row['source_dataset'], row['scene_id'], arkit_visits=arkit, rscan_references=rscan)
    rows = list(csv.DictReader(manifest.open()))
    old = [r for r in csv.DictReader(selected.open()) if r['provenance_split'] in ('train', 'validation')]
    assert len(old) == 320
    for row in rows + old:
        row['parent_group'] = resolve(row)
    source_names = sorted({r['source_dataset'] for r in rows})
    def comparison(records, field, left, right):
        a = {r['parent_group'] for r in records if r[field] == left and r['parent_group']}
        b = {r['parent_group'] for r in records if r[field] == right and r['parent_group']}
        overlap = a & b
        affected = [r for r in records if r[field] == right and r['parent_group'] in overlap]
        return {'left_groups': len(a), 'right_groups': len(b), 'overlap_group_count': len(overlap),
                'overlap_groups': sorted(overlap), 'affected_right_observations': len(affected),
                'affected_right_global_ids': sorted(r['global_id'] for r in affected)}
    audit = {}
    for name in source_names:
        full = [r for r in rows if r['source_dataset'] == name]
        bounded = [r for r in old if r['source_dataset'] == name]
        audit[name] = {'observations': len(full), 'unresolved_observations': sum(r['parent_group'] is None for r in full),
                      'old_bounded': comparison(bounded, 'provenance_split', 'train', 'validation'),
                      'provenance_train_validation': comparison(full, 'provenance_split', 'train', 'validation'),
                      'physical_train_test': comparison(full, 'archive_split', 'train', 'test')}
    eligible = [r for r in rows if r['archive_split'] == 'train' and r['parent_group'] is not None]
    assignments = grouped_development_assignment(r['parent_group'] for r in eligible)
    original_test_groups = {r['parent_group'] for r in rows if r['archive_split'] == 'test' and r['parent_group']}
    candidate = [dict(r, candidate_split=assignments[r['parent_group']],
                      parent_present_in_locked_physical_test=r['parent_group'] in original_test_groups) for r in eligible]
    groups_by_split = {s: {r['parent_group'] for r in candidate if r['candidate_split'] == s}
                       for s in ('train', 'calibration', 'validation')}
    assert not (groups_by_split['train'] & groups_by_split['validation'])
    assert not (groups_by_split['train'] & groups_by_split['calibration'])
    assert not (groups_by_split['calibration'] & groups_by_split['validation'])
    out.mkdir(parents=True)
    def write_csv(path, records):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_csv(out / 'development_candidate.csv', candidate)
    write_csv(out / 'old_train_validation_parents.csv', old)
    quarantine = [r for r in rows if r['archive_split'] == 'train' and r['parent_group'] is None]
    if quarantine:
        write_csv(out / 'unresolved_train_quarantine.csv', quarantine)
    total_affected = sum(a['old_bounded']['affected_right_observations'] for a in audit.values())
    evidence_files = [manifest, selected, sources / 'arkit_splits.csv', sources / '3RScan.json',
                      ROOT / 'src/pathrel/parent_groups.py', Path(__file__), out / 'development_candidate.csv',
                      out / 'old_train_validation_parents.csv']
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'metadata_only': True,
              'test_images_opened': False, 'training_performed': False,
              'old_physical_place_isolation_passed': False, 'old_validation_observations': 160,
              'old_validation_observations_with_training_parent': total_affected,
              'old_validation_affected_fraction': total_affected / 160,
              'old_selected_unresolved': sum(r['parent_group'] is None for r in old),
              'per_source': audit, 'candidate': {
                  'status': 'development_metadata_candidate_only', 'formal_training_ready': False,
                  'uses_physical_train_assets_only': True, 'uses_old_checkpoints': False,
                  'parent_overlap_across_candidate_splits': 0, 'observation_count': len(candidate),
                  'quarantined_train_observations': len(quarantine),
                  'split_counts': {s: {'observations': sum(r['candidate_split'] == s for r in candidate),
                                       'parent_groups': len(groups_by_split[s]),
                                       'per_source_observations': dict(Counter(r['source_dataset'] for r in candidate if r['candidate_split'] == s))}
                                   for s in groups_by_split},
                  'training_parents_present_in_original_locked_physical_test': len(groups_by_split['train'] & original_test_groups),
                  'final_test_warning': 'Original physical ID test shares parents with development data. It cannot establish unseen-place generalization. ScanNet++ remains a locked separate-source OOD candidate, not an evaluated result.',
                  'remaining_gates': ['target-blind queries and support/quality audit', 'physical scale clarification',
                                      'independent final-test design and cross-source duplication audit',
                                      'matched training recipes and data scale frozen before fresh training']},
              'interpretation': 'Old numerics remain cohort diagnostics, not unseen-building evidence. Existing scene bootstrap uses subscene IDs, not independent parent clusters. Do not discard overlapping old validation rows and reuse selected checkpoints as a fresh holdout.',
              'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in evidence_files},
              'parent_metadata_sources': json.loads((sources / 'sources.json').read_text())}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    # Public compact audit omits 270k observation records, retains old-cohort IDs and hashes.
    public = {**report, 'per_source': {k: {**v, 'provenance_train_validation': {x:y for x,y in v['provenance_train_validation'].items() if x not in ('overlap_groups','affected_right_global_ids')},
                                              'physical_train_test': {x:y for x,y in v['physical_train_test'].items() if x not in ('overlap_groups','affected_right_global_ids')}} for k,v in audit.items()}}
    (ROOT / 'site/data/flatlands_parent_group_audit.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'affected_old_val': total_affected, 'per_source': {k:v['old_bounded']['affected_right_observations'] for k,v in audit.items()}, 'candidate': report['candidate']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
