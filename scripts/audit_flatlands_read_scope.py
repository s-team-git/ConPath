#!/usr/bin/env python3
"""Metadata/saved-output audit correcting historical physical-test access claims.

Does not open archive PNGs. The original replay loader selected provenance_split,
not archive_split; therefore the old 'test_evaluated=false' receipts do not prove
physical-test images were untouched. Preserve those receipts with this erratum.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pathrel.data_access import require_development_packet


def main():
    selection = ROOT/'results/p1_flatlands_query_audit_bounded/selected_observations.csv'
    rows = list(csv.DictReader(selection.open()))
    archive_test = [r for r in rows if r['archive_split'] == 'test']
    old_train = [r for r in archive_test if r['provenance_split'] == 'train']
    old_val = [r for r in archive_test if r['provenance_split'] == 'validation']
    assert len(old_train) == 8 and len(old_val) == 5 and len(archive_test) == 53
    expected = {r['global_id'] for r in rows if r['provenance_split'] == 'validation'}
    runs = json.loads((ROOT/'results/current_baseline_k4_v1/report.json').read_text())['runs']
    for run in runs:
        with np.load(ROOT/'results/current_baseline_k4_v1'/run['method']/str(run['seed'])/'worlds.npz', allow_pickle=False) as saved:
            assert set(saved['global_ids']) == expected
    query_report = json.loads((ROOT/'results/p1_flatlands_query_audit_bounded/report.json').read_text())
    assert query_report['protocol']['gate_scope'] == 'every provenance validation/test source stratum, including ScanNet++ OOD'
    candidate_path = ROOT/'results/flatlands_parent_groups_v2/development_candidate.csv'
    candidates = list(csv.DictReader(candidate_path.open()))
    for r in candidates:
        require_development_packet(r)
    sources = [selection, candidate_path, ROOT/'src/pathrel/data_access.py', Path(__file__),
               ROOT/'src/pathrel/flatlands_data.py', ROOT/'scripts/audit_flatlands_queries.py',
               ROOT/'results/p1_flatlands_query_audit_bounded/report.json']
    report = {'erratum': True, 'this_audit_reads_images': False, 'historical_physical_test_untouched_claim_withdrawn': True,
              'current_k4_replay_physical_test_observations_opened': 5,
              'prior_development_training_physical_test_observations': old_train,
              'current_and_prior_validation_physical_test_observations': old_val,
              'historical_bounded_query_audit_physical_test_observations': 53,
              'historical_bounded_query_audit_scannetpp_observations': sum(r['source_dataset']=='ScanNet++' for r in archive_test),
              'known_image_audited_physical_test_global_ids': sorted(r['global_id'] for r in archive_test),
              'known_image_audited_scannetpp_scene_ids': sorted({r['scene_id'] for r in archive_test if r['source_dataset']=='ScanNet++'}),
              'safe_to_replay_old_validation_under_physical_test_lock': False,
              'candidate_development_access_checks_passed': len(candidates),
              'interpretation': 'Old provenance-test model evaluation was not run, but physical-test packet images were read in old development and prior query audits. Current K4 replay also read5 physical-test observations. No untouched final-test claim is supported by the old flags; do not erase or reinterpret the files to claim it. Stop further archive-test PNG reads. Audit all prior access and exclude already-inspected parent groups before designating a new final holdout; ScanNet++ is not wholly untouched.',
              'invalid_scope_flags_in': ['results/current_baseline_k4_v1/report.json:test_evaluated',
                  'results/baseline_review_20260909_v1/verification.json:test_images_opened',
                  'site/data/current_baseline_k4_analysis.json:test_evaluated'],
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    path = ROOT/'results/baseline_review_20260909_v1/read_scope_erratum.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (ROOT/'site/data/flatlands_read_scope_erratum.json').write_bytes(path.read_bytes())
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(list,dict))},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
