#!/usr/bin/env python3
"""Audit candidate gallery pixels, annotations, selection, and all 320 event votes.

Only cached development packets and previously saved worlds/CSVs are read.
The independent baseline auditor supplies explicit disk erosion and RGB oracles;
no production renderer, model, checkpoint, or event scorer is imported.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from scripts.audit_parent_pilot_gallery import (
    audit_gallery as audit_baseline_gallery, audit_svg, disk_event, expected_pixels, require, sha,
)

METHOD, SEED, K = 'coherent_categorical', 20260910, 32
SCOPE_ZH = '此地点未参与训练或检查点选优；其验证反馈已用于选择改进方向，属于开发比较，非最终测试。'


def audit_case_extension(case, original):
    """Reject any change to original case metadata or original model panels/votes."""
    restored = copy.deepcopy(case)
    require(restored.pop('candidate_model_available', None) is True, 'Candidate availability flag missing')
    require(restored.pop('candidate_scope_zh', None) == SCOPE_ZH, 'Candidate development scope missing')
    require(restored.pop('validation_reused_for_model_development', None) is True, 'Reused validation must be disclosed')
    require(set(restored['panels']) == set(original['panels']) | {METHOD, METHOD + '_sample'}, 'Candidate panel keys differ')
    for panel in (METHOD, METHOD + '_sample'):
        expected = f'assets/zh/coherent-pilot/{original["global_id"]}-{panel}.svg'
        require(restored['panels'].pop(panel) == expected, 'Candidate panel identifies a different case')
    for key in ('event_probability', 'displayed_world_event'):
        require(set(restored[key]) == set(original[key]) | {METHOD}, 'Candidate event keys differ')
        restored[key].pop(METHOD)
    require(restored == original, 'Original case metadata, input/reference, query, selection, or predictions were changed')


def audit_gallery(root=ROOT):
    root = Path(root).resolve()
    run, baseline, site = root/'results/coherent_parent_pilot_v1', root/'results/parent_group_pilot_v1', root/'site'
    hashes = {}
    def checked(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(root), 'Audit source escaped project root')
        checksum = sha(path)
        require(expected is None or checksum == expected, f'Checksum mismatch: {path}')
        hashes[str(path.relative_to(root))] = checksum
        return path
    def read_json(path, expected=None):
        return json.loads(checked(path, expected).read_text())
    analysis = read_json(run/'analysis.json')
    analysis_sha = sha(run/'analysis.json')
    verification = read_json(run/'verification.json')
    require(verification['passed'] is True and verification['analysis_sha256'] == analysis_sha, 'Candidate result verification missing or stale')
    require(analysis['training_seeds'] == [20260910, 20260911] and analysis['final_test'] is False and analysis['validation_reused_for_model_development'] is True, 'Wrong candidate result scope')
    protocol = read_json(run/'protocol.json', analysis['protocol_sha256'])
    old_gallery_path, gallery_path = site/'data/parent_pilot_gallery_zh.json', site/'data/coherent_pilot_gallery_zh.json'
    original = read_json(old_gallery_path)
    gallery = read_json(gallery_path)
    old_receipt = read_json(baseline/'gallery_verification.json', gallery['source_baseline_gallery_verification_sha256'])
    require(old_receipt['passed'] is True and old_receipt['gallery_sha256'] == sha(old_gallery_path), 'Original gallery receipt missing or stale')
    require(datetime.fromisoformat(old_receipt['created_utc']) < datetime.fromisoformat(protocol['created_before_training_utc']), 'Original selection was not verified before candidate training')
    # Reconstruct the old hash selection, input-only queries, labels, and all old
    # panels from independent oracles rather than trusting copied JSON fields.
    old_recheck = audit_baseline_gallery(root)
    require(old_recheck['passed'] is True and old_recheck['gallery_sha256'] == old_receipt['gallery_sha256'], 'Baseline gallery recheck failed')
    hashes.update(old_recheck['source_hashes'])
    require(gallery['source_analysis_sha256'] == analysis_sha and gallery['source_baseline_gallery_sha256'] == sha(old_gallery_path), 'Gallery sources differ')
    require(original['source_analysis_sha256'] == analysis['baseline_analysis_sha256'], 'Candidate baseline source differs')
    require(gallery['checkpoint_seed'] == SEED and gallery['training_seeds'] == [20260910, 20260911] and gallery['samples'] == K, 'Gallery seed or sample budget differs')
    require(gallery['final_test'] is False and gallery['validation_reused_for_model_development'] is True and gallery['candidate_scope_zh'] == SCOPE_ZH, 'Gallery scope is misleading')
    require(gallery['new_archive_images_read'] == 0 and gallery['model_inference_run'] is False, 'Gallery accessed new archive images or inference')
    require(len(gallery['examples']) == len(original['examples']) == 10, 'Wrong gallery case count')
    for case, original_case in zip(gallery['examples'], original['examples']):
        audit_case_extension(case, original_case)
    assets = {entry['path']: entry['sha256'] for entry in gallery['assets']}
    expected_assets = {case['panels'][panel] for case in gallery['examples'] for panel in (METHOD, METHOD+'_sample')}
    require(len(gallery['assets']) == len(assets) == 20 and set(assets) == expected_assets, 'New asset manifest is not exactly the 20 candidate panels')
    require(set(gallery['saved_worlds']) == {METHOD}, 'Unexpected candidate saved worlds')
    reference = gallery['saved_worlds'][METHOD]
    folder = run/'evaluation'/str(SEED)
    require(reference['path'] == str((folder/'worlds.npz').relative_to(root)), 'Candidate worlds identify a different seed')
    report = read_json(folder/'report.json')
    require(report['seed'] == SEED and report['method'] == METHOD and report['files']['worlds.npz'] == reference['sha256'], 'Candidate evaluated source differs')
    with np.load(checked(folder/'worlds.npz', reference['sha256']), allow_pickle=False) as archive:
        ids, worlds = archive['global_ids'].tolist(), archive['worlds']
    data = baseline/'data'
    seal = read_json(data/'seal.json', protocol['data_seal_sha256'])
    with checked(data/'selected.csv', seal['files']['selected.csv']).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = [row['global_id'] for row in rows if row['candidate_split'] == 'validation']
    require(ids == expected_ids and len(set(ids)) == 40 and worlds.dtype == bool and worlds.shape == (40, K, 256, 256), 'Candidate K32 saved-world identity/order/type differs')
    with checked(folder/'predictions_k32.csv', report['files']['predictions_k32.csv']).open(newline='') as handle:
        predictions = list(csv.DictReader(handle))
    prediction_map = {(entry['global_id'], int(entry['candidate_index']), int(entry['radius_cells'])): entry for entry in predictions}
    require(len(prediction_map) == len(predictions) == 1545, 'Unexpected or duplicate CSV events')
    panel_receipts, case_receipts = [], []
    for case in gallery['examples']:
        gid, start, goal = case['global_id'], case['start'], case['goal']
        packet_path = data/'packets'/(gid+'.npz')
        with np.load(checked(packet_path, seal['files']['packets/'+gid+'.npz']), allow_pickle=False) as archive:
            observation, valid, hidden, target = [archive[key] for key in ('observation', 'valid', 'hidden', 'target')]
        w = worlds[ids.index(gid)]
        require(not w[:, ~valid].any() and np.array_equal(w[:, ~hidden], np.broadcast_to(observation[0, ~hidden].astype(bool), w[:, ~hidden].shape)), f'{gid}: support or observed evidence changed')
        outcomes = [disk_event(world, start, goal, 10) for world in w]
        probability = sum(outcomes)/K
        require(type(case['event_probability'][METHOD]) in (int, float) and case['event_probability'][METHOD] == probability, f'{gid}: gallery probability differs from K32 disk votes')
        require(type(case['displayed_world_event'][METHOD]) is bool and case['displayed_world_event'][METHOD] == outcomes[0], f'{gid}: displayed event is not world zero')
        require(disk_event(target, start, goal, 10) == case['target'], f'{gid}: reference event differs')
        entry = prediction_map[(gid, case['candidate_index'], 10)]
        require(entry['parent_group'] == case['parent_group'] and float(entry['probability']) == probability, f'{gid}: evaluated CSV differs from all world votes')
        for panel, pixels, title, is_probability in [
            (METHOD+'_sample', expected_pixels(observation, valid, free_map=w[0]), 'ConPath 改进版 · 首次补全', False),
            (METHOD, expected_pixels(observation, valid, probabilities=w.sum(0, dtype=np.int64)/K), 'ConPath 改进版 · 逐格概率', True),
        ]:
            relative = case['panels'][panel]
            path = checked(site/relative, assets[relative])
            panel_receipts.append({'path': relative, 'case': gid, 'panel': panel,
                                   **audit_svg(path, pixels, start, goal, probability=is_probability, title=title,
                                               subtitle='空间相关采样 · 开发验证（非最终测试）')})
        case_receipts.append({'global_id': gid, 'candidate_index': case['candidate_index'], 'radius_cells': 10,
                              'positive_worlds': sum(outcomes), 'worlds': K, 'probability': probability,
                              'first_world_event': outcomes[0], 'reference_event': case['target']})
    checked(root/'scripts/audit_parent_pilot_gallery.py')
    checked(root/'scripts/build_coherent_gallery.py', gallery['builder_sha256'])
    return {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'analysis_sha256': analysis_sha, 'gallery_sha256': sha(gallery_path),
            'source_baseline_gallery_sha256': sha(old_gallery_path), 'script_sha256': sha(Path(__file__)),
            'cases_checked': 10, 'svg_bitmaps_checked': 20, 'rgb_pixels_checked': 20*256*256,
            'world_event_checks': 10*K, 'reference_event_checks': 10,
            'baseline_svg_bitmaps_rechecked': old_recheck['svg_bitmaps_checked'],
            'baseline_world_event_rechecks': old_recheck['world_event_checks'],
            'same_pre_candidate_cases_and_queries': True, 'first_actual_world': True,
            'independent_pixel_oracle': True, 'explicit_disk_four_neighbor_oracle': True,
            'cached_development_packets_only': True, 'cached_packets_read': 10,
            'new_archive_images_read': 0, 'model_inference_run': False, 'gpu_used': False,
            'validation_reused_for_model_development': True, 'final_test': False,
            'source_hashes': hashes, 'cases': case_receipts, 'panels': panel_receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    receipt = args.root.resolve()/'results/coherent_parent_pilot_v1/gallery_verification.json'
    try:
        result = audit_gallery(args.root)
    except Exception as error:
        result = {'passed': False, 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'script_sha256': sha(Path(__file__)), 'error': f'{type(error).__name__}: {error}'}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(receipt)
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_hashes', 'cases', 'panels')}, ensure_ascii=False))
    if not result['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
