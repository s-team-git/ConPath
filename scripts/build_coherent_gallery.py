#!/usr/bin/env python3
"""Add the candidate's saved worlds to the ten pre-existing pilot cases.

CPU only: no model inference, raw archives, new cases, or query selection.
Publication is a separate step after scripts/audit_coherent_gallery.py passes.
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import build_site_visuals as draw
from scripts.evaluate_flatlands_support_clamped import _accelerated_events

METHOD, SEED, K = 'coherent_categorical', 20260910, 32
SCOPE_ZH = '此地点未参与训练或检查点选优；其验证反馈已用于选择改进方向，属于开发比较，非最终测试。'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    run = ROOT / 'results/coherent_parent_pilot_v1'
    baseline = ROOT / 'results/parent_group_pilot_v1'
    gallery_path = ROOT / 'site/data/parent_pilot_gallery_zh.json'
    gallery = json.loads(gallery_path.read_text())
    old_audit = json.loads((baseline / 'gallery_verification.json').read_text())
    analysis = json.loads((run / 'analysis.json').read_text())
    verification = json.loads((run / 'verification.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    require(verification['passed'] is True and verification['analysis_sha256'] == draw.sha(run / 'analysis.json'), 'Candidate verification missing or stale')
    require(analysis['final_test'] is False and analysis['training_seeds'] == [20260910, 20260911], 'Unexpected candidate scope')
    require(old_audit['passed'] is True and old_audit['gallery_sha256'] == draw.sha(gallery_path), 'Original gallery verification missing or stale')
    require(datetime.fromisoformat(old_audit['created_utc']) < datetime.fromisoformat(protocol['created_before_training_utc']), 'Original gallery was not verified before candidate training')
    require(gallery['source_analysis_sha256'] == analysis['baseline_analysis_sha256'], 'Candidate uses a different baseline')
    require(len(gallery['examples']) == 10 and gallery['checkpoint_seed'] == SEED, 'Wrong original gallery size/seed')
    data = baseline / 'data'
    require(draw.sha(data / 'seal.json') == protocol['data_seal_sha256'], 'Changed data seal')
    seal = json.loads((data / 'seal.json').read_text())
    folder = run / 'evaluation' / str(SEED)
    report = json.loads((folder / 'report.json').read_text())
    world_path = folder / 'worlds.npz'
    require(report['method'] == METHOD and report['seed'] == SEED and report['files']['worlds.npz'] == draw.sha(world_path), 'Wrong candidate world source')
    with np.load(world_path, allow_pickle=False) as archive:
        ids, world_array = archive['global_ids'].tolist(), archive['worlds']
    require(len(set(ids)) == 40 and world_array.shape == (40, K, 256, 256) and world_array.dtype == bool, 'Malformed K32 saved worlds')
    (ROOT / 'site/assets/zh/coherent-pilot').mkdir(parents=True, exist_ok=True)
    cases, assets = [], []
    for original in gallery['examples']:
        case = copy.deepcopy(original)
        gid = case['global_id']
        require(case['seed'] == SEED and case['samples'] == K and case['radius_cells'] == 10 and case['cohort'] == 'new_pilot', 'Unexpected original case contract')
        packet_path = data / 'packets' / (gid + '.npz')
        require(draw.sha(packet_path) == seal['files']['packets/' + gid + '.npz'], 'Changed cached development packet')
        with np.load(packet_path, allow_pickle=False) as archive:
            observation, valid = archive['observation'], archive['valid']
            indices, starts, goals = archive['candidate_indices'], archive['starts'], archive['goals']
        require(int(indices[0]) == case['candidate_index'] and starts[0].tolist() == case['start'] and goals[0].tolist() == case['goal'], 'Gallery query differs from frozen input-first query')
        worlds = world_array[ids.index(gid)]
        events = _accelerated_events(worlds, starts[:1], goals[:1], (10,))[:, 0, 0]
        case['event_probability'][METHOD] = float(events.mean())
        case['displayed_world_event'][METHOD] = bool(events[0])
        case['candidate_model_available'] = True
        case['candidate_scope_zh'] = SCOPE_ZH
        case['validation_reused_for_model_development'] = True
        for panel, pixels, title, probability in [
            (METHOD + '_sample', draw.categorical(observation, valid, worlds[0]), 'ConPath 改进版 · 首次补全', False),
            (METHOD, draw.probability(worlds.mean(0), valid), 'ConPath 改进版 · 逐格概率', True),
        ]:
            path = draw.map_svg('coherent-pilot/' + gid + '-' + panel, pixels, title,
                                subtitle='空间相关采样 · 开发验证（非最终测试）',
                                points=[case['start'], case['goal']], ramp=probability)
            case['panels'][panel] = path
            assets.append({'path': path, 'sha256': draw.sha(ROOT / 'site' / path)})
        cases.append(case)
    result = {
        'examples': cases, 'assets': assets, 'checkpoint_seed': SEED,
        'training_seeds': [20260910, 20260911], 'samples': K,
        'source_analysis_sha256': draw.sha(run / 'analysis.json'),
        'source_baseline_gallery_sha256': draw.sha(gallery_path),
        'source_baseline_gallery_verification_sha256': draw.sha(baseline / 'gallery_verification.json'),
        'saved_worlds': {METHOD: {'path': str(world_path.relative_to(ROOT)), 'sha256': draw.sha(world_path)}},
        'candidate_scope_zh': SCOPE_ZH,
        'validation_reused_for_model_development': True, 'final_test': False,
        'selection': 'Original ten baseline gallery cases, same order, first input-hash query, first actual world; no candidate-based case selection.',
        'new_archive_images_read': 0, 'model_inference_run': False,
        'builder_sha256': draw.sha(Path(__file__)),
    }
    output = ROOT / 'site/data/coherent_pilot_gallery_zh.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cases': len(cases), 'new_svg_panels': len(assets), 'gallery_sha256': draw.sha(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
