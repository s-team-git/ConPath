#!/usr/bin/env python3
"""Fixed-order real checkpoint outputs; no best-of-K or outcome-based case picking."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.data_access import require_development_packet
from pathrel.flatlands_data import FlatLandsReplayDataset
from pathrel.flatlands_query import construct_natural_queries
from scripts import build_site_visuals as draw
from scripts.evaluate_flatlands_support_clamped import _accelerated_events, _load_model, _verify_accelerator
from scripts.render_unscenes3d_qualitative import _load_frame, _load_model as outdoor_model, _resolve_adapter

OUT = ROOT / 'results/model_gallery_20260909_v1'
ASSETS = ROOT / 'site/assets/zh/expanded-models'
SEED = 20260909
K = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rank(value):
    return hashlib.sha256(f'conpath-gallery-v1|{value}'.encode()).hexdigest()


@torch.inference_mode()
def infer(model, observation, valid, seed, independent=False):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    worlds = []
    for _ in range(K // 8):
        posterior = model(torch.from_numpy(observation[None]).cuda(),
                          valid_support_mask=torch.from_numpy(valid[None]).cuda(),
                          num_samples=8, disable_global_factors=independent,
                          generator=generator).posterior
        worlds.append(posterior.safe_samples()[0].cpu().numpy() > .5)
    result = np.concatenate(worlds)
    assert not np.any(result & ~valid[None])
    return result


def panels(case, observation, valid, target, predictions, start, goal, radius):
    points = [start, goal]
    sources = {}
    subtitle = case['source'] + ' · ' + case['global_id']
    def save(key, pixels, title, ramp=False):
        name = 'expanded-models/' + case['id'] + '-' + key
        sources[key] = draw.map_svg(name, pixels, title, subtitle=subtitle, points=points, ramp=ramp)
    save('observed', draw.categorical(observation, valid), '机器人已看到的地图')
    save('reference', draw.categorical(observation, valid, target), '数据集真实参考地图')
    event_probability, first_event, metrics = {}, {}, {}
    actual = bool(_accelerated_events(target[None], np.array([start]), np.array([goal]), (radius,))[0, 0, 0])
    for method, worlds in predictions.items():
        label = 'ConPath' if method == 'correlated' else '独立单元对照'
        save(method + '_sample', draw.categorical(observation, valid, worlds[0]), label + ' · 第一次实际补全')
        save(method, draw.probability(worlds.mean(0), valid), label + ' · 32次补全的逐格概率', True)
        events = _accelerated_events(worlds, np.array([start]), np.array([goal]), (radius,))[:, 0, 0]
        event_probability[method], first_event[method] = float(events.mean()), bool(events[0])
        mask = valid & (observation[2] > .5)
        union = np.count_nonzero((worlds[0] | target) & mask)
        metrics[method] = {'first_hidden_iou': float(np.count_nonzero(worlds[0] & target & mask) / union) if union else 1.,
                           'first_hidden_error': float(np.mean(worlds[0][mask] != target[mask])) if mask.any() else None}
    np.savez_compressed(OUT / (case['id'] + '.npz'), observation=observation, valid=valid, target=target, **predictions)
    return dict(case, panels=sources, target=actual, event_probability=event_probability,
                displayed_world_event=first_event, radius_cells=radius, start=list(start), goal=list(goal),
                seed=20260831 if case['domain'] == 'indoor' else 20260901,
                samples=K, metrics=metrics, selection='fixed_metadata_order_and_input_only_query',
                final_test=False, checkpoint_selected_on_this_cohort=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    old = list(csv.DictReader((ROOT / 'results/flatlands_parent_groups_v2/old_train_validation_parents.csv').open()))
    training_parents = {r['parent_group'] for r in old if r['provenance_split'] == 'train'}
    by_source = defaultdict(list)
    for r in old:
        if r['provenance_split'] == 'validation' and r['archive_split'] == 'train' and r['parent_group'] not in training_parents:
            by_source[r['source_dataset']].append(r)
    selected = [r for source in sorted(by_source) for r in sorted(by_source[source], key=lambda r: rank(r['global_id']))[:4]]
    assert len(selected) == 20
    manifest = draw.read(draw.MANIFEST)
    scenes = defaultdict(list)
    for r in manifest['records']['validation']:
        assert r['location'] in {'location_4', 'location_5', 'location_4_5'}
        scenes[r['scene_id']].append(r)
    outdoor = []
    for scene in sorted(scenes):
        ordered = sorted(scenes[scene], key=lambda r: float(r['timestamp']))
        outdoor.extend(ordered[int(i)] for i in np.linspace(0, len(ordered)-1, 4))
    assert len(outdoor) == 8
    # Persist all identities BEFORE opening packets or inspecting model outputs/target maps.
    selection = {'seed': SEED, 'samples': K, 'indoor': selected,
                 'outdoor': [{k: r[k] for k in ('timestamp', 'scene_id', 'location')} for r in outdoor],
                 'selection': '4 hash-ranked observations/source; 4 equally spaced timestamps/outdoor scene; no model/label filtering',
                 'indoor_parent_overlap_with_old_training': 0, 'new_physical_test_packet_images_opened': 0,
                 'scope': 'Old checkpoint-selection validation cohort, diagnostic gallery; not an independent final test.'}
    write(OUT / 'selection.json', selection)
    _verify_accelerator()
    dataset = FlatLandsReplayDataset(draw.ARCHIVE, draw.SELECTION, draw.QUERIES, split='validation')
    indices = {r.global_id: i for i, r in enumerate(dataset.observations)}
    models, checkpoints = {}, {}
    for method in ('correlated', 'independent'):
        report = draw.read(ROOT / f'results/current_baseline_k4_v1/{method}/20260831/report.json')
        source = report['source']; path = ROOT / source['path']
        assert sha(path) == source['sha256']
        models[method] = _load_model(path, method, torch.device('cuda'))[0]
        models[method].eval(); checkpoints[method] = source
    cases = []
    try:
        for row in selected:
            require_development_packet(dict(row, candidate_split='validation'))
            sample = dataset[indices[row['global_id']]]
            queries = construct_natural_queries(sample.observed_floor, sample.unobserved, sample.epistemic_mask,
                                                camera_px=sample.observation.camera_px, resolution_m=sample.observation.resolution)
            candidates = [q for q in queries if q.selection_status == 'selected']
            if not candidates:
                raise ValueError('Fixed gallery case has no input-eligible query; do not replace based on outcomes')
            q = min(candidates, key=lambda q: rank(f'{row["global_id"]}|{q.candidate_index}'))
            start, goal = [q.start_row, q.start_col], [q.goal_row, q.goal_col]
            sample_seed = SEED + int(row['global_id'].split('_')[1])
            predictions = {m: infer(model, sample.input_bev, sample.epistemic_mask, sample_seed, m == 'independent') for m, model in models.items()}
            case = {'id': 'indoor-' + row['global_id'], 'global_id': row['global_id'], 'source': row['source_dataset'],
                    'scene': row['scene_id'], 'parent_group': row['parent_group'], 'domain': 'indoor',
                    'split_zh': '室内开发验证', 'checkpoint_scope_zh': '室内训练检查点；本组曾参与旧模型选优',
                    'sampling_seed': sample_seed, 'candidate_index': q.candidate_index,
                    'source_hashes': {n: hashlib.sha256(dataset._zip().read(f'{row["packet_directory"]}/{n}')).hexdigest() for n in ('observed_floor.png', 'unobserved.png', 'epistemic_mask.png', 'floor_map.png', 'metadata.json')}}
            cases.append(panels(case, sample.input_bev, sample.epistemic_mask, sample.target_free, predictions, start, goal, 10))
            print(json.dumps({'gallery_complete': case['id'], 'count': len(cases)}), flush=True)
    finally:
        dataset.close()
    del models
    source = draw.read(ROOT / 'site/data/unscenes3d_clean_candidate_qualitative.json')['checkpoint']
    assert sha(ROOT / source['path']) == source['sha256']
    model, _, config = outdoor_model(ROOT / source['path'], torch.device('cuda'))
    assert config['checkpoint_trained_with_same_support_policy']
    checkpoints['outdoor_correlated'] = source
    for i, record in enumerate(outdoor):
        frame = _load_frame(record['timestamp'], raw_root=draw.RAW, label_root=draw.LABELS, adapter=_resolve_adapter(manifest))
        q = min(range(len(frame.starts)), key=lambda q: rank(f'{frame.timestamp}|{q}'))
        sample_seed = SEED + 1000000 + i
        predictions = {'correlated': infer(model, frame.input_bev, frame.target_valid, sample_seed)}
        case = {'id': 'outdoor-' + frame.timestamp.replace('.', '-'), 'global_id': frame.timestamp,
                'scene': record['scene_id'], 'source': 'UnScenes3D', 'domain': 'outdoor', 'location': record['location'],
                'split_zh': '室外开发验证', 'checkpoint_scope_zh': '室外单独训练检查点；不是室内权重直接迁移',
                'sampling_seed': sample_seed, 'candidate_index': q,
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [draw.RAW / 'clouds' / f'{frame.timestamp}.bin', draw.LABELS / 'occ' / f'{frame.timestamp}.npy']}}
        cases.append(panels(case, frame.input_bev, frame.target_valid, frame.target_free, predictions,
                            frame.starts[q].tolist(), frame.goals[q].tolist(), 2))
        print(json.dumps({'gallery_complete': case['id'], 'count': len(cases)}), flush=True)
    files = [ROOT / 'site' / p for case in cases for p in case['panels'].values()]
    report = {'examples': cases, 'new_cases': len(cases), 'new_panel_count': len(files), 'samples': K,
              'checkpoints': checkpoints, 'selection': selection, 'selection_sha256': sha(OUT / 'selection.json'),
              'assets': [{'path': str(p.relative_to(ROOT / 'site')), 'sha256': sha(p)} for p in files],
              'script_sha256': sha(__file__), 'new_training_performed': False,
              'probability_definition': 'Actual binary-world vote from the same 32 saved worlds; first world always displayed',
              'support_caveat': 'Evaluation support is supplied from the dataset, including outdoor target-valid support; not an end-to-end sensor-only system.'}
    write(OUT / 'report.json', report)
    write(ROOT / 'site/data/expanded_model_gallery_zh.json', report)


if __name__ == '__main__':
    main()
