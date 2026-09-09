#!/usr/bin/env python3
"""CPU-only audit of the published pilot gallery against sealed cached packets.

Run after evaluation, result verification, and report generation:
    python scripts/audit_parent_pilot_gallery.py

No model/checkpoint/archive loader or production renderer/scorer is imported.
Only ten selected development packets and the two saved K32 world files are read.
The receipt is written to results/parent_group_pilot_v1/gallery_verification.json.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, label

ROOT = Path(__file__).resolve().parents[1]
SEED, K, RADIUS = 20260910, 32, 10
METHODS = ('correlated', 'independent')
PANEL_KEYS = ('observed', 'reference', 'correlated_sample', 'correlated', 'independent_sample', 'independent')
NS = '{http://www.w3.org/2000/svg}'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rank(*parts):
    return hashlib.sha256(('parent-pilot-v1-20260909|' + '|'.join(map(str, parts))).encode()).hexdigest()


def select_gallery_rows(rows):
    result = []
    for source in sorted({row['source_dataset'] for row in rows}):
        members = [row for row in rows if row['source_dataset'] == source]
        require(len(members) >= 2, f'Insufficient gallery candidates: {source}')
        result.extend(sorted(members, key=lambda row: rank('pilot-gallery', row['global_id']))[:2])
    return result


def input_queries(observation, valid, hidden, camera_px, resolution):
    """Reconstruct the full polar stencil using input channels only, never target."""
    require(math.isfinite(resolution) and resolution > 0, 'Invalid query resolution')
    h, w = valid.shape
    require(len(camera_px) == 2, 'Invalid camera coordinate')
    camera = (int(camera_px[1]), int(camera_px[0]))
    start = None
    if 0 <= camera[0] < h and 0 <= camera[1] < w:
        coordinates = np.argwhere((observation[0] > .5) & valid)
        if len(coordinates):
            # argwhere is row-major: argmin resolves distance ties by row, then col.
            squared_distance = np.sum((coordinates - camera) ** 2, axis=1)
            start = tuple(map(int, coordinates[int(np.argmin(squared_distance))]))
    result, seen = [], set()
    for distance in (.4, .8, 1.2):
        for angle in range(0, 360, 30):
            goal = None
            status = 'no_observed_valid_start'
            if start is not None:
                theta = math.radians(angle)
                goal = (start[0] + round(math.sin(theta) * distance / resolution),
                        start[1] + round(math.cos(theta) * distance / resolution))
                if not (0 <= goal[0] < h and 0 <= goal[1] < w):
                    status = 'goal_out_of_bounds'
                elif goal == start:
                    status = 'goal_equals_start'
                elif not valid[goal]:
                    status = 'goal_outside_epistemic_mask'
                elif not hidden[goal]:
                    status = 'goal_not_unobserved'
                elif goal in seen:
                    status = 'duplicate_goal'
                else:
                    status = 'selected'
                    seen.add(goal)
            result.append({'candidate_index': len(result), 'distance_m': distance, 'angle_deg': angle,
                           'start_row': start[0] if start else None, 'start_col': start[1] if start else None,
                           'goal_row': goal[0] if goal else None, 'goal_col': goal[1] if goal else None,
                           'selection_status': status})
    return result


def disk_event(world, start, goal, radius=RADIUS):
    """Explicit integer-disk erosion, then four-neighbor connected components."""
    world = np.asarray(world)
    require(world.ndim == 2 and world.dtype == bool, 'Connectivity expects a binary world')
    require(isinstance(radius, int) and radius >= 0, 'Invalid robot radius')
    for point in (start, goal):
        require(len(point) == 2 and all(isinstance(v, (int, np.integer)) for v in point), 'Noninteger endpoint')
        require(0 <= point[0] < world.shape[0] and 0 <= point[1] < world.shape[1], 'Endpoint outside map')
    if not world[tuple(start)] or not world[tuple(goal)]:
        return False
    axis = np.arange(-radius, radius + 1)
    footprint = axis[:, None] ** 2 + axis[None, :] ** 2 <= radius ** 2
    centers = binary_erosion(world, structure=footprint, border_value=0)
    components, _ = label(centers, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    first, last = components[tuple(start)], components[tuple(goal)]
    return bool(first != 0 and first == last)


def expected_pixels(observation, valid, *, free_map=None, probabilities=None):
    """Independent RGB oracle for the documented categorical/probability palette."""
    valid = np.asarray(valid)
    require(valid.ndim == 2 and valid.dtype == bool, 'Invalid support bitmap')
    require(free_map is None or probabilities is None, 'Ambiguous map type')
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        require(probabilities.shape == valid.shape and np.isfinite(probabilities).all(), 'Invalid probability map')
        require(((probabilities >= 0) & (probabilities <= 1)).all(), 'Probability out of bounds')
        low = np.array([241, 236, 212], dtype=float)
        high = np.array([21, 127, 119], dtype=float)
        pixels = np.rint((1 - probabilities[..., None]) * low + probabilities[..., None] * high).astype(np.uint8)
    else:
        pixels = np.full((*valid.shape, 3), [247, 248, 250], dtype=np.uint8)
        if free_map is not None:
            require(free_map.shape == valid.shape and free_map.dtype == bool, 'Invalid categorical free map')
            pixels[valid & free_map] = [101, 181, 157]
            pixels[valid & ~free_map] = [54, 70, 84]
        else:
            require(observation.shape == (3, *valid.shape), 'Invalid observation shape')
            pixels[valid & (observation[2] > .5)] = [220, 228, 236]
            pixels[valid & (observation[0] > .5)] = [101, 181, 157]
            pixels[valid & (observation[1] > .5)] = [54, 70, 84]
    row, col = np.indices(valid.shape)
    pixels[~valid] = [247, 248, 250]
    pixels[~valid & ((row // 4 + col // 4) % 2 == 0)] = [232, 236, 240]
    return pixels


def audit_svg(path, expected, start, goal, *, probability, title, subtitle):
    """Compare decoded pixels and all visible SVG annotation/geometry fields."""
    tree = ET.fromstring(Path(path).read_text())
    require(tree.tag == NS + 'svg', f'{path}: not an SVG')
    require(tree.attrib == {'width': '380', 'height': '452', 'viewBox': '0 0 380 452',
                            'role': 'img', 'aria-labelledby': 'title desc'}, f'{path}: changed canvas geometry')
    # Build annotation expectations independently of the report/renderer functions.
    nodes = []
    def add(tag, attributes=None, text=None):
        nodes.append((NS + tag, {key: str(value) for key, value in (attributes or {}).items()}, text))
    legend = ('米白到青绿表示单元格可通行概率从零到一，棋盘格表示有效范围外。' if probability else
              '绿色表示可通行，深灰表示阻挡，浅灰表示未知，棋盘格表示有效范围外。')
    add('title', {'id': 'title'}, title)
    add('desc', {'id': 'desc'}, subtitle + '。' + legend + 'S 为起点，G 为目标，不画直连路径。')
    add('style', text='text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}')
    add('rect', {'width': 380, 'height': 452, 'fill': 'white'})
    add('text', {'x': 20, 'y': 26, 'fill': '#263544', 'font-size': 20, 'font-weight': 600}, title)
    add('text', {'x': 20, 'y': 48, 'fill': '#667582', 'font-size': 11}, subtitle)
    add('image', {'x': 20, 'y': 60, 'width': 340, 'height': 340, 'style': 'image-rendering:pixelated'})
    for symbol, point, color in [('S', start, '#326cbd'), ('G', goal, '#a35328')]:
        x = 20 + (point[1] + .5) * 340 / expected.shape[1]
        y = 60 + (point[0] + .5) * 340 / expected.shape[0]
        if symbol == 'S':
            add('circle', {'cx': f'{x:.2f}', 'cy': f'{y:.2f}', 'r': 7, 'fill': color, 'stroke': 'white', 'stroke-width': 2})
        else:
            add('path', {'d': f'M{x:.2f},{y-9:.2f} l9,9 l-9,9 l-9,-9 Z', 'fill': color, 'stroke': 'white', 'stroke-width': 2})
        add('text', {'x': f'{min(326, max(24, x+10)):.2f}', 'y': f'{min(391, max(75, y-11)):.2f}',
                     'font-size': 16, 'font-weight': 700, 'stroke': 'white', 'stroke-width': 4,
                     'paint-order': 'stroke', 'fill': color}, symbol)
    if probability:
        add('defs')
        add('linearGradient', {'id': 'p'})
        add('stop', {'stop-color': '#f1ecd4'})
        add('stop', {'offset': 1, 'stop-color': '#157f77'})
        add('text', {'x': 20, 'y': 419, 'font-size': 11, 'fill': '#667582'}, '单元格可通行概率')
        add('rect', {'x': 136, 'y': 409, 'width': 174, 'height': 9, 'rx': 2, 'fill': 'url(#p)'})
        add('text', {'x': 117, 'y': 419, 'font-size': 11}, '0')
        add('text', {'x': 318, 'y': 419, 'font-size': 11}, '1')
    else:
        for x, color, word in [(20, '#65b59d', '可通行'), (102, '#364654', '阻挡'), (172, '#dce4ec', '未知'), (245, '#e8ecf0', '范围外')]:
            add('rect', {'x': x, 'y': 410, 'width': 10, 'height': 10, 'rx': 2, 'fill': color})
            add('text', {'x': x+15, 'y': 420, 'font-size': 11, 'fill': '#667582'}, word)
    add('text', {'x': 20, 'y': 441, 'font-size': 11, 'fill': '#667582'}, '蓝色圆点 S：起点 · 棕色菱形 G：目标 · 不表示规划路线')
    actual_nodes = list(tree.iter())[1:]
    require(len(actual_nodes) == len(nodes), f'{path}: missing or additional annotation/overlay')
    png_bytes = None
    for element, (tag, attributes, text) in zip(actual_nodes, nodes):
        actual_attributes = element.attrib.copy()
        if tag == NS + 'image':
            uri = actual_attributes.pop('href', '')
            require(uri.startswith('data:image/png;base64,'), f'{path}: image is not embedded PNG')
            png_bytes = base64.b64decode(uri.partition(',')[2], validate=True)
        require(element.tag == tag and actual_attributes == attributes and (element.text or '').strip() == (text or ''),
                f'{path}: SVG annotation, legend, or endpoint mismatch at {tag.removeprefix(NS)}')
        require(not (element.tail or '').strip(), f'{path}: unexpected visible text')
        if tag not in (NS + 'defs', NS + 'linearGradient'):
            require(len(element) == 0, f'{path}: unexpected nested overlay')
    if probability:
        defs = tree.find(NS + 'defs')
        require(len(defs) == 1 and len(defs[0]) == 2, f'{path}: invalid probability legend structure')
    require(png_bytes is not None, f'{path}: missing image')
    with Image.open(io.BytesIO(png_bytes)) as picture:
        require(picture.format == 'PNG' and picture.mode == 'RGB' and picture.size == expected.shape[1::-1], f'{path}: wrong bitmap format or shape')
        actual = np.asarray(picture)
    require(np.array_equal(actual, expected), f'{path}: embedded RGB bitmap differs at {int(np.count_nonzero(np.any(actual != expected, axis=-1)))} pixels')
    return {'svg_sha256': sha(path), 'embedded_png_sha256': hashlib.sha256(png_bytes).hexdigest(),
            'rgb_sha256': hashlib.sha256(actual.tobytes()).hexdigest(), 'bitmap_shape': list(actual.shape),
            'all_pixels_match': True, 'endpoints_and_legend_match': True}


def audit_gallery(root=ROOT):
    root = Path(root).resolve()
    run, site = root/'results/parent_group_pilot_v1', root/'site'
    data = run/'data'
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
    analysis_sha = hashes[str((run/'analysis.json').relative_to(root))]
    verification = read_json(run/'verification.json')
    require(verification['passed'] is True and verification['analysis_sha256'] == analysis_sha, 'Independent result verification missing or stale')
    require(analysis['implementation_checks_passed'] is True and analysis['training_runs_completed'] == 9 and analysis['final_test'] is False, 'Incomplete or ineligible pilot result')
    checked(site/'data/parent_group_pilot_zh.json', analysis_sha)
    gallery_path = site/'data/parent_pilot_gallery_zh.json'
    gallery = read_json(gallery_path)
    require(gallery['source_analysis_sha256'] == analysis_sha and gallery['checkpoint_seed'] == SEED, 'Gallery provenance mismatch')
    seal = read_json(data/'seal.json')
    frozen = read_json(run/'evaluation/evaluation_frozen_before_scoring.json')
    require(frozen['data_seal_sha256'] == sha(data/'seal.json'), 'Scoring used a different data seal')
    def sealed(name):
        return checked(data/name, seal['files'][name])
    protocol = json.loads(sealed('protocol.json').read_text())
    require(protocol['query']['radii_cells'] == [0, 10, 20] and protocol['evaluation']['K_primary'] == K and protocol['seeds'][0] == SEED, 'Unexpected frozen protocol')
    require(json.loads(sealed('data_audit.json').read_text())['passed'] is True, 'Data quality gate failed')
    with sealed('selected.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len({row['global_id'] for row in rows}) == len(rows), 'Duplicate manifest global IDs')
    require(len({row['parent_group'] for row in rows}) == len(rows), 'Pilot places overlap across observations/partitions')
    for row in rows:
        require(re.fullmatch(r'obs_\d+', row['global_id']) is not None, 'Invalid observation identity')
        require(row['archive_split'] == 'train' and row['packet_directory'] == 'train/'+row['global_id'], 'Physical non-train packet is ineligible')
        require(row['candidate_split'] in ('train', 'calibration', 'validation') and row['parent_group'], 'Unresolved development partition')
    validation = [row for row in rows if row['candidate_split'] == 'validation']
    require(len(validation) == 40 and sorted(Counter(row['source_dataset'] for row in validation).values()) == [8]*5, 'Unexpected validation cohort')
    chosen = select_gallery_rows(validation)
    cases = gallery['examples']
    require(len(cases) == 10 and [case['global_id'] for case in cases] == [row['global_id'] for row in chosen], 'Gallery cases differ from frozen two-per-source hash ranking')
    query_log = [json.loads(line) for line in sealed('input_queries_before_target.jsonl').read_text().splitlines() if line.strip()]
    require(len(query_log) == len(rows) and {entry['global_id'] for entry in query_log} == {row['global_id'] for row in rows}, 'Input query log identity mismatch')
    logged = {entry['global_id']: entry['queries'] for entry in query_log}
    packets = {}
    for row, case in zip(chosen, cases):
        gid = row['global_id']
        with np.load(sealed('packets/'+gid+'.npz'), allow_pickle=False) as archive:
            packet = {name: archive[name] for name in ('observation', 'valid', 'hidden', 'target', 'starts', 'goals', 'targets', 'candidate_indices')}
        valid, observation, hidden, target = [packet[name] for name in ('valid', 'observation', 'hidden', 'target')]
        require(valid.shape == (256, 256) and valid.dtype == bool, f'{gid}: invalid support')
        require(observation.shape == (3, *valid.shape) and np.isfinite(observation).all() and np.isin(observation, [0, 1]).all(), f'{gid}: invalid observation')
        require(np.array_equal(observation.sum(0), valid), f'{gid}: overlapping or incomplete observation channels')
        require(hidden.dtype == target.dtype == bool and hidden.shape == target.shape == valid.shape, f'{gid}: invalid map shape/type')
        require(np.array_equal(hidden, observation[2].astype(bool)) and not target[~valid].any(), f'{gid}: hidden/support mismatch')
        require(np.array_equal(target[~hidden], observation[0, ~hidden].astype(bool)), f'{gid}: target contradicts observed evidence')
        queries = input_queries(observation, valid, hidden, json.loads(row['camera_px']), float(row['resolution']))
        require(queries == logged[gid], f'{gid}: input-only query reconstruction differs from frozen log')
        selected = sorted((query for query in queries if query['selection_status'] == 'selected'), key=lambda query: rank(gid, query['candidate_index']))
        require(len(selected) > 0, f'{gid}: no selected query')
        for name, values in [('candidate_indices', [query['candidate_index'] for query in selected]),
                             ('starts', [[query['start_row'], query['start_col']] for query in selected]),
                             ('goals', [[query['goal_row'], query['goal_col']] for query in selected])]:
            require(np.issubdtype(packet[name].dtype, np.integer) and packet[name].tolist() == values, f'{gid}: cached {name} ordering differs from input hash')
        start, goal = packet['starts'][0].tolist(), packet['goals'][0].tolist()
        require(packet['targets'].dtype == bool and packet['targets'].shape == (len(selected), 3), f'{gid}: target query shape mismatch')
        target_event = disk_event(target, start, goal)
        require(target_event == bool(packet['targets'][0, 1]), f'{gid}: cached target label differs from explicit disk oracle')
        expected_metadata = {'id': 'pilot-'+gid, 'global_id': gid, 'source': row['source_dataset'], 'scene': row['scene_id'],
                             'parent_group': row['parent_group'], 'domain': 'indoor', 'cohort': 'new_pilot',
                             'radius_cells': RADIUS, 'seed': SEED, 'sampling_seed': SEED+4000000+int(gid.split('_')[1]),
                             'candidate_index': int(packet['candidate_indices'][0]), 'start': start, 'goal': goal, 'samples': K,
                             'target': target_event, 'final_test': False, 'checkpoint_selected_on_this_cohort': False,
                             'selection': 'two hash-ranked observations/source, first input-hash-ordered query; first sampled world',
                             'split_zh': '新基线 · 独立地点验证',
                             'checkpoint_scope_zh': '本轮从头训练；此地点未参与训练或检查点选优，属于开发验证'}
        require(all(case.get(key) == value for key, value in expected_metadata.items()), f'{gid}: gallery case metadata differs from frozen inputs')
        require(type(case['target']) is bool and case['final_test'] is False and case['checkpoint_selected_on_this_cohort'] is False, f'{gid}: invalid boolean metadata')
        require(set(case['panels']) == set(PANEL_KEYS), f'{gid}: six expected panels are required')
        require(set(case['event_probability']) == set(case['displayed_world_event']) == set(METHODS), f'{gid}: unexpected gallery models')
        packets[gid] = packet
    asset_rows = gallery['assets']
    assets = {entry['path']: entry['sha256'] for entry in asset_rows}
    expected_paths = {f'assets/zh/parent-pilot/{row["global_id"]}-{key}.svg' for row in chosen for key in PANEL_KEYS}
    require(len(asset_rows) == len(assets) == 60 and set(assets) == expected_paths, 'Asset manifest must contain exactly 60 unique expected panels')
    require(set(gallery['saved_worlds']) == set(METHODS), 'Unexpected saved-world methods')
    worlds_by_method, csv_by_method = {}, {}
    for method in METHODS:
        folder = run/'evaluation'/method/str(SEED)
        reference = gallery['saved_worlds'][method]
        require(reference['path'] == str((folder/'worlds.npz').relative_to(root)), 'Saved-world path does not identify the fixed checkpoint seed')
        report = read_json(folder/'report.json')
        require(report['method'] == method and report['seed'] == SEED, 'Saved-world report identity mismatch')
        path = checked(root/reference['path'], reference['sha256'])
        require(report['files']['worlds.npz'] == reference['sha256'], 'Gallery worlds differ from evaluated worlds')
        with np.load(path, allow_pickle=False) as archive:
            ids, worlds = archive['global_ids'].tolist(), archive['worlds']
        require(ids == [row['global_id'] for row in validation], f'{method}: saved-world ID ordering mismatch')
        require(len(set(ids)) == len(ids) and worlds.shape == (len(ids), K, 256, 256) and worlds.dtype == bool, f'{method}: malformed or non-K32 saved worlds')
        worlds_by_method[method] = {gid: worlds[ids.index(gid)].copy() for gid in packets}
        del worlds
        with checked(folder/'predictions_k32.csv', report['files']['predictions_k32.csv']).open(newline='') as handle:
            predictions = list(csv.DictReader(handle))
        prediction_map = {(entry['global_id'], int(entry['candidate_index']), int(entry['radius_cells'])): entry for entry in predictions}
        require(len(prediction_map) == len(predictions), f'{method}: duplicate evaluated queries')
        csv_by_method[method] = prediction_map
    panel_receipts, case_receipts = [], []
    for case in cases:
        gid = case['global_id']
        packet = packets[gid]
        observation, valid, hidden, target = [packet[name] for name in ('observation', 'valid', 'hidden', 'target')]
        start, goal = case['start'], case['goal']
        images = {'observed': expected_pixels(observation, valid), 'reference': expected_pixels(observation, valid, free_map=target)}
        votes = {}
        for method in METHODS:
            worlds = worlds_by_method[method][gid]
            require(not worlds[:, ~valid].any() and np.array_equal(worlds[:, ~hidden], np.broadcast_to(observation[0, ~hidden].astype(bool), worlds[:, ~hidden].shape)), f'{gid}/{method}: world violates support or known evidence')
            outcomes = [disk_event(world, start, goal) for world in worlds]
            probability = sum(outcomes) / K
            require(type(case['event_probability'][method]) in (int, float) and case['event_probability'][method] == probability, f'{gid}/{method}: probability is not the K32 event vote')
            require(type(case['displayed_world_event'][method]) is bool and case['displayed_world_event'][method] == outcomes[0], f'{gid}/{method}: displayed event is not world zero')
            prediction = csv_by_method[method][(gid, case['candidate_index'], RADIUS)]
            require(prediction['parent_group'] == case['parent_group'] and float(prediction['probability']) == probability, f'{gid}/{method}: evaluated CSV vote mismatch')
            images[method+'_sample'] = expected_pixels(observation, valid, free_map=worlds[0])
            images[method] = expected_pixels(observation, valid, probabilities=worlds.sum(0, dtype=np.int64) / K)
            votes[method] = {'positive_worlds': sum(outcomes), 'worlds': K, 'probability': probability, 'first_world_event': outcomes[0]}
        for key in PANEL_KEYS:
            relative = f'assets/zh/parent-pilot/{gid}-{key}.svg'
            require(case['panels'][key] == relative, f'{gid}/{key}: wrong panel identity')
            path = checked(site/relative, assets[relative])
            probability = key in METHODS
            if key in ('observed', 'reference'):
                title = '机器人已看到的地图' if key == 'observed' else '数据集真实参考地图'
                subtitle = case['source'] + ' · ' + gid
            else:
                name = 'ConPath' if key.startswith('correlated') else '独立单元对照'
                title = name + (' · 32次补全的逐格概率' if probability else ' · 第一次实际补全')
                subtitle = '新基线 · 独立地点验证'
            panel_receipts.append({'path': relative, 'case': gid, 'panel': key,
                                   **audit_svg(path, images[key], start, goal, probability=probability, title=title, subtitle=subtitle)})
        case_receipts.append({'global_id': gid, 'source': case['source'], 'parent_group': case['parent_group'],
                              'candidate_index': case['candidate_index'], 'start': start, 'goal': goal,
                              'radius_cells': RADIUS, 'reference_event': case['target'], 'votes': votes})
    return {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(), 'analysis_sha256': analysis_sha,
            'gallery_sha256': sha(gallery_path), 'script_sha256': sha(Path(__file__)),
            'cases_checked': len(cases), 'svg_bitmaps_checked': len(panel_receipts), 'rgb_pixels_checked': len(panel_receipts)*256*256,
            'world_event_checks': len(cases)*len(METHODS)*K, 'reference_event_checks': len(cases),
            'selection_reconstructed_without_targets': True, 'first_input_hash_query': True, 'first_actual_world': True,
            'independent_pixel_oracle': True, 'explicit_disk_four_neighbor_oracle': True,
            'cached_development_packets_only': True, 'cached_packets_read': len(packets),
            'new_archive_images_read': 0, 'model_inference_run': False, 'gpu_used': False, 'final_test': False,
            'source_hashes': hashes, 'cases': case_receipts, 'panels': panel_receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root; defaults to this checkout')
    args = parser.parse_args()
    receipt = args.root.resolve()/'results/parent_group_pilot_v1/gallery_verification.json'
    try:
        result = audit_gallery(args.root)
    except Exception as error:
        # A failed rerun must invalidate any earlier passing receipt.
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
