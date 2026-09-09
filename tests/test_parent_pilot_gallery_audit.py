"""Synthetic gallery corruption tests; no saved packets, worlds, or models."""
import base64
import copy
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from pathrel.flatlands_query import construct_natural_queries
from scripts import build_site_visuals as draw
from scripts.audit_parent_pilot_gallery import (
    audit_gallery, audit_svg, disk_event, expected_pixels, input_queries, select_gallery_rows,
)


SVG = {'svg': 'http://www.w3.org/2000/svg'}


def synthetic_observation():
    valid = np.ones((9, 13), dtype=bool)
    valid[0] = False
    valid[4:8, 8:] = False
    free = np.zeros_like(valid)
    blocked = np.zeros_like(valid)
    free[2, 1] = True
    blocked[2, 2] = True
    hidden = valid & ~free & ~blocked
    return np.stack((free, blocked, hidden)).astype(np.float32), valid


class GalleryPixelsTests(unittest.TestCase):
    def test_categorical_colors_and_invalid_checker_are_exact(self):
        observation, valid = synthetic_observation()
        actual = expected_pixels(observation, valid)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(actual.shape, (9, 13, 3))
        for point, color in [((2, 1), [101, 181, 157]),
                             ((2, 2), [54, 70, 84]),
                             ((2, 3), [220, 228, 236]),
                             ((0, 0), [232, 236, 240]),
                             ((0, 4), [247, 248, 250]),
                             ((4, 8), [247, 248, 250]),
                             ((4, 12), [232, 236, 240])]:
            with self.subTest(point=point):
                self.assertEqual(actual[point].tolist(), color)
        np.testing.assert_array_equal(actual, draw.categorical(observation, valid))

    def test_reference_uses_binary_map_and_preserves_invalid_support(self):
        observation, valid = synthetic_observation()
        target = np.zeros_like(valid)
        target[2, 3] = True
        target[0] = True
        actual = expected_pixels(observation, valid, free_map=target)
        self.assertEqual(actual[2, 1].tolist(), [54, 70, 84])
        self.assertEqual(actual[2, 3].tolist(), [101, 181, 157])
        self.assertEqual(actual[0, 0].tolist(), [232, 236, 240])
        np.testing.assert_array_equal(actual, draw.categorical(observation, valid, target))

    def test_probability_colors_round_and_do_not_encode_event_probability(self):
        observation, valid = synthetic_observation()
        probabilities = np.zeros(valid.shape)
        probabilities[2, 1:5] = [0., .25, .5, 1.]
        actual = expected_pixels(observation, valid, probabilities=probabilities)
        self.assertEqual(actual[2, 1:5].tolist(),
                         [[241, 236, 212], [186, 209, 189],
                          [131, 182, 166], [21, 127, 119]])
        self.assertEqual(actual[0, 0].tolist(), [232, 236, 240])
        np.testing.assert_array_equal(actual, draw.probability(probabilities, valid))


class GallerySvgTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pilot-gallery-synthetic-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        asset_patch = patch.object(draw, 'ASSETS', self.directory)
        asset_patch.start()
        self.addCleanup(asset_patch.stop)
        self.observation, self.valid = synthetic_observation()
        self.pixels = draw.categorical(self.observation, self.valid)
        self.start, self.goal = (2, 1), (6, 5)
        self.title = '合成地图 <兼容检查>'
        self.subtitle = '仅合成数组 & 临时目录'

    def render(self, pixels=None, *, probability=False, points=None):
        draw.map_svg('fixture', self.pixels if pixels is None else pixels,
                     self.title, subtitle=self.subtitle,
                     points=[self.start, self.goal] if points is None else points,
                     ramp=probability)
        return self.directory / 'fixture.svg'

    def check(self, path, pixels=None, *, probability=False):
        return audit_svg(path, self.pixels if pixels is None else pixels,
                         self.start, self.goal, probability=probability,
                         title=self.title, subtitle=self.subtitle)

    def mutate(self, path, mutation):
        tree = ET.parse(path)
        mutation(tree.getroot())
        tree.write(path, encoding='unicode')

    def test_production_categorical_and_probability_svg_are_accepted(self):
        self.check(self.render())
        pixels = draw.probability(np.full(self.valid.shape, .5), self.valid)
        self.check(self.render(pixels, probability=True), pixels, probability=True)

    def test_rejects_single_decoded_png_pixel_change(self):
        path = self.render()

        def corrupt(root):
            element = root.find('svg:image', SVG)
            encoded = element.attrib['href'].split(',', 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as original:
                pixels = np.array(original)
            pixels[2, 3, 0] ^= 1
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format='PNG')
            element.set('href', 'data:image/png;base64,' +
                        base64.b64encode(stream.getvalue()).decode('ascii'))

        self.mutate(path, corrupt)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_rejects_swapped_start_and_goal_and_row_column_transposition(self):
        for points in ([self.goal, self.start],
                       [self.start[::-1], self.goal[::-1]]):
            with self.subTest(points=points):
                path = self.render(points=points)
                with self.assertRaises(ValueError):
                    self.check(path)

    def test_rejects_goal_diamond_coordinate_change(self):
        path = self.render()

        def corrupt(root):
            diamond = root.find('svg:path', SVG)
            x = 20 + (self.goal[1] + .5) * 340 / self.valid.shape[1]
            y = 60 + (self.goal[0] + .5) * 340 / self.valid.shape[0]
            diamond.set('d', f'M{x + 1:.2f},{y - 9:.2f} l9,9 l-9,9 l-9,-9 Z')

        self.mutate(path, corrupt)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_rejects_reversed_probability_legend(self):
        pixels = draw.probability(np.full(self.valid.shape, .25), self.valid)
        path = self.render(pixels, probability=True)

        def corrupt(root):
            stops = root.findall('svg:defs/svg:linearGradient/svg:stop', SVG)
            colors = [stop.attrib['stop-color'] for stop in stops]
            for stop, color in zip(stops, reversed(colors)):
                stop.set('stop-color', color)

        self.mutate(path, corrupt)
        with self.assertRaises(ValueError):
            self.check(path, pixels, probability=True)

    def test_rejects_categorical_legend_color_change(self):
        path = self.render()

        def corrupt(root):
            swatch = next(element for element in root.findall('svg:rect', SVG)
                          if element.get('x') == '20' and element.get('y') == '410')
            swatch.set('fill', '#364654')

        self.mutate(path, corrupt)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_rejects_extra_route_and_wrong_title(self):
        for case in ('route', 'title'):
            with self.subTest(case=case):
                path = self.render()

                def corrupt(root):
                    if case == 'route':
                        ET.SubElement(root, '{' + SVG['svg'] + '}line',
                                      x1='40', y1='100', x2='200', y2='200',
                                      stroke='red')
                    else:
                        root.find('svg:title', SVG).text = '另一个模型'

                self.mutate(path, corrupt)
                with self.assertRaises(ValueError):
                    self.check(path)

    def test_first_world_and_32_world_mean_cannot_replace_each_other(self):
        worlds = np.zeros((32, *self.valid.shape), dtype=bool)
        worlds[0] = self.valid
        worlds[:, self.observation[0] > .5] = True
        worlds[:, self.observation[1] > .5] = False
        first = expected_pixels(self.observation, self.valid, free_map=worlds[0])
        mean = expected_pixels(self.observation, self.valid, probabilities=worlds.mean(0))
        self.assertFalse(np.array_equal(first, mean))
        path = self.render(first)
        self.check(path, first)
        with self.assertRaises(ValueError):
            self.check(path, mean)
        path = self.render(mean, probability=True)
        self.check(path, mean, probability=True)
        with self.assertRaises(ValueError):
            self.check(path, first, probability=True)


class GalleryEventTests(unittest.TestCase):
    def test_connected_disconnected_and_blocked_goal_worlds(self):
        world = np.ones((41, 61), dtype=bool)
        start, goal = (20, 15), (20, 45)
        self.assertTrue(disk_event(world, start, goal))
        wall = world.copy()
        wall[:, 30] = False
        self.assertFalse(disk_event(wall, start, goal))
        blocked_goal = world.copy()
        blocked_goal[goal] = False
        self.assertFalse(disk_event(blocked_goal, start, goal, radius=0))
        self.assertFalse(disk_event(blocked_goal, start, goal))

    def test_diagonal_contact_is_not_four_neighbor_connectivity(self):
        world = np.eye(7, dtype=bool)
        self.assertFalse(disk_event(world, (1, 1), (5, 5), radius=0))
        world[1, 1:6] = True
        world[1:6, 5] = True
        self.assertTrue(disk_event(world, (1, 1), (5, 5), radius=0))

    def test_disk_radius_blocks_narrow_corridor_and_raster_boundary(self):
        world = np.zeros((41, 61), dtype=bool)
        world[20, 10:51] = True
        self.assertTrue(disk_event(world, (20, 15), (20, 45), radius=0))
        self.assertFalse(disk_event(world, (20, 15), (20, 45), radius=1))
        world[:] = True
        self.assertTrue(disk_event(world, (0, 15), (20, 45), radius=0))
        self.assertFalse(disk_event(world, (0, 15), (20, 45), radius=1))

    def test_event_vote_differs_from_thresholded_mean_map(self):
        worlds = np.ones((32, 7, 9), dtype=bool)
        worlds[:16, :, 3] = False
        worlds[16:, :, 5] = False
        start, goal = (3, 1), (3, 7)
        votes = [disk_event(world, start, goal, radius=0) for world in worlds]
        self.assertEqual(sum(votes), 0)
        self.assertTrue(disk_event(worlds.mean(0) >= .5, start, goal, radius=0))


class GallerySelectionTests(unittest.TestCase):
    def test_source_hash_order_is_independent_of_input_order(self):
        rows = [dict(global_id=f'obs_{i:06d}', source_dataset=source,
                     parent_group=f'{source}:parent{i}', candidate_split='validation',
                     archive_split='train', packet_directory=f'train/obs_{i:06d}')
                for source, first in [('source-b', 20), ('source-a', 0)]
                for i in range(first, first + 8)]

        def rank(row):
            value = 'parent-pilot-v1-20260909|pilot-gallery|' + row['global_id']
            return hashlib.sha256(value.encode('utf-8')).hexdigest()

        expected = [row for source in ('source-a', 'source-b')
                    for row in sorted((r for r in rows if r['source_dataset'] == source),
                                      key=rank)[:2]]
        self.assertEqual(select_gallery_rows(rows), expected)
        self.assertEqual(select_gallery_rows(list(reversed(rows))), expected)
        order = np.random.default_rng(71).permutation(len(rows))
        self.assertEqual(select_gallery_rows([rows[int(i)] for i in order]), expected)


class GalleryInputQueryTests(unittest.TestCase):
    def test_complete_stencil_keeps_hypothetical_obstacle_goals(self):
        valid = np.ones((161, 171), dtype=bool)
        observed = np.zeros_like(valid)
        observed[80, 85] = True
        hidden = valid & ~observed
        observation = np.stack((observed, np.zeros_like(valid), hidden)).astype(np.float32)
        before = [array.copy() for array in (observation, valid, hidden)]
        camera, resolution = [85, 80], .02
        queries = input_queries(observation, valid, hidden, camera, resolution)
        expected = [asdict(query) for query in construct_natural_queries(
            observed, hidden, valid, camera_px=camera, resolution_m=resolution)]
        self.assertEqual(queries, expected)
        self.assertEqual(len(queries), 36)
        self.assertEqual([query['candidate_index'] for query in queries], list(range(36)))
        self.assertTrue(all(query['selection_status'] == 'selected' for query in queries))
        self.assertEqual((queries[0]['goal_row'], queries[0]['goal_col']), (80, 105))
        # These two complete maps imply opposite labels for all selected goals.
        # Neither is an input to construction, so even blocked goals stay selected.
        for target in (observed.copy(), valid.copy()):
            self.assertEqual(input_queries(observation, valid, hidden, camera, resolution), queries)
            self.assertTrue(all(set(query) == set(expected[0]) for query in queries))
            labels = [bool(target[query['goal_row'], query['goal_col']]) for query in queries]
            self.assertEqual(set(labels), {bool(target[80, 105])})
        for actual, original in zip((observation, valid, hidden), before):
            np.testing.assert_array_equal(actual, original)

    def test_query_statuses_match_input_only_constructor(self):
        valid = np.ones((9, 11), dtype=bool)
        observed = np.zeros_like(valid)
        observed[4, 4] = observed[4, 6] = True
        valid[4, 8] = False
        hidden = valid & ~observed
        hidden[5, 4] = False
        observation = np.stack((observed, valid & ~observed & ~hidden, hidden)).astype(np.float32)
        # Nearest-start ties, repeated rounded goals, bounds, known goals and
        # missing starts must retain their complete rejection records.
        for camera, resolution in [([5, 4], .4), ([4, 4], .1), ([-1, 4], .4)]:
            with self.subTest(camera=camera, resolution=resolution):
                expected = [asdict(query) for query in construct_natural_queries(
                    observed, hidden, valid, camera_px=camera, resolution_m=resolution)]
                self.assertEqual(input_queries(observation, valid, hidden, camera, resolution), expected)
        observation[0] = 0
        queries = input_queries(observation, valid, hidden, [5, 4], .4)
        self.assertEqual(len(queries), 36)
        self.assertTrue(all(query['selection_status'] == 'no_observed_valid_start'
                            for query in queries))


def synthetic_gallery(root):
    """Build the entire audit input tree from arrays, with only ten packets."""
    run, site = root / 'results/parent_group_pilot_v1', root / 'site'
    data = run / 'data'
    (data / 'packets').mkdir(parents=True)
    (site / 'data').mkdir(parents=True)
    (site / 'assets/zh/parent-pilot').mkdir(parents=True)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False))

    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def rank(*parts):
        value = 'parent-pilot-v1-20260909|' + '|'.join(map(str, parts))
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    rows = [dict(global_id=f'obs_{i:06d}', source_dataset=f'source-{i // 8}',
                 parent_group=f'parent-{i}', scene_id=f'scene-{i}',
                 archive_split='train', packet_directory=f'train/obs_{i:06d}',
                 candidate_split='validation', camera_px='[128, 128]', resolution='.02')
            for i in range(40)]
    chosen = [row for source in sorted({r['source_dataset'] for r in rows})
              for row in sorted((r for r in rows if r['source_dataset'] == source),
                                key=lambda r: rank('pilot-gallery', r['global_id']))[:2]]
    valid = np.ones((256, 256), dtype=bool)
    free = np.zeros_like(valid)
    free[128, 128] = True
    hidden = valid & ~free
    observation = np.stack((free, np.zeros_like(valid), hidden)).astype(np.float32)
    queries = [asdict(query) for query in construct_natural_queries(
        free, hidden, valid, camera_px=[128, 128], resolution_m=.02)]
    write_csv(data / 'selected.csv', rows)
    write_json(data / 'protocol.json', {'query': {'radii_cells': [0, 10, 20]},
                                      'evaluation': {'K_primary': 32}, 'seeds': [20260910]})
    write_json(data / 'data_audit.json', {'passed': True})
    (data / 'input_queries_before_target.jsonl').write_text('\n'.join(
        json.dumps({'global_id': row['global_id'], 'queries': queries}) for row in rows) + '\n')
    packets, cases = {}, []
    for row in chosen:
        gid = row['global_id']
        selected = sorted(queries, key=lambda query: rank(gid, query['candidate_index']))
        packet = dict(observation=observation, valid=valid, hidden=hidden, target=free,
                      starts=np.array([[q['start_row'], q['start_col']] for q in selected]),
                      goals=np.array([[q['goal_row'], q['goal_col']] for q in selected]),
                      targets=np.zeros((36, 3), dtype=bool),
                      candidate_indices=np.array([q['candidate_index'] for q in selected]))
        np.savez_compressed(data / 'packets' / (gid + '.npz'), **packet)
        packets[gid] = packet
        cases.append(dict(id='pilot-' + gid, global_id=gid, source=row['source_dataset'],
                          scene=row['scene_id'], parent_group=row['parent_group'],
                          domain='indoor', cohort='new_pilot', radius_cells=10,
                          seed=20260910, sampling_seed=24260910 + int(gid.split('_')[1]),
                          candidate_index=int(packet['candidate_indices'][0]),
                          start=packet['starts'][0].tolist(), goal=packet['goals'][0].tolist(),
                          samples=32, target=False, final_test=False,
                          checkpoint_selected_on_this_cohort=False,
                          selection='two hash-ranked observations/source, first input-hash-ordered query; first sampled world',
                          split_zh='新基线 · 独立地点验证',
                          checkpoint_scope_zh='本轮从头训练；此地点未参与训练或检查点选优，属于开发验证',
                          event_probability={'correlated': 0., 'independent': 0.},
                          displayed_world_event={'correlated': False, 'independent': False},
                          panels={}))
    sealed_names = ['protocol.json', 'data_audit.json', 'selected.csv',
                    'input_queries_before_target.jsonl'] + ['packets/' + gid + '.npz' for gid in packets]
    write_json(data / 'seal.json', {'files': {name: digest(data / name) for name in sealed_names}})
    write_json(run / 'evaluation/evaluation_frozen_before_scoring.json',
               {'data_seal_sha256': digest(data / 'seal.json')})
    analysis = {'implementation_checks_passed': True, 'training_runs_completed': 9, 'final_test': False}
    write_json(run / 'analysis.json', analysis)
    write_json(site / 'data/parent_group_pilot_zh.json', analysis)
    analysis_sha = digest(run / 'analysis.json')
    write_json(run / 'verification.json', {'passed': True, 'analysis_sha256': analysis_sha})
    gallery = {'examples': cases, 'assets': [], 'checkpoint_seed': 20260910,
               'source_analysis_sha256': analysis_sha, 'saved_worlds': {}}
    # Every synthetic goal is hidden and blocked. The actual disk oracle exits
    # at those endpoints, keeping this full-size integration test inexpensive.
    worlds = np.zeros((40, 32, 256, 256), dtype=bool)
    worlds[:, :, 128, 128] = True
    for method in ('correlated', 'independent'):
        folder = run / 'evaluation' / method / '20260910'
        folder.mkdir(parents=True)
        np.savez_compressed(folder / 'worlds.npz',
                            global_ids=np.array([row['global_id'] for row in rows]), worlds=worlds)
        write_csv(folder / 'predictions_k32.csv', [
            dict(global_id=case['global_id'], candidate_index=case['candidate_index'],
                 radius_cells=10, parent_group=case['parent_group'], probability=0.) for case in cases])
        checksums = {name: digest(folder / name) for name in ('worlds.npz', 'predictions_k32.csv')}
        write_json(folder / 'report.json', {'method': method, 'seed': 20260910, 'files': checksums})
        gallery['saved_worlds'][method] = {
            'path': str((folder / 'worlds.npz').relative_to(root)), 'sha256': checksums['worlds.npz']}
    del worlds
    with patch.object(draw, 'ASSETS', site / 'assets/zh'):
        for case in cases:
            gid = case['global_id']
            for key in ('observed', 'reference', 'correlated_sample', 'correlated',
                        'independent_sample', 'independent'):
                probability = key in ('correlated', 'independent')
                if key in ('observed', 'reference'):
                    pixels = draw.categorical(observation, valid, None if key == 'observed' else free)
                    title = '机器人已看到的地图' if key == 'observed' else '数据集真实参考地图'
                    subtitle = case['source'] + ' · ' + gid
                else:
                    pixels = draw.probability(free.astype(float), valid) if probability else draw.categorical(observation, valid, free)
                    name = 'ConPath' if key.startswith('correlated') else '独立单元对照'
                    title = name + (' · 32次补全的逐格概率' if probability else ' · 第一次实际补全')
                    subtitle = '新基线 · 独立地点验证'
                path = draw.map_svg('parent-pilot/' + gid + '-' + key, pixels, title,
                                    subtitle=subtitle, points=[case['start'], case['goal']], ramp=probability)
                case['panels'][key] = path
                gallery['assets'].append({'path': path, 'sha256': digest(site / path)})
    gallery_path = site / 'data/parent_pilot_gallery_zh.json'
    write_json(gallery_path, gallery)
    return gallery_path, gallery


class GalleryIntegrationTests(unittest.TestCase):
    def test_complete_synthetic_receipt_and_rejects_changed_vote_or_selection(self):
        with tempfile.TemporaryDirectory(prefix='pilot-gallery-integration-') as directory:
            root = Path(directory)
            gallery_path, gallery = synthetic_gallery(root)
            receipt = audit_gallery(root)
            self.assertIs(receipt['passed'], True)
            self.assertEqual(receipt['cases_checked'], 10)
            self.assertEqual(receipt['svg_bitmaps_checked'], 60)
            self.assertEqual(receipt['world_event_checks'], 640)
            self.assertEqual(receipt['cached_packets_read'], 10)
            self.assertEqual(receipt['rgb_pixels_checked'], 60 * 256 * 256)
            self.assertEqual(len(receipt['panels']), 60)
            self.assertEqual(receipt['gallery_sha256'], hashlib.sha256(gallery_path.read_bytes()).hexdigest())
            self.assertIs(receipt['gpu_used'], False)
            self.assertIs(receipt['model_inference_run'], False)
            for corruption, message in [('vote', 'K32 event vote'), ('order', 'hash ranking')]:
                with self.subTest(corruption=corruption):
                    broken = copy.deepcopy(gallery)
                    if corruption == 'vote':
                        broken['examples'][0]['event_probability']['correlated'] = 1 / 32
                    else:
                        broken['examples'][0], broken['examples'][1] = broken['examples'][1], broken['examples'][0]
                    # Rewrite valid JSON while preserving all self-reported
                    # source/asset hashes; the independent checks must reject it.
                    gallery_path.write_text(json.dumps(broken, ensure_ascii=False))
                    with self.assertRaisesRegex(ValueError, message):
                        audit_gallery(root)


if __name__ == '__main__':
    unittest.main()
