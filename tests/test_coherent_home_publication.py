"""Publication gates prevent stale/mismatched experiments from reaching the homepage."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('coherent_home_builder', ROOT/'scripts/build_model_home.py')
HOME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOME)


class CoherentPublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.site = Path(self.directory.name)
        (self.site/'data').mkdir()
        self.addCleanup(patch.stopall)
        patch.object(HOME, 'SITE', self.site).start()
        self.baseline_gallery_path = self.site/'data/baseline_gallery.json'
        self.baseline_result_path = self.site/'data/baseline.json'
        self.baseline_result_path.write_text('{}')
        rows = []
        for index in range(10):
            rows.append(dict(id=f'pilot-{index}', global_id=str(index), source='synthetic',
                             scene=str(index), parent_group=str(index), domain='indoor', cohort='new_pilot',
                             radius_cells=10, seed=20260910, candidate_index=index,
                             start=[1, 2], goal=[3, 4], samples=32, target=False, final_test=False,
                             checkpoint_selected_on_this_cohort=False, panels={'observed':'input.svg'},
                             event_probability={'correlated':0.5}, displayed_world_event={'correlated':False}))
        self.baseline_gallery_path.write_text(json.dumps({'examples':rows}))
        self.pilot = {'gallery': {'examples':rows}, 'result':{'validation_parents':40, 'validation_events':1545},
                      'source_paths':[self.baseline_gallery_path, self.baseline_result_path]}
        self.result = dict(id='coherent_parent_pilot_v1', training_seeds=[20260910,20260911],
                           final_test=False, new_physical_test_reads=0, validation_reused_for_model_development=True,
                           no_direct_published_paper_ranking=True, development_screen_passed=True,
                           validation_parents=40, validation_events=1545,
                           baseline_analysis_sha256=HOME.sha(self.baseline_result_path), methods={})
        for model in HOME.COHERENT_METHODS:
            self.result['methods'][model] = {'seeds':[20260910,20260911], 'samples':1 if model=='deterministic' else 32,
                                            'brier':{'mean':0.15,'values':[0.1,0.2]},
                                            'risk30':{'mean':0.15,'values':[0.1,0.2]}, 'map':{'mean_iou':0.6}}
        self.gallery = dict(examples=copy.deepcopy(rows), checkpoint_seed=20260910,
                            training_seeds=[20260910,20260911], final_test=False,
                            validation_reused_for_model_development=True,
                            source_baseline_gallery_sha256=HOME.sha(self.baseline_gallery_path),
                            saved_worlds={'coherent_categorical':{'path':'saved/worlds.npz','sha256':'a'*64}}, assets=[])
        for index, row in enumerate(self.gallery['examples']):
            row['event_probability']['coherent_categorical'] = 0.25
            row['displayed_world_event']['coherent_categorical'] = False
            for model in ('coherent_categorical', 'coherent_categorical_sample'):
                name = f'{index}-{model}.svg'
                (self.site/name).write_text('<svg/>')
                row['panels'][model] = name
                self.gallery['assets'].append({'path':name,'sha256':HOME.sha(self.site/name)})
        (self.site/'coherent.html').write_text('<html lang="zh-CN"></html>')
        self.publish_metadata()

    def publish_metadata(self):
        result_path = self.site/'data/coherent_parent_pilot_zh.json'
        result_path.write_text(json.dumps(self.result))
        digest = HOME.sha(result_path)
        self.gallery['source_analysis_sha256'] = digest
        gallery_path = self.site/'data/coherent_pilot_gallery_zh.json'
        gallery_path.write_text(json.dumps(self.gallery))
        (self.site/'data/coherent_parent_pilot_verification.json').write_text(json.dumps({'passed':True,'analysis_sha256':digest}))
        (self.site/'data/coherent_pilot_gallery_verification.json').write_text(json.dumps({'passed':True,'analysis_sha256':digest,'gallery_sha256':HOME.sha(gallery_path)}))

    def assert_withheld(self):
        result, state = HOME.verified_coherent(self.pilot)
        self.assertIsNone(result)
        self.assertNotEqual(state, 'verified')

    def test_matching_two_seed_report_is_accepted(self):
        self.assertEqual(HOME.verified_coherent(self.pilot)[1], 'verified')

    def test_missing_image_audit_is_withheld(self):
        (self.site/'data/coherent_pilot_gallery_verification.json').unlink()
        self.assert_withheld()

    def test_stale_score_audit_is_withheld(self):
        path = self.site/'data/coherent_parent_pilot_zh.json'
        path.write_text(path.read_text()+'\n')
        self.assert_withheld()

    def test_third_seed_in_comparison_is_withheld(self):
        self.result['methods']['correlated']['seeds'].append(20260912)
        self.publish_metadata()
        self.assert_withheld()

    def test_changed_query_is_withheld_even_with_current_hashes(self):
        self.gallery['examples'][0]['goal'] = [9, 9]
        self.publish_metadata()
        self.assert_withheld()

    def test_changed_map_asset_is_withheld(self):
        (self.site/self.gallery['assets'][0]['path']).write_text('<svg>changed</svg>')
        self.assert_withheld()

    def test_final_test_or_hidden_development_reuse_is_withheld(self):
        for field, value in [('final_test',True), ('validation_reused_for_model_development',False)]:
            with self.subTest(field=field):
                old = self.result[field]
                self.result[field] = value
                self.publish_metadata()
                self.assert_withheld()
                self.result[field] = old

    def test_finished_copy_preserves_actual_completion_date(self):
        candidate = {'stage':'complete','completed_utc':'2026-09-09T10:28:51Z',
                     'timestamp_utc':'2026-09-10T14:00:00Z', 'active_training_processes':0,
                     'resource_released':True}
        note, heading, description, _ = HOME.training_copy(None, True, candidate, True)
        self.assertIn('2026-09-09 10:28 UTC', note)
        self.assertNotIn('2026-09-10', note)
        self.assertIn('发布已完成', heading)
        self.assertIn('显存已释放', description)
        self.assertIn('当前不追加训练', description)


if __name__ == '__main__':
    unittest.main()
