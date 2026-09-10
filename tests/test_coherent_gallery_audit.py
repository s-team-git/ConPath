"""Synthetic corruption checks for paired candidate-gallery provenance."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_coherent_gallery import audit_case_extension


class CandidateGalleryProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            'id': 'pilot-obs_000001', 'global_id': 'obs_000001',
            'cohort': 'new_pilot', 'seed': 20260910, 'samples': 32,
            'candidate_index': 7, 'radius_cells': 10,
            'start': [20, 20], 'goal': [40, 50], 'target': False,
            'panels': {'observed': 'original-observed.svg', 'reference': 'original-reference.svg',
                       'correlated': 'original-probability.svg', 'correlated_sample': 'original-world.svg'},
            'event_probability': {'correlated': .5}, 'displayed_world_event': {'correlated': True},
        }
        self.candidate = copy.deepcopy(self.original)
        self.candidate.update({
            'candidate_model_available': True,
            'candidate_scope_zh': '此地点未参与训练或检查点选优；其验证反馈已用于选择改进方向，属于开发比较，非最终测试。',
            'validation_reused_for_model_development': True,
        })
        self.candidate['panels'].update({
            'coherent_categorical': 'assets/zh/coherent-pilot/obs_000001-coherent_categorical.svg',
            'coherent_categorical_sample': 'assets/zh/coherent-pilot/obs_000001-coherent_categorical_sample.svg',
        })
        self.candidate['event_probability']['coherent_categorical'] = .25
        self.candidate['displayed_world_event']['coherent_categorical'] = False

    def test_paired_extension_preserves_original_without_mutating_arguments(self):
        before = copy.deepcopy(self.candidate)
        audit_case_extension(self.candidate, self.original)
        self.assertEqual(self.candidate, before)

    def test_rejects_case_query_label_and_comparator_substitution(self):
        for path, value in [
            (('global_id',), 'obs_000002'), (('candidate_index',), 8),
            (('goal',), [40, 51]), (('radius_cells',), 0), (('target',), True),
            (('panels', 'observed'), 'different-input.svg'),
            (('panels', 'reference'), 'different-reference.svg'),
            (('panels', 'coherent_categorical_sample'), 'assets/zh/coherent-pilot/obs_000002-coherent_categorical_sample.svg'),
            (('event_probability', 'correlated'), .25),
            (('displayed_world_event', 'correlated'), False),
            (('validation_reused_for_model_development',), False),
            (('candidate_scope_zh',), '最终测试集'),
        ]:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.candidate)
                target = changed if len(path) == 1 else changed[path[0]]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    audit_case_extension(changed, self.original)

    def test_failed_rerun_invalidates_an_existing_passing_receipt(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='coherent-gallery-missing-inputs-') as directory:
            fake_root = Path(directory)
            receipt = fake_root/'results/coherent_parent_pilot_v1/gallery_verification.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({'passed': True, 'stale': True}))
            result = subprocess.run([sys.executable, str(root/'scripts/audit_coherent_gallery.py'),
                                     '--root', directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            failure = json.loads(receipt.read_text())
            self.assertIs(failure['passed'], False)
            self.assertNotIn('stale', failure)
            self.assertIn('FileNotFoundError', failure['error'])


if __name__ == '__main__':
    unittest.main()
