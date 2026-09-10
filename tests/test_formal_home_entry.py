import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_model_home as home


class FormalHomeEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.site = self.root/'site'; (self.site/'data').mkdir(parents=True)
        self.addCleanup(patch.stopall)
        patch.object(home, 'ROOT', self.root).start(); patch.object(home, 'SITE', self.site).start()
        self.version = self.site/'assets/formal/snapshot_hash'; self.version.mkdir(parents=True)
        self.protocol_path = self.root/'results/flatlands_external_formal_protocol_v1/protocol.json'
        self.protocol_path.parent.mkdir(parents=True)
        self.protocol = {'source_hashes': {}, 'test_lock': {'final_test_locked': True, 'location_6_locked': True},
                         'seeds': [20260831, 20260901, 20260902], 'sampling': {'K': 4}, 'query': {'radii_cells': [0, 10, 20]},
                         'partition': {'train': 2, 'calibration': 1, 'validation': 1},
                         'scenes': {'train': [{'parent_group': 'a'}, {'parent_group': 'a'}, {'parent_group': 'b'}, {'parent_group': 'b'}],
                                    'calibration': [{'parent_group': 'c'}], 'validation': [{'parent_group': 'd'}]}}
        self.protocol_path.write_text(json.dumps(self.protocol))
        for path, content in ((self.site/'formal.html', '<html>diagnostic</html>'), (self.version/'formal.html', '<html>diagnostic</html>'),
                              (self.version/'formal.css', 'style'), (self.version/'builder_source.py', 'builder'), (self.version/'map.png', 'synthetic')):
            path.write_text(content)
        asset = {'path': 'map.png', 'sha256': home.sha(self.version/'map.png')}
        self.data = {'builder_sha256': home.sha(self.version/'builder_source.py'), 'validation_only': True, 'paper_main_or_superiority_authorized': False,
                     'final_test_locked': True, 'location_6_locked': True, 'new_physical_test_images_opened': 0, 'training_loss_images_published': False,
                     'snapshot': 'snapshot', 'reproducibility_sha256': 'repro', 'protocol_sha256': home.sha(self.protocol_path),
                     'registered_seeds': self.protocol['seeds'], 'common_K': 4, 'radii_cells': [0, 10, 20], 'figures': [{'files': [asset]}],
                     'methods': {'lama': {'verified_seeds': [20260831], 'runs': [{'audit_sha256': 'audit', 'metrics_sha256': 'metrics', 'secret_metric': .13}]}},
                     'stage_zh': '1000步诊断', 'selected_group': ['smoke', 1000], 'snapshot_created_utc': 'fixed-time'}
        self.receipt = {'version_url': 'assets/formal/snapshot_hash', 'validation_only': True, 'page_sha256': home.sha(self.site/'formal.html'),
                        'stylesheet_sha256': home.sha(self.version/'formal.css'), 'builder_sha256': home.sha(self.version/'builder_source.py'),
                        'source_snapshot': 'snapshot', 'source_reproducibility_sha256': 'repro', 'assets': [asset]}
        self.refresh()

    def refresh(self):
        (self.version/'diagnostics.json').write_text(json.dumps(self.data))
        self.receipt['data_sha256'] = home.sha(self.version/'diagnostics.json')
        content = json.dumps(self.receipt)
        (self.site/'data/formal_current.json').write_text(content)
        (self.version/'build_receipt.json').write_text(content)

    def test_verified_entry_exposes_counts_and_stage_without_importing_results(self):
        value, status = home.verified_formal_entry()
        self.assertEqual(status, 'verified')
        self.assertEqual(value['counts'], {'train': 2, 'calibration': 1, 'validation': 1})
        self.assertNotIn('secret_metric', str(value))
        self.assertNotIn('methods', value)

    def test_missing_or_changed_page_does_not_get_entry(self):
        (self.site/'formal.html').write_text('changed')
        self.assertIsNone(home.verified_formal_entry()[0])
        (self.site/'formal.html').unlink()
        self.assertIsNone(home.verified_formal_entry()[0])

    def test_changed_diagnostic_or_protocol_is_withheld(self):
        (self.version/'diagnostics.json').write_text('{}')
        self.assertIsNone(home.verified_formal_entry()[0])
        self.refresh()
        self.protocol_path.write_text(self.protocol_path.read_text()+'\n')
        self.assertIsNone(home.verified_formal_entry()[0])

    def test_wrong_test_scope_is_rejected_even_when_hashes_are_updated(self):
        self.data['location_6_locked'] = False; self.refresh()
        self.assertIsNone(home.verified_formal_entry()[0])

    def test_changed_asset_and_mismatched_partition_are_withheld(self):
        (self.version/'map.png').write_text('changed')
        self.assertIsNone(home.verified_formal_entry()[0])
        (self.version/'map.png').write_text('synthetic')
        self.protocol['partition']['train'] = 4
        self.protocol_path.write_text(json.dumps(self.protocol)); self.data['protocol_sha256'] = home.sha(self.protocol_path); self.refresh()
        self.assertIsNone(home.verified_formal_entry()[0])

    def test_current_direction_is_external_matrix_with_locked_test(self):
        value, _ = home.verified_formal_entry()
        text = str(home.formal_work_copy(value))
        for required in ('1000', '5000', 'LaMa BEV adaptation', 'FM+XAttn literature reimplementation', '20260831', 'location_6'):
            self.assertIn(required, text)
        for forbidden in ('历史融合', '注意力', '当前不追加训练', '室外'):
            self.assertNotIn(forbidden, text)


if __name__ == '__main__':
    unittest.main()
