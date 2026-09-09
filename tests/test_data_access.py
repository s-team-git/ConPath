import unittest
from pathrel.data_access import require_development_packet


class DataAccessTest(unittest.TestCase):
    def test_provenance_does_not_unlock_physical_test(self):
        with self.assertRaises(ValueError):
            require_development_packet({'global_id': 'x', 'archive_split': 'test', 'provenance_split': 'train',
                                        'candidate_split': 'train', 'parent_group': 'source:place'})

    def test_unknown_identity_is_not_a_training_group(self):
        with self.assertRaises(ValueError):
            require_development_packet({'archive_split': 'train', 'candidate_split': 'train'})
        require_development_packet({'archive_split': 'train', 'candidate_split': 'calibration', 'parent_group': 'source:place'})
