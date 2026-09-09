import unittest
from pathrel.parent_groups import parent_group, grouped_development_assignment


class ParentGroupsTest(unittest.TestCase):
    def test_siblings_share_physical_place(self):
        for source, a, b, parent in [
            ('Matterport3D', '17DRP5sb8fy_region_segmentations_region0.ply', '17DRP5sb8fy_region_segmentations_region8.ply', '17DRP5sb8fy'),
            ('ScanNet', 'scene0517_00', 'scene0517_01', 'scene0517'),
            ('ZInD', '0359_floor_01_pano_18', '0359_floor_02_pano_05', '0359'),
            ('ZInD', '1269_floor_-1_pano_46', '1269_floor_01_pano_05', '1269'),
        ]:
            with self.subTest(source=source, scene=a):
                self.assertEqual(parent_group(source, a), f'{source}:{parent}')
                self.assertEqual(parent_group(source, a), parent_group(source, b))

    def test_official_mapping_required_for_visits_and_rescans(self):
        self.assertIsNone(parent_group('ARKitScenes', '123_3dod_mesh'))
        self.assertIsNone(parent_group('ARKitScenes', '123_3dod_mesh', arkit_visits={'123': 'NA'}))
        self.assertEqual(parent_group('ARKitScenes', '123_3dod_mesh', arkit_visits={'123': 'visit4'}), 'ARKitScenes:visit4')
        self.assertIsNone(parent_group('3RScan', 'uuid'))
        self.assertEqual(parent_group('3RScan', 'rescan', rscan_references={'rescan': 'initial'}), '3RScan:initial')
        self.assertIsNone(parent_group('Matterport3D', 'bad-name'))

    def test_source_identity_and_partition_invariance(self):
        groups = [f'{s}:{i}' for s in ['a', 'b'] for i in range(100)]
        a = grouped_development_assignment(groups)
        self.assertEqual(a, grouped_development_assignment(list(reversed(groups)) + groups[:5]))
        for source in ['a', 'b']:
            self.assertEqual(sum(v == 'train' for k, v in a.items() if k.startswith(source + ':')), 80)
            self.assertEqual(sum(v == 'validation' for k, v in a.items() if k.startswith(source + ':')), 10)
        with self.assertRaises(ValueError):
            grouped_development_assignment([None])


if __name__ == '__main__':
    unittest.main()
