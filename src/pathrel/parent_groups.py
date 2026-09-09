"""Conservative upstream physical-place identities, never scan-ID fallbacks."""
import hashlib
import re
from collections import defaultdict


def parent_group(source, scene, *, arkit_visits=None, rscan_references=None):
    patterns = {
        'Matterport3D': r'([A-Za-z0-9]+)_region_segmentations_region\d+\.ply',
        'ScanNet': r'(scene\d{4})_\d{2}',
        'ZInD': r'(\d{4})_floor_-?\d+_pano_\d+',
        'ScanNet++': r'([a-f0-9]{10})',
    }
    if source in patterns:
        match = re.fullmatch(patterns[source], scene)
        parent = match.group(1) if match else None
    elif source == 'ARKitScenes':
        match = re.fullmatch(r'(\d+)_3dod_mesh', scene)
        parent = (arkit_visits or {}).get(match.group(1)) if match else None
    elif source == '3RScan':
        parent = (rscan_references or {}).get(scene)
    else:
        parent = None
    if parent is None or str(parent).strip().lower() in ('', 'na', 'nan', 'none'):
        return None
    return f'{source}:{parent}'


def grouped_development_assignment(groups, salt='conpath-parent-development-v1-20260909'):
    """Source-stratified hash ranking; ~80/10/10 groups, not image counts.

    All groups must be resolved upstream physical places. These partitions are
    development candidates, not a replacement for an untouched final test.
    """
    by_source = defaultdict(set)
    for group in groups:
        if not group or ':' not in group:
            raise ValueError('A resolved, source-qualified parent group is required')
        by_source[group.split(':', 1)[0]].add(group)
    result = {}
    for source, members in sorted(by_source.items()):
        ranked = sorted(members, key=lambda g: (hashlib.sha256(f'{salt}|{g}'.encode()).hexdigest(), g))
        n = len(ranked)
        if n < 10:
            raise ValueError(f'{source}: insufficient independent groups for 80/10/10 development split')
        n_val = n_cal = max(1, int(n * .1))
        for i, group in enumerate(ranked):
            result[group] = 'validation' if i < n_val else 'calibration' if i < n_val + n_cal else 'train'
    return result
