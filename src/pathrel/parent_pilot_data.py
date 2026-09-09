"""Guarded parent-place development packets, with target-blind query inclusion."""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_access import require_development_packet


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_rank(*values):
    return hashlib.sha256('|'.join(map(str, ('parent-pilot-v1-20260909', *values))).encode()).hexdigest()


def choose_parent_subset(rows, sizes=None):
    sizes = sizes or {'train': 20, 'calibration': 5, 'validation': 8}
    sources = sorted({r['source_dataset'] for r in rows})
    selected = []
    for split, count in sizes.items():
        for source in sources:
            members = [r for r in rows if r['candidate_split'] == split and r['source_dataset'] == source]
            parents = sorted({r['parent_group'] for r in members}, key=lambda p: stable_rank(split, source, p))
            if len(parents) < count:
                raise ValueError(f'Insufficient independent parents: {source}/{split}')
            for parent in parents[:count]:
                selected.append(min((r for r in members if r['parent_group'] == parent), key=lambda r: stable_rank(parent, r['global_id'])))
    validate_partition(selected)
    return selected


def validate_partition(rows):
    ids, groups = set(), {}
    for row in rows:
        require_development_packet(row)
        # Pilot v1 is stricter than general development: physical train only.
        if row['archive_split'] != 'train' or row['packet_directory'] != 'train/' + row['global_id']:
            raise ValueError('Pilot packets must physically belong to archive train')
        if row['global_id'] in ids:
            raise ValueError('Duplicate observation in pilot selection')
        ids.add(row['global_id'])
        parent = row['parent_group']
        if parent in groups and groups[parent] != row['candidate_split']:
            raise ValueError('Parent place crosses pilot partitions')
        groups[parent] = row['candidate_split']


def canonical_d4_digest(array):
    array = np.asarray(array, dtype=np.uint8)
    return min(hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
               for flip in (array, np.flip(array, axis=-1))
               for v in (np.rot90(flip, k, axes=(-2, -1)) for k in range(4)))


@dataclass
class PilotSample:
    row: dict
    observation: np.ndarray
    valid: np.ndarray
    target: np.ndarray
    hidden: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    targets: np.ndarray
    candidate_indices: np.ndarray


def load_pilot(root, split):
    root = Path(root)
    seal = json.loads((root / 'seal.json').read_text())
    for name in ('selected.csv', 'protocol.json', 'data_audit.json'):
        if sha(root / name) != seal['files'][name]:
            raise ValueError(f'Frozen pilot changed: {name}')
    audit = json.loads((root / 'data_audit.json').read_text())
    if not audit['passed']:
        raise ValueError('Pilot data gate did not pass')
    rows = list(csv.DictReader((root / 'selected.csv').open()))
    validate_partition(rows)
    result = []
    for row in rows:
        if row['candidate_split'] != split:
            continue
        relative = 'packets/' + row['global_id'] + '.npz'
        if sha(root / relative) != seal['files'][relative]:
            raise ValueError('Cached packet checksum changed')
        with np.load(root / relative, allow_pickle=False) as z:
            result.append(PilotSample(row, **{k: z[k] for k in ('observation', 'valid', 'target', 'hidden', 'starts', 'goals', 'targets', 'candidate_indices')}))
    if not result:
        raise ValueError('Empty development split')
    return result


def collate_pilot(samples):
    count = max(len(s.starts) for s in samples)
    starts = np.zeros((len(samples), count, 2), dtype=np.int64)
    goals = starts.copy()
    targets = np.zeros((len(samples), count, 3), dtype=bool)
    mask = np.zeros((len(samples), count), dtype=bool)
    for i, s in enumerate(samples):
        n = len(s.starts)
        starts[i, :n], goals[i, :n], targets[i, :n], mask[i, :n] = s.starts, s.goals, s.targets, True
    return {'observation': np.stack([s.observation for s in samples]), 'valid_support_mask': np.stack([s.valid for s in samples]),
            'target_free': np.stack([s.target for s in samples]), 'loss_mask': np.stack([s.hidden for s in samples]),
            'starts': starts, 'goals': goals, 'reachability_targets': targets, 'query_mask': mask}
