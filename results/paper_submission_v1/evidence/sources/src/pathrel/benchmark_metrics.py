"""Masked map metrics; paper-reported scores still require matching data protocols."""
import numpy as np


def masked_map_metrics(worlds, target, mask):
    """Score K binary worlds on unknown valid cells, without choosing a sample.

    Empty union has IoU 1. An empty evaluation mask yields None metrics. MES
    uses the off-diagonal U-statistic in FlatLands Eq.5, not the biased K^2
    denominator; K=1 has no estimable stochastic MES. Oracle best-IoU is named
    separately and never used to select an event prediction.
    """
    worlds, target, mask = map(np.asarray, (worlds, target, mask))
    if worlds.ndim != 3 or target.shape != worlds.shape[1:] or mask.shape != target.shape or len(worlds) < 1:
        raise ValueError('Expected KxHxW worlds and aligned target/mask')
    for value in (worlds, target, mask):
        if not np.isfinite(value).all() or not np.isin(value, [0, 1]).all():
            raise ValueError('Maps and mask must be finite binary arrays')
    keys = ['first_umr', 'first_iou', 'mean_iou', 'oracle_best_iou',
            'mean_blocked_iou', 'sample_vote_cell_brier', 'masked_energy_score']
    if not mask.any():
        return {**dict.fromkeys(keys), 'hidden_cells': 0, 'samples': len(worlds)}
    w, y = worlds[:, mask.astype(bool)].astype(bool), target[mask.astype(bool)].astype(bool)

    def iou(a, b):
        union = np.count_nonzero(a | b)
        return float(np.count_nonzero(a & b) / union) if union else 1.0

    overlaps = np.array([iou(v, y) for v in w])
    k = len(w)
    pair_term = sum(1-iou(w[i], w[j]) for i in range(k) for j in range(i+1, k)) / (k*(k-1)) if k > 1 else None
    return {'first_umr': float(np.mean(w[0] != y)), 'first_iou': float(overlaps[0]),
            'mean_iou': float(overlaps.mean()), 'oracle_best_iou': float(overlaps.max()),
            'mean_blocked_iou': float(np.mean([iou(~v, ~y) for v in w])),
            'sample_vote_cell_brier': float(np.mean((w.mean(0)-y)**2)),
            'masked_energy_score': float((1-overlaps).mean()-pair_term) if pair_term is not None else None,
            'hidden_cells': int(mask.sum()), 'samples': k}


def require_comparable_protocols(left, right):
    """Reject numeric rankings when an essential protocol field is absent/different."""
    fields = ('task', 'dataset_revision', 'split_manifest_sha256', 'input_information',
              'evaluation_mask', 'metric', 'aggregation', 'sample_selection', 'samples',
              'postprocessing', 'metric_parameters', 'training_protocol')
    differences = [key for key in fields if key not in left or key not in right or left[key] != right[key]]
    if differences:
        raise ValueError('Incomparable protocols: ' + ', '.join(differences))

