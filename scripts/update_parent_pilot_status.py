#!/usr/bin/env python3
"""Publish a timestamped training snapshot, never imply GitHub Pages is a live monitor."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/parent_group_pilot_v1'


def main():
    status = json.loads((RUN / 'status.json').read_text())
    jobs = []
    for seed in (20260910, 20260911, 20260912):
        for method in ('correlated', 'independent', 'deterministic'):
            folder = RUN / 'runs' / method / str(seed)
            history = json.loads((folder / 'history.json').read_text()) if (folder / 'history.json').exists() else []
            complete = json.loads((folder / 'complete.json').read_text()) if (folder / 'complete.json').exists() else None
            jobs.append({'method': method, 'seed': seed, 'epochs': len(history), 'complete': bool(complete),
                         'latest_epoch_seconds': history[-1]['epoch_seconds'] if history else None,
                         'best_calibration_brier': history[-1]['best_brier'] if history else None})
    resource_measurement = json.loads((RUN / 'resource_verification.json').read_text()) if (RUN / 'resource_verification.json').exists() else None
    if resource_measurement:
        resource_measurement = {k: v for k, v in resource_measurement.items() if k != 'processes'}
    result = {'stage': status['stage'], 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'jobs': jobs, 'completed': sum(j['complete'] for j in jobs), 'total': 9,
              'max_epochs': 24, 'train_parents': 100, 'calibration_parents': 25, 'validation_parents': 40,
              'snapshot_not_live': True, 'gallery_uses_previous_checkpoints': True,
              'final_test': False, 'new_physical_test_packets_opened': 0,
              'resource_policy': json.loads((RUN / 'resource_policy.json').read_text()) if (RUN / 'resource_policy.json').exists() else None,
              'resource_measurement': resource_measurement}
    (ROOT / 'site/data/parent_group_pilot_status_zh.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
