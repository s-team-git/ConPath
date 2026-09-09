#!/usr/bin/env python3
"""Public timestamped state of the bounded two-seed candidate."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/coherent_parent_pilot_v1'


def main():
    status = json.loads((RUN / 'status.json').read_text())
    protocol = json.loads((RUN / 'protocol.json').read_text())
    jobs = []
    for seed in protocol['seeds']:
        folder = RUN / 'runs' / str(seed)
        history = json.loads((folder / 'history.json').read_text()) if (folder / 'history.json').exists() else []
        jobs.append({'seed': seed, 'epochs': len(history), 'complete': (folder / 'complete.json').is_file(),
                     'best_calibration_brier': history[-1]['best_brier'] if history else None})
    result = {'stage': status['stage'], 'jobs': jobs, 'total': 2, 'completed': sum(j['complete'] for j in jobs),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'snapshot_not_live': True,
              'resource_policy': protocol['resource_policy'], 'model_change_zh': '固定9×9空间相关类别采样',
              'final_test': False, 'validation_reused_for_model_development': True}
    (ROOT / 'site/data/coherent_pilot_status_zh.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
