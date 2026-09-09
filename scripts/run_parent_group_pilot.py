#!/usr/bin/env python3
"""Durable owned-process queue: complete all nine training runs before scoring holdout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.parent_pilot_data import sha
from scripts.train_flatlands_conpath import _atomic_json

OUT = ROOT / 'results/parent_group_pilot_v1'
STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    protocol = json.loads((OUT / 'data/protocol.json').read_text())
    jobs = [(m, s) for s in protocol['seeds'] for m in ('correlated', 'independent', 'deterministic')]
    paths = [ROOT / 'scripts/train_parent_group_pilot.py', OUT / 'data/seal.json']
    launch = {'jobs': jobs, 'workers': args.workers, 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
              'created_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable}
    if (OUT / 'launch.json').exists():
        old = json.loads((OUT / 'launch.json').read_text())
        if old['source_sha256'] != launch['source_sha256']:
            raise ValueError('Frozen training source/data changed')
    else:
        _atomic_json(OUT / 'launch.json', launch)
    active, failed = {}, []
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while jobs or active:
            if STOP or (OUT / 'STOP').exists():
                for p, _, _ in active.values():
                    p.terminate()
                raise SystemExit(130)
            for key, (p, handle, cmd) in list(active.items()):
                code = p.poll()
                if code is not None:
                    handle.close(); del active[key]
                    if code != 0:
                        failed.append({'job': key, 'exit_code': code})
            if failed:
                for p, _, _ in active.values():
                    p.terminate()
                raise RuntimeError(f'Pilot job failed; queue stopped: {failed}')
            while jobs and len(active) < args.workers:
                method, seed = jobs.pop(0)
                folder = OUT / 'runs' / method / str(seed)
                if (folder / 'complete.json').exists():
                    continue
                cmd = [sys.executable, str(ROOT / 'scripts/train_parent_group_pilot.py'), '--method', method, '--seed', str(seed)]
                if (folder / 'latest.pt').exists():
                    cmd.append('--resume')
                handle = (OUT / f'{method}-{seed}.log').open('a')
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                active[f'{method}/{seed}'] = p, handle, cmd
            _atomic_json(OUT / 'status.json', {'stage': 'training', 'active': {k: {'pid': v[0].pid, 'command': v[2]} for k, v in active.items()},
                         'pending': jobs, 'failed': failed, 'completed': [str(p.relative_to(OUT)) for p in sorted((OUT / 'runs').glob('*/*/complete.json'))],
                         'timestamp_utc': datetime.now(timezone.utc).isoformat()})
            if active:
                time.sleep(10)
        _atomic_json(OUT / 'status.json', {'stage': 'evaluating', 'active': {}, 'pending': [], 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
        subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_parent_group_pilot.py')], cwd=ROOT, check=True)
        _atomic_json(OUT / 'status.json', {'stage': 'baseline_complete', 'active': {}, 'pending': [], 'result': 'analysis.json',
                     'timestamp_utc': datetime.now(timezone.utc).isoformat()})
    finally:
        for p, handle, _ in active.values():
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=50)
            handle.close()


if __name__ == '__main__':
    main()
