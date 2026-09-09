#!/usr/bin/env python3
"""Run exactly two candidate seeds under the user's existing resource budget."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.train_flatlands_conpath import _atomic_json

OUT = ROOT / 'results/coherent_parent_pilot_v1'
STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    lock = (OUT / '.supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / 'results/parent_group_pilot_v1/status.json').read_text())['stage'] == 'baseline_complete'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['seeds'] == [20260910, 20260911]
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    active, completed = {}, []
    try:
        for seed in protocol['seeds']:
            if STOP or (OUT / 'STOP').exists():
                raise SystemExit(130)
            folder = OUT / 'runs' / str(seed)
            if (folder / 'complete.json').exists():
                completed.append(seed)
                continue
            cmd = [sys.executable, str(ROOT / 'scripts/train_coherent_parent_pilot.py'), '--seed', str(seed)]
            if (folder / 'latest.pt').exists():
                cmd.append('--resume')
            handle = (OUT / f'{seed}.log').open('a')
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            active[seed] = process, handle, cmd
        while active:
            if STOP or (OUT / 'STOP').exists():
                _atomic_json(OUT / 'status.json', {'stage': 'paused_by_user', 'completed': completed,
                             'timestamp_utc': datetime.now(timezone.utc).isoformat()})
                raise SystemExit(130)
            for seed, (process, handle, _) in list(active.items()):
                code = process.poll()
                if code is not None:
                    handle.close(); del active[seed]
                    if code:
                        _atomic_json(OUT / 'status.json', {'stage': 'failed', 'failed_seed': seed, 'exit_code': code,
                                     'completed': completed, 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
                        raise RuntimeError(f'Candidate seed {seed} failed with exit code {code}')
                    completed.append(seed)
            _atomic_json(OUT / 'status.json', {'stage': 'training', 'completed': completed, 'total': 2,
                         'active': {str(s): {'pid': p.pid, 'command': cmd} for s, (p, _, cmd) in active.items()},
                         'resource_policy': protocol['resource_policy'], 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
            if active:
                time.sleep(10)
        _atomic_json(OUT / 'status.json', {'stage': 'evaluating', 'active': {}, 'completed': completed,
                     'timestamp_utc': datetime.now(timezone.utc).isoformat()})
        subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_coherent_parent_pilot.py')], cwd=ROOT, check=True)
        _atomic_json(OUT / 'status.json', {'stage': 'complete', 'active': {}, 'completed': completed,
                     'result': 'analysis.json', 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=50)
            handle.close()


if __name__ == '__main__':
    main()
