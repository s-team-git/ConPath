#!/usr/bin/env python3
"""Independent new-process replay of every fixed external short-profile output."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pathrel.cogniplan import configure_reproducible_cuda
from pathrel.flow_matching import FlowUNet, sample_heun
from pathrel.lama import LaMaBEV


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_tensors_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_tensors_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_tensors_finite(v) for v in value)
    return True


def main():
    out = Path('results/flatlands_external_publication_20260909')
    out.mkdir(parents=True, exist_ok=True)
    configure_reproducible_cuda(); torch.set_num_threads(4)
    checks = []
    def check(name, result):
        checks.append({'check': name, 'passed': bool(result)})
        if not result: raise ValueError(name)
    replay = []
    inputs_hashes = []
    for method in ['lama', 'flow']:
        root = Path(f'results/flatlands_{method}_profile_v1')
        report = json.loads((root / 'report.json').read_text())
        for name, record in report['outputs'].items():
            check(f'{method}/{name} hash', sha(root / name) == record['sha256'])
        for name, expected in report['source_hashes'].items():
            check(f'{method} source {name}', sha(Path(name)) == expected)
        check(f'{method} engineering-only test lock', report['engineering_only'] and report['test_assets_opened'] is False)
        inputs_hashes.append(sha(root / 'fixed_train_observations.csv'))
        inputs = np.load(root / 'fixed_train_inputs.npz')
        predictions = np.load(root / 'predictions.npz')
        # Trusted locally produced checkpoint, verified against its frozen report above.
        checkpoint = torch.load(root / 'profile_checkpoint.pt', weights_only=False, map_location='cpu')
        check(f'{method} checkpoint parameters and optimizer states finite', all_tensors_finite(checkpoint))
        model = (LaMaBEV() if method == 'lama' else FlowUNet()).cuda()
        model.load_state_dict(checkpoint['model'], strict=True); model.eval()
        drift = []
        for i in range(32):
            c = torch.from_numpy(inputs['condition'][i:i+1]).cuda()
            calls = []
            hook = model.register_forward_pre_hook(lambda module, values: calls.append(int(values[0].shape[0])))
            with torch.no_grad():
                if method == 'lama':
                    values = model(c); worlds = values > .5
                else:
                    values, worlds, _ = sample_heun(model, c, seed=20260909+i)
            hook.remove()
            p, w = values[0].cpu().numpy(), worlds[0].cpu().numpy()
            difference = float(np.max(np.abs(p - predictions[f'values_{i}'])))
            drift.append(difference)
            check(f'{method} sample{i} raw exact replay', difference == 0)
            check(f'{method} sample{i} worlds exact replay', np.array_equal(w, predictions[f'worlds_{i}']))
            known = inputs['condition'][i, 1] == 0
            expected = inputs['condition'][i, 0].astype(bool)
            check(f'{method} sample{i} known/support', np.all(w[:, known] == expected[known]))
            check(f'{method} sample{i} actual forward budget', (len(calls), sum(calls)) == ((1, 1) if method == 'lama' else (50, 400)))
            if i % 8 == 0: print(f'{method}: replay {i+1}/32, drift={difference}', flush=True)
        replay.append({'method': method, 'observations': 32, 'raw_max_absolute_drift': max(drift),
                       'worlds_replayed': 32 if method == 'lama' else 128})
        del model, checkpoint, inputs, predictions
        torch.cuda.empty_cache()
    check('identical frozen32 observation list across methods', inputs_hashes[0] == inputs_hashes[1])
    for method in ['lama', 'flow']:
        root = Path(f'results/flatlands_{method}_batch64_v1')
        report = json.loads((root / 'report.json').read_text())
        check(f'{method} batch64 actual100 updates', report['effective_batch'] == 64 and report['optimizer_updates_measured'] == 100)
        check(f'{method} batch64 no test/formal run', not report['test_assets_opened'] and not report['formal_training'])
        for name, expected in report['inputs'].items(): check(f'{method} batch64 source {name}', sha(Path(name)) == expected)
        for name, record in report['outputs'].items(): check(f'{method} batch64 output {name}', sha(root / name) == record['sha256'])
        checkpoint = torch.load(root / 'profile_checkpoint.pt', weights_only=False, map_location='cpu')
        check(f'{method} batch64 parameters and optimizer states finite', all_tensors_finite(checkpoint))
        del checkpoint
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': all(c['passed'] for c in checks),
              'check_count': len(checks), 'checks': checks, 'replay': replay, 'test_assets_opened': False,
              'script_sha256': sha(Path(__file__))}
    (out / 'replay_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2), flush=True)


if __name__ == '__main__': main()
