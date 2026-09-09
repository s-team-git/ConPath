#!/usr/bin/env python3
"""Measure 100 actual effective-batch64 updates; do not extrapolate microbatch speed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from profile_flatlands_external import gpu_state, load_fixed_training, sha, sync
from pathrel.cogniplan import configure_reproducible_cuda
from pathrel.external_training import flow_accumulated_update, lama_accumulated_update
from pathrel.flow_matching import FlowUNet
from pathrel.lama import LaMaBEV, make_discriminator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=['lama', 'flow'])
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    configure_reproducible_cuda(); torch.set_num_threads(4); torch.manual_seed(20260909)
    microbatch = 16 if args.method == 'lama' else 4
    config = {'method': args.method, 'purpose': 'fixed32 train-only actual effective-batch64 resource measurement',
              'formal_training': False, 'test_assets_opened': False, 'seed': 20260909,
              'effective_batch': 64, 'microbatch': microbatch, 'accumulation': 64 // microbatch,
              'warmup_optimizer_updates': 5, 'measured_optimizer_updates': 100,
              'sampling': 'two independently permuted passes over frozen32 per optimizer update; profiling only',
              'optimizer': 'AdamW lr1e-4 wd.01 betas(.9,.999), constant short-profile LR',
              'batch_norm_note': 'LaMa native BN sees microbatch16, with separate running-stat updates; not claimed identical to full-batch64/SyncBN/DDP.',
              'precision': 'FP32 no TF32, deterministic CUDA algorithms', 'concurrent_gpu_load': True,
              'gpu_before': gpu_state(), 'torch': torch.__version__,
              'inputs': {name: sha(Path(name)) for name in ['scripts/profile_external_batch64.py', 'scripts/profile_flatlands_external.py',
                         'src/pathrel/external_training.py', 'src/pathrel/lama.py', 'src/pathrel/flow_matching.py',
                         'src/pathrel/external_completion.py', 'src/pathrel/cogniplan.py',
                         'results/external_protocol_v1/flatlands_profile32.csv']}}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    trace = []
    try:
        rows, c, t, hashes = load_fixed_training()
        model = (LaMaBEV() if args.method == 'lama' else FlowUNet()).cuda()
        discriminator = make_discriminator().cuda() if args.method == 'lama' else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        optim_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay=.01) if discriminator is not None else None
        noise_rng = torch.Generator(device='cuda').manual_seed(20260909)
        sample_rng = torch.Generator().manual_seed(20260909)
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(1, 106):
            start = sync()
            ids = torch.cat([torch.randperm(32, generator=sample_rng) for _ in range(2)])
            batch_c, batch_t = c[ids].cuda(), t[ids].cuda()
            if args.method == 'lama':
                losses = lama_accumulated_update(model, discriminator, optimizer, optim_d, batch_c, batch_t, microbatch)
            else:
                losses = flow_accumulated_update(model, optimizer, batch_c, batch_t, microbatch, noise_rng)
            seconds = sync() - start
            assert all(np.isfinite(v) for v in losses.values())
            trace.append({'iteration': iteration, 'warmup': iteration <= 5, 'seconds': seconds, 'losses': losses,
                          'sample_indices': ids.tolist()})
            (out / 'progress.json').write_text(json.dumps({'method': args.method, 'iteration': iteration, 'target': 105,
                    'seconds': seconds, 'losses': losses, 'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
            if iteration % 5 == 0:
                print(json.dumps({'method': args.method, 'iteration': iteration, 'seconds': seconds, 'losses': losses}), flush=True)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                    'discriminator': discriminator.state_dict() if discriminator is not None else None,
                    'optimizer_d': optim_d.state_dict() if optim_d is not None else None,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'flow_rng': noise_rng.get_state(), 'sample_rng': sample_rng.get_state(), 'iterations': 105}, out / 'profile_checkpoint.pt')
        (out / 'training_trace.json').write_text(json.dumps(trace, indent=2) + '\n')
        measured = [r['seconds'] for r in trace if not r['warmup']]
        report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'method': args.method,
                  'test_assets_opened': False, 'formal_training': False, 'convergence_established': False,
                  'effective_batch': 64, 'microbatch': microbatch, 'accumulation': 64 // microbatch,
                  'optimizer_updates_measured': len(measured), 'microbatches_per_optimizer_update': 64 // microbatch,
                  'seconds_mean': float(np.mean(measured)), 'seconds_p50': float(np.median(measured)),
                  'seconds_p95': float(np.quantile(measured, .95)),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'paper_300000_update_training_only_hours_extrapolation': float(np.mean(measured)) * 300000 / 3600,
                  'budget_scope': 'Measured actual accumulated effective batch64; fixed cached32 duplicated for workload only. Full data I/O, validation, checkpoints and convergence not included; not a committed formal recipe or ETA.',
                  'generator_parameters': sum(p.numel() for p in model.parameters()),
                  'discriminator_parameters': sum(p.numel() for p in discriminator.parameters()) if discriminator is not None else 0,
                  'all_finite': True, 'gpu_after': gpu_state(), 'inputs': config['inputs'],
                  'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir() if p.name not in ['progress.json', 'report.json']}}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'progress.json').write_text(json.dumps({'status': 'complete'}) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps({'error': repr(error), 'completed_iterations': len(trace), 'test_assets_opened': False}) + '\n')
        raise


if __name__ == '__main__': main()
