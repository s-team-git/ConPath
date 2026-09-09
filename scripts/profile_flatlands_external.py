#!/usr/bin/env python3
"""Fixed-32 train-only LaMa/FM+XAttn engineering profile; no formal comparisons."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import distance_transform_edt, label
import torch

from pathrel.cogniplan import configure_reproducible_cuda
from pathrel.external_completion import validate_condition
from pathrel.flatlands_query import decode_binary_grayscale_png
from pathrel.flow_matching import FlowUNet, flow_training_loss, sample_heun
from pathrel.lama import LaMaBEV, lama_update, make_discriminator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_state():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader'], text=True).strip()


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def load_fixed_training():
    protocol = Path('results/external_protocol_v1')
    report = json.loads((protocol / 'report.json').read_text())
    path = protocol / 'flatlands_profile32.csv'
    assert sha(path) == report['outputs'][path.name]
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 32 and all(r['provenance_split'] == 'train' for r in rows)
    conditions, targets, source_hashes = [], [], {}
    with ZipFile('data/raw/flatlands/FlatLands_final_dataset.zip') as z:
        for row in rows:
            meta = json.loads(z.read(row['metadata_member']))
            assert meta['provenance']['original_split'] == 'train'
            assert meta['provenance']['global_id'] == row['global_id']
            channels = {}
            for name in ['observed_floor', 'unobserved', 'epistemic_mask', 'floor_map']:
                member = row['packet_directory'] + '/' + name + '.png'
                raw = z.read(member)
                source_hashes[member] = hashlib.sha256(raw).hexdigest()
                channels[name] = decode_binary_grayscale_png(raw)
            support = channels['epistemic_mask']
            free = channels['observed_floor'] & support
            unknown = channels['unobserved'] & support
            target = channels['floor_map'] & support
            assert np.all(target[~unknown] == free[~unknown])
            conditions.append(np.stack((free, unknown, support)).astype(np.float32))
            targets.append(target[None].astype(np.float32))
    c, t = torch.from_numpy(np.stack(conditions)), torch.from_numpy(np.stack(targets))
    validate_condition(c)
    assert c.shape == (32, 3, 256, 256) and (c[:, 1].sum((-1, -2)) > 0).all()
    return rows, c, t, source_hashes


def connectivity_workload(worlds, endpoints):
    values = []
    for world in worlds:
        distance = distance_transform_edt(np.pad(world, 1))[1:-1, 1:-1]
        for radius in (0, 10, 20):
            components, _ = label(distance > radius)
            a, b = endpoints[:, :2], endpoints[:, 2:]
            va, vb = components[a[:, 0], a[:, 1]], components[b[:, 0], b[:, 1]]
            values.append((va != 0) & (va == vb))
    return np.stack(values).reshape(len(worlds), 3, len(endpoints)).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['lama', 'flow'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--inference-observations', type=int, default=32)
    args = p.parse_args()
    if args.batch_size < 1 or args.batch_size > 32 or not 1 <= args.inference_observations <= 32:
        raise ValueError('Invalid engineering sample budget')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    configure_reproducible_cuda()
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    np.random.seed(20260909)
    config = {'method': args.method, 'seed': 20260909, 'purpose': 'fixed32 train-only engineering',
              'formal_training': False, 'pretraining': None, 'samples': 32, 'batch_size': args.batch_size,
              'gradient_accumulation': 1, 'measured_optimizer_updates': args.updates, 'warmup_optimizer_updates': args.warmup,
              'inference_observations': args.inference_observations,
              'optimizer': 'AdamW', 'learning_rate': 1e-4, 'weight_decay': .01, 'betas': [.9, .999],
              'lr_schedule': 'constant for short resource profile; full candidate recipe needs cosine schedule',
              'conditioning': ['observed_free', 'unknown', 'valid_support'], 'augmentation': 'none camera-anchored maps',
              'precision': 'FP32 without TF32', 'deterministic_algorithms': True,
              'lama': {'ngf': 64, 'residual_blocks': 18, 'global_ratio': .75, 'lfu': False, 'discriminator_layers': 4,
                       'generator_loss_weights': {'masked_l1': 10, 'hinge': 10, 'feature_mse': 250},
                       'loss_masking': 'per-observation normalized hidden-valid weights; area pooling at feature/patch scales',
                       'output_worlds': 1, 'ensemble_members_trained': 0},
              'flow': {'base_channels': 64, 'channel_multipliers': [1, 2, 4, 4], 'residual_blocks_per_stage': 2,
                       'cross_attention_resolutions': [64, 32], 'cross_attention_stages': 'down and up',
                       'attention_heads': 4, 'condition_position_encoding': 'fixed 2D sinusoidal',
                       'attention_backend': 'SDPA math, full spatial tokens', 'activation_checkpointing': True,
                       'loss': 'masked velocity MSE on OT interpolant', 'cfg_dropout': .1, 'cfg_scale_s': 2,
                       'solver': 'Heun', 'steps': 25, 'output_worlds': 4,
                       'condition_cache': False, 'input_adaptation': 'shared support added to paper Fobs/U'},
              'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu_before': gpu_state(),
                              'concurrent_gpu_workload': True, 'isolated_paper_latency': False},
              'source_hashes': {name: sha(Path(name)) for name in
                               ['src/pathrel/external_completion.py', 'src/pathrel/lama.py', 'src/pathrel/flow_matching.py',
                                'src/pathrel/cogniplan.py', 'scripts/profile_flatlands_external.py',
                                'third_party/lama/sources.json', 'results/external_protocol_v1/flatlands_profile32.csv']}}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    try:
        rows, condition, target, sources = load_fixed_training()
        np.savez_compressed(out / 'fixed_train_inputs.npz', condition=condition.numpy(), target=target.numpy())
        (out / 'source_packets.json').write_text(json.dumps(sources, indent=2) + '\n')
        with (out / 'fixed_train_observations.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        model = (LaMaBEV() if args.method == 'lama' else FlowUNet()).cuda()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        discriminator = make_discriminator().cuda() if args.method == 'lama' else None
        optim_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay=.01) if discriminator is not None else None
        rng = torch.Generator(device='cuda').manual_seed(20260909)
        sample_rng = torch.Generator().manual_seed(20260909)
        torch.cuda.reset_peak_memory_stats()
        trace = []
        for iteration in range(1, args.warmup + args.updates + 1):
            start = sync()
            ids = torch.randperm(32, generator=sample_rng)[:args.batch_size]
            c, t = condition[ids].cuda(), target[ids].cuda()
            if args.method == 'lama':
                losses = lama_update(model, discriminator, optim, optim_d, c, t)
            else:
                model.train(); optim.zero_grad(set_to_none=True)
                loss = flow_training_loss(model, c, t, rng=rng)
                loss.backward(); optim.step()
                losses = {'velocity_mse': float(loss.detach())}
            duration = sync() - start
            assert all(np.isfinite(v) for v in losses.values()), losses
            trace.append({'iteration': iteration, 'warmup': iteration <= args.warmup,
                          'seconds': duration, 'losses': losses, 'sample_indices': ids.tolist()})
            (out / 'progress.json').write_text(json.dumps({'stage': 'training_profile', 'method': args.method,
                    'iteration': iteration, 'total_iterations': args.warmup + args.updates, 'seconds': duration,
                    'losses': losses, 'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
            if iteration % 10 == 0:
                print(json.dumps({'stage': 'train', 'method': args.method, 'iteration': iteration,
                                  'seconds': duration, 'losses': losses}), flush=True)
        train_peak = torch.cuda.max_memory_allocated()
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        checkpoint_payload = {'model': model.state_dict(), 'config': config, 'optimizer': optim.state_dict(),
                              'iterations': args.updates + args.warmup, 'torch_rng': torch.get_rng_state(),
                              'cuda_rng': torch.cuda.get_rng_state_all(), 'flow_rng': rng.get_state(), 'sample_rng': sample_rng.get_state()}
        if discriminator is not None:
            checkpoint_payload.update(discriminator=discriminator.state_dict(), optimizer_d=optim_d.state_dict())
        torch.save(checkpoint_payload, out / 'profile_checkpoint.pt')
        del checkpoint_payload, discriminator, optim_d, optim
        torch.cuda.empty_cache()
        (out / 'training_trace.json').write_text(json.dumps(trace, indent=2) + '\n')
        model.eval()
        inference_rows, predicted = [], {}
        endpoints = np.random.default_rng(20260909).integers(0, 256, (64, 4))
        np.save(out / 'engineering_endpoints.npy', endpoints)
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.inference_observations):
            c = condition[i:i+1].cuda()
            calls = []
            hook = model.register_forward_pre_hook(lambda module, inputs: calls.append(int(inputs[0].shape[0])))
            start = sync()
            with torch.no_grad():
                if args.method == 'lama':
                    values = model(c)
                    worlds = values > .5
                    cost = {'samples_per_observation': 1, 'batched_model_calls': 1, 'sample_equivalent_forwards': 1}
                else:
                    values, worlds, cost = sample_heun(model, c, seed=20260909 + i)
            generation = sync() - start
            hook.remove()
            assert cost['batched_model_calls'] == len(calls) and cost['sample_equivalent_forwards'] == sum(calls)
            start = sync()
            values_np, worlds_np = values[0].cpu().numpy(), worlds[0].cpu().numpy()
            transfer = sync() - start
            assert np.isfinite(values_np).all()
            known = condition[i, 1].numpy() == 0
            expected = condition[i, 0].numpy().astype(bool)
            assert np.all(worlds_np[:, known] == expected[known])
            assert np.all(values_np[:, known] == expected[known])
            start = time.perf_counter()
            connectivity_workload(worlds_np, endpoints)
            connectivity = time.perf_counter() - start
            hidden = ~known
            vote = worlds_np.mean(0)
            score = float(np.mean((vote[hidden] - target[i, 0].numpy()[hidden]) ** 2))
            inference_rows.append({'index': i, 'global_id': rows[i]['global_id'], 'source': rows[i]['source_dataset'],
                                   'generation_seconds': generation, 'transfer_seconds': transfer,
                                   'connectivity_seconds': connectivity, 'total_seconds': generation + transfer + connectivity,
                                   'unknown_vote_brier_train_diagnostic': score, 'known_support_conflicts': 0,
                                   'distinct_worlds': len({a.tobytes() for a in worlds_np}),
                                   'cost': cost, 'hook_batch_sizes': calls})
            predicted[f'values_{i}'] = values_np
            predicted[f'worlds_{i}'] = worlds_np
            (out / 'progress.json').write_text(json.dumps({'stage': 'inference_profile', 'method': args.method,
                'observation': i + 1, 'total_observations': args.inference_observations}, indent=2) + '\n')
            if i % 4 == 0:
                print(json.dumps({'stage': 'inference', 'method': args.method, 'observation': i + 1,
                                  'generation_seconds': generation, 'actual_cost': cost}), flush=True)
        np.savez_compressed(out / 'predictions.npz', **predicted)
        (out / 'inference_rows.json').write_text(json.dumps(inference_rows, indent=2) + '\n')
        times = [t['seconds'] for t in trace if not t['warmup']]
        report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'method': args.method,
                  'engineering_only': True, 'formal_training_started': False, 'test_assets_opened': False,
                  'convergence_established': False, 'native_pretrained_quality_gate': 'Not run: no pretrained weights; full backbone fresh training engineering only.',
                  'generator_parameters': sum(v.numel() for v in model.parameters()), 'training_updates_measured': len(times),
                  'training_batch_size': args.batch_size, 'training_seconds_mean': float(np.mean(times)),
                  'training_seconds_p50': float(np.median(times)), 'training_seconds_p95': float(np.quantile(times, .95)),
                  'training_peak_allocated_bytes': train_peak, 'inference_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'inference_observations': len(inference_rows), 'inference_actual_cost_per_observation': inference_rows[0]['cost'],
                  'inference_mean_seconds_excluding_first': {k: float(np.mean([r[k] for r in inference_rows[1:] or inference_rows]))
                      for k in ['generation_seconds', 'transfer_seconds', 'connectivity_seconds', 'total_seconds']},
                  'timing_scope': 'Train includes CPU sample indexing and H2D. Generation includes solver/clamps/threshold, excludes encoding/H2D; D2H and connectivity separate. First inference excluded from means. Concurrent GPU workload, not isolated paper latency.',
                  'budget_limit': 'Batch size is an engineering microbatch, not paper effective batch64. No linear microbatch-to-batch64 ETA asserted; accumulation would change LaMa BN behavior. Full-scale/recipe quality gates remain.',
                  'gpu_after': gpu_state(), 'all_finite': True, 'known_support_conflicts': 0,
                  'source_hashes': config['source_hashes'],
                  'outputs': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in out.iterdir()
                              if p.is_file() and p.name not in {'report.json', 'progress.json'}}}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'progress.json').write_text(json.dumps({'status': 'complete', 'report': str(out / 'report.json')}) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': repr(error),
            'created_utc': datetime.now(timezone.utc).isoformat(), 'test_assets_opened': False}, indent=2) + '\n')
        raise


if __name__ == '__main__': main()
