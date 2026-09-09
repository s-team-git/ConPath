#!/usr/bin/env python3
"""Matched, resumable development training. This process cannot load validation."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.flatlands_baselines import MarginalCompletionBaseline
from pathrel.losses import reachability_brier_u_statistic, spatial_variogram_score
from pathrel.model import PathRelNet
from pathrel.parent_pilot_data import collate_pilot, load_pilot, sha
from scripts.evaluate_flatlands_support_clamped import _accelerated_events
from scripts.train_flatlands_conpath import _atomic_json, _atomic_torch_save, _seed_everything, _to_tensor_batch, _forward

DATA = ROOT / 'results/parent_group_pilot_v1/data'
RADII = (0, 10, 20)
STOP = False


def stop(*_):
    global STOP
    STOP = True


def model_for(method, device='cuda'):
    if method == 'deterministic':
        return MarginalCompletionBaseline(feature_channels=16).to(device)
    return PathRelNet(feature_channels=16, latent_dim=4, local_kernel_size=1 if method == 'independent' else 5).to(device)


def scene_nll(probability, target, mask):
    loss = F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), target.float(), reduction='none')
    return ((loss * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()


def tensor_batch(samples, device):
    batch = collate_pilot(samples)
    # Padding has zero query weight but must still satisfy shared-start geometry.
    batch['starts'][:] = batch['starts'][:, :1]
    return batch, _to_tensor_batch(batch, device)


def train_loss(model, method, samples, generator, config):
    numpy_batch, batch = tensor_batch(samples, torch.device('cuda'))
    target, mask = batch['target_free'], batch['loss_mask']
    if method == 'deterministic':
        logits = model(batch['observation'])
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
        loss = ((loss * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()
        return loss, {'map_nll': float(loss.detach()), 'hard_forward_corrections': 0}
    if batch['starts'].shape[1]:
        output = _forward(model, batch, RADII, config['train_samples'], config['max_reachability_steps'], generator,
                          disable_global_factors=method == 'independent')
        worlds = output.posterior.safe_samples().detach().cpu().numpy() > .5
        exact = np.stack([_accelerated_events(w, s, g, RADII) for w, s, g in zip(worlds, numpy_batch['starts'], numpy_batch['goals'])])
        hard = torch.from_numpy(exact).to(device='cuda', dtype=torch.float32)
        drift = int(((output.sample_reachability.detach() != hard) & batch['query_mask'][:, None, :, None]).sum().item())
        # The 256-step branch is a gradient surrogate only. Report and optimize
        # the actual unbounded hard event, including long-detour paths.
        events = output.sample_reachability + (hard - output.sample_reachability.detach())
        weights = batch['query_mask'].float() / batch['query_mask'].sum(1, keepdim=True).clamp_min(1)
        weights = weights[..., None].expand_as(batch['reachability_targets'])
        event_loss = reachability_brier_u_statistic(events, batch['reachability_targets'], weights=weights)
    else:
        output = model(batch['observation'], valid_support_mask=batch['valid_support_mask'], num_samples=config['train_samples'],
                       disable_global_factors=method == 'independent', generator=generator)
        event_loss, drift = output.posterior.sample_logits.sum() * 0, 0
    probs = output.posterior.sample_logits.softmax(2)[:, :, 0].mean(1)
    map_loss = scene_nll(probs, target, mask)
    variogram = torch.stack([spatial_variogram_score(output.posterior.safe_samples()[i:i+1], target[i:i+1].float(),
                                                    valid_mask=mask[i:i+1]) for i in range(len(samples))]).mean()
    loss = map_loss + config['variogram_weight'] * variogram + config['event_weight'] * event_loss
    return loss, {'map_nll': float(map_loss.detach()), 'event_loss': float(event_loss.detach()),
                  'variogram': float(variogram.detach()), 'hard_forward_corrections': drift}


@torch.inference_mode()
def predict(model, method, sample, seed, k):
    x = torch.from_numpy(sample.observation[None]).cuda()
    valid = torch.from_numpy(sample.valid[None]).cuda()
    if method == 'deterministic':
        raw_p = model(x).sigmoid()[0].cpu().numpy()
        p = np.where(sample.hidden, raw_p, sample.observation[0]) * sample.valid
        worlds = (p >= .5)[None]
    else:
        generator = torch.Generator(device='cuda').manual_seed(seed + int(sample.row['global_id'].split('_')[1]))
        w, probs = [], []
        for n in range(0, k, 8):
            posterior = model(x, valid_support_mask=valid, num_samples=min(8, k-n), disable_global_factors=method == 'independent', generator=generator).posterior
            w.append(posterior.safe_samples()[0].cpu().numpy() > .5)
            probs.append(posterior.sample_logits.softmax(2)[0, :, 0].cpu().numpy())
        worlds, raw_p = np.concatenate(w), np.concatenate(probs).mean(0)
        p = np.where(sample.hidden, raw_p, sample.observation[0]) * sample.valid
    assert not np.any(worlds & ~sample.valid[None])
    assert not np.any(worlds[:, ~sample.hidden] != sample.observation[0, ~sample.hidden])
    events = _accelerated_events(worlds, sample.starts, sample.goals, RADII)
    y, pv = sample.target[sample.hidden], np.clip(raw_p[sample.hidden], 1e-6, 1-1e-6)
    nll = float(np.mean(np.where(y, -np.log(pv), -np.log1p(-pv))))
    return worlds, events, p, nll


@torch.inference_mode()
def calibration(model, method, samples, seed, k):
    model.eval()
    scores, nlls = [], []
    for sample in samples:
        _, events, _, nll = predict(model, method, sample, seed + 2000000, k)
        if len(sample.starts):
            scores.append(float(np.mean((events.mean(0) - sample.targets)**2)))
        nlls.append(nll)
    return {'event_brier': float(np.mean(scores)), 'map_nll': float(np.mean(nlls))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['correlated', 'independent', 'deterministic'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile-steps', type=int, default=0, help='Training-only disposable timing; never loads validation')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader_rng, sample_rng = _seed_everything(args.seed, torch.device('cuda'))
    protocol = json.loads((DATA / 'protocol.json').read_text()); config = protocol['training']
    if args.seed not in protocol['seeds']:
        raise ValueError('Seed is not in frozen pilot protocol')
    train, cal = load_pilot(DATA, 'train'), load_pilot(DATA, 'calibration')
    limit = protocol['query']['training_queries_per_observation']
    train = [replace(s, starts=s.starts[:limit], goals=s.goals[:limit], targets=s.targets[:limit], candidate_indices=s.candidate_indices[:limit]) for s in train]
    folder = ROOT / 'results/parent_group_pilot_v1' / ('profile_micro2' if args.profile_steps else 'runs') / args.method / str(args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    recipe = {'method': args.method, 'seed': args.seed, 'protocol_sha256': sha(DATA / 'protocol.json'), 'data_seal_sha256': sha(DATA / 'seal.json'),
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/pathrel/parent_pilot_data.py', ROOT / 'src/pathrel/model.py',
                             ROOT / 'src/pathrel/stochastic_decoder.py', ROOT / 'src/pathrel/reachability.py', ROOT / 'src/pathrel/losses.py',
                             ROOT / 'src/pathrel/flatlands_baselines.py', ROOT / 'scripts/evaluate_flatlands_support_clamped.py', ROOT / 'scripts/train_flatlands_conpath.py']},
              'hard_event': 'exact disk/four-neighbor oracle; any 256-step hard-forward errors replaced, surrogate gradient remains bounded',
              'loss_weighting': 'equal parent for map NLL, variogram, event; no query observations still train with map/variogram',
              'validation_packets_loaded_by_trainer': False, 'torch': torch.__version__, 'effective_batch_size': config['batch_size'], 'micro_batch_size': 2, 'gradient_accumulation': 'size-weighted to one update per four observations'}
    model = model_for(args.method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    history, best, best_epoch, patience = [], (float('inf'), float('inf')), 0, 0
    if args.resume:
        if json.loads((folder / 'recipe.json').read_text()) != recipe:
            raise ValueError('Resume recipe/source/data mismatch')
        state = torch.load(folder / 'latest.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        history, best, best_epoch, patience = state['history'], tuple(state['best']), state['best_epoch'], state['patience']
        loader_rng.set_state(state['loader_rng'].cpu()); sample_rng.set_state(state['sample_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    else:
        if (folder / 'recipe.json').exists():
            raise ValueError('Refusing to overwrite run; use --resume')
        _atomic_json(folder / 'recipe.json', recipe)
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=loader_rng, collate_fn=list, num_workers=0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    for epoch in range(len(history)+1, config['max_epochs']+1):
        if history and epoch > config['minimum_epochs'] and patience >= config['patience']:
            break
        model.train(); epoch_start = time.monotonic(); records = []
        for step, samples in enumerate(loader, 1):
            if STOP or (folder.parents[2] / 'STOP').exists():
                _atomic_json(folder / 'paused.json', {'completed_epochs': len(history), 'resume_from_epoch_boundary': True})
                raise SystemExit(130)
            optimizer.zero_grad(set_to_none=True)
            micro_records = []
            for begin in range(0, len(samples), 2):
                micro = samples[begin:begin+2]
                loss, record = train_loss(model, args.method, micro, sample_rng, config)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss * (len(micro) / len(samples))).backward()
                micro_records.append(dict(record, total=float(loss.detach())))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite model gradients')
            optimizer.step()
            record = {key: float(np.mean([r[key] for r in micro_records])) for key in micro_records[0]}
            records.append(dict(record, grad_norm=float(norm)))
            if args.profile_steps and step >= args.profile_steps:
                torch.cuda.synchronize()
                _atomic_json(folder / 'profile.json', {'steps': step, 'elapsed_seconds': time.monotonic()-started,
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'records': records})
                print(json.dumps({'profile_steps': step, 'seconds': time.monotonic()-started, 'peak_gib': torch.cuda.max_memory_allocated()/2**30}), flush=True)
                return
        score = calibration(model, args.method, cal, args.seed, config['calibration_K'])
        current = score['event_brier'], score['map_nll']
        improved = current[0] < best[0]-1e-5 or (abs(current[0]-best[0]) <= 1e-5 and current[1] < best[1]-1e-5)
        if improved:
            best, best_epoch, patience = current, epoch, 0
        else:
            patience += 1
        record = {'epoch': epoch, 'method': args.method, 'seed': args.seed, 'train': {k: float(np.mean([r[k] for r in records])) for k in records[0]},
                  'calibration': score, 'best_epoch': best_epoch, 'best_brier': best[0], 'patience': patience, 'improved': improved,
                  'epoch_seconds': time.monotonic()-epoch_start, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        history.append(record)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history, 'best': best, 'best_epoch': best_epoch,
                 'patience': patience, 'loader_rng': loader_rng.get_state(), 'sample_rng': sample_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all(), 'recipe': recipe}
        _atomic_torch_save(folder / 'latest.pt', state)
        if improved:
            _atomic_torch_save(folder / 'best.pt', state)
        _atomic_json(folder / 'history.json', history)
        print(json.dumps(record), flush=True)
    _atomic_json(folder / 'complete.json', {'method': args.method, 'seed': args.seed, 'epochs': len(history), 'best_epoch': best_epoch,
                 'best_calibration': best, 'checkpoint_sha256': sha(folder / 'best.pt'), 'recipe_sha256': sha(folder / 'recipe.json'),
                 'wall_seconds_this_process': time.monotonic()-started, 'validation_evaluated': False})


if __name__ == '__main__':
    main()
