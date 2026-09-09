#!/usr/bin/env python3
"""Bounded two-seed follow-up; frozen baseline losses/data and one sampler change."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.coherent_categorical import CoherentCategoricalDecoder
from pathrel.model import PathRelNet
from pathrel.parent_pilot_data import load_pilot, sha
from scripts.train_parent_group_pilot import train_loss, calibration
from scripts.train_flatlands_conpath import _atomic_json, _atomic_torch_save, _seed_everything

BASE = ROOT / 'results/parent_group_pilot_v1'
OUT = ROOT / 'results/coherent_parent_pilot_v1'
DATA = BASE / 'data'
STOP = False


def stop(*_):
    global STOP
    STOP = True


def model_for(device='cuda'):
    """Keep the baseline initialization and subsequent global RNG state identical."""
    model = PathRelNet(feature_channels=16, latent_dim=4, local_kernel_size=5)
    saved_rng = torch.get_rng_state()
    decoder = CoherentCategoricalDecoder(16, latent_dim=4, local_kernel_size=5, gumbel_kernel_size=9)
    incompatible = decoder.load_state_dict(model.decoder.state_dict(), strict=False)
    assert incompatible.missing_keys == ['gumbel_kernel'] and not incompatible.unexpected_keys
    model.decoder = decoder
    torch.set_rng_state(saved_rng)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=[20260910, 20260911])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['seeds'] == [20260910, 20260911]
    assert protocol['baseline_analysis_sha256'] == sha(BASE / 'analysis.json')
    assert protocol['baseline_verification_sha256'] == sha(BASE / 'verification.json')
    assert protocol['data_seal_sha256'] == sha(DATA / 'seal.json')
    for name, checksum in protocol['source_hashes'].items():
        assert sha(ROOT / name) == checksum, name
    verification = json.loads((BASE / 'verification.json').read_text())
    assert verification['passed'] is True and verification['analysis_sha256'] == sha(BASE / 'analysis.json')
    config = json.loads((DATA / 'protocol.json').read_text())['training']
    assert protocol['training'] == config
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    loader_rng, sample_rng = _seed_everything(args.seed, torch.device('cuda'))
    train, cal = load_pilot(DATA, 'train'), load_pilot(DATA, 'calibration')
    limit = json.loads((DATA / 'protocol.json').read_text())['query']['training_queries_per_observation']
    train = [replace(s, starts=s.starts[:limit], goals=s.goals[:limit], targets=s.targets[:limit], candidate_indices=s.candidate_indices[:limit]) for s in train]
    folder = OUT / 'runs' / str(args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    recipe = {'method': 'coherent_categorical', 'seed': args.seed, 'protocol_sha256': sha(OUT / 'protocol.json'),
              'data_seal_sha256': sha(DATA / 'seal.json'), 'source_sha256': protocol['source_hashes'],
              'torch': torch.__version__, 'allocator_limit_gib': 28, 'validation_packets_loaded_by_trainer': False,
              'effective_batch_size': 4, 'micro_batch_size': 2,
              'zero_query_microbatch_limitation_shared_with_baseline': True}
    model = model_for()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    history, best, best_epoch, patience, best_model = [], (float('inf'), float('inf')), 0, 0, None
    if args.resume:
        assert json.loads((folder / 'recipe.json').read_text()) == recipe
        state = torch.load(folder / 'latest.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        history, best, best_epoch, patience = state['history'], tuple(state['best']), state['best_epoch'], state['patience']
        best_model = {k: v.cpu().clone() for k, v in state['best_model'].items()}
        loader_rng.set_state(state['loader_rng'].cpu()); sample_rng.set_state(state['sample_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        _atomic_torch_save(folder / 'best.pt', {'model': best_model, 'best': best, 'best_epoch': best_epoch, 'recipe': recipe})
        del state
    else:
        if (folder / 'recipe.json').exists():
            raise ValueError('Refusing to overwrite candidate; use --resume')
        _atomic_json(folder / 'recipe.json', recipe)
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=loader_rng, collate_fn=list, num_workers=0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    for epoch in range(len(history)+1, config['max_epochs']+1):
        if history and epoch > config['minimum_epochs'] and patience >= config['patience']:
            break
        model.train(); epoch_start = time.monotonic(); records = []
        for samples in loader:
            if STOP or (OUT / 'STOP').exists():
                _atomic_json(folder / 'paused.json', {'completed_epochs': len(history), 'resume_from_epoch_boundary': True})
                raise SystemExit(130)
            optimizer.zero_grad(set_to_none=True)
            micro_records = []
            for begin in range(0, len(samples), 2):
                micro = samples[begin:begin+2]
                loss, record = train_loss(model, 'correlated', micro, sample_rng, config)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite candidate loss')
                (loss * (len(micro) / len(samples))).backward()
                micro_records.append(dict(record, total=float(loss.detach())))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite candidate gradient')
            optimizer.step()
            records.append({**{key: float(np.mean([r[key] for r in micro_records])) for key in micro_records[0]}, 'grad_norm': float(norm)})
        score = calibration(model, 'correlated', cal, args.seed, config['calibration_K'])
        current = score['event_brier'], score['map_nll']
        improved = current[0] < best[0]-1e-5 or (abs(current[0]-best[0]) <= 1e-5 and current[1] < best[1]-1e-5)
        if improved:
            best, best_epoch, patience = current, epoch, 0
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        record = {'epoch': epoch, 'method': 'coherent_categorical', 'seed': args.seed,
                  'train': {k: float(np.mean([r[k] for r in records])) for k in records[0]},
                  'calibration': score, 'best_epoch': best_epoch, 'best_brier': best[0], 'patience': patience,
                  'improved': improved, 'epoch_seconds': time.monotonic()-epoch_start,
                  'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        history.append(record)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history,
                 'best': best, 'best_epoch': best_epoch, 'best_model': best_model, 'patience': patience,
                 'loader_rng': loader_rng.get_state(), 'sample_rng': sample_rng.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'recipe': recipe}
        _atomic_torch_save(folder / 'latest.pt', state)
        if improved:
            _atomic_torch_save(folder / 'best.pt', {'model': best_model, 'best': best, 'best_epoch': best_epoch, 'recipe': recipe})
        _atomic_json(folder / 'history.json', history)
        print(json.dumps(record), flush=True)
    _atomic_json(folder / 'complete.json', {'method': 'coherent_categorical', 'seed': args.seed, 'epochs': len(history),
                 'best_epoch': best_epoch, 'best_calibration': best, 'checkpoint_sha256': sha(folder / 'best.pt'),
                 'recipe_sha256': sha(folder / 'recipe.json'), 'wall_seconds_this_process': time.monotonic()-started,
                 'validation_evaluated': False})


if __name__ == '__main__':
    main()
