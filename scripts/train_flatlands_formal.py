#!/usr/bin/env python3
"""Train a sealed formal recipe to a permitted boundary; never loads validation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
from pathrel.cogniplan import configure_reproducible_cuda
from pathrel.formal_checkpoint import (FormalCheckpointStore, StatefulBatchSampler,
    atomic_json, capture_training_state, restore_training_state)
from pathrel.formal_training import StagedLRScheduler, training_update

STOP = False


def stop(*unused):
    global STOP
    STOP = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def project_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Frozen path escapes project: {name}')
    return path


def verify_sources(protocol):
    required = {'scripts/train_flatlands_formal.py', 'src/pathrel/formal_training.py',
                'src/pathrel/formal_checkpoint.py', 'src/pathrel/formal_data.py',
                'src/pathrel/formal_inference.py'}
    if not required.issubset(protocol['source_hashes']):
        raise ValueError('Protocol omits a required executable source hash')
    for name, digest in protocol['source_hashes'].items():
        if sha(project_path(name)) != digest:
            raise ValueError(f'Frozen implementation changed: {name}')


def snapshot(folder, step):
    path = folder / f'step_{step:08d}.pt'
    if path.exists():
        if sha(path) != sha(folder / 'latest.pt'):
            raise FileExistsError(f'Cannot replace existing checkpoint {path}')
    else:
        # latest is atomically replaced, never modified; links are immutable.
        os.link(folder / 'latest.pt', path)
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return path


def session_totals(folder):
    """Count all actual process time, including work replayed after rollback.

    A hard-killed process only has its last durable heartbeat: report a lower
    bound explicitly instead of silently claiming complete measured wall time.
    """
    total, incomplete = 0., []
    for start in sorted((folder / 'sessions').glob('*.started.json')):
        token = start.name.removesuffix('.started.json')
        end = start.with_name(token + '.finished.json')
        heartbeat = start.with_name(token + '.heartbeat.json')
        if end.exists():
            row = json.loads(end.read_text())
        else:
            row = json.loads(heartbeat.read_text()) if heartbeat.exists() else {'elapsed_seconds': 0.}
            incomplete.append(token)
        total += float(row['elapsed_seconds'])
    return {'training_wall_seconds': total, 'wall_time_is_lower_bound': bool(incomplete),
            'sessions_without_final_receipt': incomplete}


class ProcessSession:
    """Unique start/end receipts and a durable progress heartbeat per process."""
    def __init__(self, folder, metadata):
        self.folder = folder / 'sessions'
        self.folder.mkdir(exist_ok=True)
        self.token = uuid.uuid4().hex
        self.started = time.monotonic()
        self.metadata = dict(metadata, session_id=self.token, pid=os.getpid(),
                             started_utc=datetime.now(timezone.utc).isoformat())
        path = self.folder / (self.token + '.started.json')
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, self.metadata)

    def elapsed(self):
        return time.monotonic() - self.started

    def heartbeat(self, **values):
        atomic_json(self.folder / (self.token + '.heartbeat.json'),
                    dict(self.metadata, elapsed_seconds=self.elapsed(), **values))

    def finish(self, **values):
        path = self.folder / (self.token + '.finished.json')
        if path.exists():
            raise FileExistsError('Process finish receipt cannot be overwritten')
        atomic_json(path, dict(self.metadata, elapsed_seconds=self.elapsed(),
                              finished_utc=datetime.now(timezone.utc).isoformat(), **values))


def gpu_resources():
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}


def verify_completed_boundary(folder, step, protocol_sha):
    """Idempotent resume at an already committed boundary needs no model/RNG I/O."""
    receipt_path = folder / f'boundary_{step:08d}.json'
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    checkpoint = folder / f'step_{step:08d}.pt'
    if (receipt['completed_steps'] != step or receipt['protocol_sha256'] != protocol_sha
            or sha(checkpoint) != receipt['checkpoint_sha256']
            or sha(folder / 'latest.pt') != receipt['checkpoint_sha256']):
        raise ValueError('Existing completed boundary receipt/checkpoint mismatch')
    return True


def verify_full_plan(plan, protocol, protocol_sha, method):
    """Bind full training to immutable numerical AND fixed-case visual evidence."""
    if (plan.get('protocol_sha256') != protocol_sha or plan.get('method') != method
            or plan.get('pilot_gate_passed') is not True):
        raise ValueError('Full plan lacks the matching successful method/protocol gate')
    end = plan.get('full_steps')
    staged = protocol['staged']
    if (type(end) is not int or end not in staged['allowed_full_steps']
            or end <= staged['pilot_steps']):
        raise ValueError('Full budget is outside the frozen allowed step plans')
    checkpoints = sorted({staged['smoke_steps'], staged['pilot_steps'], end,
                          *range(staged['pilot_steps'] + 2500, end + 1, 2500)})
    if plan.get('checkpoint_steps') != checkpoints:
        raise ValueError('Full checkpoint boundaries differ from the frozen stage/2500-step rule')

    def bound_json(owner, path_key, hash_key):
        name, expected = owner.get(path_key), owner.get(hash_key)
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError(f'Full gate requires a project-relative {path_key} and SHA256')
        path = project_path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Full gate receipt/hash mismatch: {path_key}')
        return json.loads(path.read_text())

    numeric = bound_json(plan, 'pilot_gate_receipt', 'pilot_gate_sha256')
    visual = bound_json(plan, 'visual_review_receipt', 'visual_review_sha256')
    for kind, receipt in (('numerical', numeric), ('visual', visual)):
        if (receipt.get('passed') is not True or receipt.get('protocol_sha256') != protocol_sha
                or receipt.get('method') != method or receipt.get('seed') != staged['seed']):
            raise ValueError(f'Full {kind} gate identity/pass status differs from staged protocol')
    metrics = bound_json(numeric, 'pilot_evaluation_metrics', 'pilot_evaluation_metrics_sha256')
    if (metrics.get('protocol_sha256') != protocol_sha or metrics.get('method') != method
            or metrics.get('outer_seed') != staged['seed'] or metrics.get('stage') != 'pilot'):
        raise ValueError('Numerical gate refers to a different pilot evaluation')
    return end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'results/flatlands_external_formal_protocol_v1/protocol.json')
    parser.add_argument('--method', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--member', type=int, default=0)
    parser.add_argument('--stop-step', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.protocol = project_path(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    verify_sources(protocol)
    if args.seed not in protocol['seeds'] or args.method not in protocol['methods']:
        raise ValueError('Method/seed absent from sealed protocol')
    config = protocol['methods'][args.method]
    if not 0 <= args.member < config['members']:
        raise ValueError('Ensemble member not in sealed protocol')
    out = project_path(protocol['output_root'])
    family = 'external' if args.method in ('lama', 'flow') else 'controls'
    plan_file = out / 'full_plans' / f'{args.method}.json'
    full_plan = json.loads(plan_file.read_text()) if plan_file.exists() else None
    full_end = None
    if full_plan is not None:
        full_end = verify_full_plan(full_plan, protocol, sha(args.protocol), args.method)
    allowed = full_end or protocol['staged']['pilot_steps']
    permitted_boundaries = [protocol['staged']['smoke_steps'], protocol['staged']['pilot_steps']]
    if full_plan:
        permitted_boundaries += full_plan['checkpoint_steps']
    if args.stop_step not in permitted_boundaries:
        raise ValueError('Requested step is not a sealed stage/checkpoint boundary')
    if args.stop_step > allowed or args.stop_step < 1:
        raise ValueError('Full training requires sealed successful pilot gate')
    folder = out / 'runs' / args.method / str(args.seed) / f'member_{args.member}'
    recipe = {'protocol_sha256': sha(args.protocol), 'method': args.method, 'seed': args.seed,
              'member': args.member, 'config': config,
              'purpose': 'staged convergence then separately gated formal continuation',
              'test_assets_opened': False, 'validation_loaded_by_training_process': False}
    with FormalCheckpointStore(folder, recipe, resume=args.resume,
                               checkpoint_interval=protocol['training_runtime']['checkpoint_interval']) as store:
        state = store.load_latest()['state'] if args.resume else None
        step = state['completed_steps'] if state else 0
        durable_step = step if state else None
        if step > args.stop_step:
            raise ValueError('Cannot rewind an existing run')
        previous_full = state['extra'].get('full_plan_sha256') if state else None
        if previous_full and (not full_plan or previous_full != sha(plan_file)):
            raise ValueError('Attached full plan changed on resume')
        if step == args.stop_step and verify_completed_boundary(folder, step, sha(args.protocol)):
            print(json.dumps({'status': 'already_complete', 'step': step, 'unchanged_checkpoint': True}), flush=True)
            return
        prior_wall = session_totals(folder)
        session = ProcessSession(folder, {'method': args.method, 'seed': args.seed, 'member': args.member,
            'stop_step': args.stop_step, 'resumed_from_step': step, 'protocol_sha256': sha(args.protocol),
            'test_assets_opened': False})
        session_status, update_seconds = 'initializing', 0.
        try:
            signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
            configure_reproducible_cuda()
            torch.set_num_threads(protocol['training_runtime']['cpu_threads'])
            torch.cuda.set_per_process_memory_fraction(protocol['training_runtime']['allocator_limit_gib']*2**30 /
                                                       torch.cuda.get_device_properties(0).total_memory)
            initialization_seed = protocol['member_seeds'][args.method][str(args.seed)][args.member]
            random.seed(initialization_seed); np.random.seed(initialization_seed % 2**32); torch.manual_seed(initialization_seed)
            from pathrel.formal_data import load_formal_data
            from pathrel.formal_inference import make_models
            data_root = project_path(protocol['data_root'])
            if sha(data_root / 'seal.json') != protocol['data_seal_sha256']:
                raise ValueError('Training data seal differs from the frozen comparison protocol')
            samples = load_formal_data(data_root, 'train')
            limit = protocol['query']['training_queries_per_observation']
            samples = [replace(s, starts=s.starts[:limit], goals=s.goals[:limit], targets=s.targets[:limit],
                               candidate_indices=s.candidate_indices[:limit]) for s in samples]
            groups = {}
            for sample in samples:
                groups.setdefault(sample.row['parent_group'], []).append(sample)
            parents = [groups[key] for key in sorted(groups)]
            if len(parents) != protocol['partition']['train']:
                raise ValueError('Actual training parent count differs from the frozen protocol')
            if not parents or any(len(group) != protocol['partition']['train_views_per_parent'] for group in parents):
                raise ValueError('Training views no longer have equal parent weights')
            # Runtime metadata derives only from frozen input query eligibility;
            # never mutate the method config already hashed in run.json.
            runtime_config = dict(config,
                _n_query_views_by_parent={key: sum(bool(len(s.starts)) for s in group)
                                         for key, group in groups.items()},
                _views_per_parent=protocol['partition']['train_views_per_parent'])
            sampler = StatefulBatchSampler(len(parents), config['batch_size'], args.seed)
            generators = {'noise': torch.Generator(device='cuda').manual_seed(initialization_seed+1),
                          'views': torch.Generator(device='cpu').manual_seed(args.seed+2)}
            models = make_models(args.method, device='cuda')
            optimizers = {name: torch.optim.AdamW(module.parameters(), lr=config['learning_rate'],
                         weight_decay=config['weight_decay'], betas=tuple(config['betas'])) for name, module in models.items()}
            schedulers = {name: StagedLRScheduler(opt, pilot_steps=protocol['staged']['pilot_steps'], full_steps=full_end)
                          for name, opt in optimizers.items()}
            if state is not None:
                restored = restore_training_state(state, models=models, optimizers=optimizers, schedulers=schedulers,
                                                   sampler=sampler, generators=generators)
                update_seconds = restored['extra'].get('optimizer_update_seconds', 0.)
            torch.cuda.reset_peak_memory_stats()

            def wall_total():
                return prior_wall['training_wall_seconds'] + session.elapsed()

            def capture():
                return capture_training_state(completed_steps=step, models=models, optimizers=optimizers,
                    schedulers=schedulers, sampler=sampler, generators=generators,
                    extra={'training_wall_seconds': wall_total(),
                           'wall_time_is_lower_bound': prior_wall['wall_time_is_lower_bound'],
                           'sessions_without_final_receipt': prior_wall['sessions_without_final_receipt'],
                           'optimizer_update_seconds': update_seconds, 'active_session_id': session.token,
                           'session_receipts_directory': 'sessions',
                           'full_plan_sha256': sha(plan_file) if full_plan else None,
                           **gpu_resources()})

            def commit_if_needed():
                nonlocal durable_step
                if durable_step != step:
                    store.commit_completed_update(capture())
                    durable_step = step

            if not args.resume:
                store.initialize(capture())
                durable_step = 0
                snapshot(folder, 0)
            atomic_json(folder/'active.json', dict(session.metadata, family=family))
            session_status = 'running'
            while step < args.stop_step:
                if STOP or (out/'STOP').exists():
                    commit_if_needed()
                    store.write_interrupted('Requested pause at complete optimizer-update boundary')
                    session_status = 'paused'
                    raise SystemExit(130)
                ids = sampler.next_batch()
                views = torch.randint(protocol['partition']['train_views_per_parent'], (len(ids),), generator=generators['views']).tolist()
                batch = [parents[i][v] for i, v in zip(ids, views)]
                torch.cuda.synchronize(); began = time.monotonic()
                values = training_update(models, optimizers, args.method, batch, generators['noise'], runtime_config)
                if not all(np.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f'Nonfinite {args.method} loss')
                for schedule in schedulers.values():
                    schedule.step()
                torch.cuda.synchronize(); duration = time.monotonic()-began
                step += 1; update_seconds += duration
                progress = {'step': step, 'losses': values, 'update_seconds': duration,
                    'training_wall_seconds': wall_total(), 'wall_time_is_lower_bound': prior_wall['wall_time_is_lower_bound'],
                    'session_id': session.token, 'learning_rates': {k: o.param_groups[0]['lr'] for k, o in optimizers.items()},
                    'parent_indices': ids, 'view_indices': views, 'global_ids': [s.row['global_id'] for s in batch],
                    **gpu_resources(), 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
                store.append_progress(progress)
                session.heartbeat(last_complete_step=step, last_durable_step=durable_step, **gpu_resources())
                if step % protocol['training_runtime']['checkpoint_interval'] == 0 or step == args.stop_step or STOP:
                    commit_if_needed()
                if step % 25 == 0 or step == args.stop_step:
                    atomic_json(folder/'progress.json', progress)
                    print(json.dumps({k: progress[k] for k in ('step', 'losses', 'update_seconds', 'peak_allocated_bytes')}), flush=True)
            if STOP or (out/'STOP').exists():
                commit_if_needed()
                store.write_interrupted('Requested pause at final complete update')
                session_status = 'paused'
                raise SystemExit(130)
            checkpoint = snapshot(folder, step)
            receipt_path = folder / f'boundary_{step:08d}.json'
            if receipt_path.exists():
                verify_completed_boundary(folder, step, sha(args.protocol))
            else:
                atomic_json(receipt_path, {'method': args.method, 'seed': args.seed, 'member': args.member,
                    'completed_steps': step, 'checkpoint': str(checkpoint.relative_to(ROOT)),
                    'checkpoint_sha256': sha(checkpoint), 'protocol_sha256': sha(args.protocol),
                    'training_wall_seconds': wall_total(), 'wall_time_is_lower_bound': prior_wall['wall_time_is_lower_bound'],
                    'session_receipts_directory': 'sessions',
                    'optimizer_update_seconds': update_seconds, 'test_assets_opened': False,
                    'sufficient_training': False, 'note': 'Boundary only; sufficient training requires convergence and common evaluation.'})
            session_status = 'boundary_complete'
        except SystemExit:
            raise
        except BaseException as error:
            session_status = 'failed'
            # Live state can contain a partially updated G/D. Only durable state
            # is recoverable; initialization may have failed before it existed.
            receipt = store.write_interrupted(repr(error)) if (folder/'latest.pt').exists() else None
            if receipt:
                durable_step = receipt['completed_steps']
            failure = {'error': repr(error), 'last_valid_checkpoint': 'interrupted.pt' if receipt else None,
                'last_durable_step': receipt['completed_steps'] if receipt else None, 'last_live_complete_step': step,
                'category': 'resource' if isinstance(error, torch.cuda.OutOfMemoryError) else 'implementation_or_numerical_unresolved',
                'wall_seconds_this_session': session.elapsed(), 'session_id': session.token,
                **gpu_resources(), 'loss_curve': 'progress.jsonl', 'test_assets_opened': False}
            atomic_json(folder/'sessions'/f'{session.token}.failure.json', failure)
            atomic_json(folder/'failure.json', failure)
            raise
        finally:
            session.finish(status=session_status, last_complete_step=step, last_durable_step=durable_step,
                           **gpu_resources())
            atomic_json(folder/'time_summary.json', session_totals(folder))
            atomic_json(folder/'active.json', {'pid': None, 'method': args.method, 'seed': args.seed,
                'member': args.member, 'last_complete_step': step, 'session_id': session.token,
                'stopped_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
