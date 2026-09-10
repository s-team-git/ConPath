#!/usr/bin/env python3
"""Gate-bound three-seed full scheduling; validation occurs after calibration selection.

This new controller does not modify the frozen trainer, evaluator or protocol.
It shares the staged queue's exclusive lock and two-process GPU budget. A missing
numeric/visual pilot gate fails before any child is started. It never grants
paper-main-table eligibility: independent final auditing remains mandatory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.formal_checkpoint import atomic_json, recipe_hash
from pathrel.formal_data import sha256
from scripts import run_flatlands_formal_stages as stage_queue
from scripts.train_flatlands_formal import project_path, session_totals, verify_full_plan, verify_sources


SEEDS = [20260831, 20260901, 20260902]
SIGNAL_STOP_REQUESTED = False


def request_stop_signal(*unused):
    """Signal-safe with respect to Python locks: only assign a Boolean.

    The main thread may have been interrupted while holding stage_queue.LOCK.
    Calling its locking handler from here would deadlock that same thread.
    """
    global SIGNAL_STOP_REQUESTED
    SIGNAL_STOP_REQUESTED = True


def propagate_stop(out):
    """Call only outside stage_queue.LOCK, including during executor cleanup."""
    if (SIGNAL_STOP_REQUESTED or (out / 'STOP').exists()) and not stage_queue.STOP.is_set():
        stage_queue.request_stop()


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Formal artifact escapes project')
    return str(path.relative_to(ROOT))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def write_once(path, content):
    """Permit only exact verified reuse, never replacing an existing artifact."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise FileExistsError(f'Existing full-run artifact differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, content)


def frozen_contract(protocol_path, methods):
    protocol_path = project_path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    verify_sources(protocol)
    if protocol.get('seeds') != SEEDS or protocol['staged']['seed'] != SEEDS[0]:
        raise ValueError('Full queue requires the exact three registered outer seeds')
    if not methods or len(methods) > 2 or len(set(methods)) != len(methods):
        raise ValueError('At most two distinct registered method pipelines')
    if any(method not in protocol['methods'] for method in methods):
        raise ValueError('Unregistered full-run method')
    if sha256(project_path(protocol['data_root']) / 'seal.json') != protocol['data_seal_sha256']:
        raise ValueError('Full queue data seal differs from frozen protocol')
    if (protocol['test_lock'].get('final_test_locked') is not True
            or protocol['test_lock'].get('location_6_locked') is not True):
        raise ValueError('Final-test and location_6 locks are required')
    digest = sha256(protocol_path)
    out = project_path(protocol['output_root'])
    plans = {}
    for method in methods:
        file = out / 'full_plans' / f'{method}.json'
        if not file.is_file():
            raise ValueError(f'{method} has no approved full plan; no child may start')
        plan = json.loads(file.read_text())
        verify_full_plan(plan, protocol, digest, method)
        if plan.get('seeds') != SEEDS or plan.get('members_per_outer_seed') != protocol['methods'][method]['members']:
            raise ValueError('Full plan seed/member matrix differs from protocol')
        if plan.get('data_seal_sha256') != protocol['data_seal_sha256']:
            raise ValueError('Full plan refers to different data')
        plans[method] = {'plan': plan, 'path': file, 'sha256': sha256(file)}
    return protocol_path, protocol, digest, out, plans


def stopped(out):
    return SIGNAL_STOP_REQUESTED or stage_queue.STOP.is_set() or (out / 'STOP').exists()


def child(command, logfile, method, out):
    if stopped(out):
        raise InterruptedError('Full queue paused before child launch')
    stage_queue.run_child(command, logfile, method)


def _checkpoints(out, method, seed, members, step, protocol_hash):
    checkpoints = []
    for member in range(members):
        folder = out / 'runs' / method / str(seed) / f'member_{member}'
        path, receipt = folder / f'step_{step:08d}.pt', folder / f'boundary_{step:08d}.json'
        if not path.exists() or not receipt.exists():
            return None
        row = json.loads(receipt.read_text())
        if (row.get('method') != method or row.get('seed') != seed or row.get('member') != member
                or row.get('completed_steps') != step or row.get('protocol_sha256') != protocol_hash
                or row.get('checkpoint_sha256') != sha256(path)):
            raise ValueError('Training checkpoint boundary identity/hash mismatch')
        checkpoints.append(path)
    return checkpoints


def _model_identity(provenance):
    return [{key: member.get(key) for key in ('member_index', 'initialization_seed',
                                            'checkpoint_sha256', 'completed_steps')}
            for member in provenance]


def verify_evaluation(folder, *, protocol, protocol_hash, method, seed, step, stage, splits,
                      calibration_file=None):
    """Fully verify reuse of a completed calibration-only or validation-only run."""
    folder = Path(folder)
    metrics_file, complete_file = folder / 'metrics.json', folder / 'complete.json'
    if not complete_file.exists():
        return None
    if not metrics_file.exists():
        raise ValueError('Evaluation completion exists without metrics')
    metrics, complete = json.loads(metrics_file.read_text()), json.loads(complete_file.read_text())
    count = protocol['methods'][method]['members']
    expected_k = None if method == 'direct_query' else 1 if method == 'deterministic' else 4
    if (complete.get('passed') is not True or complete.get('metrics_sha256') != sha256(metrics_file)
            or complete.get('method') != method or complete.get('seed') != seed or complete.get('stage') != stage
            or complete.get('completed_steps') != [step] * count
            or metrics.get('method') != method or metrics.get('outer_seed') != seed
            or metrics.get('stage') != stage or metrics.get('protocol_sha256') != protocol_hash
            or set(metrics.get('splits', {})) != set(splits)
            or metrics.get('main_table_eligible') is not False
            or metrics.get('observed_and_support_constraints_passed') is not True
            or metrics.get('final_test_locked') is not True or metrics.get('location_6_locked') is not True
            or metrics.get('new_physical_test_images_opened') != 0):
        raise ValueError('Full evaluation identity/budget/split/test-lock mismatch')
    provenance = metrics['checkpoint_provenance']
    if len(provenance) != count or [p['member_index'] for p in provenance] != list(range(count)):
        raise ValueError('Full evaluation member group is incomplete or duplicated')
    for index, member in enumerate(provenance):
        if (member['completed_steps'] != step or member['initialization_seed'] !=
                protocol['member_seeds'][method][str(seed)][index]
                or not member.get('checkpoint')
                or sha256(project_path(member['checkpoint'])) != member['checkpoint_sha256']):
            raise ValueError('Full evaluation checkpoint identity changed')
        if stage == 'formal':
            # Calibration checkpoint costs are historical snapshots. Selected
            # validation is produced only after the complete budget, so its
            # recorded cost must still agree with the reconciled run summary.
            timing = project_path(member['checkpoint']).parent / 'time_summary.json'
            totals = json.loads(timing.read_text())
            if (member.get('training_time_summary_sha256') != sha256(timing)
                    or member.get('training_seconds') != totals['training_wall_seconds']
                    or member.get('training_wall_time_is_lower_bound') != totals['wall_time_is_lower_bound']):
                raise ValueError('Selected validation has stale training cost; preserve it and reconcile explicitly')
    for split, report in metrics['splits'].items():
        cases = report['cases']
        expected_ids = {s['global_id'] for s in protocol['scenes'][split]}
        if len(cases) != len(expected_ids) or {case['global_id'] for case in cases} != expected_ids:
            raise ValueError('Full evaluation does not contain every frozen split observation')
        if method != 'direct_query' and report['map']['actual_world_counts'] != [expected_k]:
            raise ValueError('Evaluation used a different actual output count')
        for case in cases:
            if (case['efficiency']['actual_world_count'] != expected_k
                    or sha256(folder / split / 'predictions' / f"{case['global_id']}.npz") != case['predictions_sha256']):
                raise ValueError('Evaluation prediction artifact or world count changed')
    calibration_path = folder / 'calibration_platt.json'
    calibrated = json.loads(calibration_path.read_text())
    if (calibrated.get('fit_split') != 'calibration' or calibrated.get('protocol_sha256') != protocol_hash
            or calibrated.get('prediction_model_identity') != _model_identity(provenance)):
        raise ValueError('Calibration file is not bound to these model checkpoints')
    if calibration_file is not None and sha256(calibration_path) != sha256(calibration_file):
        raise ValueError('Validation did not reuse the selected fixed calibration fit')
    receipt = folder / 'full_queue_artifact_receipt.json'
    files = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p != receipt}
    write_once(receipt, {'protocol_sha256': protocol_hash, 'method': method, 'seed': seed,
                        'step': step, 'stage': stage, 'files': files})
    return {'folder': folder, 'metrics': metrics, 'metrics_sha256': sha256(metrics_file),
            'calibration_file': calibration_path, 'calibration_sha256': sha256(calibration_path)}


def find_evaluation(parent, **arguments):
    if not parent.exists():
        return None
    attempts = sorted(parent.glob('attempt_*'), key=lambda p: int(p.name.split('_')[-1]))
    complete = []
    for attempt in attempts:
        result = verify_evaluation(attempt, **arguments)
        if result is not None:
            complete.append(result)
    if len(complete) > 1:
        raise ValueError('Multiple completed evaluations exist for one frozen identity')
    return complete[0] if complete else None


def next_attempt(parent):
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (parent / f'attempt_{number}').exists():
        number += 1
    return parent / f'attempt_{number}'


def candidate_record(step, evaluation):
    score = evaluation['metrics']['summary']['calibration']
    for name in ('event_brier', 'map_nll', 'map_brier'):
        if score.get(name) is not None and not _finite(score[name]):
            raise ValueError('Nonfinite calibration selection score')
    if score.get('event_brier') is None:
        raise ValueError('Calibration Event Brier is missing')
    if evaluation['metrics']['method'] != 'direct_query' and any(score.get(name) is None for name in ('map_nll', 'map_brier')):
        raise ValueError('Map-completion checkpoint is missing its calibration map scores')
    return {'step': step, 'event_brier': score['event_brier'], 'map_nll': score.get('map_nll'),
            'map_brier': score.get('map_brier'), 'evaluation': relative(evaluation['folder']),
            'metrics_sha256': evaluation['metrics_sha256'], 'calibration_file': relative(evaluation['calibration_file']),
            'calibration_sha256': evaluation['calibration_sha256'],
            'checkpoint_provenance': evaluation['metrics']['checkpoint_provenance'],
            'collapse_diagnostic': score.get('collapse_diagnostic'),
            'observed_and_support_constraints_passed': evaluation['metrics']['observed_and_support_constraints_passed']}


def select_checkpoint(candidates):
    """Global minimum and its 1e-5 tie set; independent of iteration order."""
    if not candidates or len({c['step'] for c in candidates}) != len(candidates):
        raise ValueError('Selection requires distinct complete checkpoint candidates')
    if any(not _finite(c.get('event_brier')) or (c.get('map_nll') is not None and not _finite(c['map_nll'])) for c in candidates):
        raise ValueError('Selection scores must be finite')
    minimum = min(c['event_brier'] for c in candidates)
    ties = [c for c in candidates if c['event_brier'] <= minimum + 1e-5]
    return min(ties, key=lambda c: (c['map_nll'] if c['map_nll'] is not None else 0., c['step']))


def install_selection(folder, candidates, *, protocol, protocol_hash, method, seed, full_plan_hash):
    selected = select_checkpoint(candidates)
    members = protocol['methods'][method]['members']
    if len(selected['checkpoint_provenance']) != members:
        raise ValueError('Selection has an incomplete member group')
    best_files = []
    for member in selected['checkpoint_provenance']:
        checkpoint = project_path(member['checkpoint'])
        target = checkpoint.parent / 'best.pt'
        if sha256(checkpoint) != member['checkpoint_sha256']:
            raise ValueError('Selected snapshot changed before linking best.pt')
        if target.exists():
            if sha256(target) != member['checkpoint_sha256']:
                raise FileExistsError('Existing best.pt differs; never replace a selected checkpoint')
        else:
            os.link(checkpoint, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        best_files.append(relative(target))
    minimum_full = min(protocol['staged']['allowed_full_steps'])
    selection = {'protocol_sha256': protocol_hash, 'full_plan_sha256': full_plan_hash, 'method': method,
        'seed': seed, 'candidate_steps': [c['step'] for c in candidates], 'candidates': candidates,
        'selection_rule': 'global minimum raw calibration Event Brier; within 1e-5 use map vote NLL then earlier step',
        'best_step': selected['step'], 'best_checkpoints': best_files,
        'calibration_file': selected['calibration_file'], 'calibration_sha256': selected['calibration_sha256'],
        'selected_metrics_sha256': selected['metrics_sha256'],
        'short_selected_checkpoint': selected['step'] < minimum_full,
        'main_table_eligible': False, 'final_test_locked': True,
        'note': 'Never substitute a worse long-trained checkpoint when a short checkpoint wins calibration selection.'}
    write_once(folder / 'selection.json', selection)
    return selection


def reconcile_training_time(folder):
    """Recover actual process cost even if final boundary preceded a hard kill.

    The frozen evaluator reads time_summary.json. Preserve any stale bytes in an
    immutable archive, then replace that mutable summary with the frozen driver's
    reconstruction from starts/finishes/heartbeats before evaluating again.
    """
    folder = Path(folder)
    session_paths = sorted((folder / 'sessions').glob('*.json'))
    starts = sorted((folder / 'sessions').glob('*.started.json'))
    if not starts:
        raise ValueError('Actual training session start receipts are missing')
    expected_names = {start.name.removesuffix('.started.json') + suffix
                      for start in starts for suffix in ('.started.json', '.heartbeat.json', '.finished.json', '.failure.json')}
    if any(path.name not in expected_names for path in session_paths):
        raise ValueError('Orphan or unknown training session receipt')
    evidence = []
    for start in starts:
        token = start.name.removesuffix('.started.json')
        for path in (start, start.with_name(token + '.heartbeat.json'), start.with_name(token + '.finished.json'),
                     start.with_name(token + '.failure.json')):
            if not path.exists():
                continue
            data = path.read_bytes()
            value = json.loads(data)
            if path.name.endswith(('.heartbeat.json', '.finished.json')):
                seconds = value.get('elapsed_seconds')
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError('Training session elapsed time must be finite and nonnegative')
            elif path.name.endswith('.failure.json'):
                # The frozen trainer records failure diagnostics separately;
                # their wall clock overlaps that session's finish/heartbeat.
                # Keep and bind the evidence without summing it a second time.
                seconds = value.get('wall_seconds_this_session')
                if (value.get('session_id') != token or type(seconds) not in (int, float)
                        or not math.isfinite(seconds) or seconds < 0):
                    raise ValueError('Training failure receipt identity/time is invalid')
            evidence.append({'path': relative(path), 'sha256': hashlib.sha256(data).hexdigest()})
    totals = session_totals(folder)
    if not _finite(totals['training_wall_seconds']) or totals['training_wall_seconds'] < 0:
        raise ValueError('Invalid reconstructed actual training time')
    if sorted((folder / 'sessions').glob('*.json')) != session_paths:
        raise ValueError('Training sessions changed while reconciling cost; wait for worker exit')
    for row in evidence:
        if sha256(project_path(row['path'])) != row['sha256']:
            raise ValueError('Training session changed while reconciling cost; wait for worker exit')
    target = folder / 'time_summary.json'
    before = target.read_bytes() if target.exists() else None
    old = json.loads(before) if before is not None else None
    directory = folder / 'time_reconciliation'
    directory.mkdir(exist_ok=True)
    if old != totals:
        if before is not None:
            digest = sha256(target)
            archive = directory / f'previous_summary_{digest}.json'
            if archive.exists():
                if sha256(archive) != digest:
                    raise ValueError('Immutable old training-time archive was changed')
            else:
                with archive.open('xb') as stream:
                    stream.write(before); stream.flush(); os.fsync(stream.fileno())
        atomic_json(target, totals)
    # All previous summaries stay available, so repeated reconciliation retains
    # the same provenance instead of forgetting the originally stale snapshot.
    archives = [{'path': relative(path), 'sha256': sha256(path)}
                for path in sorted(directory.glob('previous_summary_*.json'))]
    receipt = {'source': 'frozen driver.session_totals over actual process receipts',
        'totals': totals, 'session_files': evidence, 'previous_summary_archives': archives,
        'current_time_summary': relative(target), 'current_time_summary_sha256': sha256(target)}
    path = directory / f'reconciliation_{recipe_hash(receipt)}.json'
    write_once(path, receipt)
    return dict(receipt, receipt=relative(path), sha256=sha256(path))


def reconcile_checkpoint_runs(checkpoints):
    for folder in sorted({Path(path).parent for path in checkpoints}):
        reconcile_training_time(folder)


def training_audit(out, method, seed, members, full_steps, protocol_hash):
    snapshots = _checkpoints(out, method, seed, members, full_steps, protocol_hash)
    errors, records = [], []
    if snapshots is None:
        errors.append('full_budget_member_group_incomplete')
    for member in range(members):
        folder = out / 'runs' / method / str(seed) / f'member_{member}'
        progress = folder / 'progress.jsonl'
        if not progress.exists():
            errors.append(f'member{member}_progress_missing')
            continue
        rows = [json.loads(line) for line in progress.read_text().splitlines()]
        ordered = [row.get('step') for row in rows] == list(range(1, full_steps + 1))
        finite = all(row.get('losses') and all(_finite(v) for v in row['losses'].values()) for row in rows)
        if not ordered or not finite:
            errors.append(f'member{member}_incomplete_or_nonfinite_updates')
        timing = folder / 'time_summary.json'
        reconciliation = reconcile_training_time(folder)
        time_data = reconciliation['totals']
        if not reconciliation['session_files']:
            errors.append(f'member{member}_actual_training_sessions_missing')
        records.append({'member': member, 'updates': len(rows), 'all_update_losses_finite': finite,
            'progress_sha256': sha256(progress), 'time_summary': relative(timing) if timing.exists() else None,
            'time_summary_sha256': sha256(timing), 'actual_training_time': time_data,
            'time_reconciliation': reconciliation['receipt'], 'time_reconciliation_sha256': reconciliation['sha256'],
            'peak_allocated_bytes': max((r.get('peak_allocated_bytes', 0) for r in rows), default=0),
            'peak_reserved_bytes': max((r.get('peak_reserved_bytes', 0) for r in rows), default=0)})
    return {'passed': not errors, 'errors': errors, 'members': records,
        'training_seconds_sum_across_members': sum(r['actual_training_time']['training_wall_seconds']
                                                 for r in records if r['actual_training_time']),
        'time_is_lower_bound': any(r['actual_training_time'].get('wall_time_is_lower_bound', True)
                                  for r in records if r['actual_training_time'])}


def convergence_receipt(candidates, training, selection, *, protocol, protocol_hash, method, seed,
                        full_steps, full_plan_hash, out):
    tail = sorted(candidates, key=lambda c: c['step'])[-3:]
    reasons, trends = [], {}
    if full_steps < min(protocol['staged']['allowed_full_steps']) or len(tail) != 3:
        reasons.append('insufficient_full_budget_or_calibration_history')
    for metric in ('map_brier', 'event_brier'):
        values = [row.get(metric) for row in tail]
        if method == 'direct_query' and metric == 'map_brier':
            continue
        if len(values) != 3 or any(not _finite(v) for v in values):
            reasons.append(f'{metric}_missing_or_nonfinite')
            continue
        changes = [(values[0] - values[1]) / max(abs(values[0]), 1e-12),
                   (values[1] - values[2]) / max(abs(values[1]), 1e-12),
                   (values[0] - values[2]) / max(abs(values[0]), 1e-12)]
        trends[metric] = {'values': values, 'relative_reductions': changes,
                          'stable_below_two_percent': all(abs(value) < .02 for value in changes)}
        if not trends[metric]['stable_below_two_percent']:
            reasons.append(f'{metric}_still_improving_or_unstable')
    if not training['passed']:
        reasons.extend(training['errors'])
    if any(not c['observed_and_support_constraints_passed'] for c in candidates):
        reasons.append('constraint_violation')
    if method != 'direct_query' and any(c['collapse_diagnostic'] is None for c in tail):
        reasons.append('collapse_diagnostics_missing')
    numerical_passed = not reasons
    visual_path = out / 'full_reviews' / method / f'{seed}.json'
    visual, visual_hash = None, None
    if visual_path.exists():
        visual, visual_hash = json.loads(visual_path.read_text()), sha256(visual_path)
        if (visual.get('protocol_sha256') != protocol_hash or visual.get('method') != method
                or visual.get('seed') != seed or visual.get('full_plan_sha256') != full_plan_hash
                or visual.get('selected_calibration_metrics_sha256') != selection['selected_metrics_sha256']
                or visual.get('last_three_calibration_metrics_sha256') != [c['metrics_sha256'] for c in tail]):
            raise ValueError('Full visual review is not bound to the selected and final calibration outputs')
        if visual.get('passed') is not True or visual.get('no_collapse') is not True:
            reasons.append('full_visual_review_failed')
    else:
        reasons.append('full_visual_review_pending')
    if selection['short_selected_checkpoint']:
        reasons.append('short_checkpoint_selected_by_frozen_rule')
    return {'protocol_sha256': protocol_hash, 'full_plan_sha256': full_plan_hash,
        'method': method, 'seed': seed, 'full_steps': full_steps, 'selected_step': selection['best_step'],
        'last_three_calibration_steps': [c['step'] for c in tail],
        'last_three_calibration_metrics_sha256': [c['metrics_sha256'] for c in tail],
        'numerical_convergence_passed': numerical_passed, 'trends': trends,
        'collapse_diagnostics': [c['collapse_diagnostic'] for c in tail],
        'visual_review': relative(visual_path) if visual else None, 'visual_review_sha256': visual_hash,
        'sufficient_training': not reasons, 'insufficiency_reasons': reasons,
        'short_selected_checkpoint': selection['short_selected_checkpoint'],
        'main_table_eligible': False, 'final_test_locked': True, 'training_audit': training,
        'note': 'A completed budget and plateau do not establish superiority; final independent audit is still required.'}


class FullPipeline:
    def __init__(self, method, protocol, protocol_path, digest, out, plan_info, *, resume=False):
        self.method, self.protocol, self.protocol_path = method, protocol, protocol_path
        self.digest, self.out, self.plan_info = digest, out, plan_info
        self.plan = plan_info['plan']
        self.folder = out / 'full_runs' / method
        identity = {'method': method, 'protocol_sha256': digest, 'full_plan_sha256': plan_info['sha256'],
                    'controller_sha256': sha256(Path(__file__)), 'seeds': SEEDS,
                    'full_steps': self.plan['full_steps'], 'members': protocol['methods'][method]['members']}
        existed = self.folder.exists() and any(self.folder.iterdir())
        if existed and not resume:
            raise FileExistsError('Existing full pipeline requires explicit --resume')
        if resume and not existed:
            raise FileNotFoundError('Cannot resume a missing full-pipeline receipt directory')
        if resume and not (self.folder / 'run.json').is_file():
            raise ValueError('Nonempty directory has no legitimate full-pipeline run receipt')
        self.folder.mkdir(parents=True, exist_ok=True)
        write_once(self.folder / 'run.json', identity)
        self.status = dict(identity, status='ready', current_seed=None, current_step=None,
                           current_action=None, current_member=None, main_table_eligible=False)

    def update(self, **fields):
        self.status.update(fields, updated_utc=now())
        atomic_json(self.folder / 'status.json', self.status)

    def check_contract(self):
        if stopped(self.out):
            raise InterruptedError('Full queue paused')
        if sha256(self.protocol_path) != self.digest or sha256(self.plan_info['path']) != self.plan_info['sha256']:
            raise ValueError('Full queue protocol/plan changed during execution')

    def train_to(self, seed, step):
        self.check_contract()
        members = self.protocol['methods'][self.method]['members']
        ready = _checkpoints(self.out, self.method, seed, members, step, self.digest)
        if ready is not None:
            return ready
        for member in range(members):
            folder = self.out / 'runs' / self.method / str(seed) / f'member_{member}'
            # A previous complete member is retained while another resumes.
            file = folder / f'boundary_{step:08d}.json'
            if file.exists():
                continue
            self.check_contract()
            self.update(current_action='training', current_seed=seed, current_step=step, current_member=member)
            command = [sys.executable, str(ROOT / 'scripts/train_flatlands_formal.py'), '--protocol', str(self.protocol_path),
                       '--method', self.method, '--seed', str(seed), '--member', str(member), '--stop-step', str(step)]
            if folder.exists() and any(folder.iterdir()):
                command.append('--resume')
            child(command, self.out / 'logs' / f'full_{self.method}_{seed}_member{member}_step{step}', self.method, self.out)
        ready = _checkpoints(self.out, self.method, seed, members, step, self.digest)
        if ready is None:
            raise ValueError('Training returned without a complete common-step member group')
        return ready

    def calibration(self, seed, step):
        self.check_contract()
        staged_names = {self.protocol['staged']['smoke_steps']: 'smoke', self.protocol['staged']['pilot_steps']: 'pilot'}
        if seed == self.protocol['staged']['seed'] and step in staged_names:
            name = staged_names[step]
            stage = stage_queue.find_finished_eval(self.out / 'stages' / self.method / str(seed) / name,
                                                  self.digest, name, self.method, seed, step)
            if stage is None:
                raise ValueError('Approved first-seed stage evidence is missing; never rerun it silently')
            metrics = json.loads((stage / 'metrics.json').read_text())
            return {'folder': stage, 'metrics': metrics, 'metrics_sha256': sha256(stage / 'metrics.json'),
                    'calibration_file': stage / 'calibration_platt.json',
                    'calibration_sha256': sha256(stage / 'calibration_platt.json')}
        parent = self.out / 'evaluations' / self.method / str(seed) / f'checkpoint_{step:08d}'
        arguments = dict(protocol=self.protocol, protocol_hash=self.digest, method=self.method, seed=seed,
                         step=step, stage='checkpoint_calibration', splits=['calibration'])
        found = find_evaluation(parent, **arguments)
        if found is not None:
            return found
        checkpoints = self.train_to(seed, step)
        reconcile_checkpoint_runs(checkpoints)
        output = next_attempt(parent)
        self.update(current_action='calibration_only', current_seed=seed, current_step=step, current_member=None)
        command = [sys.executable, str(ROOT / 'scripts/evaluate_flatlands_formal.py'),
            '--protocol-dir', str(self.protocol_path.parent), '--data-dir', str(project_path(self.protocol['data_root'])),
            '--output-dir', str(output), '--method', self.method, '--seed', str(seed),
            '--stage', 'checkpoint_calibration', '--split', 'calibration']
        for checkpoint in checkpoints:
            command += ['--checkpoint', str(checkpoint)]
        child(command, self.out / 'logs' / f'full_{self.method}_{seed}_calibration{step}', self.method, self.out)
        result = find_evaluation(parent, **arguments)
        if result is None:
            raise ValueError('Calibration worker did not leave a valid completion receipt')
        return result

    def validation(self, seed, selection):
        self.check_contract()
        step = selection['best_step']
        parent = self.out / 'evaluations' / self.method / str(seed) / f'selected_validation_{step:08d}'
        calibration = project_path(selection['calibration_file'])
        if sha256(calibration) != selection['calibration_sha256']:
            raise ValueError('Selected calibration fit changed before final validation')
        reconcile_checkpoint_runs([project_path(path) for path in selection['best_checkpoints']])
        arguments = dict(protocol=self.protocol, protocol_hash=self.digest, method=self.method, seed=seed,
            step=step, stage='formal', splits=['validation'], calibration_file=calibration)
        found = find_evaluation(parent, **arguments)
        if found is not None:
            return found
        output = next_attempt(parent)
        self.update(current_action='selected_validation_only', current_seed=seed, current_step=step, current_member=None)
        command = [sys.executable, str(ROOT / 'scripts/evaluate_flatlands_formal.py'),
            '--protocol-dir', str(self.protocol_path.parent), '--data-dir', str(project_path(self.protocol['data_root'])),
            '--output-dir', str(output), '--method', self.method, '--seed', str(seed), '--stage', 'formal',
            '--split', 'validation', '--calibration-file', str(calibration)]
        for checkpoint in selection['best_checkpoints']:
            command += ['--checkpoint', str(project_path(checkpoint))]
        child(command, self.out / 'logs' / f'full_{self.method}_{seed}_selected_validation', self.method, self.out)
        result = find_evaluation(parent, **arguments)
        if result is None:
            raise ValueError('Selected validation worker did not leave a valid completion receipt')
        return result

    def run(self):
        self.update(status='running')
        summaries = []
        try:
            for seed in SEEDS:
                candidates = []
                for step in self.plan['checkpoint_steps']:
                    evaluation = self.calibration(seed, step)
                    candidates.append(candidate_record(step, evaluation))
                training = training_audit(self.out, self.method, seed, self.protocol['methods'][self.method]['members'],
                                          self.plan['full_steps'], self.digest)
                if not training['passed']:
                    raise ValueError('Full budget training audit failed: ' + ','.join(training['errors']))
                folder = self.folder / 'seeds' / str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                selection = install_selection(folder, candidates, protocol=self.protocol, protocol_hash=self.digest,
                    method=self.method, seed=seed, full_plan_hash=self.plan_info['sha256'])
                result = self.validation(seed, selection)
                convergence = convergence_receipt(candidates, training, selection, protocol=self.protocol,
                    protocol_hash=self.digest, method=self.method, seed=seed, full_steps=self.plan['full_steps'],
                    full_plan_hash=self.plan_info['sha256'], out=self.out)
                # New visual-review evidence may arrive after completion. Add a
                # separate immutable review version; never rerun validation.
                review_hash = recipe_hash(convergence)
                review_path = folder / 'convergence' / f'{review_hash}.json'
                write_once(review_path, convergence)
                atomic_json(folder / 'convergence.json', {'receipt': relative(review_path), 'sha256': sha256(review_path)})
                summary = {'seed': seed, 'selected_step': selection['best_step'],
                    'selection_sha256': sha256(folder / 'selection.json'), 'selected_validation': relative(result['folder']),
                    'validation_metrics_sha256': result['metrics_sha256'],
                    'short_selected_checkpoint': selection['short_selected_checkpoint'],
                    'full_steps_completed': self.plan['full_steps'], 'main_table_eligible': False,
                    'full_plan_sha256': self.plan_info['sha256'], 'protocol_sha256': self.digest}
                write_once(folder / 'complete.json', summary)
                summaries.append(dict(summary, convergence_receipt=relative(review_path),
                                      sufficient_training=convergence['sufficient_training']))
                self.update(completed_seeds=summaries)
            self.update(status='all_three_seeds_evaluated_waiting_final_audit', current_action=None,
                        current_member=None, completed_seeds=summaries)
        except BaseException as error:
            self.update(status='paused' if stopped(self.out) or isinstance(error, InterruptedError) else 'failed',
                        error=repr(error), completed_seeds=summaries)
        return self.status


def run_pipelines(pipelines, out, digest, methods):
    """Keep forwarding safe stop requests while waiting for workers to exit."""
    pool = ThreadPoolExecutor(max_workers=2)
    futures = []
    completed_normally = False
    try:
        for pipeline in pipelines:
            futures.append(pool.submit(pipeline.run))
        pending = set(futures)
        while pending:
            propagate_stop(out)
            with stage_queue.LOCK:
                children = {method: process.pid for method, process in stage_queue.CHILDREN.items()
                            if process.poll() is None}
            # A signal arriving inside the preceding lock only sets our flag.
            propagate_stop(out)
            atomic_json(out / 'queue.json', {'pid': os.getpid(), 'phase': 'approved_full_training',
                'protocol_sha256': digest, 'methods': methods, 'children': children,
                'maximum_gpu_workers': 2, 'project_gpu_budget_gib': 60, 'updated_utc': now()})
            _, pending = wait(pending, timeout=.25)
        results = [future.result() for future in futures]
        completed_normally = True
        return results
    finally:
        if not completed_normally:
            request_stop_signal()
        # Avoid ThreadPoolExecutor.__exit__ blocking inside an implicit join:
        # signals must still be propagated outside locks during error cleanup.
        pending = {future for future in futures if not future.done()}
        while pending:
            propagate_stop(out)
            _, pending = wait(pending, timeout=.1)
        propagate_stop(out)
        pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'results/flatlands_external_formal_protocol_v1/protocol.json')
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--resume', action='store_true', help='Verify and continue existing full-pipeline receipts')
    args = parser.parse_args()
    protocol_path, protocol, digest, out, plans = frozen_contract(args.protocol, args.methods)
    if stopped(out):
        raise ValueError('Shared STOP marker is set; full queue remains paused')
    with (out / '.queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check and claim every method before launching any child process.
        for method in args.methods:
            directory = out / 'full_runs' / method
            existed = directory.exists() and any(directory.iterdir())
            if existed != args.resume:
                raise ValueError('All requested pipelines must be new, or all explicitly resumed; launch mixed states separately')
        pipelines = [FullPipeline(method, protocol, protocol_path, digest, out, plans[method], resume=args.resume)
                     for method in args.methods]
        signal.signal(signal.SIGINT, request_stop_signal)
        signal.signal(signal.SIGTERM, request_stop_signal)
        session = out / 'full_queue_sessions' / (uuid.uuid4().hex + '.json')
        started = time.monotonic()
        results = []
        try:
            results = run_pipelines(pipelines, out, digest, args.methods)
        finally:
            write_once(session, {'protocol_sha256': digest, 'methods': args.methods,
                'controller_sha256': sha256(Path(__file__)), 'elapsed_seconds': time.monotonic() - started,
                'finished_utc': now(), 'results': results, 'main_table_eligible': False})
            atomic_json(out / 'queue.json', {'pid': None, 'phase': 'approved_full_training',
                'methods': args.methods, 'results': results, 'finished_utc': now(), 'main_table_eligible': False})
        print(json.dumps(results, ensure_ascii=False), flush=True)
        if any(result['status'] in ('failed', 'paused') for result in results):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
