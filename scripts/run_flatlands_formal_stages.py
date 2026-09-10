#!/usr/bin/env python3
"""Run at most two registered method pipelines through smoke and pilot.

This queue never starts full training. A completed pilot still requires the
frozen numerical gate, fixed-case visual review and a separately sealed plan.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from pathrel.formal_checkpoint import atomic_json
from pathrel.formal_data import sha256

STOP=threading.Event()
CHILDREN={}
LOCK=threading.Lock()


def request_stop(*unused):
    STOP.set()
    with LOCK:
        for child in list(CHILDREN.values()):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)


def run_child(command, logfile, method):
    if STOP.is_set():
        raise InterruptedError('Queue paused')
    logfile.parent.mkdir(parents=True,exist_ok=True)
    # Each invocation has a separate immutable log, including resumed sessions.
    attempt=1
    while logfile.with_suffix(f'.attempt{attempt}.log').exists():
        attempt+=1
    logfile=logfile.with_suffix(f'.attempt{attempt}.log')
    with logfile.open('x') as stream:
        with LOCK:
            if STOP.is_set(): raise InterruptedError('Queue paused before worker creation')
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                   env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'4'})
            CHILDREN[method]=child
        # Signal handlers can run between Popen and registration on main-thread
        # launches; the second check also covers stops racing thread startup.
        if STOP.is_set() and child.poll() is None: child.send_signal(signal.SIGINT)
        try:
            code=child.wait()
        finally:
            with LOCK: CHILDREN.pop(method,None)
    if code:
        raise RuntimeError(f'{method} child exit={code}; log={logfile}')


def find_finished_eval(parent, protocol_hash, stage, method, seed, expected_step):
    if not parent.exists(): return None
    for attempt in sorted(parent.glob('attempt_*')):
        file=attempt/'metrics.json'
        completion=attempt/'complete.json'
        if file.exists() and completion.exists():
            complete=json.loads(completion.read_text())
            if not complete.get('passed') or complete.get('metrics_sha256')!=sha256(file):
                raise ValueError('Existing completed evaluation receipt has failed verification')
            result=json.loads(file.read_text())
            member_count=4 if method=='lama' else 1
            if (result['protocol_sha256']!=protocol_hash or result['stage']!=stage
                or result['method']!=method or result['outer_seed']!=seed
                or complete.get('method')!=method or complete.get('seed')!=seed or complete.get('stage')!=stage
                or complete.get('completed_steps')!=[expected_step]*member_count
                or set(result['splits'])!={'calibration','validation'}
                or not result['observed_and_support_constraints_passed']
                or not result['final_test_locked'] or not result['location_6_locked']
                or result['new_physical_test_images_opened']!=0):
                raise ValueError('Existing completed stage identity/budget/split/test-lock mismatch')
            provenance=result['checkpoint_provenance']
            if len(provenance)!=member_count or [p['member_index'] for p in provenance]!=list(range(member_count)):
                raise ValueError('Existing stage has missing/duplicated members')
            for member in provenance:
                if member['completed_steps']!=expected_step:
                    raise ValueError('Existing stage checkpoint step mismatch')
                if member['checkpoint'] and sha256(ROOT/member['checkpoint'])!=member['checkpoint_sha256']:
                    raise ValueError('Checkpoint was changed after evaluation')
            for split,report in result['splits'].items():
                if method!='direct_query' and report['map']['actual_world_counts']!=[1 if method=='deterministic' else 4]:
                    raise ValueError('Completed stage did not use real registered K')
                for case in report['cases']:
                    if sha256(attempt/split/'predictions'/f"{case['global_id']}.npz")!=case['predictions_sha256']:
                        raise ValueError('Prediction artifact changed after evaluation')
            calibration=json.loads((attempt/'calibration_platt.json').read_text())
            identity=[{k:p.get(k) for k in ('member_index','initialization_seed','checkpoint_sha256','completed_steps')} for p in provenance]
            if (calibration['protocol_sha256']!=protocol_hash or calibration['fit_split']!='calibration'
                or calibration['prediction_model_identity']!=identity):
                raise ValueError('Completed calibration belongs to different data/models')
            artifact_receipt=attempt/'queue_artifact_receipt.json'
            files={str(p.relative_to(attempt)):sha256(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p!=artifact_receipt}
            if artifact_receipt.exists():
                if json.loads(artifact_receipt.read_text())['files']!=files:
                    raise ValueError('Previously closed stage artifacts changed')
            else:
                atomic_json(artifact_receipt,{'protocol_sha256':protocol_hash,'method':method,'seed':seed,
                    'stage':stage,'completed_step':expected_step,'files':files})
            return attempt
    return None


def method_stages(method, protocol, protocol_path):
    out=ROOT/protocol['output_root']; data=ROOT/protocol['data_root']
    seed=protocol['staged']['seed']; config=protocol['methods'][method]
    receipt=out/'stage_status'/f'{method}.json'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    status={'method':method,'seed':seed,'protocol_sha256':sha256(protocol_path),'status':'running',
            'completed_evaluations':{},'full_training_started':False,'test_assets_opened':False}
    try:
        for stage,step in [('untrained',0),('smoke',protocol['staged']['smoke_steps']),('pilot',protocol['staged']['pilot_steps'])]:
            status.update(current_stage=stage,step=step)
            atomic_json(receipt,status)
            parent=out/'stages'/method/str(seed)/stage
            finished=find_finished_eval(parent,status['protocol_sha256'],stage,method,seed,step)
            if finished:
                status['completed_evaluations'][stage]=str(finished.relative_to(ROOT))
                continue
            checkpoints=[]
            if step:
                for member in range(config['members']):
                    if STOP.is_set() or (out/'STOP').exists():
                        raise InterruptedError('Queue paused before next training worker')
                    folder=out/'runs'/method/str(seed)/f'member_{member}'
                    status.update(current_member=member,current_action='training')
                    atomic_json(receipt,status)
                    command=[sys.executable,str(ROOT/'scripts/train_flatlands_formal.py'),'--protocol',str(protocol_path),
                             '--method',method,'--seed',str(seed),'--member',str(member),'--stop-step',str(step)]
                    if folder.exists() and any(folder.iterdir()): command.append('--resume')
                    run_child(command,out/'logs'/f'{method}_{seed}_member{member}_step{step}',method)
                    checkpoints.append(folder/f'step_{step:08d}.pt')
            parent.mkdir(parents=True,exist_ok=True)
            index=1
            while (parent/f'attempt_{index}').exists(): index+=1
            evaluation=parent/f'attempt_{index}'
            if STOP.is_set() or (out/'STOP').exists():
                raise InterruptedError('Queue paused before evaluation worker')
            status.update(current_action='calibration_and_validation',evaluation=str(evaluation.relative_to(ROOT)))
            atomic_json(receipt,status)
            command=[sys.executable,str(ROOT/'scripts/evaluate_flatlands_formal.py'),'--protocol-dir',str(protocol_path.parent),
                     '--data-dir',str(data),'--output-dir',str(evaluation),'--method',method,'--seed',str(seed),'--stage',stage]
            for checkpoint in checkpoints: command+=['--checkpoint',str(checkpoint)]
            run_child(command,out/'logs'/f'{method}_{seed}_{stage}_evaluation',method)
            if find_finished_eval(parent,status['protocol_sha256'],stage,method,seed,step)!=evaluation:
                raise ValueError('Stage completion failed artifact/identity verification')
            status['completed_evaluations'][stage]=str(evaluation.relative_to(ROOT))
            atomic_json(receipt,status)
        status.update(status='pilot_evaluated_waiting_numerical_and_visual_review',current_action=None,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(status='paused' if STOP.is_set() or isinstance(error,InterruptedError) else 'failed',error=repr(error),
                      stopped_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(receipt,status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'results/flatlands_external_formal_protocol_v1/protocol.json')
    parser.add_argument('--methods',nargs='+',default=['lama','flow'])
    args=parser.parse_args()
    if len(args.methods)>2 or len(set(args.methods))!=len(args.methods):
        raise ValueError('At most two distinct simultaneous method pipelines')
    protocol=json.loads(args.protocol.read_text()); out=ROOT/protocol['output_root']
    from scripts.train_flatlands_formal import verify_sources
    verify_sources(protocol)
    if sha256(ROOT/protocol['data_root']/'seal.json')!=protocol['data_seal_sha256']:
        raise ValueError('Queue data seal differs from frozen protocol')
    if (out/'STOP').exists(): raise ValueError('STOP marker exists; queue remains paused')
    if any(m not in protocol['methods'] for m in args.methods): raise ValueError('Unregistered method')
    with (out/'.queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
        atomic_json(out/'queue.json',{'pid':os.getpid(),'methods':args.methods,'phase':'staged_only',
                    'maximum_gpu_workers':2,'started_utc':datetime.now(timezone.utc).isoformat()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(method_stages,m,protocol,args.protocol) for m in args.methods]
            results=[f.result() for f in futures]
        atomic_json(out/'queue.json',{'pid':None,'phase':'staged_only','methods':args.methods,'results':results,
                    'finished_utc':datetime.now(timezone.utc).isoformat(),'full_training_started':False})
        print(json.dumps(results,ensure_ascii=False),flush=True)
        if any(r['status'] in ('failed','paused') for r in results): raise SystemExit(1)


if __name__=='__main__': main()
