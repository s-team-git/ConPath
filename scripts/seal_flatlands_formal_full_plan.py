#!/usr/bin/env python3
"""Assess staged evidence, and seal full training only after a fixed-case review."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from pathrel.formal_checkpoint import atomic_json
from pathrel.formal_data import sha256
from scripts.run_flatlands_formal_stages import find_finished_eval


def assess(method, protocol, protocol_path):
    out=ROOT/protocol['output_root']; seed=protocol['staged']['seed']; digest=sha256(protocol_path)
    evaluations={}; files={}
    for stage,step in [('untrained',0),('smoke',protocol['staged']['smoke_steps']),('pilot',protocol['staged']['pilot_steps'])]:
        folder=find_finished_eval(out/'stages'/method/str(seed)/stage,digest,stage,method,seed,step)
        if folder is None: raise ValueError(f'{method} {stage} has not completed common evaluation')
        files[stage]={'path':str((folder/'metrics.json').relative_to(ROOT)),'sha256':sha256(folder/'metrics.json')}
        evaluations[stage]=json.loads((folder/'metrics.json').read_text())
    checks={}; losses=[]
    key='reconstruction' if method=='lama' else 'velocity_mse' if method=='flow' else 'event_loss' if method=='direct_query' else 'map_nll'
    for member in range(protocol['methods'][method]['members']):
        path=out/'runs'/method/str(seed)/f'member_{member}'/'progress.jsonl'
        lines=path.read_bytes().splitlines(keepends=True)[:protocol['staged']['pilot_steps']]
        records=[json.loads(line) for line in lines]
        if len(records)!=protocol['staged']['pilot_steps'] or [r['step'] for r in records]!=list(range(1,len(records)+1)):
            raise ValueError('Pilot progress is incomplete or out of order')
        values=np.array([r['losses'][key] for r in records])
        first,last=float(values[:100].mean()),float(values[-100:].mean())
        reduction=(first-last)/max(abs(first),1e-12)
        losses.append({'member':member,'loss':key,'first100_mean':first,'last100_mean':last,
            'relative_reduction':reduction,'pilot_progress_prefix_sha256':hashlib.sha256(b''.join(lines)).hexdigest()})
        checks[f'member{member}_finite_loss']=bool(np.isfinite(values).all())
        checks[f'member{member}_unknown_loss_improved']=bool(reduction>=protocol['staged']['automated_gate']['unknown_loss_relative_reduction_first100_to_last100_min'])
    for split in ('calibration','validation'):
        before=evaluations['untrained']['summary'][split]; after=evaluations['pilot']['summary'][split]
        for metric in (('event_brier',) if method=='direct_query' else ('map_brier','event_brier')):
            protocol_key=f'{split}_'+('hidden_map' if metric=='map_brier' else 'event')+'_brier_improvement_vs_untrained_min'
            checks[f'{split}_{metric}_improved']=bool(before[metric]-after[metric]>=protocol['staged']['automated_gate'][protocol_key])
        checks[f'{split}_evidence_and_support_preserved']=bool(evaluations['pilot']['observed_and_support_constraints_passed'])
    before=evaluations['smoke']['summary']['calibration']; after=evaluations['pilot']['summary']['calibration']
    trends={metric:(before[metric]-after[metric])/max(abs(before[metric]),1e-12)
            for metric in ('map_brier','event_brier') if before[metric] is not None}
    full_steps=10000 if all(gain<.02 for gain in trends.values()) else 20000
    return {'method':method,'seed':seed,'protocol_sha256':digest,'passed':all(checks.values()),
        'checks':checks,'losses':losses,'stage_metrics':files,
        'pilot_evaluation_metrics':files['pilot']['path'],'pilot_evaluation_metrics_sha256':files['pilot']['sha256'],
        'calibration_relative_improvements_1000_to_5000':trends,'proposed_full_steps':full_steps,
        'visual_review_required':True,'full_training_authorized_by_this_numeric_file_alone':False,
        'failure_interpretation':'A failed pilot does not establish external-method inferiority; retain checkpoints/losses/resource records and exclude superiority main table.',
        'created_utc':datetime.now(timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'results/flatlands_external_formal_protocol_v1/protocol.json')
    parser.add_argument('--method',required=True)
    parser.add_argument('--visual-review',type=Path)
    args=parser.parse_args(); protocol=json.loads(args.protocol.read_text()); out=ROOT/protocol['output_root']
    if args.method not in protocol['methods']: raise ValueError('Unregistered method')
    gate=assess(args.method,protocol,args.protocol)
    numeric_path=out/'full_gates'/args.method/'numeric.json'
    numeric_path.parent.mkdir(parents=True,exist_ok=True)
    if numeric_path.exists():
        existing=json.loads(numeric_path.read_text())
        if {k:v for k,v in gate.items() if k!='created_utc'}!={k:v for k,v in existing.items() if k!='created_utc'}:
            raise ValueError('Previously frozen numerical gate changed')
        gate=existing
    else: atomic_json(numeric_path,gate)
    if not args.visual_review:
        print(json.dumps({'numeric_gate':str(numeric_path),'passed':gate['passed'],'waiting_fixed_case_visual_review':True})); return
    visual_path=args.visual_review.resolve()
    if not visual_path.is_relative_to(ROOT): raise ValueError('Visual review must be a project artifact')
    visual=json.loads(visual_path.read_text())
    if (not gate['passed'] or visual.get('passed') is not True or visual.get('protocol_sha256')!=sha256(args.protocol)
        or visual.get('method')!=args.method or visual.get('seed')!=protocol['staged']['seed']
        or visual.get('figure_cases_sha256')!=protocol['figures']['sha256']
        or set(visual.get('reviewed_stages',[]))!={'untrained','smoke','pilot'}):
        raise ValueError('Numerical gate and matching fixed-case visual review must both pass')
    full_steps=gate['proposed_full_steps']
    if full_steps not in protocol['staged']['allowed_full_steps']: raise ValueError('Budget outside staged decision rule')
    plan_path=out/'full_plans'/f'{args.method}.json'
    plan_path.parent.mkdir(parents=True,exist_ok=True)
    if plan_path.exists(): raise FileExistsError('Full plan already frozen; refusing overwrite')
    atomic_json(plan_path,{'method':args.method,'protocol_sha256':sha256(args.protocol),'pilot_gate_passed':True,
        'pilot_gate_receipt':str(numeric_path.relative_to(ROOT)),'pilot_gate_sha256':sha256(numeric_path),
        'visual_review_receipt':str(visual_path.relative_to(ROOT)),'visual_review_sha256':sha256(visual_path),
        'full_steps':full_steps,'checkpoint_steps':[1000,5000]+list(range(7500,full_steps+1,2500)),
        'seeds':protocol['seeds'],'members_per_outer_seed':protocol['methods'][args.method]['members'],
        'schedule':'constant through5000 then cosine to.1 of registered base LR',
        'data_seal_sha256':protocol['data_seal_sha256'],'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'sufficient_training_already_established':False})
    print(json.dumps({'full_plan':str(plan_path),'sha256':sha256(plan_path),'full_steps':full_steps}))


if __name__=='__main__':main()
