#!/usr/bin/env python3
"""Seal a shared staged external-baseline protocol before any formal training."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import torch
from pathrel.formal_checkpoint import atomic_json
from pathrel.formal_data import sha256
from pathrel.formal_inference import LABELS, METHODS

PROTOCOL = ROOT/'results/flatlands_external_formal_protocol_v1'
DATA = PROTOCOL/'data_eligible_v1'
OUT = ROOT/'results/flatlands_external_formal_v1'
SEEDS = [20260831, 20260901, 20260902]


def main():
    if (PROTOCOL/'protocol.json').exists():
        raise FileExistsError('Formal protocol is already frozen; never replace it')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Formal results directory is nonempty; refusing a new experiment over it')
    audit = json.loads((DATA/'data_audit.json').read_text())
    data_protocol = json.loads((DATA/'data_protocol.json').read_text())
    resume_file = ROOT/'results/flatlands_external_resume_v1/verification.json'
    resume = json.loads(resume_file.read_text())
    if not audit['passed'] or not resume['passed']:
        raise ValueError('Data/resume gates have not passed')
    for relative, digest in json.loads((DATA/'seal.json').read_text())['files'].items():
        path = (DATA/relative).resolve()
        if not path.is_relative_to(DATA.resolve()) or sha256(path) != digest:
            raise ValueError('Sealed data artifact changed: '+relative)
    for method, receipt in resume['methods'].items():
        if sha256(ROOT/receipt['report']) != receipt['report_sha256'] or not receipt['complete_loss_curve_bitwise_equal'] or receipt['state_differences']:
            raise ValueError('Resume verification failed: '+method)
    if audit['new_physical_test_images_opened'] or audit['new_final_test_images_opened'] or audit['location_6_opened']:
        raise ValueError('Test lock has been violated')
    selected = list(csv.DictReader((DATA/'selected.csv').open()))
    cases = DATA/'figure_cases_v1.json'
    if not cases.exists():
        raise ValueError('Qualitative cases must be frozen before training/model outputs')
    if not json.loads((DATA/'figure_geometry_verification_v1.json').read_text())['passed']:
        raise ValueError('Frozen qualitative geometry did not pass independent checks')
    methods, member_seeds = {}, {}
    for method in METHODS:
        external = method in ('lama', 'flow')
        methods[method] = {
            'label': LABELS[method], 'members': 4 if method == 'lama' else 1,
            'optimizer': 'AdamW', 'learning_rate': 1e-4 if external else 3e-4,
            'weight_decay': .01 if external else .0001, 'betas': [.9, .999],
            'batch_size': 8, 'micro_batch': 8 if method == 'lama' else 4 if method == 'flow' else 2,
            'gradient_accumulation': 1 if method == 'lama' else 2 if method == 'flow' else 4,
            'train_samples': 4, 'radii_cells': [0,10,20], 'max_reachability_steps': 256,
            'event_weight': 0. if method == 'no_reach' else 2., 'variogram_weight': .1,
            'gradient_clip': None if external else 5., 'from_scratch': True, 'official_checkpoint': False,
            'actual_K': None if method == 'direct_query' else 1 if method == 'deterministic' else 4,
            'selection': 'minimum equal-parent calibration raw Event Brier at K4; ties within 1e-5 use hidden-map vote NLL then earlier step; all four LaMa members share the selected step',
        }
        member_seeds[method] = {str(seed): [seed if member == 0 else
            int(hashlib.sha256(f'formal-v1|{seed}|lama|{member}'.encode()).hexdigest()[:8],16)
            for member in range(methods[method]['members'])] for seed in SEEDS}
    methods['lama'].update({'generator': 'pinned Big-LaMa BEV ngf64,18FFC,ratio.75,3downsampling; 50966465 parameters/member',
        'discriminator': 'NLayerDiscriminator ndf64 n_layers4; 6960321 parameters/member',
        'loss': '10 hidden L1 + 10 hinge generator adversarial + 250 multi-scale discriminator feature MSE; separate hinge discriminator; G then D',
        'batchnorm': 'native BN over micro8; no BN activation recomputation, no SyncBN equivalence claimed',
        'perceptual_networks': None, 'K4_semantics': 'four independent trained members; never repeat one deterministic map'})
    methods['flow'].update({'network': 'existing U-Net width64 multipliers1/2/4/4,2resblocks/stage,4head conditional full spatial attention at64²/32²; 21202753 parameters',
        'loss': 'unknown/support masked velocity MSE; Gaussian start; linear interpolant; conditioning dropout .1',
        'solver': 'Heun25, CFG s2, per-substep unknown/support projection',
        'velocity_evaluations_per_sample': 50, 'batched_network_forwards_K4': 50,
        'single_image_equivalent_forwards_K4': 400})
    methods['conpath']['architecture_note'] = 'Previously frozen coherent categorical candidate, fixed Gaussian copula 9x9 sigma3; unchanged F16/latent4/local5 learned heads. Original decoder is a separately registered control.'
    methods['independent']['architecture_note'] = 'Matched F16 PathRelNet, local1 and disable_global_factors; independent cell draws; same map/variogram/event losses and batch/query weighting.'
    methods['direct_query']['geometry_note'] = 'Distance feature is d_cells/120, radius r_cells/20. Legacy argument adapter is dimensionless, not metres.'
    names = [str(p.relative_to(ROOT)) for p in sorted((ROOT/'src/pathrel').glob('*.py'))]
    names += ['scripts/train_flatlands_formal.py', 'scripts/evaluate_flatlands_formal.py',
              'scripts/freeze_flatlands_formal_protocol.py', 'scripts/check_formal_resume_smoke.py',
              'scripts/prepare_flatlands_formal_data.py', 'scripts/quarantine_flatlands_formal_duplicates.py',
              'scripts/run_flatlands_formal_stages.py', 'scripts/seal_flatlands_formal_full_plan.py',
              'scripts/freeze_flatlands_formal_figures.py',
              'third_party/lama/sources.json']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'third_party/lama/saicinpainting').rglob('*.py'))]
    source_hashes = {name: sha256(ROOT/name) for name in names}
    evaluator_files = {name: source_hashes[name] for name in ('scripts/evaluate_flatlands_formal.py',
        'src/pathrel/formal_metrics.py', 'src/pathrel/formal_inference.py')}
    dirty = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    for name, digest in source_hashes.items():
        committed = subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError('Commit the reviewed frozen implementation first: '+name)
    scenes = {split: [{'global_id':r['global_id'],'parent_group':r['parent_group'],'source':r['source_dataset'],
                       'packet_directory':r['packet_directory']} for r in selected if r['candidate_split']==split]
              for split in ('train','calibration','validation')}
    protocol = {
        'id': 'flatlands_external_formal_protocol_v1', 'version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'research_question': 'Joint spatial stochastic map posterior plus footprint-aware reachability versus ordinary completion, independent Bernoulli worlds, and direct event prediction',
        'scope': 'Validation-only development comparison. Three seeds and convergence required; historically reused development locations are not a fresh final test.',
        'initial_repository_head': 'e951289c27fc1f1f282b072c36f060247dc566d4',
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'unrelated_worktree_status_at_freeze':dirty,
        'output_root': str(OUT.relative_to(ROOT)), 'data_root':str(DATA.relative_to(ROOT)),
        'data_seal_sha256':sha256(DATA/'seal.json'), 'data_protocol_sha256':sha256(DATA/'data_protocol.json'),
        'data_audit_sha256':sha256(DATA/'data_audit.json'),
        'partition': {**{split:audit['summary'][split]['parents'] for split in scenes}, 'train_views_per_parent':2},
        'scene_manifest': {'path':str((DATA/'selected.csv').relative_to(ROOT)), 'sha256':sha256(DATA/'selected.csv')},
        'scenes':scenes, 'input':data_protocol['input'],
        'query': {'path':str((DATA/'queries.csv').relative_to(ROOT)), 'sha256':sha256(DATA/'queries.csv'),
            'pre_target_input_manifest':str((DATA/'input_queries_before_target.jsonl').relative_to(ROOT)),
            'pre_target_input_manifest_sha256':sha256(DATA/'input_queries_before_target.jsonl'),
            'start':'nearest observed-free cell to supplied camera; fixed before target decoding',
            'goals':'polar distances40/80/120 cells, angles0:30:330; keep input-eligible unknown/support-valid cells, including GT-blocked negatives',
            'radii_cells':[0,10,20], 'radius_unit':'grid cell; no conversion to metres',
            'training_queries_per_observation':8, 'training_order':'input-only SHA256 rank(global_id,candidate_index)',
            'evaluation_queries':'all frozen queries', 'geometry':'4-neighbor connectivity, exact Euclidean integer disk, out-of-map blocked'},
        'sampling':{'K':4,'solver_steps':25,'guidance':2.0,
            'map_score':'cellwise free frequency across the actual four binary worlds; not path probability',
            'event_score':'mean exact footprint connectivity indicator computed independently on each world; raw finite-sample event frequency',
            'deterministic_and_direct':'actual K1 and N/A separately disclosed; no copied maps',
            'decision_threshold':.5,'confidence_threshold':.8,'raw_K4_quantization':[0,.25,.5,.75,1]},
        'seeds':SEEDS, 'member_seeds':member_seeds,'lama_member_seeds':member_seeds['lama'],'methods':methods,
        'training_runtime':{'max_simultaneous_gpu_processes':2,'project_gpu_budget_gib':60,
            'allocator_limit_gib':28,'cpu_threads':4,'checkpoint_interval':100,'data_workers':0,
            'precision':'FP32, deterministic CUDA, TF32 off, cuDNN benchmark off',
            'sampling':'permuted parents, one uniformly selected frozen view per parent; dedicated sampler/view RNG shared across methods within outer seed',
            'event_parent_weighting':'For a parent with n input-query-bearing views among2, a selected query view receives weight2/n. Divide by all drawn parent slots having any query view, including a slot whose selected view has no query. This is an unbiased equal-parent event objective; microbatch accumulation preserves these full-batch weights.',
            'checkpoint':'atomic complete model/optimizers/schedulers/all RNG/sampler state; latest and interrupted; progress each complete update; safe signals save at boundary; exceptional midupdate rollback and journal-tail archive',
            'elapsed_time':'append-only per-process session receipts; replayed wall time included'},
        'staged':{'seed':SEEDS[0],'smoke_steps':1000,'pilot_steps':5000,'allowed_full_steps':[10000,20000],
            'schedule_before_full':'constant registered base LR',
            'evaluate_steps':[0,1000,5000],'evaluate_splits':['calibration','validation'],
            'LaMa':'all four independently initialized members at each stage, K4 ensemble evaluation',
            'automated_gate':{'finite_updates':True,'constraint_violations_allowed':0,
                'unknown_loss_relative_reduction_first100_to_last100_min':.02,
                'calibration_hidden_map_brier_improvement_vs_untrained_min':.0001,
                'validation_hidden_map_brier_improvement_vs_untrained_min':.0001,
                'calibration_event_brier_improvement_vs_untrained_min':.0001,
                'validation_event_brier_improvement_vs_untrained_min':.0001},
            'visual_gate':'root reviews frozen case images at0/1000/5000 for empty/full/fragmented collapse; publish observed failure and reference prevalence. No full training without recorded review.',
            'full_budget_rule':'Only after passed pilot, freeze10000 updates if cal map and event relative gains from1000to5000 are each<2%; otherwise20000. Cosine after5000 to0.1*base LR. No automatic300000-step run.',
            'full_checkpoint_steps':'every2500 updates after5000; shared rule and shared step for all LaMa members',
            'sufficiency':'At least10000 updates; final three calibration checkpoints show <2% relative improvement in both map and event score, stable finite loss and visual outputs. Reaching a cap with continuing gains is insufficient, not success.',
            'formal_repeats':'All three outer seeds and all members must finish the same method-specific sealed full schedule and common evaluation. Pilot snapshots never enter formal superiority table.'},
        'evaluation':{'version':'formal-evaluator-v1', 'files':evaluator_files,
            'sha256':hashlib.sha256(json.dumps(evaluator_files,sort_keys=True).encode()).hexdigest(),
            'map_metrics':['hidden-map sample-vote Brier','hidden-map sample-vote NLL','observed evidence violation count','valid-support violation count'],
            'native_continuous_outputs':'separate completion-score diagnostics, never silently substituted for common binary-world marginals',
            'event_metrics':['Event Brier','Event NLL','Event ECE','false-safe @0.8','coverage @0.8'],
            'strata':['pooled','equal-parent scene-weighted','reachable','unreachable','radius0','radius10','radius20','source'],
            'ece_bins':10,'nll_epsilon':1e-6,
            'platt_regularization':.01,
            'source_macro':'Arithmetic mean of source-specific equal-parent metrics; undefined source false-safe values excluded with contributing-source count, never imputed0. This is secondary to the equal-parent primary result.',
            'calibration':'raw primary; one fixed regularized monotone Platt fit on calibration only, reported separately as diagnosis; no validation/test tuning',
            'checkpoint_selection':'lowest calibration raw Event Brier at actual commonK4; tie1e-5 uses hidden-map vote NLL then earlierstep; LaMa shared ensemble step',
            'timing':'synchronized per-case generation and complete update; include actual network hooks/NFE/peak memory; shared-GPU measurements not isolated ranking'},
        'figures':{'cases':str(cases.relative_to(ROOT)),'sha256':sha256(cases),
            'geometry_verification':str((DATA/'figure_geometry_verification_v1.json').relative_to(ROOT)),
            'geometry_verification_sha256':sha256(DATA/'figure_geometry_verification_v1.json'),
            'selection':'input-hash general examples plus reference-geometry diagnostic categories frozen before any model output; no best-of-K or favorable-result selection',
            'annotations':['scene id','query id','radius cells','method','seed','validation-only'],
            'required':['same observed/reference/ConPath/LaMa/FM','all four worlds','hidden-map Brier','event reliability','event Brier/NLL/ECE','false-safe vs coverage','narrow bottleneck','unreachable','multiple alternative paths','footprint radius comparison'],
            'language':'Chinese wherever practical; never ours wins or SOTA'},
        'test_lock':{'new_physical_test_images_opened':0,'new_final_test_images_opened':0,'location_6_opened':False,
            'final_test_locked':True,'location_6_locked':True,
            'only_physical_archive_split':'train','historical_access_disclosed':True,
            'historical_erratum':'site/data/flatlands_read_scope_erratum.json',
            'final_test':'locked throughout training, calibration, validation, rendering and reporting'},
        'data_limitations':data_protocol.get('eligibility_note', 'Initial duplicate-input quarantine is documented in eligible data protocol/audit; removed open-space templates constrain scope.'),
        'old_results':'Do not rerun old P0/K128/PaSCo/S4C/profiles/batch64/UnScenes audit; old split scores are history-only. New shared data requires fresh registered controls.',
        'resume_verification':{'path':str(resume_file.relative_to(ROOT)),'sha256':sha256(resume_file)},
        'environment':{'python':sys.version,'python_executable':sys.executable,'pytorch':torch.__version__,
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'platform':platform.platform(),
            'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)},
        'source_hashes':source_hashes,
    }
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'figures').mkdir()
    atomic_json(PROTOCOL/'protocol.json',protocol)
    atomic_json(OUT/'ownership.json',{'protocol_sha256':sha256(PROTOCOL/'protocol.json'),
        'created_utc':protocol['created_utc'],'status':'protocol_frozen_stages_pending'})
    print(json.dumps({'protocol':str(PROTOCOL/'protocol.json'),'sha256':sha256(PROTOCOL/'protocol.json'),
                      'partition':protocol['partition'],'git_commit':protocol['git_commit']}))


if __name__ == '__main__':main()
