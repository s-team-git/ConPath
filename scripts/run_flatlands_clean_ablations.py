#!/usr/bin/env python3
"""Run the six frozen clean-support ablations and build audited paper candidates.

The original trainer is unchanged. The full model is reused as the reference;
each ablation changes one flag only. Results are validation-only and remain in
ignored result directories until reviewed for manuscript/site publication.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pathrel.ablation_protocol import VARIANTS, ablation_config, checkpoint_action, validate_config
from pathrel.flatlands_query import sha256_path
from scripts.audit_flatlands_conpath_k128 import (
    SEEDS, SELECTION_SHA256, QUERIES_SHA256, EXPECTED_PROTOCOL_VERSION,
    EXPECTED_SUPPORT_POLICY, _audit_checkpoint, _audit_prediction, _overall,
)
from scripts.evaluate_flatlands_support_clamped import _atomic_json
from scripts import train_flatlands_conpath as trainer

REFERENCE = Path("results/p1_flatlands_conpath_k128_support_clamped_v1")
DEFAULT_ROOT = Path("results/paper_clean_ablation_matrix_v1")
SOURCES = {"trainer": "scripts/train_flatlands_conpath.py", "model": "src/pathrel/model.py",
           "flatlands_data": "src/pathrel/flatlands_data.py"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def command(config):
    args = [sys.executable, "scripts/train_flatlands_conpath.py"]
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif value is not None:
            args.extend([flag, str(value)])
    return args


def prepare(output_root):
    sources = {key: sha256_path(ROOT / path) for key, path in SOURCES.items()}
    runs = []
    references = []
    reference_audit = read(REFERENCE / 'audit.json')
    if not reference_audit['passed']:
        raise ValueError('the full-model clean audit has not passed')
    for seed in SEEDS:
        path = REFERENCE / f"seed{seed}_conpath/run.json"
        reference = read(path)
        if (reference["forward"]["implementation_sha256"] != sources
                or reference["protocol_version"] != EXPECTED_PROTOCOL_VERSION
                or reference["test_evaluated"] is not False
                or reference["forward"]["valid_support_policy"] != EXPECTED_SUPPORT_POLICY):
            raise ValueError(f"reference/source support contract mismatch: {path}")
        for key, expected in (("selection", SELECTION_SHA256), ("queries", QUERIES_SHA256)):
            if sha256_path(Path(reference["config"][key])) != expected:
                raise ValueError(f"frozen {key} hash mismatch")
        if Path(reference['config']['archive']).stat().st_size != 2_054_773_316:
            raise ValueError('archive byte count mismatch')
        checkpoint, failures = _audit_checkpoint(ROOT / path.parent / "best.pt", decoder_variant="correlated", seed=seed)
        if failures:
            raise ValueError(f"reference checkpoint audit failed: {failures}")
        frozen_checkpoint = next(record['checkpoint'] for record in reference_audit['seeds'] if record['seed'] == seed)
        if checkpoint['sha256'] != frozen_checkpoint['sha256']:
            raise ValueError(f'reference checkpoint changed since its clean audit: {path}')
        if sha256_path(Path(reference['prediction']['path'])) != reference['prediction']['sha256']:
            raise ValueError(f'reference predictions changed: {path}')
        references.append({"seed": seed, "report": str(path), "sha256": sha256_path(path), "checkpoint": checkpoint})
        for variant in VARIANTS:
            run_dir = output_root / f"seed{seed}_{variant}"
            config = ablation_config(reference["config"], variant, str(run_dir))
            runs.append({"seed": seed, "variant": variant, "output_dir": str(run_dir),
                         "config": config, "command": command(config)})
    # Three seeds per wave. Never run more than three approximately 22-GB jobs.
    runs.sort(key=lambda run: (list(VARIANTS).index(run["variant"]), run["seed"]))
    contract = {"schema_version": 1, "kind": "flatlands_clean_causal_ablation_matrix",
                "protocol_version": EXPECTED_PROTOCOL_VERSION, "test_evaluated": False,
                "paper_result": False, "validation_only": True, "references": references,
                "training_source_sha256": sources, "runs": runs}
    path = output_root / "matrix.json"
    if path.exists():
        existing = read(path)
        if existing["contract"] != contract:
            raise ValueError("existing matrix differs from the current frozen contract")
        return existing
    matrix = {"created_utc": now(), "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              "launcher_sha256": sha256_path(Path(__file__)), "python": sys.version, "contract": contract}
    _atomic_json(path, matrix)
    return matrix


def audit_run(entry, sources, *, write_report=True):
    path = ROOT / entry["output_dir"]
    run = read(path / "run.json")
    validate_config(run['config'], entry['config'])
    if (run['protocol_version'] != EXPECTED_PROTOCOL_VERSION or not run['validation_result']
            or run['test_evaluated'] is not False or run['paper_result'] is not False):
        raise ValueError(f"invalid result scope: {path}")
    forward = run['forward']
    if (forward['implementation_sha256'] != sources or not forward['invalid_support_clamped']
            or forward['valid_support_policy'] != EXPECTED_SUPPORT_POLICY
            or not forward['validation_exact_forward'] or forward['local_kernel_size'] != 5
            or forward['effective_disable_global_factors'] != entry['config']['disable_global_factors']):
        raise ValueError(f"ablation forward contract mismatch: {path}")
    for name in ('best.pt', 'latest.pt'):
        state = torch.load(path / name, map_location='cpu', weights_only=False)
        validate_config(state['config'], entry['config'])
        if state['protocol_version'] != EXPECTED_PROTOCOL_VERSION:
            raise ValueError(f"checkpoint protocol mismatch: {path / name}")
        if name == 'latest.pt' and checkpoint_action(state, entry['config']) != 'finalize':
            raise ValueError(f"training stopped before the frozen stopping rule: {path}")
    checkpoint, failures = _audit_checkpoint(path / 'best.pt', decoder_variant='correlated', seed=entry['seed'])
    if checkpoint['best_epoch'] != run['selection']['best_epoch']:
        failures.append('best epoch differs from report')
    prediction = path / 'predictions_validation.csv'
    count, groups, prediction_failures = _audit_prediction(prediction)
    failures.extend(prediction_failures)
    if count != 4224 or groups != 1408 or sha256_path(prediction) != run['prediction']['sha256']:
        failures.append('prediction count/key/hash mismatch')
    if failures:
        raise ValueError(f"ablation audit failed: {path}: {failures}")
    replay = trainer.evaluate_flatlands_prediction_file(
        prediction, Path(entry['config']['selection']), Path(entry['config']['queries']),
        method=entry['variant'], split='validation', bootstrap_samples=2000, seed=entry['seed'])
    recorded = _overall(read(path / 'evaluation_validation/report.json'))
    actual = _overall(replay)
    for key in ('brier', 'nll', 'ece', 'false_safe_rate@0.8', 'high_confidence_safe_coverage@0.8'):
        if abs(recorded[key] - actual[key]) > 1e-12:
            raise ValueError(f"metric replay mismatch: {path}/{key}")
    result = {'passed': True, 'seed': entry['seed'], 'variant': entry['variant'], 'test_evaluated': False,
              'checkpoint': checkpoint, 'prediction_sha256': sha256_path(prediction),
              'report_sha256': sha256_path(path / 'run.json'), 'metrics': actual}
    if write_report:
        _atomic_json(path / 'clean_ablation_audit.json', result)
    return result


def finalize_stopped(entry):
    """Replay final evaluation after interruption without taking an extra epoch."""
    config = entry['config']; path = Path(entry['output_dir']); started = time.monotonic()
    latest = torch.load(path / 'latest.pt', map_location='cpu', weights_only=False)
    if checkpoint_action(latest, config) != 'finalize':
        raise ValueError('checkpoint has not reached the frozen training stop')
    selected = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
    validate_config(selected['config'], config)
    device = torch.device('cuda')
    _, generator = trainer._seed_everything(config['seed'], device)
    generator.set_state(latest['sample_generator_state'].cpu())
    samples, radii = trainer._cache_split(Path(config['archive']), Path(config['selection']),
                                        Path(config['queries']), 'validation', None)
    loader = trainer.DataLoader(samples, batch_size=1, shuffle=False,
                               collate_fn=trainer.collate_flatlands_replay, num_workers=0)
    model = trainer.PathRelNet(input_channels=3, feature_channels=16, latent_dim=4, local_kernel_size=5).to(device)
    model.load_state_dict(selected['model'], strict=True)
    rows = trainer._validation_prediction_rows(model, loader, device, radii, 128, 256, generator,
             exact_forward=True, disable_global_factors=config['disable_global_factors'], sample_chunk=8)
    prediction = path / 'predictions_validation.csv'
    count = trainer.write_prediction_manifest(prediction, rows)
    evaluation = trainer.evaluate_flatlands_prediction_file(prediction, Path(config['selection']),
        Path(config['queries']), method=entry['variant'], split='validation', bootstrap_samples=2000, seed=config['seed'])
    trainer.write_evaluation_report(path / 'evaluation_validation', evaluation)
    reference = read(REFERENCE / f"seed{entry['seed']}_conpath/run.json")
    forward = {**reference['forward'], 'effective_disable_global_factors': config['disable_global_factors']}
    report = {'schema_version': 1, 'kind': 'flatlands_conpath_training', 'paper_result': False,
              'validation_result': True, 'protocol_version': EXPECTED_PROTOCOL_VERSION, 'test_evaluated': False,
              'git': trainer._git_state(), 'config': {**config, 'resume': True}, 'data': reference['data'],
              'selection': {**reference['selection'], 'best_epoch': selected['best_epoch'],
                            'best_validation_scene_weighted_brier': selected['best_score'],
                            'epochs_completed': len(latest['history'])},
              'prediction': {'path': str(prediction), 'rows': count, 'sha256': sha256_path(prediction)},
              'forward': forward, 'evaluation': evaluation, 'history': latest['history'],
              'runtime': {'scope': 'resumed final evaluation only; training times are in history',
                          'wall_seconds': time.monotonic() - started},
              'finalization_resume': {'launcher_sha256': sha256_path(Path(__file__)), 'extra_training_epochs': 0},
              'claim_boundary': 'Clean-support ablation on frozen validation queries; test remains locked.'}
    _atomic_json(path / 'run.json', report)


def active_output_dirs():
    output_dirs = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().decode().split('\0')
            if 'scripts/train_flatlands_conpath.py' in args and '--output-dir' in args:
                output_dirs.add(str(Path(args[args.index('--output-dir') + 1]).resolve()))
            if 'scripts/run_flatlands_clean_ablations.py' in args and '--finalize-dir' in args:
                output_dirs.add(str(Path(args[args.index('--finalize-dir') + 1]).resolve()))
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            pass
    return output_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--max-parallel', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--finalize-dir', type=str)
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.output_root.resolve()
    root.relative_to(ROOT / 'results')
    root = root.relative_to(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; refusing a CPU experiment')
    matrix = prepare(root)
    entries = matrix['contract']['runs']; sources = matrix['contract']['training_source_sha256']
    if args.finalize_dir:
        finalize_stopped(next(entry for entry in entries if entry['output_dir'] == args.finalize_dir))
        return
    if args.prepare_only:
        print(json.dumps({'matrix': str(root / 'matrix.json'), 'runs': len(entries), 'gpu': torch.cuda.get_device_name(0)}))
        return
    lock = (root / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if torch.cuda.mem_get_info()[0] < args.max_parallel * 24 * 1024**3:
        raise SystemExit('insufficient free GPU memory for the requested concurrency')
    if any(str((ROOT / e['output_dir']).resolve()) in active_output_dirs() for e in entries):
        raise SystemExit('an orphan training worker is still active; refusing duplicate workers')
    log_root = root / 'logs'; log_root.mkdir(exist_ok=True)
    statuses = {e['output_dir']: {'seed': e['seed'], 'variant': e['variant'], 'status': 'queued'} for e in entries}
    active = {}; queue = list(entries)
    def write_status(status, **extra):
        _atomic_json(root / 'progress.json', {'status': status, 'updated_utc': now(), 'supervisor_pid': os.getpid(),
                     'test_evaluated': False, 'runs': list(statuses.values()), **extra})
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while queue or active:
            while queue and len(active) < args.max_parallel:
                entry = queue.pop(0); path = Path(entry['output_dir']); status = statuses[str(path)]
                if (path / 'run.json').exists():
                    audit_run(entry, sources); status['status'] = 'complete'; continue
                config = dict(entry['config']); invocation = command(config)
                if (path / 'latest.pt').exists():
                    state = torch.load(path / 'latest.pt', map_location='cpu', weights_only=False)
                    action = checkpoint_action(state, config)
                    if action == 'finalize':
                        invocation = [sys.executable, str(Path(__file__).relative_to(ROOT)), '--output-root', str(root), '--finalize-dir', str(path)]
                    else:
                        invocation.append('--resume')
                elif path.exists() and any(path.iterdir()):
                    raise ValueError(f'nonempty run has no recoverable checkpoint: {path}')
                log = (log_root / f"seed{entry['seed']}_{entry['variant']}.log").open('a')
                process = subprocess.Popen(invocation, cwd=ROOT, env={**os.environ, 'PYTHONPATH': 'src', 'PYTHONUNBUFFERED': '1'}, stdout=log, stderr=subprocess.STDOUT)
                active[str(path)] = (process, log, entry)
                status.update(status='running', pid=process.pid, started_utc=now(), command=invocation)
                print(json.dumps(status), flush=True)
            for key, (process, log, entry) in list(active.items()):
                if process.poll() is not None:
                    log.close(); del active[key]
                    if process.returncode:
                        statuses[key].update(status='failed', exit_code=process.returncode)
                        raise RuntimeError(f'worker failed: {key}; inspect its log')
                    result = audit_run(entry, sources)
                    statuses[key].update(status='complete', completed_utc=now(), brier=result['metrics']['brier'])
                    print(json.dumps(statuses[key]), flush=True)
                else:
                    progress = Path(key) / 'progress.jsonl'
                    if progress.exists():
                        rows = progress.read_text().splitlines()
                        if rows:
                            try:
                                statuses[key]['last_epoch'] = json.loads(rows[-1])
                            except json.JSONDecodeError:
                                pass
            write_status('training')
            if active:
                time.sleep(15)
        write_status('analyzing')
        subprocess.run([sys.executable, 'scripts/analyze_flatlands_clean_ablations.py', '--matrix-root', str(root)], cwd=ROOT, check=True)
        write_status('complete', evidence=str(root / 'analysis/report.json'), publication='candidate artifacts only; manuscript/site review remains')
    except BaseException as error:
        for process, log, entry in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process, log, entry in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
            log.close()
            statuses[entry['output_dir']]['status'] = 'interrupted'
        write_status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
