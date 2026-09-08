#!/usr/bin/env python3
"""Render the two previously fixed validation cases from completed seed-20260831 models.

Only checkpoint inference is performed. Case selection is inherited from the
published page before these ablations finished; no new favorable cases are chosen.
Outputs stay in results until the complete six-run statistical report is published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.flatlands_eval import join_flatlands_predictions
from pathrel.labels import clearance_radius_map
from scripts.build_ablation_summary_zh import LABELS, read, sha, write
from scripts import build_site_visuals as visual
from scripts.render_flatlands_k128_advantage import _event, _load_checkpoint_model


def posterior_visualization(model, sample, *, disable_global, seed, device):
    observation = torch.from_numpy(sample.input_bev[None]).to(device, dtype=torch.float32)
    support = torch.from_numpy(sample.epistemic_mask[None]).to(device, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(seed)
    total = np.zeros(sample.epistemic_mask.shape, dtype=np.float64)
    first_world = None
    with torch.inference_mode():
        for _ in range(16):
            posterior = model(observation, valid_support_mask=support, num_samples=8,
                              hard_samples=True, disable_global_factors=disable_global,
                              generator=generator).posterior
            mean = posterior.conditional_class_probs[0, :, 0]
            worlds = posterior.safe_samples()[0]
            if torch.any(worlds[:, ~support[0]] > .5):
                raise ValueError('invalid support became free in a rendered world')
            if first_world is None:
                first_world = worlds[0].cpu().numpy() > .5
            total += mean.sum(0).cpu().numpy()
    result = (total / 128).astype(np.float32)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('invalid mean probability')
    # Fixed logits ±10 leave softmax mass about 2e-9 on a blocked class.
    # Hard worlds were checked above; project that numerical residue to the
    # supplied support domain in the displayed continuous probability map.
    if np.any(result[~sample.epistemic_mask] > 1e-7):
        raise ValueError('unexpected probability outside valid support')
    result[~sample.epistemic_mask] = 0
    return result, first_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/ablation_publication_20260908/cases_v2')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / 'report.json').exists():
        parser.error('已完成的图片记录不覆盖；如需重新渲染，请使用新的 --output-dir。')
    output.mkdir(parents=True, exist_ok=True)
    visual.ASSETS = output
    torch.set_num_threads(2)
    torch.manual_seed(20260831)
    device = torch.device(args.device)
    source_path = ROOT / 'site/data/site_visuals_zh.json'
    fixed = read(source_path)['examples']
    if [(r['global_id'], r['candidate_index'], r['radius_cells']) for r in fixed] != [('obs_266762', 23, 10), ('obs_249512', 5, 20)]:
        raise ValueError('the pre-existing two cases must remain fixed')
    dataset = visual.flat_dataset('validation')
    indices = {r.global_id: i for i, r in enumerate(dataset.observations)}
    samples = [dataset[indices[r['global_id']]] for r in fixed]
    cases = []
    sources = {str(source_path.relative_to(ROOT)): sha(source_path),
               'scripts/render_ablation_cases_zh.py': sha(Path(__file__)),
               'scripts/build_site_visuals.py': sha(ROOT / 'scripts/build_site_visuals.py'),
               'scripts/render_flatlands_k128_advantage.py': sha(ROOT / 'scripts/render_flatlands_k128_advantage.py')}
    arrays = {}
    for case, sample in zip(fixed, samples):
        points = [tuple(case['start']), tuple(case['goal'])]
        if _event(sample.target_free, *points, case['radius_cells']) != bool(case['target']):
            raise ValueError('fixed case target/geometry mismatch')
        cases.append({k: case[k] for k in ('id', 'global_id', 'source', 'scene', 'candidate_index', 'radius_cells',
                                          'target', 'start', 'goal', 'seed', 'visual_sampling_seed', 'selection_rule')})
        cases[-1].update(panels={}, footprint_panels={}, first_world_event={}, event_probability={}, observed=case['panels']['observed'], reference=case['panels']['reference'])
        for key in ('observed', 'reference'):
            path = ROOT / 'site' / case['panels'][key]
            sources[str(path.relative_to(ROOT))] = sha(path)
    checkpoints = {}
    for variant in LABELS:
        run_dir = (ROOT / 'results/p1_flatlands_conpath_k128_support_clamped_v1/seed20260831_conpath'
                   if variant == 'conpath' else ROOT / f'results/paper_clean_ablation_matrix_v1/seed20260831_{variant}')
        checkpoint = run_dir / 'best.pt'
        run = read(run_dir / 'run.json')
        if variant != 'conpath':
            audit = read(run_dir / 'clean_ablation_audit.json')
            if not audit['passed'] or audit['checkpoint']['sha256'] != sha(checkpoint):
                raise ValueError('completed seed audit/checkpoint mismatch')
        if run['test_evaluated'] is not False:
            raise ValueError('validation-only checkpoint required')
        model, config = _load_checkpoint_model(checkpoint, 'correlated', device)
        disabled = bool(config['disable_global_factors'])
        if disabled != (variant == 'no_global'):
            raise ValueError('global-factor setting mismatch')
        if float(config['reachability_weight']) != (0 if variant == 'no_event' else 2):
            raise ValueError('event-loss setting mismatch')
        checkpoints[variant] = {'path': str(checkpoint.relative_to(ROOT)), 'sha256': sha(checkpoint),
                                'decoder_variant': 'correlated', 'local_kernel_size': 5,
                                'disable_global_factors': disabled, 'event_loss_weight': config['reachability_weight']}
        prediction = run_dir / 'predictions_validation.csv'
        records, _ = join_flatlands_predictions(prediction, ROOT / config['selection'], ROOT / config['queries'], split='validation')
        by_key = {r.key: r for r in records}
        for path in (checkpoint, prediction, run_dir / 'run.json'):
            sources[str(path.relative_to(ROOT))] = sha(path)
        for case, sample in zip(cases, samples):
            key = (case['global_id'], case['candidate_index'], case['radius_cells'])
            row = by_key[key]
            if row.target != case['target']:
                raise ValueError('prediction key/target mismatch')
            probability = row.probability
            case['event_probability'][variant] = probability
            mean, world = posterior_visualization(model, sample, disable_global=disabled, seed=case['visual_sampling_seed'], device=device)
            arrays[case['id'] + '_' + variant] = mean
            arrays[case['id'] + '_' + variant + '_first_world'] = world
            center_free = (clearance_radius_map(world) >= case['radius_cells']) & sample.epistemic_mask
            connected = _event(world, tuple(case['start']), tuple(case['goal']), case['radius_cells'])
            case['first_world_event'][variant] = connected
            name = 'ablation-case-' + case['id'] + '-' + variant
            case['panels'][variant] = visual.map_svg(name, visual.probability(mean, sample.epistemic_mask), LABELS[variant],
                subtitle=f'该查询有路概率 {probability:.1%}（原验证）',
                points=[tuple(case['start']), tuple(case['goal'])], ramp=True)
            case['footprint_panels'][variant] = visual.map_svg(name + '-footprint',
                visual.categorical(sample.input_bev, sample.epistemic_mask, center_free), LABELS[variant],
                subtitle=f"第1次采样 · 收缩{case['radius_cells']}格 · {'有路' if connected else '无路'}",
                points=[tuple(case['start']), tuple(case['goal'])])
            print(json.dumps({'case': case['global_id'], 'model': variant, 'validation_probability': probability}, ensure_ascii=False), flush=True)
        del model
    dataset.close()
    np.savez_compressed(output / 'posterior_means.npz', **arrays)
    assets = [{'path': 'assets/zh/' + p.name, 'sha256': sha(p)} for p in sorted(output.glob('*.svg'))]
    report = {'passed': True, 'validation_only': True, 'test_evaluated': False, 'training_started': False,
              'seed': 20260831, 'samples': 128, 'sample_chunk': 8, 'device': args.device,
              'labels': LABELS, 'checkpoints': checkpoints, 'sources': sources, 'assets': assets, 'cases': cases,
              'posterior_arrays': {'path': str((output / 'posterior_means.npz').relative_to(ROOT)), 'sha256': sha(output / 'posterior_means.npz')},
              'selection': 'Reuse both pre-existing, label-selected website cases without ranking new ablation outcomes. One optimization seed; explanatory, not aggregate evidence.',
              'probability_note': 'Map colors are cell probabilities from a fresh fixed visualization RNG shared by all three variants, K=128/chunks=8. Query probabilities are copied from the original exact validation CSVs; the streams differ. No inferred route is drawn.',
              'footprint_note': 'Display the first sampled world only, without outcome selection, after exact disk-clearance erosion at the fixed query radius. Green cells admit the robot center; dark cells do not. First-world connectivity is not the K=128 probability.'}
    write(output / 'report.json', report)
    print(json.dumps({'passed': True, 'cases': len(cases), 'assets': len(assets), 'report': str(output / 'report.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
