#!/usr/bin/env python3
"""Isolate spatial dependence by shuffling clean posterior samples at fixed cell counts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pathrel.flatlands_data import FlatLandsReplayDataset
from pathrel.flatlands_eval import load_prediction_manifest, write_prediction_manifest, join_flatlands_predictions, _metric_summary
from pathrel.flatlands_query import sha256_path
from pathrel.posterior_audits import shuffle_worlds_by_cell
from scripts.evaluate_flatlands_clean_controls import SELECTION, QUERIES, SEEDS, rows_for
from scripts.evaluate_flatlands_support_clamped import _load_model, _accelerated_events, _atomic_json, _verify_accelerator
from scripts.compare_flatlands_k128_paired import _validate_source_run


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for exact source RNG replay")
    torch.set_num_threads(4)
    device=torch.device('cuda')
    accelerator=_verify_accelerator()
    dataset=FlatLandsReplayDataset(Path('data/raw/flatlands/FlatLands_final_dataset.zip'), SELECTION, QUERIES, split='validation', verify_frozen=True, verify_query_geometry=True)
    try:
        samples=[dataset[i] for i in range(len(dataset))]
    finally:
        dataset.close()
    for seed in SEEDS:
        source=Path(f'results/p1_flatlands_conpath_k128_support_clamped_v1/seed{seed}_conpath')
        output=Path(f'results/paper_clean_marginal_shuffle_v1/seed{seed}')
        if (output/'run.json').exists():
            raise ValueError(f'output already completed: {output}')
        source_run=_validate_source_run(ROOT/source/'run.json','correlated','valid-support-clean-training')
        model, config, _=_load_model(source/'best.pt','correlated',device)
        latest=torch.load(source/'latest.pt',map_location='cpu',weights_only=False)
        generator=torch.Generator(device=device);generator.set_state(latest['sample_generator_state'].cpu());del latest
        shuffle_seed=seed+20260906
        shuffle_generator=np.random.default_rng(shuffle_seed)
        replay={};rows=[];cells_checked=0;started=time.monotonic()
        with torch.inference_mode():
            for index,sample in enumerate(samples):
                if not sample.retained_queries:continue
                starts=np.array([(q.start_row,q.start_col) for q in sample.retained_queries])
                goals=np.array([(q.goal_row,q.goal_col) for q in sample.retained_queries])
                observation=torch.from_numpy(sample.input_bev[None]).to(device,dtype=torch.float32)
                support=torch.from_numpy(sample.epistemic_mask[None]).to(device,dtype=torch.bool)
                worlds=np.concatenate([model(observation,valid_support_mask=support,num_samples=8,generator=generator).posterior.safe_samples()[0].cpu().numpy()>0.5 for _ in range(16)])
                shuffled=shuffle_worlds_by_cell(worlds,shuffle_generator)
                if not np.array_equal(worlds.sum(axis=0),shuffled.sum(axis=0)) or np.any(shuffled & ~sample.epistemic_mask[None]):
                    raise ValueError('fixed-marginal or valid-support contract failed')
                cells_checked+=int(sample.epistemic_mask.size)
                original_events=_accelerated_events(worlds,starts,goals,sample.radii_cells).mean(axis=0)
                shuffled_events=_accelerated_events(shuffled,starts,goals,sample.radii_cells).mean(axis=0)
                for row in rows_for(sample,original_events):
                    replay[(row['global_id'],row['candidate_index'],row['radius_cells'])]=row['probability']
                rows.extend(rows_for(sample,shuffled_events))
                if (index+1)%40==0 or index+1==len(samples):
                    print(json.dumps({'seed':seed,'scenes':index+1,'total':len(samples),'seconds':round(time.monotonic()-started,2)}),flush=True)
        canonical=load_prediction_manifest(source/'predictions_validation.csv')
        if canonical != replay:
            raise ValueError('source K=128 probabilities did not replay exactly')
        output.mkdir(parents=True,exist_ok=True)
        prediction=output/'predictions_validation.csv'
        write_prediction_manifest(prediction,rows)
        records,radii=join_flatlands_predictions(prediction,SELECTION,QUERIES,split='validation')
        metrics=_metric_summary(records,weighting='scene',bins=10)
        report={'kind':'flatlands_fixed_empirical_marginal_shuffle','validation_only':True,'paper_result':False,'test_evaluated':False,
                'seed':seed,'shuffle_seed':shuffle_seed,'samples':128,'source_run':source_run,
                'checkpoint':{'path':str(source/'best.pt'),'sha256':sha256_path(source/'best.pt')},
                'prediction':{'path':str(prediction),'sha256':sha256_path(prediction),'rows':len(rows)},
                'checks':{'canonical_k128_replay_exact':True,'empirical_cell_counts_preserved':True,'invalid_support_blocked':True,'cells_checked':cells_checked},
                'metrics':metrics,'accelerator_audit':accelerator,'seconds':time.monotonic()-started,
                'script_sha256':sha256_path(Path(__file__)),'helper_sha256':sha256_path(ROOT/'src/pathrel/posterior_audits.py'),
                'claim_boundary':'Evaluation-only intervention on fixed trained worlds. Every empirical per-cell marginal is identical; only sample alignment across cells changes. This supports a dependence mechanism on the frozen validation set, not a trained no-event/no-global ablation or final test claim.'}
        _atomic_json(output/'run.json',report)
        print(json.dumps({'complete':str(output),'brier':metrics['brier']}),flush=True)


if __name__=='__main__':main()
