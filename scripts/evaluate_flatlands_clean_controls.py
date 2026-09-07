#!/usr/bin/env python3
"""Replay clean checkpoints once for nested K budgets, mean-map and marginal controls.

The saved final training RNG state makes K=128 an exact check against the original
validation CSV. K=32/64 are nested prefixes of those same worlds. No split or query
is selected or changed using the new results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pathrel.flatlands_data import FlatLandsReplayDataset
from pathrel.flatlands_eval import write_prediction_manifest, load_prediction_manifest, join_flatlands_predictions, _metric_summary
from pathrel.flatlands_query import sha256_path
from scripts.evaluate_flatlands_support_clamped import _load_model, _accelerated_events, _verify_accelerator, _atomic_json
from scripts.compare_flatlands_k128_paired import _validate_source_run

SEEDS = (20260831, 20260901, 20260902)
BUDGETS = (32, 64, 128)
SELECTION = Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv")
QUERIES = Path("results/p1_flatlands_query_audit_bounded/queries.csv")


def rows_for(sample, probabilities):
    return [{"global_id": sample.observation.global_id, "candidate_index": q.candidate_index,
             "radius_cells": radius, "probability": float(probabilities[i, j])}
            for i, q in enumerate(sample.retained_queries)
            for j, radius in enumerate(sample.radii_cells)]


def run_one(samples, variant, seed, output, accelerator_audit, device):
    if (output / "run.json").is_file():
        report = json.loads((output / "run.json").read_text())
        if report.get("canonical_k128_max_drift") != 0 or not report.get("clean_support_training"):
            raise ValueError(f"invalid resumable result: {output}")
        for pred in report["predictions"].values():
            if sha256_path(Path(pred["path"])) != pred["sha256"]:
                raise ValueError("resumable prediction hash changed")
        print(json.dumps({"already_complete": str(output)}), flush=True)
        return
    run_root = Path("results") / ("p1_flatlands_conpath_k128_support_clamped_v1" if variant == "correlated" else "p1_flatlands_independent_k128_support_clamped_v1")
    source = run_root / f"seed{seed}_{'conpath' if variant == 'correlated' else 'independent'}"
    source_run = _validate_source_run(ROOT / source / "run.json", variant, "valid-support-clean-training")
    audit = json.loads((run_root / "audit.json").read_text())
    if audit.get("passed") is not True:
        raise ValueError("source clean checkpoint audit failed")
    checkpoint = source / "best.pt"
    model, config, forward = _load_model(checkpoint, variant, device)
    latest = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
    generator = torch.Generator(device=device)
    generator.set_state(latest["sample_generator_state"].cpu())
    del latest
    if config["batch_size"] != 1 or config["validation_samples"] != 128 or config["validation_sample_chunk"] != 8:
        raise ValueError("canonical replay expects batch=1, K=128 and chunks of eight")
    output.mkdir(parents=True, exist_ok=True)
    predictions = {str(k): [] for k in BUDGETS}
    predictions["mean_map"] = []
    map_rows = []
    started = time.monotonic()
    sampling_seconds = 0.0
    connectivity_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            queries = sample.retained_queries
            if not queries:
                continue
            starts = np.array([(q.start_row, q.start_col) for q in queries], dtype=np.int64)
            goals = np.array([(q.goal_row, q.goal_col) for q in queries], dtype=np.int64)
            observation = torch.from_numpy(sample.input_bev[None]).to(device, dtype=torch.float32)
            support = torch.from_numpy(sample.epistemic_mask[None]).to(device, dtype=torch.bool)
            event_sum = np.zeros((len(queries), len(sample.radii_cells)), dtype=np.float64)
            marginal_sum = np.zeros(sample.epistemic_mask.shape, dtype=np.float64)
            for count in range(8, 129, 8):
                torch.cuda.synchronize(device); tick = time.monotonic()
                posterior = model(observation, valid_support_mask=support, num_samples=8,
                                  disable_global_factors=variant == "independent", generator=generator).posterior
                worlds = posterior.safe_samples()[0].cpu().numpy() > 0.5
                marginal = posterior.posterior_marginal_probs[0, 0].cpu().numpy()
                torch.cuda.synchronize(device); sampling_seconds += time.monotonic() - tick
                if np.any(worlds & ~sample.epistemic_mask[None]):
                    raise ValueError("sampled a free cell outside valid support")
                tick = time.monotonic()
                event_sum += _accelerated_events(worlds, starts, goals, sample.radii_cells).sum(axis=0)
                connectivity_seconds += time.monotonic() - tick
                marginal_sum += marginal.astype(np.float64) * 8
                if count in BUDGETS:
                    predictions[str(count)].extend(rows_for(sample, event_sum / count))
            marginal = marginal_sum / 128
            mean_map = (marginal >= 0.5) & sample.epistemic_mask
            events = _accelerated_events(mean_map[None], starts, goals, sample.radii_cells)[0]
            predictions["mean_map"].extend(rows_for(sample, events))
            hidden = sample.loss_mask.astype(bool)
            if hidden.any():
                p = marginal[hidden]; y = sample.target_free[hidden].astype(float)
                clipped = np.clip(p, 1e-6, 1 - 1e-6)
                map_rows.append({"global_id": sample.observation.global_id, "hidden_cells": int(hidden.sum()),
                                 "brier": float(np.mean((p-y)**2)),
                                 "nll": float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped)))})
            if (index + 1) % 20 == 0 or index + 1 == len(samples):
                record = {"variant": variant, "seed": seed, "scenes_complete": index + 1, "total_scenes": len(samples), "seconds": round(time.monotonic()-started, 2)}
                _atomic_json(output / "progress.json", record)
                print(json.dumps(record), flush=True)
    canonical = load_prediction_manifest(source / "predictions_validation.csv")
    replay = {(r["global_id"], r["candidate_index"], r["radius_cells"]): r["probability"] for r in predictions["128"]}
    if canonical.keys() != replay.keys():
        raise ValueError("K=128 replay has different event keys")
    drift = max(abs(canonical[k]-replay[k]) for k in canonical)
    if drift != 0:
        raise ValueError(f"canonical K=128 replay drift: {drift}; do not promote these controls")
    metrics = {}; artifacts = {}
    for name, rows in predictions.items():
        path = output / f"predictions_{name}.csv"
        write_prediction_manifest(path, rows)
        records, _ = join_flatlands_predictions(path, SELECTION, QUERIES, split="validation")
        metrics[name] = _metric_summary(records, weighting="scene", bins=10)
        artifacts[name] = {"path": str(path), "sha256": sha256_path(path), "rows": len(rows)}
    report = {"kind": "flatlands_clean_checkpoint_controls", "validation_only": True, "test_evaluated": False,
              "paper_result": False, "clean_support_training": True, "canonical_k128_max_drift": drift,
              "seed": seed, "variant": variant, "source_run": source_run,
              "checkpoint": {"path": str(checkpoint), "sha256": sha256_path(checkpoint)},
              "latest_rng_checkpoint": {"path": str(source / 'latest.pt'), "sha256": sha256_path(source / 'latest.pt')},
              "software": {"script_sha256": sha256_path(Path(__file__)), "model_sha256": sha256_path(ROOT / 'src/pathrel/model.py'), "torch": torch.__version__},
              "protocol": {"budgets": BUDGETS, "sample_chunk": 8, "prefix_sampling": True, "mean_map_threshold": 0.5,
                           "selection_sha256": sha256_path(SELECTION), "queries_sha256": sha256_path(QUERIES)},
              "accelerator_audit": accelerator_audit, "predictions": artifacts, "event_metrics": metrics,
              "hidden_map": {"scene_count": len(map_rows), "brier": float(np.mean([r['brier'] for r in map_rows])), "nll": float(np.mean([r['nll'] for r in map_rows])), "scenes": map_rows},
              "runtime": {"total_seconds": time.monotonic()-started, "sampling_seconds": sampling_seconds,
                          "connectivity_seconds": connectivity_seconds, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)},
              "claim_boundary": "Nested-budget sensitivity and mean-map controls on fixed clean validation checkpoints. No retraining, threshold selection, test evaluation, or cross-domain claim."}
    _atomic_json(output / "run.json", report)
    print(json.dumps({"completed": str(output), "canonical_k128_max_drift": drift, "brier": {k: m['brier'] for k,m in metrics.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/paper_clean_checkpoint_controls_v1"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to replay the saved CUDA generator exactly")
    torch.set_num_threads(4)
    audit = _verify_accelerator()
    dataset = FlatLandsReplayDataset(Path("data/raw/flatlands/FlatLands_final_dataset.zip"), SELECTION, QUERIES, split="validation", verify_frozen=True, verify_query_geometry=True)
    try:
        samples = [dataset[i] for i in range(len(dataset))]
    finally:
        dataset.close()
    for variant in ("correlated", "independent"):
        for seed in SEEDS:
            run_one(samples, variant, seed, args.output_root / variant / f"seed{seed}", audit, torch.device("cuda"))


if __name__ == "__main__":
    main()
