#!/usr/bin/env python3
"""Evaluate completion checkpoints at several posterior sample counts.

This is an evaluation-only companion to ``train_flatlands_completion.py``.  It reuses one
selected checkpoint and the frozen validation query manifest, so K-convergence cannot change
training, checkpoint selection, or the label-free prediction contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.flatlands_baselines import MarginalCompletionBaseline  # noqa: E402
from pathrel.flatlands_data import FlatLandsReplayDataset, collate_flatlands_replay  # noqa: E402
from pathrel.flatlands_eval import evaluate_flatlands_prediction_file, write_evaluation_report, write_prediction_manifest  # noqa: E402
from pathrel.flatlands_query import sha256_path  # noqa: E402
from pathrel.flatlands_sampling import CompletionEventTask, completion_event_probabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"))
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--event-workers", type=int, default=8)
    parser.add_argument("--k", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser.parse_args()


def _to_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output = dict(batch)
    for key in ("observation", "target_free", "loss_mask"):
        output[key] = torch.from_numpy(batch[key]).to(device=device, non_blocking=True)
    return output


@torch.inference_mode()
def _probability_maps(model: MarginalCompletionBaseline, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, np.ndarray] = {}
    for numpy_batch in loader:
        probability = model.free_probability(_to_tensor_batch(numpy_batch, device)["observation"])
        for index, global_id in enumerate(numpy_batch["global_ids"]):
            output[global_id] = probability[index].cpu().numpy().astype(np.float32, copy=True)
    return output


def _tasks(samples: list[object], maps: dict[str, np.ndarray], radii: tuple[int, ...], k: int, seed: int) -> list[CompletionEventTask]:
    tasks: list[CompletionEventTask] = []
    for sample in samples:
        retained = sample.retained_queries
        if not retained:
            continue
        tasks.append(CompletionEventTask(
            global_id=sample.observation.global_id,
            free_probability=maps[sample.observation.global_id],
            observed_free=sample.observed_free,
            unknown=sample.unknown,
            starts=np.asarray([(query.start_row, query.start_col) for query in retained], dtype=np.int64),
            goals=np.asarray([(query.goal_row, query.goal_col) for query in retained], dtype=np.int64),
            candidate_indices=np.asarray([query.candidate_index for query in retained], dtype=np.int64),
            radii_cells=radii,
            posterior_samples=k,
            seed=seed,
        ))
    return tasks


def _rows(results: list[object], radii: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for query_index, candidate_index in enumerate(result.candidate_indices):
            for radius_index, radius in enumerate(radii):
                rows.append({
                    "global_id": result.global_id,
                    "candidate_index": int(candidate_index),
                    "radius_cells": radius,
                    "probability": float(result.independent[query_index, radius_index]),
                })
    return rows


def _git_state() -> dict[str, object]:
    import subprocess
    def run(*args: str) -> str:
        return subprocess.run(("git", *args), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--short").splitlines()}


def main() -> None:
    args = parse_args()
    if not args.k or any(value < 1 for value in args.k) or args.event_workers < 1:
        raise SystemExit("K and event-workers must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    feature_channels = int(config.get("feature_channels", 16))
    model = MarginalCompletionBaseline(feature_channels=feature_channels).to(device)
    model.load_state_dict(state["model"])

    dataset = FlatLandsReplayDataset(args.archive, args.selection, args.queries, split="validation", verify_frozen=True, verify_query_geometry=True)
    try:
        samples = [dataset[index] for index in range(len(dataset))]
        radii = dataset.radii_cells
    finally:
        dataset.close()
    loader = DataLoader(samples, batch_size=4, shuffle=False, collate_fn=collate_flatlands_replay, num_workers=0)
    maps = _probability_maps(model, loader, device)
    summaries: dict[str, object] = {}
    for k in sorted(set(args.k)):
        k_started = time.monotonic()
        tasks = _tasks(samples, maps, radii, k, args.seed)
        # Thread/process count affects throughput only: each observation has a stable RNG key.
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.event_workers, mp_context=mp.get_context("spawn")) as executor:
            results = list(executor.map(completion_event_probabilities, tasks, chunksize=1))
        prediction_path = args.output_dir / f"predictions_independent_k{k}_validation.csv"
        evaluation_dir = args.output_dir / f"evaluation_independent_k{k}_validation"
        count = write_prediction_manifest(prediction_path, _rows(results, radii))
        evaluation = evaluate_flatlands_prediction_file(
            prediction_path, args.selection, args.queries,
            method=f"independent_cell_completion_k{k}", split="validation",
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
        write_evaluation_report(evaluation_dir, evaluation)
        summaries[str(k)] = {
            "prediction": {"path": str(prediction_path), "rows": count, "sha256": sha256_path(prediction_path)},
            "evaluation": str(evaluation_dir / "report.json"),
            "scene_weighted": evaluation["metrics"][0]["scene_weighted"],
            "seconds": time.monotonic() - k_started,
        }
    report = {
        "schema_version": 1,
        "kind": "flatlands_completion_k_convergence",
        "paper_result": False,
        "validation_result": True,
        "test_evaluated": False,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256_path(args.checkpoint), "best_epoch": state.get("best_epoch")},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {"selection_sha256": sha256_path(args.selection), "queries_sha256": sha256_path(args.queries), "archive_bytes": args.archive.stat().st_size},
        "git": _git_state(),
        "radii_cells": list(radii),
        "results": summaries,
        "runtime": {"wall_seconds": time.monotonic() - started, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "claim_boundary": "Evaluation-only validation K convergence from a selected single-seed completion checkpoint; test remains locked and this is not a final paper result.",
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_dir": str(args.output_dir), "results": {key: value["scene_weighted"] for key, value in summaries.items()}, "paper_result": False}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
