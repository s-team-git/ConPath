#!/usr/bin/env python3
"""Evaluate a PaSCo-inspired multi-subnet uncertainty ensemble on FlatLands.

This is deliberately named an *ensemble control*: it borrows the recent paper's idea of
multiple independently trained subnetworks, but it is not a claim to reproduce PaSCo's 3-D
architecture. All members consume the frozen three-channel FlatLands input and are evaluated
with the same exact connectivity/event contract as ConPath.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
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
from pathrel.flatlands_sampling import CompletionEventTask, CompletionEventResult, completion_event_probabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"))
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--event-workers", type=int, default=8)
    parser.add_argument("--total-samples", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse existing prediction/evaluation files and only rebuild the integrity report")
    parser.add_argument("--publish-site", action="store_true", help="write a compact validation-only snapshot under site/data")
    return parser.parse_args()


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(("git", *args), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--short").splitlines()}


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _to_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output = dict(batch)
    output["observation"] = torch.from_numpy(batch["observation"]).to(device=device, non_blocking=True)
    return output


@torch.inference_mode()
def _maps(checkpoint: Path, samples: list[object], device: torch.device) -> tuple[dict[str, np.ndarray], int, str]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    channels = int(state.get("config", {}).get("feature_channels", 16))
    model = MarginalCompletionBaseline(feature_channels=channels).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(samples, batch_size=4, shuffle=False, collate_fn=collate_flatlands_replay, num_workers=0)
    output: dict[str, np.ndarray] = {}
    for batch in loader:
        probabilities = model.free_probability(_to_tensor_batch(batch, device)["observation"])
        for index, global_id in enumerate(batch["global_ids"]):
            output[global_id] = probabilities[index].cpu().numpy().astype(np.float32, copy=True)
    return output, int(state.get("best_epoch", -1)), sha256_path(checkpoint)


def _tasks(samples: list[object], maps: dict[str, np.ndarray], radii: tuple[int, ...], count: int, seed: int) -> list[CompletionEventTask]:
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
            posterior_samples=count,
            seed=seed,
        ))
    return tasks


def _combine(member_results: list[list[CompletionEventResult]], member_counts: list[int], radii: tuple[int, ...]) -> list[dict[str, object]]:
    if not member_results:
        return []
    rows: list[dict[str, object]] = []
    total = float(sum(member_counts))
    for task_index, first in enumerate(member_results[0]):
        probability = np.zeros_like(first.independent, dtype=np.float64)
        for results, count in zip(member_results, member_counts):
            current = results[task_index]
            if current.global_id != first.global_id or not np.array_equal(current.candidate_indices, first.candidate_indices):
                raise RuntimeError("ensemble members returned different query ordering")
            probability += float(count) * current.independent
        probability /= total
        for query_index, candidate_index in enumerate(first.candidate_indices):
            for radius_index, radius in enumerate(radii):
                rows.append({
                    "global_id": first.global_id,
                    "candidate_index": int(candidate_index),
                    "radius_cells": radius,
                    "probability": float(probability[query_index, radius_index]),
                })
    return rows


def _ensemble_map_metrics(samples: list[object], member_maps: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    """Report hidden-cell calibration for the ensemble's mean map posterior."""
    brier: list[float] = []
    nll: list[float] = []
    positive_rate: list[float] = []
    mean_probability: list[float] = []
    for sample in samples:
        global_id = sample.observation.global_id
        probability = np.mean([maps[global_id] for maps in member_maps], axis=0)
        mask = np.asarray(sample.loss_mask, dtype=bool)
        target = np.asarray(sample.target_free, dtype=np.float64)
        if not np.any(mask):
            continue
        p = np.clip(probability[mask].astype(np.float64), 1e-7, 1.0 - 1e-7)
        y = target[mask]
        brier.append(float(np.mean((p - y) ** 2)))
        nll.append(float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)))))
        positive_rate.append(float(np.mean(y)))
        mean_probability.append(float(np.mean(p)))
    if not brier:
        raise RuntimeError("ensemble map evaluation produced no hidden valid cells")
    return {
        "scene_weighted_brier": float(np.mean(brier)),
        "scene_weighted_nll": float(np.mean(nll)),
        "scene_weighted_positive_rate": float(np.mean(positive_rate)),
        "scene_weighted_mean_probability": float(np.mean(mean_probability)),
        "scene_count": len(brier),
    }


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("provide at least two checkpoints; the PaSCo-inspired adapter uses multiple members")
    if any(value < 1 for value in args.total_samples) or args.event_workers < 1:
        raise SystemExit("sample counts and event-workers must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(args.device)
    dataset = FlatLandsReplayDataset(args.archive, args.selection, args.queries, split="validation", verify_frozen=True, verify_query_geometry=True)
    try:
        samples = [dataset[index] for index in range(len(dataset))]
        radii = dataset.radii_cells
    finally:
        dataset.close()
    member_maps: list[dict[str, np.ndarray]] = []
    member_meta: list[dict[str, object]] = []
    for checkpoint in args.checkpoints:
        maps, best_epoch, checkpoint_hash = _maps(checkpoint, samples, device)
        member_maps.append(maps)
        member_meta.append({"path": str(checkpoint), "sha256": checkpoint_hash, "best_epoch": best_epoch})
    map_metrics = _ensemble_map_metrics(samples, member_maps)

    summaries: dict[str, object] = {}
    for total_samples in sorted(set(args.total_samples)):
        base, remainder = divmod(total_samples, len(member_maps))
        member_counts = [base + (index < remainder) for index in range(len(member_maps))]
        if min(member_counts) < 1:
            raise SystemExit("total-samples must be at least the number of ensemble members")
        k_started = time.monotonic()
        prediction_path = args.output_dir / f"predictions_pasco_ensemble_k{total_samples}_validation.csv"
        evaluation_dir = args.output_dir / f"evaluation_pasco_ensemble_k{total_samples}_validation"
        report_path = evaluation_dir / "report.json"
        if args.reuse_existing and prediction_path.exists() and report_path.exists():
            with report_path.open("r", encoding="utf-8") as handle:
                evaluation = json.load(handle)
            with prediction_path.open("r", encoding="utf-8") as handle:
                count = sum(1 for _ in handle) - 1
        else:
            member_results: list[list[CompletionEventResult]] = []
            for index, maps in enumerate(member_maps):
                tasks = _tasks(samples, maps, radii, member_counts[index], args.seed + index * 1_000_003)
                with ProcessPoolExecutor(max_workers=args.event_workers, mp_context=mp.get_context("spawn")) as executor:
                    member_results.append(list(executor.map(completion_event_probabilities, tasks, chunksize=1)))
            count = write_prediction_manifest(prediction_path, _combine(member_results, member_counts, radii))
            evaluation = evaluate_flatlands_prediction_file(
                prediction_path, args.selection, args.queries,
                method=f"pasco_inspired_ensemble_k{total_samples}", split="validation",
                bootstrap_samples=args.bootstrap_samples, seed=args.seed,
            )
            write_evaluation_report(evaluation_dir, evaluation)
        summaries[str(total_samples)] = {
            "member_samples": member_counts,
            "prediction": {"path": str(prediction_path), "rows": count, "sha256": sha256_path(prediction_path)},
            "evaluation": str(evaluation_dir / "report.json"),
            "scene_weighted": evaluation["metrics"][0]["scene_weighted"],
            "map_metrics": map_metrics,
            "seconds": time.monotonic() - k_started,
        }
    report = {
        "schema_version": 1,
        "kind": "flatlands_pasco_inspired_ensemble",
        "method_label": "PaSCo-inspired multi-subnet ensemble control",
        "paper_result": False,
        "validation_result": True,
        "test_evaluated": False,
        "members": member_meta,
        "config": _jsonable(vars(args)),
        "data": {"selection_sha256": sha256_path(args.selection), "queries_sha256": sha256_path(args.queries), "archive_bytes": args.archive.stat().st_size},
        "radii_cells": list(radii),
        "validation_map_metrics": map_metrics,
        "git": _git_state(),
        "results": summaries,
        "runtime": {"wall_seconds": time.monotonic() - started, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "claim_boundary": "Validation-only same-contract ensemble control inspired by PaSCo's multi-subnet uncertainty idea; it is not a reproduction of PaSCo's 3-D architecture and test remains locked.",
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if args.publish_site:
        site_snapshot = {
            "schema_version": 1,
            "kind": report["kind"],
            "method_label": report["method_label"],
            "paper_result": False,
            "validation_result": True,
            "test_evaluated": False,
            "radii_cells": report["radii_cells"],
            "total_samples": sorted(int(key) for key in summaries),
            "results": {
                key: {
                    "member_samples": value["member_samples"],
                    "prediction": value["prediction"],
                    "evaluation": value["evaluation"],
                    "scene_weighted": {
                        metric: value["scene_weighted"][metric]
                        for metric in (
                            "brier",
                            "nll",
                            "ece",
                            "false_safe_rate@0.8",
                            "high_confidence_safe_coverage@0.8",
                        )
                    },
                    "map_metrics": value["map_metrics"],
                }
                for key, value in summaries.items()
            },
            "claim_boundary": report["claim_boundary"],
        }
        site_path = PROJECT_ROOT / "site" / "data" / "flatlands_pasco_ensemble_validation.json"
        site_path.parent.mkdir(parents=True, exist_ok=True)
        site_path.write_text(
            json.dumps(site_snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output_dir": str(args.output_dir), "results": {key: value["scene_weighted"] for key, value in summaries.items()}, "paper_result": False}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
