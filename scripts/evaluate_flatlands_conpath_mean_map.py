#!/usr/bin/env python3
"""Evaluate a ConPath checkpoint's deterministic posterior mean-map control.

The control averages the posterior's conditional free-cell probabilities over a fixed number of
world draws, thresholds the resulting mean map at 0.5, and evaluates the resulting binary map with
the exact FlatLands connectivity oracle. It writes an ordinary label-free event manifest plus a
compact hidden-cell map summary, so event and map quality are recorded under one checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pathrel.flatlands_data import FlatLandsReplayDataset, collate_flatlands_replay
from pathrel.flatlands_eval import evaluate_flatlands_prediction_file, write_evaluation_report, write_prediction_manifest
from pathrel.labels import batched_merge_tree_bottleneck_scores, clearance_radius_map
from pathrel.model import PathRelNet
from pathrel.flatlands_query import sha256_path
from pathrel.gpu_diagnostics import cuda_unavailable_message


PROTOCOL_VERSION = "P1_BASELINE_PROTOCOL.md v1 + ConPath deterministic mean-map control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"))
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--sample-chunk", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true", help="allow an existing output directory")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cache_validation(archive: Path, selection: Path, queries: Path) -> tuple[list[object], tuple[int, ...]]:
    dataset = FlatLandsReplayDataset(archive, selection, queries, split="validation", verify_frozen=True, verify_query_geometry=True)
    try:
        return [dataset[index] for index in range(len(dataset))], dataset.radii_cells
    finally:
        dataset.close()


def _seed_generator(seed: int, device: torch.device) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 17)
    return generator


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output = dict(batch)
    for key in ("observation", "target_free", "loss_mask"):
        output[key] = torch.from_numpy(batch[key]).to(device=device, non_blocking=True)
    return output


def _map_summary(map_rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not map_rows:
        raise RuntimeError("validation produced no hidden-cell map rows")
    return {
        "scene_weighted_brier": float(np.mean([row["brier"] for row in map_rows])),
        "scene_weighted_nll": float(np.mean([row["nll"] for row in map_rows])),
        "scene_weighted_positive_rate": float(np.mean([row["positive_rate"] for row in map_rows])),
        "scene_weighted_mean_probability": float(np.mean([row["mean_probability"] for row in map_rows])),
        "scene_count": len(map_rows),
        "cell_count": int(sum(int(row["cell_count"]) for row in map_rows)),
    }


@torch.inference_mode()
def evaluate(
    model: PathRelNet,
    samples: list[object],
    radii: tuple[int, ...],
    *,
    device: torch.device,
    posterior_samples: int,
    sample_chunk: int,
    generator: torch.Generator,
    disable_global_factors: bool,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_flatlands_replay, num_workers=0)
    model.eval()
    event_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, float | int]] = []
    for numpy_batch in loader:
        batch = _to_device(numpy_batch, device)
        probability_sum: torch.Tensor | None = None
        remaining = posterior_samples
        while remaining > 0:
            current = min(sample_chunk, remaining)
            output = model(batch["observation"], num_samples=current, hard_samples=True, disable_global_factors=disable_global_factors, generator=generator)
            current_probability = output.posterior.posterior_marginal_probs[:, 0]
            probability_sum = current_probability * current if probability_sum is None else probability_sum + current_probability * current
            remaining -= current
        assert probability_sum is not None
        probability = (probability_sum / float(posterior_samples))[0].cpu().numpy()
        observation = numpy_batch["observation"][0]
        support = observation[0] > 0.5
        support |= observation[1] > 0.5
        support |= observation[2] > 0.5
        hidden = numpy_batch["loss_mask"][0].astype(bool)
        target = numpy_batch["target_free"][0].astype(np.float64)
        selected = hidden
        target_selected = target[selected]
        probability_selected = probability[selected].astype(np.float64)
        clipped = np.clip(probability_selected, 1e-6, 1.0 - 1e-6)
        map_rows.append({
            "brier": float(np.mean((probability_selected - target_selected) ** 2)),
            "nll": float(-np.mean(target_selected * np.log(clipped) + (1.0 - target_selected) * np.log1p(-clipped))),
            "positive_rate": float(np.mean(target_selected)),
            "mean_probability": float(np.mean(probability_selected)),
            "cell_count": int(selected.sum()),
        })

        deterministic_map = (probability >= 0.5) & support
        scores = clearance_radius_map(deterministic_map)[None, None].astype(np.float64, copy=False)
        starts = numpy_batch["starts"]
        goals = numpy_batch["goals"]
        bottleneck = batched_merge_tree_bottleneck_scores(scores, starts, goals)[0, 0]
        mask = numpy_batch["query_mask"][0]
        global_id = numpy_batch["global_ids"][0]
        candidates = numpy_batch["candidate_indices"][0]
        for query_index in np.flatnonzero(mask):
            for radius_index, radius in enumerate(radii):
                event_rows.append({
                    "global_id": global_id,
                    "candidate_index": int(candidates[query_index]),
                    "radius_cells": int(radius),
                    "probability": float(bottleneck[query_index] >= radius),
                })
    return event_rows, _map_summary(map_rows)


def main() -> None:
    args = parse_args()
    if args.validation_samples < 1 or args.sample_chunk < 1:
        raise SystemExit("validation-samples and sample-chunk must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_unavailable_message(torch))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; pass --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(args.device)
    generator = _seed_generator(args.seed, device)
    samples, radii = _cache_validation(args.archive, args.selection, args.queries)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    variant = str(config.get("decoder_variant", "correlated"))
    independent = variant == "independent"
    feature_channels = int(config.get("feature_channels", 16))
    latent_dim = int(config.get("latent_dim", 4))
    disable_global_factors = bool(config.get("disable_global_factors", False) or independent)
    model = PathRelNet(feature_channels=feature_channels, latent_dim=latent_dim, local_kernel_size=(1 if independent else 5)).to(device)
    model.load_state_dict(state["model"])
    rows, map_metrics = evaluate(model, samples, radii, device=device, posterior_samples=args.validation_samples, sample_chunk=args.sample_chunk, generator=generator, disable_global_factors=disable_global_factors)
    prediction_path = args.output_dir / "predictions_validation.csv"
    prediction_count = write_prediction_manifest(prediction_path, rows)
    event_evaluation = evaluate_flatlands_prediction_file(prediction_path, args.selection, args.queries, method="conpath_deterministic_mean_map", split="validation", bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_evaluation_report(args.output_dir / "evaluation_validation", event_evaluation)
    report = {
        "schema_version": 1,
        "kind": "flatlands_conpath_deterministic_mean_map",
        "paper_result": False,
        "validation_result": True,
        "protocol_version": PROTOCOL_VERSION,
        "test_evaluated": False,
        "git": {"head": subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256_path(args.checkpoint), "best_epoch": int(state.get("best_epoch", 0))},
        "data": {"validation_scenes": len(samples), "selection_sha256": sha256_path(args.selection), "queries_sha256": sha256_path(args.queries), "archive_bytes": args.archive.stat().st_size},
        "decoder": {"variant": variant, "feature_channels": feature_channels, "latent_dim": latent_dim, "local_kernel_size": 1 if independent else 5, "disable_global_factors": disable_global_factors},
        "mean_map": {"posterior_samples": args.validation_samples, "sample_chunk": args.sample_chunk, "threshold": 0.5, "metrics": map_metrics},
        "prediction": {"path": str(prediction_path), "rows": prediction_count, "sha256": sha256_path(prediction_path)},
        "event_evaluation": event_evaluation,
        "runtime": {"wall_seconds": time.monotonic() - started, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "argv": sys.argv},
        "claim_boundary": "Validation-only deterministic threshold of a ConPath posterior mean map. The event output is binary and this control is not a final paper result; FlatLands test remains locked.",
    }
    _atomic_json(args.output_dir / "map_metrics.json", map_metrics)
    _atomic_json(args.output_dir / "run.json", report)
    print(json.dumps({"output_dir": str(args.output_dir), "validation_result": True, "prediction_rows": prediction_count, "event_brier": event_evaluation["metrics"][0]["scene_weighted"]["brier"], "map_brier": map_metrics["scene_weighted_brier"], "paper_result": False}, indent=2))


if __name__ == "__main__":
    main()
