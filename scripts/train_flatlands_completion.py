#!/usr/bin/env python3
"""Train the frozen marginal-completion baseline and derive deterministic/independent events."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pathrel.flatlands_baselines import MarginalCompletionBaseline
from pathrel.flatlands_data import FlatLandsReplayDataset, collate_flatlands_replay
from pathrel.flatlands_eval import (
    evaluate_flatlands_prediction_file,
    write_evaluation_report,
    write_prediction_manifest,
)
from pathrel.flatlands_query import sha256_path
from pathrel.flatlands_sampling import (
    CompletionEventResult,
    CompletionEventTask,
    completion_event_probabilities,
)


PROTOCOL_VERSION = "P1_BASELINE_PROTOCOL.md v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/p1_flatlands_completion_seed20260831"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--event-workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments), check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "status": run("status", "--short").splitlines(),
    }


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    return generator


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_progress(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cache_split(
    archive: Path, selection: Path, queries: Path, split: str
) -> tuple[list[object], tuple[int, ...]]:
    dataset = FlatLandsReplayDataset(
        archive,
        selection,
        queries,
        split=split,
        verify_frozen=True,
        verify_query_geometry=True,
    )
    try:
        return [dataset[index] for index in range(len(dataset))], dataset.radii_cells
    finally:
        dataset.close()


def _to_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output = dict(batch)
    for key in ("observation", "target_free", "loss_mask"):
        output[key] = torch.from_numpy(batch[key]).to(device=device, non_blocking=True)
    return output


def _scene_mean_map_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    per_cell = F.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="none"
    )
    counts = mask.sum(dim=(1, 2))
    selected = counts > 0
    if not torch.any(selected):
        return logits.sum() * 0.0
    per_scene = (per_cell * mask).sum(dim=(1, 2))[selected] / counts[selected]
    return per_scene.mean()


@torch.inference_mode()
def _validation_map_summary(
    model: MarginalCompletionBaseline,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    brier: list[float] = []
    nll: list[float] = []
    positive_rate: list[float] = []
    mean_probability: list[float] = []
    cell_count = 0
    for numpy_batch in loader:
        batch = _to_tensor_batch(numpy_batch, device)
        logits = model(batch["observation"])
        targets = batch["target_free"].to(dtype=logits.dtype)
        mask = batch["loss_mask"].to(dtype=torch.bool)
        probability = torch.sigmoid(logits)
        per_cell_nll = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        for batch_index in range(logits.shape[0]):
            selected = mask[batch_index]
            if torch.any(selected):
                scene_probability = probability[batch_index][selected]
                scene_target = targets[batch_index][selected]
                brier.append(float((scene_probability - scene_target).square().mean().cpu()))
                nll.append(float(per_cell_nll[batch_index][selected].mean().cpu()))
                positive_rate.append(float(scene_target.mean().cpu()))
                mean_probability.append(float(scene_probability.mean().cpu()))
                cell_count += int(selected.sum().cpu())
    if not brier:
        raise RuntimeError("validation produced no hidden valid cells")
    return {
        "scene_weighted_brier": float(np.mean(brier)),
        "scene_weighted_nll": float(np.mean(nll)),
        "scene_weighted_positive_rate": float(np.mean(positive_rate)),
        "scene_weighted_mean_probability": float(np.mean(mean_probability)),
        "scene_count": len(brier),
        "cell_count": cell_count,
    }


@torch.inference_mode()
def _probability_maps(
    model: MarginalCompletionBaseline,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, np.ndarray] = {}
    for numpy_batch in loader:
        batch = _to_tensor_batch(numpy_batch, device)
        probability = model.free_probability(batch["observation"]).cpu().numpy()
        for index, global_id in enumerate(numpy_batch["global_ids"]):
            output[global_id] = probability[index].astype(np.float32, copy=True)
    return output


def _sampling_tasks(
    samples: Sequence[object],
    probability_maps: dict[str, np.ndarray],
    radii: tuple[int, ...],
    *,
    posterior_samples: int,
    seed: int,
) -> list[CompletionEventTask]:
    tasks: list[CompletionEventTask] = []
    for sample in samples:
        retained = sample.retained_queries
        if not retained:
            continue
        tasks.append(
            CompletionEventTask(
                global_id=sample.observation.global_id,
                free_probability=probability_maps[sample.observation.global_id],
                observed_free=sample.observed_free,
                unknown=sample.unknown,
                starts=np.asarray(
                    [(query.start_row, query.start_col) for query in retained],
                    dtype=np.int64,
                ),
                goals=np.asarray(
                    [(query.goal_row, query.goal_col) for query in retained],
                    dtype=np.int64,
                ),
                candidate_indices=np.asarray(
                    [query.candidate_index for query in retained], dtype=np.int64
                ),
                radii_cells=radii,
                posterior_samples=posterior_samples,
                seed=seed,
            )
        )
    return tasks


def _event_rows(
    results: Sequence[CompletionEventResult],
    radii: tuple[int, ...],
    *,
    kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        values = result.deterministic if kind == "deterministic" else result.independent
        for query_index, candidate_index in enumerate(result.candidate_indices):
            for radius_index, radius in enumerate(radii):
                rows.append(
                    {
                        "global_id": result.global_id,
                        "candidate_index": int(candidate_index),
                        "radius_cells": radius,
                        "probability": float(values[query_index, radius_index]),
                    }
                )
    return rows


def _checkpoint_payload(
    *,
    model: MarginalCompletionBaseline,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_nll: float,
    patience_used: int,
    history: Sequence[dict[str, object]],
    data_generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "patience_used": patience_used,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": list(history),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_generator_state": data_generator.get_state(),
        "config": vars(args),
    }


def main() -> None:
    args = parse_args()
    if args.feature_channels != 16 or args.batch_size != 4:
        raise SystemExit("protocol v1 fixes feature-channels=16 and batch-size=4")
    if args.max_epochs > 80 or args.patience != 12 or args.min_delta != 1e-5:
        raise SystemExit("protocol v1 fixes max-epochs<=80, patience=12, min-delta=1e-5")
    if args.learning_rate != 3e-4 or args.weight_decay != 1e-4:
        raise SystemExit("protocol v1 fixes AdamW learning-rate=3e-4 and weight-decay=1e-4")
    if args.posterior_samples != 32:
        raise SystemExit("protocol v1 pilot fixes --posterior-samples=32")
    if args.event_workers < 1:
        raise SystemExit("--event-workers must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; pass --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_generator = _seed_everything(args.seed)
    started = time.monotonic()
    progress_path = args.output_dir / "progress.jsonl"
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"

    print("Caching frozen train/validation packets directly from ZIP...", flush=True)
    train_samples, train_radii = _cache_split(
        args.archive, args.selection, args.queries, "train"
    )
    validation_samples, validation_radii = _cache_split(
        args.archive, args.selection, args.queries, "validation"
    )
    if train_radii != validation_radii:
        raise RuntimeError("train/validation radii differ")
    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=data_generator,
        collate_fn=collate_flatlands_replay,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_flatlands_replay,
        num_workers=0,
    )
    model = MarginalCompletionBaseline(feature_channels=args.feature_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = 0
    best_nll = float("inf")
    patience_used = 0
    if args.resume:
        if not latest_path.exists():
            raise SystemExit(f"resume checkpoint not found: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_nll = float(state["best_nll"])
        patience_used = int(state["patience_used"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda_rng_state_all"]]
            )
        data_generator.set_state(state["data_generator_state"].cpu())

    try:
        for epoch in range(start_epoch, args.max_epochs + 1):
            model.train()
            losses: list[float] = []
            epoch_start = time.monotonic()
            for numpy_batch in train_loader:
                batch = _to_tensor_batch(numpy_batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["observation"])
                loss = _scene_mean_map_loss(
                    logits,
                    batch["target_free"],
                    batch["loss_mask"].to(dtype=torch.bool),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation = _validation_map_summary(model, validation_loader, device)
            improved = (
                float(validation["scene_weighted_nll"]) < best_nll - args.min_delta
            )
            if improved:
                best_nll = float(validation["scene_weighted_nll"])
                best_epoch = epoch
                patience_used = 0
            else:
                patience_used += 1
            record = {
                "epoch": epoch,
                "train_scene_nll": float(np.mean(losses)),
                **validation,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_scene_weighted_nll": best_nll,
                "patience_used": patience_used,
                "epoch_seconds": time.monotonic() - epoch_start,
                "elapsed_seconds": time.monotonic() - started,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            history.append(record)
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_nll=best_nll,
                patience_used=patience_used,
                history=history,
                data_generator=data_generator,
                args=args,
            )
            _atomic_torch_save(latest_path, payload)
            if improved:
                _atomic_torch_save(best_path, payload)
            _append_progress(progress_path, record)
            print(json.dumps(record, allow_nan=False), flush=True)
            if patience_used >= args.patience:
                break
    except KeyboardInterrupt:
        if history:
            _atomic_torch_save(
                args.output_dir / "interrupted.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=int(history[-1]["epoch"]),
                    best_epoch=best_epoch,
                    best_nll=best_nll,
                    patience_used=patience_used,
                    history=history,
                    data_generator=data_generator,
                    args=args,
                ),
            )
        raise

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    final_map_metrics = _validation_map_summary(model, validation_loader, device)
    probability_maps = _probability_maps(model, validation_loader, device)
    tasks = _sampling_tasks(
        validation_samples,
        probability_maps,
        validation_radii,
        posterior_samples=args.posterior_samples,
        seed=args.seed,
    )
    sampling_started = time.monotonic()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.event_workers, mp_context=context
    ) as executor:
        event_results = list(
            executor.map(completion_event_probabilities, tasks, chunksize=1)
        )
    sampling_seconds = time.monotonic() - sampling_started

    method_specs = (
        (
            "deterministic_completion",
            "deterministic",
            args.output_dir / "predictions_deterministic_validation.csv",
            args.output_dir / "evaluation_deterministic_validation",
        ),
        (
            "independent_cell_completion_k32",
            "independent",
            args.output_dir / "predictions_independent_k32_validation.csv",
            args.output_dir / "evaluation_independent_k32_validation",
        ),
    )
    evaluations: dict[str, dict[str, object]] = {}
    prediction_artifacts: dict[str, dict[str, object]] = {}
    for method, kind, prediction_path, evaluation_dir in method_specs:
        prediction_count = write_prediction_manifest(
            prediction_path,
            _event_rows(event_results, validation_radii, kind=kind),
        )
        evaluation = evaluate_flatlands_prediction_file(
            prediction_path,
            args.selection,
            args.queries,
            method=method,
            split="validation",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_evaluation_report(evaluation_dir, evaluation)
        evaluations[method] = evaluation["metrics"][0]["scene_weighted"]
        prediction_artifacts[method] = {
            "path": str(prediction_path),
            "rows": prediction_count,
            "sha256": sha256_path(prediction_path),
            "evaluation": str(evaluation_dir / "report.json"),
        }

    total_seconds = time.monotonic() - started
    run_report = {
        "schema_version": 1,
        "kind": "flatlands_marginal_completion_training",
        "paper_result": False,
        "validation_result": True,
        "protocol_version": PROTOCOL_VERSION,
        "test_evaluated": False,
        "git": _git_state(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "train_scenes": len(train_samples),
            "validation_scenes": len(validation_samples),
            "selection_sha256": sha256_path(args.selection),
            "queries_sha256": sha256_path(args.queries),
            "archive_bytes": args.archive.stat().st_size,
        },
        "selection": {
            "criterion": "minimum validation equal-scene hidden-cell Bernoulli NLL",
            "best_epoch": int(selected["best_epoch"]),
            "best_scene_weighted_nll_during_training": float(selected["best_nll"]),
            "epochs_completed": len(history),
        },
        "validation_map_metrics": final_map_metrics,
        "validation_event_metrics": evaluations,
        "artifacts": {
            "best_checkpoint": {
                "path": str(best_path),
                "sha256": sha256_path(best_path),
            },
            "latest_checkpoint": {
                "path": str(latest_path),
                "sha256": sha256_path(latest_path),
            },
            "predictions": prediction_artifacts,
        },
        "runtime": {
            "wall_seconds": total_seconds,
            "event_sampling_seconds": sampling_seconds,
            "event_workers": args.event_workers,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "argv": sys.argv,
        },
        "history": history,
        "claim_boundary": (
            "Single-seed validation-only occupancy baselines. Independent events use K=32 pilot "
            "sampling. Test remains locked; these are not final paper results."
        ),
    }
    _atomic_json(args.output_dir / "run.json", run_report)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "best_epoch": selected["best_epoch"],
                "validation_map_metrics": final_map_metrics,
                "validation_event_metrics": {
                    method: {
                        key: metrics[key]
                        for key in (
                            "brier",
                            "nll",
                            "ece",
                            "false_safe_rate@0.8",
                            "count",
                            "scene_count",
                        )
                    }
                    for method, metrics in evaluations.items()
                },
                "event_sampling_seconds": sampling_seconds,
                "test_evaluated": False,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
