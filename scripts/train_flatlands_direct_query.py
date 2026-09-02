#!/usr/bin/env python3
"""Train the frozen direct-query FlatLands baseline and evaluate validation predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pathrel.flatlands_baselines import (
    DirectQueryBaseline,
    S4CInspiredCoordinateBaseline,
)
from pathrel.flatlands_data import FlatLandsReplayDataset, collate_flatlands_replay
from pathrel.flatlands_eval import (
    evaluate_flatlands_prediction_file,
    write_evaluation_report,
    write_prediction_manifest,
)
from pathrel.flatlands_query import sha256_path
from pathrel.gpu_diagnostics import cuda_unavailable_message


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
        default=Path("results/p1_flatlands_direct_query_seed20260831"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--architecture",
        choices=("direct_query", "s4c_coordinate"),
        default="direct_query",
        help="direct fixed-geometry MLP or S4C-inspired coordinate-query control",
    )
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cache_split(
    archive: Path,
    selection: Path,
    queries: Path,
    split: str,
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
        samples = [dataset[index] for index in range(len(dataset))]
        return samples, dataset.radii_cells
    finally:
        dataset.close()


def _to_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    tensor_keys = (
        "observation",
        "starts",
        "goals",
        "reachability_targets",
        "query_mask",
        "distances_m",
        "angles_deg",
        "radii_cells",
        "candidate_indices",
    )
    output = dict(batch)
    for key in tensor_keys:
        output[key] = torch.from_numpy(batch[key]).to(device=device, non_blocking=True)
    return output


def _forward(model: nn.Module, batch: dict[str, object]) -> Tensor:
    return model(
        batch["observation"],
        batch["starts"],
        batch["goals"],
        batch["distances_m"],
        batch["angles_deg"],
        batch["radii_cells"],
    )


def _scene_mean_loss(logits: Tensor, targets: Tensor, query_mask: Tensor) -> Tensor:
    mask = query_mask[:, :, None].expand_as(logits)
    if not torch.any(mask):
        return logits.sum() * 0.0
    per_event = F.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="none"
    )
    counts = mask.sum(dim=(1, 2))
    selected = counts > 0
    per_scene = (per_event * mask).sum(dim=(1, 2))[selected] / counts[selected]
    return per_scene.mean()


@torch.inference_mode()
def _validation_summary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    scene_brier: list[float] = []
    scene_nll: list[float] = []
    event_count = 0
    for numpy_batch in loader:
        batch = _to_tensor_batch(numpy_batch, device)
        logits = _forward(model, batch)
        targets = batch["reachability_targets"].to(dtype=logits.dtype)
        mask = batch["query_mask"][:, :, None].expand_as(logits)
        probabilities = torch.sigmoid(logits)
        per_event_brier = (probabilities - targets).square()
        per_event_nll = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        for batch_index in range(logits.shape[0]):
            selected = mask[batch_index]
            if torch.any(selected):
                scene_brier.append(
                    float(per_event_brier[batch_index][selected].mean().cpu())
                )
                scene_nll.append(
                    float(per_event_nll[batch_index][selected].mean().cpu())
                )
                event_count += int(selected.sum().cpu())
    if not scene_brier:
        raise RuntimeError("validation produced no retained events")
    return {
        "scene_weighted_brier": float(np.mean(scene_brier)),
        "scene_weighted_nll": float(np.mean(scene_nll)),
        "scene_count": len(scene_brier),
        "event_count": event_count,
    }


@torch.inference_mode()
def _prediction_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for numpy_batch in loader:
        batch = _to_tensor_batch(numpy_batch, device)
        probabilities = torch.sigmoid(_forward(model, batch)).cpu().numpy()
        query_mask = numpy_batch["query_mask"]
        candidate_indices = numpy_batch["candidate_indices"]
        radii = numpy_batch["radii_cells"]
        for batch_index, global_id in enumerate(numpy_batch["global_ids"]):
            for query_index in np.flatnonzero(query_mask[batch_index]):
                candidate_index = int(candidate_indices[batch_index, query_index])
                for radius_index, radius in enumerate(radii):
                    rows.append(
                        {
                            "global_id": global_id,
                            "candidate_index": candidate_index,
                            "radius_cells": int(radius),
                            "probability": float(
                                probabilities[batch_index, query_index, radius_index]
                            ),
                        }
                    )
    return rows


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_brier: float,
    patience_used: int,
    history: Sequence[dict[str, object]],
    data_generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_brier": best_brier,
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
    if args.feature_channels != 16:
        raise SystemExit("protocol v1 fixes --feature-channels=16")
    if args.batch_size != 4:
        raise SystemExit("protocol v1 fixes --batch-size=4")
    if args.max_epochs > 120 or args.patience != 20 or args.min_delta != 1e-5:
        raise SystemExit("protocol v1 fixes max-epochs<=120, patience=20, min-delta=1e-5")
    if args.learning_rate != 3e-4 or args.weight_decay != 1e-4:
        raise SystemExit("protocol v1 fixes AdamW learning-rate=3e-4 and weight-decay=1e-4")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_unavailable_message(torch))
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
    if args.architecture == "s4c_coordinate":
        model: nn.Module = S4CInspiredCoordinateBaseline(
            feature_channels=args.feature_channels
        ).to(device)
        method_label = "s4c_inspired_coordinate"
        run_kind = "flatlands_s4c_inspired_coordinate_training"
        claim_boundary = (
            "Three-seed validation-only S4C-inspired coordinate-query control. This is not "
            "a reproduction of the original S4C 3-D system; test remains locked and this is "
            "not a final public-data or paper result."
        )
    else:
        model = DirectQueryBaseline(feature_channels=args.feature_channels).to(device)
        method_label = "direct_query"
        run_kind = "flatlands_direct_query_training"
        claim_boundary = (
            "Single-seed validation-only direct-query baseline. Test remains locked; this is not "
            "a final public-data or paper result."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = 0
    best_brier = float("inf")
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
        best_brier = float(state["best_brier"])
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
            epoch_losses: list[float] = []
            epoch_start = time.monotonic()
            for numpy_batch in train_loader:
                batch = _to_tensor_batch(numpy_batch, device)
                if not torch.any(batch["query_mask"]):
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, batch)
                loss = _scene_mean_loss(
                    logits,
                    batch["reachability_targets"],
                    batch["query_mask"],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if not epoch_losses:
                raise RuntimeError("training epoch contained no retained events")
            validation = _validation_summary(model, validation_loader, device)
            improved = (
                float(validation["scene_weighted_brier"])
                < best_brier - args.min_delta
            )
            if improved:
                best_brier = float(validation["scene_weighted_brier"])
                best_epoch = epoch
                patience_used = 0
            else:
                patience_used += 1
            record = {
                "epoch": epoch,
                "train_scene_nll": float(np.mean(epoch_losses)),
                **validation,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_scene_weighted_brier": best_brier,
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
                best_brier=best_brier,
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
                    best_brier=best_brier,
                    patience_used=patience_used,
                    history=history,
                    data_generator=data_generator,
                    args=args,
                ),
            )
        raise

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    prediction_path = args.output_dir / "predictions_validation.csv"
    prediction_count = write_prediction_manifest(
        prediction_path, _prediction_rows(model, validation_loader, device)
    )
    evaluation = evaluate_flatlands_prediction_file(
        prediction_path,
        args.selection,
        args.queries,
        method=method_label,
        split="validation",
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    evaluation_dir = args.output_dir / "evaluation_validation"
    write_evaluation_report(evaluation_dir, evaluation)
    total_seconds = time.monotonic() - started
    run_report = {
        "schema_version": 1,
        "kind": run_kind,
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
            "criterion": "minimum validation equal-scene event Brier",
            "best_epoch": int(selected["best_epoch"]),
            "best_scene_weighted_brier_during_training": float(selected["best_brier"]),
            "epochs_completed": len(history),
        },
        "artifacts": {
            "best_checkpoint": {
                "path": str(best_path),
                "sha256": sha256_path(best_path),
            },
            "latest_checkpoint": {
                "path": str(latest_path),
                "sha256": sha256_path(latest_path),
            },
            "predictions_validation": {
                "path": str(prediction_path),
                "rows": prediction_count,
                "sha256": sha256_path(prediction_path),
            },
            "evaluation_validation": str(evaluation_dir / "report.json"),
        },
        "runtime": {
            "wall_seconds": total_seconds,
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
        "claim_boundary": claim_boundary,
    }
    _atomic_json(args.output_dir / "run.json", run_report)
    overall = evaluation["metrics"][0]["scene_weighted"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "architecture": args.architecture,
                "best_epoch": selected["best_epoch"],
                "prediction_rows": prediction_count,
                "scene_weighted_validation": {
                    key: overall[key]
                    for key in (
                        "brier",
                        "nll",
                        "ece",
                        "false_safe_rate@0.8",
                        "count",
                        "scene_count",
                    )
                },
                "test_evaluated": False,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
