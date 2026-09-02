#!/usr/bin/env python3
"""Train a first UnScenes3D ConPath adapter smoke on train/validation only.

The script deliberately reads the frozen train/validation manifest and never opens
location-6 labels.  Map fitting is the default smoke path; exact event probabilities
are evaluated afterward from hard posterior worlds with the NumPy oracle.  This
keeps the initial GPU run bounded while preserving the event metric semantics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.labels import clearance_radius_map, maximum_clearance_map  # noqa: E402
from pathrel.losses import (  # noqa: E402
    posterior_marginal_nll,
    reachability_brier_u_statistic,
    spatial_variogram_score,
)
from pathrel.model import PathRelNet  # noqa: E402
from pathrel.unscenes3d import load_frame  # noqa: E402


PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.1"
RADII_CELLS = (0, 1, 2)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class UnScenesFrameDataset(Dataset[dict[str, object]]):
    def __init__(self, records: list[dict[str, object]], raw_root: Path, label_root: Path) -> None:
        self.records = records
        self.raw_root = raw_root
        self.label_root = label_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        frame = load_frame(str(record["timestamp"]), raw_root=self.raw_root, label_root=self.label_root)
        unknown = frame.input_bev[2] > 0.5
        loss_mask = frame.target_valid & unknown
        return {
            "timestamp": frame.timestamp,
            "scene_id": str(record["scene_id"]),
            "location": str(record["location"]),
            "observation": frame.input_bev.astype(np.float32, copy=False),
            "target_free": frame.target_free.astype(np.float32, copy=False),
            "loss_mask": loss_mask,
            "queries": list(record["queries"]),
        }


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    query_count = max(len(item["queries"]) for item in batch)
    starts = np.zeros((len(batch), query_count, 2), dtype=np.int64)
    goals = np.zeros((len(batch), query_count, 2), dtype=np.int64)
    reachability_targets = np.zeros((len(batch), query_count, len(RADII_CELLS)), dtype=bool)
    query_mask = np.zeros((len(batch), query_count), dtype=bool)
    for batch_index, item in enumerate(batch):
        for query_index, row in enumerate(item["queries"]):
            starts[batch_index, query_index] = np.asarray(row["start"], dtype=np.int64)
            goals[batch_index, query_index] = np.asarray(row["goal"], dtype=np.int64)
            reachability_targets[batch_index, query_index] = np.asarray(
                row["reachable"], dtype=bool
            )
            query_mask[batch_index, query_index] = True
        # The shared-start propagation path requires one start per frame.  Invalid padded
        # queries are never included in the loss, but they still receive the frame's first
        # start/goal so the geometry operator sees a valid repeated-start stencil.
        item_query_count = len(item["queries"])
        if item_query_count and item_query_count < query_count:
            starts[batch_index, item_query_count:] = starts[batch_index, 0]
            goals[batch_index, item_query_count:] = goals[batch_index, 0]
    return {
        "timestamps": [str(item["timestamp"]) for item in batch],
        "scene_ids": [str(item["scene_id"]) for item in batch],
        "locations": [str(item["location"]) for item in batch],
        "observation": np.stack([item["observation"] for item in batch]),
        "target_free": np.stack([item["target_free"] for item in batch]),
        "loss_mask": np.stack([item["loss_mask"] for item in batch]),
        "starts": starts,
        "goals": goals,
        "reachability_targets": reachability_targets,
        "query_mask": query_mask,
    }


def _tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "observation": torch.from_numpy(batch["observation"]).to(device=device, non_blocking=True),
        "target_free": torch.from_numpy(batch["target_free"]).to(device=device, non_blocking=True),
        "loss_mask": torch.from_numpy(batch["loss_mask"]).to(device=device, dtype=torch.bool, non_blocking=True),
        "starts": torch.from_numpy(batch["starts"]).to(device=device, dtype=torch.long, non_blocking=True),
        "goals": torch.from_numpy(batch["goals"]).to(device=device, dtype=torch.long, non_blocking=True),
        "reachability_targets": torch.from_numpy(batch["reachability_targets"]).to(device=device, dtype=torch.bool, non_blocking=True),
        "query_mask": torch.from_numpy(batch["query_mask"]).to(device=device, dtype=torch.bool, non_blocking=True),
    }


def _target_classes(target_free: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    target = torch.full(target_free.shape, -1, dtype=torch.long, device=target_free.device)
    return torch.where(loss_mask, (1.0 - target_free).to(torch.long), target)


def _event_forward(
    model: PathRelNet,
    batch: dict[str, torch.Tensor],
    *,
    samples: int,
    max_reachability_steps: int,
    generator: torch.Generator,
) -> Any:
    return model(
        batch["observation"],
        starts=batch["starts"],
        goals=batch["goals"],
        footprint_radii_cells=RADII_CELLS,
        num_samples=samples,
        hard_samples=True,
        max_reachability_steps=max_reachability_steps,
        shared_start=True,
        generator=generator,
    )


@torch.inference_mode()
def _validation_event_brier(
    model: PathRelNet,
    loader: DataLoader,
    device: torch.device,
    *,
    samples: int,
    max_reachability_steps: int,
    generator: torch.Generator,
) -> float:
    """Bounded differentiable-event validation score for checkpoint selection."""

    model.eval()
    per_scene_scores: dict[str, list[float]] = defaultdict(list)
    for numpy_batch in loader:
        batch = _tensor_batch(numpy_batch, device)
        output = _event_forward(
            model,
            batch,
            samples=samples,
            max_reachability_steps=max_reachability_steps,
            generator=generator,
        )
        if output.reachability is None:
            raise RuntimeError("event validation forward did not return reachability")
        query_mask = batch["query_mask"]
        errors = (output.reachability - batch["reachability_targets"].to(output.reachability.dtype)).square()
        query_scores = errors.mean(dim=2)
        for index, scene_id in enumerate(numpy_batch["scene_ids"]):
            per_scene_scores[str(scene_id)].extend(
                query_scores[index][query_mask[index]].detach().cpu().tolist()
            )
    if not per_scene_scores:
        raise RuntimeError("validation contains no event queries")
    return float(np.mean([np.mean(values) for values in per_scene_scores.values()]))


@torch.inference_mode()
def _validation_map_metrics(model: PathRelNet, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    per_scene_briers: dict[str, list[float]] = defaultdict(list)
    per_scene_nlls: dict[str, list[float]] = defaultdict(list)
    valid_cells = 0
    for numpy_batch in loader:
        batch = _tensor_batch(numpy_batch, device)
        posterior = model(batch["observation"], num_samples=4, hard_samples=True).posterior
        probability = posterior.posterior_marginal_probs[:, 0]
        target = batch["target_free"]
        mask = batch["loss_mask"]
        nll = F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        for index in range(probability.shape[0]):
            selected = mask[index]
            if torch.any(selected):
                scene_id = str(numpy_batch["scene_ids"][index])
                per_scene_briers[scene_id].append(
                    float((probability[index][selected] - target[index][selected]).square().mean().cpu())
                )
                per_scene_nlls[scene_id].append(float(nll[index][selected].mean().cpu()))
                valid_cells += int(selected.sum().cpu())
    if not per_scene_briers:
        raise RuntimeError("validation contains no hidden valid cells")
    return {
        "scene_weighted_brier": float(np.mean([np.mean(values) for values in per_scene_briers.values()])),
        "scene_weighted_nll": float(np.mean([np.mean(values) for values in per_scene_nlls.values()])),
        "scene_count": len(per_scene_briers),
        "hidden_valid_cells": valid_cells,
    }


def _event_probabilities(
    model: PathRelNet,
    records: list[dict[str, object]],
    raw_root: Path,
    label_root: Path,
    device: torch.device,
    posterior_samples: int,
    generator: torch.Generator,
) -> dict[str, object]:
    model.eval()
    per_scene: dict[str, list[float]] = defaultdict(list)
    all_targets: list[float] = []
    all_predictions: list[float] = []
    monotonicity_violations = 0
    query_count = 0
    for record in records:
        frame = load_frame(str(record["timestamp"]), raw_root=raw_root, label_root=label_root)
        query_rows = list(record["queries"])
        if not query_rows:
            continue
        starts = np.asarray([row["start"] for row in query_rows], dtype=np.int64)
        goals = np.asarray([row["goal"] for row in query_rows], dtype=np.int64)
        targets = np.asarray([row["reachable"] for row in query_rows], dtype=np.float64)
        with torch.inference_mode():
            observation = torch.from_numpy(frame.input_bev[None]).to(device=device)
            posterior = model(
                observation,
                num_samples=posterior_samples,
                hard_samples=True,
                generator=generator,
            ).posterior
            worlds = posterior.safe_samples().detach().cpu().numpy()[0] > 0.5
        events = np.zeros((posterior_samples, len(query_rows), len(RADII_CELLS)), dtype=np.float64)
        for sample_index, world in enumerate(worlds):
            clearance = clearance_radius_map(world)
            start = tuple(int(value) for value in starts[0])
            best = maximum_clearance_map(
                world,
                start,
                clearance=clearance,
                stop_points=[tuple(int(value) for value in goal) for goal in goals],
            )
            max_clearance = best[goals[:, 0], goals[:, 1]]
            events[sample_index] = max_clearance[:, None] >= np.asarray(RADII_CELLS)[None, :]
        probabilities = events.mean(axis=0)
        monotonicity_violations += int(np.sum(probabilities[:, 1:] > probabilities[:, :-1] + 1e-12))
        query_count += len(query_rows)
        for query_index in range(len(query_rows)):
            values = (probabilities[query_index] - targets[query_index]) ** 2
            per_scene[str(record["scene_id"])].append(float(values.mean()))
            all_targets.extend(targets[query_index].tolist())
            all_predictions.extend(probabilities[query_index].tolist())
    if not all_predictions:
        raise RuntimeError("validation contains no event queries")
    prediction = np.asarray(all_predictions)
    target = np.asarray(all_targets)
    brier = float(np.mean([np.mean(values) for values in per_scene.values()]))
    epsilon = 1e-6
    nll = float(np.mean(-(target * np.log(np.clip(prediction, epsilon, 1 - epsilon)) + (1 - target) * np.log(np.clip(1 - prediction, epsilon, 1 - epsilon)))))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        selected = (prediction >= lower) & ((prediction < upper) if upper < 1 else (prediction <= upper))
        if np.any(selected):
            ece += float(selected.mean()) * abs(float(prediction[selected].mean()) - float(target[selected].mean()))
    high = prediction >= 0.8
    false_safe = float(np.mean(target[high] < 0.5)) if np.any(high) else 0.0
    return {
        "query_count": query_count,
        "scene_count": len(per_scene),
        "scene_weighted_brier": brier,
        "query_weighted_nll": nll,
        "query_weighted_ece": ece,
        "false_safe_rate@0.8": false_safe,
        "high_confidence_coverage@0.8": float(high.mean()),
        "positive_prevalence": float(target.mean()),
        "mean_prediction": float(prediction.mean()),
        "radius_monotonicity_violations": monotonicity_violations,
    }


def _seed(seed: int, device: torch.device) -> tuple[torch.Generator, torch.Generator]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 11)
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(seed + 17)
    return loader_generator, sample_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/unscenes3d_contract_manifest/manifest.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/unscenes3d_conpath_smoke_seed20260902"))
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--max-reachability-steps", type=int, default=64)
    parser.add_argument("--map-weight", type=float, default=1.0)
    parser.add_argument("--variogram-weight", type=float, default=0.1)
    parser.add_argument(
        "--reachability-weight",
        type=float,
        default=0.0,
        help="event Brier U-statistic weight; zero preserves the map-only adapter diagnostic",
    )
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--train-frame-limit", type=int, default=None)
    parser.add_argument("--validation-frame-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_samples < 2 or args.posterior_samples < 2:
        raise SystemExit("train/posterior samples must be at least two")
    if args.max_reachability_steps < 1:
        raise SystemExit("max-reachability-steps must be positive")
    if min(args.map_weight, args.variogram_weight, args.reachability_weight) < 0:
        raise SystemExit("loss weights must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records = list(manifest["records"]["train"])
    validation_records = list(manifest["records"]["validation"])
    if args.train_frame_limit is not None:
        train_records = train_records[: args.train_frame_limit]
    if args.validation_frame_limit is not None:
        validation_records = validation_records[: args.validation_frame_limit]
    if not train_records or not validation_records:
        raise SystemExit("train and validation records must be non-empty")
    device = torch.device(args.device)
    loader_generator, sample_generator = _seed(args.seed, device)
    train_loader = DataLoader(
        UnScenesFrameDataset(train_records, args.raw_root, args.label_root),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        UnScenesFrameDataset(validation_records, args.raw_root, args.label_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    model = PathRelNet(input_channels=3, feature_channels=args.feature_channels, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    started = time.monotonic()
    history: list[dict[str, object]] = []
    best_nll = float("inf")
    best_epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for numpy_batch in train_loader:
            batch = _tensor_batch(numpy_batch, device)
            target_classes = _target_classes(batch["target_free"], batch["loss_mask"])
            if args.reachability_weight > 0:
                output = _event_forward(
                    model,
                    batch,
                    samples=args.train_samples,
                    max_reachability_steps=args.max_reachability_steps,
                    generator=sample_generator,
                )
            else:
                output = model(
                    batch["observation"],
                    num_samples=args.train_samples,
                    hard_samples=True,
                    generator=sample_generator,
                )
            map_loss = posterior_marginal_nll(output.posterior.sample_logits, target_classes)
            variogram = spatial_variogram_score(
                output.posterior.safe_samples(),
                batch["target_free"],
                valid_mask=batch["loss_mask"],
            )
            event_loss = map_loss.new_zeros(())
            if args.reachability_weight > 0:
                if output.sample_reachability is None:
                    raise RuntimeError("event training forward did not return sample reachability")
                weights = batch["query_mask"][..., None].expand_as(batch["reachability_targets"])
                event_loss = reachability_brier_u_statistic(
                    output.sample_reachability,
                    batch["reachability_targets"],
                    weights=weights,
                )
            loss = (
                args.map_weight * map_loss
                + args.variogram_weight * variogram
                + args.reachability_weight * event_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = _validation_map_metrics(model, validation_loader, device)
        validation_event = None
        if args.reachability_weight > 0:
            validation_event = _validation_event_brier(
                model,
                validation_loader,
                device,
                samples=min(args.posterior_samples, 4),
                max_reachability_steps=args.max_reachability_steps,
                generator=sample_generator,
            )
            selection_value = validation_event
        else:
            selection_value = float(validation["scene_weighted_nll"])
        if selection_value < best_nll:
            best_nll = float(selection_value)
            best_epoch = epoch
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **validation,
            "validation_event_brier": validation_event,
            "selection_metric": "event_brier" if args.reachability_weight > 0 else "map_nll",
            "selection_value": float(selection_value),
            "best_epoch": best_epoch,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        _atomic_torch_save(args.output_dir / "latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "config": vars(args)})
        if epoch == best_epoch:
            _atomic_torch_save(args.output_dir / "best.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "config": vars(args)})
        print(json.dumps(record, sort_keys=True), flush=True)

    selected = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    event_metrics = _event_probabilities(
        model,
        validation_records,
        args.raw_root,
        args.label_root,
        device,
        args.posterior_samples,
        sample_generator,
    )
    run = {
        "schema_version": 1,
        "kind": "unscenes3d_conpath_adapter_smoke",
        "paper_result": False,
        "validation_result": True,
        "test_evaluated": False,
        "protocol_version": PROTOCOL_VERSION,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "manifest": str(args.manifest),
            "test_locked_sites": manifest["test_locked_sites"],
        },
        "selection": {
            "criterion": "validation event Brier" if args.reachability_weight > 0 else "validation hidden-valid-cell NLL",
            "best_epoch": best_epoch,
        },
        "validation_event_metrics": event_metrics,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        },
        "history": history,
        "claim_boundary": (
            "UnScenes3D adapter validation-only run; event-loss optimization is enabled only when "
            "reachability_weight is positive, and location_6 test remains locked."
        ),
    }
    _atomic_json(args.output_dir / "run.json", run)
    print(json.dumps({"output_dir": str(args.output_dir), "event_metrics": event_metrics, "test_evaluated": False}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
