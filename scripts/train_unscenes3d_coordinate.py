#!/usr/bin/env python3
"""Train a validation-only S4C-inspired coordinate-query UnScenes3D control.

The control keeps the frozen UnScenes3D three-channel observation, query stencil,
site split, optimizer, and event labels, but replaces the stochastic occupancy
posterior with a bilinear coordinate-query Fourier MLP.  It is an architectural
control inspired by recent coordinate-query work, not a reproduction of an
original 3-D system.  The location-6 test site is never opened.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_ROOT))

from pathrel.flatlands_baselines import S4CInspiredCoordinateBaseline  # noqa: E402
from pathrel.unscenes3d import load_frame  # noqa: E402
from train_unscenes3d_conpath import (  # noqa: E402
    UnScenesFrameDataset,
    _atomic_json,
    _atomic_torch_save,
    _collate,
    _resolve_adapter_config,
    _tensor_batch,
)


PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.2"
RADII_CELLS = (0, 1, 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int, device: torch.device) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 11)
    return generator


def _coordinate_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Use the shared frame collator and derive native metric query features."""

    output = _collate(batch)
    query_count = int(np.asarray(output["starts"]).shape[1])
    candidate_indices = np.zeros((len(batch), query_count), dtype=np.int64)
    for batch_index, item in enumerate(batch):
        for query_index, query in enumerate(item["queries"]):
            candidate_indices[batch_index, query_index] = int(query["candidate_index"])
        if item["queries"] and len(item["queries"]) < query_count:
            candidate_indices[batch_index, len(item["queries"]):] = candidate_indices[
                batch_index, 0
            ]
    output["candidate_indices"] = candidate_indices
    starts = np.asarray(output["starts"], dtype=np.float32)
    goals = np.asarray(output["goals"], dtype=np.float32)
    delta = goals - starts
    output["distances_m"] = np.hypot(delta[..., 0], delta[..., 1]) * 0.3
    output["angles_deg"] = np.mod(np.degrees(np.arctan2(delta[..., 1], delta[..., 0])), 360.0)
    return output


def _coordinate_tensor_batch(
    batch: dict[str, object], device: torch.device
) -> dict[str, Tensor]:
    output = _tensor_batch(batch, device)
    output["distances_m"] = torch.from_numpy(batch["distances_m"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    output["angles_deg"] = torch.from_numpy(batch["angles_deg"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return output


def _forward(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    return model(
        batch["observation"],
        batch["starts"],
        batch["goals"],
        batch["distances_m"],
        batch["angles_deg"],
        torch.as_tensor(RADII_CELLS, device=batch["observation"].device, dtype=torch.long),
    )


def _scene_mean_loss(
    logits: Tensor,
    targets: Tensor,
    query_mask: Tensor,
    scene_ids: list[str],
) -> Tensor:
    """Equal-weight the contributing scenes in each optimization batch."""

    event_loss = F.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="none"
    )
    frame_mask = query_mask[:, :, None].expand_as(logits)
    scene_losses: list[Tensor] = []
    for scene_id in sorted(set(scene_ids)):
        selected_frames = [index for index, value in enumerate(scene_ids) if value == scene_id]
        selected = frame_mask[selected_frames]
        if torch.any(selected):
            scene_losses.append(event_loss[selected_frames][selected].mean())
    if not scene_losses:
        return logits.sum() * 0.0
    return torch.stack(scene_losses).mean()


@torch.inference_mode()
def _validation_summary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    per_scene_brier: dict[str, list[float]] = defaultdict(list)
    per_scene_nll: dict[str, list[float]] = defaultdict(list)
    event_count = 0
    for numpy_batch in loader:
        batch = _coordinate_tensor_batch(numpy_batch, device)
        logits = _forward(model, batch)
        targets = batch["reachability_targets"].to(dtype=logits.dtype)
        mask = batch["query_mask"][:, :, None].expand_as(logits)
        probabilities = torch.sigmoid(logits)
        brier = (probabilities - targets).square()
        nll = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        for index, scene_id in enumerate(numpy_batch["scene_ids"]):
            selected_queries = batch["query_mask"][index]
            if torch.any(selected_queries):
                # Match the exact evaluator: each query contributes one mean-over-
                # radii score, and scenes then receive equal weight.
                per_query_brier = brier[index][selected_queries].mean(dim=1)
                per_query_nll = nll[index][selected_queries].mean(dim=1)
                per_scene_brier[str(scene_id)].extend(per_query_brier.cpu().tolist())
                per_scene_nll[str(scene_id)].extend(per_query_nll.cpu().tolist())
                event_count += int(mask[index].sum().cpu())
    if not per_scene_brier:
        raise RuntimeError("validation produced no retained events")
    return {
        "scene_weighted_brier": float(np.mean([np.mean(values) for values in per_scene_brier.values()])),
        "scene_weighted_nll": float(np.mean([np.mean(values) for values in per_scene_nll.values()])),
        "scene_count": len(per_scene_brier),
        "event_count": event_count,
    }


@torch.inference_mode()
def _prediction_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    per_scene: dict[str, list[tuple[float, float]]] = defaultdict(list)
    monotonicity_violations = 0
    for numpy_batch in loader:
        batch = _coordinate_tensor_batch(numpy_batch, device)
        probabilities = torch.sigmoid(_forward(model, batch)).cpu().numpy()
        query_mask = np.asarray(numpy_batch["query_mask"], dtype=bool)
        for batch_index, scene_id in enumerate(numpy_batch["scene_ids"]):
            for query_index in np.flatnonzero(query_mask[batch_index]):
                query_probabilities = probabilities[batch_index, query_index]
                monotonicity_violations += int(
                    np.sum(query_probabilities[1:] > query_probabilities[:-1] + 1e-12)
                )
                target = np.asarray(
                    numpy_batch["reachability_targets"][batch_index, query_index], dtype=np.float64
                )
                per_scene[str(scene_id)].append(
                    (float(np.mean((query_probabilities - target) ** 2)), float(np.mean(target)))
                )
                timestamp = str(numpy_batch["timestamps"][batch_index])
                candidate_index = int(
                    numpy_batch["candidate_indices"][batch_index, query_index]
                )
                for radius_index, radius in enumerate(RADII_CELLS):
                    rows.append(
                        {
                            "scene_id": str(scene_id),
                            "timestamp": timestamp,
                            "candidate_index": candidate_index,
                            "radius_cells": int(radius),
                            "probability": float(query_probabilities[radius_index]),
                            "target": bool(target[radius_index]),
                        }
                    )
    if not rows:
        raise RuntimeError("validation produced no prediction rows")
    predictions = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    per_scene_brier = [
        np.mean([value[0] for value in values]) for values in per_scene.values()
    ]
    epsilon = 1e-6
    clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
    nll = float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped)))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        selected = (predictions >= lower) & (
            (predictions < upper) if upper < 1.0 else (predictions <= upper)
        )
        if np.any(selected):
            ece += float(selected.mean()) * abs(
                float(predictions[selected].mean()) - float(targets[selected].mean())
            )
    high = predictions >= 0.8
    metrics = {
        "query_count": len(rows) // len(RADII_CELLS),
        "event_row_count": len(rows),
        "scene_count": len(per_scene),
        "scene_weighted_brier": float(np.mean(per_scene_brier)),
        "query_weighted_nll": nll,
        "query_weighted_ece": ece,
        "false_safe_rate@0.8": float(np.mean(targets[high] < 0.5)) if np.any(high) else 0.0,
        "high_confidence_coverage@0.8": float(high.mean()),
        "positive_prevalence": float(targets.mean()),
        "mean_prediction": float(predictions.mean()),
        "radius_monotonicity_violations": monotonicity_violations,
    }
    return rows, metrics


def _verify_manifest_geometry(
    records: list[dict[str, object]],
    *,
    raw_root: Path,
    label_root: Path,
    adapter: dict[str, object],
) -> None:
    """Check every replayed frame against the frozen manifest before training."""

    for record in records:
        frame = load_frame(
            str(record["timestamp"]),
            raw_root=raw_root,
            label_root=label_root,
            endpoint_policy=str(adapter["endpoint_policy"]),
            start_selection=str(adapter["start_selection"]),
            ground_margin_m=float(adapter["ground_margin_m"]),
            ground_bin_size_m=float(adapter["ground_bin_size_m"]),
            ground_lateral_limit_m=float(adapter["ground_lateral_limit_m"]),
            ground_quantile=float(adapter["ground_quantile"]),
        )
        queries = list(record["queries"])
        if len(frame.starts) != len(queries):
            raise RuntimeError(f"manifest/runtime query count mismatch: {record['timestamp']}")
        for index, query in enumerate(queries):
            if not np.array_equal(frame.starts[index], np.asarray(query["start"], dtype=np.int64)):
                raise RuntimeError(f"manifest/runtime start mismatch: {record['timestamp']}")
            if not np.array_equal(frame.goals[index], np.asarray(query["goal"], dtype=np.int64)):
                raise RuntimeError(f"manifest/runtime goal mismatch: {record['timestamp']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"),
    )
    parser.add_argument("--endpoint-policy", choices=("blocked", "ground"), default=None)
    parser.add_argument("--start-selection", choices=("valid", "observed_free"), default=None)
    parser.add_argument("--ground-margin-m", dest="ground_margin_m", type=float, default=None)
    parser.add_argument("--ground-bin-size-m", dest="ground_bin_size_m", type=float, default=None)
    parser.add_argument("--ground-lateral-limit-m", dest="ground_lateral_limit_m", type=float, default=None)
    parser.add_argument("--ground-quantile", dest="ground_quantile", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--train-frame-limit", type=int, default=None)
    parser.add_argument("--validation-frame-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.feature_channels != 16:
        raise SystemExit("protocol fixes --feature-channels=16")
    if args.batch_size != 4:
        raise SystemExit("protocol fixes --batch-size=4")
    if args.max_epochs < 1 or args.patience < 1:
        raise SystemExit("max-epochs and patience must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    adapter = _resolve_adapter_config(manifest, args)
    train_records = list(manifest["records"]["train"])
    validation_records = list(manifest["records"]["validation"])
    if args.train_frame_limit is not None:
        train_records = train_records[: args.train_frame_limit]
    if args.validation_frame_limit is not None:
        validation_records = validation_records[: args.validation_frame_limit]
    if not train_records or not validation_records:
        raise SystemExit("manifest train/validation records must be non-empty")
    # This is intentionally done before any optimizer step.  It makes a changed
    # adapter fail closed instead of silently changing the event contract.
    _verify_manifest_geometry(
        train_records + validation_records,
        raw_root=args.raw_root,
        label_root=args.label_root,
        adapter=adapter,
    )

    device = torch.device(args.device)
    data_generator = _seed_everything(args.seed, device)
    train_loader = DataLoader(
        UnScenesFrameDataset(train_records, args.raw_root, args.label_root, adapter),
        batch_size=args.batch_size,
        shuffle=True,
        generator=data_generator,
        num_workers=0,
        collate_fn=_coordinate_collate,
    )
    validation_loader = DataLoader(
        UnScenesFrameDataset(validation_records, args.raw_root, args.label_root, adapter),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_coordinate_collate,
    )
    model = S4CInspiredCoordinateBaseline(
        input_channels=3, feature_channels=args.feature_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    progress_path = args.output_dir / "progress.jsonl"
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = 0
    best_brier = float("inf")
    patience_used = 0
    started = time.monotonic()
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
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
        data_generator.set_state(state["data_generator_state"].cpu())

    try:
        for epoch in range(start_epoch, args.max_epochs + 1):
            model.train()
            losses: list[float] = []
            epoch_started = time.monotonic()
            for numpy_batch in train_loader:
                batch = _coordinate_tensor_batch(numpy_batch, device)
                if not torch.any(batch["query_mask"]):
                    continue
                logits = _forward(model, batch)
                loss = _scene_mean_loss(
                    logits,
                    batch["reachability_targets"],
                    batch["query_mask"],
                    [str(value) for value in numpy_batch["scene_ids"]],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise RuntimeError("training epoch contained no retained events")
            validation = _validation_summary(model, validation_loader, device)
            current_brier = float(validation["scene_weighted_brier"])
            improved = current_brier < best_brier - args.min_delta
            if improved:
                best_brier = current_brier
                best_epoch = epoch
                patience_used = 0
            else:
                patience_used += 1
            record = {
                "epoch": epoch,
                "train_event_bce": float(np.mean(losses)),
                **validation,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_scene_weighted_brier": best_brier,
                "patience_used": patience_used,
                "epoch_seconds": time.monotonic() - epoch_started,
                "elapsed_seconds": time.monotonic() - started,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            history.append(record)
            config = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            config["adapter"] = adapter
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_brier": best_brier,
                "patience_used": patience_used,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "data_generator_state": data_generator.get_state(),
                "config": config,
            }
            _atomic_torch_save(latest_path, payload)
            if improved:
                _atomic_torch_save(best_path, payload)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(record, allow_nan=False), flush=True)
            if patience_used >= args.patience:
                break
    except KeyboardInterrupt:
        raise

    if not best_path.exists():
        raise RuntimeError("no best checkpoint was produced")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    rows, event_metrics = _prediction_rows(model, validation_loader, device)
    prediction_path = args.output_dir / "predictions_validation.csv"
    import csv

    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            # Keep the exported prediction artifact label-free; targets are used only
            # in-memory for the validation report.
            fieldnames=("scene_id", "timestamp", "candidate_index", "radius_cells", "probability"),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    run = {
        "schema_version": 1,
        "kind": "unscenes3d_s4c_inspired_coordinate_training",
        "paper_result": False,
        "validation_result": True,
        "protocol_version": PROTOCOL_VERSION,
        "test_evaluated": False,
        "manifest": {
            "path": str(args.manifest),
            "sha256": _sha256(args.manifest),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "validation_queries": sum(len(record["queries"]) for record in validation_records),
            "test_locked_sites": manifest.get("test_locked_sites", []),
        },
        "adapter": adapter,
        "architecture": {
            "label": "s4c_inspired_coordinate",
            "feature_channels": args.feature_channels,
            "bilinear_coordinate_queries": True,
            "fourier_frequencies": [1.0, 2.0, 4.0, 8.0],
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selection": {
            "criterion": "minimum validation scene-weighted event Brier",
            "best_epoch": int(selected["best_epoch"]),
            "best_scene_weighted_brier_during_training": float(selected["best_brier"]),
            "epochs_completed": len(history),
        },
        "event_metrics": event_metrics,
        "artifacts": {
            "best_checkpoint": {"path": str(best_path), "sha256": _sha256(best_path)},
            "latest_checkpoint": {"path": str(latest_path), "sha256": _sha256(latest_path)},
            "predictions_validation": {
                "path": str(prediction_path),
                "rows": len(rows),
                "sha256": _sha256(prediction_path),
            },
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "argv": sys.argv,
        },
        "history": history,
        "claim_boundary": (
            "Validation-only S4C-inspired coordinate-query control on the UnScenes3D adapter; "
            "this is not a reproduction of an original 3-D S4C system, location_6 remains locked, "
            "and no final cross-domain claim is made."
        ),
    }
    _atomic_json(args.output_dir / "run.json", run)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "event_metrics": event_metrics, "test_evaluated": False},
            indent=2,
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
