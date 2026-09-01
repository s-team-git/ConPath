#!/usr/bin/env python3
"""Train ConPath on the frozen FlatLands provenance split.

This is the public-data neural path. It uses the canonical three-channel replay adapter, derives
the event from correlated stochastic map samples, and writes only label-free prediction manifests.
The default pilot uses a bounded differentiable propagation budget; final paper runs must validate
the exact-forward operator against the merge-tree contract before claiming exact connectivity.
"""

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
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.flatlands_data import FlatLandsReplayDataset, collate_flatlands_replay  # noqa: E402
from pathrel.flatlands_eval import (  # noqa: E402
    evaluate_flatlands_prediction_file,
    write_evaluation_report,
    write_prediction_manifest,
)
from pathrel.flatlands_query import sha256_path  # noqa: E402
from pathrel.gpu_diagnostics import cuda_unavailable_message  # noqa: E402
from pathrel.labels import batched_merge_tree_bottleneck_scores, clearance_radius_map  # noqa: E402
from pathrel.losses import (  # noqa: E402
    posterior_marginal_nll,
    reachability_brier,
    reachability_brier_u_statistic,
    spatial_variogram_score,
)
from pathrel.model import PathRelNet  # noqa: E402


PROTOCOL_VERSION = "P1_BASELINE_PROTOCOL.md v1 + ConPath pilot v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"))
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/p1_flatlands_conpath_seed20260831"))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-channels", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument(
        "--decoder-variant",
        choices=("correlated", "independent"),
        default="correlated",
        help="correlated ConPath posterior or an independent-cell local-noise control",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument(
        "--validation-sample-chunk",
        type=int,
        default=None,
        help="split validation posterior samples into sequential chunks to cap GPU memory; total samples stay unchanged",
    )
    parser.add_argument("--max-reachability-steps", type=int, default=256)
    parser.add_argument("--selection-query-limit", type=int, default=8)
    parser.add_argument("--map-weight", type=float, default=1.0)
    parser.add_argument("--variogram-weight", type=float, default=0.1)
    parser.add_argument("--reachability-weight", type=float, default=2.0)
    parser.add_argument("--disable-global-factors", action="store_true", help="ablation: remove the low-rank spatially global correlation term")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--train-scenes-limit", type=int, default=None, help="debug-only subset; disables paper-style full validation")
    parser.add_argument("--validation-scenes-limit", type=int, default=None, help="debug-only subset; disables exact prediction-manifest evaluation")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(("git", *arguments), check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
        return result.stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--short").splitlines()}


def _seed_everything(seed: int, device: torch.device) -> tuple[torch.Generator, torch.Generator]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 11)
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(seed + 17)
    return loader_generator, sample_generator


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


def _cache_split(archive: Path, selection: Path, queries: Path, split: str, limit: int | None) -> tuple[list[object], tuple[int, ...]]:
    dataset = FlatLandsReplayDataset(archive, selection, queries, split=split, verify_frozen=True, verify_query_geometry=True)
    try:
        samples = [dataset[index] for index in range(len(dataset))]
        if limit is not None:
            if limit < 1:
                raise ValueError("scene limits must be positive")
            samples = samples[:limit]
        return samples, dataset.radii_cells
    finally:
        dataset.close()


def _to_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output = dict(batch)
    for key in ("observation", "target_free", "loss_mask", "starts", "goals", "reachability_targets", "query_mask"):
        dtype = torch.bool if key in ("loss_mask", "reachability_targets", "query_mask") else None
        tensor = torch.from_numpy(batch[key])
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        output[key] = tensor.to(device=device, non_blocking=True)
    return output


def _limit_queries(batch: dict[str, object], limit: int | None) -> dict[str, object]:
    if limit is None:
        return batch
    if limit < 1:
        raise ValueError("selection-query-limit must be positive")
    query_count = min(int(batch["starts"].shape[1]), limit)
    output = dict(batch)
    for key in ("starts", "goals", "reachability_targets", "query_mask", "distances_m", "angles_deg", "candidate_indices"):
        if key in output:
            output[key] = output[key][:, :query_count]
    return output


def _hidden_target_classes(batch: dict[str, object]) -> Tensor:
    target_free = batch["target_free"].to(torch.bool)
    hidden = batch["loss_mask"].to(torch.bool)
    target = torch.full(target_free.shape, -1, dtype=torch.long, device=target_free.device)
    return torch.where(hidden, (~target_free).to(torch.long), target)


def _scene_mean_hidden_loss(logits: Tensor, target_classes: Tensor) -> Tensor:
    valid = target_classes >= 0
    per_cell = F.cross_entropy(logits, target_classes.clamp_min(0), reduction="none")
    counts = valid.sum(dim=(1, 2))
    selected = counts > 0
    if not torch.any(selected):
        return logits.sum() * 0.0
    return (per_cell * valid).sum(dim=(1, 2))[selected].div(counts[selected]).mean()


def _scene_event_brier(events: Tensor, targets: Tensor, query_mask: Tensor) -> Tensor:
    valid = query_mask[..., None].expand_as(targets)
    error = (events - targets.to(events.dtype)).square()
    per_scene_count = valid.sum(dim=(1, 2))
    selected = per_scene_count > 0
    if not torch.any(selected):
        return events.sum() * 0.0
    per_scene = (error * valid).sum(dim=(1, 2))[selected].div(per_scene_count[selected])
    return per_scene.mean()


def _forward(model: PathRelNet, batch: dict[str, object], radii: tuple[int, ...], samples: int, max_steps: int, generator: torch.Generator, *, disable_global_factors: bool = False) -> Any:
    return model(
        batch["observation"],
        starts=batch["starts"],
        goals=batch["goals"],
        footprint_radii_cells=radii,
        num_samples=samples,
        hard_samples=True,
        max_reachability_steps=max_steps,
        shared_start=True,
        disable_global_factors=disable_global_factors,
        generator=generator,
    )


def _exact_event_probabilities(
    safe_samples: Tensor,
    starts: np.ndarray,
    goals: np.ndarray,
    radii: tuple[int, ...],
) -> np.ndarray:
    """Evaluate hard posterior worlds with the exact NumPy clearance/merge-tree oracle."""

    safe = safe_samples.detach().cpu().numpy() > 0.5
    scores = np.stack(
        [
            np.stack([clearance_radius_map(world) for world in batch_worlds])
            for batch_worlds in safe
        ]
    ).astype(np.float64, copy=False)
    bottleneck = batched_merge_tree_bottleneck_scores(scores, starts, goals)
    return np.stack([(bottleneck >= radius).mean(axis=1) for radius in radii], axis=-1)


@torch.inference_mode()
def _validation_selection_score(model: PathRelNet, loader: DataLoader, device: torch.device, radii: tuple[int, ...], samples: int, max_steps: int, query_limit: int | None, generator: torch.Generator, *, disable_global_factors: bool = False, sample_chunk: int | None = None) -> float:
    model.eval()
    values: list[float] = []
    for numpy_batch in loader:
        batch = _limit_queries(_to_tensor_batch(numpy_batch, device), query_limit)
        if batch["starts"].shape[1] == 0:
            continue
        chunk = samples if sample_chunk is None else min(sample_chunk, samples)
        event_sum = None
        remaining = samples
        while remaining > 0:
            current = min(chunk, remaining)
            output = _forward(model, batch, radii, current, max_steps, generator, disable_global_factors=disable_global_factors)
            if output.sample_reachability is None:
                raise RuntimeError("validation forward did not return sample reachability")
            current_sum = output.sample_reachability.sum(dim=1)
            event_sum = current_sum if event_sum is None else event_sum + current_sum
            remaining -= current
        assert event_sum is not None
        values.append(float(_scene_event_brier(event_sum / float(samples), batch["reachability_targets"], batch["query_mask"]).cpu()))
    if not values:
        raise RuntimeError("validation produced no retained queries")
    return float(np.mean(values))


@torch.inference_mode()
def _validation_prediction_rows(model: PathRelNet, loader: DataLoader, device: torch.device, radii: tuple[int, ...], samples: int, max_steps: int, generator: torch.Generator, *, exact_forward: bool = True, disable_global_factors: bool = False, sample_chunk: int | None = None) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for numpy_batch in loader:
        batch = _to_tensor_batch(numpy_batch, device)
        if batch["starts"].shape[1] == 0:
            continue
        chunk = samples if sample_chunk is None else min(sample_chunk, samples)
        probability_sum = None
        remaining = samples
        while remaining > 0:
            current = min(chunk, remaining)
            output = _forward(model, batch, radii, current, max_steps, generator, disable_global_factors=disable_global_factors)
            if exact_forward:
                current_probability = _exact_event_probabilities(
                    output.posterior.safe_samples(),
                    numpy_batch["starts"],
                    numpy_batch["goals"],
                    radii,
                )
            else:
                current_probability = output.reachability.detach().cpu().numpy()
            probability_sum = current_probability * current if probability_sum is None else probability_sum + current_probability * current
            remaining -= current
        assert probability_sum is not None
        probabilities = probability_sum / float(samples)
        mask = numpy_batch["query_mask"]
        for batch_index, global_id in enumerate(numpy_batch["global_ids"]):
            for query_index in np.flatnonzero(mask[batch_index]):
                candidate_index = int(numpy_batch["candidate_indices"][batch_index, query_index])
                for radius_index, radius in enumerate(radii):
                    rows.append({
                        "global_id": global_id,
                        "candidate_index": candidate_index,
                        "radius_cells": radius,
                        "probability": float(probabilities[batch_index, query_index, radius_index]),
                    })
    if not rows:
        raise RuntimeError("validation produced no prediction rows")
    return rows


def _checkpoint_payload(model: PathRelNet, optimizer: torch.optim.Optimizer, epoch: int, best_epoch: int, best_score: float, patience_used: int, history: Sequence[dict[str, object]], loader_generator: torch.Generator, sample_generator: torch.Generator, args: argparse.Namespace) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "patience_used": patience_used,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": list(history),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator_state": loader_generator.get_state(),
        "sample_generator_state": sample_generator.get_state(),
        "config": vars(args),
    }


def main() -> None:
    args = parse_args()
    if args.train_samples < 2 or args.validation_samples < 2:
        raise SystemExit("train/validation samples must be at least two")
    if args.max_epochs < 1 or args.patience < 1 or args.max_reachability_steps < 1:
        raise SystemExit("epochs, patience, and reachability steps must be positive")
    if args.validation_sample_chunk is not None and args.validation_sample_chunk < 1:
        raise SystemExit("validation-sample-chunk must be positive when supplied")
    if args.batch_size < 1 or args.feature_channels < 1 or args.latent_dim < 1:
        raise SystemExit("batch-size, feature-channels, and latent-dim must be positive")
    if args.decoder_variant == "independent" and args.disable_global_factors:
        raise SystemExit("independent decoder already disables global factors; do not combine the flags")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_unavailable_message(torch))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; pass --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    loader_generator, sample_generator = _seed_everything(args.seed, device)
    started = time.monotonic()

    print("Caching frozen train/validation packets directly from ZIP...", flush=True)
    train_samples, train_radii = _cache_split(args.archive, args.selection, args.queries, "train", args.train_scenes_limit)
    validation_samples, validation_radii = _cache_split(args.archive, args.selection, args.queries, "validation", args.validation_scenes_limit)
    if train_radii != validation_radii:
        raise RuntimeError("train/validation radii differ")
    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, generator=loader_generator, collate_fn=collate_flatlands_replay, num_workers=0)
    validation_loader = DataLoader(validation_samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate_flatlands_replay, num_workers=0)

    independent_decoder = args.decoder_variant == "independent"
    model = PathRelNet(
        input_channels=3,
        feature_channels=args.feature_channels,
        latent_dim=args.latent_dim,
        local_kernel_size=(1 if independent_decoder else 5),
    ).to(device)
    effective_disable_global_factors = bool(args.disable_global_factors or independent_decoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, object]] = []
    start_epoch, best_epoch, best_score, patience_used = 1, 0, float("inf"), 0
    latest_path, best_path = args.output_dir / "latest.pt", args.output_dir / "best.pt"
    if args.resume:
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch, best_score, patience_used = int(state["best_epoch"]), float(state["best_score"]), int(state["patience_used"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
        loader_generator.set_state(state["loader_generator_state"].cpu())
        sample_generator.set_state(state["sample_generator_state"].cpu())

    try:
        for epoch in range(start_epoch, args.max_epochs + 1):
            model.train()
            losses: list[float] = []
            epoch_started = time.monotonic()
            for numpy_batch in train_loader:
                batch = _limit_queries(_to_tensor_batch(numpy_batch, device), args.selection_query_limit)
                if batch["starts"].shape[1] == 0:
                    continue
                output = _forward(model, batch, train_radii, args.train_samples, args.max_reachability_steps, sample_generator, disable_global_factors=effective_disable_global_factors)
                target_classes = _hidden_target_classes(batch)
                map_loss = posterior_marginal_nll(output.posterior.sample_logits, target_classes)
                target_safe = batch["target_free"].to(torch.float32)
                variogram_loss = spatial_variogram_score(output.posterior.safe_samples(), target_safe, valid_mask=batch["loss_mask"])
                weights = batch["query_mask"][..., None].expand_as(batch["reachability_targets"])
                reach_loss = reachability_brier_u_statistic(output.sample_reachability, batch["reachability_targets"], weights=weights)
                total = args.map_weight * map_loss + args.variogram_weight * variogram_loss + args.reachability_weight * reach_loss
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                losses.append(float(total.detach().cpu()))
            validation_score = _validation_selection_score(model, validation_loader, device, validation_radii, args.validation_samples, args.max_reachability_steps, args.selection_query_limit, sample_generator, disable_global_factors=effective_disable_global_factors, sample_chunk=args.validation_sample_chunk)
            improved = validation_score < best_score - args.min_delta
            if improved:
                best_score, best_epoch, patience_used = validation_score, epoch, 0
            else:
                patience_used += 1
            record = {
                "epoch": epoch,
                "train_total": float(np.mean(losses)) if losses else None,
                "validation_scene_weighted_brier": validation_score,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_validation_scene_weighted_brier": best_score,
                "patience_used": patience_used,
                "epoch_seconds": time.monotonic() - epoch_started,
                "elapsed_seconds": time.monotonic() - started,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            history.append(record)
            payload = _checkpoint_payload(model, optimizer, epoch, best_epoch, best_score, patience_used, history, loader_generator, sample_generator, args)
            _atomic_torch_save(latest_path, payload)
            if improved:
                _atomic_torch_save(best_path, payload)
            _append_progress(args.output_dir / "progress.jsonl", record)
            print(json.dumps(record, allow_nan=False), flush=True)
            if patience_used >= args.patience:
                break
    except KeyboardInterrupt:
        if history:
            _atomic_torch_save(args.output_dir / "interrupted.pt", _checkpoint_payload(model, optimizer, int(history[-1]["epoch"]), best_epoch, best_score, patience_used, history, loader_generator, sample_generator, args))
        raise

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    rows = _validation_prediction_rows(model, validation_loader, device, validation_radii, args.validation_samples, args.max_reachability_steps, sample_generator, exact_forward=True, disable_global_factors=effective_disable_global_factors, sample_chunk=args.validation_sample_chunk)
    prediction_path = args.output_dir / "predictions_validation.csv"
    prediction_count = write_prediction_manifest(prediction_path, rows)
    full_validation = args.train_scenes_limit is None and args.validation_scenes_limit is None
    evaluation: dict[str, object] | None = None
    if full_validation:
        evaluation = evaluate_flatlands_prediction_file(prediction_path, args.selection, args.queries, method="conpath", split="validation", bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        write_evaluation_report(args.output_dir / "evaluation_validation", evaluation)

    total_seconds = time.monotonic() - started
    report = {
        "schema_version": 1,
        "kind": "flatlands_conpath_training",
        "paper_result": False,
        "validation_result": bool(full_validation),
        "protocol_version": PROTOCOL_VERSION,
        "test_evaluated": False,
        "git": _git_state(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "train_scenes": len(train_samples),
            "validation_scenes": len(validation_samples),
            "selection_sha256": sha256_path(args.selection),
            "queries_sha256": sha256_path(args.queries),
            "archive_bytes": args.archive.stat().st_size,
        },
        "selection": {
            "criterion": "minimum validation scene-weighted event Brier on target-blind fixed queries",
            "best_epoch": int(selected["best_epoch"]),
            "best_validation_scene_weighted_brier": float(selected["best_score"]),
            "epochs_completed": len(history),
        },
        "prediction": {"path": str(prediction_path), "rows": prediction_count, "sha256": sha256_path(prediction_path)},
        "forward": {
            "training_event_operator": "shared-start differentiable max-min propagation",
            "validation_prediction_operator": "exact NumPy disk-clearance + batched Kruskal merge-tree",
            "validation_exact_forward": True,
            "training_max_reachability_steps": args.max_reachability_steps,
            "decoder_variant": args.decoder_variant,
            "effective_disable_global_factors": effective_disable_global_factors,
            "local_kernel_size": 1 if independent_decoder else 5,
            "validation_sample_chunk": args.validation_sample_chunk,
        },
        "evaluation": evaluation,
        "runtime": {
            "wall_seconds": total_seconds,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "argv": sys.argv},
        "history": history,
        "claim_boundary": (
            f"Single-seed {args.decoder_variant} decoder validation run. The differentiable event uses a bounded "
            f"shared-start propagation budget of {args.max_reachability_steps} steps; test is locked and this is not a final paper result."
        ),
    }
    _atomic_json(args.output_dir / "run.json", report)
    print(json.dumps({"output_dir": str(args.output_dir), "validation_result": full_validation, "prediction_rows": prediction_count, "evaluation_brier": None if evaluation is None else evaluation["metrics"][0]["scene_weighted"]["brier"], "paper_result": False}, indent=2))


if __name__ == "__main__":
    main()
