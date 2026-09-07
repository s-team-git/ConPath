#!/usr/bin/env python3
"""Evaluate a UnScenes3D ConPath checkpoint as a deterministic mean-map control.

The evaluator is validation-only.  It averages Rao--Blackwellized posterior free-cell
probabilities over a fixed number of stochastic worlds, thresholds one map at 0.5, and
passes that map through the exact bottleneck/clearance oracle used by the manifest.
The adapter configuration is read from the manifest so the input and query contracts
cannot silently diverge.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.labels import clearance_radius_map, maximum_clearance_map  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402
from pathrel.unscenes3d import load_frame  # noqa: E402


PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.2"
VALID_SUPPORT_POLICY = (
    "UnScenes3D target_valid complement is deterministically blocked before posterior sampling"
)
RADII_CELLS = (0, 1, 2)
DEFAULT_ADAPTER: dict[str, object] = {
    "endpoint_policy": "blocked",
    "start_selection": "valid",
    "ground_margin_m": 0.35,
    "ground_bin_size_m": 1.2,
    "ground_lateral_limit_m": 20.0,
    "ground_quantile": 0.15,
}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_adapter(manifest: dict[str, object]) -> dict[str, object]:
    value = manifest.get("adapter", {})
    if not isinstance(value, dict):
        raise SystemExit("manifest adapter metadata must be an object")
    result: dict[str, object] = {}
    for key, default in DEFAULT_ADAPTER.items():
        result[key] = value.get(key, default)
    return {
        "endpoint_policy": str(result["endpoint_policy"]),
        "start_selection": str(result["start_selection"]),
        "ground_margin_m": float(result["ground_margin_m"]),
        "ground_bin_size_m": float(result["ground_bin_size_m"]),
        "ground_lateral_limit_m": float(result["ground_lateral_limit_m"]),
        "ground_quantile": float(result["ground_quantile"]),
    }


def _load_frame(
    timestamp: str,
    *,
    raw_root: Path,
    label_root: Path,
    adapter: dict[str, object],
) -> Any:
    return load_frame(
        timestamp,
        raw_root=raw_root,
        label_root=label_root,
        endpoint_policy=str(adapter["endpoint_policy"]),
        start_selection=str(adapter["start_selection"]),
        ground_margin_m=float(adapter["ground_margin_m"]),
        ground_bin_size_m=float(adapter["ground_bin_size_m"]),
        ground_lateral_limit_m=float(adapter["ground_lateral_limit_m"]),
        ground_quantile=float(adapter["ground_quantile"]),
    )


def _seed(seed: int, device: torch.device) -> torch.Generator:
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


def _event_metrics(
    rows: list[dict[str, object]],
    targets: dict[tuple[str, str, int, int], bool],
) -> dict[str, object]:
    if not rows:
        raise RuntimeError("no validation event rows were produced")
    per_scene: dict[str, list[float]] = defaultdict(list)
    predictions: list[float] = []
    labels: list[float] = []
    for row in rows:
        key = (str(row["scene_id"]), str(row["timestamp"]), int(row["candidate_index"]), int(row["radius_cells"]))
        prediction = float(row["probability"])
        target = float(targets[key])
        predictions.append(prediction)
        labels.append(target)
        per_scene[str(row["scene_id"])].append((prediction - target) ** 2)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    epsilon = 1e-6
    clipped = np.clip(prediction_array, epsilon, 1.0 - epsilon)
    nll = float(-np.mean(label_array * np.log(clipped) + (1.0 - label_array) * np.log1p(-clipped)))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        selected = (prediction_array >= lower) & (
            (prediction_array < upper) if upper < 1.0 else (prediction_array <= upper)
        )
        if np.any(selected):
            ece += float(selected.mean()) * abs(
                float(prediction_array[selected].mean()) - float(label_array[selected].mean())
            )
    high = prediction_array >= 0.8
    return {
        "query_count": int(len(rows) // len(RADII_CELLS)),
        "event_row_count": len(rows),
        "scene_count": len(per_scene),
        "scene_weighted_brier": float(np.mean([np.mean(values) for values in per_scene.values()])),
        "query_weighted_nll": nll,
        "query_weighted_ece": ece,
        "false_safe_rate@0.8": float(np.mean(label_array[high] < 0.5)) if np.any(high) else 0.0,
        "high_confidence_coverage@0.8": float(high.mean()),
        "positive_prevalence": float(label_array.mean()),
        "mean_prediction": float(prediction_array.mean()),
        "radius_monotonicity_violations": 0,
    }


def _map_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("no hidden validation cells were produced")
    per_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_scene[str(row["scene_id"])].append(row)
    return {
        "scene_weighted_brier": float(
            np.mean([np.mean([float(row["brier"]) for row in values]) for values in per_scene.values()])
        ),
        "scene_weighted_nll": float(
            np.mean([np.mean([float(row["nll"]) for row in values]) for values in per_scene.values()])
        ),
        "scene_weighted_positive_rate": float(
            np.mean(
                [np.mean([float(row["positive_rate"]) for row in values]) for values in per_scene.values()]
            )
        ),
        "scene_weighted_mean_probability": float(
            np.mean(
                [np.mean([float(row["mean_probability"]) for row in values]) for values in per_scene.values()]
            )
        ),
        "scene_count": len(per_scene),
        "frame_count": len(rows),
        "cell_count": int(sum(int(row["cell_count"]) for row in rows)),
    }


def _validate_record_geometry(record: dict[str, object], frame: Any) -> None:
    """Ensure the replayed adapter produces the frozen manifest query geometry."""

    queries = list(record["queries"])
    if len(frame.starts) != len(queries) or len(frame.goals) != len(queries):
        raise RuntimeError(
            f"adapter/manifest query count mismatch for {record['timestamp']}: "
            f"runtime={len(frame.starts)} manifest={len(queries)}"
        )
    for index, query in enumerate(queries):
        expected_start = np.asarray(query["start"], dtype=np.int64)
        expected_goal = np.asarray(query["goal"], dtype=np.int64)
        if not np.array_equal(frame.starts[index], expected_start) or not np.array_equal(
            frame.goals[index], expected_goal
        ):
            raise RuntimeError(
                f"adapter/manifest query geometry mismatch for {record['timestamp']} "
                f"candidate_index={query['candidate_index']}"
            )


def evaluate(
    model: PathRelNet,
    records: list[dict[str, object]],
    *,
    raw_root: Path,
    label_root: Path,
    adapter: dict[str, object],
    device: torch.device,
    posterior_samples: int,
    sample_chunk: int,
    generator: torch.Generator,
    disable_global_factors: bool,
) -> tuple[list[dict[str, object]], dict[str, object], dict[tuple[str, str, int, int], bool]]:
    model.eval()
    event_rows: list[dict[str, object]] = []
    targets: dict[tuple[str, str, int, int], bool] = {}
    map_rows: list[dict[str, object]] = []
    for record in records:
        timestamp = str(record["timestamp"])
        frame = _load_frame(timestamp, raw_root=raw_root, label_root=label_root, adapter=adapter)
        _validate_record_geometry(record, frame)
        observation = torch.from_numpy(frame.input_bev[None]).to(device=device)
        valid_support_mask = torch.from_numpy(frame.target_valid[None]).to(
            device=device, dtype=torch.bool
        )
        probability_sum: torch.Tensor | None = None
        remaining = posterior_samples
        while remaining > 0:
            current = min(sample_chunk, remaining)
            output = model(
                observation,
                valid_support_mask=valid_support_mask,
                num_samples=current,
                hard_samples=True,
                disable_global_factors=disable_global_factors,
                generator=generator,
            )
            current_probability = output.posterior.posterior_marginal_probs[:, 0]
            probability_sum = (
                current_probability * current
                if probability_sum is None
                else probability_sum + current_probability * current
            )
            remaining -= current
        assert probability_sum is not None
        probability = (probability_sum / float(posterior_samples))[0].detach().cpu().numpy()
        unknown = frame.input_bev[2] > 0.5
        hidden = frame.target_valid & unknown
        target = frame.target_free.astype(np.float64, copy=False)
        if np.any(hidden):
            selected_target = target[hidden]
            selected_probability = probability[hidden].astype(np.float64)
            clipped = np.clip(selected_probability, 1e-6, 1.0 - 1e-6)
            map_rows.append(
                {
                    "scene_id": str(record["scene_id"]),
                    "brier": float(np.mean((selected_probability - selected_target) ** 2)),
                    "nll": float(-np.mean(selected_target * np.log(clipped) + (1.0 - selected_target) * np.log1p(-clipped))),
                    "positive_rate": float(np.mean(selected_target)),
                    "mean_probability": float(np.mean(selected_probability)),
                    "cell_count": int(hidden.sum()),
                }
            )

        deterministic_map = probability >= 0.5
        deterministic_map[frame.input_bev[1] > 0.5] = False
        deterministic_map[frame.input_bev[0] > 0.5] = True
        clearance = clearance_radius_map(deterministic_map)
        query_rows = list(record["queries"])
        if query_rows:
            starts = tuple(int(value) for value in query_rows[0]["start"])
            goals = [tuple(int(value) for value in row["goal"]) for row in query_rows]
            bottleneck = maximum_clearance_map(
                deterministic_map,
                starts,
                clearance=clearance,
                stop_points=goals,
            )
            values = bottleneck[np.asarray([goal[0] for goal in goals]), np.asarray([goal[1] for goal in goals])]
            for query_index, row in enumerate(query_rows):
                candidate_index = int(row["candidate_index"])
                for radius_index, radius in enumerate(RADII_CELLS):
                    key = (str(record["scene_id"]), timestamp, candidate_index, int(radius))
                    targets[key] = bool(row["reachable"][radius_index])
                    event_rows.append(
                        {
                            "scene_id": str(record["scene_id"]),
                            "timestamp": timestamp,
                            "candidate_index": candidate_index,
                            "radius_cells": int(radius),
                            "probability": float(values[query_index] >= radius),
                        }
                    )
    return event_rows, _map_metrics(map_rows), targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--sample-chunk", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_samples < 2 or args.sample_chunk < 2:
        raise SystemExit("validation-samples and sample-chunk must both be at least two")
    if args.validation_samples % args.sample_chunk:
        raise SystemExit("validation-samples must be divisible by sample-chunk")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; pass --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    adapter = _resolve_adapter(manifest)
    records = list(manifest["records"]["validation"])
    if not records:
        raise SystemExit("manifest validation records are empty")
    device = torch.device(args.device)
    generator = _seed(args.seed, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    checkpoint_trained_with_support = (
        config.get("valid_support_policy") == VALID_SUPPORT_POLICY
    )
    variant = str(config.get("decoder_variant", "correlated"))
    if variant not in {"correlated", "independent"}:
        raise SystemExit(f"unsupported decoder_variant in checkpoint: {variant!r}")
    independent = variant == "independent"
    feature_channels = int(config.get("feature_channels", 16))
    latent_dim = int(config.get("latent_dim", 4))
    model = PathRelNet(
        feature_channels=feature_channels,
        latent_dim=latent_dim,
        local_kernel_size=(1 if independent else 5),
    ).to(device)
    model.load_state_dict(state["model"])
    event_rows, map_metrics, targets = evaluate(
        model,
        records,
        raw_root=args.raw_root,
        label_root=args.label_root,
        adapter=adapter,
        device=device,
        posterior_samples=args.validation_samples,
        sample_chunk=args.sample_chunk,
        generator=generator,
        disable_global_factors=independent,
    )
    prediction_path = args.output_dir / "predictions_validation.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scene_id", "timestamp", "candidate_index", "radius_cells", "probability"),
        )
        writer.writeheader()
        writer.writerows(event_rows)
    metrics = _event_metrics(event_rows, targets)
    report = {
        "schema_version": 1,
        "kind": "unscenes3d_conpath_deterministic_mean_map",
        "paper_result": False,
        "validation_result": True,
        "protocol_version": PROTOCOL_VERSION,
        "test_evaluated": False,
        "posthoc_checkpoint_evaluation": not checkpoint_trained_with_support,
        "retraining_required": not checkpoint_trained_with_support,
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "manifest": {
            "path": str(args.manifest),
            "sha256": _sha256(args.manifest),
            "validation_records": len(records),
            "validation_queries": sum(len(record["queries"]) for record in records),
            "test_locked_sites": manifest.get("test_locked_sites", []),
        },
        "adapter": adapter,
        "decoder": {"variant": variant, "feature_channels": feature_channels, "latent_dim": latent_dim, "local_kernel_size": 1 if independent else 5, "disable_global_factors": independent},
        "forward": {
            "invalid_support_clamped": True,
            "valid_support_policy": VALID_SUPPORT_POLICY,
            "checkpoint_trained_with_same_support_policy": checkpoint_trained_with_support,
        },
        "mean_map": {
            "posterior_samples": args.validation_samples,
            "sample_chunk": args.sample_chunk,
            "evaluation_seed": args.seed,
            "threshold": 0.5,
            "metrics": map_metrics,
        },
        "prediction": {"path": str(prediction_path), "rows": len(event_rows), "sha256": _sha256(prediction_path)},
        "event_metrics": metrics,
        "runtime": {"wall_seconds": time.monotonic() - started, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None},
        "claim_boundary": (
            "Validation-only post-hoc support-clamped deterministic mean-map diagnostic; the "
            "checkpoint predates this support policy and requires clean retraining; location_6 "
            "remains locked."
            if not checkpoint_trained_with_support
            else "Validation-only deterministic threshold of a support-consistent UnScenes3D "
            "posterior mean map; location_6 remains locked and this is not a final paper claim."
        ),
    }
    _atomic_json(args.output_dir / "map_metrics.json", map_metrics)
    _atomic_json(args.output_dir / "run.json", report)
    print(json.dumps({"output_dir": str(args.output_dir), "event_metrics": metrics, "map_metrics": map_metrics, "test_evaluated": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
