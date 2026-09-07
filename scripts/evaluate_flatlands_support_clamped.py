#!/usr/bin/env python3
"""Re-evaluate one FlatLands checkpoint with invalid support clamped as blocked.

This is a validation-only recovery evaluator.  It never accepts a split argument and therefore
cannot open the frozen test labels.  Posterior worlds are generated without the differentiable
reachability operator, then scored with a SciPy-accelerated implementation of the same discrete
disk and four-neighbour connectivity contract.  The accelerator is checked against the canonical
NumPy clearance/merge-tree oracle before any checkpoint is loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

try:
    from scipy import ndimage
except ImportError as error:  # pragma: no cover - depends on the execution environment
    raise SystemExit(
        "scipy is required only for this accelerated recovery evaluator; install scipy>=1.10"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.flatlands_data import FlatLandsReplayDataset  # noqa: E402
from pathrel.flatlands_eval import (  # noqa: E402
    evaluate_flatlands_prediction_file,
    write_evaluation_report,
    write_prediction_manifest,
)
from pathrel.flatlands_query import sha256_path  # noqa: E402
from pathrel.gpu_diagnostics import cuda_unavailable_message  # noqa: E402
from pathrel.labels import (  # noqa: E402
    batched_merge_tree_bottleneck_scores,
    clearance_radius_map,
)
from pathrel.model import PathRelNet  # noqa: E402


FOUR_NEIGHBOUR_STRUCTURE = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--decoder-variant", choices=("correlated", "independent"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sample-chunk", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "results/p1_flatlands_query_audit_bounded/selected_observations.csv"
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"),
    )
    parser.add_argument(
        "--validation-scenes-limit",
        type=int,
        default=None,
        help="debug-only prefix; a limited run is not passed to the metric evaluator",
    )
    return parser.parse_args()


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


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "status": run("status", "--short").splitlines(),
    }


def _accelerated_clearance(world: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=bool)
    if world.ndim != 2 or not world.size:
        raise ValueError("world must be a non-empty [H,W] map")
    distance = ndimage.distance_transform_edt(
        np.pad(world, 1, mode="constant", constant_values=False)
    )[1:-1, 1:-1]
    clearance = np.ceil(distance).astype(np.int64) - 1
    clearance[~world] = -1
    return clearance


def _accelerated_events(
    worlds: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    radii: tuple[int, ...],
) -> np.ndarray:
    worlds = np.asarray(worlds, dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    goals = np.asarray(goals, dtype=np.int64)
    if worlds.ndim != 3:
        raise ValueError("worlds must have shape [K,H,W]")
    if starts.ndim != 2 or starts.shape != goals.shape or starts.shape[1:] != (2,):
        raise ValueError("starts/goals must have matching shape [Q,2]")
    output = np.zeros((len(worlds), len(starts), len(radii)), dtype=bool)
    for sample_index, world in enumerate(worlds):
        clearance = _accelerated_clearance(world)
        for radius_index, radius in enumerate(radii):
            components, _ = ndimage.label(
                clearance >= int(radius), structure=FOUR_NEIGHBOUR_STRUCTURE
            )
            start_components = components[starts[:, 0], starts[:, 1]]
            goal_components = components[goals[:, 0], goals[:, 1]]
            output[sample_index, :, radius_index] = (
                (start_components != 0) & (start_components == goal_components)
            )
    return output


def _verify_accelerator(seed: int = 20260903, cases: int = 12) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    radii = (0, 1, 2, 4)
    comparisons = 0
    for case_index in range(cases):
        height = 13 + case_index % 4
        width = 15 + case_index % 5
        world = rng.random((height, width)) > (0.18 + 0.03 * (case_index % 3))
        start = (height // 2, width // 3)
        goals = np.asarray(
            [(height // 2, 2 * width // 3), (height // 3, width // 2)],
            dtype=np.int64,
        )
        world[start] = True
        world[goals[:, 0], goals[:, 1]] = True
        canonical = clearance_radius_map(world)
        accelerated = _accelerated_clearance(world)
        if not np.array_equal(canonical, accelerated):
            raise RuntimeError(f"accelerated clearance mismatch in audit case {case_index}")
        starts = np.repeat(np.asarray(start, dtype=np.int64)[None], len(goals), axis=0)
        bottleneck = batched_merge_tree_bottleneck_scores(
            canonical[None, None].astype(np.float64), starts[None], goals[None]
        )[0, 0]
        expected = bottleneck[:, None] >= np.asarray(radii)[None]
        actual = _accelerated_events(world[None], starts, goals, radii)[0]
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"accelerated connectivity mismatch in audit case {case_index}")
        comparisons += int(expected.size)
    return {
        "passed": True,
        "seed": seed,
        "random_maps": cases,
        "event_comparisons": comparisons,
        "reference": "pathrel.labels clearance_radius_map + batched merge-tree",
    }


def _load_model(
    checkpoint_path: Path,
    decoder_variant: str,
    device: torch.device,
) -> tuple[PathRelNet, dict[str, Any], dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    recorded_variant = str(config.get("decoder_variant", "correlated"))
    if recorded_variant != decoder_variant:
        raise ValueError(
            f"checkpoint decoder variant is {recorded_variant!r}, requested {decoder_variant!r}"
        )
    local_kernel_size = 1 if decoder_variant == "independent" else 5
    model = PathRelNet(
        input_channels=3,
        feature_channels=int(config.get("feature_channels", 32)),
        latent_dim=int(config.get("latent_dim", 8)),
        local_kernel_size=local_kernel_size,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    forward = {
        "decoder_variant": decoder_variant,
        "local_kernel_size": local_kernel_size,
        "effective_disable_global_factors": decoder_variant == "independent",
        "invalid_support_policy": "epistemic_mask complement clamped to blocked before sampling",
        "event_operator": "exact discrete-disk erosion + four-neighbour connected components",
    }
    return model, config, forward


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.sample_chunk < 2:
        raise SystemExit("samples and sample-chunk must both be at least two")
    if args.samples % args.sample_chunk:
        raise SystemExit("samples must be divisible by sample-chunk")
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap-samples must be positive")
    if args.validation_scenes_limit is not None and args.validation_scenes_limit < 1:
        raise SystemExit("validation-scenes-limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_unavailable_message(torch))

    started = time.monotonic()
    accelerator_audit = _verify_accelerator()
    device = torch.device(args.device)
    model, checkpoint_config, forward = _load_model(
        args.checkpoint, args.decoder_variant, device
    )
    generator = torch.Generator(device=device).manual_seed(args.evaluation_seed)
    dataset = FlatLandsReplayDataset(
        args.archive,
        args.selection,
        args.queries,
        split="validation",
        verify_frozen=True,
        verify_query_geometry=True,
    )
    limit = len(dataset)
    if args.validation_scenes_limit is not None:
        limit = min(limit, args.validation_scenes_limit)

    rows: list[dict[str, object]] = []
    try:
        for scene_index in range(limit):
            sample = dataset[scene_index]
            queries = sample.retained_queries
            if not queries:
                continue
            starts = np.asarray(
                [(int(query.start_row), int(query.start_col)) for query in queries],
                dtype=np.int64,
            )
            goals = np.asarray(
                [(int(query.goal_row), int(query.goal_col)) for query in queries],
                dtype=np.int64,
            )
            event_sum = np.zeros(
                (len(queries), len(sample.radii_cells)), dtype=np.float64
            )
            observation = torch.from_numpy(sample.input_bev[None]).to(
                device=device, dtype=torch.float32
            )
            support = torch.from_numpy(sample.epistemic_mask[None]).to(
                device=device, dtype=torch.bool
            )
            with torch.inference_mode():
                for _ in range(args.samples // args.sample_chunk):
                    posterior = model(
                        observation,
                        valid_support_mask=support,
                        num_samples=args.sample_chunk,
                        hard_samples=True,
                        disable_global_factors=args.decoder_variant == "independent",
                        generator=generator,
                    ).posterior
                    worlds = (
                        posterior.safe_samples()[0].detach().cpu().numpy() > 0.5
                    )
                    if np.any(worlds & ~sample.epistemic_mask[None]):
                        raise RuntimeError(
                            f"invalid support was sampled free for {sample.observation.global_id}"
                        )
                    event_sum += _accelerated_events(
                        worlds, starts, goals, sample.radii_cells
                    ).sum(axis=0)
            probabilities = event_sum / float(args.samples)
            for query_index, query in enumerate(queries):
                for radius_index, radius in enumerate(sample.radii_cells):
                    rows.append(
                        {
                            "global_id": sample.observation.global_id,
                            "candidate_index": query.candidate_index,
                            "radius_cells": radius,
                            "probability": float(probabilities[query_index, radius_index]),
                        }
                    )
            if (scene_index + 1) % 10 == 0 or scene_index + 1 == limit:
                print(
                    json.dumps(
                        {
                            "validation_scenes_complete": scene_index + 1,
                            "validation_scenes_total": limit,
                            "prediction_rows": len(rows),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    finally:
        dataset.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions_validation.csv"
    prediction_count = write_prediction_manifest(prediction_path, rows)
    full_validation = args.validation_scenes_limit is None
    evaluation = None
    if full_validation:
        evaluation = evaluate_flatlands_prediction_file(
            prediction_path,
            args.selection,
            args.queries,
            method=f"{args.decoder_variant}_support_clamped_posthoc",
            split="validation",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.evaluation_seed,
        )
        write_evaluation_report(args.output_dir / "evaluation_validation", evaluation)

    report = {
        "schema_version": 1,
        "kind": "flatlands_support_clamped_checkpoint_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_result": False,
        "validation_result": full_validation,
        "posthoc_checkpoint_evaluation": True,
        "retraining_required": True,
        "test_evaluated": False,
        "git": _git_state(),
        "config": {
            "checkpoint": str(args.checkpoint),
            "decoder_variant": args.decoder_variant,
            "evaluation_seed": args.evaluation_seed,
            "samples": args.samples,
            "sample_chunk": args.sample_chunk,
            "bootstrap_samples": args.bootstrap_samples,
            "device": args.device,
            "archive": str(args.archive),
            "selection": str(args.selection),
            "queries": str(args.queries),
            "validation_scenes_limit": args.validation_scenes_limit,
        },
        "checkpoint": {
            "sha256": sha256_path(args.checkpoint),
            "training_seed": checkpoint_config.get("seed"),
            "feature_channels": checkpoint_config.get("feature_channels"),
            "latent_dim": checkpoint_config.get("latent_dim"),
        },
        "forward": forward,
        "accelerator_audit": accelerator_audit,
        "prediction": {
            "path": str(prediction_path),
            "rows": prediction_count,
            "sha256": sha256_path(prediction_path),
        },
        "evaluation": evaluation,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "claim_boundary": (
            "Validation-only post-hoc recovery diagnostic for checkpoints trained before the "
            "invalid-support clamp was added. It can quantify the bug's effect but cannot replace "
            "a clean retraining run. No test labels were opened."
        ),
    }
    _atomic_json(args.output_dir / "run.json", report)
    overall = None
    if evaluation is not None:
        overall = next(
            item["scene_weighted"]
            for item in evaluation["metrics"]
            if item["scope"] == "overall"
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "prediction_rows": prediction_count,
                "overall_scene_weighted": overall,
                "test_evaluated": False,
                "retraining_required": True,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
