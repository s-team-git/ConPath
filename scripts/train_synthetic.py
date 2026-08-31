#!/usr/bin/env python3
"""Small synthetic training run for the PathRel contract.

This script is a code-path smoke test, not a paper result. It intentionally trains on rasterized
BEV observations so the stochastic-map/reachability hypothesis can be tested before adding a
dataset-specific RGB/LiDAR frontend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from pathrel.losses import (  # noqa: E402
    posterior_marginal_nll,
    reachability_brier,
    reachability_brier_u_statistic,
    spatial_variogram_score,
)
from pathrel.model import PathRelNet  # noqa: E402
from pathrel.gpu_diagnostics import cuda_unavailable_message  # noqa: E402
from pathrel.synthetic import ambiguous_corridor_scene, stack_scenes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "synthetic.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def tensor_batch(
    indices: list[int],
    *,
    height: int,
    width: int,
    radii: tuple[int, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    scenes = [
        ambiguous_corridor_scene(
            index,
            height=height,
            width=width,
            footprint_radii_cells=radii,
        )
        for index in indices
    ]
    arrays = stack_scenes(scenes)
    return {
        "observation": torch.from_numpy(arrays["observation_bev"]).to(device),
        "target_classes": torch.from_numpy(arrays["target_classes"]).long().to(device),
        "starts": torch.from_numpy(arrays["starts"]).long().to(device),
        "goals": torch.from_numpy(arrays["goals"]).long().to(device),
        "reachability_targets": torch.from_numpy(
            arrays["reachability_targets"]
        ).to(device),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(cuda_unavailable_message(torch))

    radii = tuple(int(value) for value in config["footprint_radii_cells"])
    steps = int(args.steps if args.steps is not None else config["steps"])
    batch_size = int(config["batch_size"])
    if steps < 1:
        raise ValueError("steps must be at least one")
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError("batch_size must be a positive even number to preserve open/closed pairs")
    height, width = int(config["height"]), int(config["width"])
    loss_weights = config["loss_weights"]

    model = PathRelNet(
        input_channels=3,
        feature_channels=int(config["feature_channels"]),
        latent_dim=int(config["latent_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-4
    )

    first_total = None
    model.train()
    for step in range(steps):
        # Keep consecutive open/closed pairs in each batch. They share their visible observation
        # but differ coherently behind the unknown band.
        base = (step * batch_size) % 1024
        if base % 2:
            base -= 1
        batch = tensor_batch(
            list(range(base, base + batch_size)),
            height=height,
            width=width,
            radii=radii,
            device=device,
        )
        generator = torch.Generator(device=device).manual_seed(seed * 10000 + step)
        output = model(
            batch["observation"],
            starts=batch["starts"],
            goals=batch["goals"],
            footprint_radii_cells=radii,
            num_samples=int(config["num_samples"]),
            hard_samples=True,
            max_reachability_steps=int(config["max_reachability_steps"]),
            generator=generator,
        )

        map_loss = posterior_marginal_nll(
            output.posterior.sample_logits, batch["target_classes"]
        )
        target_safe = (batch["target_classes"] == 0).to(torch.float32)
        variogram_loss = spatial_variogram_score(
            output.posterior.safe_samples(), target_safe
        )
        reach_loss = reachability_brier_u_statistic(
            output.sample_reachability, batch["reachability_targets"]
        )
        total = (
            float(loss_weights["map"]) * map_loss
            + float(loss_weights["variogram"]) * variogram_loss
            + float(loss_weights["reachability"]) * reach_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if first_total is None:
            first_total = float(total.detach())

        if step == 0 or (step + 1) % max(1, steps // 6) == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "total": round(float(total.detach()), 6),
                        "map": round(float(map_loss.detach()), 6),
                        "variogram": round(float(variogram_loss.detach()), 6),
                        "reach_u": round(float(reach_loss.detach()), 6),
                        "reach_brier": round(
                            float(
                                reachability_brier(
                                    output.reachability,
                                    batch["reachability_targets"],
                                ).detach()
                            ),
                            6,
                        ),
                    }
                )
            )

    model.eval()
    if args.validation_size < 1 or args.validation_samples < 2:
        raise ValueError("validation size must be positive and samples must be at least two")
    validation = tensor_batch(
        list(range(2048, 2048 + args.validation_size)),
        height=height,
        width=width,
        radii=radii,
        device=device,
    )
    with torch.no_grad():
        output = model(
            validation["observation"],
            starts=validation["starts"],
            goals=validation["goals"],
            footprint_radii_cells=radii,
            num_samples=max(args.validation_samples, int(config["num_samples"])),
            hard_samples=True,
            max_reachability_steps=int(config["max_reachability_steps"]),
            generator=torch.Generator(device=device).manual_seed(seed + 999999),
        )
        validation_brier = reachability_brier(
            output.reachability, validation["reachability_targets"]
        )

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = PROJECT_ROOT / "checkpoints" / "synthetic_smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config, "validation_brier": float(validation_brier)},
        checkpoint,
    )
    print(
        json.dumps(
            {
                "first_step_objective": first_total,
                "last_step_objective": float(total.detach()),
                "objectives_use_different_batches": steps > 1,
                "validation_brier": float(validation_brier),
                "checkpoint": str(checkpoint),
                "paper_result": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
