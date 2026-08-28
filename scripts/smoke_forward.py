#!/usr/bin/env python3
"""Run one deterministic PathRel forward pass on synthetic BEV observations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from pathrel.model import PathRelNet  # noqa: E402
from pathrel.synthetic import ambiguous_corridor_scene, stack_scenes  # noqa: E402


def main() -> None:
    torch.manual_seed(11)
    radii = (0, 1, 2)
    batch_np = stack_scenes(
        [
            ambiguous_corridor_scene(0, footprint_radii_cells=radii),
            ambiguous_corridor_scene(1, footprint_radii_cells=radii),
        ]
    )

    observation = torch.from_numpy(batch_np["observation_bev"])
    starts = torch.from_numpy(batch_np["starts"]).long()
    goals = torch.from_numpy(batch_np["goals"]).long()

    model = PathRelNet(input_channels=3, feature_channels=16, latent_dim=4)
    model.eval()
    generator = torch.Generator(device=observation.device).manual_seed(29)
    with torch.no_grad():
        output = model(
            observation,
            starts=starts,
            goals=goals,
            footprint_radii_cells=radii,
            num_samples=4,
            hard_samples=True,
            max_reachability_steps=24 * 24,
            generator=generator,
        )

    report = {
        "observation": list(observation.shape),
        "mean_logits": list(output.posterior.mean_logits.shape),
        "map_samples": list(output.posterior.sample_probs.shape),
        "reachability": list(output.reachability.shape),
        "reachability_values": np.round(
            output.reachability.detach().cpu().numpy(), 3
        ).tolist(),
        "target_values": batch_np["reachability_targets"].tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
