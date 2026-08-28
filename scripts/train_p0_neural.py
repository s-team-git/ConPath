#!/usr/bin/env python3
"""Train/evaluate the neural PathRel head on the held-out-template P0 protocol.

This is intentionally separate from the short ``train_synthetic.py`` code-path smoke.  It uses the
same context-family/random-visible-query generator as ``evaluate_p0.py`` so a neural checkpoint can
be compared to the P0 baselines without changing the split.  CUDA is recommended; CPU runs with a
small ``--steps`` value are useful only for debugging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

from evaluate_p0 import make_split, stack  # noqa: E402
from pathrel.losses import (  # noqa: E402
    map_cross_entropy,
    posterior_marginal_nll,
    reachability_brier_u_statistic,
    spatial_variogram_score,
)
from pathrel.metrics import summarize  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-templates", type=int, default=12)
    parser.add_argument("--test-templates", type=int, default=4)
    parser.add_argument("--worlds-per-template", type=int, default=24)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument(
        "--categorical-noise-scale",
        type=float,
        default=0.25,
        help="Scale of per-cell Concrete/Gumbel noise; lower values preserve learned spatial correlation.",
    )
    parser.add_argument("--reachability-weight", type=float, default=2.0)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "p0_neural")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.warmup_steps < 0 or args.warmup_steps > args.steps or args.batch_size < 2 or args.train_samples < 2 or args.validation_samples < 2 or args.categorical_noise_scale < 0:
        raise ValueError("steps, batch size, and sample counts must be >=2 where applicable")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; run scripts/check_environment.py first."
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    radii = (0, 1, 2)
    train_arrays = stack(
        make_split(
            range(args.train_templates), worlds_per_template=args.worlds_per_template,
            height=args.height, width=args.width, radii=radii,
        )
    )
    test_arrays = stack(
        make_split(
            range(args.train_templates, args.train_templates + args.test_templates),
            worlds_per_template=args.worlds_per_template, height=args.height, width=args.width, radii=radii,
        )
    )
    model = PathRelNet(input_channels=3, feature_channels=32, latent_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    model.train()
    for step in range(args.steps):
        batch_rng = np.random.default_rng(args.seed + step)
        if args.batch_size % 2:
            raise ValueError("batch_size must be even so each context family is represented equally")
        context_zero = np.flatnonzero(train_arrays["context"] == 0)
        context_one = np.flatnonzero(train_arrays["context"] == 1)
        half_batch = args.batch_size // 2
        indices = np.concatenate(
            [
                batch_rng.choice(context_zero, half_batch, replace=False),
                batch_rng.choice(context_one, half_batch, replace=False),
            ]
        )
        observation = torch.from_numpy(train_arrays["observation"][indices]).to(device=device, dtype=torch.float32)
        target_classes = torch.from_numpy((~train_arrays["target_free"][indices]).astype(np.int64)).to(device=device, dtype=torch.long)
        starts = torch.from_numpy(train_arrays["starts"][indices]).to(device=device, dtype=torch.long)
        goals = torch.from_numpy(train_arrays["goals"][indices]).to(device=device, dtype=torch.long)
        targets = torch.from_numpy(train_arrays["events"][indices]).to(device=device, dtype=torch.float32)
        if step < args.warmup_steps:
            # Stage A stabilizes the deterministic mean map before the high-variance event
            # estimator is enabled.  The stochastic heads receive no gradient in this stage.
            output = model(
                observation, num_samples=args.train_samples, hard_samples=True,
                categorical_noise_scale=args.categorical_noise_scale, generator=generator
            )
            map_loss = map_cross_entropy(output.posterior.mean_logits, target_classes)
            variogram_loss = output.posterior.mean_logits.sum() * 0.0
            reach_loss = output.posterior.mean_logits.sum() * 0.0
            total = map_loss
        else:
            output = model(
                observation, starts=starts, goals=goals, footprint_radii_cells=radii,
                num_samples=args.train_samples, hard_samples=True,
                categorical_noise_scale=args.categorical_noise_scale,
                max_reachability_steps=args.height * args.width, generator=generator,
            )
            map_loss = posterior_marginal_nll(output.posterior.sample_logits, target_classes)
            variogram_loss = spatial_variogram_score(
                output.posterior.safe_samples(), (target_classes == 0).to(torch.float32)
            )
            reach_loss = reachability_brier_u_statistic(output.sample_reachability, targets)
            total = map_loss + 0.2 * variogram_loss + args.reachability_weight * reach_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        row = {"step": step + 1, "stage": "warmup" if step < args.warmup_steps else "joint", "total": float(total.detach()), "map": float(map_loss.detach()), "variogram": float(variogram_loss.detach()), "reach_u": float(reach_loss.detach())}
        losses.append(row)
        if step == 0 or (step + 1) % max(1, args.steps // 6) == 0:
            print(json.dumps(row))

    model.eval()
    test_observation = torch.from_numpy(test_arrays["observation"]).to(device=device, dtype=torch.float32)
    test_starts = torch.from_numpy(test_arrays["starts"]).to(device=device, dtype=torch.long)
    test_goals = torch.from_numpy(test_arrays["goals"]).to(device=device, dtype=torch.long)
    with torch.no_grad():
        output = model(
            test_observation, starts=test_starts, goals=test_goals, footprint_radii_cells=radii,
            num_samples=args.validation_samples, hard_samples=True,
            categorical_noise_scale=args.categorical_noise_scale,
            max_reachability_steps=args.height * args.width,
            generator=torch.Generator(device=device).manual_seed(args.seed + 999999),
        )
    event_report = summarize(output.reachability.detach().cpu().numpy(), test_arrays["events"])
    map_report = summarize(
        output.posterior.posterior_marginal_probs[:, 0].detach().cpu().numpy(),
        test_arrays["target_free"].astype(np.float64),
    )
    report = {
        "protocol": {"seed": args.seed, "radii_cells": list(radii), "map_shape": [args.height, args.width], "train_templates": args.train_templates, "test_templates": args.test_templates, "worlds_per_template": args.worlds_per_template, "train_samples": args.train_samples, "validation_samples": args.validation_samples, "warmup_steps": args.warmup_steps, "reachability_weight": args.reachability_weight, "categorical_noise_scale": args.categorical_noise_scale},
        "event_metrics": event_report,
        "event_mean_by_radius": output.reachability.detach().cpu().numpy().mean(axis=(0, 1)).tolist(),
        "target_event_mean_by_radius": test_arrays["events"].mean(axis=(0, 1)).tolist(),
        "map_marginal_metrics": {key: value for key, value in map_report.items() if key != "reliability"},
        "last_training": losses[-1],
        "paper_result": False,
        "interpretation": "A neural P0 checkpoint can be compared to results/p0_death_test only when this run completes on the same device/split; CPU smoke is not a paper result.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "report": report}, args.output_dir / "checkpoint.pt")
    print(json.dumps({"output_dir": str(args.output_dir), "event_metrics": {key: value for key, value in event_report.items() if isinstance(value, (float, int))}, "paper_result": False}, indent=2))


if __name__ == "__main__":
    main()
