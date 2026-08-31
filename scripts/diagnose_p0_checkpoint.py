#!/usr/bin/env python3
"""Audit a neural P0 checkpoint's hard-map joint statistics.

The trainer historically reported posterior marginals from conditional softmax probabilities while
path events were computed from straight-through hard maps.  This script reports both views, plus
doorway coherence and exact NumPy event metrics, so a good voxel score cannot hide an over-open
joint posterior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

from evaluate_p0 import exact_events, make_split, stack  # noqa: E402
from train_p0_neural import add_visible_context_plane  # noqa: E402
from pathrel.metrics import summarize  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402
from pathrel.gpu_diagnostics import cuda_unavailable_message  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--categorical-noise-scale", type=float)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip the exact NumPy event pass and only audit map/joint sample statistics.",
    )
    return parser.parse_args()


def finite_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    return summarize(probabilities.astype(np.float64), targets.astype(np.float64))


def by_radius(probabilities: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    return [
        finite_metrics(probabilities[..., index], targets[..., index])
        for index in range(probabilities.shape[-1])
    ]


def longest_true_runs(values: np.ndarray) -> np.ndarray:
    """Return the longest contiguous true run along the final dimension."""

    flat = np.asarray(values, dtype=bool).reshape(-1, values.shape[-1])
    result = np.zeros(len(flat), dtype=np.int64)
    for index, row in enumerate(flat):
        longest = current = 0
        for item in row:
            current = current + 1 if item else 0
            longest = max(longest, current)
        result[index] = longest
    return result.reshape(values.shape[:-1])


def doorway_summary(wall_free: np.ndarray) -> dict[str, float]:
    wall_free = np.asarray(wall_free, dtype=bool)
    transitions = np.count_nonzero(wall_free[..., 1:] != wall_free[..., :-1], axis=-1)
    longest = longest_true_runs(wall_free)
    any_open = wall_free.any(axis=-1)
    return {
        "any_open_rate": float(any_open.mean()),
        "all_blocked_rate": float((~any_open).mean()),
        "mean_free_cells": float(wall_free.sum(axis=-1).mean()),
        "mean_longest_open_run": float(longest.mean()),
        "mean_transitions": float(transitions.mean()),
        "fragmented_rate": float((transitions > 2).mean()),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def protocol_value(protocol: dict[str, Any], key: str, default: Any) -> Any:
    value = protocol.get(key, default)
    return default if value is None else value


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.batch_size < 1:
        raise ValueError("samples must be at least two and batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(cuda_unavailable_message(torch))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("checkpoint does not contain a model state")
    state = checkpoint["model"]
    prior_report = checkpoint.get("report", {})
    protocol = prior_report.get("protocol", checkpoint.get("config", {}))
    train_templates = int(protocol_value(protocol, "train_templates", 12))
    test_templates = int(protocol_value(protocol, "test_templates", 4))
    worlds_per_template = int(protocol_value(protocol, "worlds_per_template", 24))
    height, width = (int(value) for value in protocol_value(protocol, "map_shape", [24, 24]))
    radii = tuple(int(value) for value in protocol_value(protocol, "radii_cells", [0, 1, 2]))
    seed = int(protocol_value(protocol, "seed", 20260827))
    noise_scale = (
        float(args.categorical_noise_scale)
        if args.categorical_noise_scale is not None
        else float(protocol_value(protocol, "categorical_noise_scale", 1.0))
    )

    mean_weight = state["decoder.mean_head.weight"]
    factor_weight = state["decoder.factor_head.weight"]
    num_classes = int(mean_weight.shape[0])
    feature_channels = int(mean_weight.shape[1])
    if factor_weight.shape[0] % num_classes:
        raise ValueError("cannot infer latent dimension from factor head")
    latent_dim = int(factor_weight.shape[0] // num_classes)
    encoder_first_weight = state["encoder.enc0.block.0.weight"]
    use_coordinate_channels = bool(
        protocol_value(protocol, "encoder_context_mode", "local") == "coord_global"
        or any(key.startswith("encoder.context_projection.") for key in state)
    )
    input_channels = int(encoder_first_weight.shape[1]) - (2 if use_coordinate_channels else 0)
    context_input = str(
        protocol_value(protocol, "context_input", "plane" if input_channels == 4 else "marker")
    )
    use_global_context = any(key.startswith("encoder.context_projection.") for key in state)
    model = PathRelNet(
        input_channels=input_channels,
        feature_channels=feature_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        use_coordinate_channels=use_coordinate_channels,
        use_global_context=use_global_context,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    template_ids = (
        range(train_templates)
        if args.split == "train"
        else range(train_templates, train_templates + test_templates)
    )
    arrays = stack(
        make_split(
            template_ids,
            worlds_per_template=worlds_per_template,
            height=height,
            width=width,
            radii=radii,
        )
    )
    generator = torch.Generator(device=device).manual_seed(seed + 999999)
    soft_marginals: list[np.ndarray] = []
    empirical_marginals: list[np.ndarray] = []
    sampled_maps: list[np.ndarray] = []
    event_probabilities: list[np.ndarray] = []
    factor_standard_deviations: list[np.ndarray] = []
    local_standard_deviations: list[np.ndarray] = []
    adjacent_factor_cosines: list[np.ndarray] = []
    mean_logit_differences: list[np.ndarray] = []

    for begin in range(0, len(arrays["observation"]), args.batch_size):
        end = min(begin + args.batch_size, len(arrays["observation"]))
        observation_numpy = add_visible_context_plane(
            arrays["observation"][begin:end],
            arrays["context"][begin:end],
            mode=context_input,
        )
        observation = torch.from_numpy(observation_numpy).to(
            device=device, dtype=torch.float32
        )
        with torch.no_grad():
            output = model(
                observation,
                num_samples=args.samples,
                hard_samples=True,
                categorical_noise_scale=noise_scale,
                generator=generator,
            )
        posterior = output.posterior
        soft_marginals.append(posterior.posterior_marginal_probs[:, 0].cpu().numpy())
        empirical_marginals.append(posterior.empirical_class_frequencies[:, 0].cpu().numpy())
        maps = posterior.safe_samples().cpu().numpy().astype(bool)
        sampled_maps.append(maps)

        factor_difference = posterior.factor_maps[:, 0] - posterior.factor_maps[:, 1]
        factor_std = factor_difference.square().mean(dim=1).sqrt()
        factor_standard_deviations.append(factor_std.cpu().numpy())
        local_std = posterior.local_scale.square().sum(dim=1).sqrt()
        local_standard_deviations.append(local_std.cpu().numpy())
        mean_logit_differences.append(
            (posterior.mean_logits[:, 0] - posterior.mean_logits[:, 1]).cpu().numpy()
        )
        wall_factors = factor_difference[..., width // 2]
        numerator = (wall_factors[:, :, 1:] * wall_factors[:, :, :-1]).sum(dim=1)
        denominator = (
            wall_factors[:, :, 1:].norm(dim=1) * wall_factors[:, :, :-1].norm(dim=1)
        ).clamp_min(1e-8)
        adjacent_factor_cosines.append((numerator / denominator).cpu().numpy())

        if not args.skip_events:
            sample_events = np.stack(
                [
                    exact_events(
                        maps[:, sample_index],
                        arrays["starts"][begin:end],
                        arrays["goals"][begin:end],
                        radii,
                    )
                    for sample_index in range(args.samples)
                ],
                axis=0,
            )
            event_probabilities.append(sample_events.mean(axis=0))
        print(json.dumps({"processed": end, "total": len(arrays["observation"])}), flush=True)

    soft = np.concatenate(soft_marginals, axis=0)
    empirical = np.concatenate(empirical_marginals, axis=0)
    maps = np.concatenate(sampled_maps, axis=0)
    factor_std = np.concatenate(factor_standard_deviations, axis=0)
    local_std = np.concatenate(local_standard_deviations, axis=0)
    factor_cosine = np.concatenate(adjacent_factor_cosines, axis=0)
    mean_logit_difference = np.concatenate(mean_logit_differences, axis=0)
    target_free = arrays["target_free"].astype(np.float64)
    unknown = arrays["observation"][:, 2] > 0.5
    wall_col = width // 2

    target_wall = arrays["target_free"][:, :, wall_col]
    sampled_wall = maps[..., wall_col]
    doorway: dict[str, Any] = {"target_all": doorway_summary(target_wall)}
    for family in (0, 1):
        selected = arrays["context"] == family
        doorway[f"target_context_{family}"] = doorway_summary(target_wall[selected])
        doorway[f"sampled_context_{family}"] = doorway_summary(sampled_wall[selected])

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_report_event_brier": prior_report.get("event_metrics", {}).get("brier"),
        "protocol": {
            "seed": seed,
            "radii_cells": list(radii),
            "map_shape": [height, width],
            "train_templates": train_templates,
            "test_templates": test_templates,
            "worlds_per_template": worlds_per_template,
            "diagnostic_samples": args.samples,
            "batch_size": args.batch_size,
            "categorical_noise_scale": noise_scale,
            "device": str(device),
            "split": args.split,
            "encoder_context_mode": (
                "coord_global" if use_coordinate_channels and use_global_context else "local"
            ),
            "context_input": context_input,
        },
        "map_marginal_metrics": {
            "conditional_softmax_full": finite_metrics(soft, target_free),
            "empirical_hard_full": finite_metrics(empirical, target_free),
            "conditional_softmax_unknown": finite_metrics(soft[unknown], target_free[unknown]),
            "empirical_hard_unknown": finite_metrics(empirical[unknown], target_free[unknown]),
        },
        "posterior_scale": {
            "global_logit_difference_std_mean": float(factor_std.mean()),
            "global_logit_difference_std_unknown_mean": float(factor_std[unknown].mean()),
            "global_logit_difference_std_wall_mean": float(factor_std[:, :, wall_col].mean()),
            "local_logit_difference_std_mean": float(local_std.mean()),
            "local_logit_difference_std_unknown_mean": float(local_std[unknown].mean()),
            "local_logit_difference_std_wall_mean": float(local_std[:, :, wall_col].mean()),
            "wall_adjacent_factor_cosine_mean": float(factor_cosine.mean()),
            "categorical_noise_scale": noise_scale,
        },
        "posterior_by_context": {
            str(family): {
                "conditional_wall_free_mean": float(
                    soft[arrays["context"] == family, :, wall_col].mean()
                ),
                "empirical_wall_free_mean": float(
                    empirical[arrays["context"] == family, :, wall_col].mean()
                ),
                "mean_logit_difference_wall_mean": float(
                    mean_logit_difference[arrays["context"] == family, :, wall_col].mean()
                ),
                "global_logit_difference_std_wall_mean": float(
                    factor_std[arrays["context"] == family, :, wall_col].mean()
                ),
                "local_logit_difference_std_wall_mean": float(
                    local_std[arrays["context"] == family, :, wall_col].mean()
                ),
            }
            for family in (0, 1)
        },
        "doorway_joint": doorway,
        "known_evidence_violation_rate": float(
            (
                (maps != arrays["observation"][:, None, 0].astype(bool))
                & (arrays["observation"][:, None, 2] < 0.5)
            ).sum()
            / max(1, int((arrays["observation"][:, 2] < 0.5).sum()) * args.samples)
        ),
    }
    if not args.skip_events:
        events = np.concatenate(event_probabilities, axis=0)
        report["event_metrics"] = finite_metrics(events, arrays["events"])
        report["event_metrics_by_radius"] = by_radius(events, arrays["events"])
        report["event_mean_by_radius"] = events.mean(axis=(0, 1)).tolist()
        report["target_event_mean_by_radius"] = arrays["events"].mean(axis=(0, 1)).tolist()
        report["event_mean_by_context_and_radius"] = {
            str(family): events[arrays["context"] == family].mean(axis=(0, 1)).tolist()
            for family in (0, 1)
        }
        report["target_event_mean_by_context_and_radius"] = {
            str(family): arrays["events"][arrays["context"] == family]
            .mean(axis=(0, 1))
            .tolist()
            for family in (0, 1)
        }

    output_path = args.output or args.checkpoint.parent / "diagnostics.json"
    atomic_json(output_path, report)
    print(json.dumps({"output": str(output_path), **report}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
