#!/usr/bin/env python3
"""Render a same-scene FlatLands map comparison from real checkpoints.

The renderer deliberately requires a ConPath checkpoint.  It never creates a synthetic
"ours is better" panel: every map is read from the frozen ZIP replay or a supplied checkpoint,
and the SVG records dataset, provenance split, scene/query identity, radius, and event scores.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.flatlands_data import FlatLandsReplayDataset  # noqa: E402
from pathrel.gpu_diagnostics import cuda_unavailable_message  # noqa: E402
from pathrel.labels import batched_merge_tree_bottleneck_scores, clearance_radius_map  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained ConPath best.pt")
    parser.add_argument("--completion-checkpoint", type=Path, default=None, help="optional deterministic completion best.pt")
    parser.add_argument("--baseline-predictions", type=Path, default=None, help="optional direct-query prediction CSV")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/flatlands/FlatLands_final_dataset.zip"))
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--global-id")
    selector.add_argument("--index", type=int)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--max-reachability-steps", type=int, default=256)
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    return state, config


def _find_sample(dataset: FlatLandsReplayDataset, global_id: str | None, index: int | None):
    if index is not None:
        return dataset[index]
    assert global_id is not None
    for row_index, observation in enumerate(dataset.observations):
        if observation.global_id == global_id:
            return dataset[row_index]
    raise ValueError(f"global_id not found in the requested provenance split: {global_id}")


def _choose_query(sample: Any, candidate_index: int | None):
    retained = sample.retained_queries
    if candidate_index is not None:
        retained = tuple(query for query in retained if query.candidate_index == candidate_index)
    if not retained:
        raise ValueError("selected scene has no retained query matching --candidate-index")
    query = retained[0]
    if any(value is None for value in (query.start_row, query.start_col, query.goal_row, query.goal_col)):
        raise ValueError("qualitative query has missing endpoints")
    return query


def _exact_event_probability(safe_samples: torch.Tensor, query: Any, radius: int) -> float:
    worlds = safe_samples.detach().cpu().numpy() > 0.5
    scores = np.stack([clearance_radius_map(world) for world in worlds[0]], axis=0)[None]
    starts = np.asarray([[[query.start_row, query.start_col]]], dtype=np.int64)
    goals = np.asarray([[[query.goal_row, query.goal_col]]], dtype=np.int64)
    bottleneck = batched_merge_tree_bottleneck_scores(scores, starts, goals)
    return float(np.mean(bottleneck[0, :, 0] >= int(radius)))


def _load_event_prediction(path: Path | None, global_id: str, candidate_index: int, radius: int) -> float | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("global_id") == global_id
                and int(row.get("candidate_index", -1)) == candidate_index
                and int(row.get("radius_cells", -1)) == radius
            ):
                return float(row["probability"])
    return None


def _downsample(value: np.ndarray, max_side: int, *, binary: bool = False) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError("map values must be two-dimensional")
    if max_side < 8:
        raise ValueError("max-side must be at least eight")
    factor = max(1, int(np.ceil(max(value.shape) / float(max_side))))
    if factor == 1:
        return value.astype(np.float32, copy=False)
    height = int(np.ceil(value.shape[0] / factor))
    width = int(np.ceil(value.shape[1] / factor))
    output = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            block = value[row * factor : (row + 1) * factor, col * factor : (col + 1) * factor]
            output[row, col] = float(np.max(block) if binary else np.mean(block))
    return output


def _hex_rgb(red: float, green: float, blue: float) -> str:
    values = [max(0, min(255, int(round(channel * 255)))) for channel in (red, green, blue)]
    return "#%02x%02x%02x" % tuple(values)


def _probability_color(probability: float) -> str:
    # Dark blue -> teal -> warm yellow, readable on a white page and without a misleading
    # red/green safety convention.
    value = max(0.0, min(1.0, float(probability)))
    if value < 0.5:
        t = value * 2.0
        return _hex_rgb(0.08 * (1 - t) + 0.08 * t, 0.24 * (1 - t) + 0.62 * t, 0.36 * (1 - t) + 0.60 * t)
    t = (value - 0.5) * 2.0
    return _hex_rgb(0.08 * (1 - t) + 0.97 * t, 0.62 * (1 - t) + 0.76 * t, 0.60 * (1 - t) + 0.35 * t)


def _panel_svg(
    title: str,
    values: np.ndarray,
    *,
    mode: str,
    x: int,
    y: int,
    panel_size: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> str:
    height, width = values.shape
    cell = panel_size / float(max(height, width))
    offset_x = x + (panel_size - width * cell) / 2.0
    offset_y = y + (panel_size - height * cell) / 2.0
    chunks = [f'<text class="panel-title" x="{x + panel_size / 2:.1f}" y="{y - 12}" text-anchor="middle">{html.escape(title)}</text>']
    for row in range(height):
        for col in range(width):
            value = float(values[row, col])
            if mode == "probability":
                color = _probability_color(value)
            elif mode == "observed":
                color = {0.0: "#f0f3f5", 1.0: "#3d997c", 2.0: "#354451"}.get(value, "#ffffff")
            else:
                color = "#ffffff" if value < 0.0 else ("#3d997c" if value > 0.5 else "#354451")
            chunks.append(
                f'<rect x="{offset_x + col * cell:.2f}" y="{offset_y + row * cell:.2f}" '
                f'width="{cell + 0.12:.2f}" height="{cell + 0.12:.2f}" fill="{color}"/>'
            )
    for label, point, color in (("S", start, "#315b86"), ("G", goal, "#c47b38")):
        row, col = point
        px = offset_x + (col + 0.5) * cell
        py = offset_y + (row + 0.5) * cell
        radius = max(4.0, cell * 1.7)
        chunks.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        chunks.append(f'<text class="marker" x="{px:.2f}" y="{py + 3.5:.2f}" text-anchor="middle">{label}</text>')
    chunks.append(f'<rect x="{offset_x:.2f}" y="{offset_y:.2f}" width="{width * cell:.2f}" height="{height * cell:.2f}" fill="none" stroke="#cdd6dd"/>')
    return "".join(chunks)


def _build_svg(
    panels: Iterable[tuple[str, np.ndarray, str]],
    *,
    metadata: dict[str, Any],
    start: tuple[int, int],
    goal: tuple[int, int],
    panel_size: int,
) -> str:
    panels = tuple(panels)
    margin = 34
    gap = 28
    header = 138
    width = margin * 2 + len(panels) * panel_size + (len(panels) - 1) * gap
    height = header + panel_size + 44
    title = (
        f"FlatLands · {metadata['split']} · {metadata['source']} · "
        f"scene {metadata['scene_id']} · r={metadata['radius']} cells"
    )
    subtitle = (
        f"global_id={metadata['global_id']} · candidate={metadata['candidate_index']} · "
        f"S=({start[0]},{start[1]}) G=({goal[0]},{goal[1]}) · "
        f"ConPath exact event={metadata['conpath_event']:.3f}"
    )
    if metadata.get("direct_query_event") is not None:
        subtitle += f" · direct-query={metadata['direct_query_event']:.3f}"
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#263744}.title{font-size:20px;font-weight:700}.sub{font-size:11px;fill:#66737e}.panel-title{font-size:13px;font-weight:700}.marker{font-size:9px;font-weight:800;fill:#fff}</style>',
        f'<text class="title" x="{margin}" y="31">{html.escape(title)}</text>',
        f'<text class="sub" x="{margin}" y="54">{html.escape(subtitle)}</text>',
        f'<text class="sub" x="{margin}" y="75">All panels use the same frozen ZIP replay and target-blind query; reference is shown for diagnosis, not used by the model.</text>',
    ]
    for index, (panel_title, values, mode) in enumerate(panels):
        x = margin + index * (panel_size + gap)
        chunks.append(_panel_svg(panel_title, values, mode=mode, x=x, y=header, panel_size=panel_size, start=start, goal=goal))
    chunks.append(f'<text class="sub" x="{margin}" y="{height - 15}">Legend: S start · G goal · observed free green · observed blocked charcoal · probability map dark blue→yellow · white invalid/out-of-support</text>')
    chunks.append("</svg>")
    return "".join(chunks)


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.max_reachability_steps < 1:
        raise SystemExit("samples must be at least two and max-reachability-steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_unavailable_message(torch))
    device = torch.device(args.device)
    state, config = _load_checkpoint(args.checkpoint, device)
    dataset = FlatLandsReplayDataset(
        args.archive,
        args.selection,
        args.queries,
        split=args.split,
        verify_frozen=True,
        verify_query_geometry=True,
    )
    try:
        sample = _find_sample(dataset, args.global_id, args.index)
        query = _choose_query(sample, args.candidate_index)
        if args.radius not in sample.radii_cells:
            raise ValueError(f"radius {args.radius} is not in frozen radii {sample.radii_cells}")
        observation = torch.from_numpy(sample.input_bev[None]).to(device=device, dtype=torch.float32)
        starts = torch.tensor([[[query.start_row, query.start_col]]], dtype=torch.long, device=device)
        goals = torch.tensor([[[query.goal_row, query.goal_col]]], dtype=torch.long, device=device)
        feature_channels = int(config.get("feature_channels", 32))
        latent_dim = int(config.get("latent_dim", 8))
        model = PathRelNet(input_channels=3, feature_channels=feature_channels, latent_dim=latent_dim).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with torch.inference_mode():
            output = model(
                observation,
                starts=starts,
                goals=goals,
                footprint_radii_cells=(args.radius,),
                num_samples=args.samples,
                hard_samples=True,
                max_reachability_steps=args.max_reachability_steps,
                shared_start=True,
                disable_global_factors=bool(config.get("disable_global_factors", False)),
                generator=generator,
            )
        conpath_probability = output.posterior.conditional_class_probs[0, :, 0].mean(dim=0).cpu().numpy()
        conpath_event = _exact_event_probability(output.posterior.safe_samples(), query, args.radius)

        completion_probability = None
        completion_event = None
        if args.completion_checkpoint is not None:
            completion_state, completion_config = _load_checkpoint(args.completion_checkpoint, device)
            from pathrel.flatlands_baselines import MarginalCompletionBaseline

            completion_model = MarginalCompletionBaseline(feature_channels=int(completion_config.get("feature_channels", 16))).to(device)
            completion_model.load_state_dict(completion_state["model"])
            completion_model.eval()
            with torch.inference_mode():
                completion_probability = completion_model.free_probability(observation)[0].cpu().numpy()
            deterministic_world = completion_probability > 0.5
            completion_event = float(
                batched_merge_tree_bottleneck_scores(
                    clearance_radius_map(deterministic_world)[None, None],
                    np.asarray([[[query.start_row, query.start_col]]]),
                    np.asarray([[[query.goal_row, query.goal_col]]]),
                )[0, 0, 0] >= args.radius
            )
        direct_event = _load_event_prediction(args.baseline_predictions, sample.observation.global_id, query.candidate_index, args.radius)

        observed = np.full(sample.observed_floor.shape, 0.0, dtype=np.float32)
        valid = sample.epistemic_mask.astype(bool)
        observed[sample.observed_free.astype(bool)] = 1.0
        observed[(valid & ~sample.unknown.astype(bool) & ~sample.observed_free.astype(bool))] = 2.0
        reference = np.full(sample.floor_map.shape, -1.0, dtype=np.float32)
        reference[valid & sample.target_free.astype(bool)] = 1.0
        reference[valid & ~sample.target_free.astype(bool)] = 0.0
        panel_values = [
            ("Observed BEV", _downsample(observed, args.max_side, binary=True), "observed"),
            ("ConPath P(traversable)", _downsample(conpath_probability, args.max_side), "probability"),
        ]
        if completion_probability is not None:
            panel_values.append(("Deterministic completion", _downsample(completion_probability, args.max_side), "probability"))
        panel_values.append(("Reference map", _downsample(reference, args.max_side), "reference"))
        metadata = {
            "split": sample.observation.provenance_split,
            "source": sample.observation.source_dataset,
            "scene_id": sample.observation.scene_id,
            "global_id": sample.observation.global_id,
            "candidate_index": query.candidate_index,
            "radius": args.radius,
            "conpath_event": conpath_event,
            "completion_event": completion_event,
            "direct_query_event": direct_event,
        }
        svg = _build_svg(
            panel_values,
            metadata=metadata,
            start=(int(query.start_row), int(query.start_col)),
            goal=(int(query.goal_row), int(query.goal_col)),
            panel_size=280,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg, encoding="utf-8")
        summary = {
            "output": str(args.output),
            "dataset": "FlatLands",
            "split": sample.observation.provenance_split,
            "source_dataset": sample.observation.source_dataset,
            "scene_id": sample.observation.scene_id,
            "global_id": sample.observation.global_id,
            "candidate_index": query.candidate_index,
            "radius_cells": args.radius,
            "conpath_event_exact": conpath_event,
            "completion_event": completion_event,
            "direct_query_event": direct_event,
            "test_evaluated": False,
        }
        args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
