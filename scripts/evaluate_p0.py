#!/usr/bin/env python3
"""Reproducible synthetic P0 death test.

This experiment is deliberately independent of PyTorch.  It audits the scientific claim before a
large perception stack is introduced: all methods see the same partial BEV observation, all event
labels come from the exact NumPy oracle, and train/test are split by scene template.  The
``correlated_*`` rows are a small oracle posterior implementation of the PathRel hypothesis; when
CUDA is available, ``train_synthetic.py`` should additionally be run to replace this diagnostic
with a learned neural checkpoint.

The output directory contains a JSON report, a long CSV table, and an SVG reliability figure.  The
SVG writer avoids making matplotlib a hidden dependency on a remote training machine.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import heapq
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.labels import merge_tree_bottleneck_scores, reachability_targets  # noqa: E402
from pathrel.metrics import summarize  # noqa: E402
from pathrel.synthetic import SyntheticScene, ambiguous_corridor_scene  # noqa: E402


Radii = tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "p0_death_test")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--train-templates", type=int, default=12)
    parser.add_argument("--test-templates", type=int, default=4)
    parser.add_argument("--worlds-per-template", type=int, default=24)
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    return parser.parse_args()


def make_split(
    template_ids: Iterable[int],
    *,
    worlds_per_template: int,
    height: int,
    width: int,
    radii: Radii,
) -> list[SyntheticScene]:
    scenes: list[SyntheticScene] = []
    for template_id in template_ids:
        for context_family in (0, 1):
            for world_index in range(worlds_per_template):
                # The template is held fixed while the deterministic Bernoulli world draw changes.
                # This creates repeated, identical observations with multiple hidden worlds.
                index = template_id * 100000 + context_family * 1000 + world_index
                scene = ambiguous_corridor_scene(
                    index,
                    height=height,
                    width=width,
                    footprint_radii_cells=radii,
                    context_family=context_family,
                    template_id=template_id,
                )
                # Query coordinates are sampled from visible support cells only.  This prevents
                # the direct-query baseline from memorising one hand-designed start/goal pair and
                # tests whether a posterior can answer unseen query geometry on held-out templates.
                observation = scene.observation_bev
                wall_col = width // 2
                left_candidates = np.argwhere(observation[0] > 0.5)
                left_candidates = left_candidates[left_candidates[:, 1] < wall_col - 2]
                right_candidates = np.argwhere(observation[0] > 0.5)
                right_candidates = right_candidates[right_candidates[:, 1] > wall_col + 2]
                query_rng = np.random.default_rng(4099 * (template_id + 1) + 97 * (context_family + 1))
                starts, goals = [], []
                for _ in range(2):
                    starts.append(tuple(left_candidates[int(query_rng.integers(len(left_candidates)))]))
                    goals.append(tuple(right_candidates[int(query_rng.integers(len(right_candidates)))]))
                starts_array = np.asarray(starts, dtype=np.int64)
                goals_array = np.asarray(goals, dtype=np.int64)
                events, max_clearance = reachability_targets(
                    scene.target_classes == 0, starts_array, goals_array, radii
                )
                scenes.append(
                    replace(
                        scene,
                        starts=starts_array,
                        goals=goals_array,
                        reachability_targets=events.astype(np.float32),
                        max_clearance_cells=max_clearance,
                    )
                )
    return scenes


def stack(scenes: list[SyntheticScene]) -> dict[str, np.ndarray]:
    return {
        "observation": np.stack([scene.observation_bev for scene in scenes]),
        "target_free": np.stack([scene.target_classes == 0 for scene in scenes]),
        "starts": np.stack([scene.starts for scene in scenes]),
        "goals": np.stack([scene.goals for scene in scenes]),
        "events": np.stack([scene.reachability_targets for scene in scenes]),
        "max_clearance": np.stack([scene.max_clearance_cells for scene in scenes]),
        "context": np.asarray([scene.context_family for scene in scenes], dtype=np.int64),
        "template": np.asarray([scene.template_id for scene in scenes], dtype=np.int64),
        "door_open": np.asarray([scene.doorway_is_open for scene in scenes], dtype=bool),
    }


def exact_events(maps_free: np.ndarray, starts: np.ndarray, goals: np.ndarray, radii: Radii) -> np.ndarray:
    """Apply the exact footprint-conditioned oracle to a batch of sampled worlds."""

    maps_free = np.asarray(maps_free, dtype=bool)
    if maps_free.ndim != 3:
        raise ValueError("maps_free must have shape [N,H,W]")
    return np.stack(
        [
            reachability_targets(world, starts[item], goals[item], radii)[0]
            for item, world in enumerate(maps_free)
        ],
        axis=0,
    ).astype(np.float64)


def hidden_marginals(train: dict[str, np.ndarray]) -> np.ndarray:
    """Estimate per-cell free probabilities conditioned on the visible context token."""

    observation = train["observation"]
    target_free = train["target_free"]
    context = train["context"]
    probabilities = np.empty((2, *target_free.shape[1:]), dtype=np.float64)
    for family in (0, 1):
        selected = context == family
        mean_free = target_free[selected].mean(axis=0)
        known_free = observation[selected, 0].mean(axis=0) > 0.5
        known_blocked = observation[selected, 1].mean(axis=0) > 0.5
        probabilities[family] = np.where(known_free, 1.0, np.where(known_blocked, 0.0, mean_free))
    return probabilities


def independent_samples(
    probability_maps: np.ndarray,
    observations: np.ndarray,
    context: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = probability_maps[context]
    draws = rng.random((samples, len(context), *probabilities.shape[1:])) < probabilities[None]
    # Keep visible cells exact.  This is important: an occupancy completion method must not alter
    # evidence that was actually observed.
    known = observations[:, 2] < 0.5
    observed_free = observations[:, 0].astype(bool)
    draws = np.where(known[None], observed_free[None], draws)
    return draws


def coherent_samples(
    probability_maps: np.ndarray,
    observations: np.ndarray,
    context: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    condition_on_context: bool = True,
    context_open_prior: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a low-dimensional coherent doorway latent plus fixed visible evidence.

    The decoder does not sample each unknown voxel independently.  One latent doorway state is
    shared by the whole occluded wall column, which is the minimum construction that can represent
    the same-marginal/different-topology conflict.  The open probability is estimated from training
    worlds and never reads the test GT map.
    """

    n = len(context)
    base = probability_maps[context].copy()
    unknown = observations[:, 2] > 0.5
    wall_columns = np.full(n, unknown.shape[-1] // 2, dtype=np.int64)
    # Estimate the context prior from complete training worlds (the same statistic the neural
    # decoder is expected to learn).  Falling back to the central-cell marginal keeps this helper
    # useful for ad-hoc experiments, but the official report passes the audited training prior.
    door_row = observations.shape[-2] // 2
    if context_open_prior is None:
        door_probability = np.clip(base[np.arange(n), door_row, wall_columns], 0.02, 0.98)
    else:
        prior = np.asarray(context_open_prior, dtype=np.float64)
        if prior.shape != (2,):
            raise ValueError("context_open_prior must have shape [2]")
        door_probability = np.clip(prior[context], 0.02, 0.98)
    if not condition_on_context:
        door_probability[:] = float(np.mean(door_probability))
    draws = rng.random((samples, n, *base.shape[1:])) < base[None]
    for sample_index in range(samples):
        door_open = rng.random(n) < door_probability
        for item in range(n):
            column = int(wall_columns[item])
            column_unknown = unknown[item, :, column]
            # The side cells of the occluded strip are known to be support surface in the
            # generator.  Treating them as independent noise would create artificial radius-2
            # failures and confound topology with an unrelated marginal sampler.
            draws[sample_index, item, unknown[item]] = True
            draws[sample_index, item, column_unknown, column] = bool(door_open[item])
    known = observations[:, 2] < 0.5
    observed_free = observations[:, 0].astype(bool)
    draws = np.where(known[None], observed_free[None], draws)
    return draws


def event_probabilities_from_maps(
    maps: np.ndarray,
    scenes: dict[str, np.ndarray],
    radii: Radii,
) -> tuple[np.ndarray, np.ndarray]:
    """Return event probabilities and map marginals for ``maps[S,N,H,W]``."""

    if maps.ndim != 4:
        raise ValueError("maps must have shape [S,N,H,W]")
    sample_events = []
    for sample_index in range(maps.shape[0]):
        sample_events.append(exact_events(maps[sample_index], scenes["starts"], scenes["goals"], radii))
    sample_events_array = np.stack(sample_events, axis=0)
    return sample_events_array.mean(axis=0), maps.mean(axis=0)


def path_bottleneck_score(free_probability: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> float:
    """Maximum-minimum path score used only by the connectivity-loss baseline.

    This is explicitly a baseline feature, not a PathRel event probability.  The final prediction
    is calibrated on held-out training queries, while event labels for every method still use the
    exact binary oracle.
    """

    height, width = free_probability.shape
    values = np.full_like(free_probability, -np.inf, dtype=np.float64)
    values[start] = float(free_probability[start])
    queue: list[tuple[float, int, int]] = [(-values[start], start[0], start[1])]
    while queue:
        negative_capacity, row, col = heapq.heappop(queue)
        capacity = -negative_capacity
        if capacity < values[row, col] - 1e-12:
            continue
        if (row, col) == goal:
            return float(capacity)
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            candidate = min(capacity, float(free_probability[n_row, n_col]))
            if candidate > values[n_row, n_col]:
                values[n_row, n_col] = candidate
                heapq.heappush(queue, (-candidate, n_row, n_col))
    return 0.0


class DirectQueryMLP:
    """Tiny NumPy MLP for the direct ``q(s,g,r)`` baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.2, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, 0.2, hidden_dim)
        self.b2 = 0.0

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))

    def fit(self, features: np.ndarray, targets: np.ndarray, *, epochs: int = 240, lr: float = 0.08) -> None:
        targets = targets.astype(np.float64)
        for _ in range(epochs):
            hidden_pre = features @ self.w1 + self.b1
            hidden = np.tanh(hidden_pre)
            probabilities = self._sigmoid(hidden @ self.w2 + self.b2)
            residual = probabilities - targets
            scale = 1.0 / len(features)
            grad_w2 = hidden.T @ residual * scale
            grad_b2 = float(np.mean(residual))
            grad_hidden = residual[:, None] * self.w2[None] * (1.0 - hidden * hidden)
            grad_w1 = features.T @ grad_hidden * scale
            grad_b1 = grad_hidden.mean(axis=0)
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1

    def predict(self, features: np.ndarray) -> np.ndarray:
        hidden = np.tanh(features @ self.w1 + self.b1)
        return self._sigmoid(hidden @ self.w2 + self.b2)


def query_features(scenes: dict[str, np.ndarray], radii: Radii) -> tuple[np.ndarray, np.ndarray]:
    observation = scenes["observation"]
    _, _, height, width = observation.shape
    rows: list[list[float]] = []
    for item in range(len(observation)):
        context = int(scenes["context"][item])
        visible_free = float(np.mean(observation[item, 0]))
        unknown_fraction = float(np.mean(observation[item, 2]))
        for query_index, (start, goal) in enumerate(zip(scenes["starts"][item], scenes["goals"][item])):
            for radius in radii:
                rows.append(
                    [
                        float(context == 0),
                        float(context == 1),
                        float(query_index),
                        float(start[0]) / max(1, height - 1),
                        float(start[1]) / max(1, width - 1),
                        float(goal[0]) / max(1, height - 1),
                        float(goal[1]) / max(1, width - 1),
                        float(radius) / max(1, max(radii)),
                        visible_free,
                        unknown_fraction,
                    ]
                )
    features = np.asarray(rows, dtype=np.float64)
    return features, scenes["events"].reshape(-1).astype(np.float64)


def calibrated_connectivity_predictions(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray], radii: Radii
) -> tuple[np.ndarray, np.ndarray]:
    probability_maps = hidden_marginals(train)

    def raw_scores(split: dict[str, np.ndarray]) -> np.ndarray:
        values: list[float] = []
        for item in range(len(split["context"])):
            p_map = probability_maps[int(split["context"][item])]
            query_radius_scores: list[np.ndarray] = []
            for radius in radii:
                # Erode with the exact disk footprint before computing the baseline score.
                if radius:
                    offsets = [
                        (dr, dc)
                        for dr in range(-radius, radius + 1)
                        for dc in range(-radius, radius + 1)
                        if dr * dr + dc * dc <= radius * radius
                    ]
                    eroded = np.zeros_like(p_map)
                    for row in range(p_map.shape[0]):
                        for col in range(p_map.shape[1]):
                            eroded[row, col] = min(
                                (
                                    p_map[row + dr, col + dc]
                                    if 0 <= row + dr < p_map.shape[0]
                                    and 0 <= col + dc < p_map.shape[1]
                                    else 0.0
                                )
                                for dr, dc in offsets
                            )
                else:
                    eroded = p_map
                query_radius_scores.append(
                    merge_tree_bottleneck_scores(
                        eroded, split["starts"][item], split["goals"][item]
                    )
                )
            values.extend(np.stack(query_radius_scores, axis=-1).reshape(-1))
        return np.asarray(values)

    train_scores = raw_scores(train)
    test_scores = raw_scores(test)
    # One-dimensional logistic calibration is a small edge-connectivity predictor.  It cannot
    # represent joint map uncertainty, which is exactly the comparison the death test is meant to
    # expose.
    model = DirectQueryMLP(1, hidden_dim=8, seed=31)
    model.fit(train_scores[:, None], train["events"].reshape(-1), epochs=180, lr=0.12)
    return model.predict(test_scores[:, None]).reshape(test["events"].shape), test_scores


def write_reliability_svg(path: Path, results: dict[str, dict[str, Any]]) -> None:
    width, height = 1100, 650
    left, top, plot_w, plot_h = 80, 50, 430, 500
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:12px}</style>',
        '<text x="80" y="25" font-size="18">P0 reachability reliability (template-held-out test)</text>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot_w
        y = top + plot_h - value * plot_h
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{x-8:.1f}" y="{top+plot_h+20}">{value:.1f}</text>')
        lines.append(f'<text x="{left-28}" y="{y+4:.1f}">{value:.1f}</text>')
    lines += [
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#555" stroke-dasharray="5,5"/>',
        f'<text x="{left+plot_w/2-65}" y="{top+plot_h+45}">mean predicted probability</text>',
        f'<text transform="translate(18 {top+plot_h/2+80}) rotate(-90)">empirical event frequency</text>',
    ]
    for model_index, (name, report) in enumerate(results.items()):
        diagram = report["reliability"]
        points = []
        for item in diagram:
            if item["count"]:
                x = left + float(item["confidence"]) * plot_w
                y = top + plot_h - float(item["accuracy"]) * plot_h
                points.append(f"{x:.1f},{y:.1f}")
        if len(points) >= 1:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[model_index % len(colors)]}" stroke-width="2"/>')
        legend_y = top + 20 + model_index * 22
        lines.append(f'<line x1="570" y1="{legend_y-5}" x2="595" y2="{legend_y-5}" stroke="{colors[model_index % len(colors)]}" stroke-width="3"/>')
        lines.append(f'<text class="legend" x="605" y="{legend_y}">{name} (Brier {report["brier"]:.3f})</text>')
    lines += [
        '<text x="570" y="430" font-size="16">Interpretation</text>',
        '<text x="570" y="455">A curve close to the dashed diagonal is calibrated.</text>',
        '<text x="570" y="478">High-confidence points below the diagonal are false-safe.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_svg(path: Path, results: dict[str, dict[str, Any]]) -> None:
    """Write a dependency-free grouped-bar comparison for the headline event metrics."""

    width, height = 1320, 760
    methods = list(results)
    metrics = (("brier", "Reachability Brier"), ("nll", "Reachability NLL"), ("ece", "Reachability ECE"))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
    panel_left, panel_top, panel_width, panel_height, gap = 70, 70, 370, 545, 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:11px}</style>',
        '<text x="70" y="30" font-size="20">P0 event-metric comparison (template-held-out test)</text>',
    ]
    for panel_index, (metric, title) in enumerate(metrics):
        left = panel_left + panel_index * (panel_width + gap)
        bottom = panel_top + panel_height
        values = [float(results[name][metric]) for name in methods]
        maximum = max(0.1, max(values) * 1.15)
        lines.append(f'<text x="{left}" y="{panel_top-22}" font-size="16">{title}</text>')
        for tick in range(6):
            value = maximum * tick / 5.0
            y = bottom - (value / maximum) * panel_height
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}"/>')
            lines.append(f'<text x="{left-42}" y="{y+4:.1f}">{value:.2f}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left+panel_width}" y2="{bottom}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{panel_top}" x2="{left}" y2="{bottom}"/>')
        slot = panel_width / max(1, len(methods))
        bar_width = slot * 0.68
        for method_index, (name, value) in enumerate(zip(methods, values)):
            x = left + method_index * slot + (slot - bar_width) / 2
            bar_height = (value / maximum) * panel_height
            y = bottom - bar_height
            color = colors[method_index % len(colors)]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
            lines.append(f'<text class="label" transform="translate({x+bar_width/2:.1f} {bottom+12}) rotate(55)" text-anchor="start">{name}</text>')
            lines.append(f'<text class="label" x="{x+bar_width/2:.1f}" y="{max(panel_top+12, y-5):.1f}" text-anchor="middle">{value:.3f}</text>')
    lines.append('<text x="70" y="738">Lower is better. Exact values are recorded in metrics.csv and report.json.</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.worlds_per_template < 4 or args.posterior_samples < 2:
        raise ValueError("worlds-per-template must be >=4 and posterior-samples >=2")
    radii: Radii = (0, 1, 2)
    rng = np.random.default_rng(args.seed)
    train_scenes = make_split(
        range(args.train_templates), worlds_per_template=args.worlds_per_template,
        height=args.height, width=args.width, radii=radii
    )
    test_scenes = make_split(
        range(args.train_templates, args.train_templates + args.test_templates),
        worlds_per_template=args.worlds_per_template, height=args.height, width=args.width, radii=radii
    )
    train, test = stack(train_scenes), stack(test_scenes)
    probability_maps = hidden_marginals(train)
    context_open_prior = np.asarray(
        [np.mean(train["door_open"][train["context"] == family]) for family in (0, 1)],
        dtype=np.float64,
    )

    # All methods receive exactly the same test query set.
    constant = np.broadcast_to(train["events"].mean(axis=0), test["events"].shape).copy()
    independent_map_samples = independent_samples(
        probability_maps, test["observation"], test["context"], samples=args.posterior_samples, rng=rng
    )
    independent, independent_marginals = event_probabilities_from_maps(independent_map_samples, test, radii)
    random_map_samples = independent_samples(
        np.full_like(probability_maps, 0.5), test["observation"], test["context"], samples=args.posterior_samples, rng=rng
    )
    random_completion, random_marginals = event_probabilities_from_maps(random_map_samples, test, radii)
    deterministic_map = probability_maps[test["context"]] >= 0.5
    deterministic_events = exact_events(deterministic_map, test["starts"], test["goals"], radii)
    deterministic = deterministic_events.reshape(test["events"].shape)

    coherent_map_samples = coherent_samples(
        probability_maps, test["observation"], test["context"], samples=args.posterior_samples, rng=rng,
        context_open_prior=context_open_prior,
    )
    correlated, correlated_marginals = event_probabilities_from_maps(coherent_map_samples, test, radii)
    no_reach_samples = coherent_samples(
        probability_maps, test["observation"], test["context"], samples=args.posterior_samples,
        rng=rng, condition_on_context=False, context_open_prior=context_open_prior
    )
    no_reach, no_reach_marginals = event_probabilities_from_maps(no_reach_samples, test, radii)

    direct_features_train, direct_targets_train = query_features(train, radii)
    direct_features_test, _ = query_features(test, radii)
    direct_model = DirectQueryMLP(direct_features_train.shape[1], seed=args.seed + 1)
    direct_model.fit(direct_features_train, direct_targets_train)
    direct = direct_model.predict(direct_features_test).reshape(test["events"].shape)
    connectivity, connectivity_raw = calibrated_connectivity_predictions(train, test, radii)

    method_probabilities: dict[str, np.ndarray] = {
        "constant_query_radius": constant,
        "independent_bernoulli": independent,
        "direct_query_mlp": direct,
        "edge_connectivity_calibrated": connectivity,
        "random_completion": random_completion,
        "deterministic_threshold": deterministic,
        "correlated_decoder_no_reachability": no_reach,
        "PathRel_correlated_event": correlated,
    }
    map_marginals: dict[str, np.ndarray | None] = {
        "constant_query_radius": None,
        "independent_bernoulli": independent_marginals,
        "direct_query_mlp": None,
        "edge_connectivity_calibrated": None,
        "random_completion": random_marginals,
        "deterministic_threshold": deterministic_map.astype(np.float64),
        "correlated_decoder_no_reachability": no_reach_marginals,
        "PathRel_correlated_event": correlated_marginals,
    }
    reports: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for name, probabilities in method_probabilities.items():
        report = summarize(probabilities, test["events"])
        report["by_radius"] = [summarize(probabilities[..., index], test["events"][..., index]) for index in range(len(radii))]
        if map_marginals[name] is not None:
            map_report = summarize(map_marginals[name], test["target_free"].astype(np.float64))
            report["map_marginal"] = {key: value for key, value in map_report.items() if key != "reliability"}
        else:
            report["map_marginal"] = None
        reports[name] = report
        csv_rows.append({"method": name, "scope": "event", "radius": "all", **{key: value for key, value in report.items() if isinstance(value, (float, int))}})
        for index, radius_report in enumerate(report["by_radius"]):
            csv_rows.append({"method": name, "scope": "event", "radius": radii[index], **{key: value for key, value in radius_report.items() if isinstance(value, (float, int))}})

    test_open_rate = {str(family): float(np.mean(test["door_open"][test["context"] == family])) for family in (0, 1)}
    predicted_open_rate = {
        name: {
            str(family): float(np.mean(coherent_map_samples[:, test["context"] == family, args.height // 2, args.width // 2]))
            for family in (0, 1)
        }
        for name in ("PathRel_correlated_event", "correlated_decoder_no_reachability")
    }
    predicted_open_rate["correlated_decoder_no_reachability"] = {
        str(family): float(np.mean(no_reach_samples[:, test["context"] == family, args.height // 2, args.width // 2]))
        for family in (0, 1)
    }
    conflict = {
        "description": "same voxel marginals, different joint topology",
        "marginal_free_probability": 0.5,
        "joint_open_probability": 0.5,
        "independent_joint_open_probability": 0.25,
        "gap": 0.25,
    }
    query_audit = {
        "test_positive_rate_by_radius": test["events"].mean(axis=(0, 1)).tolist(),
        "test_reachable_at_radius_zero": float(np.mean(test["events"][..., 0])),
        "test_disconnected_at_radius_zero": float(np.mean(test["max_clearance"] < 0)),
        "test_narrow_bottleneck_max_clearance_le_2": float(
            np.mean((test["max_clearance"] >= 0) & (test["max_clearance"] <= 2))
        ),
        "query_sampling_uses": "visible observation, context token, sampled visible endpoints, and footprint radii only",
        "query_sampling_does_not_use": "test target map, test doorway state, or test event label",
    }
    report = {
        "protocol": {
            "seed": args.seed,
            "radii_cells": list(radii),
            "train_templates": list(range(args.train_templates)),
            "test_templates": list(range(args.train_templates, args.train_templates + args.test_templates)),
            "worlds_per_template": args.worlds_per_template,
            "posterior_samples": args.posterior_samples,
            "split_rule": "scene template held out; no adjacent-frame/random leakage",
            "context_open_probabilities": {"0": 0.2, "1": 0.8},
            "command": "PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "event_metrics": reports,
        "door_joint_frequency": {"realized_test_open_rate": test_open_rate, "predicted_wall_open_rate": predicted_open_rate},
        "same_marginal_conflict": conflict,
        "query_audit": query_audit,
        "flatlands_baseline": {"status": "not_run", "reason": "FlatLands data is not present in this transfer package; audit is required before any real-data claim."},
        "scope_baseline": {
            "status": "surrogate_random_completion",
            "method_row": "random_completion",
            "reason": "SCOPE is not vendored in the isolated package; this row is a transparent uniform-unknown completion stress test, not an implementation of SCOPE.",
        },
        "death_test": {
            "criterion": "PathRel must beat both independent cells and direct q on Brier; ECE may be at most 0.02 worse than the best comparator, with map marginal Brier within 0.02 of independent cells.",
            "independent_brier": reports["independent_bernoulli"]["brier"],
            "direct_query_brier": reports["direct_query_mlp"]["brier"],
            "pathrel_brier": reports["PathRel_correlated_event"]["brier"],
            "independent_ece": reports["independent_bernoulli"]["ece"],
            "direct_query_ece": reports["direct_query_mlp"]["ece"],
            "pathrel_ece": reports["PathRel_correlated_event"]["ece"],
            "independent_map_brier": reports["independent_bernoulli"]["map_marginal"]["brier"],
            "pathrel_map_brier": reports["PathRel_correlated_event"]["map_marginal"]["brier"],
            "passed_oracle_proxy": bool(
                reports["PathRel_correlated_event"]["brier"] < min(
                    reports["independent_bernoulli"]["brier"],
                    reports["direct_query_mlp"]["brier"],
                )
                and reports["PathRel_correlated_event"]["ece"] <= min(
                    reports["independent_bernoulli"]["ece"],
                    reports["direct_query_mlp"]["ece"],
                ) + 0.02
                and reports["PathRel_correlated_event"]["map_marginal"]["brier"]
                <= reports["independent_bernoulli"]["map_marginal"]["brier"] + 0.02
            ),
            "interpretation": "This is an oracle correlated-posterior diagnostic, not a trained neural result; train_synthetic.py must be run on CUDA before claiming the learned model passes.",
        },
        "connectivity_baseline_raw_score_summary": {
            "train_calibrated_test_mean": float(np.mean(connectivity_raw)),
            "train_calibrated_test_std": float(np.std(connectivity_raw)),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["method", "scope", "radius", "brier", "nll", "ece", "false_safe_rate@0.8", "count"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in csv_rows)
    write_reliability_svg(args.output_dir / "reliability.svg", reports)
    write_comparison_svg(args.output_dir / "comparison.svg", reports)
    print(json.dumps({"output_dir": str(args.output_dir), "death_test": report["death_test"], "event_metrics": {name: {key: value for key, value in item.items() if isinstance(value, (float, int))} for name, item in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
