#!/usr/bin/env python3
"""Render a checkpoint-derived FlatLands correlation comparison for the project page.

The figure is deliberately validation-only.  It consumes support-consistent prediction manifests,
recomputes two examples from real K=128 checkpoints, records every input hash, and marks whether the
source checkpoints were cleanly retrained or only post-hoc corrected.  No test split or test-label
path is accepted.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import textwrap
from typing import Any

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:  # pragma: no cover - depends on the execution environment
    raise SystemExit("Pillow is required to render the comparison image") from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.flatlands_data import FlatLandsReplayDataset  # noqa: E402
from pathrel.flatlands_eval import join_flatlands_predictions  # noqa: E402
from pathrel.flatlands_query import sha256_path  # noqa: E402
from pathrel.gpu_diagnostics import cuda_unavailable_message  # noqa: E402
from pathrel.labels import clearance_radius_map  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402


SEEDS = (20260831, 20260901, 20260902)
CORRELATED = "correlated"
INDEPENDENT = "independent"
VARIANTS = (CORRELATED, INDEPENDENT)
CANVAS_SIZE = (1800, 1330)

INK = "#17242f"
MUTED = "#61707c"
LIGHT_MUTED = "#84909a"
LINE = "#dce4e9"
TEAL = "#258b72"
TEAL_LIGHT = "#e7f5f0"
ORANGE = "#c87434"
ORANGE_LIGHT = "#fff1e6"
BLUE = "#315b86"
BACKGROUND = "#f5f8fa"
CARD = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("results/p1_flatlands_support_clamped_posthoc_k128_validation"),
    )
    parser.add_argument("--correlated-evaluation-root", type=Path, default=None)
    parser.add_argument("--independent-evaluation-root", type=Path, default=None)
    parser.add_argument("--correlated-run-suffix", default=None)
    parser.add_argument("--independent-run-suffix", default=None)
    parser.add_argument("--paired-report", type=Path, default=None)
    parser.add_argument(
        "--evidence-contract",
        choices=("posthoc-support-clamp", "clean-support-training"),
        default="posthoc-support-clamp",
    )
    parser.add_argument(
        "--correlated-checkpoint-root",
        type=Path,
        default=Path("results/p1_flatlands_conpath_k128_validation_v2"),
    )
    parser.add_argument(
        "--independent-checkpoint-root",
        type=Path,
        default=Path("results/p1_flatlands_independent_k128_validation_v2"),
    )
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
    parser.add_argument("--canonical-seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sample-chunk", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/assets/flatlands_k128_support_clamped_advantage.png"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("site/data/flatlands_k128_support_clamped_advantage.json"),
    )
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(
    run_roots: dict[str, Path],
    run_suffixes: dict[str, str],
    paired_path: Path,
    evidence_contract: str,
    selection: Path,
    queries: Path,
) -> tuple[dict[tuple[str, int], dict[tuple[str, int, int], Any]], dict[str, Any]]:
    records: dict[tuple[str, int], dict[tuple[str, int, int], Any]] = {}
    provenance: dict[str, Any] = {"prediction_manifests": []}
    reference_keys: set[tuple[str, int, int]] | None = None
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = run_roots[variant] / f"seed{seed}_{run_suffixes[variant]}"
            run_path = run_dir / "run.json"
            prediction_path = run_dir / "predictions_validation.csv"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run.get("test_evaluated") is not False:
                raise ValueError(f"run is not explicitly test-locked: {run_path}")
            if evidence_contract == "posthoc-support-clamp":
                valid_contract = (
                    run.get("posthoc_checkpoint_evaluation") is True
                    and run.get("retraining_required") is True
                    and run.get("forward", {}).get("invalid_support_policy")
                    == "epistemic_mask complement clamped to blocked before sampling"
                )
            else:
                valid_contract = (
                    run.get("kind") == "flatlands_conpath_training"
                    and run.get("validation_result") is True
                    and run.get("protocol_version")
                    == "P1_BASELINE_PROTOCOL.md v1 + ConPath valid-support v2"
                    and run.get("forward", {}).get("invalid_support_clamped") is True
                )
            if not valid_contract:
                raise ValueError(f"invalid {evidence_contract} run contract: {run_path}")
            joined, radii = join_flatlands_predictions(
                prediction_path,
                selection,
                queries,
                split="validation",
            )
            if tuple(radii) != (0, 10, 20) or len(joined) != 4224:
                raise ValueError(f"unexpected validation coverage: {prediction_path}")
            by_key = {record.key: record for record in joined}
            if len(by_key) != len(joined):
                raise ValueError(f"duplicate prediction keys: {prediction_path}")
            if reference_keys is None:
                reference_keys = set(by_key)
            elif set(by_key) != reference_keys:
                raise ValueError("prediction manifests do not share identical event keys")
            records[(variant, seed)] = by_key
            provenance["prediction_manifests"].append(
                {
                    "variant": variant,
                    "seed": seed,
                    "path": str(prediction_path),
                    "sha256": sha256_path(prediction_path),
                    "rows": len(joined),
                }
            )
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("test_evaluated") is not False:
        raise ValueError("paired comparison is not explicitly test-locked")
    if evidence_contract == "posthoc-support-clamp":
        valid_paired = (
            paired.get("retraining_required") is True
            and paired.get("posthoc_checkpoint_evaluation") is True
            and paired.get("protocol", {}).get("support_policy")
            == "invalid-clamped-posthoc"
        )
    else:
        valid_paired = (
            paired.get("retraining_required") is False
            and paired.get("posthoc_checkpoint_evaluation") is False
            and paired.get("clean_support_training") is True
            and paired.get("protocol", {}).get("support_policy")
            == "valid-support-clean-training"
        )
    if not valid_paired:
        raise ValueError(f"paired comparison is not the {evidence_contract} contract")
    provenance["paired_comparison"] = {
        "path": str(paired_path),
        "sha256": sha256_path(paired_path),
    }
    return records, {"provenance": provenance, "paired": paired}


def _select_cases(
    records: dict[tuple[str, int], dict[tuple[str, int, int], Any]],
) -> list[dict[str, Any]]:
    keys = sorted(records[(CORRELATED, SEEDS[0])])

    def probabilities(variant: str, key: tuple[str, int, int]) -> list[float]:
        return [float(records[(variant, seed)][key].probability) for seed in SEEDS]

    positive: list[tuple[float, tuple[str, int, int]]] = []
    false_safe: list[tuple[float, tuple[str, int, int]]] = []
    for key in keys:
        record = records[(CORRELATED, SEEDS[0])][key]
        correlated = probabilities(CORRELATED, key)
        independent = probabilities(INDEPENDENT, key)
        target = int(record.target)
        brier_gain = statistics.mean(
            (independent[index] - target) ** 2
            - (correlated[index] - target) ** 2
            for index in range(len(SEEDS))
        )
        if target == 1 and all(
            correlated[index] > independent[index] for index in range(len(SEEDS))
        ):
            positive.append((brier_gain, key))
        if (
            target == 0
            and all(value >= 0.8 for value in independent)
            and all(value < 0.8 for value in correlated)
        ):
            false_safe.append((brier_gain, key))
    if not positive or not false_safe:
        raise RuntimeError("the predeclared validation example rules returned no cases")
    positive.sort(key=lambda item: (-item[0], item[1]))
    false_safe.sort(key=lambda item: (-item[0], item[1]))

    specifications = (
        (
            "positive_recovery",
            positive[0],
            len(positive),
            "target=1, correlated probability exceeds independent probability in all three seeds; rank by mean per-event Brier gain",
            "Recovers a real path",
            "The independent decoder nearly rules the path out in every seed.",
        ),
        (
            "false_safe_avoided",
            false_safe[0],
            len(false_safe),
            "target=0, independent probability >=0.8 and correlated probability <0.8 in all three seeds; rank by mean per-event Brier gain",
            "Avoids a false-safe decision",
            "The independent decoder is confidently unsafe-to-trust in every seed.",
        ),
    )
    output: list[dict[str, Any]] = []
    for case_id, (gain, key), eligible_count, rule, title, description in specifications:
        record = records[(CORRELATED, SEEDS[0])][key]
        correlated = probabilities(CORRELATED, key)
        independent = probabilities(INDEPENDENT, key)
        output.append(
            {
                "case_id": case_id,
                "key": key,
                "global_id": key[0],
                "candidate_index": key[1],
                "radius_cells": key[2],
                "target": int(record.target),
                "source_dataset": record.source_dataset,
                "scene_id": record.scene_id,
                "title": title,
                "description": description,
                "selection_rule": rule,
                "eligible_count": eligible_count,
                "rank": 1,
                "mean_brier_gain": gain,
                "event_probability": {
                    CORRELATED: dict(zip((str(seed) for seed in SEEDS), correlated)),
                    INDEPENDENT: dict(zip((str(seed) for seed in SEEDS), independent)),
                },
                "mean_event_probability": {
                    CORRELATED: statistics.mean(correlated),
                    INDEPENDENT: statistics.mean(independent),
                },
            }
        )
    return output


def _checkpoint_path(root: Path, variant: str, seed: int) -> Path:
    suffix = "conpath" if variant == CORRELATED else "independent"
    return root / f"seed{seed}_{suffix}" / "best.pt"


def _load_checkpoint_model(
    path: Path,
    variant: str,
    device: torch.device,
) -> tuple[PathRelNet, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    if str(config.get("decoder_variant", "correlated")) != variant:
        raise ValueError(f"decoder variant mismatch in {path}")
    model = PathRelNet(
        input_channels=3,
        feature_channels=int(config.get("feature_channels", 32)),
        latent_dim=int(config.get("latent_dim", 8)),
        local_kernel_size=1 if variant == INDEPENDENT else 5,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    metadata = dict(config)
    metadata["checkpoint_protocol_version"] = state.get("protocol_version")
    return model, metadata


def _posterior_visuals(
    model: PathRelNet,
    sample: Any,
    *,
    variant: str,
    samples: int,
    sample_chunk: int,
    sampling_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    observation = torch.from_numpy(sample.input_bev[None]).to(
        device=device, dtype=torch.float32
    )
    support = torch.from_numpy(sample.epistemic_mask[None]).to(
        device=device, dtype=torch.bool
    )
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    probability_sum = np.zeros(sample.epistemic_mask.shape, dtype=np.float64)
    retained_worlds: list[np.ndarray] = []
    with torch.inference_mode():
        remaining = samples
        while remaining:
            current = min(sample_chunk, remaining)
            posterior = model(
                observation,
                valid_support_mask=support,
                num_samples=current,
                hard_samples=True,
                disable_global_factors=variant == INDEPENDENT,
                generator=generator,
            ).posterior
            probability_sum += posterior.conditional_class_probs[0, :, 0].sum(
                dim=0
            ).cpu().numpy()
            worlds = posterior.safe_samples()[0].cpu().numpy() > 0.5
            if np.any(worlds & ~sample.epistemic_mask[None]):
                raise RuntimeError("rendered posterior violates the valid-support clamp")
            for world in worlds:
                if len(retained_worlds) < 4:
                    retained_worlds.append(world.copy())
            remaining -= current
    return {
        "posterior_mean": (probability_sum / float(samples)).astype(np.float32),
        "sample_worlds": retained_worlds,
    }


def _event(world: np.ndarray, start: tuple[int, int], goal: tuple[int, int], radius: int) -> bool:
    clearance = clearance_radius_map(world)
    safe = clearance >= int(radius)
    if not safe[start] or not safe[goal]:
        return False
    stack = [start]
    visited = np.zeros(safe.shape, dtype=bool)
    visited[start] = True
    while stack:
        row, col = stack.pop()
        if (row, col) == goal:
            return True
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_row, n_col = row + d_row, col + d_col
            if (
                0 <= n_row < safe.shape[0]
                and 0 <= n_col < safe.shape[1]
                and safe[n_row, n_col]
                and not visited[n_row, n_col]
            ):
                visited[n_row, n_col] = True
                stack.append((n_row, n_col))
    return False


def _observed_rgb(sample: Any) -> np.ndarray:
    output = np.empty(sample.epistemic_mask.shape + (3,), dtype=np.uint8)
    output[:] = (219, 226, 231)
    valid = sample.epistemic_mask.astype(bool)
    output[valid] = (238, 242, 244)
    output[sample.observed_free.astype(bool)] = (79, 174, 147)
    output[sample.observed_blocked.astype(bool)] = (42, 56, 68)
    return output


def _reference_rgb(sample: Any) -> np.ndarray:
    output = np.empty(sample.epistemic_mask.shape + (3,), dtype=np.uint8)
    output[:] = (219, 226, 231)
    valid = sample.epistemic_mask.astype(bool)
    output[valid] = (43, 58, 70)
    output[sample.target_free.astype(bool)] = (112, 192, 166)
    return output


def _probability_rgb(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    low = np.asarray((20, 52, 73), dtype=np.float32)
    middle = np.asarray((31, 151, 139), dtype=np.float32)
    high = np.asarray((247, 190, 73), dtype=np.float32)
    t_low = np.minimum(values * 2.0, 1.0)[..., None]
    t_high = np.maximum((values - 0.5) * 2.0, 0.0)[..., None]
    output = low + (middle - low) * t_low
    upper = middle + (high - middle) * t_high
    output = np.where((values >= 0.5)[..., None], upper, output)
    output = np.rint(output).astype(np.uint8)
    output[~valid.astype(bool)] = (219, 226, 231)
    return output


def _world_rgb(world: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.empty(world.shape + (3,), dtype=np.uint8)
    output[:] = (219, 226, 231)
    output[valid.astype(bool)] = (43, 58, 70)
    output[np.asarray(world, dtype=bool)] = (78, 174, 147)
    return output


def _draw_map(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    rgb: np.ndarray,
    *,
    x: int,
    y: int,
    size: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    radius_cells: int,
    event: bool | None = None,
) -> None:
    image = Image.fromarray(rgb, mode="RGB").resize(
        (size, size), resample=Image.Resampling.NEAREST
    )
    canvas.paste(image, (x, y))
    border = BLUE if event is True else (ORANGE if event is False else LINE)
    draw.rectangle((x, y, x + size, y + size), outline=border, width=3 if event is not None else 2)
    height, width = rgb.shape[:2]
    for point, color, label in ((start, BLUE, "S"), (goal, ORANGE, "G")):
        px = x + (point[1] + 0.5) * size / width
        py = y + (point[0] + 0.5) * size / height
        marker_radius = max(4, int(round(size / 42)))
        footprint_radius = radius_cells * size / width
        if footprint_radius > marker_radius + 2:
            draw.ellipse(
                (
                    px - footprint_radius,
                    py - footprint_radius,
                    px + footprint_radius,
                    py + footprint_radius,
                ),
                outline="#ffffff",
                width=max(1, size // 125),
            )
        draw.ellipse(
            (
                px - marker_radius,
                py - marker_radius,
                px + marker_radius,
                py + marker_radius,
            ),
            fill=color,
            outline="#ffffff",
            width=max(1, size // 120),
        )
        if size >= 180:
            marker_font = _font(max(9, size // 25), bold=True)
            bbox = draw.textbbox((0, 0), label, font=marker_font)
            draw.text(
                (px - (bbox[2] - bbox[0]) / 2, py - (bbox[3] - bbox[1]) / 2 - 1),
                label,
                font=marker_font,
                fill="#ffffff",
            )


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    correlated: str,
    independent: str,
    footer: str,
    correlated_label: str = "ConPath",
    independent_label: str = "indep.",
    labels_below: bool = False,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=CARD, outline=LINE, width=2)
    x0, y0, x1, _ = box
    draw.text((x0 + 25, y0 + 18), title, font=_font(17, bold=True), fill=MUTED)
    value_y = y0 + (42 if labels_below else 48)
    draw.text((x0 + 25, value_y), correlated, font=_font(35, bold=True), fill=TEAL)
    draw.text((x0 + 305, value_y), independent, font=_font(35, bold=True), fill=ORANGE)
    if labels_below:
        draw.text((x0 + 27, y0 + 82), correlated_label, font=_font(13, bold=True), fill=TEAL)
        draw.text((x0 + 307, y0 + 82), independent_label, font=_font(13, bold=True), fill=ORANGE)
    else:
        draw.text((x0 + 191, y0 + 61), correlated_label, font=_font(15, bold=True), fill=TEAL)
        draw.text((x0 + 440, y0 + 61), independent_label, font=_font(15, bold=True), fill=ORANGE)
    draw.line((x0 + 25, y0 + 101, x1 - 25, y0 + 101), fill=LINE, width=1)
    draw.text((x0 + 25, y0 + 109), footer, font=_font(14, bold=True), fill=INK)


def _draw_probability_bars(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    correlated: list[float],
    independent: list[float],
) -> None:
    draw.text((x, y), "EVENT PROBABILITY BY TRAINING SEED", font=_font(13, bold=True), fill=LIGHT_MUTED)
    draw.rounded_rectangle((x, y + 27, x + 12, y + 39), radius=4, fill=TEAL)
    draw.text((x + 19, y + 23), "ConPath", font=_font(13), fill=MUTED)
    draw.rounded_rectangle((x + 98, y + 27, x + 110, y + 39), radius=4, fill=ORANGE)
    draw.text((x + 117, y + 23), "Independent", font=_font(13), fill=MUTED)
    track_x = x + 70
    track_width = 226
    for index, seed in enumerate(SEEDS):
        row_y = y + 58 + index * 53
        draw.text((x, row_y + 7), str(seed)[-4:], font=_font(13, bold=True), fill=MUTED)
        draw.rounded_rectangle(
            (track_x, row_y, track_x + track_width, row_y + 10),
            radius=5,
            fill="#e9eef1",
        )
        draw.rounded_rectangle(
            (track_x, row_y, track_x + max(2, int(track_width * correlated[index])), row_y + 10),
            radius=5,
            fill=TEAL,
        )
        draw.rounded_rectangle(
            (track_x, row_y + 17, track_x + track_width, row_y + 27),
            radius=5,
            fill="#e9eef1",
        )
        draw.rounded_rectangle(
            (track_x, row_y + 17, track_x + max(2, int(track_width * independent[index])), row_y + 27),
            radius=5,
            fill=ORANGE,
        )
        draw.text(
            (track_x + track_width + 9, row_y - 5),
            f"{correlated[index]:.3f}",
            font=_font(12, bold=True),
            fill=TEAL,
        )
        draw.text(
            (track_x + track_width + 9, row_y + 12),
            f"{independent[index]:.3f}",
            font=_font(12, bold=True),
            fill=ORANGE,
        )


def _draw_case(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    case: dict[str, Any],
    sample: Any,
    query: Any,
    visuals: dict[str, dict[str, Any]],
    letter: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=22, fill=CARD, outline=LINE, width=2)
    draw.rounded_rectangle((x0 + 24, y0 + 24, x0 + 62, y0 + 62), radius=10, fill=INK)
    draw.text((x0 + 36, y0 + 28), letter, font=_font(20, bold=True), fill="#ffffff")
    draw.text((x0 + 78, y0 + 23), case["title"], font=_font(27, bold=True), fill=INK)
    target_text = "TARGET: PATH EXISTS" if case["target"] else "TARGET: NO PATH"
    target_fill = TEAL_LIGHT if case["target"] else ORANGE_LIGHT
    target_color = TEAL if case["target"] else ORANGE
    draw.rounded_rectangle((x0 + 78, y0 + 61, x0 + 268, y0 + 89), radius=14, fill=target_fill)
    draw.text((x0 + 91, y0 + 66), target_text, font=_font(12, bold=True), fill=target_color)
    draw.text((x0 + 24, y0 + 105), case["description"], font=_font(15), fill=MUTED)
    details = (
        f"{case['source_dataset']} · {case['scene_id']}\n"
        f"{case['global_id']} · candidate {case['candidate_index']} · r={case['radius_cells']} cells"
    )
    draw.multiline_text((x0 + 24, y0 + 132), details, font=_font(12), fill=LIGHT_MUTED, spacing=5)
    probabilities = {
        variant: [case["event_probability"][variant][str(seed)] for seed in SEEDS]
        for variant in VARIANTS
    }
    _draw_probability_bars(
        draw,
        x=x0 + 24,
        y=y0 + 188,
        correlated=probabilities[CORRELATED],
        independent=probabilities[INDEPENDENT],
    )

    start = (int(query.start_row), int(query.start_col))
    goal = (int(query.goal_row), int(query.goal_col))
    radius = int(case["radius_cells"])
    map_y = y0 + 119
    observed_x = x0 + 438
    correlated_x = x0 + 715
    independent_x = x0 + 1040
    reference_x = x0 + 1365

    draw.text((observed_x, y0 + 82), "OBSERVED", font=_font(14, bold=True), fill=MUTED)
    draw.text((correlated_x, y0 + 82), "CONPATH · CORRELATED", font=_font(14, bold=True), fill=TEAL)
    draw.text((independent_x, y0 + 82), "INDEPENDENT CONTROL", font=_font(14, bold=True), fill=ORANGE)
    draw.text((reference_x, y0 + 82), "REFERENCE", font=_font(14, bold=True), fill=MUTED)

    _draw_map(
        canvas,
        draw,
        _observed_rgb(sample),
        x=observed_x,
        y=map_y,
        size=250,
        start=start,
        goal=goal,
        radius_cells=radius,
    )
    _draw_map(
        canvas,
        draw,
        _probability_rgb(
            visuals[CORRELATED]["posterior_mean"], sample.epistemic_mask
        ),
        x=correlated_x,
        y=map_y,
        size=200,
        start=start,
        goal=goal,
        radius_cells=radius,
    )
    _draw_map(
        canvas,
        draw,
        _probability_rgb(
            visuals[INDEPENDENT]["posterior_mean"], sample.epistemic_mask
        ),
        x=independent_x,
        y=map_y,
        size=200,
        start=start,
        goal=goal,
        radius_cells=radius,
    )
    _draw_map(
        canvas,
        draw,
        _reference_rgb(sample),
        x=reference_x,
        y=map_y,
        size=250,
        start=start,
        goal=goal,
        radius_cells=radius,
        event=bool(case["target"]),
    )
    for variant, group_x in ((CORRELATED, correlated_x), (INDEPENDENT, independent_x)):
        events = visuals[variant]["sample_events"]
        for index, world in enumerate(visuals[variant]["sample_worlds"]):
            thumb_y = map_y + index * 51
            _draw_map(
                canvas,
                draw,
                _world_rgb(world, sample.epistemic_mask),
                x=group_x + 212,
                y=thumb_y,
                size=46,
                start=start,
                goal=goal,
                radius_cells=radius,
                event=events[index],
            )
            tag_color = BLUE if events[index] else ORANGE
            draw.rounded_rectangle(
                (group_x + 215, thumb_y + 3, group_x + 230, thumb_y + 18),
                radius=3,
                fill=tag_color,
            )
            draw.text(
                (group_x + 220, thumb_y + 3),
                "P" if events[index] else "N",
                font=_font(8, bold=True),
                fill="#ffffff",
            )
    draw.text((correlated_x, map_y + 210), "mean + first 4 samples · P path / N no path", font=_font(10), fill=LIGHT_MUTED)
    draw.text((independent_x, map_y + 210), "mean + first 4 samples · P path / N no path", font=_font(10), fill=LIGHT_MUTED)
    draw.text((observed_x, map_y + 260), "green free · charcoal blocked · pale unknown", font=_font(10), fill=LIGHT_MUTED)
    draw.text((reference_x, map_y + 260), "white rings show the queried footprint", font=_font(10), fill=LIGHT_MUTED)


def _aggregate_metrics(
    run_roots: dict[str, Path], run_suffixes: dict[str, str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in VARIANTS:
        seed_metrics: list[dict[str, float]] = []
        for seed in SEEDS:
            run = json.loads(
                (
                    run_roots[variant]
                    / f"seed{seed}_{run_suffixes[variant]}"
                    / "run.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
            overall = next(
                item["scene_weighted"]
                for item in run["evaluation"]["metrics"]
                if item["scope"] == "overall"
            )
            seed_metrics.append(overall)
        output[variant] = {}
        for metric in (
            "brier",
            "nll",
            "ece",
            "false_safe_rate@0.8",
            "high_confidence_safe_coverage@0.8",
        ):
            values = [float(item[metric]) for item in seed_metrics]
            output[variant][metric] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "seed_values": dict(zip((str(seed) for seed in SEEDS), values)),
            }
    return output


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.sample_chunk < 2 or args.samples % args.sample_chunk:
        raise SystemExit("samples must be divisible by sample-chunk and both must be at least two")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(cuda_unavailable_message(torch))

    clean_support_training = args.evidence_contract == "clean-support-training"
    run_roots = {
        CORRELATED: args.correlated_evaluation_root or args.evaluation_root,
        INDEPENDENT: args.independent_evaluation_root or args.evaluation_root,
    }
    run_suffixes = {
        CORRELATED: args.correlated_run_suffix
        or ("conpath" if clean_support_training else "correlated"),
        INDEPENDENT: args.independent_run_suffix or "independent",
    }
    paired_path = args.paired_report or args.evaluation_root / "paired_comparison.json"
    records, comparison = _load_predictions(
        run_roots,
        run_suffixes,
        paired_path,
        args.evidence_contract,
        args.selection,
        args.queries,
    )
    cases = _select_cases(records)
    aggregate = _aggregate_metrics(run_roots, run_suffixes)
    paired_delta = comparison["paired"]["aggregate_delta"]["overall"]["brier"]
    paired_per_seed_overall = [
        {
            "seed": int(seed_result["seed"]),
            "brier": seed_result["strata"]["overall"]["delta"]["brier"],
        }
        for seed_result in comparison["paired"]["seeds"]
    ]
    if [item["seed"] for item in paired_per_seed_overall] != list(SEEDS):
        raise ValueError("paired comparison seeds do not match the figure contract")
    if paired_delta["mean"] <= 0.0 or any(
        item["brier"]["bootstrap_95"] is None
        or item["brier"]["bootstrap_95"][0] <= 0.0
        for item in paired_per_seed_overall
    ):
        raise ValueError(
            "refusing to render an advantage figure without a positive aggregate delta and "
            "positive lower bootstrap bound in every seed"
        )

    device = torch.device(args.device)
    checkpoint_roots = {
        CORRELATED: args.correlated_checkpoint_root,
        INDEPENDENT: args.independent_checkpoint_root,
    }
    models: dict[str, PathRelNet] = {}
    checkpoints: dict[str, Any] = {}
    for variant in VARIANTS:
        path = _checkpoint_path(
            checkpoint_roots[variant], variant, args.canonical_seed
        )
        model, config = _load_checkpoint_model(path, variant, device)
        if clean_support_training and config.get("checkpoint_protocol_version") != (
            "P1_BASELINE_PROTOCOL.md v1 + ConPath valid-support v2"
        ):
            raise ValueError(f"clean figure checkpoint protocol mismatch: {path}")
        models[variant] = model
        checkpoints[variant] = {
            "path": str(path),
            "sha256": sha256_path(path),
            "training_seed": config.get("seed"),
            "feature_channels": config.get("feature_channels"),
            "latent_dim": config.get("latent_dim"),
            "decoder_variant": config.get("decoder_variant"),
            "protocol_version": config.get("checkpoint_protocol_version"),
        }

    dataset = FlatLandsReplayDataset(
        args.archive,
        args.selection,
        args.queries,
        split="validation",
        verify_frozen=True,
        verify_query_geometry=True,
    )
    observation_indices = {
        observation.global_id: index
        for index, observation in enumerate(dataset.observations)
    }
    rendered: list[tuple[dict[str, Any], Any, Any, dict[str, dict[str, Any]]]] = []
    try:
        for case_index, case in enumerate(cases):
            sample = dataset[observation_indices[case["global_id"]]]
            query = next(
                query
                for query in sample.retained_queries
                if query.candidate_index == case["candidate_index"]
            )
            if bool(query.reachable[sample.radii_cells.index(case["radius_cells"])]) != bool(case["target"]):
                raise RuntimeError("selected case target disagrees with frozen replay")
            visuals: dict[str, dict[str, Any]] = {}
            for variant in VARIANTS:
                visual = _posterior_visuals(
                    models[variant],
                    sample,
                    variant=variant,
                    samples=args.samples,
                    sample_chunk=args.sample_chunk,
                    sampling_seed=args.sampling_seed + case_index,
                    device=device,
                )
                start = (int(query.start_row), int(query.start_col))
                goal = (int(query.goal_row), int(query.goal_col))
                visual["sample_events"] = [
                    _event(world, start, goal, case["radius_cells"])
                    for world in visual["sample_worlds"]
                ]
                visuals[variant] = visual
            rendered.append((case, sample, query, visuals))
    finally:
        dataset.close()

    canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 49), "Spatial correlation improves path reliability", font=_font(48, bold=True), fill=INK)
    draw.text(
        (72, 111),
        "FlatLands provenance validation · K=128 posterior worlds · three matched training seeds",
        font=_font(21),
        fill=MUTED,
    )
    badge = (1320, 50, 1730, 96)
    draw.rounded_rectangle(badge, radius=23, fill=ORANGE_LIGHT, outline="#edcfb8", width=2)
    badge_text = (
        "CLEAN VALIDATION · TEST LOCKED"
        if clean_support_training
        else "POST-HOC CHECK · RETRAIN REQUIRED"
    )
    draw.text((1346, 62), badge_text, font=_font(15, bold=True), fill=ORANGE)

    correlated_brier = aggregate[CORRELATED]["brier"]["mean"]
    independent_brier = aggregate[INDEPENDENT]["brier"]["mean"]
    correlated_false_safe = aggregate[CORRELATED]["false_safe_rate@0.8"]["mean"]
    independent_false_safe = aggregate[INDEPENDENT]["false_safe_rate@0.8"]["mean"]
    brier_reduction = 100.0 * (independent_brier - correlated_brier) / independent_brier
    false_safe_reduction = 100.0 * (
        independent_false_safe - correlated_false_safe
    ) / independent_false_safe
    _draw_metric_card(
        draw,
        (70, 157, 600, 304),
        title="EVENT BRIER ↓ · mean across 3 seeds",
        correlated=f"{correlated_brier:.4f}",
        independent=f"{independent_brier:.4f}",
        footer=(
            f"{brier_reduction:.1f}% lower after clean support training"
            if clean_support_training
            else f"{brier_reduction:.1f}% lower after support clamping"
        ),
    )
    _draw_metric_card(
        draw,
        (635, 157, 1165, 304),
        title="FALSE-SAFE @ 0.8 ↓ · mean across 3 seeds",
        correlated=f"{correlated_false_safe:.4f}",
        independent=f"{independent_false_safe:.4f}",
        footer=f"{false_safe_reduction:.1f}% lower false-safe rate",
    )
    _draw_metric_card(
        draw,
        (1200, 157, 1730, 304),
        title="PAIRED SCENE BRIER Δ · independent − ConPath",
        correlated=f"+{paired_delta['mean']:.4f}",
        independent=f"± {paired_delta['sample_sd']:.4f}",
        footer="All 3 per-seed bootstrap 95% intervals > 0",
        correlated_label="mean Δ",
        independent_label="seed SD",
        labels_below=True,
    )

    _draw_case(
        canvas,
        draw,
        box=(70, 333, 1730, 767),
        case=rendered[0][0],
        sample=rendered[0][1],
        query=rendered[0][2],
        visuals=rendered[0][3],
        letter="A",
    )
    _draw_case(
        canvas,
        draw,
        box=(70, 792, 1730, 1226),
        case=rendered[1][0],
        sample=rendered[1][1],
        query=rendered[1][2],
        visuals=rendered[1][3],
        letter="B",
    )
    footer = (
        "Validation only; physical test labels remain unopened. Examples are label-derived diagnostics "
        "selected by the stated all-seed rules, not unbiased effect estimates. Maps use canonical seed "
        f"{args.canonical_seed}; sample worlds 0–3 use fixed visual seeds {args.sampling_seed} and {args.sampling_seed + 1}."
    )
    lines = textwrap.wrap(footer, width=175)
    draw.multiline_text((72, 1252), "\n".join(lines), font=_font(13), fill=LIGHT_MUTED, spacing=4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    case_metadata: list[dict[str, Any]] = []
    for case, _, _, visuals in rendered:
        value = dict(case)
        value["key"] = list(value["key"])
        value["visual_sampling_seed"] = args.sampling_seed + len(case_metadata)
        value["displayed_sample_indices"] = [0, 1, 2, 3]
        value["displayed_sample_events"] = {
            variant: [bool(event) for event in visuals[variant]["sample_events"]]
            for variant in VARIANTS
        }
        case_metadata.append(value)
    metadata = {
        "schema_version": 1,
        "kind": "flatlands_k128_support_clamped_model_advantage_figure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "paper_result": False,
        "posthoc_checkpoint_evaluation": not clean_support_training,
        "retraining_required": not clean_support_training,
        "clean_support_training": clean_support_training,
        "test_evaluated": False,
        "figure": {
            "path": str(args.output),
            "sha256": _sha256(args.output),
            "width": CANVAS_SIZE[0],
            "height": CANVAS_SIZE[1],
        },
        "software": {
            "renderer_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "renderer_sha256": sha256_path(Path(__file__).resolve()),
            "model_path": "src/pathrel/model.py",
            "model_sha256": sha256_path(PROJECT_ROOT / "src/pathrel/model.py"),
            "torch_version": torch.__version__,
            "render_device": str(device),
        },
        "protocol": {
            "provenance_split": "validation",
            "official_archive_split_used": False,
            "posterior_samples": args.samples,
            "sample_chunk": args.sample_chunk,
            "seeds": list(SEEDS),
            "canonical_map_seed": args.canonical_seed,
            "support_policy": "epistemic_mask complement clamped to blocked",
            "example_selection": "validation-label-derived diagnostic with predeclared all-seed rules",
            "test_lock": "No test labels read; physical test remains locked",
        },
        "benchmark": {
            "selection_path": str(args.selection),
            "selection_sha256": sha256_path(args.selection),
            "queries_path": str(args.queries),
            "queries_sha256": sha256_path(args.queries),
            "prediction_rows_per_seed": 4224,
            "scene_count": 142,
        },
        "aggregate_scene_weighted": aggregate,
        "paired_overall_brier_delta": paired_delta,
        "paired_per_seed_overall": paired_per_seed_overall,
        "cases": case_metadata,
        "canonical_checkpoints": checkpoints,
        "source_artifacts": comparison["provenance"],
        "claim_boundary": (
            "This visual demonstrates a clean support-consistent validation comparison. Every "
            "source run is independently audited and the physical test remains locked; remaining "
            "cross-domain and test gates still prohibit a final paper or leaderboard claim."
            if clean_support_training
            else "This visual demonstrates a validation diagnostic after correcting invalid-support "
            "routing in checkpoints trained under the earlier contract. It is suitable for an "
            "honest project-page progress visual, but clean retraining is required before any "
            "paper, leaderboard, test, or generalization claim."
        ),
    }
    _atomic_json(args.metadata_output, metadata)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(args.metadata_output),
                "sha256": metadata["figure"]["sha256"],
                "cases": [
                    [case["global_id"], case["candidate_index"], case["radius_cells"]]
                    for case in case_metadata
                ],
                "aggregate_brier": {
                    CORRELATED: correlated_brier,
                    INDEPENDENT: independent_brier,
                },
                "test_evaluated": False,
                "retraining_required": not clean_support_training,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
