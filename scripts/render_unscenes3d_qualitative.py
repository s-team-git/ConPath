#!/usr/bin/env python3
"""Render validation-only UnScenes3D posterior, footprint, and failure panels.

The renderer selects one true-positive and one false-safe validation query from a
frozen manifest, then shows the camera projection, three-channel observation,
posterior mean, and the radius-specific safe-center maps.  Selection may inspect
validation labels for the purpose of making a diagnostic failure case; no test
site files are opened and the exported figure is not a paper result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_unscenes3d_coordinate import _parse_calibration, _project  # noqa: E402
from pathrel.labels import clearance_radius_map, maximum_clearance_map  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402
from pathrel.posterior_audits import MEAN_MAP_PROJECTION_VERSION, threshold_supported_mean_map  # noqa: E402
from pathrel.unscenes3d import GRID_SHAPE, load_frame  # noqa: E402


PROTOCOL_VERSION = "UNSCENES3D_PROTOCOL.md v0.2"
VALID_SUPPORT_POLICY = (
    "UnScenes3D target_valid complement is deterministically blocked before posterior sampling"
)
RADII_CELLS = (0, 1, 2)


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


def _resolve_adapter(manifest: dict[str, object]) -> dict[str, object]:
    value = manifest.get("adapter", {})
    if not isinstance(value, dict):
        raise SystemExit("manifest adapter metadata must be an object")
    return {
        "endpoint_policy": str(value.get("endpoint_policy", "blocked")),
        "start_selection": str(value.get("start_selection", "valid")),
        "ground_margin_m": float(value.get("ground_margin_m", 0.35)),
        "ground_bin_size_m": float(value.get("ground_bin_size_m", 1.2)),
        "ground_lateral_limit_m": float(value.get("ground_lateral_limit_m", 20.0)),
        "ground_quantile": float(value.get("ground_quantile", 0.15)),
    }


def _load_frame(
    timestamp: str,
    *,
    raw_root: Path,
    label_root: Path,
    adapter: dict[str, object],
):
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


def _load_model(checkpoint: Path, device: torch.device) -> tuple[PathRelNet, str, dict[str, object]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    variant = str(config.get("decoder_variant", "correlated"))
    if variant not in {"correlated", "independent"}:
        raise SystemExit(f"unsupported decoder variant: {variant!r}")
    feature_channels = int(config.get("feature_channels", 16))
    latent_dim = int(config.get("latent_dim", 4))
    model = PathRelNet(
        input_channels=int(config.get("input_channels", 3)),
        feature_channels=feature_channels,
        latent_dim=latent_dim,
        local_kernel_size=1 if variant == "independent" else 5,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, variant, {
        "feature_channels": feature_channels,
        "latent_dim": latent_dim,
        "local_kernel_size": 1 if variant == "independent" else 5,
        "disable_global_factors": variant == "independent",
        "checkpoint_trained_with_same_support_policy": (
            config.get("valid_support_policy") == VALID_SUPPORT_POLICY
        ),
    }


@torch.inference_mode()
def _posterior_mean(
    model: PathRelNet,
    frame,
    *,
    device: torch.device,
    samples: int,
    sample_chunk: int,
    generator: torch.Generator,
    disable_global_factors: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = torch.from_numpy(frame.input_bev[None]).to(device=device)
    valid_support_mask = torch.from_numpy(frame.target_valid[None]).to(
        device=device, dtype=torch.bool
    )
    total: torch.Tensor | None = None
    remaining = samples
    while remaining:
        current = min(sample_chunk, remaining)
        posterior = model(
            observation,
            valid_support_mask=valid_support_mask,
            num_samples=current,
            hard_samples=True,
            disable_global_factors=disable_global_factors,
            generator=generator,
        ).posterior
        chunk = posterior.posterior_marginal_probs[:, 0]
        total = chunk * current if total is None else total + chunk * current
        remaining -= current
    assert total is not None
    probability = (total / float(samples))[0].cpu().numpy()
    deterministic = threshold_supported_mean_map(
        probability, frame.input_bev, frame.target_valid
    )
    clearance = clearance_radius_map(deterministic)
    return probability, deterministic, clearance


def _geometry_check(record: dict[str, object], frame) -> None:
    queries = list(record["queries"])
    if len(frame.starts) != len(queries) or len(frame.goals) != len(queries):
        raise RuntimeError(f"manifest/runtime query mismatch for {record['timestamp']}")
    for index, query in enumerate(queries):
        if not np.array_equal(frame.starts[index], np.asarray(query["start"], dtype=np.int64)):
            raise RuntimeError(f"manifest/runtime start mismatch for {record['timestamp']}")
        if not np.array_equal(frame.goals[index], np.asarray(query["goal"], dtype=np.int64)):
            raise RuntimeError(f"manifest/runtime goal mismatch for {record['timestamp']}")


def _case_score(
    *,
    event_prediction: np.ndarray,
    target: np.ndarray,
    max_clearance: int,
    query_index: int,
) -> tuple[float, ...]:
    false_safe = bool(event_prediction[2] and not target[2])
    true_positive = bool(np.all(event_prediction == target) and np.all(target))
    if false_safe:
        # Prefer a high-capacity wrong path, then a larger query index only as a
        # deterministic tie-breaker.
        return (1.0, float(max_clearance), float(-query_index))
    if true_positive:
        return (1.0, float(max_clearance), float(-query_index))
    return (0.0, float(-abs(max_clearance)), float(-query_index))


def _select_cases(
    model: PathRelNet,
    records: list[dict[str, object]],
    *,
    raw_root: Path,
    label_root: Path,
    adapter: dict[str, object],
    device: torch.device,
    samples: int,
    sample_chunk: int,
    generator: torch.Generator,
    disable_global_factors: bool,
) -> tuple[dict[str, object], dict[str, object], int]:
    false_safe_cases: list[dict[str, object]] = []
    positive_cases: list[dict[str, object]] = []
    processed = 0
    for record in records:
        frame = _load_frame(
            str(record["timestamp"]),
            raw_root=raw_root,
            label_root=label_root,
            adapter=adapter,
        )
        _geometry_check(record, frame)
        probability, deterministic, clearance = _posterior_mean(
            model,
            frame,
            device=device,
            samples=samples,
            sample_chunk=sample_chunk,
            generator=generator,
            disable_global_factors=disable_global_factors,
        )
        queries = list(record["queries"])
        if not queries:
            continue
        starts = tuple(int(value) for value in queries[0]["start"])
        goals = [tuple(int(value) for value in query["goal"]) for query in queries]
        best = maximum_clearance_map(
            deterministic,
            starts,
            clearance=clearance,
            stop_points=goals,
        )
        for query_index, query in enumerate(queries):
            target = np.asarray(query["reachable"], dtype=bool)
            max_clearance = int(best[goals[query_index]])
            prediction = max_clearance >= np.asarray(RADII_CELLS)
            case = {
                "record": record,
                "frame": frame,
                "probability": probability,
                "deterministic": deterministic,
                "clearance": clearance,
                "start": starts,
                "goal": goals[query_index],
                "query_index": query_index,
                "candidate_index": int(query["candidate_index"]),
                "target": target,
                "prediction": prediction,
                "max_clearance": max_clearance,
            }
            if prediction[2] and not target[2]:
                false_safe_cases.append(case)
            if np.all(prediction == target) and np.all(target):
                positive_cases.append(case)
        processed += 1
    if not false_safe_cases:
        raise RuntimeError("no false-safe validation query was found for the requested checkpoint")
    if not positive_cases:
        raise RuntimeError("no true-positive validation query was found for the requested checkpoint")
    # Stable ordering by capacity, then scene/timestamp/query index.
    false_safe_cases.sort(
        key=lambda case: (
            -int(case["max_clearance"]),
            str(case["record"]["scene_id"]),
            str(case["record"]["timestamp"]),
            int(case["query_index"]),
        )
    )
    positive_cases.sort(
        key=lambda case: (
            -int(case["max_clearance"]),
            str(case["record"]["scene_id"]),
            str(case["record"]["timestamp"]),
            int(case["query_index"]),
        )
    )
    return positive_cases[0], false_safe_cases[0], processed


def _font_bundle():
    from PIL import ImageFont  # type: ignore

    try:
        return (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 25),
            ImageFont.truetype("DejaVuSans-Bold.ttf", 19),
            ImageFont.truetype("DejaVuSans.ttf", 15),
        )
    except OSError:  # pragma: no cover
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback


def _map_panel(case: dict[str, object], mode: str, size: int):
    from PIL import Image, ImageDraw  # type: ignore

    frame = case["frame"]
    probability = np.asarray(case["probability"], dtype=np.float64)
    deterministic = np.asarray(case["deterministic"], dtype=bool)
    clearance = np.asarray(case["clearance"], dtype=np.int64)
    observed_free = frame.input_bev[0] > 0.5
    observed_blocked = frame.input_bev[1] > 0.5
    unknown = frame.input_bev[2] > 0.5
    rgb = np.full((*GRID_SHAPE, 3), (28, 38, 48), dtype=np.uint8)
    if mode == "observed":
        rgb[unknown] = (57, 66, 77)
        rgb[observed_free] = (41, 180, 198)
        rgb[observed_blocked] = (239, 143, 48)
    elif mode == "posterior":
        # Blue -> yellow probability ramp for unknown cells.
        p = np.clip(probability, 0.0, 1.0)
        rgb[unknown, 0] = np.asarray(30 + 220 * p[unknown], dtype=np.uint8)
        rgb[unknown, 1] = np.asarray(70 + 170 * p[unknown], dtype=np.uint8)
        rgb[unknown, 2] = np.asarray(210 - 150 * p[unknown], dtype=np.uint8)
        rgb[observed_free] = (41, 180, 198)
        rgb[observed_blocked] = (239, 143, 48)
    elif mode.startswith("radius"):
        radius = int(mode[-1])
        safe = deterministic & (clearance >= radius)
        rgb[deterministic] = (73, 127, 83)
        rgb[~deterministic] = (159, 62, 67)
        rgb[safe] = (48, 194, 112)
        rgb[observed_free] = (41, 180, 198)
        rgb[observed_blocked] = (239, 143, 48)
    else:
        raise ValueError(mode)

    image = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    start = tuple(int(value) for value in case["start"])
    goal = tuple(int(value) for value in case["goal"])
    scale = size / float(GRID_SHAPE[0])
    start_xy = (int(round((start[1] + 0.5) * scale)), int(round((start[0] + 0.5) * scale)))
    goal_xy = (int(round((goal[1] + 0.5) * scale)), int(round((goal[0] + 0.5) * scale)))
    draw.line((start_xy, goal_xy), fill=(255, 240, 90), width=max(2, size // 150))
    draw.ellipse((start_xy[0] - 6, start_xy[1] - 6, start_xy[0] + 6, start_xy[1] + 6), fill=(255, 255, 255))
    draw.ellipse((goal_xy[0] - 6, goal_xy[1] - 6, goal_xy[0] + 6, goal_xy[1] + 6), fill=(255, 240, 90))
    return image


def _camera_panel(case: dict[str, object], raw_root: Path, size: tuple[int, int]):
    from PIL import Image, ImageDraw  # type: ignore

    record = case["record"]
    timestamp = str(record["timestamp"])
    image_path = raw_root / "images" / f"{timestamp}.jpg"
    calibration_path = raw_root / "calibs" / f"{timestamp}.txt"
    cloud_path = raw_root / "clouds" / f"{timestamp}.bin"
    if not image_path.exists() or not calibration_path.exists() or not cloud_path.exists():
        raise FileNotFoundError(f"missing camera panel inputs for {timestamp}")
    image = Image.open(image_path).convert("RGB")
    values = np.fromfile(cloud_path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"malformed cloud: {cloud_path}")
    points = values.reshape(-1, 4)
    calibration = _parse_calibration(calibration_path)
    camera, uv = _project(points[:, :3], calibration)
    width, height = image.size
    visible = (
        np.isfinite(uv).all(axis=1)
        & (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (12, 18, 24))
    x0 = (size[0] - image.width) // 2
    y0 = (size[1] - image.height) // 2
    canvas.paste(image, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    sx, sy = image.width / width, image.height / height
    indices = np.flatnonzero(visible)
    if len(indices) > 2500:
        indices = indices[np.linspace(0, len(indices) - 1, 2500, dtype=np.int64)]
    for index in indices.tolist():
        u, v = uv[index]
        x, y = x0 + int(round(u * sx)), y0 + int(round(v * sy))
        depth = float(camera[index, 2])
        color = (54, 210, 157) if depth < 12 else (240, 169, 63) if depth < 30 else (91, 132, 212)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return canvas, int(np.sum(visible)), int(len(points))


def _render_case(
    case: dict[str, object],
    *,
    output: Path,
    raw_root: Path,
    label: str,
) -> dict[str, object]:
    from PIL import Image, ImageDraw  # type: ignore

    title_font, panel_font, small_font = _font_bundle()
    width, height = 1800, 1280
    panel_size = 540
    canvas = Image.new("RGB", (width, height), (245, 247, 249))
    draw = ImageDraw.Draw(canvas)
    record = case["record"]
    target = np.asarray(case["target"], dtype=bool)
    prediction = np.asarray(case["prediction"], dtype=bool)
    status = ", ".join(
        f"r={radius}: {'✓' if prediction[i] else '×'} (truth {'✓' if target[i] else '×'})"
        for i, radius in enumerate(RADII_CELLS)
    )
    draw.text(
        (30, 18),
        f"UnScenes3D {label} · {record['scene_id']} · {record['timestamp']}",
        fill=(25, 38, 50),
        font=title_font,
    )
    draw.text((30, 49), f"candidate={case['candidate_index']} · max-clearance={case['max_clearance']} cells · {status}", fill=(64, 76, 88), font=small_font)
    camera, projected, point_count = _camera_panel(case, raw_root, (panel_size, 390))
    positions = ((30, 85), (630, 85), (1230, 85), (30, 685), (630, 685), (1230, 685))
    panels = (
        camera,
        _map_panel(case, "observed", panel_size),
        _map_panel(case, "posterior", panel_size),
        _map_panel(case, "radius0", panel_size),
        _map_panel(case, "radius1", panel_size),
        _map_panel(case, "radius2", panel_size),
    )
    captions = (
        f"Camera + LiDAR projection ({projected}/{point_count})",
        "Observed BEV · cyan free / orange endpoint",
        "Posterior mean free probability · line=start→goal",
        f"Footprint radius 0 · predicted {'reachable' if prediction[0] else 'blocked'}",
        f"Footprint radius 1 · predicted {'reachable' if prediction[1] else 'blocked'}",
        f"Footprint radius 2 · predicted {'reachable' if prediction[2] else 'blocked'}",
    )
    for (x, y), panel, caption in zip(positions, panels, captions):
        canvas.paste(panel, (x, y))
        draw.text((x, y - 24), caption, fill=(25, 38, 50), font=panel_font)
    draw.text(
        (30, 1242),
        "green safe centers · red blocked centers · cyan observed free · orange observed endpoint · yellow query · truth is used only to select this validation diagnostic",
        fill=(70, 82, 92),
        font=small_font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    temporary.replace(output)
    return {
        "path": str(output),
        "label": label,
        "scene_id": str(record["scene_id"]),
        "timestamp": str(record["timestamp"]),
        "candidate_index": int(case["candidate_index"]),
        "target": target.tolist(),
        "prediction": prediction.tolist(),
        "max_clearance_cells": int(case["max_clearance"]),
        "projected_points": projected,
        "point_count": point_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/unscenes3d/raw_package/unscenes3d-mini_raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/unscenes3d/label_package/unscenes3d-mini_label"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--sample-chunk", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.posterior_samples < 2 or args.sample_chunk < 2:
        raise SystemExit("posterior-samples and sample-chunk must both be at least two")
    if args.posterior_samples % args.sample_chunk:
        raise SystemExit("posterior-samples must be divisible by sample-chunk")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    adapter = _resolve_adapter(manifest)
    records = list(manifest["records"]["validation"])
    if not records:
        raise SystemExit("validation records are empty")
    device = torch.device(args.device)
    generator = _seed(args.seed, device)
    model, variant, decoder = _load_model(args.checkpoint, device)
    positive, failure, processed = _select_cases(
        model,
        records,
        raw_root=args.raw_root,
        label_root=args.label_root,
        adapter=adapter,
        device=device,
        samples=args.posterior_samples,
        sample_chunk=args.sample_chunk,
        generator=generator,
        disable_global_factors=bool(decoder["disable_global_factors"]),
    )
    positive_report = _render_case(
        positive,
        output=args.output_dir / "qualitative_positive.png",
        raw_root=args.raw_root,
        label="true-positive",
    )
    failure_report = _render_case(
        failure,
        output=args.output_dir / "qualitative_failure.png",
        raw_root=args.raw_root,
        label="false-safe failure",
    )
    report = {
        "schema_version": 2,
        "kind": "unscenes3d_qualitative_validation_panel",
        "paper_result": False,
        "validation_result": True,
        "protocol_version": PROTOCOL_VERSION,
        "test_evaluated": False,
        "posthoc_checkpoint_evaluation": not bool(
            decoder["checkpoint_trained_with_same_support_policy"]
        ),
        "retraining_required": not bool(
            decoder["checkpoint_trained_with_same_support_policy"]
        ),
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "manifest": {
            "path": str(args.manifest),
            "sha256": _sha256(args.manifest),
            "validation_records": len(records),
            "validation_queries": sum(len(record["queries"]) for record in records),
            "test_locked_sites": manifest.get("test_locked_sites", []),
        },
        "adapter": adapter,
        "decoder": {"variant": variant, **decoder},
        "forward": {
            "invalid_support_clamped": True,
            "mean_map_projection_version": MEAN_MAP_PROJECTION_VERSION,
            "implementation_sha256": {
                "renderer": _sha256(Path(__file__)),
                "mean_map_projection": _sha256(PROJECT_ROOT / "src/pathrel/posterior_audits.py"),
            },
            "valid_support_policy": VALID_SUPPORT_POLICY,
        },
        "selection": {
            "posterior_samples": args.posterior_samples,
            "sample_chunk": args.sample_chunk,
            "seed": args.seed,
            "frames_scanned": processed,
            "rule": "highest deterministic max-clearance false-safe radius-2 query and highest max-clearance all-radii true-positive query",
        },
        "panels": {"positive": positive_report, "failure": failure_report},
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        },
        "claim_boundary": (
            "Validation-only post-hoc support-clamped qualitative diagnostic from a checkpoint "
            "trained before the support policy; clean retraining is required, target labels select "
            "the displayed cases, and location_6 remains locked."
            if not decoder["checkpoint_trained_with_same_support_policy"]
            else "Validation-only support-consistent qualitative diagnostic; target labels select "
            "the displayed cases, location_6 remains locked, and no final paper claim is made."
        ),
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({"output_dir": str(args.output_dir), "panels": report["panels"], "test_evaluated": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
