#!/usr/bin/env python3
"""Run the ConPath stochastic head on a real TUM RGB-D-derived BEV observation.

This is an integration smoke, not a trained result.  First run
``scripts/run_tum_rgbd_pilot.py`` with its default output; that command writes a compact
``model_input.npz`` containing the real RGB-D-derived prefix evidence.  This torch-only script then
executes ``PathRelNet`` and its footprint-conditioned reachability head.  Keeping image processing
in the system interpreter and model processing in the project virtualenv avoids an unnecessary
OpenCV dependency in the research environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from pathrel.model import PathRelNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "tum_rgbd_freiburg1_desk_pilot" / "model_input.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tum_rgbd_model_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"Real BEV input not found: {args.input}\n"
            "Run /usr/bin/python3 scripts/run_tum_rgbd_pilot.py first."
        )
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but this execution session cannot see a CUDA device")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    with np.load(args.input) as data:
        observation = data["observation"].astype(np.float32)
        queries = data["queries"].astype(np.int64)
        support_height = float(data["support_height_m"][0])
        split = int(data["split"][0])
    height, width = observation.shape[-2:]
    # Preserve the measured pattern but bound the smoke tensor and the number of relaxation steps.
    target_h, target_w = min(40, height), min(56, width)
    tensor = torch.from_numpy(observation[None]).to(device=device, dtype=torch.float32)
    if (target_h, target_w) != (height, width):
        tensor = torch.nn.functional.interpolate(tensor, size=(target_h, target_w), mode="area")
    scaled = queries.astype(np.float32)
    scaled[..., 0] = np.clip(scaled[..., 0] * target_h / height, 0, target_h - 1)
    scaled[..., 1] = np.clip(scaled[..., 1] * target_w / width, 0, target_w - 1)
    starts = torch.from_numpy(np.rint(scaled[:, 0]).astype(np.int64)[None]).to(device)
    goals = torch.from_numpy(np.rint(scaled[:, 1]).astype(np.int64)[None]).to(device)
    model = PathRelNet(input_channels=3, feature_channels=16, latent_dim=4, local_kernel_size=3).to(device).eval()
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model(
            tensor,
            starts=starts,
            goals=goals,
            footprint_radii_cells=(0, 1, 2),
            num_samples=max(1, int(args.samples)),
            hard_samples=True,
            max_reachability_steps=max(0, int(args.max_steps)),
        )
    elapsed = time.perf_counter() - start_time
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "real-data model integration smoke",
        "claim_boundary": "Randomly initialised PathRelNet forward on a TUM RGB-D-derived BEV; not a trained or paper-valid result.",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "dataset": "TUM RGB-D Freiburg1/desk",
        "source_input": str(args.input),
        "observed_prefix_frames": split,
        "support_height_m": support_height,
        "input_shape": list(tensor.shape),
        "map_sample_shape": list(output.posterior.safe_samples(0).shape),
        "reachability_shape": list(output.reachability.shape) if output.reachability is not None else None,
        "reachability_values": output.reachability.detach().cpu().tolist() if output.reachability is not None else None,
        "elapsed_seconds": elapsed,
        "max_reachability_steps": int(args.max_steps),
        "queries": int(queries.shape[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
