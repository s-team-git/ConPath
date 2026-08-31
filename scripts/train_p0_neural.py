#!/usr/bin/env python3
"""Train/evaluate the neural ConPath head on the held-out-template P0 protocol.

This trainer uses the same context-family and visible-query generator as ``evaluate_p0.py``. It
also writes a durable JSONL journal and atomic latest checkpoint so an interrupted shell or Codex
session can resume without reconstructing optimizer or random-generator state from chat history.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

from evaluate_p0 import make_split, stack  # noqa: E402
from pathrel.losses import (  # noqa: E402
    map_cross_entropy,
    posterior_marginal_nll_from_probs,
    reachability_brier_u_statistic,
    spatial_variogram_score,
)
from pathrel.metrics import summarize  # noqa: E402
from pathrel.model import PathRelNet  # noqa: E402


P0_COMPARATORS = {
    "independent_brier": 0.18317159016927084,
    "direct_query_brier": 0.16988758349570215,
    "independent_ece": 0.1762424045138889,
    "direct_query_ece": 0.08912776173751391,
    "independent_map_brier": 0.002829818813889115,
}


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
        help="Positive Gumbel scale; conditional marginals use softmax(logits / scale).",
    )
    parser.add_argument("--reachability-weight", type=float, default=2.0)
    parser.add_argument("--variogram-weight", type=float, default=0.2)
    parser.add_argument(
        "--empirical-map-weight",
        type=float,
        default=1.0,
        help="Weight for the U-statistic Brier score of empirical hard-map marginals.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the resume checkpoint directory or results/p0_neural.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def append_progress(path: Path, payload: dict[str, Any], *, durable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": utc_now(), **payload}, allow_nan=True) + "\n")
        stream.flush()
        if durable:
            os.fsync(stream.fileno())


def training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "radii_cells": [0, 1, 2],
        "map_shape": [args.height, args.width],
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "train_templates": args.train_templates,
        "test_templates": args.test_templates,
        "worlds_per_template": args.worlds_per_template,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "categorical_noise_scale": args.categorical_noise_scale,
        "reachability_weight": args.reachability_weight,
        "variogram_weight": args.variogram_weight,
        "empirical_map_weight": args.empirical_map_weight,
        "learning_rate": args.learning_rate,
    }


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    mutable = {"steps", "validation_samples"}
    mismatches = {
        key: {"checkpoint": saved.get(key), "requested": value}
        for key, value in current.items()
        if key not in mutable and saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume configuration mismatch: {mismatches}")


def training_checkpoint(
    *,
    model: PathRelNet,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    next_step: int,
    losses: list[dict[str, Any]],
    config: dict[str, Any],
    status: str,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "status": status,
        "saved_at_utc": utc_now(),
        "git_head": git_head(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "losses": losses,
        "config": config,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if report is not None:
        payload["report"] = report
    return payload


def compact_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    return {
        key: value
        for key, value in summarize(probabilities, targets).items()
        if key != "reliability"
    }


def event_metrics_by_radius(probabilities: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    return [
        compact_metrics(probabilities[..., index], targets[..., index])
        for index in range(probabilities.shape[-1])
    ]


def doorway_summary(wall_free: np.ndarray) -> dict[str, float]:
    wall_free = np.asarray(wall_free, dtype=bool)
    transitions = np.count_nonzero(wall_free[..., 1:] != wall_free[..., :-1], axis=-1)
    return {
        "any_open_rate": float(wall_free.any(axis=-1).mean()),
        "mean_free_cells": float(wall_free.sum(axis=-1).mean()),
        "mean_transitions": float(transitions.mean()),
        "fragmented_rate": float((transitions > 2).mean()),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("steps must be positive and warmup steps must lie in [0, steps]")
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("batch size must be an even integer of at least two")
    if args.train_samples < 2 or args.validation_samples < 2:
        raise ValueError("train and validation sample counts must be at least two")
    if args.categorical_noise_scale <= 0:
        raise ValueError("categorical noise scale must be positive during training")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if min(
        args.reachability_weight,
        args.variogram_weight,
        args.empirical_map_weight,
        args.learning_rate,
    ) < 0:
        raise ValueError("loss weights and learning rate must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir or (
        args.resume.parent
        if args.resume is not None
        else PROJECT_ROOT / "results" / "p0_neural"
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; "
            "run scripts/check_environment.py first."
        )

    output_dir: Path = args.output_dir
    progress_path = output_dir / "progress.jsonl"
    latest_path = output_dir / "latest.pt"
    if args.resume is None and any(
        path.exists() for path in (progress_path, latest_path, output_dir / "checkpoint.pt")
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing run in {output_dir}; choose a new --output-dir "
            "or pass --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    radii = (0, 1, 2)
    config = training_config(args)
    train_arrays = stack(
        make_split(
            range(args.train_templates),
            worlds_per_template=args.worlds_per_template,
            height=args.height,
            width=args.width,
            radii=radii,
        )
    )
    test_arrays = stack(
        make_split(
            range(args.train_templates, args.train_templates + args.test_templates),
            worlds_per_template=args.worlds_per_template,
            height=args.height,
            width=args.width,
            radii=radii,
        )
    )
    model = PathRelNet(input_channels=3, feature_channels=32, latent_dim=8).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    losses: list[dict[str, Any]] = []
    start_step = 0

    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format_version") != 2 or "optimizer" not in resume:
            raise ValueError("resume requires an interruption-safe format_version=2 checkpoint")
        validate_resume_config(resume["config"], config)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        # ``map_location=cuda`` also moves ByteTensor RNG states, but Generator.set_state
        # requires its serialized state on CPU even when the generator itself targets CUDA.
        generator.set_state(resume["generator_state"].cpu())
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in resume:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume["cuda_rng_state_all"]]
            )
        start_step = int(resume["next_step"])
        losses = list(resume.get("losses", []))
        if start_step > args.steps:
            raise ValueError(
                f"checkpoint is at step {start_step}, beyond requested --steps {args.steps}"
            )

    append_progress(
        progress_path,
        {
            "event": "resume" if args.resume is not None else "start",
            "git_head": git_head(),
            "device": str(device),
            "start_step": start_step,
            "config": config,
            "command": sys.argv,
        },
        durable=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    context_zero = np.flatnonzero(train_arrays["context"] == 0)
    context_one = np.flatnonzero(train_arrays["context"] == 1)

    model.train()
    try:
        for step in range(start_step, args.steps):
            batch_rng = np.random.default_rng(args.seed + step)
            half_batch = args.batch_size // 2
            indices = np.concatenate(
                [
                    batch_rng.choice(context_zero, half_batch, replace=False),
                    batch_rng.choice(context_one, half_batch, replace=False),
                ]
            )
            observation = torch.from_numpy(train_arrays["observation"][indices]).to(
                device=device, dtype=torch.float32
            )
            target_classes = torch.from_numpy(
                (~train_arrays["target_free"][indices]).astype(np.int64)
            ).to(device=device, dtype=torch.long)
            unknown = observation[:, 2] > 0.5
            hidden_target_classes = target_classes.masked_fill(~unknown, -1)
            target_safe = (target_classes == 0).to(torch.float32)
            starts = torch.from_numpy(train_arrays["starts"][indices]).to(
                device=device, dtype=torch.long
            )
            goals = torch.from_numpy(train_arrays["goals"][indices]).to(
                device=device, dtype=torch.long
            )
            targets = torch.from_numpy(train_arrays["events"][indices]).to(
                device=device, dtype=torch.float32
            )

            if step < args.warmup_steps:
                output = model(
                    observation,
                    num_samples=args.train_samples,
                    hard_samples=True,
                    categorical_noise_scale=args.categorical_noise_scale,
                    generator=generator,
                )
                map_loss = map_cross_entropy(
                    output.posterior.mean_logits, hidden_target_classes
                )
                empirical_map_loss = output.posterior.mean_logits.sum() * 0.0
                variogram_loss = output.posterior.mean_logits.sum() * 0.0
                reach_loss = output.posterior.mean_logits.sum() * 0.0
                total = map_loss
            else:
                output = model(
                    observation,
                    starts=starts,
                    goals=goals,
                    footprint_radii_cells=radii,
                    num_samples=args.train_samples,
                    hard_samples=True,
                    categorical_noise_scale=args.categorical_noise_scale,
                    max_reachability_steps=args.height * args.width,
                    generator=generator,
                )
                map_loss = posterior_marginal_nll_from_probs(
                    output.posterior.conditional_class_probs,
                    hidden_target_classes,
                )
                empirical_map_loss = reachability_brier_u_statistic(
                    output.posterior.safe_samples(),
                    target_safe,
                    weights=unknown.to(torch.float32),
                )
                variogram_loss = spatial_variogram_score(
                    output.posterior.safe_samples(),
                    target_safe,
                    valid_mask=unknown,
                )
                reach_loss = reachability_brier_u_statistic(
                    output.sample_reachability, targets
                )
                total = (
                    map_loss
                    + args.empirical_map_weight * empirical_map_loss
                    + args.variogram_weight * variogram_loss
                    + args.reachability_weight * reach_loss
                )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            row = {
                "event": "step",
                "step": step + 1,
                "stage": "warmup" if step < args.warmup_steps else "joint",
                "total": float(total.detach()),
                "map_nll": float(map_loss.detach()),
                "empirical_map_u": float(empirical_map_loss.detach()),
                "variogram": float(variogram_loss.detach()),
                "reach_u": float(reach_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            losses.append(row)
            append_progress(progress_path, row)
            if step == start_step or (step + 1) % max(1, args.steps // 6) == 0:
                print(json.dumps(row), flush=True)
            if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
                atomic_torch_save(
                    training_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        generator=generator,
                        next_step=step + 1,
                        losses=losses,
                        config=config,
                        status="running",
                    ),
                    latest_path,
                )
                append_progress(
                    progress_path,
                    {"event": "checkpoint", "next_step": step + 1, "path": str(latest_path)},
                    durable=True,
                )
    except (Exception, KeyboardInterrupt) as error:
        interrupted_path = output_dir / "interrupted.pt"
        completed_steps = int(losses[-1]["step"]) if losses else start_step
        atomic_torch_save(
            training_checkpoint(
                model=model,
                optimizer=optimizer,
                generator=generator,
                next_step=completed_steps,
                losses=losses,
                config=config,
                status="interrupted",
            ),
            interrupted_path,
        )
        append_progress(
            progress_path,
            {
                "event": "interrupted",
                "next_step": completed_steps,
                "checkpoint": str(interrupted_path),
                "error_type": type(error).__name__,
                "error": str(error),
            },
            durable=True,
        )
        raise

    model.eval()
    test_observation = torch.from_numpy(test_arrays["observation"]).to(
        device=device, dtype=torch.float32
    )
    test_starts = torch.from_numpy(test_arrays["starts"]).to(
        device=device, dtype=torch.long
    )
    test_goals = torch.from_numpy(test_arrays["goals"]).to(
        device=device, dtype=torch.long
    )
    with torch.no_grad():
        output = model(
            test_observation,
            starts=test_starts,
            goals=test_goals,
            footprint_radii_cells=radii,
            num_samples=args.validation_samples,
            hard_samples=True,
            categorical_noise_scale=args.categorical_noise_scale,
            max_reachability_steps=args.height * args.width,
            generator=torch.Generator(device=device).manual_seed(args.seed + 999999),
        )

    event_probabilities = output.reachability.detach().cpu().numpy()
    conditional_marginal = (
        output.posterior.posterior_marginal_probs[:, 0].detach().cpu().numpy()
    )
    empirical_marginal = (
        output.posterior.empirical_class_frequencies[:, 0].detach().cpu().numpy()
    )
    target_free = test_arrays["target_free"].astype(np.float64)
    unknown_np = test_arrays["observation"][:, 2] > 0.5
    hard_maps = output.posterior.safe_samples().detach().cpu().numpy().astype(bool)
    wall_free = hard_maps[..., args.width // 2]

    event_report = summarize(event_probabilities, test_arrays["events"])
    conditional_full = compact_metrics(conditional_marginal, target_free)
    empirical_full = compact_metrics(empirical_marginal, target_free)
    map_reports = {
        "conditional_categorical_full": conditional_full,
        "empirical_hard_full": empirical_full,
        "conditional_categorical_unknown": compact_metrics(
            conditional_marginal[unknown_np], target_free[unknown_np]
        ),
        "empirical_hard_unknown": compact_metrics(
            empirical_marginal[unknown_np], target_free[unknown_np]
        ),
    }
    official_protocol = bool(
        args.height == 24
        and args.width == 24
        and args.train_templates == 12
        and args.test_templates == 4
        and args.worlds_per_template == 24
        and args.validation_samples >= 64
    )
    p0_pass = bool(
        official_protocol
        and event_report["brier"] < P0_COMPARATORS["independent_brier"]
        and event_report["brier"] < P0_COMPARATORS["direct_query_brier"]
        and event_report["ece"]
        <= min(P0_COMPARATORS["independent_ece"], P0_COMPARATORS["direct_query_ece"])
        + 0.02
        and empirical_full["brier"]
        <= P0_COMPARATORS["independent_map_brier"] + 0.02
    )
    doorway: dict[str, Any] = {}
    for family in (0, 1):
        selected = test_arrays["context"] == family
        doorway[str(family)] = {
            "target": doorway_summary(test_arrays["target_free"][selected, :, args.width // 2]),
            "sampled": doorway_summary(wall_free[selected]),
        }

    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "protocol": {
            **config,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_head": git_head(),
            "command": sys.argv,
        },
        "event_metrics": event_report,
        "event_metrics_by_radius": event_metrics_by_radius(
            event_probabilities, test_arrays["events"]
        ),
        "event_mean_by_radius": event_probabilities.mean(axis=(0, 1)).tolist(),
        "target_event_mean_by_radius": test_arrays["events"].mean(axis=(0, 1)).tolist(),
        "event_mean_by_context_and_radius": {
            str(family): event_probabilities[test_arrays["context"] == family]
            .mean(axis=(0, 1))
            .tolist()
            for family in (0, 1)
        },
        "target_event_mean_by_context_and_radius": {
            str(family): test_arrays["events"][test_arrays["context"] == family]
            .mean(axis=(0, 1))
            .tolist()
            for family in (0, 1)
        },
        "map_marginal_metrics": map_reports,
        "doorway_joint": doorway,
        "last_training": losses[-1],
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
        "p0_gate": {
            "passed": p0_pass,
            "official_protocol": official_protocol,
            "comparators": P0_COMPARATORS,
            "criterion": (
                "beat independent and direct-query event Brier; ECE no more than 0.02 above "
                "the better comparator; empirical hard-map Brier within 0.02 of independent"
            ),
        },
        "paper_result": False,
        "interpretation": (
            "Synthetic neural P0 gate passed; public-data audit is still required."
            if p0_pass
            else "Neural P0 remains NO-GO; do not expand to public-data claims."
        ),
    }
    atomic_json(report, output_dir / "report.json")
    completed_payload = training_checkpoint(
        model=model,
        optimizer=optimizer,
        generator=generator,
        next_step=args.steps,
        losses=losses,
        config=config,
        status="complete",
        report=report,
    )
    atomic_torch_save(completed_payload, output_dir / "checkpoint.pt")
    atomic_torch_save(completed_payload, latest_path)
    append_progress(
        progress_path,
        {
            "event": "complete",
            "next_step": args.steps,
            "event_brier": event_report["brier"],
            "event_ece": event_report["ece"],
            "empirical_map_brier": empirical_full["brier"],
            "p0_gate_passed": p0_pass,
            "report": str(output_dir / "report.json"),
        },
        durable=True,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "event_metrics": {
                    key: value
                    for key, value in event_report.items()
                    if isinstance(value, (float, int))
                },
                "empirical_hard_map_metrics": empirical_full,
                "p0_gate_passed": p0_pass,
                "paper_result": False,
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
