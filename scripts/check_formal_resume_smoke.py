#!/usr/bin/env python3
"""Small cross-process resume proof; never a convergence or evaluation result.

Default method is a tiny CPU fixture. ``--method lama --device cuda`` and
``--method flow --device cuda`` exercise the unchanged full-size external models
on tiny synthetic BEV tensors. They do not read any dataset or test assets.
Only root should launch GPU tests after checking the project resource budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from pathrel.formal_checkpoint import (FormalCheckpointStore, StatefulBatchSampler,
    atomic_json, capture_training_state, restore_training_state)


def configure(seed, device):
    # This also precedes CUDA initialization in every fresh subprocess.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def synthetic_data(size):
    # Generator is disjoint from every training stream; mask and targets do not
    # depend on any trained model. The three channels remain [free, U, support].
    rng = torch.Generator().manual_seed(970)
    target = (torch.rand((7, 1, size, size), generator=rng) > .35).float()
    condition = torch.zeros((7, 3, size, size))
    condition[:, 2, 1:-1, 1:-1] = 1
    condition[:, 1, 4:-4, 4:-4] = 1
    target *= condition[:, 2:3]
    condition[:, :1] = target * (condition[:, 2:3] - condition[:, 1:2])
    return condition, target


def make_objects(method, device):
    if method == "lama":
        from pathrel.lama import LaMaBEV, make_discriminator
        models = {"model": LaMaBEV().to(device), "discriminator": make_discriminator().to(device)}
    elif method == "flow":
        from pathrel.flow_matching import FlowUNet
        models = {"model": FlowUNet().to(device)}
    else:
        models = {"model": torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.2),
                                               torch.nn.Linear(8, 1)).to(device),
                  "discriminator": torch.nn.Linear(1, 1).to(device)}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
                  for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
                  for name, optimizer in optimizers.items()}
    return {"models": models, "optimizers": optimizers, "schedulers": schedulers,
            "sampler": StatefulBatchSampler(7, 2, seed=20260831),
            "generators": {"noise": torch.Generator(device=device).manual_seed(209),
                           "sample": torch.Generator(device="cpu").manual_seed(311)}}


def run_update(args, objects, conditions, targets, ids):
    models, optimizers = objects["models"], objects["optimizers"]
    if args.method == "lama":
        from pathrel.external_training import lama_accumulated_update
        return lama_accumulated_update(models["model"], models["discriminator"], optimizers["model"],
            optimizers["discriminator"], conditions[ids].to(args.device), targets[ids].to(args.device), microbatch=2)
    if args.method == "flow":
        from pathrel.external_training import flow_accumulated_update
        return flow_accumulated_update(models["model"], optimizers["model"],
            conditions[ids].to(args.device), targets[ids].to(args.device), microbatch=2,
            rng=objects["generators"]["noise"])
    noise = torch.randn((2, 3), device=args.device, generator=objects["generators"]["noise"])
    x = noise + torch.tensor(ids, device=args.device)[:, None] / 7
    y = torch.rand((2, 1), generator=objects["generators"]["sample"]).to(args.device)
    y += random.random() + float(np.random.random())
    model, discriminator = models["model"], models["discriminator"]
    optimizers["model"].zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    prediction = model(x)
    loss_g = (discriminator(prediction) - y).square().mean()
    loss_g.backward()
    optimizers["model"].step()
    discriminator.requires_grad_(True)
    optimizers["discriminator"].zero_grad(set_to_none=True)
    loss_d = (discriminator(prediction.detach()) - y).square().mean()
    loss_d.backward()
    optimizers["discriminator"].step()
    return {"generator": float(loss_g.detach()), "discriminator": float(loss_d.detach())}


def worker(args):
    configure(20260831, args.device)
    started = time.perf_counter()
    if args.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024 ** 3 / total, device=0)
    objects = make_objects(args.method, args.device)
    conditions, targets = synthetic_data(args.size)
    recipe = {"method": args.method, "device": args.device, "seed": 20260831,
              "steps": args.steps, "batch": 2, "microbatch": 2, "size": args.size,
              "purpose": "numerical recovery proof on synthetic BEV; no dataset assets",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": .01},
              "scheduler": {"name": "StepLR", "step_size": 2, "gamma": .9},
              "checkpoint_interval": args.checkpoint_interval,
              "gpu_allocator_limit_gib": args.memory_gib if args.device == "cuda" else None}
    def resources():
        return {"elapsed_seconds_this_process": time.perf_counter() - started,
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if args.device == "cuda" else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if args.device == "cuda" else 0,
                "gpu_allocator_limit_gib": args.memory_gib if args.device == "cuda" else None,
                "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda}
    with FormalCheckpointStore(args.output_dir, recipe, resume=args.resume,
                               checkpoint_interval=args.checkpoint_interval) as store:
        if args.resume:
            restored = restore_training_state(store.load_latest()["state"], **objects)
            start = restored["completed_steps"]
            elapsed_before = restored["extra"]["elapsed_seconds"]
        else:
            start, elapsed_before = 0, 0.
            store.initialize(capture_training_state(completed_steps=0, **objects,
                             extra={"elapsed_seconds": 0.}))
        try:
            for step in range(start + 1, args.steps + 1):
                ids = objects["sampler"].next_batch()
                optimizer = objects["optimizers"]["model"]
                original_step = optimizer.step
                if step == args.interrupt_at:
                    # A real SIGINT is delivered AFTER optimizer G changes its
                    # parameters but BEFORE D or the scheduler completes.
                    def interrupting_step(*positional, **keywords):
                        result = original_step(*positional, **keywords)
                        os.kill(os.getpid(), signal.SIGINT)
                        return result
                    optimizer.step = interrupting_step
                losses = run_update(args, objects, conditions, targets, ids)
                optimizer.step = original_step
                if not all(np.isfinite(value) for value in losses.values()):
                    raise ValueError("Nonfinite synthetic smoke loss")
                for scheduler in objects["schedulers"].values():
                    scheduler.step()
                store.append_progress({"step": step, "ids": ids, "losses": losses})
                if step % args.checkpoint_interval == 0 or step == args.steps:
                    store.commit_completed_update(capture_training_state(completed_steps=step, **objects,
                        extra={"elapsed_seconds": elapsed_before + time.perf_counter() - started}))
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            store.write_interrupted("test SIGINT after first optimizer changed parameters")
            atomic_json(args.output_dir / "interruption_resources.json", resources())
            return 86
        atomic_json(args.output_dir / "completion.json", {"completed_steps": args.steps,
            **resources(),
            "test_assets_opened": False, "synthetic_only": True})
    return 0


def compare_nested(left, right, prefix="state"):
    errors = []
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            errors.append(prefix)
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            errors.append(prefix)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return [prefix]
        for key in left:
            if prefix == "state.extra" and key == "elapsed_seconds":
                continue
            errors.extend(compare_nested(left[key], right[key], f"{prefix}.{key}"))
    elif isinstance(left, (tuple, list)):
        if not isinstance(right, (tuple, list)) or len(left) != len(right):
            return [prefix]
        for index, (a, b) in enumerate(zip(left, right)):
            errors.extend(compare_nested(a, b, f"{prefix}.{index}"))
    elif left != right:
        errors.append(prefix)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["toy", "lama", "flow"], default="toy")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=28.)
    parser.add_argument("--interrupt-at", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.memory_gib <= 28:
        parser.error("GPU allocator limit must be between 0 and 28 GiB")
    if args.size < 32 or args.size % 8 or not 1 < args.interrupt_at <= args.steps:
        # Worker continuous/resume phases use interrupt_at=0 explicitly.
        if not (args.worker and args.interrupt_at == 0 and args.size >= 32 and not args.size % 8):
            parser.error("Need size >= 32 divisible by 8 and 1 < interrupt-at <= steps")
    if args.worker:
        return worker(args)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke artifact directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    base = [sys.executable, str(Path(__file__).resolve()), "--worker", "--method", args.method,
            "--device", args.device, "--steps", str(args.steps), "--size", str(args.size),
            "--checkpoint-interval", str(args.checkpoint_interval), "--memory-gib", str(args.memory_gib)]
    phases = [("continuous", "continuous", 0, False, 0),
              ("interrupted", "split", args.interrupt_at, False, 86),
              ("resumed", "split", 0, True, 0)]
    runs = []
    for name, directory, interruption, resume, expected_return in phases:
        command = base + ["--output-dir", str(args.output_dir / directory), "--interrupt-at", str(interruption)]
        if resume:
            command.append("--resume")
        started = time.perf_counter()
        with (args.output_dir / f"{name}.log").open("wb") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        runs.append({"phase": name, "returncode": result.returncode, "expected_returncode": expected_return,
                     "elapsed_seconds": time.perf_counter() - started, "command": command})
        atomic_json(args.output_dir / "subprocesses.json", runs)
        if result.returncode != expected_return:
            raise RuntimeError(f"{name} returned {result.returncode}; see its log")
    continuous = torch.load(args.output_dir / "continuous/latest.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(args.output_dir / "split/latest.pt", map_location="cpu", weights_only=False)
    state_differences = compare_nested(continuous["state"], resumed["state"])
    left = (args.output_dir / "continuous/progress.jsonl").read_bytes()
    right = (args.output_dir / "split/progress.jsonl").read_bytes()
    root = Path(__file__).resolve().parents[1]
    sources = ["scripts/check_formal_resume_smoke.py", "src/pathrel/formal_checkpoint.py",
               "src/pathrel/external_training.py", "src/pathrel/lama.py", "src/pathrel/flow_matching.py"]
    report = {"passed": not state_differences and left == right, "method": args.method, "device": args.device,
              "complete_loss_curve_bitwise_equal": left == right, "state_differences": state_differences,
              "compared_states": ["model", "all optimizers", "all schedulers", "sampler permutation/cursor/epoch",
                                  "Python/NumPy/Torch CPU/CUDA RNG", "named generators", "progress index"],
              "exclusions": ["elapsed wall-clock seconds"], "steps": args.steps,
              "interrupt_at": args.interrupt_at, "checkpoint_interval": args.checkpoint_interval,
              "interruption": json.loads((args.output_dir / "split/interruption.json").read_text()),
              "archived_uncommitted_journals": [str(path.relative_to(args.output_dir)) for path in
                  (args.output_dir / "split/recovery_journal").glob("uncommitted_tail_*.jsonl")],
              "loss_curve_sha256": hashlib.sha256(left).hexdigest(), "runs": runs,
              "resources": {"continuous": json.loads((args.output_dir / "continuous/completion.json").read_text()),
                            "interrupted": json.loads((args.output_dir / "split/interruption_resources.json").read_text()),
                            "resumed": json.loads((args.output_dir / "split/completion.json").read_text())},
              "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "synthetic_only": True, "test_assets_opened": False, "formal_training": False,
              "model_recipe": "unchanged full-size model; synthetic batch 2, microbatch 2" if args.method != "toy" else "tiny CPU fixture",
              "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}}
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
