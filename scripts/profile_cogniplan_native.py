#!/usr/bin/env python3
"""Public-weight train-only sanity and fixed-32 native training resource profile.

No held-out evaluation, checkpoint selection, or convergence claim is made here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
import torch

from pathrel.cogniplan import (NATIVE_CONDITIONS, configure_reproducible_cuda, encode_partial, encode_target,
                              load_native, native_update, postprocess)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_state():
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def load_config(raw):
    # Released config is a small scalar/list mapping; no environment-wide installs.
    config, nested = {}, None
    import ast
    for line in raw.decode().splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        key, value = text.split(":", 1)
        value = value.strip()
        if not line.startswith(" "):
            nested = None
        if not value and key in {"netG", "netD"}:
            config[key] = {}
            nested = config[key]
            continue
        try:
            value = ast.literal_eval(value) if value else None
        except (ValueError, SyntaxError):
            pass
        (nested if nested is not None else config)[key] = value
    return config


def event_workload(worlds, endpoints):
    """192 target-blind endpoint/radius events per world; padded exact disk / 4-neighbor."""
    predictions = []
    for world in worlds:
        distance = distance_transform_edt(np.pad(world, 1))[1:-1, 1:-1]
        for radius in (0, 3, 6):
            components, _ = label(distance > radius)
            a, b = endpoints[:, :2], endpoints[:, 2:]
            ca, cb = components[a[:, 0], a[:, 1]], components[b[:, 0], b[:, 1]]
            predictions.append((ca != 0) & (ca == cb))
    return np.asarray(predictions).reshape(4, 3, 64).mean(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-dir", type=Path, default=Path("results/external_protocol_v1"))
    p.add_argument("--output-dir", type=Path, default=Path("results/cogniplan_native_profile_v2"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    configure_reproducible_cuda()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(20260908)
    np.random.seed(20260908)
    random.seed(20260908)
    rows = list(csv.DictReader((args.protocol_dir / "cogniplan_profile32.csv").open()))
    assert len(rows) == 32 and all(r["split"] == "train" for r in rows)
    protocol = json.loads((args.protocol_dir / "report.json").read_text())
    for filename, expected in protocol["outputs"].items():
        assert sha(args.protocol_dir / filename) == expected
    root = Path("data/raw/cogniplan/paper_model_exploration")
    members = {r[k] for r in rows for k in ("member", "target_member")}
    arrays = {}
    with tarfile.open(root / "maps_train_inpaint.tar.gz", "r|gz") as tar:
        for member in tar:
            if member.name in members:
                raw = tar.extractfile(member).read()
                arrays[member.name] = np.asarray(Image.open(io.BytesIO(raw))).copy()
                matches = [r for r in rows if r["member"] == member.name]
                if matches:
                    assert hashlib.sha256(raw).hexdigest() == matches[0]["png_sha256"]
    assert set(arrays) == members
    with tarfile.open(root / "checkpoints_wgan_inpainting.tar.gz") as tar:
        checkpoint = tar.extractfile("wgan_inpainting/gen_00500000.pt").read()
        config_raw = tar.extractfile("wgan_inpainting/config.yaml").read()
    old_audit = json.loads(Path("results/cogniplan_training_asset_audit_v1/report.json").read_text())
    assert hashlib.sha256(checkpoint).hexdigest() == old_audit["released_generator"]["checkpoint_sha256"]
    config = load_config(config_raw)
    assert config["batch_size"] == 24 and config["n_critic"] == 5 and config["warmup_iter"] == 10000
    (out / "released_config.json").write_text(json.dumps(config, indent=2) + "\n")
    native = load_native()
    generator = native["Generator"](config["netG"], True).cuda()
    generator.load_state_dict(torch.load(io.BytesIO(checkpoint), weights_only=True, map_location="cpu"), strict=True)
    generator.eval()
    evaluator = native["Evaluator"](config, generator, True, 4)
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu_before": gpu_state(),
                   "concurrent_workload": True, "isolated_paper_latency": False,
                   "precision": "float32_no_tf32", "torch_threads": 4, "cudnn_benchmark": False,
                   "deterministic_algorithms": True, "cudnn_deterministic": True,
                   "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
                   "upstream_execution_adaptation": "Freeze arithmetic and algorithm choices; native architecture/losses unchanged. Native autotuning v1 did not replay across processes."}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    sample_rows, sample_arrays = [], {}
    endpoints = np.random.default_rng(20260908).integers(0, 250, size=(64, 4))
    np.save(out / "engineering_endpoints.npy", endpoints)
    # Warm up the exact native 4-hypothesis inference workload.
    x, mask = (t.cuda() for t in encode_partial(arrays[rows[0]["member"]]))
    with torch.no_grad():
        for _ in range(5):
            for cond in NATIVE_CONDITIONS:
                evaluator.eval_step(x, mask, torch.tensor([cond], device="cuda"), (250, 250))
    torch.cuda.reset_peak_memory_stats()
    for idx, row in enumerate(rows):
        partial = arrays[row["member"]]
        target = arrays[row["target_member"]] != 127
        x, mask = (t.cuda() for t in encode_partial(partial))
        raw = []
        start = sync_time()
        with torch.no_grad():
            for cond in NATIVE_CONDITIONS:
                _, prediction = evaluator.eval_step(x, mask, torch.tensor([cond], device="cuda"), (250, 250))
                raw.append(prediction)
        generation_s = sync_time() - start
        start = sync_time()
        output = torch.cat(raw).cpu().numpy()[:, 0]
        support = np.ones_like(partial, dtype=bool)
        worlds = np.stack([postprocess(v, partial, support) for v in output])
        post_s = sync_time() - start
        # Native reference postprocessing: required exact equality on released inputs.
        reference = np.stack([evaluator.post_process(t, x[:, :, 3:253, 3:253]) > 0 for t in raw])
        assert np.array_equal(worlds, reference)
        assert np.isfinite(output).all()
        assert np.all(worlds[:, partial != 127] == (partial[partial != 127] == 255))
        start = time.perf_counter()
        event_workload(worlds, endpoints)
        event_s = time.perf_counter() - start
        unknown = partial == 127
        vote = worlds.mean(0)
        sample_rows.append({"index": idx, "member": row["member"], "mother_id": row["mother_id"],
                            "layout": row["layout"], "split": "train", "hypotheses": 4,
                            "generation_seconds": generation_s, "postprocess_transfer_seconds": post_s,
                            "connectivity_seconds": event_s, "total_seconds": generation_s + post_s + event_s,
                            "hidden_vote_brier_sanity_only": float(np.mean((vote[unknown] - target[unknown]) ** 2)),
                            "distinct_worlds": len({v.tobytes() for v in worlds}),
                            "native_replay_equal": True, "known_conflicts": 0})
        sample_arrays[f"partial_{idx}"] = partial
        sample_arrays[f"target_{idx}"] = target
        sample_arrays[f"raw_{idx}"] = output
        sample_arrays[f"worlds_{idx}"] = worlds
    inference_peak = torch.cuda.max_memory_allocated()
    np.savez_compressed(out / "native_outputs.npz", **sample_arrays)
    with (out / "native_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0])); w.writeheader(); w.writerows(sample_rows)
    print(json.dumps({"stage": "native_sanity_passed", "samples": 32, "worlds": 128,
                      "known_conflicts": 0, "native_replay_equal": True}), flush=True)
    del evaluator, generator, x, mask, raw, prediction
    torch.cuda.empty_cache()

    # Fixed 32 train-only samples, CPU decoded cache, native rotation augmentation,
    # released batch=24. Both phases use fresh random weights, not a tuned short run.
    phases = []
    trace = []
    for phase in ("reconstruction", "adversarial"):
        torch.manual_seed(20260908)
        random.seed(20260908)
        trainer = native["Trainer"](config)
        generator_parameters = sum(p.numel() for p in trainer.netG.parameters())
        discriminator_parameters = sum(p.numel() for p in trainer.netD.parameters())
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for iteration in range(1, 111):
            start = sync_time()
            ids = torch.randperm(32)[:24].tolist()
            xs, masks, targets, conditions = [], [], [], []
            for idx in ids:
                row = rows[idx]
                rotation = random.randrange(4)
                partial = np.ascontiguousarray(np.rot90(arrays[row["member"]], rotation))
                target = np.ascontiguousarray(np.rot90(arrays[row["target_member"]], rotation))
                a, b = encode_partial(partial)
                xs.append(a); masks.append(b); targets.append(encode_target(target))
                conditions.append([float(row["layout"] == name) for name in ("room", "tunnel", "outdoor")])
            batch = (torch.cat(xs).cuda(), torch.cat(masks).cuda(), torch.cat(targets).cuda(),
                     torch.tensor(conditions, device="cuda"))
            g_step = iteration % config["n_critic"] == 0
            losses = native_update(trainer, batch, adversarial=phase == "adversarial", generator_update=g_step)
            duration = sync_time() - start
            if iteration > 10:
                elapsed.append(duration)
                trace.append({"phase": phase, "iteration": iteration - 10,
                              "generator_updated": phase == "reconstruction" or g_step,
                              "seconds": duration, "losses": losses})
            if iteration % 10 == 0:
                (out / "progress.json").write_text(json.dumps({"phase": phase, "updates_with_warmup": iteration,
                    "target_updates_with_warmup": 110, "last_seconds": duration, "last_losses": losses}, indent=2) + "\n")
                print(json.dumps({"stage": phase, "updates": iteration, "seconds": round(duration, 3)}), flush=True)
        assert all(torch.isfinite(p).all() for p in trainer.parameters())
        phases.append({"phase": phase, "warmup_iterations": 10, "measured_iterations": 100,
                       "generator_updates_measured": 100 if phase == "reconstruction" else 20,
                       "discriminator_updates_measured": 0 if phase == "reconstruction" else 100,
                       "batch_size": 24, "seconds_mean": float(np.mean(elapsed)),
                       "seconds_p50": float(np.median(elapsed)), "seconds_p95": float(np.quantile(elapsed, .95)),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "all_finite": True})
        del trainer, batch
        torch.cuda.empty_cache()
    (out / "training_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    estimated = phases[0]["seconds_mean"] * 10000 + phases[1]["seconds_mean"] * 490000
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
              "kind": "train-only native compatibility and resource profile",
              "test_assets_opened": False, "formal_training_started": False,
              "convergence_demonstrated": False, "native_interface_passed": True,
              "native_quality_status": "All preselected outputs preserved; visual review still required. No original-test reproduction claim.",
              "native_samples": 32, "native_worlds": 128, "known_conflicts": 0,
              "native_postprocess_exact_replay": True,
              "hypotheses": NATIVE_CONDITIONS, "public_checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
              "inference_peak_allocated_bytes": inference_peak,
              "generator_parameters": generator_parameters, "discriminator_parameters": discriminator_parameters,
              "inference_seconds_mean": {k: float(np.mean([r[k] for r in sample_rows])) for k in
                                         ("generation_seconds", "postprocess_transfer_seconds", "connectivity_seconds", "total_seconds")},
              "inference_workload": "4 serial native forwards; CPU native morphology; 64 fixed label-free endpoints x radii 0/3/6 cells; 4-neighbor exact-disk components. Encoding/H2D excluded, D2H included. Engineering only.",
              "training_phases": phases, "original_recipe_training_only_hours_per_seed_estimate": estimated / 3600,
              "original_recipe_training_only_hours_three_seeds_serial_estimate": estimated * 3 / 3600,
              "estimate_limitations": "Extrapolated from 32 decoded cached train views with native rotation under concurrent GPU load. Excludes full-loader I/O, validation/checkpointing; not an ETA or accepted formal budget. Adversarial phase starts fresh solely to measure its computation.",
              "environment": environment, "gpu_after": gpu_state(),
              "inputs": {str(p): sha(p) for p in (Path(__file__), Path("src/pathrel/cogniplan.py"), args.protocol_dir / "report.json", args.protocol_dir / "cogniplan_profile32.csv")},
              "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in {"progress.json", "report.json"}}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "progress.json").write_text(json.dumps({"status": "complete", "report": str(out / "report.json")}, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
