#!/usr/bin/env python3
"""Independent, CPU-only final audit for all three registered method repeats.

This file does not start models or repair predictions. All candidate receipts,
training history, selected calibration/validation artifacts and convergence are
checked before an immutable final_audits/<method>.json can be issued.

Pilot visual receipt contract, in addition to the frozen sealer fields:
  stage_metrics_sha256: {untrained: SHA, smoke: SHA, pilot: SHA}
  render_manifest: {path: project-relative convergence manifest.json, sha256: SHA}
  pilot_quality_passed: true; pilot_no_persistent_collapse: true
  case_reviews: all 3 stages x 18 original frozen category entries, each with
    stage, case_category, global_id, query_id, predictions_sha256,
    reviewed: true, no_unexplained_collapse: true, notes_zh: nonempty string.
    Pilot entries additionally require persistent_collapse_detected: false.
Earlier stages may have explained initialization collapse; they are references,
not required to pass the pilot's quality standard. Human judgment is explicitly
identified, while all candidate collapse statistics are independently computed.
The actual convergence manifest must be complete and bind every 16-query /
20-figure / 60-export artifact; a report-layout approval is not this review.
Renderer schema v2 preserves actual worlds (four, deterministic one, or no map
for direct-query control), with raw direct event scores explicitly identified.

Run only after all three real full repeats exist. CPU fixture tests never write
a passed final receipt into the actual experiment output tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pathrel.formal_data import sha256, load_formal_data
from scripts import audit_flatlands_formal_stage as independent

SEEDS = [20260831, 20260901, 20260902]


def need(condition, message):
    independent.require(condition, message)


def finite(value, *, nonnegative=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not nonnegative or value >= 0))


def recovery_identity(sampler, views_state):
    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return dict(size=sampler["size"], batch_size=sampler["batch_size"], cursor=sampler["cursor"], epoch=sampler["epoch"],
        permutation_dtype=str(sampler["permutation"].dtype), permutation_sha256=tensor_hash(sampler["permutation"]),
        sampler_generator_sha256=tensor_hash(sampler["generator_state"]), views_generator_sha256=tensor_hash(views_state))


def path(name):
    target = (ROOT / name).resolve()
    need(target.is_relative_to(ROOT.resolve()), "Artifact escapes project")
    return target


def read(name):
    return json.loads(Path(name).read_text())


def bound(owner, name_key, hash_key):
    target = path(owner[name_key])
    need(target.is_file() and sha256(target) == owner[hash_key], "Referenced artifact/hash mismatch: " + name_key)
    return read(target), target


def write_new(target, payload):
    """Atomically publish a complete new artifact; never replace a prior result."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, prefix=".audit-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, target)
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def choose_checkpoint(candidates):
    """Independently form the global tie set, then prefer NLL and earlier step."""
    need(bool(candidates) and len({c["step"] for c in candidates}) == len(candidates), "Duplicate/empty candidate steps")
    for row in candidates:
        need(type(row["step"]) is int and row["step"] > 0, "Invalid candidate step")
        need(finite(row["event_brier"], nonnegative=True), "Nonfinite candidate Event Brier")
        need(row["map_nll"] is None or finite(row["map_nll"], nonnegative=True), "Nonfinite candidate map NLL")
    minimum = min(row["event_brier"] for row in candidates)
    eligible = [row for row in candidates if row["event_brier"] <= minimum + .00001]
    return sorted(eligible, key=lambda row: (0. if row["map_nll"] is None else row["map_nll"], row["step"]))[0]


def session_cost(folder):
    """Recompute real session time; a missing finish retains a visible lower bound."""
    total, missing, files = 0., [], []
    starts = sorted((folder / "sessions").glob("*.started.json"))
    need(bool(starts), "Actual training session receipts are absent")
    names = {start.name.removesuffix(".started.json") + suffix for start in starts
             for suffix in (".started.json", ".heartbeat.json", ".finished.json", ".failure.json")}
    need(all(file.name in names for file in (folder / "sessions").glob("*.json")), "Orphan or unknown session receipt")
    for start in starts:
        token = start.name.removesuffix(".started.json")
        need(read(start).get("session_id") == token, "Session start token differs")
        finished = start.with_name(token + ".finished.json")
        heartbeat = start.with_name(token + ".heartbeat.json")
        source = finished if finished.exists() else heartbeat if heartbeat.exists() else None
        if not finished.exists():
            missing.append(token)
        elapsed = 0. if source is None else read(source).get("elapsed_seconds")
        need(finite(elapsed, nonnegative=True), "Invalid actual session duration")
        total += elapsed
        files.append(dict(path=str(start.relative_to(ROOT)), sha256=sha256(start)))
        for receipt in (finished, heartbeat, start.with_name(token + ".failure.json")):
            if receipt.exists():
                record = read(receipt)
                need(record.get("session_id") == token, "Session receipt token differs")
                value = record.get("wall_seconds_this_session" if receipt.name.endswith(".failure.json") else "elapsed_seconds")
                need(finite(value, nonnegative=True), "Invalid session heartbeat/finish/failure")
                files.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha256(receipt)))
        if heartbeat.exists() and finished.exists():
            need(read(heartbeat)["elapsed_seconds"] <= read(finished)["elapsed_seconds"], "Heartbeat exceeds completed session time")
    need(finite(total, nonnegative=True), "Session total overflow")
    return dict(training_wall_seconds=total, wall_time_is_lower_bound=bool(missing),
                sessions_without_final_receipt=missing), files


def check_training(folder, method, seed, member, full_steps, protocol_hash, full_plan_hash, protocol):
    receipt = read(folder / "run.json")
    independent.compare(receipt, dict(schema_version=1, data_workers=0, prefetch=False,
        checkpoint_interval=protocol["training_runtime"]["checkpoint_interval"],
        maximum_journaled_updates_replayed_after_unsafe_interruption=protocol["training_runtime"]["checkpoint_interval"]), "Durable training store policy")
    recipe = receipt["recipe"]
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    need(digest == receipt["recipe_sha256"], "Training recipe digest changed")
    need(recipe["protocol_sha256"] == protocol_hash and recipe["method"] == method
         and independent.recipe_outer_seed(recipe) == seed and recipe["member"] == member, "Training identity mismatch")
    independent.compare(recipe["config"], protocol["methods"][method], "registered optimizer/model recipe")
    need(recipe["test_assets_opened"] is False and recipe["validation_loaded_by_training_process"] is False,
         "Training declared validation or locked-test access")
    progress_path = folder / "progress.jsonl"
    lines = progress_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    need([r["step"] for r in rows] == list(range(1, full_steps + 1)), "Training progress is not exactly the complete full budget")
    train_ids = {r["global_id"] for r in protocol["scenes"]["train"]}
    grouped = {}
    for sample in protocol["scenes"]["train"]:
        grouped.setdefault(sample["parent_group"], []).append(sample["global_id"])
    parent_views = [grouped[key] for key in sorted(grouped)]
    need(len(parent_views) == protocol["partition"]["train"] and all(len(v) == protocol["partition"]["train_views_per_parent"] for v in parent_views),
         "Training parent/view matrix differs")
    import torch
    parent_rng = torch.Generator(device="cpu").manual_seed(seed)
    view_rng = torch.Generator(device="cpu").manual_seed(seed + 2)
    permutation, cursor, epoch = torch.randperm(len(parent_views), generator=parent_rng).tolist(), 0, 0
    milestones = {protocol["staged"]["smoke_steps"], protocol["staged"]["pilot_steps"], full_steps,
                  *range(protocol["staged"]["pilot_steps"] + 2500, full_steps + 1, 2500)}
    recovery_states, update_total = {}, 0.
    for row in rows:
        need(bool(row["losses"]) and all(finite(v) for v in row["losses"].values()),
             "Nonfinite/missing full-training losses")
        need(len(row["global_ids"]) == protocol["methods"][method]["batch_size"]
             and set(row["global_ids"]).issubset(train_ids), "Training batch escaped frozen training observations")
        need(finite(row["update_seconds"], nonnegative=True), "Invalid optimizer update duration")
        update_total += row["update_seconds"]
        need(all(type(row[key]) is int and row[key] >= 0 for key in ("peak_allocated_bytes", "peak_reserved_bytes")), "Invalid journal peak GPU memory")
        ids = []
        for _ in range(protocol["methods"][method]["batch_size"]):
            if cursor == len(permutation):
                permutation, cursor = torch.randperm(len(parent_views), generator=parent_rng).tolist(), 0
                epoch += 1
            ids.append(permutation[cursor]); cursor += 1
        views = torch.randint(protocol["partition"]["train_views_per_parent"], (len(ids),), generator=view_rng).tolist()
        need(row["parent_indices"] == ids and row["view_indices"] == views
             and row["global_ids"] == [parent_views[i][v] for i, v in zip(ids, views)], "Independent parent/view RNG replay differs from training progress")
        fraction = max(0., min(1., (row["step"] - protocol["staged"]["pilot_steps"]) / (full_steps - protocol["staged"]["pilot_steps"])))
        expected_lr = protocol["methods"][method]["learning_rate"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
        independent.compare(row["learning_rates"], {key: expected_lr for key in (["model", "discriminator"] if method == "lama" else ["model"])},
                            "Full progress scheduled learning rate")
        if row["step"] in milestones:
            recovery_states[str(row["step"])] = recovery_identity(dict(size=len(parent_views), batch_size=len(ids), epoch=epoch,
                cursor=cursor, permutation=torch.tensor(permutation, dtype=torch.int64), generator_state=parent_rng.get_state()), view_rng.get_state())
    costs, source_files = session_cost(folder)
    for file in source_files:
        if not file["path"].endswith(".failure.json"):
            independent.compare(read(path(file["path"])), dict(method=method, seed=seed, member=member,
                protocol_sha256=protocol_hash, test_assets_opened=False), "Training process session identity")
    session_ids = {Path(f["path"]).name.removesuffix(".started.json") for f in source_files if f["path"].endswith(".started.json")}
    need(all(row["session_id"] in session_ids for row in rows), "Progress lacks an actual training process session")
    time_summary = read(folder / "time_summary.json")
    independent.compare(time_summary, costs, "reconciled actual training duration")
    need(update_total <= costs["training_wall_seconds"] + 1e-8, "Effective optimizer update duration exceeds actual session wall time")
    snapshot = folder / f"step_{full_steps:08d}.pt"
    boundary = read(folder / f"boundary_{full_steps:08d}.json")
    need(boundary["checkpoint_sha256"] == sha256(snapshot) and boundary["protocol_sha256"] == protocol_hash,
         "Final training boundary digest mismatch")
    need(sha256(folder / "latest.pt") == boundary["checkpoint_sha256"], "Latest recoverable checkpoint differs from the final completed boundary")
    independent.compare(boundary, dict(method=method, seed=seed, member=member, completed_steps=full_steps,
                                      test_assets_opened=False), "Final boundary identity")
    record = dict(method=method, seed=seed, member=member, completed_steps=full_steps,
                  run_receipt_sha256=sha256(folder / "run.json"), progress_sha256=sha256(progress_path),
                  actual_training_cost=costs, session_files=source_files,
                  time_summary_sha256=sha256(folder / "time_summary.json"),
                  peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows),
                  peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in rows),
                  final_checkpoint=str(snapshot.relative_to(ROOT)), final_checkpoint_sha256=sha256(snapshot),
                  full_plan_sha256=full_plan_hash)
    record.update(expected_recovery_state_by_step=recovery_states, canonical_optimizer_update_seconds=update_total)
    return record, rows, lines


def check_checkpoint(member, method, seed, index, expected_step, protocol, protocol_hash, full_plan_hash, *, full_steps, expected_recovery, deep=False):
    """Metadata for every candidate; real tensor finiteness for selected/final weights."""
    checkpoint = path(member["checkpoint"])
    need(sha256(checkpoint) == member["checkpoint_sha256"], "Model checkpoint checksum mismatch")
    need(member["member_index"] == index and member["initialization_seed"] == protocol["member_seeds"][method][str(seed)][index]
         and member["completed_steps"] == expected_step, "Checkpoint seed/member/step mismatch")
    recipe_file = path(member["run_receipt"])
    run = read(recipe_file)
    encoded = json.dumps(run["recipe"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    recipe_hash = hashlib.sha256(encoded).hexdigest()
    need(recipe_hash == run["recipe_sha256"] == member["recipe_sha256"], "Model recipe hash mismatch")
    recipe = run["recipe"]
    need(recipe["protocol_sha256"] == protocol_hash and recipe["method"] == method
         and independent.recipe_outer_seed(recipe) == seed and recipe["member"] == index, "Checkpoint recipe identity mismatch")
    import torch
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = envelope["state"]
    need(envelope["schema_version"] == state["schema_version"] == run["schema_version"] == 1, "Unsupported/missing checkpoint schema")
    need(envelope["recipe_sha256"] == recipe_hash and envelope["progress_index"] == state["completed_steps"] == expected_step,
         "Checkpoint envelope progress mismatch")
    independent.compare(member["checkpoint_snapshot_cost"], dict(
        training_wall_seconds=state["extra"]["training_wall_seconds"], optimizer_update_seconds=state["extra"]["optimizer_update_seconds"],
        peak_allocated_bytes=state["extra"].get("peak_allocated_bytes", state["extra"].get("peak_allocated_bytes_this_session")),
        peak_reserved_bytes=state["extra"].get("peak_reserved_bytes", state["extra"].get("peak_reserved_bytes_this_session"))), "Actual checkpoint resource snapshot")
    lines = (recipe_file.parent / "progress.jsonl").read_bytes().splitlines(keepends=True)
    need(len(lines) >= expected_step and envelope["journal_sha256"] == hashlib.sha256(b"".join(lines[:expected_step])).hexdigest(),
         "Checkpoint journal hash differs from the exact complete-update prefix")
    independent.compare(state["extra"]["optimizer_update_seconds"], sum(json.loads(line)["update_seconds"] for line in lines[:expected_step]),
                        "Checkpoint canonical optimizer-update cost")
    names = {"model", "discriminator"} if method == "lama" else {"model"}
    for group in ("models", "optimizers", "schedulers", "model_training_modes", "parameter_requires_grad"):
        need(set(state[group]) == names, "Incomplete checkpoint model/optimizer registration")
    for group in ("model_training_modes", "parameter_requires_grad"):
        need(all(values and all(flag is True for flag in values.values()) for values in state[group].values()), "Checkpoint retained incorrect training/gradient flags")
    need(set(state["rng"]) >= {"python", "numpy", "torch_cpu", "torch_cuda"}, "Incomplete checkpoint RNG state")
    need(set(state["generators"]) == {"noise", "views"}, "Missing dedicated noise/view RNG")
    random.Random().setstate(state["rng"]["python"])
    np.random.RandomState().set_state(state["rng"]["numpy"])
    torch.Generator(device="cpu").set_state(state["rng"]["torch_cpu"])
    torch.Generator(device="cpu").set_state(state["generators"]["views"]["state"])
    need(state["generators"]["views"]["device"] == "cpu" and state["generators"]["noise"]["device"].startswith("cuda"),
         "Dedicated RNG devices differ from training recipe")
    need(state["rng"]["cuda_initialized"] is True and bool(state["rng"]["torch_cuda"]), "Formal GPU training CUDA RNG state missing")
    for value in [*state["rng"]["torch_cuda"], state["generators"]["noise"]["state"], state["sampler"]["generator_state"]]:
        need(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
             "Malformed saved random-state tensor")
    need(state["sampler"]["size"] == protocol["partition"]["train"]
         and state["sampler"]["batch_size"] == protocol["methods"][method]["batch_size"], "Wrong parent sampler size/batch")
    permutation = state["sampler"]["permutation"].cpu().numpy()
    need(state["sampler"]["permutation"].dtype == torch.int64 and np.array_equal(np.sort(permutation), np.arange(protocol["partition"]["train"])), "Invalid parent sampler permutation")
    need(type(state["sampler"]["cursor"]) is int and 0 <= state["sampler"]["cursor"] <= protocol["partition"]["train"]
         and type(state["sampler"]["epoch"]) is int and state["sampler"]["epoch"] >= 0, "Invalid sampler cursor/epoch")
    independent.compare(recovery_identity(state["sampler"], state["generators"]["views"]["state"]), expected_recovery,
                        "Checkpoint exact replayed parent/view random terminal state")
    if expected_step > protocol["staged"]["pilot_steps"]:
        need(state["extra"]["full_plan_sha256"] == full_plan_hash, "Checkpoint did not attach the sealed full schedule")
    for name in names:
        schedule = state["schedulers"][name]
        need(schedule["completed_steps"] == expected_step and schedule["pilot_steps"] == protocol["staged"]["pilot_steps"],
             "Scheduler state does not match optimizer progress")
        expected_end = full_steps if state["extra"].get("full_plan_sha256") is not None else None
        need(state["extra"].get("full_plan_sha256") in (None, full_plan_hash)
             and schedule["full_steps"] == expected_end and schedule["minimum_ratio"] == .1,
             "Checkpoint LR schedule does not match the gated full plan")
        parameters = state["parameter_requires_grad"][name]
        need(set(parameters).issubset(state["models"][name]), "Missing registered parameter values")
        optim = state["optimizers"][name]
        ids = [p for group in optim["param_groups"] for p in group["params"]]
        need(bool(parameters) and len(ids) == len(parameters) and len(ids) == len(set(ids))
             and bool(optim["state"]) and set(optim["state"]).issubset(ids), "Optimizer parameter identities incomplete/duplicated")
        config = protocol["methods"][method]
        need(schedule["base_lrs"] == [config["learning_rate"]] * len(optim["param_groups"]), "Checkpoint base LR differs from protocol")
        fraction = 0. if schedule["full_steps"] is None else max(0., min(1.,
            (expected_step - schedule["pilot_steps"]) / (schedule["full_steps"] - schedule["pilot_steps"])))
        expected_lr = config["learning_rate"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
        for group in optim["param_groups"]:
            independent.compare(group["lr"], expected_lr, "Actual optimizer scheduled LR")
            need(group["weight_decay"] == config["weight_decay"] and list(group["betas"]) == config["betas"], "Optimizer decay/betas differ from protocol")
        for parameter_id, parameter_name in zip(ids, parameters):
            if parameter_id not in optim["state"]:
                continue  # Legal unused factor-head parameters retain no Adam state.
            parameter_state = optim["state"][parameter_id]
            need(set(parameter_state) >= {"step", "exp_avg", "exp_avg_sq"}
                 and 0 < float(parameter_state["step"]) <= expected_step, "AdamW moment/step state is incomplete")
            if method in ("lama", "flow"):
                need(float(parameter_state["step"]) == expected_step, "External model AdamW progress differs from complete updates")
            parameter = state["models"][name][parameter_name]
            for key in ("exp_avg", "exp_avg_sq"):
                need(parameter_state[key].shape == parameter.shape and parameter_state[key].dtype == parameter.dtype,
                     "AdamW moment shape/dtype differs from registered model parameter")
        if name == "model":
            count = sum(state["models"][name][key].numel() for key in parameters)
            need(count == member["model_parameters"], "Actual model parameter count differs from evaluation provenance")
        if deep:
            def tensors(value):
                if isinstance(value, torch.Tensor):
                    yield value
                elif isinstance(value, dict):
                    for child in value.values():
                        yield from tensors(child)
                elif isinstance(value, (tuple, list)):
                    for child in value:
                        yield from tensors(child)
            for value in tensors((state["models"][name], optim)):
                if value.is_floating_point() or value.is_complex():
                    need(bool(torch.isfinite(value).all()), "Nonfinite stored model/optimizer tensor")
    need(not torch.cuda.is_initialized(), "Final audit initialized CUDA")
    result = dict(checkpoint=str(checkpoint.relative_to(ROOT)), sha256=member["checkpoint_sha256"],
                  step=expected_step, tensor_finiteness_checked=deep, model_parameter_count=member["model_parameters"])
    del envelope, state
    gc.collect()
    return result


def check_evaluation(folder, method, seed, step, protocol, protocol_hash, required_splits):
    metrics, complete = read(folder / "metrics.json"), read(folder / "complete.json")
    need(complete["passed"] is True and complete["metrics_sha256"] == sha256(folder / "metrics.json"), "Evaluation completion hash mismatch")
    count = protocol["methods"][method]["members"]
    need(metrics["method"] == complete["method"] == method and metrics["outer_seed"] == complete["seed"] == seed
         and metrics["protocol_sha256"] == protocol_hash and complete["completed_steps"] == [step] * count,
         "Evaluation method/repeat/step mismatch")
    need(complete["stage"] == metrics["stage"] and complete["final_test_locked"] is True
         and complete["new_physical_test_images_opened"] == 0, "Completion stage/test identity differs")
    need(set(metrics["splits"]) == set(required_splits) and metrics["main_table_eligible"] is False,
         "Evaluation split coverage or eligibility mismatch")
    need(metrics["final_test_locked"] and metrics["location_6_locked"] and metrics["new_physical_test_images_opened"] == 0
         and metrics["observed_and_support_constraints_passed"], "Evaluation test/constraint gate failed")
    members = metrics["checkpoint_provenance"]
    need(len(members) == count and [m["member_index"] for m in members] == list(range(count)), "Missing/duplicate members")
    need([m["initialization_seed"] for m in members] == protocol["member_seeds"][method][str(seed)], "Wrong member seeds")
    need([m["completed_steps"] for m in members] == [step] * count, "Member progress differs from completed evaluation")
    before = read(folder / "evaluation_frozen_before_scoring.json")
    independent.compare(before, dict(method=method, seed=seed, stage=metrics["stage"], protocol_sha256=protocol_hash,
        evaluator_sha256=protocol["source_hashes"]["scripts/evaluate_flatlands_formal.py"],
        metric_version="flatlands-formal-metrics-v1", checkpoint_provenance=members, one_member_diagnostic=False,
        new_physical_test_images_opened=0, final_test_locked=True, location_6_locked=True), "Pre-scoring evaluation identity")
    need(finite(metrics["evaluation_seconds"], nonnegative=True), "Invalid total evaluation duration")
    for split in required_splits:
        expected = {r["global_id"] for r in protocol["scenes"][split]}
        cases = metrics["splits"][split]["cases"]
        need(len(cases) == len(expected) and {c["global_id"] for c in cases} == expected, "Incomplete case coverage")
        for case in cases:
            need(sha256(folder / split / "predictions" / (case["global_id"] + ".npz")) == case["predictions_sha256"],
                 "Candidate prediction artifact changed")
        check_efficiency(metrics["splits"][split], method)
    fitted = read(folder / "calibration_platt.json")
    identity = [{k: m.get(k) for k in ("member_index", "initialization_seed", "checkpoint_sha256", "completed_steps")} for m in members]
    need(fitted["fit_split"] == "calibration" and fitted["protocol_sha256"] == protocol_hash
         and fitted["prediction_model_identity"] == identity, "Calibration/checkpoint identity mismatch")
    return metrics, fitted, identity


def check_efficiency(report, method):
    costs = [row["efficiency"] for row in report["cases"]]
    need(bool(costs), "Empty efficiency case set")
    expected_k = None if method == "direct_query" else 1 if method == "deterministic" else 4
    for row in costs:
        independent.compare(row, dict(actual_world_count=expected_k, common_budget_K=4,
            actual_members=4 if method == "lama" else 1, stage_one_member_diagnostic=False), "Common evaluation cost")
        for key in ("generation_seconds", "full_update_seconds", "actual_batch_forward_calls", "actual_model_input_examples"):
            need(finite(row[key], nonnegative=True), "Nonfinite/negative inference cost")
        need(row["generation_seconds"] <= row["full_update_seconds"], "Generation exceeds complete update time")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            need(row[key] is None or type(row[key]) is int and row[key] >= 0, "Invalid peak GPU memory")
        if method == "lama":
            independent.compare(row, dict(actual_batch_forward_calls=4, actual_model_input_examples=4), "LaMa K4 forward budget")
        if method == "flow":
            independent.compare(row, dict(velocity_evaluations_per_sample=50, actual_batch_forward_calls=50,
                actual_model_input_examples=400, solver="Heun", steps_per_sample=25, samples_per_observation=4,
                cfg_scale_s=2., cfg_branches=2, batched_model_calls=50, sample_equivalent_forwards=400,
                condition_cache_used=False), "FM Heun25 actual forward budget")
    expected = dict(case_count=len(costs), includes_first_case_cold_execution=True)
    for timing in ("generation_seconds", "full_update_seconds"):
        expected[timing + "_mean"] = float(np.mean([c[timing] for c in costs]))
        expected[timing + "_median"] = float(np.median([c[timing] for c in costs]))
    for key in ("actual_batch_forward_calls", "actual_model_input_examples"):
        expected[key] = sum(c[key] for c in costs)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        expected[key] = max((c[key] for c in costs if c[key] is not None), default=None)
    independent.compare(report["efficiency"], expected, "Independently aggregated inference cost")
    return expected


def check_cached_inputs(sample):
    """Audit cached masks and query eligibility without opening original images."""
    observation = np.asarray(sample.observation)
    need(observation.shape == (3, *sample.target.shape) and np.isin(observation, (0, 1)).all(), "Nonbinary/invalid observation channels")
    need(np.isin(sample.target, (0, 1)).all() and np.isin(sample.valid, (0, 1)).all()
         and np.isin(sample.hidden, (0, 1)).all(), "Nonbinary cached labels/masks")
    valid, hidden = np.asarray(sample.valid, bool), np.asarray(sample.hidden, bool)
    need(not (hidden & ~valid).any() and np.array_equal(observation[2], hidden), "Hidden/unknown/support mismatch")
    need(np.array_equal(observation.sum(0), valid), "Observation channels overlap or disagree with support")
    need(not observation[0, ~valid].any() and not np.asarray(sample.target)[~valid].any(), "Input/reference escapes support")
    known = valid & ~hidden
    need(np.array_equal(observation[0, known], sample.target[known]), "Known observation/reference contradiction")
    need(sample.starts.shape == sample.goals.shape == (len(sample.candidate_indices), 2)
         and sample.targets.shape == (len(sample.starts), 3), "Malformed cached query matrix")
    for start, goal in zip(sample.starts, sample.goals):
        need(all(0 <= int(v) < size for rc in (start, goal) for v, size in zip(rc, valid.shape)), "Out-of-bounds cached query")
        need(observation[(0, *start)] == 1 and valid[tuple(start)] and hidden[tuple(goal)]
             and valid[tuple(goal)] and not np.array_equal(start, goal), "Query violates input-only eligibility")


def candidate_scores(evaluation, split, samples, metrics):
    """Recompute every candidate's selection scalars from label-free saved worlds.

    Connectivity uses the separately version-bound historical exact CPU oracle;
    selected outputs additionally undergo explicit-disk cross-checks downstream.
    """
    method = metrics["method"]
    expected_k = None if method == "direct_query" else 1 if method == "deterministic" else 4
    cases = metrics["splits"][split]["cases"]
    need([c["global_id"] for c in cases] == [s.row["global_id"] for s in samples], "Candidate case order differs")
    maps, ps, ys, parents = [], [], [], []
    for sample, case in zip(samples, cases):
        check_cached_inputs(sample)
        file = evaluation / split / "predictions" / (sample.row["global_id"] + ".npz")
        need(sha256(file) == case["predictions_sha256"], "Candidate NPZ hash differs")
        with np.load(file, allow_pickle=False) as z:
            fields = {"global_id", "candidate_indices", "starts", "goals", "radii_cells", "event_scores"}
            if expected_k is not None:
                fields |= {"worlds", "world_events", "continuous_score"}
            need(set(z.files) == fields and str(z["global_id"]) == sample.row["global_id"], "Candidate predictions are not label-free")
            for key, expected in (("candidate_indices", sample.candidate_indices), ("starts", sample.starts),
                                  ("goals", sample.goals), ("radii_cells", np.array([0, 10, 20]))):
                need(np.array_equal(z[key], expected), "Candidate changed frozen query/radius")
            p = z["event_scores"]
            need(p.shape == sample.targets.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid candidate event score")
            if expected_k is not None:
                worlds = z["worlds"]
                need(worlds.shape == (expected_k, *sample.target.shape), "Candidate actual K differs")
                value = independent.independent_map_case(sample, worlds, z["continuous_score"])
                independent.compare(case["map"], value, "Candidate independent map/collapse")
                maps.append(value)
                events = independent._accelerated_events(worlds, sample.starts, sample.goals, (0, 10, 20))
                need(np.array_equal(events, z["world_events"]) and np.array_equal(events.mean(0), p), "Candidate world-event geometry differs")
            ps.extend(p.ravel().tolist()); ys.extend(sample.targets.ravel().tolist())
            parents.extend([sample.scene_key] * p.size)
    event = independent.independent_event_summary(ps, ys, parents, "scene_weighted")
    map_report = independent.independent_map_report(maps, [s.scene_key for s in samples]) if maps else None
    result = dict(event_brier=event["event_brier"], map_brier=map_report["scene_weighted"]["sample_vote_cell_brier"] if maps else None,
                  map_nll=map_report["scene_weighted"]["sample_vote_cell_nll"] if maps else None,
                  collapse_diagnostic=map_report["collapse_diagnostic"] if maps else None)
    independent.compare(metrics["summary"][split], result, "Candidate independently recomputed selection scalars")
    return result


def frozen_figure_cases(protocol):
    figures = read(path(protocol["figures"]["cases"]))
    return figures["general_cases"] + figures["multi_radius_cases"] + [
        case for category in figures["reference_categories"].values() for case in category["cases"]]


def check_pilot_visual(visual, numeric, protocol, stage_metrics):
    """Check human model-output reviews against exact preselected case artifacts."""
    need(visual["passed"] is True and visual["figure_cases_sha256"] == protocol["figures"]["sha256"]
         and set(visual["reviewed_stages"]) == {"untrained", "smoke", "pilot"}, "Pilot visual review lacks the frozen case/stage scope")
    expected_hashes = {stage: numeric["stage_metrics"][stage]["sha256"] for stage in ("untrained", "smoke", "pilot")}
    need(visual["stage_metrics_sha256"] == expected_hashes, "Pilot visual review refers to other stage predictions")
    check_convergence_render(visual["render_manifest"], numeric, protocol, stage_metrics)
    need(visual["pilot_quality_passed"] is True and visual["pilot_no_persistent_collapse"] is True,
         "Pilot model quality has not passed independent visual review")
    cases = frozen_figure_cases(protocol)
    expected = {(stage, case["case_category"], case["global_id"], case["query_id"])
                for stage in expected_hashes for case in cases}
    rows = visual["case_reviews"]
    need(len(rows) == len(expected) and
         {(r["stage"], r["case_category"], r["global_id"], r["query_id"]) for r in rows} == expected,
         "Pilot model visual review omitted/duplicated a frozen stage case")
    by_stage = {stage: {r["global_id"]: r for r in metrics["splits"]["validation"]["cases"]} for stage, metrics in stage_metrics.items()}
    for review in rows:
        need(review["reviewed"] is True and review["no_unexplained_collapse"] is True
             and isinstance(review["notes_zh"], str) and bool(review["notes_zh"].strip()), "Pilot case lacks a model-behavior visual review")
        need(review["predictions_sha256"] == by_stage[review["stage"]][review["global_id"]]["predictions_sha256"],
             "Pilot case review refers to different generated maps")
        if review["stage"] == "pilot":
            need(review["persistent_collapse_detected"] is False, "Pilot retained visually detected collapse")


def check_convergence_render(reference, numeric, protocol, stage_metrics):
    manifest, file = bound(reference, "path", "sha256")
    pilot = stage_metrics["pilot"]
    expected_k = None if pilot["method"] == "direct_query" else 1 if pilot["method"] == "deterministic" else 4
    semantics = "direct_query_sigmoid" if expected_k is None else "raw_world_event_frequency"
    columns = ["observed", "reference"] + (["direct_event_score"] if expected_k is None else [f"world_{i}" for i in range(expected_k)])
    stage_hashes = {stage: numeric["stage_metrics"][stage]["sha256"] for stage in ("untrained", "smoke", "pilot")}
    independent.compare(manifest, dict(schema_version=2, method=pilot["method"], outer_seed=pilot["outer_seed"],
        protocol_sha256=pilot["protocol_sha256"], figure_cases_sha256=protocol["figures"]["sha256"],
        figure_geometry_audit_sha256=protocol["figures"]["geometry_verification_sha256"],
        data_seal_sha256=protocol["data_seal_sha256"], complete=True, missing_stages=[],
        visual_gate_passed=None, validation_only=True, new_physical_test_images_opened=0,
        raw_archive_opened=False, final_test_locked=True, location_6_locked=True,
        stage_order=["untrained", "smoke", "pilot"], stage_metrics_sha256=stage_hashes,
        actual_world_count=expected_k, common_world_budget=4, event_score_semantics=semantics,
        column_layout=columns), "Actual convergence render identity and completion")
    need(manifest["world_order"] is None if expected_k is None else isinstance(manifest["world_order"], str) and bool(manifest["world_order"]),
         "Convergence world-order declaration disagrees with actual output type")
    need(sha256(file.parent / "renderer_source.py") == manifest["renderer_sha256"], "Convergence renderer source snapshot changed")
    for relative, digest in manifest["dependency_source_hashes"].items():
        # Frozen numerical dependencies must still match. Mutable renderer/UI
        # dependencies are identified by their render-time hash in this receipt.
        if relative in protocol["source_hashes"]:
            need(digest == protocol["source_hashes"][relative], "Convergence renderer used changed frozen numerical dependency")
    expected_figures, expected_queries = set(), set()
    for case in frozen_figure_cases(protocol):
        key = (case["global_id"], case["query_id"])
        expected_queries.add(key)
        radii = [0, 10, 20] if case["case_category"] == "multi_radius" else [case.get("witness_radius_cells", case.get("bottleneck_failed_radius_cells", 10))]
        expected_figures.update((*key, radius) for radius in radii)
    need(manifest["unique_frozen_queries"] == manifest["rendered_unique_queries"] == len(expected_queries)
         and len(manifest["cases"]) == len(expected_queries)
         and {(c["global_id"], c["query_id"]) for c in manifest["cases"]} == expected_queries, "Convergence render omitted frozen queries")
    need(len(manifest["figures"]) == len(expected_figures)
         and {(f["global_id"], f["query_id"], f["radius_cells"]) for f in manifest["figures"]} == expected_figures,
         "Convergence render omitted/duplicated frozen query-radius figures")
    for row in manifest["cases"] + manifest["figures"]:
        need(set(row["stage_sources"]) == {"untrained", "smoke", "pilot"}, "Convergence panel stage coverage differs")
        for stage, source in row["stage_sources"].items():
            independent.compare(source, dict(actual_world_count=expected_k, event_score_semantics=semantics), "Convergence panel actual output semantics")
            case = next(c for c in stage_metrics[stage]["splits"]["validation"]["cases"] if c["global_id"] == row["global_id"])
            need(source["status"] == "complete" and source["metrics_sha256"] == numeric["stage_metrics"][stage]["sha256"]
                 and source["prediction_sha256"] == case["predictions_sha256"]
                 and sha256(path(source["prediction_path"])) == case["predictions_sha256"], "Convergence panel refers to another saved prediction")
            independent.compare(source["model_provenance"], stage_metrics[stage]["checkpoint_provenance"], "Convergence panel model provenance")
            frozen = next(c for c in frozen_figure_cases(protocol) if (c["global_id"], c["query_id"]) == (row["global_id"], row["query_id"]))
            with np.load(path(source["prediction_path"]), allow_pickle=False) as z:
                indices = np.flatnonzero(z["candidate_indices"] == frozen["candidate_index"]) if frozen["candidate_index"] is not None else []
                need(len(indices) == (0 if frozen["candidate_index"] is None else 1), "Convergence panel dropped/duplicated its frozen query")
                query_index = int(indices[0]) if len(indices) else None
                need(source["query_array_index"] == query_index, "Convergence panel query-array index differs")
                scores = z["event_scores"][query_index].tolist() if query_index is not None else None
                need(source["event_scores_by_radius"] == scores
                     and source["reference_events_by_radius"] == (frozen["reference_events"] if query_index is not None else None),
                     "Convergence panel event-score/reference annotations differ")
                if expected_k is None:
                    need("worlds" not in z.files and "world_events" not in z.files and "continuous_score" not in z.files
                         and source["world_order_sha256"] is None, "Direct-query convergence panel fabricated map/world outputs")
                    expected_events = None
                else:
                    worlds = z["worlds"]
                    world_hashes = [hashlib.sha256(np.ascontiguousarray(world, dtype=np.uint8).tobytes()).hexdigest() for world in worlds]
                    need(world_hashes == source["world_order_sha256"] and len(worlds) == expected_k, "Convergence panel reordered/omitted saved worlds")
                    expected_events = z["world_events"][:, query_index].tolist() if query_index is not None else None
                need(source["events_by_world_and_radius"] == expected_events, "Convergence panel reference event annotations differ")
    for figure in manifest["figures"]:
        need(len(figure["files"]) == 3 and {Path(f["path"]).suffix for f in figure["files"]} == {".svg", ".png", ".pdf"},
             "Missing convergence figure export")
        for artifact in figure["files"]:
            target = (file.parent / artifact["path"]).resolve()
            need(target.is_relative_to(file.parent.resolve()) and sha256(target) == artifact["sha256"], "Convergence image export hash/path differs")


def check_plan(plan, protocol, protocol_hash, method, first_seed_training, samples_by_split):
    need(plan["method"] == method and plan["protocol_sha256"] == protocol_hash
         and plan["seeds"] == SEEDS and plan["data_seal_sha256"] == protocol["data_seal_sha256"],
         "Full plan identity mismatch")
    need(plan["members_per_outer_seed"] == protocol["methods"][method]["members"], "Wrong full member matrix")
    end = plan["full_steps"]
    need(type(end) is int and end in protocol["staged"]["allowed_full_steps"], "Full budget outside frozen choices")
    checkpoints = sorted({protocol["staged"]["smoke_steps"], protocol["staged"]["pilot_steps"], end,
                          *range(protocol["staged"]["pilot_steps"] + 2500, end + 1, 2500)})
    need(plan["checkpoint_steps"] == checkpoints, "Unregistered checkpoint schedule")
    numeric, _ = bound(plan, "pilot_gate_receipt", "pilot_gate_sha256")
    visual, _ = bound(plan, "visual_review_receipt", "visual_review_sha256")
    for review in (numeric, visual):
        need(review["protocol_sha256"] == protocol_hash and review["method"] == method and review["seed"] == SEEDS[0],
             "Pilot review identities do not match")
    stage_metrics, independent_scores = {}, {}
    for name, expected_step in (("untrained", 0), ("smoke", protocol["staged"]["smoke_steps"]), ("pilot", protocol["staged"]["pilot_steps"])):
        reference = numeric["stage_metrics"][name]
        file = path(reference["path"])
        need(sha256(file) == reference["sha256"], "Pilot source metrics changed")
        values, _, _ = check_evaluation(file.parent, method, SEEDS[0], expected_step, protocol, protocol_hash,
                                        ["calibration", "validation"])
        need(values["method"] == method and values["outer_seed"] == SEEDS[0]
             and values["protocol_sha256"] == protocol_hash and values["stage"] == name,
             "Pilot stage identity differs")
        need([m["completed_steps"] for m in values["checkpoint_provenance"]] == [expected_step] * protocol["methods"][method]["members"],
             "Pilot stage training step/member count differs")
        need(values["observed_and_support_constraints_passed"], "Pilot projection gate failed")
        stage_metrics[name] = values
        independent_scores[name] = {split: candidate_scores(file.parent, split, samples_by_split[split], values)
                                    for split in ("calibration", "validation")}
    checks, losses = {}, []
    loss_key = "reconstruction" if method == "lama" else "velocity_mse" if method == "flow" else "event_loss" if method == "direct_query" else "map_nll"
    for member, (rows, lines) in enumerate(first_seed_training):
        size = protocol["staged"]["pilot_steps"]
        values = np.array([r["losses"][loss_key] for r in rows[:size]], float)
        before, after = float(values[:100].mean()), float(values[-100:].mean())
        gain = (before - after) / max(abs(before), 1e-12)
        checks[f"member{member}_finite_loss"] = bool(np.isfinite(values).all())
        checks[f"member{member}_unknown_loss_improved"] = bool(gain >= protocol["staged"]["automated_gate"]["unknown_loss_relative_reduction_first100_to_last100_min"])
        losses.append(dict(member=member, loss=loss_key, first100_mean=before, last100_mean=after,
                           relative_reduction=gain, pilot_progress_prefix_sha256=hashlib.sha256(b"".join(lines[:size])).hexdigest()))
    for split in ("calibration", "validation"):
        before, after = independent_scores["untrained"][split], independent_scores["pilot"][split]
        for metric in (("event_brier",) if method == "direct_query" else ("map_brier", "event_brier")):
            threshold = f"{split}_" + ("hidden_map" if metric == "map_brier" else "event") + "_brier_improvement_vs_untrained_min"
            checks[f"{split}_{metric}_improved"] = bool(before[metric] - after[metric] >= protocol["staged"]["automated_gate"][threshold])
        checks[f"{split}_evidence_and_support_preserved"] = bool(stage_metrics["pilot"]["observed_and_support_constraints_passed"])
    independent.compare(numeric["checks"], checks, "independently recomputed pilot gate")
    independent.compare(numeric["losses"], losses, "pilot unknown-loss convergence")
    before, after = independent_scores["smoke"]["calibration"], independent_scores["pilot"]["calibration"]
    gains = {key: (before[key] - after[key]) / max(abs(before[key]), 1e-12)
             for key in ("map_brier", "event_brier") if before[key] is not None}
    expected_end = 10000 if all(value < .02 for value in gains.values()) else 20000
    independent.compare(numeric["calibration_relative_improvements_1000_to_5000"], gains, "Pilot schedule gains")
    need(end == expected_end == numeric["proposed_full_steps"], "Full budget was not selected by the frozen staged rule")
    need(all(checks.values()) and numeric["passed"] and plan["pilot_gate_passed"], "Pilot numerical gate did not actually pass")
    check_pilot_visual(visual, numeric, protocol, stage_metrics)
    return dict(passed=True, full_steps=end, numerical_checks=checks, staged_relative_gains=gains,
                independently_recomputed_stage_scores=independent_scores,
                visual_review_sha256=plan["visual_review_sha256"], pilot_gate_sha256=plan["pilot_gate_sha256"])


def check_training_reconciliation(stored, records):
    """Bind controller cost metadata to independent sessions and immutable hashes."""
    need(stored["passed"] is True and stored["errors"] == [], "Controller training audit failed")
    need(len(stored["members"]) == len(records), "Training reconciliation member matrix differs")
    for saved, actual in zip(stored["members"], records):
        independent.compare(saved, dict(member=actual["member"], updates=actual["completed_steps"],
            all_update_losses_finite=True, progress_sha256=actual["progress_sha256"],
            time_summary_sha256=actual["time_summary_sha256"], actual_training_time=actual["actual_training_cost"],
            peak_allocated_bytes=actual["peak_allocated_bytes"], peak_reserved_bytes=actual["peak_reserved_bytes"]),
            "Controller full-training evidence")
        receipt, _ = bound(saved, "time_reconciliation", "time_reconciliation_sha256")
        independent.compare(receipt["totals"], actual["actual_training_cost"], "Reconciliation actual totals")
        need(path(saved["time_summary"]) == path(receipt["current_time_summary"])
             and sha256(path(receipt["current_time_summary"])) == receipt["current_time_summary_sha256"] == actual["time_summary_sha256"],
             "Reconciliation time summary changed")
        evidence = receipt["session_files"]
        need(len(evidence) == len(actual["session_files"]) and
             {str(path(r["path"])): r["sha256"] for r in evidence} ==
             {str(path(r["path"])): r["sha256"] for r in actual["session_files"]}, "Reconciliation omitted/changed session evidence")
        for item in evidence + receipt["previous_summary_archives"]:
            need(sha256(path(item["path"])) == item["sha256"], "Session/reconciliation archive hash changed")
    independent.compare(stored["training_seconds_sum_across_members"],
        sum(r["actual_training_cost"]["training_wall_seconds"] for r in records), "Full member training cost sum")
    need(stored["time_is_lower_bound"] == any(r["actual_training_cost"]["wall_time_is_lower_bound"] for r in records),
         "Full member time lower-bound flag differs")


def check_validation_training_cost(members, records):
    need(len(members) == len(records), "Validation training cost member count differs")
    for member, actual in zip(members, records):
        independent.compare(member, dict(member_index=actual["member"], training_time_summary_sha256=actual["time_summary_sha256"],
            training_seconds=actual["actual_training_cost"]["training_wall_seconds"],
            training_wall_time_is_lower_bound=actual["actual_training_cost"]["wall_time_is_lower_bound"],
            sessions_without_final_receipt=actual["actual_training_cost"]["sessions_without_final_receipt"]), "Validation captured full actual training cost")
        need(sha256(path(member["training_time_summary_path"])) == actual["time_summary_sha256"], "Validation training summary path/hash differs")
        expected = [r for r in actual["session_files"] if r["path"].endswith((".finished.json", ".heartbeat.json"))]
        resources = member["training_resource_receipts"]
        need(len(resources) == len(expected) and
             {str(path(r["path"])): r["sha256"] for r in resources} ==
             {str(path(r["path"])): r["sha256"] for r in expected}, "Validation omitted actual training resource receipts")
        snapshot = member["checkpoint_snapshot_cost"]
        for key in ("training_wall_seconds", "optimizer_update_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            need(finite(snapshot[key], nonnegative=True), "Invalid selected checkpoint resource snapshot")
        captured = datetime.fromisoformat(member["training_cost_captured_utc"])
        finished = []
        for resource in resources:
            need(sha256(path(resource["path"])) == resource["sha256"], "Validation resource digest differs")
            if resource["path"].endswith(".finished.json"):
                stamp = read(path(resource["path"]))["finished_utc"]
                need(datetime.fromisoformat(stamp) <= captured, "Validation cost cutoff predates actual finished training")
                finished.append(stamp)
        need(member["latest_finished_training_session_utc"] == (max(finished) if finished else None), "Validation latest training finish differs")
        for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
            peaks = [snapshot[metric]] + [read(path(r["path"]))[metric] for r in resources]
            need(all(finite(v, nonnegative=True) for v in peaks), "Invalid training resource peak")
            independent.compare(member["training_" + metric], max(peaks), "Validation full run peak memory")


def check_convergence(selection, candidates, pointer_file, method, seed, protocol, protocol_hash, plan_hash, training_records):
    pointer = read(pointer_file)
    target = path(pointer["receipt"])
    need(sha256(target) == pointer["sha256"], "Convergence receipt digest differs")
    receipt = read(target)
    check_training_reconciliation(receipt["training_audit"], training_records)
    tail = sorted(candidates, key=lambda c: c["step"])[-3:]
    need(len(tail) == 3, "Fewer than three final calibration checkpoints")
    numerical = True
    trends = {}
    for key in ("event_brier", "map_brier"):
        if method == "direct_query" and key == "map_brier":
            continue
        values = [c[key] for c in tail]
        need(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values), "Missing final convergence metric")
        changes = [(values[0] - values[1]) / max(abs(values[0]), 1e-12),
                   (values[1] - values[2]) / max(abs(values[1]), 1e-12),
                   (values[0] - values[2]) / max(abs(values[0]), 1e-12)]
        stable = all(abs(change) < .02 for change in changes)
        numerical &= stable
        trends[key] = dict(values=values, relative_reductions=changes, stable_below_two_percent=stable)
    independent.compare(receipt["trends"], trends, "recomputed convergence trend")
    need(receipt["protocol_sha256"] == protocol_hash and receipt["full_plan_sha256"] == plan_hash
         and receipt["method"] == method and receipt["seed"] == seed, "Convergence identity differs")
    need(receipt["selected_step"] == selection["best_step"] and receipt["full_steps"] == candidates[-1]["step"]
         and receipt["last_three_calibration_steps"] == [c["step"] for c in tail]
         and receipt["final_test_locked"] is True and receipt["main_table_eligible"] is False, "Convergence budget/selection/test scope differs")
    need(receipt["last_three_calibration_metrics_sha256"] == [c["metrics_sha256"] for c in tail], "Convergence not bound to final calibration candidates")
    independent.compare(receipt["collapse_diagnostics"], [c["collapse_diagnostic"] for c in tail], "Final reference collapse diagnostics")
    visual_passed = False
    if receipt.get("visual_review"):
        visual, _ = bound(receipt, "visual_review", "visual_review_sha256")
        need(visual["protocol_sha256"] == protocol_hash and visual["full_plan_sha256"] == plan_hash
             and visual["method"] == method and visual["seed"] == seed
             and visual["selected_calibration_metrics_sha256"] == selection["selected_metrics_sha256"]
             and visual["last_three_calibration_metrics_sha256"] == [c["metrics_sha256"] for c in tail],
             "Final visual review refers to different selected/final maps")
        visual_passed = visual.get("passed") is True and visual.get("no_collapse") is True
    short = selection["best_step"] < min(protocol["staged"]["allowed_full_steps"])
    need(selection["short_selected_checkpoint"] == short and receipt["short_selected_checkpoint"] == short, "Short selected checkpoint flag is wrong")
    expected = numerical and visual_passed and not short
    need(receipt["numerical_convergence_passed"] == numerical, "Reported convergence flag disagrees with numerical trend")
    need(receipt["sufficient_training"] == expected, "Reported sufficient-training flag is unsupported")
    return dict(passed=expected, numerical_convergence_passed=numerical, visual_review_passed=visual_passed,
                selected_checkpoint_is_short=short, convergence_receipt_sha256=sha256(target),
                visual_review_sha256=receipt.get("visual_review_sha256"), trends=trends,
                note="Visual judgment remains a referenced human assessment; independently recomputed collapse statistics accompany the audit.")


def rescore_selected(cal_dir, val_dir, cal_metrics, val_metrics, protocol, protocol_hash, explicit_ids):
    flat_cal, results_cal, receipt_cal = independent.score_split(cal_dir, "calibration", protocol, cal_metrics, explicit_ids)
    flat_val, results_val, receipt_val = independent.score_split(val_dir, "validation", protocol, val_metrics, explicit_ids)
    cal_fit, val_fit = read(cal_dir / "calibration_platt.json"), read(val_dir / "calibration_platt.json")
    need(sha256(cal_dir / "calibration_platt.json") == sha256(val_dir / "calibration_platt.json"), "Validation refitted/changed the selected calibration mapping")
    identity = [{key: m.get(key) for key in ("member_index", "initialization_seed", "checkpoint_sha256", "completed_steps")}
                for m in cal_metrics["checkpoint_provenance"]]
    val_identity = [{key: m.get(key) for key in ("member_index", "initialization_seed", "checkpoint_sha256", "completed_steps")}
                    for m in val_metrics["checkpoint_provenance"]]
    need(identity == val_identity, "Selected calibration and validation used different checkpoints")
    platt = independent.verify_platt(cal_fit, flat_cal, identity, protocol_hash)
    for split, flat, computed, stored in (("calibration", flat_cal, results_cal, cal_metrics),
                                          ("validation", flat_val, results_val, val_metrics)):
        clipped = np.clip(flat["p"], 1e-6, 1 - 1e-6)
        calibrated = expit(cal_fit["a"] * (np.log(clipped) - np.log1p(-clipped)) + cal_fit["b"])
        computed["event_calibrated_diagnostic"] = independent.independent_event_report(calibrated, flat["y"], flat["parents"], flat["radii"])
        independent.compare(stored["splits"][split]["event_calibrated_diagnostic"], computed["event_calibrated_diagnostic"], split + " calibrated")
        for source, detail in computed["by_source"].items():
            mask = flat["sources"] == source
            detail["event_calibrated_diagnostic"] = independent.independent_event_report(
                calibrated[mask], flat["y"][mask], flat["parents"][mask], flat["radii"][mask]) if mask.any() else None
            independent.compare(stored["splits"][split]["by_source"][source]["event_calibrated_diagnostic"],
                                detail["event_calibrated_diagnostic"], split + " calibrated source " + source)
        independent.verify_source_macro(stored["splits"][split], computed)
    return dict(calibration=results_cal, validation=results_val), dict(calibration=receipt_cal, validation=receipt_val, platt=platt)


def audit(method, protocol_file, attempt):
    protocol, protocol_hash = read(protocol_file), sha256(protocol_file)
    need(protocol["seeds"] == SEEDS and method in protocol["methods"], "Wrong three-seed/method contract")
    need(protocol["sampling"]["K"] == 4 and protocol["query"]["radii_cells"] == [0, 10, 20], "Changed common K or cell radius")
    out, data = path(protocol["output_root"]), path(protocol["data_root"])
    plan_path = out / "full_plans" / f"{method}.json"
    plan, plan_hash = read(plan_path), sha256(plan_path)
    need(protocol["test_lock"]["final_test_locked"] and protocol["test_lock"]["location_6_locked"], "Locked-test policy absent")
    need(sha256(data / "seal.json") == protocol["data_seal_sha256"], "Data seal differs")
    for relative, digest in read(data / "seal.json")["files"].items():
        need(sha256(data / relative) == digest, "Frozen data file changed")
    for relative, digest in protocol["source_hashes"].items():
        need(sha256(path(relative)) == digest, "Frozen training/evaluator source changed")
    helper = "scripts/evaluate_flatlands_support_clamped.py"
    frozen_helper = subprocess.check_output(["git", "show", protocol["git_commit"] + ":" + helper], cwd=ROOT)
    need(hashlib.sha256(frozen_helper).hexdigest() == sha256(path(helper)), "Historical independent geometry helper differs from frozen Git version")
    for key, digest_key in (("cases", "sha256"), ("geometry_verification", "geometry_verification_sha256")):
        need(sha256(path(protocol["figures"][key])) == protocol["figures"][digest_key], "Frozen figure scope/geometry changed")
    need(read(path(protocol["figures"]["geometry_verification"]))["passed"], "Figure geometry was not verified")
    pipeline = out / "full_runs" / method
    identity = read(pipeline / "run.json")
    need(identity["protocol_sha256"] == protocol_hash and identity["full_plan_sha256"] == plan_hash
         and identity["seeds"] == SEEDS and identity["members"] == protocol["methods"][method]["members"]
         and identity["method"] == method and identity["full_steps"] == plan["full_steps"],
         "Full pipeline identity differs")
    controller = ROOT / "scripts/run_flatlands_formal_full.py"
    need(sha256(controller) == identity["controller_sha256"], "Controller differs from its run provenance")
    need(all((pipeline / "seeds" / str(seed) / "complete.json").exists() for seed in SEEDS),
         "All three completed repeats are required before final auditing")
    parent_sets = {split: {r["parent_group"] for r in protocol["scenes"][split]} for split in ("train", "calibration", "validation")}
    need(all(not (parent_sets[a] & parent_sets[b]) for a, b in (("train", "calibration"), ("train", "validation"), ("calibration", "validation"))),
         "Frozen train/calibration/validation parent partitions overlap")
    need(all(row["packet_directory"] == "train/" + row["global_id"] for rows in protocol["scenes"].values() for row in rows),
         "Frozen manifest contains a physical test packet")
    samples_by_split = {split: load_formal_data(data, split) for split in ("calibration", "validation")}
    fixed = frozen_figure_cases(protocol)
    explicit_ids = {c["global_id"] for c in fixed}
    for split in ("calibration", "validation"):
        ids = sorted((r["global_id"] for r in protocol["scenes"][split]),
                     key=lambda gid: hashlib.sha256(f"final-audit-v1|20260910|{split}|{gid}".encode()).hexdigest())
        explicit_ids.update(ids[:5])
    all_training, first_seed_training = {}, []
    for seed in SEEDS:
        all_training[str(seed)] = []
        for member in range(protocol["methods"][method]["members"]):
            folder = out / "runs" / method / str(seed) / f"member_{member}"
            record, rows, lines = check_training(folder, method, seed, member, plan["full_steps"], protocol_hash, plan_hash, protocol)
            all_training[str(seed)].append(record)
            if seed == SEEDS[0]:
                first_seed_training.append((rows, lines))
    plan_review = check_plan(plan, protocol, protocol_hash, method, first_seed_training, samples_by_split)
    seeds, qualified, evaluation_hashes = [], [], []
    for seed in SEEDS:
        folder = pipeline / "seeds" / str(seed)
        selection, complete = read(folder / "selection.json"), read(folder / "complete.json")
        need(complete["selection_sha256"] == sha256(folder / "selection.json")
             and complete["protocol_sha256"] == selection["protocol_sha256"] == protocol_hash
             and complete["full_plan_sha256"] == selection["full_plan_sha256"] == plan_hash,
             "Seed completion/selection digest mismatch")
        independent.compare(selection, dict(method=method, seed=seed, final_test_locked=True, main_table_eligible=False), "Selected repeat identity")
        independent.compare(complete, dict(seed=seed, full_steps_completed=plan["full_steps"], main_table_eligible=False), "Completed repeat identity")
        candidates = selection["candidates"]
        need(selection["candidate_steps"] == plan["checkpoint_steps"] == [c["step"] for c in candidates], "Candidate checkpoint matrix incomplete")
        candidate_checks, eval_by_step, rescored_candidates = [], {}, []
        for candidate in candidates:
            evaluation = path(candidate["evaluation"])
            need(sha256(evaluation / "metrics.json") == candidate["metrics_sha256"], "Candidate metrics checksum differs")
            splits = ["calibration", "validation"] if seed == SEEDS[0] and candidate["step"] in (1000, 5000) else ["calibration"]
            metrics, fitted, _ = check_evaluation(evaluation, method, seed, candidate["step"], protocol, protocol_hash, splits)
            independent.compare(candidate["checkpoint_provenance"], metrics["checkpoint_provenance"], "candidate member provenance")
            recalculated = candidate_scores(evaluation, "calibration", samples_by_split["calibration"], metrics)
            independent.compare(candidate, recalculated, "Candidate calibration score and collapse receipt")
            rescored_candidates.append({**candidate, **recalculated})
            need(candidate["calibration_sha256"] == sha256(evaluation / "calibration_platt.json"), "Candidate calibration file differs")
            need(path(candidate["calibration_file"]) == evaluation / "calibration_platt.json"
                 and candidate["observed_and_support_constraints_passed"] is True, "Candidate calibration path/constraint receipt differs")
            for index, member in enumerate(metrics["checkpoint_provenance"]):
                canonical = out / "runs" / method / str(seed) / f"member_{index}"
                need(path(member["run_receipt"]) == canonical / "run.json"
                     and path(member["checkpoint"]) == canonical / f"step_{candidate['step']:08d}.pt",
                     "Candidate checkpoint escaped its audited canonical member run")
                candidate_checks.append(check_checkpoint(member, method, seed, index, candidate["step"], protocol, protocol_hash, plan_hash,
                    full_steps=plan["full_steps"], expected_recovery=all_training[str(seed)][index]["expected_recovery_state_by_step"][str(candidate["step"])],
                    deep=candidate["step"] in {selection["best_step"], plan["full_steps"]}))
            eval_by_step[candidate["step"]] = (evaluation, metrics)
        winner = choose_checkpoint(rescored_candidates)
        need(selection["best_step"] == complete["selected_step"] == winner["step"]
             and selection["selected_metrics_sha256"] == winner["metrics_sha256"],
             "Checkpoint selection did not follow the independently recomputed global tie rule")
        need(complete["short_selected_checkpoint"] == selection["short_selected_checkpoint"], "Completion short-checkpoint flag differs")
        cal_dir, cal_metrics = eval_by_step[winner["step"]]
        need(selection["calibration_sha256"] == winner["calibration_sha256"]
             and path(selection["calibration_file"]) == cal_dir / "calibration_platt.json", "Selection did not retain the winning calibration mapping")
        need(len(selection["best_checkpoints"]) == protocol["methods"][method]["members"], "Best checkpoint member count differs")
        for best, member in zip(selection["best_checkpoints"], winner["checkpoint_provenance"]):
            need(sha256(path(best)) == member["checkpoint_sha256"], "best.pt differs from selected original checkpoint")
        val_dir = path(complete["selected_validation"])
        need(sha256(val_dir / "metrics.json") == complete["validation_metrics_sha256"], "Selected validation hash differs")
        val_metrics, _, _ = check_evaluation(val_dir, method, seed, winner["step"], protocol, protocol_hash, ["validation"])
        check_validation_training_cost(val_metrics["checkpoint_provenance"], all_training[str(seed)])
        completed_attempts = list((out / "evaluations" / method / str(seed)).glob("selected_validation_*/attempt_*/complete.json"))
        need(len(completed_attempts) == 1 and completed_attempts[0].parent == val_dir,
             "More than one successful selected-validation evaluation exists")
        computed, numerical_audit = rescore_selected(cal_dir, val_dir, cal_metrics, val_metrics, protocol, protocol_hash, explicit_ids)
        independent_file = attempt / f"seed_{seed}_independent_metrics.json"
        write_new(independent_file, computed)
        convergence = check_convergence(selection, rescored_candidates, folder / "convergence.json", method, seed, protocol, protocol_hash, plan_hash,
                                        all_training[str(seed)])
        if convergence["passed"]:
            qualified.append(seed)
        evaluation_hashes.append(complete["validation_metrics_sha256"])
        seeds.append(dict(seed=seed, selected_step=winner["step"], selection_sha256=sha256(folder / "selection.json"),
                          validation_metrics_sha256=complete["validation_metrics_sha256"], candidate_checkpoints=candidate_checks,
                          independent_metrics=str(independent_file.relative_to(ROOT)), independent_metrics_sha256=sha256(independent_file),
                          independent_evaluation=numerical_audit, convergence=convergence, full_training=all_training[str(seed)]))
    return dict(passed=qualified == SEEDS, test_lock_passed=True, checkpoint_selection_passed=True,
                implementation_checks_passed=True, convergence_review_passed=qualified == SEEDS,
                protocol_sha256=protocol_hash, fully_trained_outer_seeds=qualified,
                full_budget_completed_outer_seeds=SEEDS, evaluation_metrics_sha256=evaluation_hashes,
                full_plan_sha256=plan_hash, method=method, seeds=seeds, full_plan_review=plan_review,
                main_table_eligible=qualified == SEEDS,
                claim_scope="Unified validation-only comparison; no final-test or external-paper leaderboard claim",
                new_raw_archive_or_test_images_opened=0, gpu_inference_performed=False,
                historical_geometry_helper_frozen_git_verified=True,
                cached_input_audit_scope="Binary/disjoint channels, hidden/support/known reference consistency, query eligibility; original archive generation was not rerun",
                training_seconds_sum_across_seeds_and_members=sum(r["actual_training_cost"]["training_wall_seconds"] for runs in all_training.values() for r in runs),
                training_time_is_lower_bound=any(r["actual_training_cost"]["wall_time_is_lower_bound"] for runs in all_training.values() for r in runs))


def publish_final_receipt(report, attempt, out):
    """A diagnostic or incomplete repeat matrix never creates a final receipt."""
    required = ("passed", "test_lock_passed", "checkpoint_selection_passed", "implementation_checks_passed", "convergence_review_passed")
    need(all(report.get(key) is True for key in required)
         and report.get("fully_trained_outer_seeds") == SEEDS
         and len(report.get("evaluation_metrics_sha256", [])) == 3
         and len(set(report["evaluation_metrics_sha256"])) == 3,
         "A final receipt requires independently verified convergence and all three repeats")
    verification = attempt / "verification.json"
    need(read(verification) == report, "Publication differs from the actual audit verification")
    publication = {**report, "audit_attempt": str(attempt.relative_to(ROOT)),
                   "verification": str(verification.relative_to(ROOT)), "verification_sha256": sha256(verification)}
    write_new(out / "final_audits" / f"{report['method']}.json", publication)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/flatlands_external_formal_protocol_v1/protocol.json")
    parser.add_argument("--output-dir", type=Path, help="New audit attempt directory; existing paths are rejected")
    args = parser.parse_args()
    args.protocol = path(args.protocol)
    protocol = read(args.protocol)
    need(args.method in protocol["methods"], "Unregistered audit method")
    out = path(protocol["output_root"])
    if args.output_dir:
        attempt = args.output_dir.resolve()
        need(attempt.is_relative_to(ROOT), "Audit output escapes project")
    else:
        parent = out / "audit_attempts" / "final" / args.method
        number = 1
        while (parent / f"attempt_{number}").exists():
            number += 1
        attempt = parent / f"attempt_{number}"
    attempt.mkdir(parents=True, exist_ok=False)
    source_names = ["scripts/audit_flatlands_formal_final.py", "scripts/audit_flatlands_formal_stage.py",
                    "scripts/evaluate_flatlands_support_clamped.py", "src/pathrel/formal_data.py",
                    "src/pathrel/parent_pilot_data.py", "src/pathrel/data_access.py"]
    started = time.perf_counter()
    report = dict(passed=False, method=args.method, protocol_sha256=sha256(args.protocol),
                  test_lock_passed=False, checkpoint_selection_passed=False, implementation_checks_passed=False,
                  convergence_review_passed=False, fully_trained_outer_seeds=[], evaluation_metrics_sha256=[],
                  full_plan_sha256=None, new_raw_archive_or_test_images_opened=0, gpu_inference_performed=False)
    try:
        report.update(audit(args.method, args.protocol, attempt))
    except BaseException as error:
        report["error"] = repr(error)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                  auditor_sources={name: sha256(ROOT / name) for name in source_names},
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  independent_check_counts=dict(independent.CHECKS),
                  maximum_metric_difference=independent.MAX_DIFFERENCE)
    import scipy
    report["audit_environment"] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)
    verification = attempt / "verification.json"
    write_new(verification, report)
    if report["passed"]:
        publish_final_receipt(report, attempt, out)
    print(json.dumps({key: value for key, value in report.items() if key not in ("seeds", "auditor_sources")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
