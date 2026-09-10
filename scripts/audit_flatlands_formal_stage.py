#!/usr/bin/env python3
"""Independent CPU audit of a complete, validation-only formal stage.

No model construction, inference, raw archive or test loader is used. Arithmetic
is implemented independently of formal_metrics.py. Every saved world uses the
historical exact oracle; fixed figures and an input-ranked subset additionally
use explicit integer-disk erosion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pathrel.formal_data import load_formal_data, sha256
from scripts.evaluate_flatlands_support_clamped import _accelerated_events

RADII, EPSILON = (0, 10, 20), 1e-6
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
CHECKS = Counter()
MAX_DIFFERENCE = 0.0


def require(condition, message):
    if not condition:
        raise ValueError(message)
    CHECKS["assertions"] += 1


def compare(actual, expected, label, tolerance=2e-10):
    """Expected dictionaries select the independently audited fields."""
    global MAX_DIFFERENCE
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ": expected mapping")
        for key, value in expected.items():
            require(key in actual, label + ": missing " + key)
            compare(actual[key], value, label + "/" + key, tolerance)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), label + ": length mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, label + f"/{i}", tolerance)
    elif expected is None or isinstance(expected, (str, bool)):
        require(actual == expected, f"{label}: {actual!r} != {expected!r}")
    elif isinstance(expected, (int, np.integer)):
        require(actual == expected, f"{label}: integer identity {actual!r} != {expected!r}")
        CHECKS["numeric_values"] += 1
    elif isinstance(expected, (int, float, np.number)):
        require(actual is not None and np.isfinite(actual), label + ": missing/nonfinite number")
        difference = abs(float(actual) - float(expected))
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, difference)
        require(difference <= tolerance * max(1., abs(float(expected))),
                f"{label}: {actual!r} != {expected!r}")
        CHECKS["numeric_values"] += 1
    else:
        require(actual == expected, label + ": value mismatch")


def parent_weights(parents):
    counts = Counter(map(str, parents))
    return np.array([1. / (len(counts) * counts[str(p)]) for p in parents])


def independent_event_summary(p, y, parents, weighting):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if not len(p):
        return dict(events=0, scenes=0, event_brier=None, event_nll=None, event_ece=None,
                    coverage_at_confidence=None, false_safe_at_confidence=None,
                    false_safe_mass=None, reachable_prevalence=None, accepted_events=0,
                    reliability=[], false_safe_coverage_curve=[])
    w = parent_weights(parents) if weighting == "scene_weighted" else np.ones(len(p)) / len(p)
    w /= w.sum()
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    accepted = p >= .8
    coverage, false_mass = float(w[accepted].sum()), float(w[accepted & ~y].sum())
    reliability, ece = [], 0.
    for index in range(10):
        mask = (p >= index / 10) & ((p < (index + 1) / 10) if index < 9 else (p <= 1))
        mass = float(w[mask].sum())
        predicted = float(np.dot(w[mask], p[mask]) / mass) if mass else None
        observed = float(np.dot(w[mask], y[mask]) / mass) if mass else None
        if mass:
            ece += mass * abs(predicted - observed)
        reliability.append(dict(lower=index / 10, upper=(index + 1) / 10, upper_inclusive=index == 9,
                                events=int(mask.sum()), weight=mass, predicted=predicted, observed=observed))
    curve = [dict(threshold=None, acceptance="none", coverage=0., false_safe=None)]
    for threshold in sorted(set(p.tolist()), reverse=True):
        mask = p >= threshold
        mass = float(w[mask].sum())
        curve.append(dict(threshold=threshold, acceptance="score >= threshold", coverage=min(1., mass),
                          false_safe=float(w[mask & ~y].sum() / mass)))
    return dict(events=len(p), scenes=len(set(parents)), event_brier=float(np.dot(w, (p - y) ** 2)),
                event_nll=float(np.dot(w, -np.log(np.where(y, clipped, 1 - clipped)))), event_ece=float(ece),
                coverage_at_confidence=coverage, false_safe_at_confidence=false_mass / coverage if coverage else None,
                false_safe_mass=false_mass, reachable_prevalence=float(np.dot(w, y)), accepted_events=int(accepted.sum()),
                reliability=reliability, false_safe_coverage_curve=curve)


def independent_event_report(p, y, parents, radii):
    p, y, parents, radii = np.asarray(p, float), np.asarray(y, bool), np.asarray(parents, str), np.asarray(radii, int)
    result = dict(events=len(p), scenes=len(set(parents)), ece_bins=10, confidence=.8,
                  nll_clip_epsilon=EPSILON, radius_unit="grid cell")
    for weighting in ("pooled", "scene_weighted"):
        def subset(mask):
            return independent_event_summary(p[mask], y[mask], parents[mask], weighting)
        result[weighting] = {
            "overall": subset(np.ones(len(p), bool)),
            "by_truth": {"reachable": subset(y), "unreachable": subset(~y)},
            "by_radius": {str(r): subset(radii == r) for r in sorted(set(radii))},
            "by_radius_and_truth": {str(r): {"reachable": subset((radii == r) & y),
                                             "unreachable": subset((radii == r) & ~y)} for r in sorted(set(radii))}}
    return result


def explicit_disk_events(worlds, starts, goals):
    output = np.zeros((len(worlds), len(starts), 3), bool)
    for k, world in enumerate(worlds):
        for index, radius in enumerate(RADII):
            dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
            safe = ndimage.binary_erosion(world, structure=dx * dx + dy * dy <= radius * radius, border_value=0)
            components, _ = ndimage.label(safe, structure=FOUR)
            a, b = components[starts[:, 0], starts[:, 1]], components[goals[:, 0], goals[:, 1]]
            output[k, :, index] = (a != 0) & (a == b)
    return output


def independent_map_case(sample, worlds, native):
    hidden, valid, known = sample.hidden, sample.valid, sample.valid & ~sample.hidden
    require(worlds.ndim == 3 and worlds.shape[1:] == sample.target.shape, "map dimensions differ")
    require(np.isfinite(worlds).all() and np.isin(worlds, (0, 1)).all(), "nonbinary world")
    worlds = worlds.astype(bool)
    obs_errors = int(np.sum(worlds[:, known] != sample.observation[0, known]))
    support_errors = int(worlds[:, ~valid].sum())
    require(obs_errors == 0 and support_errors == 0, "saved world violates evidence/support")
    p, y = worlds.mean(0)[hidden], sample.target[hidden]
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    result = dict(actual_world_count=len(worlds), hidden_cells=int(hidden.sum()),
        sample_vote_cell_brier=float(np.mean((p - y) ** 2)) if len(p) else None,
        sample_vote_cell_nll=float(np.mean(-np.log(np.where(y, clipped, 1 - clipped)))) if len(p) else None,
        observed_evidence_violation_count=obs_errors, valid_support_violation_count=support_errors,
        observed_cell_world_checks=int(known.sum()) * len(worlds), invalid_cell_world_checks=int((~valid).sum()) * len(worlds),
        hidden_world_free_fraction=worlds[:, hidden].mean(1).tolist() if hidden.any() else [],
        reference_hidden_free_fraction=float(y.mean()) if len(y) else None,
        valid_free_components_per_world=[int(ndimage.label(w & valid, FOUR)[1]) for w in worlds],
        reference_valid_free_components=int(ndimage.label(sample.target & valid, FOUR)[1]),
        distinct_world_count=len({np.packbits(w).tobytes() for w in worlds}))
    native = np.asarray(native, float)
    if native.ndim == 2:
        native = native[None]
    require(native.shape[1:] == sample.target.shape and len(native) in (1, len(worlds)), "native score shape mismatch")
    require(np.isfinite(native).all() and ((native >= 0) & (native <= 1)).all(), "invalid native score")
    require(not (native[:, known] != sample.observation[0, known]).any() and not native[:, ~valid].any(),
            "continuous score violates evidence/support")
    p = native.mean(0)[hidden]
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    result["continuous_completion_score"] = dict(
        hidden_brier=float(np.mean((p - y) ** 2)) if len(p) else None,
        hidden_nll=float(np.mean(-np.log(np.where(y, clipped, 1 - clipped)))) if len(p) else None,
        observed_evidence_violation_count=0, valid_support_violation_count=0)
    return result


def independent_map_report(cases, parents):
    counts, parents = np.array([c["hidden_cells"] for c in cases]), np.asarray(parents, str)
    active = counts > 0
    active_cases = [c for c, flag in zip(cases, active) if flag]
    result = dict(cases=len(cases), scenes=len(set(parents)), hidden_cells=int(counts.sum()),
        cases_with_hidden_cells=int(active.sum()), actual_world_counts=sorted({c["actual_world_count"] for c in cases}),
        observed_evidence_violation_count=sum(c["observed_evidence_violation_count"] for c in cases),
        valid_support_violation_count=sum(c["valid_support_violation_count"] for c in cases))
    native = dict(observed_evidence_violation_count=0, valid_support_violation_count=0)
    for weighting in ("pooled", "scene_weighted"):
        w = counts[active].astype(float) if weighting == "pooled" else parent_weights(parents[active])
        result[weighting] = {key: float(np.average([c[key] for c in active_cases], weights=w)) if len(w) else None
                             for key in ("sample_vote_cell_brier", "sample_vote_cell_nll")}
        native[weighting] = {key: float(np.average([c["continuous_completion_score"][key] for c in active_cases], weights=w)) if len(w) else None
                             for key in ("hidden_brier", "hidden_nll")}
    result["continuous_completion_score_diagnostic"] = native
    fractions = np.array([f for c in cases for f in c["hidden_world_free_fraction"]])
    reference = np.array([c["reference_hidden_free_fraction"] for c in cases if c["reference_hidden_free_fraction"] is not None])
    result["collapse_diagnostic"] = dict(
        generated_worlds_with_hidden=len(fractions),
        generated_nearly_all_blocked_fraction=float(np.mean(fractions <= .01)) if len(fractions) else None,
        generated_nearly_all_free_fraction=float(np.mean(fractions >= .99)) if len(fractions) else None,
        reference_nearly_all_blocked_case_fraction=float(np.mean(reference <= .01)) if len(reference) else None,
        reference_nearly_all_free_case_fraction=float(np.mean(reference >= .99)) if len(reference) else None,
        mean_valid_free_components_per_world=float(np.mean([v for c in cases for v in c["valid_free_components_per_world"]])),
        mean_reference_valid_free_components=float(np.mean([c["reference_valid_free_components"] for c in cases])))
    return result


def input_identity(seed, sample):
    key = dict(seed=seed, global_id=sample.row["global_id"], parent_group=sample.scene_key,
               starts=sample.starts.tolist(), goals=sample.goals.tolist(),
               candidate_indices=sample.candidate_indices.tolist(), radii_cells=RADII)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, int(digest[:16], 16) % (2**63 - 1)


def recipe_outer_seed(recipe):
    """Frozen trainer stores 'seed'; accept explicit legacy alias consistently."""
    value = recipe.get("outer_seed", recipe.get("seed"))
    require(isinstance(value, int) and not isinstance(value, bool), "missing integer training seed")
    if "seed" in recipe and "outer_seed" in recipe:
        require(recipe["seed"] == recipe["outer_seed"], "training seed aliases disagree")
    return value


def verify_provenance(evaluation, metrics, protocol, protocol_hash):
    complete = json.loads((evaluation / "complete.json").read_text())
    method, seed, stage = metrics["method"], metrics["outer_seed"], metrics["stage"]
    require(complete["passed"] and complete["metrics_sha256"] == sha256(evaluation / "metrics.json"), "completion hash mismatch")
    compare(complete, dict(method=method, seed=seed, stage=stage, final_test_locked=True, new_physical_test_images_opened=0), "completion")
    require(metrics["protocol_sha256"] == protocol_hash, "different evaluation protocol")
    require(seed in protocol["seeds"] and protocol["seeds"] == [20260831, 20260901, 20260902], "unregistered repeat")
    require(method in protocol["methods"] and metrics["main_table_eligible"] is False, "single stage cannot claim formal superiority")
    require(metrics["observed_and_support_constraints_passed"] and metrics["final_test_locked"]
            and metrics["location_6_locked"] and metrics["new_physical_test_images_opened"] == 0, "lock/constraint failure")
    before = json.loads((evaluation / "evaluation_frozen_before_scoring.json").read_text())
    compare(before, dict(method=method, seed=seed, stage=stage, protocol_sha256=protocol_hash,
                        checkpoint_provenance=metrics["checkpoint_provenance"], one_member_diagnostic=False), "pre-scoring receipt")
    members, seeds = metrics["checkpoint_provenance"], protocol["member_seeds"][method][str(seed)]
    require(len(members) == len(seeds) == (4 if method == "lama" else 1), "actual member count differs")
    require(len(set(seeds)) == len(seeds), "member seeds not independent")
    expected_step = dict(untrained=0, smoke=protocol["staged"]["smoke_steps"], pilot=protocol["staged"]["pilot_steps"]).get(stage)
    identities, checked = [], []
    for index, member in enumerate(members):
        compare(member, dict(member_index=index, initialization_seed=seeds[index]), "member identity")
        if expected_step is not None:
            require(member["completed_steps"] == expected_step, "checkpoint is at wrong stage step")
        identities.append({key: member.get(key) for key in ("member_index", "initialization_seed", "checkpoint_sha256", "completed_steps")})
        if stage == "untrained":
            require(member["checkpoint"] is None and member["completed_steps"] == 0, "untrained stage has trained weights")
            continue
        checkpoint = (ROOT / member["checkpoint"]).resolve()
        require(checkpoint.is_relative_to(ROOT) and sha256(checkpoint) == member["checkpoint_sha256"], "checkpoint hash/path mismatch")
        receipt = json.loads((ROOT / member["run_receipt"]).read_text())
        encoded = json.dumps(receipt["recipe"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        require(digest == receipt["recipe_sha256"] == member["recipe_sha256"], "checkpoint recipe mismatch")
        compare(receipt["recipe"], dict(method=method, member=index, protocol_sha256=protocol_hash), "checkpoint recipe")
        require(recipe_outer_seed(receipt["recipe"]) == seed, "checkpoint recipe outer-seed mismatch")
        import torch
        envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = envelope["state"]
        require(envelope["recipe_sha256"] == digest and envelope["progress_index"] == state["completed_steps"] == member["completed_steps"], "checkpoint progress mismatch")
        require(all(k in state for k in ("models", "optimizers", "schedulers", "sampler", "generators", "rng")), "incomplete resumable state")
        require(set(state["rng"]) >= {"python", "numpy", "torch_cpu", "torch_cuda"}, "missing RNG states")
        for scheduler in state["schedulers"].values():
            require(scheduler["completed_steps"] == state["completed_steps"], "scheduler progress mismatch")
        require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
        checked.append(dict(path=str(checkpoint.relative_to(ROOT)), sha256=member["checkpoint_sha256"], step=state["completed_steps"]))
        del envelope, state
        gc.collect()
    compare(complete["completed_steps"], [m["completed_steps"] for m in members], "completed steps")
    return identities, checked


def verify_platt(fitted, calibration, identity, protocol_hash):
    require(fitted["fit_split"] == "calibration" and fitted["protocol_sha256"] == protocol_hash
            and fitted["optimizer_success"] and fitted["guaranteed_calibration"] is False, "Platt scope mismatch")
    compare(fitted["prediction_model_identity"], identity, "calibration model identity")
    compare(fitted, dict(regularization=.01, clip_epsilon=EPSILON), "calibration fixed hyperparameters")
    p, y, parents, keys = (calibration[k] for k in ("p", "y", "parents", "keys"))
    payload = dict(scores=p.tolist(), targets=y.astype(int).tolist(), scene_ids=parents.tolist(), query_keys=keys)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == fitted["calibration_data_sha256"], "fit data are not exactly calibration predictions/labels")
    compare(fitted, dict(events=len(p), scenes=len(set(parents))), "calibration support")
    clipped, weights = np.clip(p, EPSILON, 1 - EPSILON), parent_weights(parents)
    x = np.log(clipped) - np.log1p(-clipped)
    def objective(ab):
        a, b = ab
        z = a * x + b
        return float(np.dot(weights, np.logaddexp(0, z) - y * z) + .005 * ((a - 1) ** 2 + b ** 2))
    fit = minimize(objective, [1., 0.], method="L-BFGS-B", bounds=((0., 20.), (-20., 20.)),
                   options=dict(ftol=1e-12, gtol=1e-8, maxiter=1000))
    require(fit.success and np.isfinite(fit.x).all(), "independent calibration optimization failed")
    compare(fitted["objective"], objective([fitted["a"], fitted["b"]]), "saved Platt objective", 1e-9)
    compare(objective(fit.x), fitted["objective"], "independent calibration-only optimum", 1e-8)
    require(abs(fit.x[0] - fitted["a"]) < 2e-4 and abs(fit.x[1] - fitted["b"]) < 2e-4, "independent Platt parameters disagree")
    return dict(calibration_input_hash_verified=True, validation_labels_used_for_refit=False,
                independent_a=float(fit.x[0]), independent_b=float(fit.x[1]), independent_objective=float(fit.fun))


def score_split(evaluation, split, protocol, metrics, explicit_ids):
    samples = load_formal_data(ROOT / protocol["data_root"], split)
    recorded = metrics["splits"][split]
    require([c["global_id"] for c in recorded["cases"]] == [s.row["global_id"] for s in samples], "case order mismatch")
    compare(json.loads((evaluation / split / "metrics.json").read_text()), recorded, split + " serialized metrics")
    prediction_dir = evaluation / split / "predictions"
    require({p.stem for p in prediction_dir.glob("*.npz")} == {s.row["global_id"] for s in samples}, "prediction coverage mismatch")
    maps, ps, ys, parents, radii, sources, keys = [], [], [], [], [], [], []
    method, seed = metrics["method"], metrics["outer_seed"]
    expected_k = None if method == "direct_query" else 1 if method == "deterministic" else 4
    allowed = {"global_id", "candidate_indices", "starts", "goals", "radii_cells", "event_scores"}
    if expected_k is not None:
        allowed |= {"worlds", "world_events", "continuous_score"}
    explicit_checked, world_count, event_checks = [], 0, 0
    monotonic = dict(raw_score_pair_violations=0, reference_pair_violations=0, world_event_pair_violations=0, cases=[])
    for position, (sample, row) in enumerate(zip(samples, recorded["cases"])):
        gid = sample.row["global_id"]
        path = prediction_dir / (gid + ".npz")
        require(sha256(path) == row["predictions_sha256"], "prediction digest mismatch: " + gid)
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == allowed, "unexpected/target fields in prediction artifact: " + gid)
            require(str(z["global_id"]) == gid, "prediction ID mismatch")
            for key, expected in (("candidate_indices", sample.candidate_indices), ("starts", sample.starts),
                                   ("goals", sample.goals), ("radii_cells", np.array(RADII))):
                require(np.array_equal(z[key], expected), "frozen query mismatch: " + gid + "/" + key)
            p = z["event_scores"].copy()
            require(p.shape == sample.targets.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid event scores")
            digest, sampling_seed = input_identity(seed, sample)
            compare(row, dict(input_key_sha256=digest, sampling_seed=sampling_seed, parent_group=sample.scene_key,
                              source=sample.row["source_dataset"], queries=len(sample.starts)), "input identity")
            target_events = _accelerated_events(sample.target[None], sample.starts, sample.goals, RADII)[0]
            require(np.array_equal(target_events, sample.targets), "independent reference oracle mismatch")
            CHECKS["reference_events"] += sample.targets.size
            case_monotonic = dict(global_id=gid, raw_score_pair_violations=int((np.diff(p, axis=-1) > 0).sum()),
                                  reference_pair_violations=int((np.diff(sample.targets.astype(int), axis=-1) > 0).sum()),
                                  world_event_pair_violations=0)
            require(case_monotonic["reference_pair_violations"] == 0, "reference radius monotonicity violation")
            if expected_k is not None:
                worlds = z["worlds"].copy()
                require(worlds.shape == (expected_k, 256, 256), "actual K differs from registered budget")
                map_case = independent_map_case(sample, worlds, z["continuous_score"])
                compare(row["map"], map_case, split + "/" + gid + "/map")
                maps.append(map_case)
                events = _accelerated_events(worlds, sample.starts, sample.goals, RADII)
                require(np.array_equal(events, z["world_events"]), "saved-world connectivity mismatch")
                require(np.array_equal(events.mean(0), p), "score is not actual world-event frequency")
                case_monotonic["world_event_pair_violations"] = int((np.diff(events.astype(int), axis=-1) > 0).sum())
                require(not case_monotonic["world_event_pair_violations"] and not case_monotonic["raw_score_pair_violations"],
                        "map-derived radius monotonicity violation")
                event_checks += events.size
                world_count += len(worlds)
                if gid in explicit_ids:
                    require(np.array_equal(explicit_disk_events(worlds, sample.starts, sample.goals), events), "explicit disk world oracle mismatch")
                    require(np.array_equal(explicit_disk_events(sample.target[None], sample.starts, sample.goals)[0], sample.targets), "explicit disk target oracle mismatch")
                    explicit_checked.append(gid)
                    CHECKS["explicit_disk_world_events"] += events.size
                    CHECKS["explicit_disk_reference_events"] += sample.targets.size
            else:
                require(row["map"] is None, "direct-query control contains map claims")
            for key in ("raw_score_pair_violations", "reference_pair_violations", "world_event_pair_violations"):
                monotonic[key] += case_monotonic[key]
            monotonic["cases"].append(case_monotonic)
            costs = row["efficiency"]
            compare(costs, dict(actual_world_count=expected_k, common_budget_K=4,
                    actual_members=4 if method == "lama" else 1, stage_one_member_diagnostic=False), "output cost identity")
            require(0 <= costs["generation_seconds"] <= costs["full_update_seconds"], "inconsistent timing scopes")
            if method == "lama":
                compare(costs, dict(actual_batch_forward_calls=4, actual_model_input_examples=4), "LaMa K4 cost")
            elif method == "flow":
                compare(costs, dict(velocity_evaluations_per_sample=50, actual_batch_forward_calls=50,
                                   actual_model_input_examples=400), "flow Heun25 CFG cost")
        for index, candidate in enumerate(sample.candidate_indices):
            for ri, radius in enumerate(RADII):
                ps.append(p[index, ri]); ys.append(sample.targets[index, ri]); parents.append(sample.scene_key)
                radii.append(radius); sources.append(sample.row["source_dataset"]); keys.append(f"{gid}/{int(candidate)}/{radius}")
        if (position + 1) % 50 == 0:
            print(json.dumps(dict(auditing_split=split, cases=position + 1, worlds_checked=world_count)), flush=True)
    csv_rows = list(csv.DictReader((evaluation / split / "predictions.csv").open()))
    require(len(csv_rows) == len(ps), "CSV event count mismatch")
    for index, row in enumerate(csv_rows):
        require(not ({"target", "label", "reference"} & set(row)), "prediction CSV includes labels")
        require(f"{row['global_id']}/{row['query_id']}/{row['radius_cells']}" == keys[index], "CSV event key mismatch")
        require(float(row["raw_event_score"]) == ps[index] and row["parent_group"] == parents[index]
                and row["source_dataset"] == sources[index], "CSV prediction/parent/source mismatch")
    flat = dict(p=np.asarray(ps, float), y=np.asarray(ys, bool), parents=np.asarray(parents, str),
                radii=np.asarray(radii, int), sources=np.asarray(sources, str), keys=keys)
    independent = dict(event_raw=independent_event_report(flat["p"], flat["y"], flat["parents"], flat["radii"]),
                       map=independent_map_report(maps, [s.scene_key for s in samples]) if maps else None, by_source={})
    compare(recorded["event_raw"], independent["event_raw"], split + "/event")
    compare(recorded["map"], independent["map"], split + "/map")
    for source in sorted({s.row["source_dataset"] for s in samples}):
        mask = flat["sources"] == source
        ss = [s for s in samples if s.row["source_dataset"] == source]
        sm = [m for s, m in zip(samples, maps) if s.row["source_dataset"] == source]
        report = dict(cases=len(ss),
            event_raw=independent_event_report(flat["p"][mask], flat["y"][mask], flat["parents"][mask], flat["radii"][mask]) if mask.any() else None,
            map=independent_map_report(sm, [s.scene_key for s in ss]) if sm else None)
        compare(recorded["by_source"][source], report, split + "/source/" + source)
        independent["by_source"][source] = report
    actual = dict(cases=len(samples), worlds=world_count, world_events=event_checks, explicit_disk_cases=explicit_checked,
                  map_parent_count=len({s.scene_key for s in samples}), event_parent_count=len(set(parents)),
                  events=len(ps), radius_monotonicity=monotonic)
    audit = json.loads((ROOT / protocol["data_root"] / "data_audit.json").read_text())["summary"][split]
    compare(actual, dict(cases=audit["observations"], map_parent_count=audit["parents"],
                         event_parent_count=audit["parents_with_queries"], events=audit["events"]), split + " complete scene support")
    return flat, independent, actual


def verify_source_macro(recorded, independent):
    macro = recorded["equal_source_macro"]
    for score_type in ("event_raw", "event_calibrated_diagnostic", "map"):
        names = ("sample_vote_cell_brier", "sample_vote_cell_nll") if score_type == "map" else (
            "event_brier", "event_nll", "event_ece", "false_safe_at_confidence", "coverage_at_confidence")
        for key in names:
            values = {}
            for source, detail in independent["by_source"].items():
                item = detail.get(score_type)
                if item is not None:
                    value = item["scene_weighted"][key] if score_type == "map" else item["scene_weighted"]["overall"][key]
                    if value is not None:
                        values[source] = value
            compare(macro[score_type][key], dict(mean=float(np.mean(list(values.values()))) if values else None,
                    contributing_sources=len(values), source_names=sorted(values)), "source macro/" + score_type + "/" + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/flatlands_external_formal_protocol_v1/protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = dict(passed=False, created_utc=datetime.now(timezone.utc).isoformat(),
                   evaluation_dir=str(args.evaluation_dir.resolve()), auditor_sha256=sha256(__file__),
                   scope="Single-stage engineering audit; not a formal three-repeat result or superiority claim",
                   gpu_inference_performed=False, new_raw_archive_or_test_images_opened=0)
    try:
        evaluation = args.evaluation_dir.resolve()
        require(evaluation.is_relative_to(ROOT), "evaluation directory escapes project")
        metrics = json.loads((evaluation / "metrics.json").read_text())
        protocol, protocol_hash = json.loads(args.protocol.read_text()), sha256(args.protocol)
        data = ROOT / protocol["data_root"]
        require(sha256(data / "seal.json") == protocol["data_seal_sha256"], "protocol/data seal mismatch")
        require(protocol["test_lock"]["final_test_locked"] and protocol["test_lock"]["location_6_locked"], "test lock absent")
        require(protocol["sampling"]["K"] == 4 and protocol["query"]["radii_cells"] == list(RADII), "K/radius mismatch")
        for relative, digest in protocol["source_hashes"].items():
            require(sha256(ROOT / relative) == digest, "frozen source changed: " + relative)
        for relative, digest in json.loads((data / "seal.json").read_text())["files"].items():
            require(sha256(data / relative) == digest, "sealed data changed: " + relative)
        figures, geometry = ROOT / protocol["figures"]["cases"], ROOT / protocol["figures"]["geometry_verification"]
        require(sha256(figures) == protocol["figures"]["sha256"] and sha256(geometry) == protocol["figures"]["geometry_verification_sha256"], "figure/geometry hash mismatch")
        require(json.loads(geometry.read_text())["passed"], "fixed figure geometry failed")
        cases = json.loads(figures.read_text())
        fixed = cases["general_cases"] + cases["multi_radius_cases"] + [c for g in cases["reference_categories"].values() for c in g["cases"]]
        explicit_ids = {case["global_id"] for case in fixed}
        for split in ("calibration", "validation"):
            ids = [r["global_id"] for r in protocol["scenes"][split]]
            ids.sort(key=lambda gid: hashlib.sha256(f"audit-explicit-v1|20260910|{split}|{gid}".encode()).hexdigest())
            explicit_ids.update(ids[:5])
        identity, checkpoints = verify_provenance(evaluation, metrics, protocol, protocol_hash)
        require(set(metrics["splits"]) == {"calibration", "validation"}, "Complete-stage audit requires both development splits")
        independent, flats, split_receipts = {}, {}, {}
        for split in ("calibration", "validation"):
            flats[split], independent[split], split_receipts[split] = score_split(evaluation, split, protocol, metrics, explicit_ids)
        fitted = json.loads((evaluation / "calibration_platt.json").read_text())
        platt_receipt = verify_platt(fitted, flats["calibration"], identity, protocol_hash)
        for split, flat in flats.items():
            clipped = np.clip(flat["p"], EPSILON, 1 - EPSILON)
            transformed = expit(fitted["a"] * (np.log(clipped) - np.log1p(-clipped)) + fitted["b"])
            calibrated = independent_event_report(transformed, flat["y"], flat["parents"], flat["radii"])
            compare(metrics["splits"][split]["event_calibrated_diagnostic"], calibrated, split + "/calibrated")
            independent[split]["event_calibrated_diagnostic"] = calibrated
            for source, detail in independent[split]["by_source"].items():
                mask = flat["sources"] == source
                value = independent_event_report(transformed[mask], flat["y"][mask], flat["parents"][mask], flat["radii"][mask]) if mask.any() else None
                compare(metrics["splits"][split]["by_source"][source]["event_calibrated_diagnostic"], value, split + "/source calibrated/" + source)
                detail["event_calibrated_diagnostic"] = value
            verify_source_macro(metrics["splits"][split], independent[split])
            event, maps = independent[split]["event_raw"]["scene_weighted"]["overall"], independent[split]["map"]
            compare(metrics["summary"][split], dict(event_brier=event["event_brier"], event_nll=event["event_nll"],
                    event_ece=event["event_ece"], false_safe_at_0_8=event["false_safe_at_confidence"],
                    coverage_at_0_8=event["coverage_at_confidence"], events=event["events"], scenes=event["scenes"],
                    map_brier=maps["scene_weighted"]["sample_vote_cell_brier"] if maps else None,
                    map_nll=maps["scene_weighted"]["sample_vote_cell_nll"] if maps else None), split + "/top summary")
        import torch
        require(not torch.cuda.is_initialized(), "audit created CUDA context")
        receipt.update(passed=True, method=metrics["method"], seed=metrics["outer_seed"], stage=metrics["stage"],
            protocol_sha256=protocol_hash, metrics_sha256=sha256(evaluation / "metrics.json"),
            complete_sha256=sha256(evaluation / "complete.json"), data_seal_sha256=sha256(data / "seal.json"),
            splits=split_receipts, checkpoints=checkpoints, calibration=platt_receipt,
            real_world_budget=protocol["methods"][metrics["method"]]["actual_K"], main_table_or_superiority_authorized=False,
            independence=dict(metrics="independent numpy arithmetic, no formal_metrics imports",
                all_world_geometry="historical _accelerated_events, independent evaluator wrapper",
                subset_geometry="explicit integer-disk erosion plus four-connected components", model_inference_replayed=False))
        with (args.output_dir / "independent_metrics.json").open("x") as handle:
            json.dump(independent, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    except BaseException as error:
        receipt.update(error=repr(error))
        raise
    finally:
        receipt.update(check_counts=dict(CHECKS), maximum_numeric_difference=MAX_DIFFERENCE, elapsed_seconds=time.perf_counter() - started)
        with (args.output_dir / "verification.json").open("x") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "splits"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
