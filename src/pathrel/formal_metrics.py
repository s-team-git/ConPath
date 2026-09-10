"""CPU-only scoring contract for the new, locked FlatLands external comparison.

This module does not load datasets, choose checkpoints, clamp predictions, or
fit calibration on evaluation labels. Cell free frequency and event frequency
are different quantities: the latter is computed AFTER exact connectivity in
each actual binary world. Deterministic one-world outputs remain K=1.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.special import expit
import hashlib
import json


VERSION = "flatlands-formal-metrics-v1"
NLL_EPSILON = 1e-6
ECE_BINS = 10
CONFIDENCE = 0.8
FOUR_NEIGHBOURS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _binary(value, name):
    value = np.asarray(value)
    if not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be finite and binary")
    return value.astype(bool)


def _scores(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _integer(value, name):
    value = np.asarray(value)
    if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise ValueError(f"{name} must contain integers")
    return value.astype(np.int64)


def _losses(p, y, epsilon=NLL_EPSILON):
    if not 0 < epsilon < 0.5:
        raise ValueError("Invalid NLL clipping epsilon")
    clipped = np.clip(p, epsilon, 1 - epsilon)
    return (p - y) ** 2, np.where(y, -np.log(clipped), -np.log1p(-clipped))


def scene_weights(scene_ids):
    """Equal scene mass; equal event mass within each scene in this subset."""
    ids = np.asarray(scene_ids, dtype=str)
    if ids.ndim != 1 or not len(ids) or np.any(ids == ""):
        raise ValueError("Nonempty one-dimensional scene IDs are required")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def _weighted_summary(p, y, weights, confidence, bins, epsilon):
    if not len(p):
        return {"events": 0, "event_brier": None, "event_nll": None,
                "event_ece": None, "coverage_at_confidence": None,
                "false_safe_at_confidence": None, "false_safe_mass": None,
                "reachable_prevalence": None, "accepted_events": 0,
                "reliability": [], "false_safe_coverage_curve": []}
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != p.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid event weights")
    w = w / w.sum()
    brier, nll = _losses(p, y, epsilon)
    accepted = p >= confidence
    coverage = float(w[accepted].sum())
    false_mass = float(np.sum(w[accepted] * (1 - y[accepted])))
    # Intervals are [i/B, (i+1)/B), except the final interval includes 1.
    assignments = np.minimum(np.floor(p * bins).astype(np.int64), bins - 1)
    reliability, ece = [], 0.0
    for index in range(bins):
        included = assignments == index
        mass = float(w[included].sum())
        predicted = float(np.sum(w[included] * p[included]) / mass) if mass else None
        empirical = float(np.sum(w[included] * y[included]) / mass) if mass else None
        if mass:
            ece += mass * abs(predicted - empirical)
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "upper_inclusive": index == bins - 1,
                            "events": int(included.sum()), "weight": mass,
                            "predicted": predicted, "observed": empirical})
    # Threshold sets include complete ties; no ordering uses evaluation labels.
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_w, sorted_y = p[order], w[order], y[order]
    cumulative_mass = np.cumsum(sorted_w)
    cumulative_errors = np.cumsum(sorted_w * (1 - sorted_y))
    endpoints = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    curve = [{"threshold": None, "acceptance": "none", "coverage": 0.0, "false_safe": None}]
    curve.extend({"threshold": float(sorted_p[j]), "acceptance": "score >= threshold",
                  "coverage": float(min(1.0, cumulative_mass[j])),
                  "false_safe": float(cumulative_errors[j] / cumulative_mass[j])} for j in endpoints)
    return {"events": len(p), "event_brier": float(np.sum(w * brier)),
            "event_nll": float(np.sum(w * nll)), "event_ece": float(ece),
            "coverage_at_confidence": coverage,
            "false_safe_at_confidence": false_mass / coverage if coverage else None,
            "false_safe_mass": false_mass, "reachable_prevalence": float(np.sum(w * y)),
            "accepted_events": int(accepted.sum()), "reliability": reliability,
            "false_safe_coverage_curve": curve}


def event_report(scores, targets, scene_ids, radius_cells, *, confidence=CONFIDENCE,
                 bins=ECE_BINS, epsilon=NLL_EPSILON, score_semantics="raw_world_event_frequency"):
    """Pooled and scene-weighted metrics, stratified by truth and grid radius.

    Inputs are aligned flat arrays, one row per frozen (scene, query, radius).
    Scene-weighted ECE is ECE of the globally scene-weighted events, not an
    average of per-scene ECE. Each stratum renormalizes over its eligible scenes.
    False-safe means P(unreachable | accepted); false_safe_mass is the separate
    unconditional weighted error mass. An empty acceptance set has undefined
    false-safe (None), never a misleading zero.
    """
    p, y = _scores(scores, "event scores"), _binary(targets, "event targets")
    scenes, radii = np.asarray(scene_ids, dtype=str), _integer(radius_cells, "radius cells")
    if p.ndim != 1 or not len(p) or not (p.shape == y.shape == scenes.shape == radii.shape):
        raise ValueError("Expected nonempty aligned one-dimensional event arrays")
    if (radii < 0).any() or np.any(scenes == ""):
        raise ValueError("Negative radius or empty scene ID")
    if not 0 <= confidence <= 1 or not isinstance(bins, int) or bins < 1:
        raise ValueError("Invalid confidence or reliability bin count")
    if not score_semantics:
        raise ValueError("Event score semantics must be recorded")

    def subset(mask, weighting):
        weights = scene_weights(scenes[mask]) if mask.any() and weighting == "scene_weighted" else np.ones(int(mask.sum()))
        result = _weighted_summary(p[mask], y[mask], weights, confidence, bins, epsilon)
        result["scenes"] = int(len(np.unique(scenes[mask])))
        return result

    out = {"version": VERSION, "score_semantics": score_semantics,
           "nll_clip_epsilon": epsilon, "ece_bins": bins, "confidence": confidence,
           "weighting_contract": "equal scenes, then equal events per scene; recomputed inside each stratum",
           "false_safe_definition": "weighted unreachable-and-accepted mass / weighted accepted mass",
           "radius_unit": "grid cell", "events": len(p), "scenes": len(np.unique(scenes))}
    for weighting in ("pooled", "scene_weighted"):
        out[weighting] = {"overall": subset(np.ones(len(p), dtype=bool), weighting),
                          "by_truth": {"reachable": subset(y, weighting), "unreachable": subset(~y, weighting)},
                          "by_radius": {str(r): subset(radii == r, weighting) for r in sorted(np.unique(radii))},
                          "by_radius_and_truth": {
                              str(r): {"reachable": subset((radii == r) & y, weighting),
                                       "unreachable": subset((radii == r) & ~y, weighting)}
                              for r in sorted(np.unique(radii))}}
    return out


def exact_world_events(worlds, starts, goals, radii_cells=(0, 10, 20)):
    """Closed integer disk footprints and exact four-neighbour connectivity.

    The map exterior is blocked. Every integer offset satisfying dx²+dy²<=r²
    must be free. EDT distance strictly greater than r implements that disk.
    No differentiable finite-iteration reachability or mean-map thresholding.
    """
    w = _binary(worlds, "worlds")
    starts, goals = _integer(starts, "starts"), _integer(goals, "goals")
    radii = _integer(radii_cells, "radius cells")
    if w.ndim != 3 or not all(w.shape):
        raise ValueError("Expected nonempty KxHxW binary worlds")
    if starts.ndim != 2 or starts.shape != goals.shape or starts.shape[1:] != (2,):
        raise ValueError("Expected aligned Qx2 row/column endpoints")
    if radii.ndim != 1 or not len(radii) or (radii < 0).any() or len(np.unique(radii)) != len(radii):
        raise ValueError("Expected distinct nonnegative integer radii")
    for endpoints in (starts, goals):
        if (endpoints < 0).any() or (endpoints >= np.array(w.shape[1:])).any():
            raise ValueError("Query endpoint outside the fixed grid")
    result = np.zeros((len(w), len(starts), len(radii)), dtype=bool)
    for k, world in enumerate(w):
        distance = ndimage.distance_transform_edt(np.pad(world, 1, constant_values=False))[1:-1, 1:-1]
        for r, radius in enumerate(radii):
            components, _ = ndimage.label(distance > radius, structure=FOUR_NEIGHBOURS)
            a = components[starts[:, 0], starts[:, 1]]
            b = components[goals[:, 0], goals[:, 1]]
            result[k, :, r] = (a > 0) & (a == b)
    return result


def score_map_case(worlds, target, hidden, support, observed_free, *, continuous_score=None,
                   continuous_score_semantics=None, epsilon=NLL_EPSILON):
    """Score actual worlds without repairing their constraint violations.

    Primary cell metrics use binary-world free frequency, NOT a path score.
    Optional native continuous scores are a separately named diagnostic; FM
    clipped flow output is not automatically a calibrated occupancy probability.
    A continuous_score may be HxW or KxHxW; the latter is averaged across K.
    """
    worlds, target, hidden, support, observed = [_binary(v, n) for v, n in (
        (worlds, "worlds"), (target, "target"), (hidden, "hidden"),
        (support, "support"), (observed_free, "observed free"))]
    if worlds.ndim != 3 or not all(worlds.shape) or any(v.shape != worlds.shape[1:] for v in (target, hidden, support, observed)):
        raise ValueError("Expected aligned KxHxW worlds and HxW masks")
    if (hidden & ~support).any() or (observed & (~support | hidden)).any():
        raise ValueError("Inconsistent hidden/support/observed masks")
    known = support & ~hidden
    if (target[known] != observed[known]).any():
        raise ValueError("Target disagrees with frozen observed evidence")
    hidden_count = int(hidden.sum())
    brier, nll = _losses(worlds.mean(0)[hidden], target[hidden], epsilon)
    result = {"actual_world_count": len(worlds), "hidden_cells": hidden_count,
              "sample_vote_cell_brier": float(brier.mean()) if hidden_count else None,
              "sample_vote_cell_nll": float(nll.mean()) if hidden_count else None,
              "observed_evidence_violation_count": int(np.count_nonzero(worlds[:, known] != observed[known])),
              "valid_support_violation_count": int(np.count_nonzero(worlds[:, ~support])),
              "observed_cell_world_checks": int(known.sum()) * len(worlds),
              "invalid_cell_world_checks": int((~support).sum()) * len(worlds),
              "hidden_world_free_fraction": worlds[:, hidden].mean(1).tolist() if hidden_count else [],
              "reference_hidden_free_fraction": float(target[hidden].mean()) if hidden_count else None,
              "valid_free_components_per_world": [int(ndimage.label(w & support, structure=FOUR_NEIGHBOURS)[1]) for w in worlds],
              "reference_valid_free_components": int(ndimage.label(target & support, structure=FOUR_NEIGHBOURS)[1]),
              "distinct_world_count": len({np.ascontiguousarray(w).tobytes() for w in worlds}),
              "continuous_completion_score": None}
    if continuous_score is not None:
        p = _scores(continuous_score, "continuous completion score")
        if p.shape == target.shape:
            p = p[None]
        if p.ndim != 3 or p.shape[1:] != target.shape or len(p) not in (1, len(worlds)):
            raise ValueError("Continuous completion score must align with actual worlds")
        if not continuous_score_semantics:
            raise ValueError("Native continuous score semantics are required")
        cb, cn = _losses(p.mean(0)[hidden], target[hidden], epsilon)
        result["continuous_completion_score"] = {
            "semantics": continuous_score_semantics,
            "hidden_brier": float(cb.mean()) if hidden_count else None,
            "hidden_nll": float(cn.mean()) if hidden_count else None,
            "observed_evidence_violation_count": int(np.count_nonzero(p[:, known] != observed[known])),
            "valid_support_violation_count": int(np.count_nonzero(p[:, ~support]))}
    return result


def map_report(case_reports, scene_ids):
    """Aggregate cells pooled, or equally within observations then scenes."""
    cases, scenes = list(case_reports), np.asarray(scene_ids, dtype=str)
    if not cases or scenes.shape != (len(cases),) or np.any(scenes == ""):
        raise ValueError("Expected aligned nonempty cases and scene IDs")
    counts = np.asarray([c["hidden_cells"] for c in cases], dtype=np.int64)
    active = counts > 0
    metrics = ("sample_vote_cell_brier", "sample_vote_cell_nll")
    result = {"version": VERSION, "cases": len(cases), "scenes": len(np.unique(scenes)),
              "hidden_cells": int(counts.sum()), "cases_with_hidden_cells": int(active.sum()),
              "cell_score_semantics": "per-cell free frequency among actual binary worlds; not event probability",
              "actual_world_counts": sorted({int(c["actual_world_count"]) for c in cases}),
              "observed_evidence_violation_count": sum(c["observed_evidence_violation_count"] for c in cases),
              "valid_support_violation_count": sum(c["valid_support_violation_count"] for c in cases),
              "weighting_contract": "pooled cells; or equal scenes, equal nonempty observations, equal hidden cells within observation"}
    fractions = [f for c in cases for f in c["hidden_world_free_fraction"]]
    reference_fractions = [c["reference_hidden_free_fraction"] for c in cases if c["reference_hidden_free_fraction"] is not None]
    result["collapse_diagnostic"] = {
        "thresholds": {"nearly_all_blocked": "hidden free fraction <= .01", "nearly_all_free": "hidden free fraction >= .99"},
        "generated_worlds_with_hidden": len(fractions),
        "generated_nearly_all_blocked_fraction": float(np.mean(np.asarray(fractions) <= .01)) if fractions else None,
        "generated_nearly_all_free_fraction": float(np.mean(np.asarray(fractions) >= .99)) if fractions else None,
        "reference_nearly_all_blocked_case_fraction": float(np.mean(np.asarray(reference_fractions) <= .01)) if reference_fractions else None,
        "reference_nearly_all_free_case_fraction": float(np.mean(np.asarray(reference_fractions) >= .99)) if reference_fractions else None,
        "mean_valid_free_components_per_world": float(np.mean([v for c in cases for v in c["valid_free_components_per_world"]])),
        "mean_reference_valid_free_components": float(np.mean([c["reference_valid_free_components"] for c in cases])),
        "weighting": "diagnostic output/case fractions; raw per-case quantities retained, no outcome filtering"}
    for weighting in ("pooled", "scene_weighted"):
        weights = counts[active] if weighting == "pooled" else scene_weights(scenes[active]) if active.any() else np.array([])
        result[weighting] = {m: float(np.average([c[m] for c, a in zip(cases, active) if a], weights=weights)) if active.any() else None for m in metrics}
    native = [c.get("continuous_completion_score") for c in cases]
    if any(v is not None for v in native):
        if not all(v is not None for v in native) or len({v["semantics"] for v in native}) != 1:
            raise ValueError("Cannot combine missing or semantically different native scores")
        diagnostic = {"semantics": native[0]["semantics"]}
        for weighting in ("pooled", "scene_weighted"):
            weights = counts[active] if weighting == "pooled" else scene_weights(scenes[active]) if active.any() else np.array([])
            diagnostic[weighting] = {m: float(np.average([c[m] for c, a in zip(native, active) if a], weights=weights)) if active.any() else None
                                     for m in ("hidden_brier", "hidden_nll")}
        diagnostic["observed_evidence_violation_count"] = sum(v["observed_evidence_violation_count"] for v in native)
        diagnostic["valid_support_violation_count"] = sum(v["valid_support_violation_count"] for v in native)
        result["continuous_completion_score_diagnostic"] = diagnostic
    return result


def fit_calibration_platt(scores, targets, scene_ids, *, split, query_keys,
                          protocol_sha256, regularization=0.01, epsilon=NLL_EPSILON):
    """Fit one global, monotone, identity-regularized mapping on calibration only.

    This optional diagnostic is separate from the raw event-frequency main row.
    It uses no per-radius/per-source model selection. Its finite-data fit does
    not establish guaranteed calibration. a>=0 preserves score ordering.
    """
    if split != "calibration":
        raise ValueError("Probability calibration may only fit the calibration split")
    if len(protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in protocol_sha256):
        raise ValueError("A frozen protocol SHA-256 is required")
    p, y = _scores(scores, "calibration scores"), _binary(targets, "calibration targets")
    scenes, keys = np.asarray(scene_ids, dtype=str), list(query_keys)
    if p.ndim != 1 or not len(p) or p.shape != y.shape or p.shape != scenes.shape or len(keys) != len(p) or len(set(keys)) != len(keys):
        raise ValueError("Calibration inputs and unique frozen event keys must align")
    if regularization <= 0 or not 0 < epsilon < .5:
        raise ValueError("Positive regularization and valid clipping required")
    weights = scene_weights(scenes)
    clipped = np.clip(p, epsilon, 1 - epsilon)
    x = np.log(clipped) - np.log1p(-clipped)

    def objective(parameters):
        a, b = parameters
        z = a * x + b
        loss = np.sum(weights * (np.logaddexp(0, z) - y * z)) + .5 * regularization * ((a - 1)**2 + b*b)
        error = weights * (expit(z) - y)
        gradient = np.array([np.sum(error*x) + regularization*(a-1), np.sum(error) + regularization*b])
        return loss, gradient

    optimized = minimize(objective, [1., 0.], method="L-BFGS-B", jac=True,
                         bounds=[(0., 20.), (-20., 20.)],
                         options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000})
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise ValueError("Calibration optimizer failed: " + str(optimized.message))
    payload = {"scores": p.tolist(), "targets": y.astype(int).tolist(), "scene_ids": scenes.tolist(), "query_keys": keys}
    return {"version": VERSION, "fit_split": split, "protocol_sha256": protocol_sha256,
            "method": "sigmoid(a*logit(clip(raw))+b), a>=0; identity L2 regularization",
            "a": float(optimized.x[0]), "b": float(optimized.x[1]),
            "regularization": regularization, "clip_epsilon": epsilon,
            "events": len(p), "scenes": len(np.unique(scenes)),
            "calibration_data_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "optimizer_success": True, "objective": float(optimized.fun), "guaranteed_calibration": False}


def apply_calibration_platt(scores, fitted, *, protocol_sha256):
    if fitted.get("fit_split") != "calibration" or fitted.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Calibration scope/protocol mismatch")
    a, b, epsilon = fitted["a"], fitted["b"], fitted["clip_epsilon"]
    if not np.isfinite([a, b, epsilon]).all() or a < 0 or not 0 < epsilon < .5:
        raise ValueError("Invalid calibration parameters")
    clipped = np.clip(_scores(scores, "event scores"), epsilon, 1 - epsilon)
    return expit(a * (np.log(clipped) - np.log1p(-clipped)) + b)
