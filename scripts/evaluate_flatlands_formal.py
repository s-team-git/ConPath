#!/usr/bin/env python3
"""Evaluate one registered method/seed on the sealed calibration and validation.

Every prediction artifact is label-free. Inference consumes only input fields;
the scorer uses reference fields after the prediction has been persisted.
The sealed packet loader includes references. There is no test split option.
This per-run evaluator never declares a three-seed superiority result.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathrel.formal_checkpoint import atomic_json, recipe_hash
from pathrel.formal_data import load_formal_data, sha256
from pathrel.formal_inference import LABELS, METHODS, RADII, make_models, sample_models
from pathrel.formal_metrics import (VERSION, apply_calibration_platt, event_report, exact_world_events,
                                   fit_calibration_platt, map_report, score_map_case)


def verify_protocol(protocol_dir, data_dir):
    path = Path(protocol_dir) / "protocol.json"
    protocol = json.loads(path.read_text())
    if protocol.get("seeds") != [20260831, 20260901, 20260902]:
        raise ValueError("Formal seeds differ from the user-specified three repeats")
    if protocol.get("data_seal_sha256") != sha256(Path(data_dir) / "seal.json"):
        raise ValueError("Formal protocol/data seal mismatch")
    hashes = protocol.get("source_sha256", protocol.get("source_hashes", {}))
    required = {"scripts/evaluate_flatlands_formal.py", "src/pathrel/formal_metrics.py", "src/pathrel/formal_inference.py", "src/pathrel/formal_data.py"}
    if not required.issubset(hashes):
        raise ValueError("Evaluator/data/inference sources were not frozen")
    for relative, expected in hashes.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != expected:
            raise ValueError("Frozen formal source changed: " + relative)
    sampling = protocol.get("sampling", {})
    if sampling.get("K") != 4 or sampling.get("solver_steps") != 25 or sampling.get("guidance") != 2.0:
        raise ValueError("Formal K/Heun/CFG contract is missing or changed")
    lock = protocol.get("test_lock", {})
    if lock.get("final_test_locked") is not True or lock.get("location_6_locked") is not True:
        raise ValueError("Explicit final-test/location_6 locks are required")
    return protocol, sha256(path)


def _receipt_for(checkpoint):
    for parent in (checkpoint.parent, *checkpoint.parents):
        if not parent.is_relative_to(ROOT):
            break
        candidate = parent / "run.json"
        if candidate.is_file():
            return json.loads(candidate.read_text()), candidate
    raise ValueError("No run.json provenance receipt for " + str(checkpoint))


def training_cost_receipt(folder, checkpoint_extra):
    """Actual run cost through the captured session receipts, not selected step."""
    folder = Path(folder)
    summary_path = folder / "time_summary.json"
    snapshot = {"training_wall_seconds": checkpoint_extra.get("training_wall_seconds"),
                "optimizer_update_seconds": checkpoint_extra.get("optimizer_update_seconds"),
                "peak_allocated_bytes": checkpoint_extra.get("peak_allocated_bytes", checkpoint_extra.get("peak_allocated_bytes_this_session")),
                "peak_reserved_bytes": checkpoint_extra.get("peak_reserved_bytes", checkpoint_extra.get("peak_reserved_bytes_this_session"))}
    totals = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    resources, cutoff_times = [], []
    peak_allocated = [snapshot["peak_allocated_bytes"]] if snapshot["peak_allocated_bytes"] is not None else []
    peak_reserved = [snapshot["peak_reserved_bytes"]] if snapshot["peak_reserved_bytes"] is not None else []
    for path in sorted((folder / "sessions").glob("*.json")):
        if not (path.name.endswith(".finished.json") or path.name.endswith(".heartbeat.json")):
            continue
        value = json.loads(path.read_text())
        if value.get("peak_allocated_bytes") is not None:
            peak_allocated.append(value["peak_allocated_bytes"])
        if value.get("peak_reserved_bytes") is not None:
            peak_reserved.append(value["peak_reserved_bytes"])
        if value.get("finished_utc"):
            cutoff_times.append(value["finished_utc"])
        resources.append({"path": str(path), "sha256": sha256(path)})
    return {"training_seconds": totals.get("training_wall_seconds", snapshot["training_wall_seconds"]),
            "training_wall_time_is_lower_bound": totals.get("wall_time_is_lower_bound", True),
            "training_time_summary_path": str(summary_path) if summary_path.exists() else None,
            "training_time_summary_sha256": sha256(summary_path) if summary_path.exists() else None,
            "training_cost_captured_utc": datetime.now(timezone.utc).isoformat(),
            "latest_finished_training_session_utc": max(cutoff_times) if cutoff_times else None,
            "training_cost_scope": "all run sessions recorded when evaluated, including replayed work; independent of selected checkpoint step",
            "sessions_without_final_receipt": totals.get("sessions_without_final_receipt", []),
            "training_peak_allocated_bytes": max(peak_allocated) if peak_allocated else None,
            "training_peak_reserved_bytes": max(peak_reserved) if peak_reserved else None,
            "training_resource_receipts": resources,
            "checkpoint_snapshot_cost": snapshot}


def load_registered_models(method, outer_seed, checkpoints, protocol, protocol_hash, device, *, one_member=False):
    expected = 4 if method == "lama" and not one_member else 1
    member_seeds = protocol["member_seeds"][method][str(outer_seed)]
    if method == "lama" and member_seeds != protocol["lama_member_seeds"][str(outer_seed)]:
        raise ValueError("LaMa member seed aliases disagree")
    if len(member_seeds) != (4 if method == "lama" else 1) or len(set(member_seeds)) != len(member_seeds):
        raise ValueError("Independent member seeds were not frozen")
    if checkpoints and len(checkpoints) != expected:
        raise ValueError("Expected one real checkpoint for each independent member")
    models, provenance = [], []
    for member_index in range(expected):
        seed = int(member_seeds[member_index])
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        group = make_models(method, device)
        # Discriminators are training-only and do not participate in inference.
        group = {"model": group["model"]}
        record = {"member_index": member_index, "initialization_seed": seed, "checkpoint": None,
                  "completed_steps": 0, "training_seconds": 0.0,
                  "model_parameters": sum(p.numel() for p in group["model"].parameters())}
        if checkpoints:
            checkpoint = Path(checkpoints[member_index]).resolve()
            receipt, receipt_path = _receipt_for(checkpoint)
            recipe = receipt["recipe"]
            if recipe_hash(recipe) != receipt["recipe_sha256"]:
                raise ValueError("Training recipe receipt hash mismatch")
            if recipe.get("method") != method or recipe.get("outer_seed", recipe.get("seed")) != outer_seed:
                raise ValueError("Checkpoint method/outer seed mismatch")
            if recipe.get("protocol_sha256") != protocol_hash:
                raise ValueError("Checkpoint was not trained under this formal protocol")
            if recipe.get("member") != member_index:
                raise ValueError("Checkpoint does not identify the expected independent member")
            envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = envelope["state"]
            if envelope["recipe_sha256"] != receipt["recipe_sha256"] or envelope["progress_index"] != state["completed_steps"]:
                raise ValueError("Checkpoint envelope/progress mismatch")
            group["model"].load_state_dict(state["models"]["model"])
            extra = state.get("extra", {})
            record.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha256(checkpoint),
                          run_receipt=str(receipt_path.relative_to(ROOT)), recipe_sha256=receipt["recipe_sha256"],
                          completed_steps=int(state["completed_steps"]),
                          **training_cost_receipt(receipt_path.parent, extra))
        models.append(group); provenance.append(record)
    if method == "lama" and len({p["completed_steps"] for p in provenance}) != 1:
        raise ValueError("LaMa ensemble members must share the registered milestone step")
    return models if method == "lama" else models[0], provenance


def _atomic_npz(path, arrays):
    temporary = path.with_name("." + path.name + ".tmp." + str(os.getpid()))
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_source_macro(report):
    """Equal dataset-source diagnostic; retain undefined selective risks."""
    macro = {"definition": "arithmetic mean of source-specific scene-weighted metrics; this does not replace the primary scene-weighted result",
             "undefined_policy": "exclude undefined values, report contributing source count/names; never substitute zero"}
    for score_type in ("event_raw", "event_calibrated_diagnostic"):
        metric_values = {}
        for key in ("event_brier", "event_nll", "event_ece", "false_safe_at_confidence", "coverage_at_confidence"):
            values = {source: detail[score_type]["scene_weighted"]["overall"][key]
                      for source, detail in report["by_source"].items() if detail.get(score_type) is not None}
            values = {source: value for source, value in values.items() if value is not None}
            metric_values[key] = {"mean": float(np.mean(list(values.values()))) if values else None,
                                  "contributing_sources": len(values), "source_names": sorted(values)}
        if any(v["contributing_sources"] for v in metric_values.values()):
            macro[score_type] = metric_values
    map_values = {}
    for key in ("sample_vote_cell_brier", "sample_vote_cell_nll"):
        values = {source: detail["map"]["scene_weighted"][key]
                  for source, detail in report["by_source"].items() if detail.get("map") is not None}
        values = {source: value for source, value in values.items() if value is not None}
        map_values[key] = {"mean": float(np.mean(list(values.values()))) if values else None,
                           "contributing_sources": len(values), "source_names": sorted(values)}
    macro["map"] = map_values
    report["equal_source_macro"] = macro


def evaluate_split(models, method, samples, seed, output, *, one_member=False):
    """Save inference, then score frozen labels; all supplied cases are included."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    maps, scenes, scores, targets, radii, keys, costs, rows, cases, sources = [], [], [], [], [], [], [], [], [], []
    semantics = None
    for sample in samples:
        gid, scene = sample.row["global_id"], sample.row["parent_group"]
        prediction = sample_models(models, method, sample, seed, one_member=one_member)
        artifacts = {"global_id": np.array(gid), "candidate_indices": sample.candidate_indices,
                     "starts": sample.starts, "goals": sample.goals, "radii_cells": np.array(RADII),
                     "event_scores": prediction["event_scores"]}
        if prediction["worlds"] is not None:
            artifacts.update(worlds=prediction["worlds"], world_events=prediction["world_events"],
                             continuous_score=prediction["continuous_score"])
        path = output / "predictions" / (gid + ".npz")
        _atomic_npz(path, artifacts)
        # Targets are first consumed by this function here, after artifact write.
        reference_events = exact_world_events(sample.target[None], sample.starts, sample.goals, RADII)[0]
        if not np.array_equal(reference_events, sample.targets):
            raise ValueError("Independent target geometry replay disagrees: " + gid)
        if prediction["worlds"] is not None:
            maps.append(score_map_case(prediction["worlds"], sample.target, sample.hidden, sample.valid,
                                       sample.observation[0], continuous_score=prediction["continuous_score"],
                                       continuous_score_semantics=prediction["continuous_score_semantics"]))
        costs.append(prediction["efficiency"])
        p = prediction["event_scores"]
        if p.shape != sample.targets.shape:
            raise ValueError("Prediction/query shape mismatch")
        semantics = prediction["event_score_semantics"]
        cases.append({"global_id": gid, "parent_group": scene, "source": sample.row["source_dataset"],
                      "input_key_sha256": prediction["input_key_sha256"], "sampling_seed": prediction["sampling_seed"],
                      "predictions_sha256": sha256(path), "queries": len(sample.starts),
                      "map": maps[-1] if prediction["worlds"] is not None else None, "efficiency": costs[-1]})
        for qi, candidate in enumerate(sample.candidate_indices):
            for ri, radius in enumerate(RADII):
                key = f"{gid}/{int(candidate)}/{radius}"
                scores.append(float(p[qi, ri])); targets.append(bool(sample.targets[qi, ri]))
                scenes.append(scene); radii.append(radius); keys.append(key); sources.append(sample.row["source_dataset"])
                rows.append({"global_id": gid, "parent_group": scene, "query_id": int(candidate),
                             "radius_cells": radius, "source_dataset": sample.row["source_dataset"],
                             "raw_event_score": float(p[qi, ri]), "score_semantics": semantics})
        print(json.dumps({"evaluation_split": output.name, "method": method, "seed": seed,
                          "completed_cases": len(cases), "total_cases": len(samples), "global_id": gid}), flush=True)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate frozen event key")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_id", "parent_group", "query_id", "radius_cells", "source_dataset", "raw_event_score", "score_semantics"])
        writer.writeheader(); writer.writerows(rows)
    report = {"event_raw": event_report(scores, targets, scenes, radii, score_semantics=semantics),
              "map": map_report(maps, [s.row["parent_group"] for s in samples]) if maps else None,
              "cases": cases, "evaluation_split": output.name, "development_only": True,
              "validation_only": output.name == "validation", "final_test": False,
              "efficiency": {"case_count": len(costs),
                             "generation_seconds_mean": float(np.mean([c["generation_seconds"] for c in costs])),
                             "generation_seconds_median": float(np.median([c["generation_seconds"] for c in costs])),
                             "full_update_seconds_mean": float(np.mean([c["full_update_seconds"] for c in costs])),
                             "full_update_seconds_median": float(np.median([c["full_update_seconds"] for c in costs])),
                             "actual_batch_forward_calls": sum(c["actual_batch_forward_calls"] for c in costs),
                             "actual_model_input_examples": sum(c["actual_model_input_examples"] for c in costs),
                             "peak_allocated_bytes": max((c["peak_allocated_bytes"] for c in costs if c["peak_allocated_bytes"] is not None), default=None),
                             "peak_reserved_bytes": max((c["peak_reserved_bytes"] for c in costs if c["peak_reserved_bytes"] is not None), default=None),
                             "includes_first_case_cold_execution": True}}
    report["by_source"] = {}
    for source in sorted({s.row["source_dataset"] for s in samples}):
        mask = np.asarray(sources) == source
        source_samples = [s for s in samples if s.row["source_dataset"] == source]
        source_maps = [m for s, m in zip(samples, maps) if s.row["source_dataset"] == source]
        report["by_source"][source] = {
            "cases": len(source_samples),
            "event_raw": event_report(np.asarray(scores)[mask], np.asarray(targets)[mask], np.asarray(scenes)[mask],
                                      np.asarray(radii)[mask], score_semantics=semantics) if mask.any() else None,
            "map": map_report(source_maps, [s.row["parent_group"] for s in source_samples]) if source_maps else None}
    update_source_macro(report)
    atomic_json(output / "raw_metrics.json", report)
    return report, {"scores": scores, "targets": targets, "scene_ids": scenes, "radius_cells": radii, "query_keys": keys, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[20260831, 20260901, 20260902])
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--stage", choices=["untrained", "smoke", "pilot", "formal", "checkpoint_calibration"], required=True)
    parser.add_argument("--one-member", action="store_true")
    parser.add_argument("--split", choices=["calibration", "validation", "both"], default="both")
    parser.add_argument("--calibration-file", type=Path, help="Required for validation-only execution; fitted on the same checkpoint(s)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing nonempty evaluation output directory")
    if args.stage == "untrained" and args.checkpoint or args.stage != "untrained" and not args.checkpoint:
        raise ValueError("Untrained stage must start from random weights; trained stages require checkpoints")
    if args.one_member and args.stage in ("formal", "checkpoint_calibration"):
        raise ValueError("A formal LaMa comparison requires the actual four-member ensemble")
    if args.split == "validation" and args.calibration_file is None:
        raise ValueError("Validation-only execution requires a previously fitted calibration file")
    declared_data = None if args.data_dir is not None else json.loads((args.protocol_dir / "protocol.json").read_text())["data_root"]
    data_dir = args.data_dir or ROOT / declared_data
    protocol, protocol_hash = verify_protocol(args.protocol_dir, data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    models, provenance = load_registered_models(args.method, args.seed, args.checkpoint, protocol, protocol_hash, args.device, one_member=args.one_member)
    atomic_json(args.output_dir / "evaluation_frozen_before_scoring.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "method": args.method, "label": LABELS[args.method],
        "stage": args.stage, "seed": args.seed, "protocol_sha256": protocol_hash,
        "evaluator_sha256": sha256(Path(__file__)), "metric_version": VERSION,
        "checkpoint_provenance": provenance, "one_member_diagnostic": args.one_member,
        "new_physical_test_images_opened": 0, "final_test_locked": True, "location_6_locked": True})
    started = time.perf_counter()
    reports, flat = {}, {}
    splits = ("calibration",) if args.stage == "checkpoint_calibration" else ("calibration", "validation") if args.split == "both" else (args.split,)
    for split in splits:
        samples = load_formal_data(data_dir, split)
        reports[split], flat[split] = evaluate_split(models, args.method, samples, args.seed, args.output_dir / split, one_member=args.one_member)
    model_identity = [{k: p.get(k) for k in ("member_index", "initialization_seed", "checkpoint_sha256", "completed_steps")} for p in provenance]
    if "calibration" in flat:
        cal = flat["calibration"]
        fitted = fit_calibration_platt(cal["scores"], cal["targets"], cal["scene_ids"], split="calibration",
                                       query_keys=cal["query_keys"], protocol_sha256=protocol_hash)
        fitted["prediction_model_identity"] = model_identity
    else:
        fitted = json.loads(args.calibration_file.read_text())
        if fitted.get("prediction_model_identity") != model_identity:
            raise ValueError("Calibration was fitted to different model checkpoints")
    atomic_json(args.output_dir / "calibration_platt.json", fitted)
    for split in splits:
        values = flat[split]
        calibrated = apply_calibration_platt(values["scores"], fitted, protocol_sha256=protocol_hash)
        reports[split]["event_calibrated_diagnostic"] = event_report(calibrated, values["targets"], values["scene_ids"], values["radius_cells"], score_semantics="calibration-only monotone Platt diagnostic")
        for source, detail in reports[split]["by_source"].items():
            mask = np.asarray(values["sources"]) == source
            detail["event_calibrated_diagnostic"] = event_report(
                calibrated[mask], np.asarray(values["targets"])[mask], np.asarray(values["scene_ids"])[mask],
                np.asarray(values["radius_cells"])[mask], score_semantics="same global calibration-only Platt diagnostic") if mask.any() else None
        update_source_macro(reports[split])
        atomic_json(args.output_dir / split / "metrics.json", reports[split])
    violations = sum(r["map"]["observed_evidence_violation_count"] + r["map"]["valid_support_violation_count"]
                     + r["map"].get("continuous_completion_score_diagnostic", {}).get("observed_evidence_violation_count", 0)
                     + r["map"].get("continuous_completion_score_diagnostic", {}).get("valid_support_violation_count", 0)
                     for r in reports.values() if r["map"] is not None)
    summary = {"method": args.method, "label": LABELS[args.method], "outer_seed": args.seed, "stage": args.stage,
               "splits": reports, "protocol_sha256": protocol_hash, "checkpoint_provenance": provenance,
               "observed_and_support_constraints_passed": violations == 0,
               "evaluation_seconds": time.perf_counter() - started,
               "main_table_eligible": False, "main_table_gate": "Requires full-plan convergence, all three independent outer repeats, and unified final audit",
               "new_physical_test_images_opened": 0, "final_test_locked": True, "location_6_locked": True,
               "external_paper_scores_imported": False}
    summary["summary"] = {}
    for split, report in reports.items():
        event = report["event_raw"]["scene_weighted"]["overall"]
        maps = report["map"]
        native = maps.get("continuous_completion_score_diagnostic", {}).get("scene_weighted", {}) if maps else {}
        summary["summary"][split] = {
            "event_brier": event["event_brier"], "event_nll": event["event_nll"], "event_ece": event["event_ece"],
            "false_safe_at_0_8": event["false_safe_at_confidence"], "coverage_at_0_8": event["coverage_at_confidence"],
            "map_brier": maps["scene_weighted"]["sample_vote_cell_brier"] if maps else None,
            "map_nll": maps["scene_weighted"]["sample_vote_cell_nll"] if maps else None,
            "continuous_score_brier_diagnostic": native.get("hidden_brier"),
            "continuous_score_nll_diagnostic": native.get("hidden_nll"),
            "observed_evidence_violation_count": maps["observed_evidence_violation_count"] if maps else None,
            "valid_support_violation_count": maps["valid_support_violation_count"] if maps else None,
            "collapse_diagnostic": maps["collapse_diagnostic"] if maps else None,
            "scenes": event["scenes"], "events": event["events"]}
    atomic_json(args.output_dir / "metrics.json", summary)
    if violations:
        raise ValueError("Observed/support constraint violations; saved evaluation is a failed diagnostic")
    atomic_json(args.output_dir / "complete.json", {"passed": True, "metrics_sha256": sha256(args.output_dir / "metrics.json"),
                "method": args.method, "seed": args.seed, "stage": args.stage, "completed_steps": [p["completed_steps"] for p in provenance],
                "final_test_locked": True, "new_physical_test_images_opened": 0})


if __name__ == "__main__":
    main()
