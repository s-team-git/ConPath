#!/usr/bin/env python3
"""Audit a validation-only FlatLands F=16 K=128 decoder run.

The audit is deliberately label-safe: it reads only the frozen selection/query files through the
``split=validation`` evaluator, never opens the archive, and rejects any attempt to evaluate a test
split.  It checks run flags/configuration, prediction hashes and keys, probability finiteness,
radius ordering, and an independent metric replay for all three seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from pathrel.flatlands_eval import evaluate_flatlands_prediction_file
from pathrel.model import PathRelNet


SEEDS = (20260831, 20260901, 20260902)
RADIUS = (0, 10, 20)
EXPECTED_ROWS = 4224
EXPECTED_GROUPS = 1408
SELECTION_SHA256 = "4e7ae4c992cf943ab81618e3826c4748fcaaa97c3c4d7cb187518ee3fe6a9409"
QUERIES_SHA256 = "33e7f8a0343269b0dde47b428b3be622c80effdb0f80ae34b352ca282018d60d"
METRICS = ("brier", "nll", "ece", "false_safe_rate@0.8", "high_confidence_safe_coverage@0.8")
EXPECTED_PROTOCOL_VERSION = "P1_BASELINE_PROTOCOL.md v1 + ConPath valid-support v2"
EXPECTED_SUPPORT_POLICY = (
    "FlatLands epistemic_mask complement is deterministically blocked before posterior sampling"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _close(actual: object, expected: object, tolerance: float = 1e-10) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    return actual == expected


def _overall(report: dict[str, Any]) -> dict[str, Any]:
    for metric in report.get("metrics", []):
        if isinstance(metric, dict) and metric.get("scope") == "overall":
            value = metric.get("scene_weighted")
            if isinstance(value, dict):
                return value
    raise ValueError("validation report has no overall scene_weighted metric")


def _audit_prediction(path: Path) -> tuple[int, int, list[str]]:
    failures: list[str] = []
    groups: dict[tuple[str, str], dict[int, float]] = {}
    required = {"global_id", "candidate_index", "radius_cells", "probability"}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != required:
            failures.append(f"prediction header mismatch: {reader.fieldnames!r}")
        rows = 0
        for row in reader:
            rows += 1
            try:
                radius = int(row["radius_cells"])
                probability = float(row["probability"])
                key = (str(row["global_id"]), str(row["candidate_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"row {rows} parse error: {exc}")
                continue
            if radius not in RADIUS:
                failures.append(f"row {rows} unexpected radius {radius}")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                failures.append(f"row {rows} probability outside [0,1]: {probability!r}")
            if radius in RADIUS and radius in groups.setdefault(key, {}):
                failures.append(f"duplicate prediction key: {key}/{radius}")
            groups.setdefault(key, {})[radius] = probability
    if rows != EXPECTED_ROWS:
        failures.append(f"prediction row count {rows} != {EXPECTED_ROWS}")
    if len(groups) != EXPECTED_GROUPS:
        failures.append(f"endpoint group count {len(groups)} != {EXPECTED_GROUPS}")
    for key, values in groups.items():
        if set(values) != set(RADIUS):
            failures.append(f"incomplete radius group: {key}")
        elif values[0] < values[10] - 1e-12 or values[10] < values[20] - 1e-12:
            failures.append(f"radius monotonicity violation: {key} {values}")
    return rows, len(groups), failures


def _audit_checkpoint(
    path: Path, *, decoder_variant: str, seed: int
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint payload is not an object: {path}")
    if state.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        failures.append("checkpoint protocol version mismatch")
    config = state.get("config")
    if not isinstance(config, dict):
        failures.append("checkpoint config is missing")
        config = {}
    if config.get("seed") != seed:
        failures.append("checkpoint seed mismatch")
    if config.get("decoder_variant") != decoder_variant:
        failures.append("checkpoint decoder variant mismatch")
    model = PathRelNet(
        input_channels=3,
        feature_channels=int(config.get("feature_channels", 16)),
        latent_dim=int(config.get("latent_dim", 4)),
        local_kernel_size=1 if decoder_variant == "independent" else 5,
    )
    model_state = state.get("model")
    if not isinstance(model_state, dict):
        failures.append("checkpoint model state is missing")
        finite = False
    else:
        model.load_state_dict(model_state, strict=True)
        finite = all(
            not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
            for value in model_state.values()
            if isinstance(value, torch.Tensor)
        )
        if not finite:
            failures.append("checkpoint contains non-finite model tensors")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 120_108:
        failures.append(f"checkpoint model parameter count {parameter_count} != 120108")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(path),
        "epoch": state.get("epoch"),
        "best_epoch": state.get("best_epoch"),
        "best_score": state.get("best_score"),
        "parameter_count": parameter_count,
        "strict_restore": True,
        "finite_tensors": finite,
    }, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/p1_flatlands_conpath_k128_validation_v2"))
    parser.add_argument(
        "--run-suffix",
        default="conpath",
        help="per-seed directory suffix (default: conpath; use independent for the matched control)",
    )
    parser.add_argument(
        "--decoder-variant",
        choices=("correlated", "independent"),
        default="correlated",
        help="decoder variant required in each run metadata file",
    )
    parser.add_argument("--method", default="conpath_k128_audit")
    parser.add_argument("--selection", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"))
    parser.add_argument("--queries", type=Path, default=Path("results/p1_flatlands_query_audit_bounded/queries.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = PROJECT_ROOT / args.root
    selection = PROJECT_ROOT / args.selection
    queries = PROJECT_ROOT / args.queries
    output = PROJECT_ROOT / args.output if args.output else root / "audit.json"
    failures: list[str] = []
    seed_records: list[dict[str, Any]] = []
    expected_implementation_hashes = {
        "model": _sha256(PROJECT_ROOT / "src/pathrel/model.py"),
        "flatlands_data": _sha256(PROJECT_ROOT / "src/pathrel/flatlands_data.py"),
        "trainer": _sha256(PROJECT_ROOT / "scripts/train_flatlands_conpath.py"),
    }
    if _sha256(selection) != SELECTION_SHA256:
        failures.append("selection SHA-256 does not match frozen contract")
    if _sha256(queries) != QUERIES_SHA256:
        failures.append("queries SHA-256 does not match frozen contract")

    for seed in SEEDS:
        run_dir = root / f"seed{seed}_{args.run_suffix}"
        run_path = run_dir / "run.json"
        report_path = run_dir / "evaluation_validation" / "report.json"
        prediction_path = run_dir / "predictions_validation.csv"
        try:
            run = _read_json(run_path)
            report = _read_json(report_path)
            if run.get("paper_result") is not False or run.get("test_evaluated") is not False:
                failures.append(f"seed {seed}: run flags are not paper_result=false/test_evaluated=false")
            if run.get("validation_result") is not True:
                failures.append(f"seed {seed}: validation_result is not true")
            if run.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
                failures.append(f"seed {seed}: protocol_version is not the valid-support v2 contract")
            config = run.get("config", {})
            expected_config = {
                "seed": seed,
                "device": "cuda",
                "feature_channels": 16,
                "latent_dim": 4,
                "batch_size": 1,
                "max_epochs": 40,
                "patience": 8,
                "min_delta": 1e-5,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "gradient_clip": 5.0,
                "train_samples": 8,
                "validation_samples": 128,
                "validation_sample_chunk": 8,
                "max_reachability_steps": 256,
                "selection_query_limit": 8,
                "map_weight": 1.0,
                "variogram_weight": 0.1,
                "reachability_weight": 2.0,
                "bootstrap_samples": 2000,
                "train_scenes_limit": None,
                "validation_scenes_limit": None,
                "resume": False,
            }
            for key, expected in expected_config.items():
                if not isinstance(config, dict) or config.get(key) != expected:
                    failures.append(f"seed {seed}: config {key} != {expected!r}")
            if not isinstance(config, dict) or config.get("decoder_variant") != args.decoder_variant:
                failures.append(
                    f"seed {seed}: config decoder_variant != {args.decoder_variant!r}"
                )
            expected_independent = args.decoder_variant == "independent"
            if isinstance(config, dict) and bool(config.get("disable_global_factors")):
                # The independent variant is selected by --decoder-variant and the trainer rejects
                # combining it with the explicit ablation flag.  Keep the CLI contract separate
                # from the effective setting recorded in the forward envelope below.
                failures.append(
                    f"seed {seed}: explicit disable_global_factors flag must remain false"
                )
            data = run.get("data", {})
            if not isinstance(data, dict) or data.get("selection_sha256") != SELECTION_SHA256 or data.get("queries_sha256") != QUERIES_SHA256:
                failures.append(f"seed {seed}: run data hashes differ from frozen contract")
            if isinstance(data, dict):
                if data.get("train_scenes") != 160 or data.get("validation_scenes") != 160:
                    failures.append(f"seed {seed}: run scene counts differ from 160/160")
                if data.get("archive_bytes") != 2_054_773_316:
                    failures.append(f"seed {seed}: FlatLands archive byte count mismatch")
            checkpoint, checkpoint_failures = _audit_checkpoint(
                run_dir / "best.pt",
                decoder_variant=args.decoder_variant,
                seed=seed,
            )
            failures.extend(f"seed {seed}: {item}" for item in checkpoint_failures)
            if checkpoint.get("best_epoch") != run.get("selection", {}).get("best_epoch"):
                failures.append(f"seed {seed}: checkpoint/run best epoch mismatch")
            if not _close(
                checkpoint.get("best_score"),
                run.get("selection", {}).get("best_validation_scene_weighted_brier"),
            ):
                failures.append(f"seed {seed}: checkpoint/run best score mismatch")
            rows, groups, prediction_failures = _audit_prediction(prediction_path)
            failures.extend(f"seed {seed}: {item}" for item in prediction_failures)
            actual_hash = _sha256(prediction_path)
            if actual_hash != run.get("prediction", {}).get("sha256"):
                failures.append(f"seed {seed}: prediction hash mismatch in run metadata")
            if run.get("prediction", {}).get("rows") != rows:
                failures.append(f"seed {seed}: prediction row metadata mismatch")
            if report.get("paper_result") is not False:
                failures.append(f"seed {seed}: report is marked as paper result")
            protocol = report.get("protocol", {})
            if not isinstance(protocol, dict) or protocol.get("provenance_split") != "validation":
                failures.append(f"seed {seed}: report provenance split is not validation")
            forward = run.get("forward", {})
            if not isinstance(forward, dict):
                failures.append(f"seed {seed}: missing forward envelope")
            else:
                if forward.get("validation_exact_forward") is not True:
                    failures.append(f"seed {seed}: validation_exact_forward is not true")
                if forward.get("decoder_variant") != args.decoder_variant:
                    failures.append(f"seed {seed}: forward decoder_variant mismatch")
                if bool(forward.get("effective_disable_global_factors")) != expected_independent:
                    failures.append(f"seed {seed}: effective global-factor setting mismatch")
                expected_kernel = 1 if expected_independent else 5
                if forward.get("local_kernel_size") != expected_kernel:
                    failures.append(f"seed {seed}: local kernel size mismatch")
                if forward.get("validation_sample_chunk") != 8:
                    failures.append(f"seed {seed}: validation sample chunk != 8")
                if forward.get("invalid_support_clamped") is not True:
                    failures.append(f"seed {seed}: invalid support is not explicitly clamped")
                if forward.get("valid_support_policy") != EXPECTED_SUPPORT_POLICY:
                    failures.append(f"seed {seed}: valid-support policy mismatch")
                if forward.get("implementation_sha256") != expected_implementation_hashes:
                    failures.append(f"seed {seed}: implementation hash envelope mismatch")
            replay = evaluate_flatlands_prediction_file(prediction_path, selection, queries, method=args.method, split="validation", bootstrap_samples=2000, seed=seed)
            recorded = _overall(report)
            replayed = _overall(replay)
            for key in METRICS:
                if not _close(recorded.get(key), replayed.get(key)):
                    failures.append(f"seed {seed}: replay mismatch for {key}")
            seed_records.append({
                "seed": seed,
                "best_epoch": run.get("selection", {}).get("best_epoch"),
                "epochs_completed": run.get("selection", {}).get("epochs_completed"),
                "selection_brier": run.get("selection", {}).get("best_validation_scene_weighted_brier"),
                "prediction_rows": rows,
                "prediction_sha256": actual_hash,
                "scene_weighted": {key: recorded.get(key) for key in METRICS},
                "radius_monotonicity_failures": sum("radius monotonicity violation" in item for item in prediction_failures),
                "replay_matches": all(_close(recorded.get(key), replayed.get(key)) for key in METRICS),
                "invalid_support_clamped": forward.get("invalid_support_clamped") is True,
                "implementation_hashes_match": forward.get("implementation_sha256")
                == expected_implementation_hashes,
                "checkpoint": checkpoint,
            })
        except Exception as exc:  # report all seeds before exiting
            failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")

    aggregate: dict[str, dict[str, float]] = {}
    for key in METRICS:
        values = [float(record["scene_weighted"][key]) for record in seed_records if record["scene_weighted"].get(key) is not None]
        if len(values) == len(SEEDS):
            aggregate[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    result = {
        "schema_version": 1,
        "kind": f"flatlands_{args.decoder_variant}_k128_validation_audit",
        "validation_only": True,
        "paper_result": False,
        "test_evaluated": False,
        "root": str(args.root),
        "selection": str(args.selection),
        "queries": str(args.queries),
        "run_suffix": args.run_suffix,
        "decoder_variant": args.decoder_variant,
        "seeds": seed_records,
        "aggregate_scene_weighted": aggregate,
        "checks": {
            "selection_hash": _sha256(selection) == SELECTION_SHA256,
            "queries_hash": _sha256(queries) == QUERIES_SHA256,
            "seed_count": len(seed_records) == len(SEEDS),
            "all_replays_match": all(record.get("replay_matches") for record in seed_records),
            "all_prediction_rows": all(record.get("prediction_rows") == EXPECTED_ROWS for record in seed_records),
            "all_invalid_support_clamped": all(record.get("invalid_support_clamped") is True for record in seed_records),
            "all_implementation_hashes_match": all(
                record.get("implementation_hashes_match") is True for record in seed_records
            ),
            "all_checkpoints_restore_strictly": all(
                record.get("checkpoint", {}).get("strict_restore") is True
                for record in seed_records
            ),
            "all_checkpoint_tensors_finite": all(
                record.get("checkpoint", {}).get("finite_tensors") is True
                for record in seed_records
            ),
            "radius_monotonicity_violations": sum(record.get("radius_monotonicity_failures", 0) for record in seed_records),
            "label_scope": "validation evaluator only",
        },
        "passed": not failures,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(args.output or output.relative_to(PROJECT_ROOT)), "passed": not failures, "failure_count": len(failures), "seed_count": len(seed_records)}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
