#!/usr/bin/env python3
"""Audit three clean, validation-only UnScenes3D ConPath adapter runs.

The audit never loads a frame or label.  It verifies the frozen manifest envelope, training
configuration, support-boundary contract, reported validation metrics, and a strict CPU restore
of every selected checkpoint.  The physical ``location_6`` test site therefore stays locked.
"""

from __future__ import annotations

import argparse
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

from pathrel.model import PathRelNet


SEEDS = (20260831, 20260901, 20260902)
EXPECTED_PROTOCOL = "UNSCENES3D_PROTOCOL.md v0.2"
EXPECTED_MANIFEST = Path("results/unscenes3d_contract_manifest_ground_valid/manifest.json")
EXPECTED_MANIFEST_SHA256 = "d866be0b25ba6635d5ad934bac9c822c55be8034fabd8f4627803bb9efa972d5"
EXPECTED_SUPPORT_POLICY = (
    "UnScenes3D target_valid complement is deterministically blocked before posterior sampling"
)
EXPECTED_ADAPTER = {
    "endpoint_policy": "ground",
    "start_selection": "valid",
    "ground_margin_m": 0.35,
    "ground_bin_size_m": 1.2,
    "ground_lateral_limit_m": 20.0,
    "ground_quantile": 0.15,
}
METRICS = (
    "scene_weighted_brier",
    "query_weighted_nll",
    "query_weighted_ece",
    "false_safe_rate@0.8",
    "high_confidence_coverage@0.8",
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


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _audit_checkpoint(
    path: Path,
    *,
    decoder_variant: str,
    seed: int,
    best_epoch: int,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint payload is not an object: {path}")
    config = state.get("config")
    if not isinstance(config, dict):
        failures.append("checkpoint config is missing")
        config = {}
    if config.get("seed") != seed:
        failures.append("checkpoint seed mismatch")
    if config.get("decoder_variant") != decoder_variant:
        failures.append("checkpoint decoder variant mismatch")
    if config.get("valid_support_policy") != EXPECTED_SUPPORT_POLICY:
        failures.append("checkpoint valid-support policy mismatch")
    if str(config.get("manifest")) != str(EXPECTED_MANIFEST):
        failures.append("checkpoint manifest path mismatch")
    if config.get("adapter") != EXPECTED_ADAPTER:
        failures.append("checkpoint adapter contract mismatch")

    model = PathRelNet(
        input_channels=3,
        feature_channels=int(config.get("feature_channels", 16)),
        latent_dim=int(config.get("latent_dim", 4)),
        local_kernel_size=1 if decoder_variant == "independent" else 5,
    )
    model_state = state.get("model")
    finite = False
    strict_restore = False
    if not isinstance(model_state, dict):
        failures.append("checkpoint model state is missing")
    else:
        model.load_state_dict(model_state, strict=True)
        strict_restore = True
        finite = all(
            not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
            for value in model_state.values()
            if isinstance(value, torch.Tensor)
        )
        if not finite:
            failures.append("checkpoint contains non-finite model tensors")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 120_108:
        failures.append(f"checkpoint parameter count {parameter_count} != 120108")
    history = state.get("history")
    if not isinstance(history, list) or not history:
        failures.append("checkpoint history is empty")
        checkpoint_epoch = None
    else:
        checkpoint_epoch = history[-1].get("epoch") if isinstance(history[-1], dict) else None
        if checkpoint_epoch != best_epoch:
            failures.append("selected checkpoint does not end at the reported best epoch")
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "parameter_count": parameter_count,
        "checkpoint_epoch": checkpoint_epoch,
        "strict_restore": strict_restore,
        "finite_tensors": finite,
    }, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decoder-variant", choices=("correlated", "independent"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    output = (PROJECT_ROOT / args.output).resolve() if args.output else root / "audit.json"
    manifest_path = PROJECT_ROOT / EXPECTED_MANIFEST
    failures: list[str] = []
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        failures.append("frozen manifest SHA-256 mismatch")
    expected_implementation_hashes = {
        "model": _sha256(PROJECT_ROOT / "src/pathrel/model.py"),
        "unscenes3d": _sha256(PROJECT_ROOT / "src/pathrel/unscenes3d.py"),
        "trainer": _sha256(PROJECT_ROOT / "scripts/train_unscenes3d_conpath.py"),
    }

    expected_config = {
        "device": "cuda",
        "feature_channels": 16,
        "latent_dim": 4,
        "batch_size": 4,
        "max_epochs": 3,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "train_samples": 4,
        "max_reachability_steps": 64,
        "map_weight": 1.0,
        "variogram_weight": 0.1,
        "reachability_weight": 0.0,
        "posterior_samples": 8,
        "train_frame_limit": None,
        "validation_frame_limit": None,
    }
    seed_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = root / f"seed{seed}"
        try:
            run = _read_json(run_dir / "run.json")
            prefix = f"seed {seed}: "
            if run.get("kind") != "unscenes3d_conpath_adapter_smoke":
                failures.append(prefix + "unexpected run kind")
            if (
                run.get("paper_result") is not False
                or run.get("validation_result") is not True
                or run.get("test_evaluated") is not False
            ):
                failures.append(prefix + "paper/validation/test flags violate the contract")
            if run.get("protocol_version") != EXPECTED_PROTOCOL:
                failures.append(prefix + "protocol version mismatch")
            config = run.get("config")
            if not isinstance(config, dict):
                failures.append(prefix + "run config is missing")
                config = {}
            if config.get("seed") != seed:
                failures.append(prefix + "config seed mismatch")
            if config.get("decoder_variant") != args.decoder_variant:
                failures.append(prefix + "config decoder variant mismatch")
            if str(config.get("manifest")) != str(EXPECTED_MANIFEST):
                failures.append(prefix + "config manifest path mismatch")
            if config.get("adapter") != EXPECTED_ADAPTER:
                failures.append(prefix + "config adapter contract mismatch")
            for key, expected in expected_config.items():
                if config.get(key) != expected:
                    failures.append(prefix + f"config {key} != {expected!r}")

            data = run.get("data")
            envelope = run.get("manifest")
            if not isinstance(data, dict) or data.get("train_records") != 478 or data.get("validation_records") != 62:
                failures.append(prefix + "train/validation record counts differ from 478/62")
            if not isinstance(envelope, dict):
                failures.append(prefix + "manifest envelope is missing")
                envelope = {}
            if (
                envelope.get("sha256") != EXPECTED_MANIFEST_SHA256
                or envelope.get("validation_queries") != 1529
                or envelope.get("test_locked_sites") != ["location_6"]
            ):
                failures.append(prefix + "manifest hash/query/test-lock envelope mismatch")

            forward = run.get("forward")
            if not isinstance(forward, dict):
                failures.append(prefix + "forward envelope is missing")
                forward = {}
            if forward.get("invalid_support_clamped") is not True:
                failures.append(prefix + "invalid support is not clamped")
            if forward.get("valid_support_policy") != EXPECTED_SUPPORT_POLICY:
                failures.append(prefix + "valid-support policy mismatch")
            if forward.get("decoder_variant") != args.decoder_variant:
                failures.append(prefix + "forward decoder variant mismatch")
            if forward.get("implementation_sha256") != expected_implementation_hashes:
                failures.append(prefix + "implementation hash envelope mismatch")

            history = run.get("history")
            if not isinstance(history, list) or len(history) != 3:
                failures.append(prefix + "expected exactly three training epochs")
                history = []
            for epoch_index, record in enumerate(history, 1):
                if not isinstance(record, dict) or record.get("epoch") != epoch_index:
                    failures.append(prefix + f"malformed history record {epoch_index}")
                    continue
                for key in ("train_loss", "scene_weighted_brier", "scene_weighted_nll", "selection_value"):
                    if not _finite_number(record.get(key)):
                        failures.append(prefix + f"non-finite history value {key} at epoch {epoch_index}")
            selection = run.get("selection")
            if not isinstance(selection, dict):
                failures.append(prefix + "selection envelope is missing")
                selection = {}
            best_epoch = selection.get("best_epoch")
            if best_epoch not in (1, 2, 3):
                failures.append(prefix + "best epoch is outside [1,3]")
                best_epoch = 1
            if selection.get("criterion") != "validation hidden-valid-cell NLL":
                failures.append(prefix + "selection criterion is not map NLL")

            metrics = run.get("validation_event_metrics")
            if not isinstance(metrics, dict):
                failures.append(prefix + "validation event metrics are missing")
                metrics = {}
            if metrics.get("query_count") != 1529 or metrics.get("scene_count") != 2:
                failures.append(prefix + "validation event coverage mismatch")
            if metrics.get("radius_monotonicity_violations") != 0:
                failures.append(prefix + "radius monotonicity violations are nonzero")
            for key in METRICS:
                if not _finite_number(metrics.get(key)):
                    failures.append(prefix + f"non-finite validation metric {key}")

            checkpoint, checkpoint_failures = _audit_checkpoint(
                run_dir / "best.pt",
                decoder_variant=args.decoder_variant,
                seed=seed,
                best_epoch=int(best_epoch),
            )
            failures.extend(prefix + item for item in checkpoint_failures)
            seed_records.append(
                {
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_event_metrics": {key: metrics.get(key) for key in METRICS},
                    "invalid_support_clamped": forward.get("invalid_support_clamped") is True,
                    "implementation_hashes_match": forward.get("implementation_sha256")
                    == expected_implementation_hashes,
                    "checkpoint": checkpoint,
                }
            )
        except Exception as exc:
            failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")

    aggregate: dict[str, dict[str, float]] = {}
    for key in METRICS:
        values = [
            float(record["validation_event_metrics"][key])
            for record in seed_records
            if _finite_number(record["validation_event_metrics"].get(key))
        ]
        if len(values) == len(SEEDS):
            aggregate[key] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
    result = {
        "schema_version": 1,
        "kind": f"unscenes3d_{args.decoder_variant}_clean_support_training_audit",
        "validation_only": True,
        "paper_result": False,
        "test_evaluated": False,
        "root": _relative(root),
        "decoder_variant": args.decoder_variant,
        "manifest": {
            "path": str(EXPECTED_MANIFEST),
            "sha256": manifest_hash,
            "test_locked_sites": ["location_6"],
        },
        "seeds": seed_records,
        "aggregate_validation_event_metrics": aggregate,
        "checks": {
            "manifest_hash": manifest_hash == EXPECTED_MANIFEST_SHA256,
            "seed_count": len(seed_records) == len(SEEDS),
            "all_invalid_support_clamped": all(
                record.get("invalid_support_clamped") is True for record in seed_records
            ),
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
            "test_site": "location_6 locked; audit loads no frames or labels",
        },
        "passed": not failures,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": _relative(output),
                "passed": not failures,
                "failure_count": len(failures),
                "seed_count": len(seed_records),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
