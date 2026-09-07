#!/usr/bin/env python3
"""Compare a canonical UnScenes3D run with a fresh v0.2 replay.

The comparison is a deterministic-contract audit, not a new benchmark.  It verifies
that both runs use the same frozen manifest/adapter and that a fresh GPU replay does
not materially change the validation event metrics.  It never opens data files and
never enumerates the locked test site.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "scene_weighted_brier",
    "query_weighted_nll",
    "query_weighted_ece",
    "false_safe_rate@0.8",
    "high_confidence_coverage@0.8",
    "mean_prediction",
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=Path("results/unscenes3d_ground_valid_f16"))
    parser.add_argument("--replay-root", type=Path, default=Path("results/unscenes3d_ground_valid_f16_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/unscenes3d_replay_audit/report.json"))
    args = parser.parse_args()
    canonical_root = PROJECT_ROOT / args.canonical_root
    replay_root = PROJECT_ROOT / args.replay_root
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for seed in (20260831, 20260901, 20260902):
        canonical_path = canonical_root / f"seed{seed}" / "run.json"
        replay_path = replay_root / f"seed{seed}" / "run.json"
        canonical = _read(canonical_path)
        replay = _read(replay_path)
        c_metrics = canonical.get("validation_event_metrics", {})
        r_metrics = replay.get("validation_event_metrics", {})
        if not isinstance(c_metrics, dict) or not isinstance(r_metrics, dict):
            failures.append(f"seed {seed}: missing event metrics")
            continue
        c_config = canonical.get("config", {})
        r_config = replay.get("config", {})
        c_adapter = c_config.get("adapter") if isinstance(c_config, dict) else None
        r_adapter = r_config.get("adapter") if isinstance(r_config, dict) else None
        contract_equal = (
            c_config.get("manifest") == r_config.get("manifest")
            and c_adapter == r_adapter
            and c_config.get("feature_channels") == r_config.get("feature_channels")
            and c_config.get("latent_dim") == r_config.get("latent_dim")
            and c_config.get("max_epochs") == r_config.get("max_epochs")
            and c_config.get("reachability_weight") == r_config.get("reachability_weight")
        ) if isinstance(c_config, dict) and isinstance(r_config, dict) else False
        if not contract_equal:
            failures.append(f"seed {seed}: config/adapter contract differs")
        if canonical.get("test_evaluated") is not False or replay.get("test_evaluated") is not False:
            failures.append(f"seed {seed}: test_evaluated is not false")
        deltas: dict[str, float] = {}
        for key in METRIC_KEYS:
            try:
                delta = float(r_metrics[key]) - float(c_metrics[key])
            except (KeyError, TypeError, ValueError):
                failures.append(f"seed {seed}: missing metric {key}")
                continue
            deltas[key] = delta
        records.append(
            {
                "seed": seed,
                "canonical_protocol": canonical.get("protocol_version"),
                "replay_protocol": replay.get("protocol_version"),
                "contract_equal": contract_equal,
                "canonical_metrics": {key: c_metrics.get(key) for key in METRIC_KEYS},
                "replay_metrics": {key: r_metrics.get(key) for key in METRIC_KEYS},
                "deltas_replay_minus_canonical": deltas,
                "max_abs_delta": max((abs(value) for value in deltas.values()), default=0.0),
            }
        )
    max_abs = max((float(row["max_abs_delta"]) for row in records), default=float("inf"))
    # GPU kernels can differ in the last few ulps between processes.  This bound is
    # deliberately much smaller than any displayed table precision.
    if max_abs > 1e-3:
        failures.append(f"replay metric drift exceeds 1e-3: {max_abs}")
    report = {
        "schema_version": 1,
        "kind": "unscenes3d_replay_contract_audit",
        "validation_only": True,
        "test_evaluated": False,
        "canonical_root": str(args.canonical_root),
        "replay_root": str(args.replay_root),
        "records": records,
        "max_abs_metric_delta": max_abs,
        "passed": not failures,
        "failures": failures,
        "claim_boundary": "GPU replay stability check only; no test labels or cross-domain claim.",
    }
    _atomic_write(PROJECT_ROOT / args.output, report)
    print(json.dumps({"output": str(args.output), "passed": not failures, "max_abs_metric_delta": max_abs, "failure_count": len(failures)}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
