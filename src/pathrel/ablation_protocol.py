"""Configuration and resume guards for the frozen clean FlatLands ablations."""

from __future__ import annotations

from pathlib import Path

VARIANTS = {
    "no_event": {"reachability_weight": 0.0},
    "no_global": {"disable_global_factors": True},
}


def ablation_config(reference: dict, variant: str, output_dir: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown ablation: {variant}")
    if reference.get("reachability_weight") != 2.0 or reference.get("disable_global_factors") is not False:
        raise ValueError("the reference must be the full model")
    if reference.get("decoder_variant") != "correlated":
        raise ValueError("the reference must use the correlated decoder")
    return {**reference, **VARIANTS[variant], "output_dir": output_dir, "resume": False}


def validate_config(actual: dict, expected: dict) -> None:
    """Allow only the operational resume flag to differ from the frozen config."""
    def normalize(config):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in config.items() if key != "resume"}
    actual, expected = normalize(actual), normalize(expected)
    changed = sorted(key for key in actual.keys() | expected.keys()
                     if key not in actual or key not in expected or actual[key] != expected[key])
    if changed:
        raise ValueError(f"frozen ablation configuration changed: {changed}")


def checkpoint_action(state: dict, expected: dict) -> str:
    validate_config(state["config"], expected)
    if not state.get("history") or int(state["epoch"]) != int(state["history"][-1]["epoch"]):
        raise ValueError("checkpoint epoch/history mismatch")
    if int(state["epoch"]) >= expected["max_epochs"] or int(state["patience_used"]) >= expected["patience"]:
        return "finalize"
    return "resume"
