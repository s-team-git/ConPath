"""Frozen model factories and label-free inference for the formal BEV matrix."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch

from .coherent_categorical import CoherentCategoricalDecoder
from .external_completion import validate_condition
from .flatlands_baselines import DirectQueryBaseline, MarginalCompletionBaseline
from .flow_matching import FlowUNet, sample_heun
from .formal_metrics import exact_world_events
from .lama import LaMaBEV, make_discriminator
from .model import PathRelNet


METHODS = ("lama", "flow", "conpath", "conpath_original", "independent", "no_reach", "deterministic", "direct_query")
RADII = (0, 10, 20)
LABELS = {"lama": "LaMa BEV adaptation", "flow": "FM+XAttn literature reimplementation",
          "conpath": "ConPath coherent categorical", "conpath_original": "ConPath original categorical",
          "independent": "Independent completion", "no_reach": "ConPath without reachability loss",
          "deterministic": "Deterministic completion control", "direct_query": "Direct-query control"}


def make_models(method, device="cuda"):
    """Construct exactly the registered architecture, with no pretrained weights."""
    if method not in METHODS:
        raise ValueError("Unknown formal method: " + str(method))
    if method == "lama":
        models = {"model": LaMaBEV(), "discriminator": make_discriminator()}
    elif method == "flow":
        models = {"model": FlowUNet(base_channels=64, heads=4, activation_checkpointing=True)}
    elif method == "direct_query":
        models = {"model": DirectQueryBaseline(feature_channels=16)}
    elif method == "deterministic":
        models = {"model": MarginalCompletionBaseline(feature_channels=16)}
    else:
        model = PathRelNet(feature_channels=16, latent_dim=4, local_kernel_size=1 if method == "independent" else 5)
        if method in ("conpath", "no_reach"):
            # Match the accepted candidate initialization, including the RNG
            # consumed by the original model but not by decoder replacement.
            saved_rng = torch.get_rng_state()
            decoder = CoherentCategoricalDecoder(16, latent_dim=4, local_kernel_size=5, gumbel_kernel_size=9)
            incompatible = decoder.load_state_dict(model.decoder.state_dict(), strict=False)
            if incompatible.missing_keys != ["gumbel_kernel"] or incompatible.unexpected_keys:
                raise ValueError("Candidate decoder initialization drift")
            model.decoder = decoder
            torch.set_rng_state(saved_rng)
        models = {"model": model}
    return {key: model.to(device) for key, model in models.items()}


def observation_from_condition(condition):
    """Lossless [free,unknown,support] -> [free,blocked,unknown] adapter."""
    validate_condition(condition)
    free, unknown, support = condition.split(1, dim=1)
    return torch.cat((free, support - free - unknown, unknown), dim=1)


def direct_query_logits(model, observation, starts, goals, radii_cells=RADII):
    """Geometry is in cells; the legacy argument name does not denote metres."""
    delta = goals.to(observation.dtype) - starts.to(observation.dtype)
    distance_cells = torch.linalg.vector_norm(delta, dim=-1)
    # The unchanged control internally divides its legacy distances_m argument
    # by 1.2. Supplying 1.2*d_cells/120 makes its feature exactly d_cells/120.
    dimensionless_legacy_adapter = 1.2 * distance_cells / 120.0
    angle_degrees = torch.rad2deg(torch.atan2(delta[..., 0], delta[..., 1]))
    radii = torch.as_tensor(radii_cells, device=observation.device, dtype=observation.dtype)
    return model(observation, starts, goals, dimensionless_legacy_adapter, angle_degrees, radii)


# Public trainer/evaluator adapter name, shared to avoid geometry drift.
direct_forward = direct_query_logits


def sample_seed(seed, sample):
    """Repeatable input-only stream; independent of method order and targets."""
    key = {"seed": int(seed), "global_id": sample.row["global_id"],
           "parent_group": sample.row["parent_group"], "starts": sample.starts.tolist(),
           "goals": sample.goals.tolist(), "candidate_indices": sample.candidate_indices.tolist(), "radii_cells": RADII}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1), digest


@torch.inference_mode()
def sample_models(models_or_list, method, sample, seed, k=4, *, one_member=False, solver_steps=25, guidance=2.0):
    """Generate from input-only fields; never access sample.target/targets.

    LaMa needs four independently trained model dictionaries for a formal row.
    one_member=True is an explicitly K=1 stage diagnostic, not an ensemble.
    Deterministic/direct controls retain actual K=1/N/A. Efficiency includes
    device transfer and exact event calculation, excludes loading weights/data.
    """
    if method not in METHODS or k != 4:
        raise ValueError("The common formal output budget is exactly four")
    members = list(models_or_list) if isinstance(models_or_list, (tuple, list)) else [models_or_list]
    expected = 1 if method != "lama" or one_member else 4
    if len(members) != expected or any("model" not in m for m in members):
        raise ValueError("Actual independently loaded member count does not match the protocol")
    if one_member and method != "lama":
        raise ValueError("one_member is only a LaMa stage diagnostic")
    device = next(members[0]["model"].parameters()).device
    if any(next(m["model"].parameters()).device != device for m in members):
        raise ValueError("All inference members must be on the same device")
    for member in members:
        member["model"].eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    condition = torch.as_tensor(np.asarray(sample.condition)[None], device=device, dtype=torch.float32)
    observation = observation_from_condition(condition)
    support = condition[:, 2].bool()
    if not np.array_equal(observation[0].cpu().numpy(), sample.observation):
        raise ValueError("Lossless common input adaptation mismatch")
    actual_seed, input_key = sample_seed(seed, sample)
    generator = torch.Generator(device=device).manual_seed(actual_seed)
    counts = {"actual_batch_forward_calls": 0, "actual_model_input_examples": 0}

    def count_forward(_model, inputs, _output):
        counts["actual_batch_forward_calls"] += 1
        counts["actual_model_input_examples"] += int(inputs[0].shape[0])

    hooks = [m["model"].register_forward_hook(count_forward) for m in members]
    native_semantics, sampler = None, {"solver": None, "velocity_evaluations_per_sample": None}
    sync(); generation_started = time.perf_counter()
    try:
        model = members[0]["model"]
        if method == "lama":
            continuous = torch.cat([m["model"](condition)[:, 0] for m in members], dim=0).cpu().numpy()
            worlds = continuous > .5
            native_semantics = "mean continuous BEV generator completion score; not guaranteed occupancy probability"
        elif method == "flow":
            p, w, sampler = sample_heun(model, condition, samples=k, steps=solver_steps, guidance=guidance, seed=actual_seed)
            continuous, worlds = p[0].cpu().numpy(), w[0].cpu().numpy()
            native_semantics = "mean clipped continuous flow completion value; not guaranteed occupancy probability"
        elif method == "deterministic":
            raw = model(observation).sigmoid()
            p = (raw * condition[:, 1] + condition[:, 0]) * condition[:, 2]
            continuous = p.cpu().numpy()
            worlds = continuous > .5
            native_semantics = "sigmoid marginal-completion model score"
        elif method == "direct_query":
            starts = torch.as_tensor(sample.starts[None], device=device, dtype=torch.long)
            goals = torch.as_tensor(sample.goals[None], device=device, dtype=torch.long)
            event_scores = direct_query_logits(model, observation, starts, goals).sigmoid()[0].cpu().numpy()
            worlds, continuous = None, None
        else:
            posterior = model(observation, valid_support_mask=support, num_samples=k,
                              disable_global_factors=method == "independent", generator=generator).posterior
            worlds = posterior.safe_samples()[0].cpu().numpy() > .5
            raw = posterior.sample_logits.softmax(2)[0, :, 0]
            continuous = ((raw * condition[0, 1] + condition[0, 0]) * condition[0, 2]).cpu().numpy()
            native_semantics = "mean categorical conditional free probability across sampled latent factors"
        sync()
    finally:
        for hook in hooks:
            hook.remove()
    generation_seconds = time.perf_counter() - generation_started
    if worlds is not None:
        events = exact_world_events(worlds, sample.starts, sample.goals, RADII)
        event_scores = events.mean(0)
    else:
        events = None
    full_update_seconds = time.perf_counter() - began
    return {"worlds": worlds, "continuous_score": continuous,
            "continuous_score_semantics": native_semantics, "world_events": events,
            "event_scores": event_scores, "event_score_semantics": "direct_query_sigmoid" if method == "direct_query" else "raw_world_event_frequency",
            "sampling_seed": actual_seed, "input_key_sha256": input_key,
            "efficiency": {**counts, **sampler, "generation_seconds": generation_seconds,
                           "full_update_seconds": full_update_seconds,
                           "generation_scope": "one observation; actual output bundle, projection/binarization and transfer to CPU",
                           "full_update_scope": "input transfer plus generation plus exact CPU reachability; excludes disk and checkpoint loading",
                           "actual_world_count": len(worlds) if worlds is not None else None,
                           "actual_members": len(members), "common_budget_K": k,
                           "stage_one_member_diagnostic": bool(one_member),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}}
