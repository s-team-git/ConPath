"""Pinned CogniPlan completion module; inference takes observations only.

Native definitions are compiled unchanged from the bundled MIT-licensed source.
Optional upstream visualization/CLI imports are excluded, not numerical code.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "third_party/cogniplan"
NATIVE_CONDITIONS = ((0.333, 0.333, 0.333), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def configure_reproducible_cuda():
    """Freeze arithmetic/algorithm choices for cross-process scientific replay."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def load_native():
    """Verify every source byte before importing official numerical definitions."""
    for row in json.loads((SOURCE_ROOT / "sources.json").read_text()):
        if hashlib.sha256((SOURCE_ROOT / row["name"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Modified pinned CogniPlan source: {row['name']}")
    path = SOURCE_ROOT / "mapinpaint/networks.py"
    spec = importlib.util.spec_from_file_location("conpath_pinned_cogniplan_networks", path)
    networks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(networks)
    scope = {"torch": torch, "nn": nn, "F": F, "np": np, "cv2": cv2,
             "autograd": torch.autograd, "distance_transform_edt": distance_transform_edt,
             "Generator": networks.Generator, "Discriminator": networks.Discriminator}
    for name, definitions in (("tools.py", {"spatial_discounting_mask"}),
                              ("trainer.py", {"Trainer"}),
                              ("evaluator.py", {"Evaluator", "calc_similarity"})):
        path = SOURCE_ROOT / "mapinpaint" / name
        parsed = ast.parse(path.read_text())
        body = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in definitions]
        if {node.name for node in body} != definitions:
            raise ValueError(f"Missing native definitions in {name}")
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


def encode_partial(partial: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Native grayscale normalization and mask, center edge padding to 256."""
    partial = np.asarray(partial)
    if partial.shape != (250, 250) or not np.isin(partial, (1, 127, 255)).all():
        raise ValueError("Expected a native 250x250 partial map with values 1/127/255")
    x = torch.from_numpy(partial.astype(np.float32))[None, None] / 255.
    x = x.mul_(2).add_(-1)
    mask = torch.from_numpy((partial == 127).astype(np.float32))[None, None]
    return F.pad(x, (3, 3, 3, 3), mode="replicate"), F.pad(mask, (3, 3, 3, 3), mode="replicate")


def encode_target(target: np.ndarray) -> torch.Tensor:
    target = np.asarray(target)
    if target.shape != (250, 250) or not np.isin(target, (127, 208, 255)).all():
        raise ValueError("Expected native target values 127/208/255")
    pixels = np.where(target == 127, 1, 255).astype(np.float32)
    tensor = (torch.from_numpy(pixels)[None, None] / 255.).mul_(2).add_(-1)
    return F.pad(tensor, (3, 3, 3, 3), mode="replicate")


def postprocess(raw: np.ndarray, partial: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Native threshold/close/open, restore known cells, then close invalid cells.

    Explicit semantic values also handle observations missing one or more classes;
    upstream top-three-value inference is ambiguous on those boundary inputs.
    Native valid three-class observations are required to replay exactly.
    """
    raw, partial, support = np.asarray(raw), np.asarray(partial), np.asarray(support, dtype=bool)
    if raw.shape != partial.shape or support.shape != partial.shape or not np.isfinite(raw).all():
        raise ValueError("Invalid raw output, partial map or support shape")
    if not np.isin(partial, (1, 127, 255)).all():
        raise ValueError("Invalid partial-map encoding")
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = (raw > -0.3).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return np.where(partial == 127, binary == 255, partial == 255) & support


@torch.no_grad()
def predict_four(generator, partial: np.ndarray, support: np.ndarray | None = None):
    """Four native layout hypotheses; no target or true layout argument."""
    device = next(generator.parameters()).device
    x, mask = (v.to(device) for v in encode_partial(partial))
    if support is None:
        support = np.ones_like(partial, dtype=bool)
    generator.eval()
    raw, worlds = [], []
    for condition in NATIVE_CONDITIONS:
        vector = torch.tensor([condition], dtype=x.dtype, device=device)
        output = generator(x, mask, vector) * mask + x * (1 - mask)
        array = output[0, 0, 3:253, 3:253].cpu().numpy()
        raw.append(array)
        worlds.append(postprocess(array, partial, support))
    return np.stack(raw), np.stack(worlds)


def native_update(trainer, batch, *, adversarial: bool, generator_update: bool):
    """Official train.py update order; no loss caching or reduced precision."""
    x, mask, target, onehot = batch
    config = trainer.config
    compute_g = not adversarial or generator_update
    if not adversarial:
        trainer.optimizer_g.zero_grad()
    losses, _ = trainer(x, mask, target, onehot, compute_g)
    if adversarial:
        trainer.optimizer_d.zero_grad()
        losses["d"] = losses["wgan_d"] + losses["wgan_gp"] * config["wgan_gp_lambda"]
        losses["d"].backward()
    if compute_g:
        if adversarial:
            trainer.optimizer_g.zero_grad()
        losses["g"] = (losses["ae"] * config["ae_loss_alpha"]
                       + losses["l1"] * config["l1_loss_alpha"]
                       + losses["f1"] * config["f1_loss_alpha"])
        if adversarial:
            losses["g"] = losses["g"] + losses["wgan_g"] * config["gan_loss_alpha"]
        losses["g"].backward()
        trainer.optimizer_g.step()
    if adversarial:
        trainer.optimizer_d.step()
    values = {key: float(value.detach()) for key, value in losses.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError(values)
    return values
