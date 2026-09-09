"""Pinned official LaMa Fourier backbone with a documented FlatLands BEV loss adapter."""
from __future__ import annotations

import abc
import ast
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .external_completion import clamp_completion, masked_mean

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "third_party/lama"


def load_native_lama():
    for row in json.loads((SOURCE_ROOT / "sources.json").read_text()):
        if hashlib.sha256((SOURCE_ROOT / row["name"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Modified pinned LaMa source: {row['name']}")
    scope = {"torch": torch, "nn": nn, "F": F, "np": np, "abc": abc, "Tuple": Tuple, "List": List}
    definitions = [("modules/base.py", {"BaseDiscriminator", "get_activation"}),
                   ("modules/squeeze_excitation.py", {"SELayer"}),
                   ("modules/ffc.py", None), ("modules/pix2pixhd.py", {"NLayerDiscriminator"}),
                   ("losses/feature_matching.py", {"masked_l1_loss", "feature_matching_loss"})]
    for filename, names in definitions:
        path = SOURCE_ROOT / "saicinpainting/training" / filename
        body = [node for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (names is None or node.name in names)]
        if names and {node.name for node in body} != names:
            raise ValueError(f"Missing native definitions in {filename}")
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


class LaMaBEV(nn.Module):
    """Big-LaMa defaults: 64 channels, 18 FFC blocks, .75 global ratio, no LFU.

    Input adaptation adds shared support to paper [Fobs,U], preserving the complete
    3-state/support information given to all competing models. No pretrained weights.
    """
    def __init__(self, *, ngf=64, n_blocks=18):
        super().__init__()
        native = load_native_lama()
        self.generator = native["FFCResNetGenerator"](
            input_nc=3, output_nc=1, ngf=ngf, n_downsampling=3, n_blocks=n_blocks,
            add_out_act="sigmoid",
            init_conv_kwargs={"ratio_gin": 0., "ratio_gout": 0., "enable_lfu": False},
            downsample_conv_kwargs={"ratio_gin": 0., "ratio_gout": 0., "enable_lfu": False},
            resnet_conv_kwargs={"ratio_gin": .75, "ratio_gout": .75, "enable_lfu": False})

    def forward(self, condition):
        return clamp_completion(self.generator(condition), condition)


def make_discriminator(ndf=64):
    return load_native_lama()["NLayerDiscriminator"](input_nc=1, ndf=ndf, n_layers=4)


def hinge_discriminator_loss(real_scores, fake_scores, mask):
    # Patch-level weights record the fraction of the supervised region at this scale.
    weights = F.adaptive_avg_pool2d(mask, real_scores.shape[-2:])
    return .5 * (masked_mean(F.relu(1 - real_scores), weights)
                 + masked_mean(F.relu(1 + fake_scores), weights))


def generator_losses(prediction, target, condition, scores, fake_features, real_features):
    mask = condition[:, 1:2]
    reconstruction = masked_mean((prediction - target).abs(), mask)
    patch_mask = F.adaptive_avg_pool2d(mask, scores.shape[-2:])
    adversarial = -masked_mean(scores, patch_mask)
    feature_matching = torch.stack([
        masked_mean((a - b.detach()).square(), F.adaptive_avg_pool2d(mask, a.shape[-2:]))
        for a, b in zip(fake_features, real_features)]).mean()
    # Paper S8.2: no RGB perceptual losses; reconstruction / hinge / feature MSE.
    total = 10 * reconstruction + 10 * adversarial + 250 * feature_matching
    return total, {"reconstruction": reconstruction, "adversarial": adversarial,
                   "feature_matching": feature_matching}


def lama_update(generator, discriminator, optim_g, optim_d, condition, target):
    """One documented G-then-D update; shared known/support projection on both maps."""
    generator.train(); discriminator.train()
    target = clamp_completion(target, condition)
    optim_g.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    predicted = generator(condition)
    real_scores, real_features = discriminator(target)
    fake_scores, fake_features = discriminator(predicted)
    loss_g, terms = generator_losses(predicted, target, condition, fake_scores, fake_features, real_features)
    loss_g.backward()
    optim_g.step()
    discriminator.requires_grad_(True)
    optim_d.zero_grad(set_to_none=True)
    real_scores, _ = discriminator(target)
    fake_scores, _ = discriminator(predicted.detach())
    loss_d = hinge_discriminator_loss(real_scores, fake_scores, condition[:, 1:2])
    loss_d.backward()
    optim_d.step()
    return {"generator": float(loss_g.detach()), "discriminator": float(loss_d.detach()),
            **{k: float(v.detach()) for k, v in terms.items()}}
