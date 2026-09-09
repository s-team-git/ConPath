"""Documented FlatLands-style conditional flow matching with spatial cross-attention.

Literature reimplementation: velocity MSE, OT linear interpolant, CFG s=2,
Heun-25, per-substep evidence/support projection. Width, head count and exact
U-Net block layout are declared implementation choices, not unreleased author code.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .external_completion import clamp_completion, masked_mean


def time_embedding(t, width):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=t.device) / (width // 2))
    phase = t[:, None] * frequencies[None] * 1000
    return torch.cat((phase.cos(), phase.sin()), dim=1)


class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(F.silu(t))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.GroupNorm(min(32, channels), channels)
        self.norm_c = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Conv2d(channels, channels * 2, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, condition):
        b, c, h, w = x.shape
        # Fixed 2D sinusoidal positions preserve the spatial meaning of condition
        # tokens; this is an explicitly documented reimplementation choice.
        frequencies = torch.exp(-math.log(10000) * torch.arange(c // 4, device=x.device) / (c // 4))
        yy = torch.arange(h, device=x.device, dtype=x.dtype)[:, None] * frequencies[None]
        xx = torch.arange(w, device=x.device, dtype=x.dtype)[:, None] * frequencies[None]
        py = torch.cat((yy.sin(), yy.cos()), dim=1).T[:, :, None].expand(-1, h, w)
        px = torch.cat((xx.sin(), xx.cos()), dim=1).T[:, None, :].expand(-1, h, w)
        position = torch.cat((py, px))[None]
        q = self.q(self.norm_q(x) + position).reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        k, v = self.kv(self.norm_c(condition) + position).chunk(2, dim=1)
        k = k.reshape(b, self.heads, c // self.heads, -1).transpose(-1, -2)
        v = v.reshape(b, self.heads, c // self.heads, -1).transpose(-1, -2)
        # Explicit FP32 math backend; full spatial tokens, no approximate/window attention.
        with sdpa_kernel(SDPBackend.MATH):
            attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(attention)


class FlowUNet(nn.Module):
    def __init__(self, base_channels=64, heads=4, activation_checkpointing=True):
        super().__init__()
        self.base_channels = base_channels
        self.activation_checkpointing = activation_checkpointing
        channels = [base_channels * m for m in (1, 2, 4, 4)]
        time_channels = base_channels * 4
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, time_channels), nn.SiLU(), nn.Linear(time_channels, time_channels))
        self.input = nn.Conv2d(4, channels[0], 3, padding=1)
        self.condition_encoder = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.down_attn = nn.ModuleDict()
        previous = 3
        for level, ch in enumerate(channels):
            self.condition_encoder.append(nn.Sequential(nn.Conv2d(previous, ch, 3, stride=1 if level == 0 else 2, padding=1),
                                                        nn.GroupNorm(min(32, ch), ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1)))
            previous = ch
            self.down_blocks.append(nn.ModuleList([TimeResBlock(ch, ch, time_channels), TimeResBlock(ch, ch, time_channels)]))
            if level >= 2:
                self.down_attn[str(level)] = SpatialCrossAttention(ch, heads)
            if level < 3:
                self.downsample.append(nn.Conv2d(ch, channels[level + 1], 3, stride=2, padding=1))
        self.middle = nn.ModuleList([TimeResBlock(channels[-1], channels[-1], time_channels),
                                     TimeResBlock(channels[-1], channels[-1], time_channels)])
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleDict()
        self.up_attn = nn.ModuleDict()
        for level in reversed(range(4)):
            ch = channels[level]
            self.up_blocks.append(nn.ModuleList([TimeResBlock(ch * 2, ch, time_channels), TimeResBlock(ch, ch, time_channels)]))
            if level >= 2:
                self.up_attn[str(level)] = SpatialCrossAttention(ch, heads)
            if level:
                self.upsample[str(level)] = nn.Conv2d(ch, channels[level - 1], 3, padding=1)
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out = nn.Conv2d(channels[0], 1, 3, padding=1)

    def run_block(self, block, *inputs):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, state, t, condition):
        if state.shape[-2:] != condition.shape[-2:] or state.shape[-1] % 8 or state.shape[-2] % 8:
            raise ValueError("State and condition need matching spatial dimensions divisible by eight")
        embedding = self.time_mlp(time_embedding(t, self.base_channels))
        c, encoded = condition, []
        for block in self.condition_encoder:
            c = self.run_block(block, c)
            encoded.append(c)
        h = self.input(torch.cat((state, condition), dim=1))
        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = self.run_block(block, h, embedding)
            if str(level) in self.down_attn:
                h = self.run_block(self.down_attn[str(level)], h, encoded[level])
            skips.append(h)
            if level < 3:
                h = self.downsample[level](h)
        for block in self.middle:
            h = self.run_block(block, h, embedding)
        for index, level in enumerate(reversed(range(4))):
            h = torch.cat((h, skips[level]), dim=1)
            for block in self.up_blocks[index]:
                h = self.run_block(block, h, embedding)
            if str(level) in self.up_attn:
                h = self.run_block(self.up_attn[str(level)], h, encoded[level])
            if level:
                h = self.upsample[str(level)](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(h)))


def flow_training_loss(model, condition, target, *, rng, drop_probability=.1):
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=rng)
    t = torch.rand((target.shape[0],), device=target.device, generator=rng)
    state = clamp_completion((1 - t[:, None, None, None]) * noise + t[:, None, None, None] * target, condition)
    drop = torch.rand((target.shape[0], 1, 1, 1), device=target.device, generator=rng) < drop_probability
    prediction = model(state, t, condition * (~drop))
    velocity = target - noise
    return masked_mean((prediction - velocity).square(), condition[:, 1:2])


@torch.no_grad()
def sample_heun(model, condition, *, samples=4, steps=25, guidance=2., seed=20260909, noise=None):
    if samples < 1 or steps < 1 or guidance < 0:
        raise ValueError("Invalid output budget, solver steps or CFG scale")
    model.eval()
    b, _, h, w = condition.shape
    repeated = condition.repeat_interleave(samples, dim=0)
    rng = torch.Generator(device=condition.device).manual_seed(seed)
    shape = (b * samples, 1, h, w)
    state = torch.randn(shape, device=condition.device, dtype=condition.dtype, generator=rng) if noise is None else noise.clone()
    if state.shape != shape:
        raise ValueError("Noise shape must match actual output count")
    state = clamp_completion(state, repeated)
    calls = 0

    def velocity(value, time):
        nonlocal calls
        t = torch.full((shape[0],), time, dtype=condition.dtype, device=condition.device)
        if guidance == 0:
            answer = model(value, t, repeated)
        else:
            both = model(torch.cat((value, value)), torch.cat((t, t)),
                         torch.cat((repeated, torch.zeros_like(repeated))))
            conditional, unconditional = both.chunk(2, dim=0)
            answer = (1 + guidance) * conditional - guidance * unconditional
        calls += 1
        return answer

    dt = 1 / steps
    for step in range(steps):
        first = velocity(state, step / steps)
        proposal = clamp_completion(state + dt * first, repeated)
        second = velocity(proposal, (step + 1) / steps)
        state = clamp_completion(state + .5 * dt * (first + second), repeated)
    # Binarization only after integration; intermediate states remain real-valued.
    probability_like = clamp_completion(state.clamp(0, 1), repeated)
    worlds = (probability_like > .5).reshape(b, samples, h, w)
    return probability_like.reshape(b, samples, h, w), worlds, {
        "solver": "Heun", "steps_per_sample": steps, "samples_per_observation": samples,
        "cfg_scale_s": guidance, "cfg_branches": 1 if guidance == 0 else 2,
        "batched_model_calls": calls, "velocity_evaluations_per_sample": 2 * steps,
        "sample_equivalent_forwards": calls * b * samples * (1 if guidance == 0 else 2),
        "condition_cache_used": False}
