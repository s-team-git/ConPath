"""Explicit effective-batch updates for the documented external BEV recipes."""
import torch

from .external_completion import clamp_completion
from .flow_matching import flow_training_loss
from .lama import generator_losses, hinge_discriminator_loss


def lama_accumulated_update(model, discriminator, optim_g, optim_d, condition, target, microbatch):
    if len(condition) % microbatch:
        raise ValueError('Effective batch must be divisible by microbatch')
    chunks = len(condition) // microbatch
    model.train(); discriminator.train()
    optim_g.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    cached = []
    g_value, d_value, reconstruction = 0., 0., 0.
    for c, t in zip(condition.split(microbatch), target.split(microbatch)):
        t = clamp_completion(t, c)
        prediction = model(c)
        _, real_features = discriminator(t)
        fake_scores, fake_features = discriminator(prediction)
        loss, terms = generator_losses(prediction, t, c, fake_scores, fake_features, real_features)
        (loss / chunks).backward()
        g_value += float(loss.detach()) / chunks
        reconstruction += float(terms['reconstruction'].detach()) / chunks
        cached.append((c, t, prediction.detach()))
    optim_g.step()
    discriminator.requires_grad_(True)
    optim_d.zero_grad(set_to_none=True)
    for c, t, prediction in cached:
        real_scores, _ = discriminator(t)
        fake_scores, _ = discriminator(prediction)
        loss = hinge_discriminator_loss(real_scores, fake_scores, c[:, 1:2])
        (loss / chunks).backward()
        d_value += float(loss.detach()) / chunks
    optim_d.step()
    return {'generator': g_value, 'discriminator': d_value, 'reconstruction': reconstruction}


def flow_accumulated_update(model, optimizer, condition, target, microbatch, rng):
    if len(condition) % microbatch:
        raise ValueError('Effective batch must be divisible by microbatch')
    chunks = len(condition) // microbatch
    model.train(); optimizer.zero_grad(set_to_none=True)
    value = 0.
    for c, t in zip(condition.split(microbatch), target.split(microbatch)):
        loss = flow_training_loss(model, c, t, rng=rng)
        (loss / chunks).backward()
        value += float(loss.detach()) / chunks
    optimizer.step()
    return {'velocity_mse': value}
