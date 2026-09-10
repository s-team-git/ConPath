"""New formal-run training glue; historical experiment implementations stay frozen."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from .external_completion import clamp_completion
from .flow_matching import flow_training_loss
from .lama import generator_losses, hinge_discriminator_loss
from .losses import reachability_brier_u_statistic, spatial_variogram_score


class StagedLRScheduler:
    """Constant through pilot, then a separately sealed cosine continuation.

    A full budget may be attached only at/before the pilot boundary. Once attached
    it cannot change on resume. State is explicit, independent of wall-clock time.
    """
    def __init__(self, optimizer, *, pilot_steps=5000, full_steps=None, minimum_ratio=.1):
        if not isinstance(pilot_steps, int) or pilot_steps < 1 or not 0 <= minimum_ratio <= 1:
            raise ValueError('Invalid pilot steps or minimum learning-rate ratio')
        if full_steps is not None and full_steps <= pilot_steps:
            raise ValueError('Full budget must exceed pilot budget')
        self.optimizer = optimizer
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.pilot_steps, self.full_steps = pilot_steps, full_steps
        self.minimum_ratio = minimum_ratio
        self.completed_steps = 0

    def _apply(self):
        fraction = 0. if self.full_steps is None else max(0., min(1.,
            (self.completed_steps-self.pilot_steps)/(self.full_steps-self.pilot_steps)))
        factor = self.minimum_ratio + (1-self.minimum_ratio)*.5*(1+math.cos(math.pi*fraction))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base * factor

    def step(self):
        self.completed_steps += 1
        self._apply()

    def state_dict(self):
        return {key: getattr(self, key) for key in ('base_lrs', 'pilot_steps', 'full_steps',
                                                   'minimum_ratio', 'completed_steps')}

    def load_state_dict(self, state):
        for name in ('base_lrs', 'pilot_steps', 'minimum_ratio'):
            if state[name] != getattr(self, name):
                raise ValueError(f'Scheduler resume changed {name}')
        old_end = state['full_steps']
        if old_end != self.full_steps and not (old_end is None and self.full_steps is not None
                                               and state['completed_steps'] <= self.pilot_steps):
            raise ValueError('Frozen full schedule cannot change on resume')
        self.completed_steps = state['completed_steps']
        if not isinstance(self.completed_steps, int) or self.completed_steps < 0:
            raise ValueError('Invalid scheduler completed step')
        self._apply()


def _finite_loss(loss, name):
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError(f'Nonfinite {name} loss before optimizer update')


def _finite_gradients(module, name):
    gradients = [parameter.grad.detach() for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        raise FloatingPointError(f'{name} has no gradients')
    # One host synchronization checks all tensors; no clipping or rescaling is
    # introduced into the pinned external AdamW recipe.
    if not bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()):
        raise FloatingPointError(f'Nonfinite {name} gradients before optimizer update')


def checked_lama_update(model, discriminator, optim_g, optim_d, condition, target, microbatch):
    """Same G-then-D accumulated math as the frozen adapter, with fail-fast checks.

    If D fails after G has stepped, the enclosing driver must discard live state
    and restore its last complete checkpoint. It must not save a partial GAN.
    """
    if microbatch < 1 or not len(condition) or len(condition) % microbatch:
        raise ValueError('Effective batch must be nonempty and divisible by microbatch')
    chunks = len(condition) // microbatch
    model.train(); discriminator.train()
    optim_g.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    cached = []
    stats = dict(generator=0., discriminator=0., reconstruction=0., adversarial=0., feature_matching=0.)
    for c, t in zip(condition.split(microbatch), target.split(microbatch)):
        t = clamp_completion(t, c)
        prediction = model(c)
        _, real_features = discriminator(t)
        fake_scores, fake_features = discriminator(prediction)
        loss, terms = generator_losses(prediction, t, c, fake_scores, fake_features, real_features)
        _finite_loss(loss, 'LaMa generator')
        for name, value in terms.items():
            _finite_loss(value, f'LaMa {name}')
            stats[name] += float(value.detach()) / chunks
        (loss / chunks).backward()
        stats['generator'] += float(loss.detach()) / chunks
        cached.append((c, t, prediction.detach()))
    _finite_gradients(model, 'LaMa generator')
    optim_g.step()
    discriminator.requires_grad_(True)
    optim_d.zero_grad(set_to_none=True)
    for c, t, prediction in cached:
        real_scores, _ = discriminator(t)
        fake_scores, _ = discriminator(prediction)
        loss = hinge_discriminator_loss(real_scores, fake_scores, c[:, 1:2])
        _finite_loss(loss, 'LaMa discriminator')
        (loss / chunks).backward()
        stats['discriminator'] += float(loss.detach()) / chunks
    _finite_gradients(discriminator, 'LaMa discriminator')
    optim_d.step()
    return stats


def checked_flow_update(model, optimizer, condition, target, microbatch, rng):
    """Unchanged velocity objective/accumulation with checks before AdamW."""
    if microbatch < 1 or not len(condition) or len(condition) % microbatch:
        raise ValueError('Effective batch must be nonempty and divisible by microbatch')
    chunks = len(condition) // microbatch
    model.train(); optimizer.zero_grad(set_to_none=True)
    value = 0.
    for c, t in zip(condition.split(microbatch), target.split(microbatch)):
        loss = flow_training_loss(model, c, t, rng=rng)
        _finite_loss(loss, 'FM velocity')
        (loss / chunks).backward()
        value += float(loss.detach()) / chunks
    _finite_gradients(model, 'FM velocity')
    optimizer.step()
    return {'velocity_mse': value}


def tensor_batch(samples, device):
    from .formal_data import collate_formal_data
    batch = collate_formal_data(samples)
    # All genuine queries in a scene share a start; align padded queries too.
    if batch['starts'].shape[1]:
        batch['starts'][:] = batch['starts'][:, :1]
    tensors = {}
    for key in ('observation', 'valid_support_mask', 'target_free', 'loss_mask',
                'starts', 'goals', 'reachability_targets', 'query_mask', 'condition'):
        tensors[key] = torch.from_numpy(batch[key]).to(device)
    return batch, tensors


def per_scene_masked_mean(values, mask):
    count = mask.flatten(1).sum(1)
    return (values * mask).flatten(1).sum(1) / count.clamp_min(1)


def weighted_micro_loss(map_loss, variogram, event_loss, *, micro_count, batch_count,
                        micro_query_scenes, batch_query_scenes, event_weight, variogram_weight):
    """Preserve full-batch parent weights, including scenes without any queries."""
    value = (map_loss + variogram_weight * variogram) * (micro_count / batch_count)
    if batch_query_scenes:
        value = value + event_weight * event_loss * (micro_query_scenes / batch_query_scenes)
    return value


def parent_event_numerator(per_view_losses, samples, config):
    """Unbiased sum of parent event losses under one uniformly sampled view.

    For V fixed views and n query-bearing views, each selected query-bearing
    view contributes V/n times its own query/radius mean. A no-query selected
    view contributes zero, while its parent remains in the full-batch divisor.
    Counts depend exclusively on input-generated query eligibility.
    """
    counts = config['_n_query_views_by_parent']
    views = config['_views_per_parent']
    multipliers = []
    for sample in samples:
        count = counts[sample.row['parent_group']]
        if not isinstance(count, int) or not 0 <= count <= views:
            raise ValueError('Invalid query-bearing view count for parent')
        if len(sample.starts) and count == 0:
            raise ValueError('Query-bearing view belongs to a declared zero-query parent')
        multipliers.append(views / count if count and len(sample.starts) else 0.)
    weights = per_view_losses.new_tensor(multipliers)
    return (per_view_losses * weights).sum()


def eligible_parent_slots(samples, config):
    counts = config['_n_query_views_by_parent']
    return sum(counts[sample.row['parent_group']] > 0 for sample in samples)


def parent_weighted_micro_loss(map_loss, variogram, event_numerator, *, micro_count,
                               batch_count, batch_eligible_parents, event_weight,
                               variogram_weight):
    value = (map_loss + variogram_weight * variogram) * (micro_count / batch_count)
    if batch_eligible_parents:
        value = value + event_weight * event_numerator / batch_eligible_parents
    return value


def _control_losses(model, method, samples, rng, config, device):
    from .formal_inference import direct_forward
    from .formal_metrics import exact_world_events
    arrays, batch = tensor_batch(samples, device)
    target, mask = batch['target_free'].float(), batch['loss_mask']
    zero = next(model.parameters()).sum() * 0
    active = batch['query_mask'].any(1)
    if method == 'direct_query':
        if active.any():
            logits = direct_forward(model, batch['observation'], batch['starts'], batch['goals'],
                                    config['radii_cells'])
            errors = F.binary_cross_entropy_with_logits(logits, batch['reachability_targets'].float(), reduction='none')
            per_view = per_scene_masked_mean(errors, batch['query_mask'][..., None].expand_as(errors))
            event = parent_event_numerator(per_view, samples, config)
        else:
            event = zero
        return zero, zero, event, 0
    if method == 'deterministic':
        logits = model(batch['observation'])
        error = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return per_scene_masked_mean(error, mask).mean(), zero, zero, 0
    has_event_loss = method != 'no_reach' and bool(active.any())
    kwargs = dict(valid_support_mask=batch['valid_support_mask'], num_samples=config['train_samples'],
                  disable_global_factors=method == 'independent', generator=rng)
    if has_event_loss:
        kwargs.update(starts=batch['starts'], goals=batch['goals'],
                      footprint_radii_cells=config['radii_cells'], hard_samples=True,
                      max_reachability_steps=config['max_reachability_steps'], shared_start=True)
    output = model(batch['observation'], **kwargs)
    worlds = output.posterior.safe_samples()
    marginal = output.posterior.sample_logits.softmax(2)[:, :, 0].mean(1)
    errors = F.binary_cross_entropy(marginal.clamp(1e-6, 1-1e-6), target, reduction='none')
    map_loss = per_scene_masked_mean(errors, mask).mean()
    variogram = torch.stack([spatial_variogram_score(worlds[i:i+1], target[i:i+1],
                                                   valid_mask=mask[i:i+1]) for i in range(len(samples))]).mean()
    event_loss, corrections = zero, 0
    if has_event_loss:
        hard = np.stack([exact_world_events(w, s, g, config['radii_cells']) for w, s, g in
                        zip(worlds.detach().cpu().numpy() > .5, arrays['starts'], arrays['goals'])])
        hard = torch.from_numpy(hard).to(device=device, dtype=torch.float32)
        corrections = int(((output.sample_reachability.detach() != hard) & batch['query_mask'][:, None, :, None]).sum())
        events = output.sample_reachability + (hard-output.sample_reachability.detach())
        # Retain one scalar per view until the parent importance weight is
        # applied. Normalizing across only this microbatch's active views would
        # silently downweight parents with a single query-bearing view.
        per_view = torch.stack([reachability_brier_u_statistic(events[i:i+1],
            batch['reachability_targets'][i:i+1],
            weights=batch['query_mask'][i:i+1, :, None].expand_as(batch['reachability_targets'][i:i+1]))
            for i in range(len(samples))])
        event_loss = parent_event_numerator(per_view, samples, config)
    return map_loss, variogram, event_loss, corrections


def training_update(models, optimizers, method, samples, rng, config, device='cuda'):
    """One complete update; the driver checkpoints only after this returns."""
    model = models['model']
    for module in models.values():
        module.train()
    micro = config['micro_batch']
    if method in ('lama', 'flow'):
        condition = torch.from_numpy(np.stack([s.condition for s in samples])).to(device)
        target = torch.from_numpy(np.stack([s.target for s in samples])[:, None]).to(device=device, dtype=torch.float32)
        if method == 'lama':
            return checked_lama_update(model, models['discriminator'], optimizers['model'],
                optimizers['discriminator'], condition, target, micro)
        return checked_flow_update(model, optimizers['model'], condition, target, micro, rng)
    optimizer = optimizers['model']
    optimizer.zero_grad(set_to_none=True)
    query_scenes = eligible_parent_slots(samples, config)
    stats = dict(map_nll=0., variogram=0., event_loss=0., total=0., hard_forward_corrections=0)
    for begin in range(0, len(samples), micro):
        part = samples[begin:begin+micro]
        map_loss, vario, event, corrections = _control_losses(model, method, part, rng, config, device)
        weight = 1. if method == 'direct_query' else config['event_weight']
        loss = parent_weighted_micro_loss(map_loss, vario, event, micro_count=len(part), batch_count=len(samples),
            batch_eligible_parents=query_scenes,
            event_weight=weight, variogram_weight=config['variogram_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite control loss')
        loss.backward()
        stats['map_nll'] += float(map_loss.detach())*len(part)/len(samples)
        stats['variogram'] += float(vario.detach())*len(part)/len(samples)
        stats['event_loss'] += float(event.detach())/max(query_scenes, 1)
        stats['total'] += float(loss.detach())
        stats['hard_forward_corrections'] += corrections
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(stats, gradient_norm=float(grad_norm))
