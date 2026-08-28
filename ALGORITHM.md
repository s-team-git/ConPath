# ConPath Algorithm Contract

## Research question

Voxel-wise calibration does not determine path reliability. PathRel asks whether a learned joint
occupancy posterior can calibrate the non-local event that at least one four-connected,
circular-footprint topological path exists.

For an observation `X`, latent map `M`, start `s`, goal `g`, and footprint radius `r`:

```text
q_theta(s,g,r|X) = E_{M ~ p_theta(M|X)}[rho(M;s,g,r)]
```

where `rho` is one exactly when `s` and `g` are connected in a sampled hidden-world map after
treating blocked and out-of-map cells as occupied and inflating them by `r`.

## Model contract

1. `Encoder(X) -> F`: dataset-specific observations become a shared BEV feature map.
2. `MeanLogitHead(F) -> mu`: the location of a binary logistic-normal field, not the posterior
   marginal itself.
3. `CorrelationHead(F) -> (B, sigma)`: low-rank global factors and local noise scale.
4. Reparameterized samples:

   ```text
   L_k = mu + B z_k / sqrt(D) + sigma * LocalFilter(epsilon_k)
   M_k = StraightThroughGumbelSoftmax(L_k)
   ```

5. The latent world has `TRAVERSABLE` and `BLOCKED` states. Observation unknown/occlusion is an
   input channel and label-validity condition, not a third sampled world class.
6. `rho(M_k;s,g,r)` is derived from the map. A direct query classifier is only a baseline.

## Discrete clearance target

For a binary support-valid map, let `D_M(v)` be the largest integer disk radius centered at `v`
that contains no occupied or out-of-map cell. The maximum-clearance path value is

```text
C*(s,g;M) = max_path min_{v on path} D_M(v).
```

Then

```text
rho(M;s,g,r) = 1[C*(s,g;M) >= r].
```

`labels.py` computes this target exactly on the discrete grid using a clearance map and a
maximum-bottleneck graph search. The current learned layer uses the same disk-footprint convention
and four-neighbor motion, so the training target and surrogate do not silently use different
geometry.

## Training losses

```text
L = L_posterior_marginal
  + lambda_vario * L_variogram
  + lambda_reach * L_discrete_CRPS.
```

- `L_posterior_marginal`: log score on the mean conditional class probability across latent
  samples. `softmax(mean logits)` is never reported as the posterior marginal.
- `L_variogram`: encourages correct local and longer-range pair dependence; it does not by itself
  guarantee recovery of the complete high-order topology.
- `L_discrete_CRPS`: the mean Brier score across start-goal queries and footprint radii.

The event label is generated from the complete target map. Observation channels may hide the
critical bottleneck; this is what creates a conditional uncertainty problem.

## Training stages

### P0: synthetic contract test

- Train on ambiguous walls and doorways.
- Confirm radius monotonicity, start-goal symmetry, non-zero stochastic-head gradients, and
  decreasing Brier score.
- Compare against a constant-rate predictor and independent cell samples.

`scripts/evaluate_p0.py` implements this protocol with two visible context families whose hidden
door priors are 0.2 and 0.8. Worlds are repeated within a scene template and test templates are
held out. It reports event Brier/NLL/ECE, false-safe rate, reliability bins, radius curves, voxel
marginal scores, and the joint open-door frequency. The correlated row is an oracle posterior
diagnostic; it is not evidence that the neural decoder has learned the posterior.

`scripts/train_p0_neural.py` follows the same split and now exposes a staged warm-up: map
cross-entropy on the mean head first, then the posterior-marginal/variogram/U-statistic event losses.
This reduces the chance that a noisy reachability gradient is mistaken for a learned occupancy
posterior.

### P1: public-data audit

- Start with FlatLands to audit partial/full/valid masks, query balance, bottlenecks, and to build a
  completion-plus-post-hoc-connectivity baseline; use ORFD as a secondary label-semantics audit.
- Use sequence/site splits, never random adjacent frames.
- Reject the direction if disconnected/bottleneck queries are rare or independent Bernoulli
  sampling matches the correlated model.

### P2: main public experiment

- If FlatLands event calibration survives, add an official/public encoder adapter for UnScenes3D.
- Use road elevation and occupancy to construct a 2.5-D support surface.
- Validate on WildOcc without adapting the reliability metric.

### P3: final method

- Replace the iterative prototype layer with a differentiable scalable path/cut implementation. A
  NumPy exact-forward merge-tree reference now lives in `labels.py`; it is the contract and
  correctness oracle for a future CUDA kernel, not yet a replacement for the PyTorch backward path.
- Add an SE(2) rectangular footprint only after the 2.5-D scientific claim survives P1/P2.

## Required baselines

- deterministic occupancy plus thresholding;
- voxel temperature scaling;
- independent Bernoulli samples with the same marginal logits;
- deep ensemble;
- direct `q(s,g,r)` head;
- topology loss/MALIS-style supervision;
- the same correlated decoder without reachability loss.

## Death tests

Stop the project if any of the following holds:

1. calibrated independent cells match task-level Brier/ECE;
2. improvements disappear under sequence/site splits;
3. gains occur only on hand-designed queries;
4. map quality is traded away to game reachability labels;
5. observation-validity masks or accumulated hidden-world occupancy truth cannot be audited;
6. a direct query MLP matches PathRel while being equally calibrated across unseen sites and
   footprint sizes.
