# ConPath: Connectivity-Calibrated Path Reliability from Partial BEV Observations

This is the writing scaffold for the first IROS submission. It deliberately separates results that
are already reproducible in this repository from claims that require a trained CUDA model and public
data.

## Abstract (working version)

Most uncertainty-aware navigation systems calibrate occupancy at individual voxels, while a robot
ultimately needs a decision about a non-local event: whether a start and goal are connected for its
footprint. These objectives are not equivalent. ConPath predicts a correlated posterior over binary
support maps and derives the two-terminal, footprint-conditioned path event from sampled maps. The
posterior is trained with a proper event score together with marginal and spatial scores; the event
operator uses exact binary geometry in the forward pass and a straight-through surrogate only for
learning. We introduce a held-out-template synthetic death test in which visible observations have
different context-conditioned doorway priors and repeated hidden worlds, making independent-cell
completion statistically misspecified. Two CUDA-trained optimization seeds obtain event Brier
scores of 0.1164 and 0.1116, compared with 0.1832 for independent cells and 0.1699 for direct query;
a matched model without the event loss obtains 0.1914 despite comparable map-marginal Brier. These
numbers close a synthetic hypothesis audit on one fixed split, not a public-data or real-robot
result. The final paper must evaluate event calibration on versioned occupancy data with
site/sequence-held-out splits and public completion baselines.

## Claim boundary

The paper claims a task-level probabilistic interface:

```text
q_theta(s, g, r | X) = P(there exists a support-valid four-connected path
                         for a disk footprint of radius r | partial BEV X).
```

It does not claim a complete RGB-to-BEV system, dynamics or collision guarantee, arbitrary SE(2)
footprints, or calibrated performance on a public dataset until those experiments are complete.
`TRAVERSABLE` and `BLOCKED` are the only latent world classes; `UNKNOWN` is an observation/validity
state.

## Current public-data checkpoint (not a paper result)

The first FlatLands pass uses a bounded 512-scene sample from the explicitly non-official
`provenance.original_split` (160 train / 160 validation / 192 test scenes). The published archive
directory split is scene-leaky and is never used for cross-scene evidence. On the frozen validation
queries, the radius-prior, deterministic completion, independent-cell K=32, and direct-query
baselines obtain scene-weighted Brier scores 0.15870, 0.08556, 0.22546, and 0.09119, respectively.
The direct-query comparator has the best NLL/ECE, while deterministic completion has the best Brier.
These are evaluator and difficulty diagnostics only: one optimization seed, validation only, no
official completion weights, no final ConPath result, and no test labels read. The site labels this
section “validation-only / test locked / not final paper result.”

## Intended contributions

1. **Event-level formulation.** We distinguish voxel marginal calibration, joint occupancy
   calibration, and two-terminal footprint-conditioned event calibration, and evaluate each with its
   own proper score.
2. **Correlated posterior-to-event model.** Low-rank global factors and local correlated noise produce
   coherent map hypotheses; reachability is computed from each hypothesis rather than from a direct
   query head or an independent-cell factorization.
3. **Auditable benchmark protocol.** The synthetic generator creates same-visible-observation,
   multi-world conflicts, context-dependent hidden-door priors, random visible-support queries, and
   scene-template-held-out splits. The protocol reports Brier/NLL/ECE, false-safe rate, radius curves,
   map marginals, and joint doorway frequency.
4. **Scalable exact-forward contract.** A NumPy Kruskal merge-tree/LCA reference answers many
   maximum-bottleneck queries exactly and provides the correctness contract for a future CUDA
   exact-forward/soft-backward operator.

The first contribution is the central scientific claim. The other contributions support its
identifiability and reproducibility; they are not presented as novel occupancy completion by
themselves.

## Experiment matrix for the full paper

| Question | Required comparison | Metric / split | Status |
|---|---|---|---|
| Does joint structure matter? | constant, independent Bernoulli, direct query, edge-connectivity, deterministic, ConPath | event Brier/NLL/ECE and false-safe; held-out templates | neural P0 passes in two optimization seeds |
| Is the map still useful? | independent vs ConPath posterior samples | map NLL/Brier/ECE, variogram, joint doorway frequency | neural synthetic map scores recorded; fragmentation remains |
| Does the loss matter? | ConPath vs no reachability loss | same checkpoint budget and query draws | matched CUDA ablation fails P0 |
| Does geometry matter? | radii 0/1/2 (then dataset-specific radii) | per-radius event curves and bottleneck strata | synthetic recorded |
| Does it transfer? | FlatLands completion samples, then UnScenes3D/WildOcc if needed | source/scene-held-out event calibration | bounded data gate and first validation baselines pass on a non-official provenance split; multi-seed ConPath/second domain pending |
| Is inference scalable? | iterative propagation vs merge-tree forward | exact error, latency, peak memory vs map/query count | NumPy reference recorded |

Every reported checkpoint must include dataset version, split manifest, query-generation seed,
posterior sample count, geometry convention, and the exact command. No adjacent-frame random split is
allowed.

## Reviewer-facing ablations

- Hold voxel marginal quality fixed while changing only the joint sampler.
- Hold the sampler fixed while removing the reachability proper score.
- Match parameter count and encoder features for direct-query and connectivity baselines.
- Report calibration at equal coverage and a high-confidence false-safe threshold (0.8), not only
  average accuracy.
- Verify monotonicity in footprint radius and symmetry under swapping start and goal.
- Show a same-marginal conflict figure: independent samples create fragmented door states while a
  correlated posterior produces whole-door open/closed worlds.

## Current go/no-go decision

The learned model passes the tightened P0 gate in both tested optimization seeds, while the matched
no-reach ablation fails. A 512-scene, target-blind FlatLands audit and the first validation-only
baseline slate also pass their respective plumbing gates on an explicitly non-official
upstream-provenance split. This is **GO for multi-seed ConPath development**, but remains **NO-GO for
public-data or paper claims** until the gain survives multi-seed ConPath, causal ablations, scalable
connectivity, calibration/false-safe analysis, official/public completion baselines, and a second
domain. The query distribution is source/radius dependent—test ARKitScenes has no 20 cm positives in
the bounded manifest—so pooled metrics are insufficient. The synthetic full model also retains
nonzero doorway fragmentation, which must not be hidden by aggregate event metrics.
