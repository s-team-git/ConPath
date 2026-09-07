# ConPath: Connectivity-Calibrated Path Reliability from Partial BEV Observations

Working manuscript, updated 2026-09-07 (literature and prospective experimental design; numerical results unchanged). Current numerical tables and standalone figures are generated in [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md). Earlier tables and superseded experiments are preserved in [PAPER_DRAFT_HISTORY.md](PAPER_DRAFT_HISTORY.md).

## Abstract

Path reliability depends on the joint distribution of free space, whereas occupancy scores usually evaluate individual cells. ConPath learns a correlated posterior over binary support maps and estimates the probability that a path exists between two terminals for a specified disk footprint. Training combines map and spatial losses with a proper event score; evaluation uses exact discrete connectivity. A synthetic repeated-world experiment passes in two optimization seeds, while a matched model without the event loss fails. On a frozen, scene-disjoint, non-official FlatLands validation split, three corrected training runs achieve event Brier 0.06749 versus 0.09521 for a matched independent decoder. Permuting sample indices separately at every cell preserves all empirical occupancy marginals but worsens event Brier to 0.16139, isolating a role for spatial dependence. The advantage over a deterministic map from the same checkpoint is smaller and inconsistent across seeds. At equal 30% coverage, false-safe rates of 3.60% and 3.90% do not establish a stable risk improvement over the independent decoder. A second-domain audit identifies an observation-conditioned error floor that limits the current UnScenes3D adapter. These validation results support an event-level research formulation while leaving final testing, robust selective-risk improvement, and external generalization unresolved.

## 1. Problem and scope

For a partial BEV observation `X`, a supplied validity domain `V`, start `s`, goal `g`, and disk radius `r`, the quantity of interest is

```text
q_theta(s,g,r | X,V)
  = E_{M ~ p_theta(M | X,V)} [1{a support-valid path exists from s to g for radius r}].
```

Identical probabilities at individual cells do not determine this expectation: dependence can change whether a bottleneck opens as a coherent region. We evaluate map marginals, joint sample structure, and path events separately. Lower map error alone does not establish better event reliability.

The prototype operates on rasters, circular footprints, and four-neighbor connectivity. It does not implement vehicle dynamics, an arbitrary SE(2) swept footprint, or a collision guarantee. Released validity masks define the benchmark support domain; UnScenes3D uses label validity as a supplied support mask. This is a benchmark condition, not evidence that the same support domain is available to a deployed robot.

## 2. Comparison with related methods

The comparison separates deterministic completion, independent stochastic completion, joint stochastic completion, and direct event prediction. The primary dependence comparison matches encoder capacity and sample budget. The fixed-marginal shuffle additionally changes the joint arrangement of generated worlds while preserving every empirical cell probability, addressing a confound that separately trained models cannot eliminate.

The coordinate-query control is inspired by implicit-field representations; it is not a faithful S4C reproduction. Sources, task differences, and implementation status are recorded in [RECENT_BASELINES.md](RECENT_BASELINES.md). An eight-paper review now prioritizes BEV completion over a wholesale transition to 3-D SSC: the FlatLands benchmark compares completion families under shared BEV conditioning; MapEx uses LaMa-based map predictions; CogniPlan provides a conditional generative map module with native layout-type supervision. PaSCo's separate common-backbone uncertainty comparisons and S4C's re-evaluation under a shared support mask inform the proposed controls. The review and source locations are recorded in [LITERATURE_REVIEW_ZH.md](LITERATURE_REVIEW_ZH.md).

The prospective external comparison includes LaMa/ensemble and conditional flow completion on FlatLands, followed by CogniPlan's released generator on its native map data. The generator's layout labels prevent treating a file-path replacement as a faithful FlatLands transfer. These experiments have not been run. Module adaptations will retain their core mechanisms and disclose changes; full navigation-system claims require separate closed-loop evaluation. Cross-task 3-D mIoU, FID, or exploration scores do not enter the event table. [EXPERIMENT_DESIGN_ZH.md](EXPERIMENT_DESIGN_ZH.md) specifies a new matched-output K=4 comparison and measured cost curves; neither overwrites the frozen K=128 validation evidence.

## 3. Method

### 3.1 Correlated map posterior

Three input channels encode observed free, observed blocked, and unknown cells. A compact BEV encoder supplies a two-class mean-logit head, low-rank global factors, and locally correlated stochastic logits. The final FlatLands model uses feature width 16 and four global latent dimensions, with 120,108 parameters. The matched independent decoder uses the same capacity, disables global factors, and uses a one-cell local kernel.

Observed evidence is clamped in each world, and the complement of `V` is always blocked before sampling. `UNKNOWN` is an observation state, not a third physical world class. The exact target geometry uses the same support boundary. Earlier forwards omitted this boundary; only checkpoints retrained with the corrected contract supply current main results.

### 3.2 Footprint-conditioned connectivity

Eroding a sampled free set by a discrete radius-`r` disk gives valid vehicle-center positions. The event is one when `s` and `g` belong to the same four-connected component after erosion. Equivalently, let `C*(s,g;M)` be the maximum, over all paths, of the minimum clearance along the path. Then

```text
q_hat(s,g,r) = (1/K) sum_k 1{C*(s,g;M_k) >= r}.
```

Using the same worlds for every radius makes probability non-increasing with vehicle size. Invalid endpoints are unreachable. Final predictions use exact disk clearance and connectivity. A NumPy merge-tree reference supplies the correctness oracle; the accelerated connected-component evaluator checks itself against that oracle before loading checkpoints.

### 3.3 Training and checkpoint selection

The objective combines posterior-marginal NLL, a variogram term, and an event U-statistic Brier estimator. For binary sample events `z_k` and target `y`,

```text
L_U = ((sum_k z_k)^2 - sum_k z_k) / (K(K-1)) - 2 y mean_k(z_k) + y^2.
```

Its expectation is the squared error of the underlying event probability without the additional Monte Carlo variance penalty of the plug-in training Brier. Evaluation uses ordinary Brier, NLL, and ECE. Straight-through samples and a relaxed connectivity backward path provide gradients. Public-data training uses a bounded 256-step propagation budget; this is not a globally exact large-map training operator.

Clean FlatLands runs use eight training worlds, 128 validation worlds in chunks of eight, AdamW with learning rate 0.0003, at most 40 epochs, and patience eight. Both decoders use the same three optimization seeds. Checkpoint selection evaluates up to eight fixed target-blind queries per observation with bounded propagation; final exact evaluation includes all retained queries. The selection/evaluation operator difference remains a limitation. New analyses do not retune checkpoints, query selection, split, or radii.

## 4. Experimental protocol

### 4.1 Synthetic hypothesis test

P0 holds out scene templates, supplies visible context distinguishing hidden-door priors, and repeats hidden worlds for the same observation. Two full-model seeds obtain event Brier 0.116377 and 0.111621, versus 0.183172 for independent Bernoulli and 0.169888 for direct query. A matched no-event-loss run obtains 0.191368 despite comparable map marginals. Residual fragmented doorway samples remain a limitation. These are synthetic results; exact gates and configurations are in [P0_DEATH_TEST.md](P0_DEATH_TEST.md).

### 4.2 FlatLands validation

The published observation-directory split fails scene isolation. We use the explicitly non-official `provenance.original_split`, preserving upstream scene memberships. The bounded dataset/query manifest is frozen. The current comparison has 4,224 event rows from 1,408 endpoint groups across 142 contributing validation scenes, at radii 0, 10, and 20 cells. All methods use identical event keys. No test set is used for model selection or evaluation in this analysis.

Each scene receives equal total weight and each event within it equal weight. Three-seed variability is sample standard deviation. Paired uncertainty resamples whole scenes 2,000 times within each seed; event rows are not independent observations. The exploratory comparisons are not adjusted confirmatory significance tests.

Selective risk uses scene-weighted coverage. Every event tied at the boundary probability receives the same acceptance fraction independently of its label. This achieves requested coverage in expectation, including binary predictors. Each bootstrap replicate recomputes its coverage boundary. These curves describe validation ranking behavior; they are not fitted deployment thresholds.

### 4.3 Second-domain diagnosis

The frozen UnScenes3D LiDAR/ground-valid adapter uses nine training scenes and two validation scenes, with 478/62 frames and 15,567/1,529 queries. Three radii yield 46,701/4,587 events. The test site `location_6` remains unopened. Six fresh correlated/independent adapters pass support/checkpoint audits. Their deterministic K=128 mean-map events form a transfer diagnostic with a different predictive object from the primary stochastic event table.

## 5. Results

The full nine-control table, per-seed paired intervals, source/radius results, and prediction hashes are in [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md) and [the machine-readable analysis](site/data/flatlands_clean_paper_analysis.json).

### 5.1 Event accuracy and dependence

ConPath Brier is 0.06749 ± 0.00936 versus 0.09521 ± 0.00703 for the independent decoder. The paired mean reduction is 0.02772, with positive scene-bootstrap intervals in all three seeds. Deterministic completion and coordinate query obtain 0.08857 and 0.09204; their third-seed paired Brier intervals include zero, so the evidence does not establish a uniformly decisive improvement over every comparator.

Separately trained correlated and independent models have different hidden-map Brier (0.15451 versus 0.16789), so that comparison alone does not hold marginals fixed. The shuffle intervention preserves every empirical per-cell count in the original 128 worlds, changing only world-index alignment between cells. Event Brier rises to 0.16139 ± 0.00677. This supports a dependence mechanism in the frozen samples; it does not replace clean no-event/no-global training ablations or prove generalization.

![Fixed-marginal dependence intervention](site/assets/flatlands_clean_marginal_shuffle.svg)

### 5.2 Strong mean-map control and selective risk

Thresholding the same ConPath checkpoint's posterior mean at 0.5 gives Brier 0.06957 ± 0.00380; the independent checkpoint's mean map gives 0.07184 ± 0.00349. The stochastic-versus-mean-map gap is small and reverses in one seed. The stochastic method cannot be described as uniformly better than deterministic thresholding on this benchmark.

At confidence 0.8, ConPath has lower false-safe rate (0.04552 versus 0.05689) but different coverage (0.33644 versus 0.36592). At equal 30% coverage, risks are 0.03600 and 0.03901; all three per-seed paired intervals include zero. This does not support a robust equal-coverage safety claim.

![Equal-coverage selective risk](site/assets/flatlands_clean_equal_coverage.svg)

### 5.3 Sample-budget sensitivity

The saved final sampling state reproduces every original K=128 event probability across six checkpoints with zero drift. Nested prefixes give ConPath Brier 0.06861 / 0.06771 / 0.06749 at K=32 / 64 / 128, and independent Brier 0.09575 / 0.09540 / 0.09521. Aggregate changes over these budgets are small relative to seed variation. This is fixed-checkpoint sensitivity, not arbitrary-K convergence or a training-budget ablation.

### 5.4 Second-domain observation constraints

Clean UnScenes3D mean-map Brier is 0.51142 ± 0.00198 versus 0.51130 ± 0.00218 for the independent decoder. The near-zero difference is not evidence of successful transfer. A final mean-map projection audit found that restoring observed-free cells could reopen invalid support after correct model sampling. Version 2 applies the support mask last in both the evaluator and renderer; all six checkpoints and hidden-map metrics reproduce exactly, and all 27,522 corrected event predictions satisfy the observation bounds. Superseded v1 predictions remain available for audit.

Let `F_min` contain only observed-free valid cells, and let `F_max` also free every unknown valid cell. Every world respecting the hard observation lies between these sets. Disk erosion and connectivity are monotone, so its event is bounded by the pessimistic and optimistic events. A positive target with optimistic event zero, or a negative target with pessimistic event one, forces Brier error one for every such posterior. This yields a lower bound without selecting model parameters.

All 51,288 train/validation event labels were replayed against the exact oracle. The scene-weighted Brier lower bound is 0.13097 on training and 0.47574 on validation. Only 15.66% of validation events are mutable through unknown completion; the lower bound accounts for about 93% of current mean-map error. Further optimization cannot remove this forced component. A future adapter should first revisit the observation likelihood using training data and a separately versioned protocol. The bound applies to the frozen hard-observation adapter, not to all sensor models or the dataset in general.

![Observation-conditioned error floor](site/assets/unscenes3d_observation_ceiling.svg)

## 6. Reproducibility and limitations

Source, protocol, checkpoint and prediction hashes accompany the reports. The checkpoint environment uses Python 3.13.13. Reruns restore the saved CUDA sampling state and validate exact geometry. Commands and downloadable SVG/PDF figures are in [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md). Real checkpoint-derived positive and false-safe panels disclose their label-based explanatory case selection.

Limitations include validation reuse for checkpoint selection, the non-official split, source/radius-dependent prevalence, two second-domain validation scenes, hard observation assumptions, and bounded differentiable propagation. Current NumPy/accelerated CPU inference diagnostics do not establish a scalable CUDA backward operator. There is no claimed end-to-end sensor fusion, real-robot navigation, rectangular footprint, or certified collision result.

## 7. Current research decision

The evidence supports further study of joint map structure and event scoring. It does not yet meet the final-paper gate: deterministic controls remain close, equal-coverage risk improvement over the independent decoder is not stable, and the second-domain adapter has a large observation-conditioned error floor.

The next design milestone is to validate external-method interfaces and dataset provenance, including a discrepancy between the FlatLands paper's nominal spatial resolution and the local release metadata, before expanding physical-footprint claims or scheduling expensive runs. The clean three-seed no-event/no-global matrix remains a required causal experiment under its frozen contract and is paused. Before another UnScenes3D matrix, a training-only observation model audit must address the bound without held-out test labels. Scalable training-operator evidence, the external-method comparison, and a frozen test protocol remain necessary. This remains a working draft; final public-data, transfer, and submission-ready claims are unsupported.

The clean no-event/no-global training matrix was launched on 2026-09-07 under the frozen [ablation analysis plan](CLEAN_ABLATION_PLAN.md), then paused with three no-event runs at completed epoch 4 and no-global not yet started. No ablation result is claimed before all six seeds/runs complete their audits.
