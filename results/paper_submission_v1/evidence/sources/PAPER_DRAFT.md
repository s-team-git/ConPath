# ConPath: Joint Spatial Map Posteriors for Footprint-Aware Reachability Prediction

**完整英文工作稿 · 2026-09-10 · validation-only。** 已停止外部基线及新增调参。本文的方法版本固定为原版 ConPath；父级地点隔离的三种子开发实验是当前主要验证证据，旧六方法表仅作历史机制诊断。最终测试未在本轮打开，但历史物理 test 访问不能隐瞒。尚不能称为已达到 IROS/ICRA 投稿证据门槛。数字、缺项和来源分别见 [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md)、[PAPER_TABLES.md](PAPER_TABLES.md)、[PAPER_FIGURES.md](PAPER_FIGURES.md) 和 [最终模型选择](results/FINAL_MODEL_SELECTION.md)。

## Abstract

A robot's ability to reach a goal through partly observed space depends on how free cells occur together, as well as on their individual probabilities. We study the probability that a path exists for a specified circular robot footprint, conditioned on a partial bird's-eye-view map and a supplied valid-support mask. ConPath represents a joint stochastic occupancy posterior with global latent factors and local spatial noise, conditions sampled worlds to preserve observed evidence under the registered sampling rule, and estimates reachability by exact disk erosion and connectivity. A sample-pair event objective trains the posterior toward the downstream probability forecast. On a small FlatLands development split with disjoint parent places, three training seeds obtain Event Brier 0.10380 versus 0.11201 for independently sampled completion. A separate historical-cohort intervention preserves every empirical cell marginal while increasing Event Brier from 0.06749 to 0.16139 after spatial dependence is disrupted. Historical matched training ablations also support event supervision and global decoder factors, but that cohort has subsequently identified place overlap and cannot establish unseen-place generalization. Selective-risk improvements are not uniformly supported, and deterministic completion remains competitive. These validation-only findings support task-level evaluation of joint map posteriors; they do not establish final-test calibration, external-model superiority, or closed-loop navigation performance.

## 1. Introduction

A partially observed map can assign high free-space probability to every cell in a doorway while rarely generating a doorway wide enough for a robot. Conversely, a model can assign uncertain occupancy to many cells yet consistently generate one of several connected passages. A cellwise completion score cannot distinguish these situations completely. For a navigation decision, the relevant random variable may be whether **any** footprint-feasible path exists between a particular start and goal.

Uncertain mapping, topology-aware roadmap construction, and uncertainty-aware planning already address related problems. MRFMap models occupancy dependencies induced by sensor rays, and uncertainty-aware planners can use alternative route hypotheses. ConPath's scope is narrower: learning and evaluating a conditional probability of a path-existence event from a partial BEV map, rather than constructing a sensor map or executing a trajectory. [MRFMap](https://www.roboticsproceedings.org/rss16/p060.html), [Banfi et al.](https://arxiv.org/abs/2205.14251)

Our working hypothesis is that spatially correlated map samples and explicit event supervision improve this forecast beyond independently sampled cells. A fair test must separate three effects: changing map marginals, changing dependence at fixed marginals, and changing the training objective. It must also retain strong deterministic and direct-event controls, disclose finite-sample effects, and evaluate errors at comparable coverage.

We organize the paper around three questions:

- **RQ1:** At identical empirical map marginals, does the joint arrangement of sampled maps improve path-event prediction relative to disrupted spatial dependence?
- **RQ2:** Does reachability supervision improve Event Brier, NLL, ECE, and false-safe behavior relative to optimizing map completion alone?
- **RQ3:** How reliable are the predictions across footprint radii, partial observations, bottlenecks, and unreachable queries?

The contributions are (i) an explicit joint-map-to-footprint-event formulation; (ii) a stochastic decoder trained with an event-level sample-pair score; (iii) an auditable fixed-marginal intervention and matched training ablations that examine the dependence and supervision mechanisms; and (iv) a reproducible evaluation contract connecting map quality, event scores, and selective risk. The empirical support has different strengths across the questions: the newer place-isolated development study supports the independent-sampling comparison, whereas the complete ablation and fixed-marginal evidence remains historical and exploratory. We do not claim the first correlated occupancy model, a general proof of calibration, or uniformly superior safety.

## 2. Problem Formulation

Let the finite grid be \(\Omega\), the supplied valid-support domain be \(V\subseteq\Omega\), and the observed free and blocked cells be \(O_f\) and \(O_b\). The hidden evaluation region is \(H=V\setminus(O_f\cup O_b)\). The input \(X\) encodes free, blocked, and unknown states; the support mask is supplied separately. Unknown is an observation state, not a third physical map class. A binary world \(M(v)\in\{0,1\}\) denotes free space when one. The posterior must satisfy

\[
M(v)=1\quad(v\in O_f\cap V),\qquad
M(v)=0\quad(v\in O_b\text{ or }v\notin V).
\]

For input-chosen terminals \((s,g)\) and integer disk radius \(r\), define \(\rho_r(M;s,g)\) as the indicator of four-connected reachability after footprint erosion. The desired forecast is

\[
q_\theta(s,g,r\mid X,V)=\mathbb E_{M\sim p_\theta(\cdot\mid X,V)}[\rho_r(M;s,g)].
\]

This is a probability under the learned conditional model. Whether it agrees with empirical event frequencies is an evaluation question, not an architectural guarantee. It is not the probability of a particular trajectory, and no control sequence is returned.

**Why cell marginals are insufficient.** Consider a route that requires two hidden cells to be free, each with marginal probability one half. If their states always agree, the route event has probability one half; if independent, it has probability one quarter; if exactly one is free, it has probability zero. These distributions have identical cell marginals. This elementary example establishes non-identifiability from marginals alone; it is an analytical illustration, not an experimental result.

## 3. Method Overview

ConPath first encodes the partial BEV raster. A stochastic decoder then produces complete binary worlds sharing observed evidence and the same support boundary. Each world induces an exact footprint-feasible graph. Averaging binary connectivity events yields a Monte Carlo estimate of \(q_\theta\). During training, reference maps supply both hidden-cell supervision and query-event labels. At inference, the reference map and event labels are unavailable to the model.

A single sampled world is reused across all footprint radii for a query. The sampler is independent of the query terminals: different queries can therefore be answered from the same world ensemble. This separates map inference from query evaluation while letting the event objective influence the map posterior during learning. The architecture and implementation provenance are documented in [the figure inventory](PAPER_FIGURES.md).

![ConPath implementation diagram](results/paper_figures/20260910T063236.828092Z/01_architecture.png)

*Implemented original ConPath: known-class logit conditioning, spatially correlated logits, independent class noise, and exact footprint events. This is a code schematic, not measured experimental data.*

## 4. Correlated Stochastic Occupancy Posterior

A compact U-Net with feature width 16 produces a feature field \(F_\theta(X)\). Global pooled context and coordinate features are part of the existing encoder. Three decoder heads produce class-location logits \(\mu_c(v)\), bounded low-rank factors \(B_{cd}(v)\), and bounded local noise scales \(\sigma_c(v)\). For world \(k\),

\[
L_{kc}(v)=\mu_c(v)+D^{-1/2}\sum_{d=1}^{D} B_{cd}(v)z_{kd}
 +\sigma_c(v)\,\widetilde{G\!\ast\!\epsilon}_{kc}(v),
\]

where \(z\) and \(\epsilon\) are independent standard Gaussian draws across worlds, \(D=4\), and the local Gaussian kernel is \(5\times5\). The filtered noise is normalized by the in-bounds squared kernel weights, maintaining its pointwise variance near boundaries. The model has 120,108 learned parameters.

The original decoder adds independent per-cell Gumbel class noise to these correlated logits and uses a hard argmax in the forward pass with a Concrete relaxation in the backward pass. The implementation encodes observed and invalid cells as known classes and replaces their class logits by +10/−10 before categorical sampling. At the registered unit categorical-noise scale, the 20-logit gap exceeds the possible class-noise difference after uniform draws are clipped to [10⁻⁶, 1−10⁻⁶]; the resulting hard worlds therefore preserve these constraints. Saved-world audits verify zero violations. Continuous marginal maps are separately projected by the evaluator. This is a property of the registered implementation, not an assertion of an arbitrary-scale final projection layer. Global and local logit dependence therefore coexist with independent final categorical noise; coherent wall geometry is not guaranteed. The two-seed spatially correlated categorical-noise variant is a separate development experiment, not the selected paper method.

With ideal untruncated Gumbel noise and positive categorical-noise scale \(a\), the conditional class probability is \(\mathrm{softmax}(L_k/a)\); clipping the uniform draws introduces a numerical approximation to that identity in the implementation. Its average across latent worlds estimates the model marginal. \(\mathrm{softmax}(\mu)\) is generally not that marginal. We distinguish this continuous conditional-probability estimate from the empirical fraction of hard worlds that mark a cell free; archived map metrics must be interpreted using their recorded estimator. Both are distinct from the query's path-event probability.

## 5. Footprint-aware Reachability

Let \(D_r=\{u\in\mathbb Z^2:\|u\|_2^2\leq r^2\}\). A center cell is valid for the robot if its translated disk lies entirely inside sampled free support:

\[
F_r(M)=\{v\in\Omega:v+D_r\subseteq V\cap\{u:M(u)=1\}\}.
\]

We set \(\rho_r(M;s,g)=1\) precisely when both endpoints belong to \(F_r(M)\) and share a four-connected component. Endpoints invalid for the footprint produce event zero. Out-of-map cells are blocked. All reported radii are **grid cells**, specifically 0, 10, and 20 on FlatLands; no independently verified conversion to meters is assumed.

For \(K\) actual worlds,

\[
\widehat q_r=\frac1K\sum_{k=1}^K\rho_r(M_k;s,g).
\]

Exact evaluation is checked against the project's discrete geometric oracle. The training backward path uses bounded relaxed propagation and is not a collision-certified differentiable planner.

**Radius monotonicity.** If \(r_2\geq r_1\), then \(D_{r_1}\subseteq D_{r_2}\), hence \(F_{r_2}(M)\subseteq F_{r_1}(M)\). Any path valid for \(r_2\) is valid for \(r_1\), so \(\rho_{r_2}\leq\rho_{r_1}\). Averaging the same worlds preserves this inequality exactly. This structural property does not imply calibrated probabilities at either radius. Direct-query predictors have no corresponding architectural guarantee.

## 6. Training Objectives

The public-data stochastic models use

\[
\mathcal L=\mathcal L_{\mathrm{map\mbox{-}NLL}}+0.1\mathcal L_{\mathrm{vario}}+2\mathcal L_{\mathrm{event\mbox{-}U}}.
\]

The map log score applies only to hidden valid cells. The variogram term compares reference pair differences with sampled pair differences at fixed offsets. This encourages selected spatial dependencies but cannot identify an arbitrary joint distribution. Variogram scoring has a broader probabilistic-forecasting foundation. [Scheuerer and Hamill](https://repository.library.noaa.gov/view/noaa/22327)

For query label \(y\in\{0,1\}\) and binary sample events \(z_k\), the event term is

\[
\mathcal L_{\mathrm{event\mbox{-}U}}
=\frac{\sum_{k\neq\ell}z_kz_\ell}{K(K-1)}-\frac{2y}{K}\sum_k z_k+y^2.
\]

For independent world draws conditional on the observation, this is an unbiased estimator of \((q_\theta-y)^2\). In contrast, the expected squared error of the finite-ensemble mean contains an additional \(q_\theta(1-q_\theta)/K\) term. An individual sample-pair estimate can be negative; the expected target remains nonnegative. Evaluation uses ordinary Brier, not this training estimator. Proper-score theory motivates the objective, but finite data, limited model capacity, and approximate gradients prevent a calibration guarantee. [Gneiting and Raftery](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf)

Hard sampled events use a straight-through backward surrogate with at most 256 propagation steps. In the newer parent-isolated experiment, exact hard connectivity replaces any truncated forward event while the bounded surrogate supplies gradients. The historical cohort used a bounded training/selection operator and a separate exact final evaluator. These protocols are not interchangeable.

In the newer experiment, AdamW uses learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\), gradient clipping at 5, effective batch 4, and micro-batch 2. Training uses \(K=4\); model selection uses \(K=16\) on 25 separate selection places. The frozen budget is at most 24 epochs/600 updates, with its recorded early-stopping rule. The selected checkpoint minimizes selection Event Brier, breaking ties within \(10^{-5}\) by map NLL. Two zero-query training places expose a known micro-batch event-weighting limitation shared by the stochastic methods. We retain the archived models and disclose this limitation rather than silently changing the objective after evaluation.

## 7. Baselines

The intended compact comparison contains six internal methods:

1. **Deterministic occupancy:** a completion network trained on map loss, thresholded at the archived fixed threshold, followed by the same footprint connectivity evaluator.
2. **Independent Bernoulli:** a matched stochastic completion model with global factors disabled and a one-cell local kernel; map and event training terms remain enabled. It is a separately trained independent model, not an exactly marginal-matched intervention.
3. **Direct-query:** a classifier predicting the start–goal–radius event without a sampled map. Its existing historical result has one training seed; the distinct coordinate-query control is not relabeled as three direct-query repeats.
4. **No-global:** the full decoder without its low-rank global factors. Encoder context, local correlation, and the event loss remain present.
5. **No-event:** the full stochastic architecture with event-loss weight zero. Historical checkpoint selection still uses Event Brier, so the contrast concerns training supervision, not complete removal of event information.
6. **ConPath:** the original correlated decoder with map, variogram, and event objectives.

All numerical comparisons require common inputs, labels, query keys, support and hidden masks, radii, split, weighting, and evaluator contract. Available evidence does not provide all six methods on the newer place-isolated split. Missing cells remain missing. The historical cohort contains the requested six-row comparison, explicitly marked as a diagnostic table. The stronger same-checkpoint deterministic mean-map control is retained in the text and supplementary tables even though it is outside the compact six-row list.

A separate **fixed-marginal shuffle** independently permutes world indices at every cell of the saved ConPath ensemble. It preserves each cell's empirical free count exactly while disrupting their alignment across worlds. This is a finite-ensemble dependence intervention, not a claim that each shuffled world is an independent draw from an exactly factorized infinite posterior.

External completions are discussed in Related Work and an under-convergence supplement only. LaMa BEV adaptation and FM+XAttn literature reimplementation are not official pretrained checkpoints or adequately converged principal baselines.

## 8. Experimental Protocol

### 8.1 Evidence cohorts and data separation

The primary data source is FlatLands, a partial-view BEV floor-completion benchmark. Its published completion task is related but has different targets and metrics from our path-event evaluation. [FlatLands v3](https://arxiv.org/abs/2603.16016v3)

**Parent-isolated development cohort.** Five indoor sources—3RScan, ARKitScenes, Matterport3D, ScanNet, and ZInD—contribute 20/5/8 parent places each to training/selection/validation, totaling 100/25/40. Each place contributes one fixed observation. Parent-place overlap and cross-split exact D4 duplicate checks pass. Input-defined query generation precedes target inspection and retains goals whose reference cell is blocked. The 40 validation places contain 515 queries and 1,545 radius-conditioned events. Seeds are the existing **20260910, 20260911, and 20260912**. These identifiers are not replaced with the different seeds requested for a possible new optimization study.

The original nine model runs finished and fixed their checkpoints before validation scoring. Data-preparation quality checks had already inspected the development references, and validation was later reused to choose a sampling modification. Thus, the cohort is development validation, not a pristine confirmatory test. It also has a small training budget that does not establish full convergence.

**Historical diagnostic cohort.** The archived six-method comparison uses 160 training and 160 validation packets, 142 event-contributing validation subscenes, 1,408 terminal groups, and 4,224 events. Full, independent, no-global, and no-event use seeds **20260831, 20260901, and 20260902**. A later parent audit found training-place overlap in 27 of the 160 validation observations. Historical metrics are reproducible but cannot substantiate unseen-place generalization. Resampling subscenes does not correct this overlap.

**Two-seed sampling follow-up.** The modified sampler uses the same 100/25/40 cohort and existing seeds 20260910/20260911. Its comparisons are restricted to those two matched seeds and remain supplementary. It is not pooled with the three-seed main development result.

**Stopped external cohort.** The separately frozen 985/97/191 training/calibration/validation-place protocol remains archived. No ConPath result from that cohort is available for the present paper. LaMa/FM staging was stopped when the user froze the paper scope; these results never fill missing internal-baseline rows.

### 8.2 Test lock and access history

No final test is opened for this manuscript consolidation. UnScenes3D `location_6` remains locked. Earlier FlatLands work did access physical archive-test observations: eight occurred in the old training manifest, five in old validation, and an earlier query audit inspected 53 archive-test observations. These counts refer to overlapping histories and must not be added as disjoint totals. We withdraw any historical claim that all physical test assets were untouched. Final holdout eligibility requires a complete access-history audit and exclusion of inspected parent places. Old Boolean `test_evaluated=false` fields do not establish that condition. [Access erratum](site/data/flatlands_read_scope_erratum.json)

### 8.3 Metrics, weighting, and uncertainty

For event rows \(i\) with fixed nonnegative weights \(w_i\) summing to one (\(w_i=1/(S n_s)\) for a row in a parent with \(n_s\) events among \(S\) contributing parents), we report weighted Brier \(\sum_i w_i(\widehat q_i-y_i)^2\), binary NLL with the archived clipping rule, and ECE with the archived bins. Hidden-map Brier applies only to valid unobserved cells, using the specific stored marginal estimator. No bin boundaries, probability thresholds, or query lists are retuned. The parent-isolated experiment originally registered Brier and equal-coverage risk; NLL, ECE, and confidence-0.8 summaries added here are retrospective calculations from its saved validation predictions, using the existing project convention (NLL clipping at 10⁻⁶ and ten equal-width ECE bins). They are not relabeled as preregistered selection criteria. Per-query calibration predictions were not archived, so missing calibration summaries remain unavailable. Low ECE alone can accompany uninformative forecasts, and NLL is particularly sensitive to confidently wrong finite-ensemble zeros and ones.

At confidence \(0.8\), acceptance is \(A_i=\mathbf1\{\widehat q_i\geq0.8\}\). Coverage is \(\sum_iw_iA_i\), and false-safe rate is \(\sum_iw_iA_i(1-y_i)/\sum_iw_iA_i\). If nothing is accepted, risk is undefined, not zero. The same threshold has different available probability bins for different \(K\). Equal-coverage curves additionally compare ranking: every row tied at the boundary receives the same fractional acceptance without using its label. A 30%-coverage risk is not relabeled as risk at confidence 0.8.

The newer cohort weights parent places equally; the historical cohort weights contributing subscenes equally. Seed summaries use sample standard deviations, not confidence intervals. Recorded paired bootstrap intervals retain their original resampling unit and condition on the stated seed averages. Event rows and generated worlds are not treated as independent experimental repetitions. Complete available threshold metrics, radius strata, monotonicity and constraint audits are indexed in [the numeric snapshot](results/paper_validation_snapshot.json).

### 8.4 Freeze at paper convergence

The existing parent-isolated original model beats independent sampling in all three recorded training seeds, satisfying the requested stop condition. Consequently no A/B/C configurations are launched, no temperature calibration is fitted, and no loss, sampler, data split, threshold, or query set is changed. The selected method is the original ConPath family with all three original selection checkpoints; it is not a best-seed pick or an ensemble. The complete checkpoint identities and remaining test requirements are recorded in [FINAL_MODEL_SELECTION.md](results/FINAL_MODEL_SELECTION.md).

## 9. Main Results

### 9.1 Current place-isolated development evidence

Table 1 uses only the 100/25/40 parent-isolated protocol and its three existing seeds. The risk column is explicitly **equal-30%-coverage risk**. Additional metrics from saved predictions are kept in the separate metric tables; absence of a baseline is never represented by zero.

| Method | Actual K | Event Brier ↓ | False-safe at 30% coverage ↓ |
|---|---:|---:|---:|
| Deterministic occupancy | 1 | 0.11152 ± 0.00256 | 0.2516 |
| Independent Bernoulli | 32 | 0.11201 ± 0.00366 | 0.2443 |
| Direct-query | — | Not evaluated on this cohort | — |
| No-global | — | Not evaluated on this cohort | — |
| No-event | — | Not evaluated on this cohort | — |
| ConPath | 32 | 0.10380 ± 0.00350 | 0.2395 |

*Table 1. Validation-only, bounded development experiment. Brier is mean ± sample SD over seeds 20260910/11/12; displayed risks are mean point estimates. Source: `site/data/parent_group_pilot_zh.json`; exact values and hashes are in PAPER_TABLES.md and the snapshot. Dashes denote missing measurements.*

Retrospective calculations from the unchanged saved validation predictions give mean Event NLL/ECE of 0.79330/0.08622 for ConPath and 0.94364/0.10131 for independent sampling. At confidence 0.8, ConPath has mean false-safe rate 0.18649 and coverage 0.21340, versus 0.20847 and 0.22453 for independent sampling. These additional scores describe the selected checkpoints; they were not used to redefine the original selection or screening rules, and lower threshold risk at lower coverage is not a matched-coverage safety claim. The existing train-fitted radius prior is a useful counterexample: its NLL 0.48499 and ECE 0.05653 are lower than ConPath's, although its Brier is higher and its confidence-0.8 coverage is zero. Its false-safe rate at that threshold is undefined. This rule remains in the supplementary table; we do not claim the lowest NLL or ECE among all available controls.

The independent-minus-ConPath Brier difference is 0.00821, with the recorded source-stratified paired parent interval [0.00439, 0.01251], conditional on the three seed means. Against deterministic completion the corresponding difference is 0.00772, with interval [−0.01616, 0.03377]. This supports the independent-sampling comparison within the available development cohort, but not a stable advantage over deterministic completion.

Difficulty is not uniform. In a recorded post-evaluation diagnosis, 750/1,545 events are forced unreachable by the observed map; ConPath has zero error on that subset. On events mutable by hidden completion, its parent-weighted Brier is 0.19912. The smaller overall Brier partly reflects easy known negatives. Unknown-region per-sample free-space IoU is 0.62238 for ConPath versus 0.66228 for deterministic completion; improved event accuracy does not imply better map reconstruction on every metric. In the saved continuous-probability estimator, hidden-map Brier is 0.15147 for ConPath, 0.15158 for independent, and 0.13869 for deterministic completion. The deterministic model's worse thresholded binary-map Brier must not be substituted for its better continuous score to manufacture a probabilistic mapping advantage.

A previously saved post-hoc mean-map analysis on the same current checkpoints uses the fixed 0.5 threshold. ConPath's conditional mean map has Event Brier 0.11410 and risk at 30% coverage 0.27791; the independent checkpoint's conditional mean map has Brier 0.11640 and risk 0.27739. These derived controls were not original registered screening rows and do not replace the separately trained deterministic baseline. Their saved report is retained in T1e, without fresh inference or a retuned threshold.

### 9.2 Historical six-method diagnostic

The complete historical six-row table is retained in [PAPER_TABLES.md](PAPER_TABLES.md), with the overlap caveat in its caption. ConPath's Event Brier/NLL/ECE are 0.06749/0.28854/0.05715; independent sampling obtains 0.09521/0.78503/0.08720. The direct-query control, available for one seed, has ECE 0.04076, below ConPath's three-seed mean. Consequently the evidence cannot support an all-metric or uniformly better direct-query-calibration claim. Counts and uncertainty remain visible rather than presenting one direct-query seed as three repetitions.

The same-checkpoint ConPath mean map is an important counterexample to a broad stochastic-versus-deterministic claim: its Brier is 0.06957, near the stochastic value 0.06749, and the direction reverses in one seed. These historical findings describe a reused, overlapping development cohort, not independent-building generalization.

![Event-score comparison](results/paper_figures/20260910T063236.828092Z/03_event_metrics.png)

*Event Brier, NLL, and ECE. The upper row is the original three-seed comparison; the lower row is the distinct two-seed sampling exploration. Dots are individual seeds and bars are seed SD. NLL/ECE are retrospective summaries of unchanged validation predictions.*

## 10. Ablation Study

### 10.1 RQ1: dependence at fixed empirical marginals

In the saved historical K=128 samples, independently permuting world indices at every cell leaves each empirical occupancy marginal exactly unchanged. Event Brier increases from 0.06749 ± 0.00936 to 0.16139 ± 0.00677. The intervention therefore shows that the original sample alignment contains useful event information beyond its finite-sample marginals. It does not prove the learned posterior is the true conditional distribution or establish this effect on an untouched test cohort.

Separately trained ConPath and independent models have different hidden-map Brier, 0.15451 and 0.16789 in the archived marginal estimator. Their comparison alone cannot answer a strictly fixed-marginal RQ1. The paired permutation supplies that narrower mechanism test. The newer place-isolated comparison supplies complementary development evidence, but has no newly executed fixed-marginal intervention in this consolidation.

![Historical fixed-marginal intervention](results/paper_figures/historical_aggregate/flatlands_clean_marginal_shuffle.svg)

*Historical diagnostic only: original K128 samples versus cellwise world-index permutations with exactly preserved empirical marginals. This plot predates the parent-overlap discovery and does not establish unseen-place generalization.*

### 10.2 RQ2: event supervision and global factors

The historical matched retrainings use the same three seeds and common query/evaluator contract. Removing the event loss increases Event Brier to 0.20425 ± 0.00322, NLL to 2.53067 ± 0.05699, and ECE to 0.21690 ± 0.01933. Removing global decoder factors gives 0.09499 ± 0.00157, 0.78486 ± 0.05495, and 0.08592 ± 0.00058, respectively. Every recorded within-seed subscene-bootstrap Brier interval favors the full model. These are conditional historical diagnostics; the place-overlap finding prevents a confirmatory interpretation.

The no-event model assigns zero reachability to every radius-20 validation query across all three seeds, despite an archived scene-weighted positive rate of 20.67%. This is consistent with fragmented sampled support destroying finite-footprint connections. It does not imply that all map-only completion architectures must fail. No-global removes low-rank decoder factors only; it does not remove the encoder's global context.

![Historical training ablations](results/paper_figures/historical_aggregate/flatlands_clean_training_ablations.svg)

*Historical diagnostic only: full/no-event/no-global on their common archived cohort. The later audit found train-place overlap in 27/160 validation observations; the displayed old subscene intervals are not independent-building intervals.*

### 10.3 Archived sampling modification

The two-seed correlated categorical-noise variant obtains K32 Brier 0.09191 versus 0.10325 for the original model on the same two seeds. The recorded original-minus-variant interval [−0.00117, 0.02437] includes zero. Empirical hard-world hidden-map Brier rises from 0.15192 to 0.15362, and at K4 the 30%-coverage risk rises from 23.73% to 23.84%. The variant therefore does not establish the requested stable multi-metric improvement and is retained as supplementary exploration. No third seed or additional search is initiated.

## 11. Calibration and Selective Risk

Reliability diagrams display predicted event scores against observed event frequencies using the saved weighting and bins. They must be read with their counts, radius composition, and cohort labels. The archived historical reliability evidence supports lower overall event error relative to independent sampling and the two training ablations, but it does not support uniformly improved selective risk.

At confidence 0.8 on the historical cohort, ConPath's mean false-safe rate is 0.04552 at coverage 0.33644; independent sampling has risk 0.05689 at coverage 0.36592. At equal 30% coverage the risks are 0.03600 and 0.03901, and every recorded within-seed paired risk interval includes zero. At the fixed 0.8 threshold, no-event actually has a lower mean false-safe rate, 0.02561, than full ConPath, but also much lower mean coverage, 0.19316. No-event and no-global have mean equal-coverage risks 0.05655 and 0.04288, but their paired risk intervals also include zero. No-global's point contrast reverses in two seeds.

These results answer only part of RQ2: overall probabilistic scores improve in the historical ablation, while a robust false-safe improvement is not established. A lower risk at a fixed threshold can reflect lower coverage. No claim of a safety guarantee follows from any of these point estimates. Post-hoc temperature or threshold fitting is not performed during manuscript consolidation.

![Event reliability](results/paper_figures/20260910T063236.828092Z/04_reliability.png)

*Saved event forecasts and observed frequencies on the parent-isolated development cohort; each seed remains visible. Connecting segments are visual guides, not calibrated fits.*

![Selective risk and coverage](results/paper_figures/20260910T063236.828092Z/05_risk_coverage.png)

*Equal-parent selective risk with fractional, label-independent handling of score ties. Coverage is the fraction of accepted query events, not the confidence threshold.*

## 12. Qualitative Results and Footprint Analysis

The figure bundle reuses stored predictions and previously recorded case lists. The newer gallery fixes ten parent-place examples across the five indoor sources using input identifiers and fixed query ordering. Comparisons preserve the same observation, reference, seed, radius, and first sampled world; they do not select the best-looking sample. Earlier historical positive/failure examples were selected using outcomes and came from the overlap-affected cohort. They remain traceable in the archive but are excluded from the new figure bundle; they are not presented as blind-selected main evidence.

Each map distinguishes observed free space, observed blocked space, unknown valid cells, and closed invalid support. Posterior sample panels show actual binary worlds. Cell-probability panels show the recorded marginal estimate, and query annotations show a separate event forecast. S and G identify query terminals; a straight segment between them would not constitute a predicted path. Any footprint-eroded panel is explicitly labeled as robot-center support rather than the original occupancy map.

Archived historical no-event/no-global/full panels illustrate how similar-looking cell probabilities can produce different eroded connected components, but their outcome-based selection prevents including them as unbiased main qualitative evidence. No matching no-event/no-global worlds were saved for the newer cohort, so its requested qualitative ablation panel remains unavailable. Unreachable examples are retained with their source and selection rationale. No strict interior-bottleneck example is available among the ten existing fixed queries after requiring both endpoints to fit the footprint; this missing qualitative category is disclosed rather than filled by a new outcome-selected example. Radius curves report the actual 0/10/20-cell strata. Although shared-world connectivity guarantees non-increasing forecasts with radius, the archived no-event collapse and radius/source differences show that this is weaker than empirical reliability. No preregistered occlusion-stratified sweep or complete narrow-bottleneck validation figure set is available. RQ3 is therefore partially supported by an exact structural property and diagnostic figures, with remaining evidence gaps recorded in [PAPER_FIGURES.md](PAPER_FIGURES.md).

The current three-seed radius breakdown exposes an additional limitation hidden by the aggregate scores. At radii 0, 10, and 20, original ConPath's mean Brier is 0.15329, 0.09668, and 0.06142, compared with 0.16474, 0.10249, and 0.06880 for independent sampling. Deterministic completion is better at radius 20, with Brier 0.04025. At that radius, ConPath accepts no query at confidence 0.8 in any seed: coverage is zero and false-safe rate is undefined. At radius 10 its mean false-safe rate is 0.30851 at only 0.03547 coverage. Thus, lower aggregate Brier and structural radius monotonicity do not establish reliable high-confidence predictions across footprints. These are descriptive views of the same frozen predictions, with parent weights normalized within each radius; no radius, threshold, or query was changed. Full seed-level values are provided in [the radius table](PAPER_TABLES.md#t1f-current-footprint-strata-and-abstention) and [the portable data index](PAPER_DATA_PACKAGE.md).

![First previously fixed query and posterior samples](results/paper_figures/20260910T063236.828092Z/02_fixed_case_00.png)

*The first case in the existing fixed ten-case list, obs_014028:q024, seed 20260910, radius 10 cells. Original ConPath and independent samples appear above the supplementary sampling variant. These are the first four saved worlds in order; query scores use all 32 worlds. All ten cases remain in the figure inventory, including failures.*

![Footprint-radius results](results/paper_figures/20260910T063236.828092Z/06_radius.png)

*Brier at the fixed 0/10/20-cell radii; the same query set and saved worlds are used. Monotonic probabilities do not imply equal error or equal calibration across radii.*

## 13. Limitations

**Development evidence and historical contamination.** The current principal results are validation-only. The newer 40-place cohort has been reused for development and is small; the older full ablation cohort overlaps training parent places and includes historical archive-test access. No final test has been opened in this consolidation, and final holdout eligibility has not been established. Existing subscene confidence intervals cannot be reinterpreted as independent-building uncertainty.

**Incomplete matched baseline matrix.** The newer split lacks direct-query, no-event, and no-global runs, while the historical direct-query result has only one seed. We neither mix the two cohorts nor start replacement training. External generative models have not served as sufficiently converged formal main baselines. We make no ranking claim against LaMa, FM+XAttn, or published navigation systems.

**Task and sensor scope.** Inputs are supplied BEV maps and valid-support masks, not end-to-end RGB or LiDAR. Current primary evidence concerns indoor FlatLands development protocols. Outdoor generalization, real-robot closed-loop operation, and historical-frame reasoning are unproven. The method forecasts path existence and does not directly output a control trajectory.

**Geometry and computation.** Footprints are discrete circular disks with four-neighbor motion; nonholonomic dynamics, arbitrary robot orientation, and continuous collision checking are outside the model. Radius units remain grid cells because physical-scale calibration is unresolved. Bounded relaxed gradients are approximate. Finite-world probability estimates have sampling noise and clipped NLL depends on the recorded clipping convention. No controlled-device throughput superiority is asserted.

**Learning and inference limitations.** The parent-isolated experiment has at most 600 updates, a known zero-query micro-batch event-weighting limitation, and no claim of sufficient convergence for every model. The original categorical noise can fragment narrow passages. Better Brier does not imply uniformly better NLL, ECE, map quality, or false-safe risk; deterministic controls remain strong, and original ConPath has zero confidence-0.8 coverage at radius 20 on current development validation. Event-based selection of the no-event checkpoint limits attribution to the training-loss intervention. Fixed-marginal shuffling is a saved-sample intervention rather than a complete causal account of separately trained architectures.

## 14. Related Work

**Probabilistic mapping and planning.** MRFMap uses a Markov random field and forward ray sensor models for occupancy inference, establishing that dependencies in occupancy mapping are not new. ConPath studies a learned hidden-region posterior under an already supplied BEV observation and scores a downstream footprint-conditioned event. [Shankar and Michael](https://www.roboticsproceedings.org/rss16/p060.html) Topology-informed growing neural gas uses probabilistic map topology to construct navigation roadmaps, and uncertainty-aware planning can reason about alternative paths and information-gathering actions. Those outputs and experiments differ from a calibrated scalar path-existence forecast. [Saroya et al.](https://doi.org/10.1109/LRA.2021.3068886), [Banfi et al.](https://arxiv.org/abs/2205.14251)

**Connectivity supervision.** MALIS trains affinity graphs through maximin connectivity rather than only individual-edge error. Learned promising-region methods also supervise connectivity for sampling-based planning. These precedents motivate task-sensitive supervision and prevent claiming connectivity learning itself as novel. ConPath combines a stochastic map posterior, a robot-footprint event, and probabilistic scoring; its bounded surrogate is not presented as a new general maximin algorithm. [Turaga et al.](https://arxiv.org/abs/0911.5372), [Ma et al.](https://arxiv.org/abs/2112.08106)

**Generative map completion.** FlatLands directly studies partial-view floor completion using deterministic and stochastic completion families, including flow-based methods. It is a close data/task neighbor, so our contribution cannot be the mere generation of multiple BEV maps. Our comparison asks whether their joint structure supports a specific reachability forecast. [Bhattacharjee et al.](https://arxiv.org/abs/2603.16016v3) LaMa addresses large-mask image inpainting using Fourier convolutions and a wide-receptive-field training design. Its BEV adaptation is a distinct implementation contract. [Suvorov et al.](https://arxiv.org/abs/2109.07161) Our LaMa BEV adaptation and FM+XAttn literature reimplementation received incomplete convergence-stage experiments and were stopped before adequate three-seed confirmation. They remain related methods and supplementary engineering records, with no numerical superiority claim.

**Probabilistic evaluation.** Proper scoring rules evaluate forecasts against realized outcomes; variogram scores address aspects of multivariate dependence, and temperature scaling is an established post-hoc calibration method. ConPath uses these ideas to distinguish map-marginal quality, event accuracy, and selective risk. Temperature scaling is not newly implemented or fitted here, and lower binned ECE is not equated with a safety certificate. [Gneiting and Raftery](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf), [Scheuerer and Hamill](https://repository.library.noaa.gov/view/noaa/22327), [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)

Published external mIoU, FID, and navigation-success values are omitted from our numerical event tables. Where cited in the archived literature review, they refer to **different datasets, tasks, outputs, and metrics and cannot be compared directly** with ConPath Event Brier.

## 15. Conclusion

ConPath connects a joint stochastic map posterior to the probability of footprint-feasible start–goal reachability. The current evidence supports the importance of evaluating map dependence at the event level: a small place-isolated development study improves on independent sampling, and a separate historical fixed-marginal intervention and training ablations expose the roles of sample structure and event supervision. The evidence does not establish uniformly better selective risk, superiority to strong deterministic completion, or final-test generalization. We freeze the original method and the existing checkpoints, stop additional tuning and external training, and preserve these limits in the manuscript. The remaining submission question is the sufficiency of the documented validation evidence and a properly audited final holdout, not whether further parameter searches can improve every reported decimal.

## Reproducibility and manuscript assets

The [evidence register](PAPER_EVIDENCE.md), [tables](PAPER_TABLES.md), [figures](PAPER_FIGURES.md), [JSON snapshot](results/paper_validation_snapshot.json), [CSV snapshot](results/paper_validation_snapshot.csv), and [model-selection record](results/FINAL_MODEL_SELECTION.md) bind results to their original protocols, seeds, checkpoints, and saved predictions. No training, new model inference, raw test-data reading, threshold fitting, or query revision is required to regenerate this manuscript bundle. Source references above identify prior work; all ConPath numbers derive exclusively from local auditable experiment records.
