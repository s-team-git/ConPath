# Paper tables / 论文数字表

2026-09-10 · 所有数字为已有验证/历史诊断；无最终测试成绩。不同cohort、K和seed集合不混排。±为训练seed样本标准差，不是置信区间。原报告SHA见文末，详细预测来源与补算指标见 [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md) 及 [snapshot](results/paper_validation_snapshot.json)。

## T1. Current parent-isolated development comparison

100 training / 25 selection / 40 validation parent places; same 515 queries × radii 0/10/20 cells, 1545 events. Seeds 20260910/20260911/20260912. Parent-weighted, validation-only, bounded training. Missing methods were not run on this cohort.

| Method | Actual K | Seeds | Event Brier ↓ | Risk at 30% coverage ↓ | Hidden-map Brier ↓ |
|---|---:|---:|---:|---:|---:|
| Deterministic occupancy | 1 | 3 | 0.11152 ± 0.00256 | 0.25164 ± 0.01024 | 0.18988 |
| Independent Bernoulli | 32 | 3 | 0.11201 ± 0.00366 | 0.24432 ± 0.00928 | 0.15402 |
| Direct-query | — | 0 | Not evaluated | — | — |
| No-global | — | 0 | Not evaluated | — | — |
| No-event | — | 0 | Not evaluated | — | — |
| ConPath | 32 | 3 | 0.10380 ± 0.00350 | 0.23952 ± 0.00578 | 0.15388 |

Hidden-map Brier above uses the stored **hard-world free-frequency** estimator on unknown valid cells; deterministic K1 is its binary-map squared error, not continuous marginal Brier. Risk30 is not risk at confidence 0.8. The continuous native marginal metric, if saved, is separately named in the snapshot.

The source-stratified paired parent interval for independent minus ConPath is [0.00439, 0.01251]; deterministic minus ConPath is [−0.01616, 0.03377]. Intervals condition on the three recorded seed means. The 40 validation places were later reused for development.

## T1b. Retrospective event and marginal-probability diagnostics

Same fixed predictions, same three training seeds as T1. NLL/ECE/confidence-0.8 are newly summarized descriptive metrics, not original selection criteria. NLL clips at 1e-6, ECE uses ten equal-width bins; no calibration is fitted.

| Method | Actual K | Event NLL ↓ | Event ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 | Continuous hidden-map Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic occupancy | 1 | 1.54065 ± 0.03537 | 0.11152 ± 0.00256 | 0.25164 ± 0.01024 | 0.34198 ± 0.00883 | 0.13869 ± 0.00542 |
| Independent Bernoulli | 32 | 0.94364 ± 0.06507 | 0.10131 ± 0.00578 | 0.20847 ± 0.01295 | 0.22453 ± 0.01525 | 0.15158 ± 0.00852 |
| ConPath | 32 | 0.79330 ± 0.10289 | 0.08622 ± 0.00315 | 0.18649 ± 0.01396 | 0.21340 ± 0.00842 | 0.15147 ± 0.00378 |

Continuous hidden-map Brier uses the saved original sigmoid probabilities for deterministic and the saved K32 conditional-probability average for the stochastic models. **Deterministic continuous Brier is better here**; its thresholded binary Brier in T1 must not be substituted to claim better probabilistic map quality. No continuous K4 re-inference is performed.

## T1c. Frozen sample budget and constraint checks

K4 is the saved nested prefix of K32, with identical queries and checkpoints. All 23 current method/seed/budget score files have zero radius-monotonicity violations across their 515 query triples (including rules); this is a structural check, not calibration. The existing saved-world audits report zero observed-evidence and valid-support violations for evaluated worlds; source/extent are detailed in PAPER_EVIDENCE.md.

| Method | K | Brier ↓ | ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 | Radius monotonicity violations |
|---|---:|---:|---:|---:|---:|---:|
| ConPath | 4 | 0.11064 ± 0.00666 | 0.09633 ± 0.00698 | 0.17801 ± 0.01356 | 0.19624 ± 0.00358 | 0 |
| ConPath | 32 | 0.10380 ± 0.00350 | 0.08622 ± 0.00315 | 0.18649 ± 0.01396 | 0.21340 ± 0.00842 | 0 |
| Independent Bernoulli | 4 | 0.12028 ± 0.00844 | 0.10985 ± 0.01085 | 0.20673 ± 0.01934 | 0.21058 ± 0.01425 | 0 |
| Independent Bernoulli | 32 | 0.11201 ± 0.00366 | 0.10131 ± 0.00578 | 0.20847 ± 0.01295 | 0.22453 ± 0.01525 | 0 |

## T1d. Existing training-free/radius-prior controls (supplement)

The single train-fitted radius prior has lower NLL/ECE than ConPath but accepts no event at confidence 0.8. Its risk is undefined, not zero. Rules are one fixed evaluation each, not optimization-seed repeats. These controls remain visible without expanding the requested six-method main table.

| Control | Actual maps | Brier ↓ | NLL ↓ | ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 |
|---|---:|---:|---:|---:|---:|---:|
| All unknown free | 1 | 0.18048 | 2.49341 | 0.18048 | 0.39087 | 0.46174 |
| All unknown blocked | 1 | 0.28126 | 3.88578 | 0.28126 | — | 0.00000 |
| Nearest observed | 1 | 0.13834 | 1.91117 | 0.13834 | 0.30460 | 0.36573 |
| Train-fitted radius prior | N/A | 0.15606 | 0.48499 | 0.05653 | — | 0.00000 |

Source for T1b–T1d: [results/paper_validation_snapshot.json](results/paper_validation_snapshot.json), SHA-256 `8a3b81ef1f01dec72e57f92d7c02d39ed0e3697e12389143a755a836cc4bbc32`. Per-radius, source, reachable/unreachable, pooled and parent-weighted summaries are preserved without changing the original query set.

## T1e. Previously saved mean-map and observation-constraint diagnostics

This post-hoc report predates manuscript consolidation. It uses the same current three checkpoints, references, query keys and fixed 0.5 threshold. It was not an original registered screening row, and is not a newly run evaluation. NLL/ECE were not stored for these derived controls and are not fabricated.

| Derived output | Actual maps | Event Brier ↓ | Risk at 30% coverage ↓ |
|---|---:|---:|---:|
| ConPath same-checkpoint conditional-mean threshold | 1 | 0.11410 ± 0.00301 | 0.27791 ± 0.00724 |
| Independent same-checkpoint conditional-mean threshold | 1 | 0.11640 ± 0.00694 | 0.27739 ± 0.01524 |

ConPath Brier on observation-mutable events: 0.19912 ± 0.00967. The known-immutable subset is 750/1545 events (unweighted), with zero ConPath error. Full-cohort and subset metrics have their own parent normalization.
Source: [results/parent_group_pilot_v1/diagnostics/report.json](results/parent_group_pilot_v1/diagnostics/report.json), SHA-256 `5da858eaae25681f6f8c1428393e8612bfb35161ecc7391dad806fb132375363`.


## T1f. Current footprint strata and abstention

Same fixed three seeds, 40 development parents, and 515 events per radius. Values below are means; per-seed values and sample SD remain in the linked CSV. All parent weights are renormalized within the radius. No new evaluation or threshold fitting.

| Radius (cells) | Method | K | Event Brier ↓ | False-safe @0.8 ↓ | Coverage @0.8 |
|---:|---|---:|---:|---:|---:|
| 0 | Deterministic occupancy | 1 | 0.18790 | 0.19302 | 0.66478 |
| 0 | Independent Bernoulli | 32 | 0.16474 | 0.18898 | 0.59613 |
| 0 | ConPath | 32 | 0.15329 | 0.17754 | 0.60473 |
| 10 | Deterministic occupancy | 1 | 0.10639 | 0.39336 | 0.24830 |
| 10 | Independent Bernoulli | 32 | 0.10249 | 0.34346 | 0.07316 |
| 10 | ConPath | 32 | 0.09668 | 0.30851 | 0.03547 |
| 20 | Deterministic occupancy | 1 | 0.04025 | 0.28529 | 0.11286 |
| 20 | Independent Bernoulli | 32 | 0.06880 | 0.41239 | 0.00430 |
| 20 | ConPath | 32 | 0.06142 | Undefined | 0.00000 |

ConPath has **zero coverage at confidence 0.8 for radius 20 in all three seeds**; its false-safe rate is undefined, not zero. Deterministic occupancy has lower Brier at radius 20. Monotone scores and lower aggregate error therefore do not establish calibrated high-confidence predictions across footprints.

Source: unchanged [frozen snapshot](results/paper_validation_snapshot.json), `/development_run_metrics/*/by_radius`; [three-seed strata CSV](results/paper_submission_v1/analysis/primary_three_seed_strata.csv) and [export receipt](results/paper_submission_v1/analysis/metric_view_receipt.json). Source/truth strata and pooled metrics remain available in the [276-row seed-level view](results/paper_submission_v1/analysis/development_metrics_by_seed_and_stratum.csv). Truth-conditioned rows are error diagnostics, not standalone calibration evidence.

## T2. Historical six-method diagnostic (not unseen-place main evidence)

160 train / 160 validation packets; 4224 common event rows from 142 contributing subscenes. The later audit found train-place overlap in 27/160 validation observations and historical archive-test access. Common input/labels/query/radii/support/evaluation contract within this table does not repair that flaw. Do not copy these numbers into T1.

| Method | Actual K | Training seeds | Event Brier ↓ | Event NLL ↓ | Event ECE ↓ | False-safe @0.8 ↓ | Coverage @0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic occupancy | 1 | 20260831,20260901,20260902 | 0.08857 ± 0.00323 | 1.22362 ± 0.04463 | 0.08857 ± 0.00323 | 0.07293 ± 0.00682 | 0.44278 ± 0.01024 |
| Independent Bernoulli | 128 | 20260831,20260901,20260902 | 0.09521 ± 0.00703 | 0.78503 ± 0.04690 | 0.08720 ± 0.00741 | 0.05689 ± 0.00832 | 0.36592 ± 0.01692 |
| Direct-query | N/A | 20260831 | 0.09119 | 0.29788 | 0.04076 | 0.08335 | 0.37412 |
| No-global | 128 | 20260831,20260901,20260902 | 0.09499 ± 0.00157 | 0.78486 ± 0.05495 | 0.08592 ± 0.00058 | 0.05739 ± 0.01662 | 0.36354 ± 0.00957 |
| No-event | 128 | 20260831,20260901,20260902 | 0.20425 ± 0.00322 | 2.53067 ± 0.05699 | 0.21690 ± 0.01933 | 0.02561 ± 0.01609 | 0.19316 ± 0.10435 |
| ConPath | 128 | 20260831,20260901,20260902 | 0.06749 ± 0.00936 | 0.28854 ± 0.01662 | 0.05715 ± 0.00683 | 0.04552 ± 0.01429 | 0.33644 ± 0.00579 |

The direct-query row has only one training seed; it is not the separately trained three-seed coordinate-query control. A dash/NA is never zero. All historical full/no-event/no-global within-seed Brier intervals favor full, but the subscene bootstrap does not establish independent-building generalization. No-event still uses event-based checkpoint selection. No-global retains encoder context and local decoder correlation.

## T3. Historical dependence and strong deterministic checks

Same historical cohort as T2; three seeds 20260831/20260901/20260902. A derived mean map is one actual binary output, computed from the recorded K128 posterior estimate.

| Control | Event Brier ↓ | Event NLL ↓ | Event ECE ↓ | Risk at 30% coverage ↓ |
|---|---:|---:|---:|---:|
| ConPath, K128 | 0.06749 ± 0.00936 | 0.28854 ± 0.01662 | 0.05715 ± 0.00683 | 0.03600 ± 0.00986 |
| Cellwise world-index permutation, K128 | 0.16139 ± 0.00677 | 1.53947 ± 0.18205 | 0.16433 ± 0.01252 | 0.03908 ± 0.01622 |
| ConPath same-checkpoint mean map | 0.06957 ± 0.00380 | 0.96118 ± 0.05253 | 0.06957 ± 0.00380 | 0.10509 ± 0.01468 |
| Independent same-checkpoint mean map | 0.07184 ± 0.00349 | 0.99248 ± 0.04822 | 0.07184 ± 0.00349 | 0.11023 ± 0.01558 |

The permutation preserves every **empirical hard-world cell marginal** and changes dependence. This is distinct from separately trained independent sampling. Mean-map and equal-coverage comparisons are strong counterevidence to broad stochastic/safety superiority. Every historical independent-versus-full equal-30%-coverage risk interval includes zero.

## T4. Two-seed sampling variant (supplementary only)

Same 100/25/40 parent-place development cohort, restricted to seeds 20260910 and 20260911 in every row. Do not compare this two-seed mean with T1’s three-seed mean.

| Method | K | Event Brier ↓ | Risk at 30% coverage ↓ | Free-class sample IoU ↑ |
|---|---:|---:|---:|---:|
| Correlated categorical-noise variant (candidate) | 32 | 0.09191 ± 0.00902 | 0.22604 ± 0.00413 | 0.61919 |
| Original ConPath | 32 | 0.10325 ± 0.00476 | 0.23649 ± 0.00342 | 0.62009 |
| Independent Bernoulli | 32 | 0.11014 ± 0.00239 | 0.24873 ± 0.00748 | 0.62522 |
| Deterministic occupancy | 1 | 0.11132 ± 0.00359 | 0.25047 ± 0.01419 | 0.66254 |

Original minus candidate Brier: 0.01135, recorded paired parent interval [−0.00117, 0.02437]. Candidate is not selected as the paper method; no third seed is added. Its K4 risk30 is 23.84% versus original 23.73%, despite lower Brier.

## Traceability and exclusions

All tables are computed from the following existing reports. No new network evaluation is performed by this renderer. Original prediction hashes, evaluator identity, query agreement, threshold summaries and new/old metric status are retained in the snapshot. External under-convergence numbers, P0 synthetic scores and UnScenes3D results are not inserted into these FlatLands comparisons.

| Source | SHA-256 |
|---|---|
| [results/parent_group_pilot_v1/analysis.json](results/parent_group_pilot_v1/analysis.json) | `0875640afd84a7d5a1446943328ae90e40f3b4cdc8cbe02af04d01d428ee8cc5` |
| [results/paper_clean_analysis_v1/report.json](results/paper_clean_analysis_v1/report.json) | `29ca7640437695dbb432c256dc6a0ab0f176ddfb108c17bd8723e17a223e9de3` |
| [results/paper_clean_ablation_matrix_v1/analysis/report.json](results/paper_clean_ablation_matrix_v1/analysis/report.json) | `d54ade10698106c1b36b11fd73fb3f90ac1da1c70b1b8ab780d0e6c687f2daab` |
| [results/coherent_parent_pilot_v1/analysis.json](results/coherent_parent_pilot_v1/analysis.json) | `6da43493e5e50b975b4d5c08036c85fdd0dca3c6ee2e078e54ff80646f8496b6` |
