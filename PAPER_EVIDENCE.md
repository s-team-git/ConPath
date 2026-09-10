# Evidence ledger for the ConPath paper

Updated 2026-09-10 UTC after the user cancelled further training. This ledger freezes the interpretation of **existing evidence**; it grants no training, test access, external-superiority, or submission approval.

中文说明：目前可以写成论文的是已有验证证据及其限制，而不是补齐了外部正式矩阵。原 ConPath 的父地点隔离三种子结果、coherent 的同两种子补充、历史消融、固定边际机制实验和 synthetic P0 必须分别解释。不能把它们拼成一个“全面领先”的主表。

## 1. Machine-readable evidence and audit scope

- [JSON snapshot](results/paper_validation_snapshot.json): `evidence_rows` contains 29 aggregate rows with source paths, SHA-256 hashes, JSON pointers, actual seeds, budgets, evidence levels and permitted claims. `development_run_metrics` contains 23 existing method/seed/budget reports with independently recomputed event metrics, reliability bins, parent/source/radius/truth strata and monotonicity checks. `development_posthoc_diagnostics` separately binds the existing conditional-mean threshold controls and input-determined/mutable event subgroups; these are not newly registered main rows. `source_registry` records 189 source files.
- [CSV snapshot](results/paper_validation_snapshot.csv): one row per cohort/method/budget aggregate. Empty cells mean unavailable or inapplicable, never zero. Sample SD is across actual optimization seeds; a single rule or seed has no estimated SD. The CSV reports real map counts separately from archived budget labels.
- Existing geometry checks are reused from [parent-pilot verification](results/parent_group_pilot_v1/verification.json) and [coherent independent verification](results/coherent_parent_pilot_v1/independent_verification.json), with their analysis/protocol bindings checked. The former audited 7,920 saved worlds, including 720 explicit-disk geometric checks and 27,810 prediction votes. This snapshot does **not** claim to repeat those geometry audits.
- This audit decoded only 40 sealed, physical-archive-`train` development-validation packets and saved continuous prediction maps. It joined the existing label-free CSV scores to all 1,545 frozen event keys per report, verified their file hashes, and reproduced original equal-parent Brier, parent/source Brier and risk-at-30%-coverage values. It did not perform model inference, optimization, threshold fitting, raw-PNG access, physical-test access, final-test access, or location_6 access.
- Training recipes, complete receipts, histories and actual best-checkpoint byte hashes were retained in each trained run's provenance. Old mixed-cohort evidence below was read as JSON aggregates/audit metadata only, without reopening old prediction rows, worlds or target maps.
- The exact CPU snapshot builder is embedded in `software.snapshot_builder_source`, with its own SHA-256 and Python/NumPy versions. This is a finite saved-prediction reaggregation, not a new experimental run.

## 2. Cohorts that can and cannot share a comparison table

| Evidence level | Data and repeats | Comparisons allowed | Claim boundary |
|---|---|---|---|
| D1: parent-isolated small development | 100 train / 25 calibration / 40 validation parents; seeds 20260910/11/12 | Original ConPath, independent, deterministic and four input/train-only rules under the same frozen observations and queries; stochastic K4 and K32 in separate budget columns; deterministic actual K1 | Current development comparison; maximum 24 epochs, not proof of full convergence or final-test generalization |
| D2: reused-validation candidate | Same parent split; only seeds 20260910/11 | Coherent categorical candidate against the **same two seeds** of original/independent/deterministic | Supplementary exploratory result; validation influenced the development direction; no third seed and no superiority to original established |
| H1: historical diagnostics | 160 training / 160 validation observations, 142 event-bearing subscenes, 4,224 events/seed; mostly seeds 20260831/0901/0902 | Historical ConPath, independent, deterministic completion, direct-query, no-event, no-global and derived mean-map controls, with repeat counts disclosed | Parent overlap and prior physical-test access invalidate unseen-place/final-test claims; direct-query has one seed |
| M1: fixed-marginal conditional mechanism | The historical saved K128 world ensembles and three paired shuffles | Each source ensemble against its own independently cell-shuffled world indices | Exact empirical-marginal preservation isolates a finite-ensemble joint-dependence intervention, not current-cohort generalization or training-loss causality |
| S1: synthetic mechanism check | Four held-out synthetic templates; full model two seeds, no-event one | Synthetic controls within that synthetic protocol | Different inputs, radii and data construction; not comparable numerically to FlatLands |
| External formal matrix | Separate 985 train parents × 2 views / 97 calibration / 191 validation; external K4 and outer seeds 20260831/0901/0902 | No completed fully trained three-seed comparison exists | Cancelled engineering work only; no external main-table claims |

All three groups of seeds above retain their actual identities. Existing 20260910/11/12 runs must not be renamed as the planned external seeds. The same source dataset name does not make two cohorts comparable: parent identities, observation rules, target-query eligibility, weighting, training selection and actual sample counts also differ.

## 3. Current primary evidence: original ConPath, three seeds

The [frozen parent protocol](results/parent_group_pilot_v1/data/protocol.json) selects one observation per parent by source/partition-stratified hash, with 20/5/8 parents per source in train/calibration/validation. Sources are 3RScan, ARKitScenes, Matterport3D, ScanNet and ZInD. All selected assets belong to physical archive `train`; candidate parent partitions are disjoint. These validation parents are now development-known, not fresh final test.

The input is observed-free / observed-blocked / unknown channels with supplied valid support. Start points are observed free; goals are unknown and support-valid. Candidate geometry is input-only and frozen before target-map access. Blocked target endpoints remain negatives; they are not filtered after looking at the reference. There are 515 validation queries, each at radii **0, 10 and 20 grid cells**, giving 1,545 events over 40 parents. Radius is not converted to metres. Exact four-neighbour connectivity uses an integer-disk footprint.

All learned baseline runs use the registered F16 feature width, batch4, learning rate 3e-4, weight decay 1e-4, gradient clip5, maximum24 epochs and calibration-based early stopping. Only the stochastic models use latent4 and train K4; deterministic completion has no stochastic latent and outputs one map. Original loss weights are map1, variogram0.1 and reachability2; deterministic uses unknown-valid map NLL. Independent removes the registered decoder dependence terms; it is not the exact fixed-marginal intervention in Section 5. All baseline checkpoint selections preceded their model-scored validation reports. Calibration uses 25 separate parents and fixed K16 stochastic streams, or one deterministic map: lowest equal-parent Event Brier, map NLL to resolve a tie within 1e-5. Two training parents have no eligible query; the original microbatch event-weighting limitation is disclosed and was not silently replaced by the later formal trainer's fix.

The original ConPath remains the selected development method, using all three registered runs. Their selected epochs are 21/17/22 after 24/23/24 completed epochs. The fixed display seed is 20260910, not the seed with a validation-selected score. Exact paths, recipes, histories and checkpoint hashes are in `development_run_metrics[].checkpoint` and [FINAL_MODEL_SELECTION](results/FINAL_MODEL_SELECTION.md).

The following newly summarized NLL/ECE/risk metrics are **post hoc secondary diagnostics from saved raw scores**. Original registered primary metrics were Event Brier and risk at 30% coverage. NLL clips scores to [1e-6, 1−1e-6]; ECE uses ten equal-width bins, with one included in the last bin; high-confidence acceptance is raw score ≥0.8. Parent weights are recomputed within each reported subset. No calibration curve, Platt mapping or threshold was fit using validation.

| Method | Actual worlds | Event Brier | Event NLL | Event ECE | False-safe at 0.8 | Coverage at 0.8 | Risk at 30% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConPath (original) | 32 | 0.10380 ± 0.00350 | 0.79330 ± 0.10289 | 0.08622 ± 0.00315 | 0.18649 ± 0.01396 | 0.21340 ± 0.00842 | 0.23952 ± 0.00578 |
| Independent completion | 32 | 0.11201 ± 0.00366 | 0.94364 ± 0.06507 | 0.10131 ± 0.00578 | 0.20847 ± 0.01295 | 0.22453 ± 0.01525 | 0.24432 ± 0.00928 |
| Deterministic completion | 1 | 0.11152 ± 0.00256 | 1.54065 ± 0.03537 | 0.11152 ± 0.00256 | 0.25164 ± 0.01024 | 0.34198 ± 0.00883 | 0.25164 ± 0.01024 |
| Train-fitted radius prior | N/A | 0.15606 | 0.48499 | 0.05653 | N/A | 0.00000 | 0.40411 |

Values are equal-parent mean ± sample SD over three optimization seeds; the train-fitted prior is one deterministic fit. False-safe and coverage are fractions, not percentages. The prior's archived `K=1` filename tag does not mean it generates a map: `actual_map_count=null` and map metrics are N/A. Its zero coverage makes false-safe undefined, not zero.

Original ConPath's K32 Brier is lower than independent in each of the three matched seeds. The existing source-stratified parent bootstrap gives independent minus ConPath = 0.008212, 95% interval [0.004391, 0.012508]. Against deterministic, the difference is 0.007717 with interval [−0.016161, 0.033766], which crosses zero. These intervals condition on the displayed seed average, not arbitrary optimization randomness. At K4 the ConPath/independent Brier means are 0.110638/0.120281; these common-budget results must remain visible alongside K32.

The radius-only prior has **lower NLL and ECE** than ConPath while having worse Brier, poorer top-30% risk and no accepted query at 0.8. It must remain in the table: low ECE alone does not establish informative reachability. Hard-world scores can be exactly zero or one; their NLL is sensitive to the declared clipping convention. Raw event frequencies are Monte Carlo model scores, not measured probabilities of safe physical execution.

For all 23 current saved method/seed/budget reports, all 515 query triples satisfy nonincreasing raw scores as radius increases, and all reference label triples satisfy the same ordering: zero violating triples, without postprocessing. This consistency is partly structural for geometry-based outputs; it is not an independent accuracy guarantee. No current direct-query model was evaluated, so this check says nothing about its monotonicity.

**Map estimators must remain separate.** The first column below scores empirical free probabilities from actual binary worlds. The second scores separately saved conditional categorical probabilities over 32 latent draws, or the deterministic network's sigmoid. A saved K32 continuous estimate is not renamed as a K4 estimate. `mean_iou` is the mean **free-class** IoU over worlds, not multi-class mIoU.

| Method | Hidden Brier, binary-world estimator | Hidden Brier, continuous estimator | Hidden free-class IoU |
|---|---:|---:|---:|
| ConPath (original) | 0.15388 | 0.15147 | 0.62238 |
| Independent completion | 0.15402 | 0.15158 | 0.62120 |
| Deterministic completion | 0.18988 | 0.13869 | 0.66228 |

Deterministic completion has better continuous hidden-map Brier and free-class IoU. Thresholding its sigmoid and then comparing the resulting binary map with a 32-world marginal cannot support a claim of universally better map probabilities. The near-identical continuous map Brier of original and independent alongside different Event Brier is consistent with the motivation for event-level evaluation, but their retraining comparison does not isolate dependence at exactly fixed marginals.

Per-query calibration predictions were not saved for this cohort. Training histories retain checkpoint-selection calibration Brier/map NLL scalars; calibration NLL/ECE/confidence reaggregation is therefore **unavailable**. No missing calibration table is filled from validation and no new inference is authorized to reconstruct it.

The existing [post hoc diagnostics](results/parent_group_pilot_v1/diagnostics/report.json) also retain **same-checkpoint conditional-mean threshold controls** for the current three seeds. A fixed, untuned 0.5 threshold is applied to the saved 32-draw continuous mean, with evidence/support projection, producing one map rather than 32 worlds. Original ConPath's mean-map control has Event Brier 0.114101 and risk at 30% coverage 0.277913; independent's corresponding values are 0.116398 and 0.277390. These controls were added after the original evaluation without retraining, so they are saved supplementary diagnostics rather than missing evaluations or prespecified primary comparisons.

That report identifies 750/1,545 input-determined unreachable events (48.54% by unweighted count), on which ConPath has zero Brier. The remaining 795 mutable events span 39 parents; their separately parent-weighted ConPath Brier is 0.199123 and risk at 30% coverage is 0.140514. This subgroup explains why the overall 0.103799 should not be read as the difficulty of only the unknown-dependent queries. Subgroup weights and the risk-ranking population are recomputed within the subgroup, so its scores do not replace the registered full-cohort table. This audit binds the existing report and source hashes; it does not rerun geometry or create new predictions.

## 4. Coherent categorical candidate: matched two-seed supplement

The [coherent protocol](results/coherent_parent_pilot_v1/protocol.json) uses the same parent data and training budget, with a 9×9, sigma3 Gaussian-copula categorical-noise change. It is not a new backbone and its extra compute was not benchmarked as identical cost. The correct original comparator is the original model's first two seeds, not its three-seed average.

| Method, same seeds 20260910/11 | K32 Brier | K32 NLL | K32 ECE | K32 risk at 30% | K4 risk at 30% |
|---|---:|---:|---:|---:|---:|
| ConPath (original) | 0.10325 ± 0.00476 | 0.78457 ± 0.14393 | 0.08711 ± 0.00389 | 0.23649 ± 0.00342 | 0.23728 ± 0.00988 |
| ConPath coherent categorical candidate | 0.09191 ± 0.00902 | 0.55795 ± 0.08265 | 0.07379 ± 0.01474 | 0.22604 ± 0.00413 | 0.23836 ± 0.00037 |

Original minus coherent Brier is 0.011345 with a source-stratified parent interval [−0.001173, 0.024370], crossing zero. The candidate's K4 equal-coverage risk is slightly worse. Its K32 hidden binary-vote Brier is 0.153618 versus original's matched 0.151921; continuous Brier is 0.150856 versus 0.149308. These map tradeoffs and the missing third seed prevent a blanket improvement claim. Validation has been reused for development, so this supplement does not replace the selected original method or act as an untouched confirmation.

## 5. RQ1: what the fixed-marginal evidence actually establishes

[Historical analysis](results/paper_clean_analysis_v1/report.json) and the three [20260831](results/paper_clean_marginal_shuffle_v1/seed20260831/run.json), [20260901](results/paper_clean_marginal_shuffle_v1/seed20260901/run.json), [20260902](results/paper_clean_marginal_shuffle_v1/seed20260902/run.json) shuffle receipts report exact canonical K128 replay, preserved empirical free-cell counts and support closure. Each receipt records 9,306,112 checked cell locations. Their source-run hashes were checked here; the old ensembles themselves were not reopened.

The intervention independently permutes the world index at each cell of the same saved ensemble. Each cell therefore keeps its empirical marginal exactly while the cross-cell joint arrangement changes. Event Brier rises from 0.067494 to 0.161391 on that historical query cohort. This is evidence that **equal empirical cell marginals need not imply equal connectivity-event predictions**. Its arithmetic/mechanism interpretation does not become invalid merely because the cohort later proved unsuitable as unseen-place evaluation.

The scope is narrower than several possible claims: it does not establish generalization to new buildings, a true environmental posterior, causality of reachability training loss, universal dominance of any correlated model, or a physical safety guarantee. Finite-world shuffling is an intervention on empirical dependence, not an independently trained Bernoulli baseline. The historical mean-map control is also important: thresholding the same model's mean map gives Brier 0.069572, close to full 0.067494, with no resolved paired Brier advantage. Keep this contrary evidence visible; do not replace it with an easier unrelated deterministic baseline.

## 6. Historical training ablations and direct prediction

The [historical training-ablation report](site/data/flatlands_clean_training_ablations.json) retains three seeds for no-event/no-global, checkpoint paths/hashes, validation-selection and strict-restore audits. They belong only with the historical cohort, not the new parent-isolated table.

| Historical method | Repeats | Event Brier | Event NLL | Event ECE | False-safe at 0.8 | Coverage at 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| ConPath K=128 | 3 | 0.06749 ± 0.00936 | 0.28854 ± 0.01662 | 0.05715 ± 0.00683 | 0.04552 ± 0.01429 | 0.33644 ± 0.00579 |
| Independent K=128 | 3 | 0.09521 ± 0.00703 | 0.78503 ± 0.04690 | 0.08720 ± 0.00741 | 0.05689 ± 0.00832 | 0.36592 ± 0.01692 |
| Deterministic completion | 3 | 0.08857 ± 0.00323 | 1.22362 ± 0.04463 | 0.08857 ± 0.00323 | 0.07293 ± 0.00682 | 0.44278 ± 0.01024 |
| Direct query (one seed) | 1 | 0.09119 | 0.29788 | 0.04076 | 0.08335 | 0.37412 |
| Without event training loss | 3 | 0.20425 ± 0.00322 | 2.53067 ± 0.05699 | 0.21690 ± 0.01933 | 0.02561 ± 0.01609 | 0.19316 ± 0.10435 |
| Without global decoder factors | 3 | 0.09499 ± 0.00157 | 0.78486 ± 0.05495 | 0.08592 ± 0.00058 | 0.05739 ± 0.01662 | 0.36354 ± 0.00957 |
| ConPath mean-map K=128 | 3 | 0.06957 ± 0.00380 | 0.96118 ± 0.05253 | 0.06957 ± 0.00380 | 0.10509 ± 0.01468 | 0.50314 ± 0.01479 |

The no-event variant sets the reachability-loss weight to zero, while retaining historical event-based checkpoint selection. The no-global variant removes decoder global factors but preserves encoder context and local correlated noise; it is not removal of all spatial dependence. Both have worse Brier in that setting. However, no-event has lower false-safe at 0.8 than full while accepting fewer queries (about 19.3% versus 33.6%); this reverse indicator must not be hidden. Historical equal-30%-coverage risk comparisons do not establish uniformly improved safety. Direct-query is one seed; a different three-seed coordinate-query control cannot be substituted for it.

The [parent-overlap audit](site/data/flatlands_parent_group_audit.json) found 27/160 validation observations sharing a training parent. The [read-scope erratum](site/data/flatlands_read_scope_erratum.json) records prior physical-test access, including eight selected historical training and five selected validation observations. Historical `test_evaluated=false` flags therefore cannot prove an untouched test split. Old validation also influenced checkpoint selection. These are development diagnostics, and cannot be repaired into fresh final-test evidence simply by dropping a few rows after seeing scores.

There are **no matched current-parent no-event, no-global or direct-query evaluations**. Those cells remain missing. Current same-checkpoint mean-map controls do exist as the saved post hoc diagnostics described in Section 3; they must not be presented as prespecified primary outcomes. The old training ablations support a limited historical mechanism narrative, not a new-cohort causal ablation claim.

## 7. Synthetic P0 and cancelled external baselines

Synthetic P0 uses 24×24 maps, radii0/1/2 cells, 12 training and four held-out templates, 24 worlds/template and 1,152 events. It supplies a fourth visible context plane, unlike the three-channel FlatLands input. Full has two seeds (20260827/28), no-event only one (20260827); the selected template split is held fixed across optimization seeds.

The saved full-model Event Brier mean is 0.113999 across two seeds, versus no-event 0.191368 for one seed. This is a synthetic mechanism check, not a three-repeat public-data experiment. Full-map Brier and unknown-only Brier must also remain distinct: known cells dilute all-cell scores. Exact protocol, per-seed metrics and both map scopes are retained in `synthetic_runs` in the snapshot, with checkpoint byte hashes. These reports themselves mark `paper_result=false`.

The cancelled external protocol differs from both current and historical internal cohorts. LaMa is **LaMa BEV adaptation** and FM is **FM+XAttn literature reimplementation**; neither is an official author checkpoint. There is no fully trained three-seed external matrix, no completed full plan and no final-test result. LaMa's completed first-seed 5,000-step pilot is an engineering diagnostic. FM's incomplete pilot validation is not promoted to a new 117-case comparison set. Cancellation and partial completion do not demonstrate that either method loses to ConPath. The existing external progress, checkpoints and STOP marker are preserved; this ledger authorizes no continuation.

## 8. Paper claims that survive this audit

The defensible claim is that the existing finite-ensemble intervention shows connectivity can change with joint spatial dependence even at fixed empirical marginals, and a separate, modest parent-isolated development comparison gives lower Event Brier for original ConPath than independent completion. Event-level evaluation reveals tradeoffs that map accuracy alone does not capture.

The evidence does not establish superiority over sufficiently trained external literature methods, a robust advantage over deterministic completion, uniformly improved risk/reliability, generalization to an untouched public-data test, or a current-cohort causal attribution to each loss/component. The stronger mean-map control, continuous-map results, radius prior and risk/coverage reversals belong in the paper, not only an internal note.

The paper can therefore present D1 as its current principal **development-validation** table, D2 as an exploratory same-two-seed supplement, M1 as a qualified fixed-marginal mechanism result, and H1/S1 as separately labelled diagnostics. None receives formal external or final-test main-table eligibility from this audit. Future final testing would require a separately audited untouched cohort and frozen evaluation decision, but this round stops at existing evidence and writing.

中文结论：原模型对独立补全的当前小规模验证优势可以写；“优于全部补全模型”“安全性全面提高”“最新消融已证明各模块必要”“已战胜外部论文”都不能写。当前应完成论文的限定结论与图表，不继续训练追分，也不打开测试集。

## 9. Snapshot integrity

- `results/paper_validation_snapshot.json`: SHA-256 `8a3b81ef1f01dec72e57f92d7c02d39ed0e3697e12389143a755a836cc4bbc32`.
- `results/paper_validation_snapshot.csv`: SHA-256 `a6ea41fe2a6fce10af290dc401a526b1b25fc6bd6ac0a0a8c41d2a19d9981475`.

Every displayed experimental value is derived from a row or referenced report registered in the JSON snapshot. Secondary event metrics and continuous-map diagnostics are explicitly distinguished from original registered outcomes. No missing standard deviation, calibration score, external seed or final-test result has been fabricated.
