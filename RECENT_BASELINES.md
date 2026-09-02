# Recent-baseline bridge for ConPath

Status: **protocol v1; first same-contract recent-method control evaluated (validation-only)**
Frozen: 2026-09-01 (America/New_York)

This file records how the FlatLands comparison will include strong methods from the last few
years without mixing incompatible tasks. A number copied from another paper is **not** a FlatLands
result: the input sensor, label space, scene split, resolution, and evaluation metric must match
before a row can enter the paper table.

## Why a direct leaderboard copy would be invalid

Recent occupancy papers mostly target camera/LiDAR 3-D semantic scene completion or future BEV
forecasting. ConPath is a 2-D, three-channel, partial-support problem whose primary target is a
calibrated two-terminal footprint event. A 3-D mIoU or FID number therefore cannot be compared with
our event Brier/NLL/ECE. We use the papers below in two explicit tiers:

1. **Same-contract adaptations.** Re-run the method (or its released uncertainty mechanism) on the
   frozen FlatLands input, hidden-cell mask, query manifest, and scene-weighted evaluator.
2. **Reference-only controls.** Record the published parameter/training setting and implement a
   capacity- or uncertainty-matched control when a faithful same-contract port is not possible.

## Recent methods and the planned use

| Method (year) | What is recent and relevant | Published setting / parameter fact | FlatLands status |
|---|---|---|---|
| [PaSCo, CVPR 2024](https://github.com/astra-vision/PaSCo) | Panoptic 3-D scene completion with uncertainty-aware multi-inference (MIMO) | Official recipe uses 1 or 3 subnets; 3-subnet training uses batch 2 on 2 A100-80G GPUs and `lr=1e-4`. | **Same-contract adapter evaluated:** three independently trained 16-channel completion members, fixed total event budget K=32. Labelled an ensemble control, not “PaSCo SOTA”. |
| [SGN, IEEE TIP 2024](https://github.com/Jieqianyu/SGN) | Sparse-guided, multi-scale semantic scene completion; a published lightweight model | SGN-L reports 12.5M parameters and 7.16G training memory on SemanticKITTI. | **Reference-only:** camera/3-D voxel labels do not match FlatLands. The 12.5M figure is used as an external scale reference; no SGN score is copied into our table. |
| [S4C, 3DV 2024](https://ahayler.github.io/publications/s4c/) | Self-supervised implicit semantic fields; arbitrary point queries and multi-view consistency | Implicit field rather than a dense voxel decoder; the project reports self-supervised training from video and pseudo-labels. | **Same-contract adapter evaluated:** an S4C-inspired coordinate-query event field with bilinear feature sampling and Fourier geometry encoding. This is explicitly not a reproduction of the original S4C 3-D system; report event calibration, not SSC mIoU. |
| [SceneSense / frontier diffusion, 2024](https://arxiv.org/abs/2409.10681) | Diffusion occupancy completion and probabilistic map reconciliation for frontier navigation | The online variant reports 73% end-to-end runtime reduction and 28% fewer trainable parameters after removing conditioning; 3–5 predictions are merged per pose. | **Reference-only after contract audit:** official code is a 3-D point-cloud/voxel ROS system with robot-map inputs and FID/KID/exploration metrics, not the FlatLands 2-D event contract. No score is copied. |
| [ReliOcc, 2024](https://arxiv.org/abs/2409.18026) | Reliability-focused uncertainty learning and calibration for semantic occupancy | Plug-in uncertainty and calibration strategies are evaluated under sensor failures and out-of-domain noise. | **Metric/control reference:** use its reliability perspective to motivate coverage and false-safe curves; do not claim a direct architecture reproduction until its input contract is aligned. |
| [COTR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_COTR_Compact_Occupancy_TRansformer_for_Vision-based_3D_Occupancy_Prediction_CVPR_2024_paper.html) | Compact transformer for vision-based 3-D occupancy | Public code/paper provide a compact-vs-backbone comparison on Occ3D; the task remains camera/3-D voxel occupancy. | **Reference-only:** use as a recent compact-architecture citation; no cross-task score transfer. |
| [FlatLands, 2026](https://arxiv.org/abs/2603.16016) | The closest task: partial-view BEV completion with multiple valid layouts and stochastic/flow completion | Dataset-native stochastic completion benchmark and official provenance must be checked before importing weights. | **Official check completed:** dataset and documentation are public, but the official repository currently says model weights/construction code/tooling are planned for release. No importable checkpoint is available; reference-only pending a future artifact audit. |

## Parameter and optimization lock

The first running pilot used `feature_channels=8` (30,428 trainable parameters) to diagnose CUDA
and propagation. That pilot is explicitly **not** a capacity-matched paper comparison. The final
FlatLands matrix will use `feature_channels=16` for ConPath and the existing completion/direct-query
baselines. Measured trainable counts are:

| Control | Configuration | Parameters |
|---|---|---:|
| ConPath pilot | `feature_channels=8, latent_dim=4` | 30,428 |
| ConPath final | `feature_channels=16, latent_dim=4` | 120,108 |
| Independent completion | `feature_channels=16` | 119,921 |
| Direct query | `feature_channels=16` | 127,905 |
| S4C-inspired coordinate query | `feature_channels=16`, Fourier geometry, bilinear queries | 132,449 |

The final report records parameter count, input channels, raster size, hidden-cell mask, optimizer,
learning-rate schedule, batch size, epoch/patience budget, posterior sample count, peak GPU memory,
and wall time for every row. Recent controls receive the same train/validation scenes, query rows,
radius set, three seeds (`20260831`, `20260901`, `20260902`), checkpoint-selection rule, and test
lock. Any method requiring a different sensor or label space is kept in the reference-only column.

## Required recent-control matrix before the paper freeze

1. Finish the current CUDA diagnostic matrix and retain it as a low-capacity pilot.
2. Re-run the three-seed ConPath/ablation matrix with the capacity lock above.
3. **Done for the first validation pass:** add the 3-subnet ensemble control (PaSCo-inspired
   uncertainty control) and compare event and map calibration at equal sample budget. A follow-up
   K-sensitivity run now evaluates total posterior budgets 32/64/128 under the same exact event
   contract; the compact report is tracked at `site/data/flatlands_pasco_ensemble_validation.json`.
   It remains validation-only until the full recent-control matrix and final test gate are complete.
4. **Done for the current validation pass:** evaluate the S4C-inspired coordinate-query control under
   the exact FlatLands event contract. It uses three seeds and remains validation-only; the original
   S4C 3-D architecture is not claimed as reproduced.
5. **Done:** audit SceneSense as the diffusion reference. Its official 3-D pointmap/ROS contract does
   not map to the FlatLands event protocol without creating a new method; it is recorded as
   reference-only in [`SCENESENSE_COMPATIBILITY_CHECK.md`](SCENESENSE_COMPATIBILITY_CHECK.md).
6. **Done:** check the official FlatLands implementation/weights first. The public repository currently
   has no importable checkpoint and marks weights/construction code/tooling as planned; this is recorded
   in [`OFFICIAL_FLATLANDS_CHECK.md`](OFFICIAL_FLATLANDS_CHECK.md), so no incompatible number is copied.
7. Publish one table with same-dataset rows only. Published cross-dataset numbers appear in related
   work, never in the “ours vs baseline” cells.

## S4C-inspired coordinate-query result

The adapter keeps the canonical three-channel input, F=16 encoder, frozen train/validation scenes,
4,224 retained validation events, radii 0/10/20, AdamW protocol, and test lock. It changes only the
query representation: start/goal/delta, distance, angle, and footprint radius are Fourier encoded,
and start/goal features are sampled bilinearly from the raster feature field. Across seeds
`20260831`, `20260901`, and `20260902`, the exact scene-weighted validation metrics are:

| Control | Event Brier | Event NLL | Event ECE | False-safe @0.8 | Coverage @0.8 |
|---|---:|---:|---:|---:|---:|
| S4C-inspired coordinate query (F=16) | 0.09204 ± 0.00582 | 0.33411 ± 0.01859 | 0.04302 ± 0.00803 | 0.09555 ± 0.00357 | 0.39193 ± 0.03865 |

The Brier/NLL/ECE values are competitive with the ConPath validation control, but the selective
false-safe rate is higher (`0.09555` versus `0.06004` for ConPath). This result supports keeping
the coordinate-query adapter in the comparison matrix; it does not establish a paper-level claim,
SOTA status, or faithful S4C reproduction. Per-seed checkpoints, label-free predictions, exact
reports, and the four-method calibration snapshot are retained under the ignored results tree.

## Official FlatLands artifact status

The [official repository](https://github.com/1ssb/Flat_Lands/) and [project page](https://1ssb.github.io/Flat_Lands/)
were checked on 2026-09-02. The archive and documentation are available, while the repository states
that model weights, construction code, and additional benchmark tooling are planned for release. The
official model therefore remains **not yet importable / reference-only** for ConPath. See
[`OFFICIAL_FLATLANDS_CHECK.md`](OFFICIAL_FLATLANDS_CHECK.md) for the compatibility checklist and the
rule that future weights must pass before entering the same-dataset table.

## Reproducibility rule

Every recent-control run gets a config hash, git commit, environment JSON, seed, checkpoint, label-free
prediction CSV, exact evaluator report, and a metadata-labelled map. The website may show a recent
method as **planned**, **not comparable**, or **evaluated on the same FlatLands contract**; it must not
render an empty/planned row as a zero or as evidence that ConPath is better.

## First same-contract recent-control result

The PaSCo-inspired adapter uses the three selected completion checkpoints from seeds `20260831`,
`20260901`, and `20260902`. Each member consumes the canonical three-channel FlatLands input;
event probabilities are pooled across the three members with a fixed total of 32 posterior worlds
(11/11/10 per member). The exact scene-weighted validation result is:

| Control | Map Brier | Event Brier | Event NLL | Event ECE | False-safe @0.8 |
|---|---:|---:|---:|---:|---:|
| PaSCo-inspired 3-subnet ensemble (K=32) | 0.11054 | 0.22898 | 2.89817 | 0.24841 | 0.01538 |

The fixed three-member ensemble's event metrics are stable as posterior sampling increases:

| Total K | Event Brier | Event NLL | Event ECE | False-safe @0.8 | Coverage @0.8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.22898 | 2.89817 | 0.24841 | 0.01538 | 0.18981 |
| 64 | 0.22842 | 2.86991 | 0.24810 | 0.01465 | 0.18691 |
| 128 | 0.22806 | 2.84268 | 0.24830 | 0.01228 | 0.19122 |

This budget check addresses Monte-Carlo sensitivity only; it does not turn the PaSCo-inspired
adapter into a PaSCo architecture reproduction or a final paper result.

The corresponding member checkpoint SHA-256 values and prediction/evaluation hashes are in the
machine-readable report. This row is a fair same-contract uncertainty control, not a reproduction
of PaSCo's camera/3-D architecture; no incompatible 3-D score is copied into the ConPath table.
