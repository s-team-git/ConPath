# Recent-baseline bridge for ConPath

Status: **protocol v1; no recent-paper result is claimed yet**  
Frozen: 2026-08-31 (America/New_York)

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
| [PaSCo, CVPR 2024](https://github.com/astra-vision/PaSCo) | Panoptic 3-D scene completion with uncertainty-aware multi-inference (MIMO) | Official recipe uses 1 or 3 subnets; 3-subnet training uses batch 2 on 2 A100-80G GPUs and `lr=1e-4`. | **Adaptation planned:** a 3-subnet ensemble control with the same 3-channel input and event evaluator. It will be labelled an ensemble control, not “PaSCo SOTA”. |
| [SGN, IEEE TIP 2024](https://github.com/Jieqianyu/SGN) | Sparse-guided, multi-scale semantic scene completion; a published lightweight model | SGN-L reports 12.5M parameters and 7.16G training memory on SemanticKITTI. | **Reference-only:** camera/3-D voxel labels do not match FlatLands. The 12.5M figure is used as an external scale reference; no SGN score is copied into our table. |
| [S4C, 3DV 2024](https://ahayler.github.io/publications/s4c/) | Self-supervised implicit semantic fields; arbitrary point queries and multi-view consistency | Implicit field rather than a dense voxel decoder; the project reports self-supervised training from video and pseudo-labels. | **Adaptation planned if compute permits:** coordinate-query hidden-cell completion, with observed cells clamped. Report event calibration, not SSC mIoU. |
| [SceneSense / frontier diffusion, 2024](https://arxiv.org/abs/2409.10681) | Diffusion occupancy completion and probabilistic map reconciliation for frontier navigation | The online variant reports 73% end-to-end runtime reduction and 28% fewer trainable parameters after removing conditioning; 3–5 predictions are merged per pose. | **Reference + port candidate:** a 2-D conditional diffusion control with fixed sample count. The preprint's robot/3-D metrics are not used as FlatLands evidence. |
| [ReliOcc, 2024](https://arxiv.org/abs/2409.18026) | Reliability-focused uncertainty learning and calibration for semantic occupancy | Plug-in uncertainty and calibration strategies are evaluated under sensor failures and out-of-domain noise. | **Metric/control reference:** use its reliability perspective to motivate coverage and false-safe curves; do not claim a direct architecture reproduction until its input contract is aligned. |
| [COTR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_COTR_Compact_Occupancy_TRansformer_for_Vision-based_3D_Occupancy_Prediction_CVPR_2024_paper.html) | Compact transformer for vision-based 3-D occupancy | Public code/paper provide a compact-vs-backbone comparison on Occ3D; the task remains camera/3-D voxel occupancy. | **Reference-only:** use as a recent compact-architecture citation; no cross-task score transfer. |
| [FlatLands, 2026](https://arxiv.org/abs/2603.16016) | The closest task: partial-view BEV completion with multiple valid layouts and stochastic/flow completion | Dataset-native stochastic completion benchmark and official provenance must be checked before importing weights. | **Mandatory same-task check:** audit official code/weights; if released and compatible, evaluate its samples through the exact ConPath event evaluator. |

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

The final report records parameter count, input channels, raster size, hidden-cell mask, optimizer,
learning-rate schedule, batch size, epoch/patience budget, posterior sample count, peak GPU memory,
and wall time for every row. Recent controls receive the same train/validation scenes, query rows,
radius set, three seeds (`20260831`, `20260901`, `20260902`), checkpoint-selection rule, and test
lock. Any method requiring a different sensor or label space is kept in the reference-only column.

## Required recent-control matrix before the paper freeze

1. Finish the current CUDA diagnostic matrix and retain it as a low-capacity pilot.
2. Re-run the three-seed ConPath/ablation matrix with the capacity lock above.
3. Add the 3-subnet ensemble control (PaSCo-inspired uncertainty control) and compare both event and
   map calibration at equal sample budget.
4. Attempt the S4C-style coordinate-query and diffusion-style 2-D completion ports only if they pass
   exact input/target leakage checks; otherwise document the incompatibility and keep them as related
   work.
5. Check the official FlatLands implementation/weights first. If unavailable or incompatible, state
   this explicitly in the limitations instead of silently substituting a different dataset.
6. Publish one table with same-dataset rows only. Published cross-dataset numbers appear in related
   work, never in the “ours vs baseline” cells.

## Reproducibility rule

Every recent-control run gets a config hash, git commit, environment JSON, seed, checkpoint, label-free
prediction CSV, exact evaluator report, and a metadata-labelled map. The website may show a recent
method as **planned**, **not comparable**, or **evaluated on the same FlatLands contract**; it must not
render an empty/planned row as a zero or as evidence that ConPath is better.

