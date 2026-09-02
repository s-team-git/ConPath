# SceneSense compatibility check

Audit date: **2026-09-02 (America/New_York)**  
Decision: **reference-only for the current ConPath validation matrix**

## Sources checked

- [SceneSense IROS 2024 paper](https://arxiv.org/abs/2403.11985)
- [Online SceneSense paper](https://arxiv.org/abs/2409.10681)
- [Official SceneSense repository](https://github.com/arpg/SceneSense)
- [SceneSense project page](https://arpg.colorado.edu/scenesense/)

## Contract mismatch

SceneSense is a 3-D occupancy inpainting system for a running robot map. Its public code expects
point-cloud/voxel preprocessing, sparse 3-D convolution dependencies, and a ROS/runtime map rather
than the FlatLands packet used here. The method preserves observed occupied/free voxels while
diffusing unobserved 3-D geometry around a robot; it is not a single-pass 2-D floormap generator.

The reported evaluations use 3-D occupancy/map-quality and robot-exploration measures (for example
FID/KID, frontier prediction, and traversability time), not the exact FlatLands hidden-cell mask,
4,224 retained start-goal-radius events, or ConPath's scene-weighted event Brier/NLL/ECE/false-safe
metrics. The online paper and repository do not provide a checkpoint or preprocessing path that can
be loaded without changing those contracts.

## Decision

No SceneSense score is copied into the ConPath same-dataset table. Reimplementing a new 2-D diffusion
network from the paper's high-level idea would be an original adapter, not a SceneSense reproduction;
it is deferred until the core ground-robot visual and second-domain gates are complete. The current
recent-method matrix therefore labels SceneSense as **reference-only / incompatible 3-D pointmap
contract**, while the PaSCo- and S4C-inspired rows remain explicitly same-contract adapters.
