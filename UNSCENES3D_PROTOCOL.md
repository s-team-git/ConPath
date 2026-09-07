# UnScenes3D ConPath adapter protocol v0.2

Status: **coordinate-audited contract; clean validation diagnostic complete; no final model score**
Frozen: 2026-09-04 (America/New_York)

Evaluation correction, 2026-09-06: mean-map/qualitative v1 restored observed-free cells after
sampling and could reopen invalid support. Version 2 gives blocked observations priority and
applies `target_valid` last. All six original clean checkpoints and their hidden-map metrics
replay exactly; all 27,522 corrected validation predictions satisfy the observation bounds.
Use `*support_clamped_mean_map_k128_v2` and `*support_clamped_qualitative_v2` for current
evidence. Training checkpoints, adapter parameters, frozen queries and physical test locks
are unchanged; v1 predictions and figures are retained as superseded history.

Diagnostic added 2026-09-06 (the frozen adapter/query contract is unchanged): exact optimistic
and pessimistic worlds imply a scene-weighted Brier lower bound of 0.13097 on train and 0.47574
on validation. Only 15.66% of validation events can change through unknown completion. See
`scripts/audit_unscenes3d_observation_ceiling.py`, `PAPER_EVIDENCE.md`, and
`site/data/unscenes3d_observation_ceiling.json`. This identifies an observation-model limitation;
future changes must be trained/tuned on training data under a separate versioned protocol.

The previous v0.1 contract (all LiDAR returns treated as blocked endpoints) is
retained as a legacy diagnostic. The canonical train/validation control contract
below uses the label-free ground-endpoint policy after the coordinate audit.

This protocol defines the second-domain bridge before any learned run. It is intentionally separate
from the FlatLands protocol: the voxel resolution matches 0.3 m, but the query distances and
scene/site split are native to the UnScenes3D release.

## Assets and split

The 13-scene mini raw package, 629-frame label package, and two local-map archives are recorded with
byte counts and hashes in `UNSCENES3D_COMPATIBILITY_CHECK.md` and `RECOVERY_STATE.json`. All raw
image/cloud/calibration stems join exactly. Occupancy/elevation/depth labels and local maps join all
629 released `scene_info.json` samples; 3-D object labels are optional and are not used by this
adapter.

The site-held-out split is fixed before training:

| Split | Locations | Scenes | Use |
|---|---|---:|---|
| train | `location_1`, `location_2`, `location_3` | 9 | fitting and checkpoint updates |
| validation | `location_4_5` | 2 | checkpoint selection only |
| test (locked) | `location_6` | 2 | no labels read until all methods are frozen |

No adjacent frame is assigned across splits. Scene and location identifiers are taken from the
official `scene_info.json`; no random frame split is allowed.

## Three-channel bridge

The official occupancy grid is `[x, y, z] = 256 x 256 x 32` with 0.3 m voxels and point range
`x=[0,76.8)`, `y=[-38.4,38.4)`, `z=[-4,5.6)`. `src/pathrel/unscenes3d.py` projects each sparse
occupancy file to a 2-D support slice:

* class 11 (`driveable_surface`) is traversable;
* any other labeled class blocks the `(x,y)` cell, even if a driveable voxel is also present; and
* an `(x,y)` cell with no occupancy row is outside the target-valid mask, never an implicit free
  cell.

The input is label-free LiDAR ray rasterization with the same `[observed-free, observed-blocked,
unknown]` channel order as ConPath. The legacy v0.1 policy marks every return as a blocked endpoint.
The canonical control policy (`endpoint_policy=ground`) estimates a longitudinal support-height
profile from raw points only (1.2 m bins, lateral limit 20 m, 0.15 quantile) and marks the lowest
return in a cell free when it lies within 0.35 m of that profile; otherwise the endpoint is blocked.
Cells traversed by a ray before the endpoint are observed free. Occupancy/elevation classes are
never passed to the input.

### Valid-support amendment (2026-09-03)

The exact event oracle treats the complement of `target_valid` as outside the evaluation support,
so every map-derived posterior world must deterministically mark that complement blocked before
sampling, footprint clearance, or connectivity evaluation. The mask discloses only where the
release defines a target; it never discloses the free/blocked class. Training forwards, validation
forwards, deterministic mean-map controls, and qualitative rendering must all use the identical
mask. Older ConPath, independent-decoder, mean-map, calibration, and qualitative artifacts were
generated without this clamp and are therefore superseded for paper and cross-domain claims.
Coordinate-query and train-fitted radius-prior controls do not sample maps and are unaffected.
The six-run clean support-consistent map-model retraining and K=128 replay now complete gate 4 for
the validation diagnostic. The paired Brier delta is near zero on the two held-out scenes, so this
does not establish a cross-domain method advantage; the one-time test gate remains separate.

## Queries and events

Candidate endpoints use a fixed polar stencil of distances 13/27/40 cells (3.9/8.1/12.0 m) and
angles 0--330 degrees in 30-degree increments. The canonical start is the nearest target-valid
cell to the fixed geometry hint `(row=96, col=128)`; only bounds and validity are used to select it,
not the free/blocked class or sensor endpoint status. Invalid endpoints are omitted from the
retained manifest. Event labels use the exact four-neighbor maximum-bottleneck/disk-clearance oracle
with footprint radii 0/1/2 cells (0/0.3/0.6 m). The same query rows and radii are used by every
control. The canonical manifest is
`results/unscenes3d_contract_manifest_ground_valid/manifest.json` (15,567 train and 1,529
validation queries); the older v0.1 manifest remains at `results/unscenes3d_contract_manifest/`.

## Gates before test

1. Run `scripts/audit_unscenes3d_mini.py` with both local-map directories and preserve its report.
2. Render at least one camera-to-LiDAR/BEV overlay and verify the local-map/LiDAR coordinate
   agreement. The coordinate audit sampled 55 train/validation frames: direct LiDAR/local-map
   nearest-neighbour overlap averaged 0.791 within 0.3 m and 0.863 within 1 m; both IMU-transform
   alternatives had zero overlap at those thresholds. Calibration rotation diagnostics remained
   near-orthonormal (maximum error `2.13e-6`, minimum determinant `0.9999986`).
3. Freeze label-validity and event-balance counts on train/validation without reading test labels.
4. Train capacity-matched ConPath, independent completion, deterministic completion, and the
   recent coordinate-query control with three seeds from the canonical ground-valid manifest;
   report scene-weighted Brier/NLL/ECE, false-safe risk, radius monotonicity, and K-sensitivity.
   **This gate was reopened by the 2026-09-03 valid-support amendment and is now complete for the
   validation diagnostic.** The historical,
   pre-correction validation matrix reported correlated ConPath at
   `0.56272 ± 0.00744` Brier, independent is `0.56297 ± 0.00541`, the K=128 mean-map control is
   `0.62774 ± 0.00145`, and the S4C-inspired coordinate query is `0.20379 ± 0.03485`; these
   predictive objects are not interchangeable. The map-derived values are superseded, while the
   coordinate-query row remains a valid non-map control. The clean K=128 map-derived diagnostic is
   `0.51142 ± 0.00198` (correlated) versus `0.51130 ± 0.00218` (independent), with no measurable
   correlation gain on two scenes; none is a final cross-domain claim.
   `scripts/evaluate_unscenes3d_calibration.py` additionally writes validation-only reliability
   and selective-risk snapshots for the controls that export per-event rows; scalar-only stochastic
   reports are not interpolated into those curves.
5. Only after these controls and checkpoint rules are frozen may the test site be unlocked once.

The clean validation diagnostic may be reported with its two-scene and `location_6` caveats; no
UnScenes3D test or final cross-domain result is reported until the separate test go/no-go passes.
