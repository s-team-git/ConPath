# UnScenes3D compatibility check

Status: **coordinate/manifest-audited candidate; clean target-valid-support validation diagnostic complete; test locked**
Checked: 2026-09-04 (America/New_York)

UnScenes3D is currently the strongest candidate for the second-domain experiment and for a later
ground-robot visual. The site hero is now a checkpoint-derived FlatLands comparison, and the clean
UnScenes3D panels are retained as a transfer diagnostic. Unlike ORFD's image-plane-only target, its release is organized around
3D occupancy and road-surface geometry with an explicit vehicle/world coordinate chain.

## Official release evidence

The official [UnScenes3D repository](https://github.com/ruiqi-song/UnScenes3D) describes a mini
release with synchronized images and point clouds, calibration, 3D semantic occupancy labels,
road-elevation labels, local dense point-cloud maps, vehicle information, and `scene_info.json`
split metadata. The accompanying [Scientific Data article](https://www.nature.com/articles/s41597-025-05532-5)
reports approximately 23,549 frames captured with a monocular camera, LiDAR, IMU, and RTK, and
states that occupancy labels distinguish drivable and non-drivable regions for trajectory planning.
The article reports a six-region generalization design in which regions 1--4 are used for training
and regions 5--6 are unseen test regions. The repository is marked Apache-2.0; the article's usage
notes state that the dataset is downloadable under CC BY 4.0, so release/archive terms must be
recorded separately before redistributing any derived media.

The repository's pipeline documentation gives the raw package layout:

```text
raw_data/
  scene_*/
    calib/
    camera_1/
    ego_pose/
    label_3d/
    lidar_1/
  scene_info.json
```

The public release also lists separate label and local-map archives. The larger mini raw package
acquired locally is a 13-scene release with 1,336 synchronized image/cloud/calibration stems; the
smaller `raw_data.zip` asset remains available as a one-scene parser/pose smoke package. The label
package contains 629 aligned occupancy/elevation/depth/caption timestamps, 1,336 vehicle
information files, and 269 sparse 3-D label files. These are acquisition and join diagnostics only,
not model scores.

## Code-level coordinate audit

The official processing code was cloned at commit `baab668ce487af674637f10a606a9e919acfd19f` and
read without modifying it. Its database loader:

* reads frame timestamps from `scene_info.json` (`samples` and optional `sweeps`);
* parses camera projection, rectification, LiDAR-to-camera, and LiDAR-to-IMU calibration;
* loads per-frame odometry poses from `pose_odom/<timestamp>.txt` (and keeps an ego-pose path for
  the raw data); and
* projects dense local-map points into camera/LiDAR frames while constructing occupancy and road
  elevation labels.

This is sufficient in principle to build a metric support raster in a vehicle/world frame and to
make a ground-vehicle visual with camera, observed support, posterior occupancy, footprint erosion,
and path probability. The downloaded raw/label join is complete for all 629 label timestamps. The
raw tree itself has no `pose_odom` or local-map directory, but the separate local-map parts now
join all 629 labels. The read-only coordinate audit sampled 55 train/validation frames and found
direct LiDAR/local-map nearest-neighbour overlap of 0.791 within 0.3 m and 0.863 within 1 m;
both forward and inverse `Tr_velo_to_imu` alternatives had zero overlap at those thresholds. Camera
projection had positive depth for all 800,742 sampled returns, and calibration rotations stayed
near orthonormal (maximum error `2.13e-6`, minimum determinant `0.9999986`). This supports using
the raw LiDAR frame directly for the local-map comparison; it does not establish a world-pose
claim.

## Contract comparison and adapter gates

| Dimension | ConPath/FlatLands | UnScenes3D | Decision |
|---|---|---|---|
| Input | Partial three-channel support raster | RGB, LiDAR, calibration, synchronized frames; reduced to the same three channels | Coordinate-audited sensor adapter; ground-endpoint candidate |
| Truth | Hidden metric support grid and two-terminal footprint event | 3D semantic occupancy plus road elevation/local map | Map official `driveable_surface` (class 11) to traversable and all obstacle/terrain classes conservatively to blocked; keep elevation/occupancy validity masks |
| World reference | Full support map with provenance | Local dense map and pose/odometry files | Direct LiDAR/local-map frame agrees in audit; world-pose use remains out of scope |
| Splits | Scene/site/sequence held out, no adjacent-frame leakage | `scene_info.json` plus six-region generalization description; mini raw smoke currently one scene | Freeze an explicit scene/site split before reading test labels |
| Scale | 256x256 FlatLands raster | Release is multi-archive; raw mini is ~104.5 MB, labels/maps are larger | Download only after archive hashes and disk budget are recorded |
| Visual fit | Desk pilot currently semantically weak | Ground vehicle, off-road support surface, occupancy/elevation | Preferred source for the replacement visual |

Before training, all of the following must pass:

1. hash and license audit for each official archive (raw, labels, local maps);
2. complete timestamp/scene joins across RGB, LiDAR, calibration, pose, occupancy, elevation, and
   local-map files;
3. coordinate-frame and voxel-bound sanity checks, including a rendered camera-to-BEV overlay;
4. a scene/site/sequence-held-out split with an adjacent-frame leakage report;
5. a conservative mapping from occupancy/elevation to `TRAVERSABLE`/`BLOCKED` plus a separate
   observation/label-validity mask; and
6. a target-blind query manifest and exact event evaluator before any test labels are read.

The coordinate-audit gate is now recorded as passed for train/validation. The canonical candidate
uses `endpoint_policy=ground` (1.2 m longitudinal bins, 20 m lateral limit, 0.15 height quantile,
0.35 m margin) and retains the fixed validity-only start. Its manifest is
`results/unscenes3d_contract_manifest_ground_valid/manifest.json`, with 15,567 train and 1,529
validation queries. A later support audit found that the old PathRelNet map worlds did not hard-block
the complement of `target_valid`, although the exact oracle treats it as invalid support. Therefore
the historical three-seed F=16 map-only event Brier `0.56272 ± 0.00744` is superseded, not a method
win or paper result.
The earlier bounded event-loss number `0.59126 ± 0.00855` belongs to the rejected
observed-free-start ablation and is not a canonical-contract result. On the canonical contract,
the historical capacity-matched independent decoder is `0.56297 ± 0.00541`; the deterministic K=128
posterior-mean-map control is `0.62774 ± 0.00145` (hidden-cell map Brier `0.15025` on average).
Both are likewise superseded. The trainer, mean-map evaluator, and qualitative renderer now pass
`target_valid` as a hard support mask and record whether the source checkpoint used the same policy;
Six fresh F=16 map adapters were then trained under the corrected contract. Their exact K=128
mean-map replay gives event Brier `0.51142 ± 0.00198` for correlated ConPath and
`0.51130 ± 0.00218` for the independent control (paired delta `−0.00012 ± 0.00021`). With only
two held-out validation scenes, this is a reproducible support/transfer diagnostic with no measurable
correlation advantage, not a cross-domain method claim. The unaffected S4C-inspired coordinate-query control is
`0.20379 ± 0.03485` with NLL
`0.57289 ± 0.10572`. These rows are deliberately reported as controls: they use different
predictive objects and do not establish a cross-domain method win. The archived validation-only
ground-robot posterior/footprint/failure panel is stored under
`results/unscenes3d_qualitative_validation/` and mirrored in the project site; `location_6`
remains locked. Before the support issue was discovered, the read-only `scripts/audit_unscenes3d_controls.py` replayed the frozen
manifest, all 15 control reports/CSV contracts, K=32/64/128 sensitivity, calibration snapshot,
site links, published audit evidence, manifest replay, portable paths, and PNG dimensions in 599 checks with zero failures. A fresh v0.2 GPU replay of the three correlated
checkpoints differed from the earlier metadata-only reports by at most `1.51e-4` on any displayed
metric; the three older reports retain a v0.1 protocol tag and are treated as legacy metadata,
not silently rewritten. That audit remains a structural history, not current model evidence. The compact archived snapshot is
`site/data/unscenes3d_controls_validation.json`.

The clean candidate JSON and panels are under
`site/data/unscenes3d_clean_support_k128_candidate.json`,
`site/data/unscenes3d_clean_candidate_qualitative.json`, and
`site/assets/unscenes3d_clean_candidate_{positive,failure}.png`; all record
`paper_result=false` and `test_evaluated=false`.

No UnScenes3D score is reported as a final or cross-domain result. The acquired raw and label
packages and the runs above remain validation diagnostics, not evidence of ConPath superiority.

## Local acquisition record

The official `raw_data.zip` smoke asset was downloaded on 2026-09-02, verified as 104,519,506 bytes,
and hashed as `d1050b22d0eb31ea7199d2625accbfb54ae3034b3b57653a9790cdbdfa522ae0`. The larger
`unscenes3d-mini_raw.zip` package was verified as 983,750,002 bytes with SHA-256
`4e14f1b02e350f8c332032c50b2e3cb4c538f7a2cea0676733840539c02b8494`; its 4,014 members passed a
path-traversal audit and were extracted under ignored `data/raw/unscenes3d/raw_package/`. The
`unscenes3d-mini_label.zip` package was verified as 577,517,527 bytes with SHA-256
`267b6744e6417826d8cceb3304495a3139ae6104984f74b8f07cb1cf5d163a1a`; its 4,128 members also passed
the path audit and were extracted under ignored `data/raw/unscenes3d/label_package/`. No FlatLands
archive was extracted or modified. The two local-map parts were then verified and path-audited:
`unscenes3d-mini_loaclmap-1.zip` is 1,521,872,067 bytes with SHA-256
`67ddd40392dbeb29dc9f40e209f320ace6ca1f83d0177a009b3f9457738c44ae`, and
`unscenes3d-mini_loaclmap-2.zip` is 1,390,845,431 bytes with SHA-256
`f947a738db47c95282b8b6d2bcd39bb35fa113324dd5c27ddfe1f2bdf83364dd`. Together they provide
629/629 local-map files for the 629 occupancy timestamps. The read-only
`scripts/audit_unscenes3d_mini.py` report passes basic integrity: zero missing raw stems, zero
duplicate scene timestamps, zero occupancy shape/bound/class violations, zero bad map floats, and
629/629 map joins. A separate 50-frame scene-spread nearest-neighbour smoke check found a
median 0.804 fraction of LiDAR samples within 0.3 m of the corresponding map (0.875 within 1 m);
this supports the same-frame coordinate hypothesis but is not a final model result. No FlatLands
archive was extracted or modified.

## Sources

* [Official UnScenes3D repository](https://github.com/ruiqi-song/UnScenes3D)
* [UnScenes3D Scientific Data article](https://www.nature.com/articles/s41597-025-05532-5)
* [Official mini-release page](https://github.com/ruiqi-song/UnScenes3D/releases)
