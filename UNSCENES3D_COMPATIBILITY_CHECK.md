# UnScenes3D compatibility check

Status: **priority second-domain candidate; raw mini package acquired, label/map adapter pending**  
Checked: 2026-09-02 (America/New_York)

UnScenes3D is currently the strongest candidate for the second-domain experiment and for replacing
the desk-surface hero visual. Unlike ORFD's image-plane-only target, its release is organized around
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

The public release also lists separate label and local-map archives. The mini release page lists a
14-scene package; the raw package currently acquired locally is the smaller `raw_data.zip` asset
and contains one scene (`scene_00000`) for parser/pose smoke testing.

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
and path probability. It is still necessary to verify coordinate handedness, voxel bounds, temporal
leakage, and the exact meaning of occupancy/elevation values on the downloaded label/map archives.

## Contract comparison and adapter gates

| Dimension | ConPath/FlatLands | UnScenes3D | Decision |
|---|---|---|---|
| Input | Partial three-channel support raster | RGB, LiDAR, calibration, synchronized frames; can be reduced to the same three channels | Adapter required, but sensor geometry is available |
| Truth | Hidden metric support grid and two-terminal footprint event | 3D semantic occupancy plus road elevation/local map | Project a conservative support slice; keep elevation/occupancy validity masks |
| World reference | Full support map with provenance | Local dense map and pose/odometry files | Promising; verify map-to-pose transform and bounds on labels |
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

No UnScenes3D score is reported yet. The acquired raw package is a parser/pose smoke artifact, not
a result and not evidence of ConPath superiority.

## Local acquisition record

The official release asset
`https://github.com/ruiqi-song/UnScenes3D/releases/download/unscenes-mini/raw_data.zip` was
downloaded on 2026-09-02, verified as 104,519,506 bytes, and hashed as
`d1050b22d0eb31ea7199d2625accbfb54ae3034b3b57653a9790cdbdfa522ae0`. It was safely inspected for
path traversal and extracted under ignored `data/raw/unscenes3d/raw_data/`; no FlatLands archive
was extracted or modified.

## Sources

* [Official UnScenes3D repository](https://github.com/ruiqi-song/UnScenes3D)
* [UnScenes3D Scientific Data article](https://www.nature.com/articles/s41597-025-05532-5)
* [Official mini-release page](https://github.com/ruiqi-song/UnScenes3D/releases)

