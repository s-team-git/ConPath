# UnScenes3D ConPath adapter protocol v0.1

Status: **frozen adapter draft; no model score yet**
Frozen: 2026-09-02 (America/New_York)

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
unknown]` channel order as ConPath. Returns are conservative blocked endpoints; cells traversed by
a ray before the endpoint are observed free. Occupancy classes are never passed to the input.

## Queries and events

Candidate endpoints use a fixed polar stencil of distances 13/27/40 cells (3.9/8.1/12.0 m) and
angles 0--330 degrees in 30-degree increments. The start is the nearest target-valid cell to the
fixed geometry hint `(row=96, col=128)`; only bounds and validity are used to select endpoints, not
the free/blocked class. Invalid endpoints are omitted from the retained manifest. Event labels use
the exact four-neighbor maximum-bottleneck/disk-clearance oracle with footprint radii 0/1/2 cells
(0/0.3/0.6 m). The same query rows and radii are used by every control.

## Gates before test

1. Run `scripts/audit_unscenes3d_mini.py` with both local-map directories and preserve its report.
2. Render at least one camera-to-LiDAR/BEV overlay and verify the local-map/LiDAR coordinate
   agreement; the current 50-frame smoke has 0.804 mean overlap within 0.3 m and 0.875 within 1 m.
3. Freeze label-validity and event-balance counts on train/validation without reading test labels.
4. Train capacity-matched ConPath, independent completion, deterministic completion, and the
   recent coordinate-query control with three seeds; report scene-weighted Brier/NLL/ECE,
   false-safe risk, radius monotonicity, and K-sensitivity.
5. Only after these controls and checkpoint rules are frozen may the test site be unlocked once.

No UnScenes3D result is reported until all five gates pass.
