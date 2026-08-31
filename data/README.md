# Data

Raw public datasets are not committed here (`data/raw/` is ignored). The repository now contains a
real-data pilot for the official TUM RGB-D `freiburg1/desk` sequence. It uses RGB, registered depth,
and the supplied motion-capture trajectory to build a geometric world-frame reference map; see
[`REAL_DATA_PILOT.md`](../REAL_DATA_PILOT.md) and `scripts/run_tum_rgbd_pilot.py` for the exact
command and claim boundary.

The official FlatLands archive is also present locally under ignored `data/raw/flatlands/` and has
been verified against its published byte count and SHA-256. Do not extract or train on its official
train/val/test split for a cross-scene claim: the full metadata audit found thousands of shared
`(source, scene_id)` values across all three in-distribution partitions. See
[`P1_DATA_AUDIT.md`](../P1_DATA_AUDIT.md).

The deterministic synthetic corridor generator in `src/pathrel/synthetic.py` remains the contract
harness for the P0 death test, not a substitute for the real-data pilot.

Planned paper-grade audits/adapters, in order:

1. FlatLands with a versioned scene-disjoint split and natural-query audit; ScanNet++ stays OOD.
2. ORFD for a secondary off-road label-semantics audit.
3. UnScenes3D for the main support-surface occupancy experiment.
4. WildOcc for cross-dataset evaluation.

Each adapter must map valid physical truth to `TRAVERSABLE` or `BLOCKED` and preserve a separate
observation/label-validity mask. `UNKNOWN` is an observation condition, not a third terrain
state. Unobserved cells must never be assigned free ground truth unless a future or multi-view
reference supplies verified hidden-world supervision.
