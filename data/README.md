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

The packet-level `provenance.original_split` values do recover a complete scene-disjoint candidate:
203,373 train, 25,555 validation, and 41,647 test observations, with ScanNet++ only in test. Its
deterministic manifest is ignored under `results/p1_flatlands_provenance_manifest/`, but its exact
size/SHA and gate result are tracked in `RECOVERY_STATE.json`. This split is non-official FlatLands
and currently authorizes the bounded direct-from-ZIP validation pipeline; the published K=128 runs
stream this archive without extracting it and remain validation-only because the split is non-official.
Their compact hand-offs are `site/data/flatlands_conpath_k128_validation.json`,
`site/data/flatlands_independent_k128_validation.json`, and
`site/data/flatlands_k128_paired_comparison.json`; the physical test labels remain locked.

The deterministic synthetic corridor generator in `src/pathrel/synthetic.py` remains the contract
harness for the P0 death test, not a substitute for the real-data pilot.

The official ORFD release has also been audited as the first second-domain candidate. It is a
ground-vehicle off-road dataset with synchronized RGB/LiDAR and image-plane traversability labels,
but it does not directly provide the metric hidden-grid/footprint-event contract used here. No
ORFD archive is present locally and no ORFD score is reported. The compatibility decision and the
adapter gates are recorded in [`ORFD_COMPATIBILITY_CHECK.md`](../ORFD_COMPATIBILITY_CHECK.md).

The official UnScenes3D mini raw and label packages are present under ignored
`data/raw/unscenes3d/`. The larger raw package has 13 scenes and 1,336 synchronized
image/cloud/calibration samples; the label package has 629 aligned occupancy, elevation, and
depth timestamps. Both local-map parts are acquired and the coordinate/projection audit is
complete. The ground-valid site-held-out manifest has 15,567 train and 1,529 validation queries.
Older correlated/independent/mean-map outputs omitted the `target_valid` hard support boundary and
are now superseded; their entry points are corrected and six clean target-valid-support adapters plus
K=128 paired evaluation now pass strict audits. The clean event Brier is 0.51142 (correlated) versus
0.51130 (independent) on two validation scenes, so no correlation advantage is claimed. The non-map
S4C-inspired coordinate-query control is unaffected. `location_6` test files/labels remain locked. See
[`UNSCENES3D_COMPATIBILITY_CHECK.md`](../UNSCENES3D_COMPATIBILITY_CHECK.md) and
[`UNSCENES3D_PROTOCOL.md`](../UNSCENES3D_PROTOCOL.md).
The frozen bridge rules are in [`UNSCENES3D_PROTOCOL.md`](../UNSCENES3D_PROTOCOL.md).

Planned paper-grade audits/adapters, in order:

1. FlatLands with a versioned scene-disjoint split and natural-query audit; ScanNet++ stays OOD.
2. ORFD for a secondary off-road label-semantics audit (**official semantics audit complete;
   adapter and sequence-held-out split still pending**).
3. UnScenes3D for the main support-surface occupancy experiment (**coordinate/manifest gates and
   clean target-valid-support validation diagnostic pass; two-scene result is not a cross-domain
   claim and any test use requires an explicit go/no-go**).
4. WildOcc for cross-dataset evaluation.

Each adapter must map valid physical truth to `TRAVERSABLE` or `BLOCKED` and preserve a separate
observation/label-validity mask. `UNKNOWN` is an observation condition, not a third terrain
state. Unobserved cells must never be assigned free ground truth unless a future or multi-view
reference supplies verified hidden-world supervision.
