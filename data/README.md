# Data

Raw public datasets are not committed here (`data/raw/` is ignored). The repository now contains a
real-data pilot for the official TUM RGB-D `freiburg1/desk` sequence. It uses RGB, registered depth,
and the supplied motion-capture trajectory to build a geometric world-frame reference map; see
[`REAL_DATA_PILOT.md`](../REAL_DATA_PILOT.md) and `scripts/run_tum_rgbd_pilot.py` for the exact
command and claim boundary.

The deterministic synthetic corridor generator in `src/pathrel/synthetic.py` remains the contract
harness for the P0 death test, not a substitute for the real-data pilot.

Planned paper-grade adapters, in order:

1. ORFD for a small 2-D/2.5-D problem audit.
2. UnScenes3D for the main support-surface occupancy experiment.
3. WildOcc for cross-dataset evaluation.

Each adapter must map valid physical truth to `TRAVERSABLE` or `BLOCKED` and preserve a separate
observation/label-validity mask. `UNKNOWN` is an observation condition, not a third terrain
state. Unobserved cells must never be assigned free ground truth unless a future or multi-view
reference supplies verified hidden-world supervision.
