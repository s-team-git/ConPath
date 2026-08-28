# Data

Raw public datasets are not committed here. The first runnable milestone uses the deterministic
synthetic corridor generator in `src/pathrel/synthetic.py`.

Planned adapters, in order:

1. ORFD for a small 2-D/2.5-D problem audit.
2. UnScenes3D for the main support-surface occupancy experiment.
3. WildOcc for cross-dataset evaluation.

Each adapter must map valid physical truth to `TRAVERSABLE` or `BLOCKED` and preserve a separate
observation/label-validity mask. `UNKNOWN` is an observation condition, not a third terrain
state. Unobserved cells must never be assigned free ground truth unless a future or multi-view
reference supplies verified hidden-world supervision.
