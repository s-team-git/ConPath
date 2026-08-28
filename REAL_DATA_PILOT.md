# Real-data pilot: TUM RGB-D Freiburg1/desk

The repository now includes a reproducible adapter for a real RGB-D sequence. It is intentionally
an intermediate experiment: the TUM RGB-D benchmark provides registered RGB/depth frames and a
motion-capture camera trajectory, but it does **not** provide traversability, collision, or robot
footprint labels.

## Reproduce

The raw archive is kept outside Git (`data/raw/` is ignored):

```bash
mkdir -p data/raw/tum_rgbd_freiburg1_desk
curl -L --fail --continue-at - \
  https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz \
  -o data/raw/tum_rgbd_freiburg1_desk/rgbd_dataset_freiburg1_desk.tgz
mkdir -p data/raw/tum_rgbd_freiburg1_desk/extracted
tar -xzf data/raw/tum_rgbd_freiburg1_desk/rgbd_dataset_freiburg1_desk.tgz \
  -C data/raw/tum_rgbd_freiburg1_desk/extracted
/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --publish-site
PYTHONPATH=src .venv/bin/python scripts/run_tum_rgbd_model_smoke.py --device cpu
```

The script associates RGB/depth timestamps with the official ground-truth poses, lifts sampled
depth pixels using the Freiburg-1 intrinsics, estimates the dominant horizontal support plane, and
builds a world-frame raster. It then evaluates future-frame start/goal queries using an
independent-cell completion and a spatially correlated temporal completion. The generated report,
CSV, figures, and temporary frames live under `results/tum_rgbd_freiburg1_desk_pilot/`; only the
small derived website assets are tracked.

The optional second command consumes the ignored `model_input.npz` hand-off and runs a bounded
randomly initialised `PathRelNet` forward pass. It checks that the real BEV reaches the stochastic
decoder and footprint-conditioned operator; it is an integration smoke, not training or a paper
result.

## Interpretation

The event labels are connectivity labels on the fused geometric reference raster. They are useful
for checking the data path and the event-calibration machinery, but they are not ground-truth robot
navigation labels. The pilot must not be cited as a public benchmark result or as evidence that the
current neural P0 checkpoint has passed. A paper-ready experiment still needs a dataset with an
auditable support/collision label, sequence/site-held-out splits, and a trained ConPath model.

## Source and attribution

TUM RGB-D dataset: [Computer Vision Group, TU Munich](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
Please retain the dataset's CC BY 4.0 attribution and terms when redistributing derived media.
