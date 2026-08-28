# ConPath Pro 6000 transfer package

This directory is a self-contained research prototype. It intentionally does **not** depend on
the repository's imported `vi/` code, ROS, G1 drivers, or a local virtual environment. Transfer
the clean source tree to the GPU machine and create a new environment there.

## 1. Verify the target machine

Run these commands before installing anything:

```bash
nvidia-smi
python --version
```

Use Python 3.10--3.12. Install a CUDA-enabled PyTorch wheel matching the driver and platform from
the official PyTorch selector. Then install the small project dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
# Install torch from https://pytorch.org/get-started/locally/
python -m pip install -r requirements-gpu.txt
python scripts/check_environment.py
```

The check must report `cuda_available: true` and the intended GPU name before a long run.

## 2. Run the code-path smoke test

```bash
export PYTHONPATH=src                 # PowerShell: $env:PYTHONPATH="src"
python -m unittest discover -s tests -v
python scripts/smoke_forward.py
python scripts/train_synthetic.py \\
  --config configs/synthetic.json \\
  --device cuda \\
  --steps 1 --validation-size 2 --validation-samples 2 \\
  --checkpoint checkpoints/pro6000_quick_smoke.pt
```

This only checks forward/backward/optimizer/checkpoint execution. It is not a paper result.

`CONFERENCE_READINESS.md` records the current, deliberately conservative ICRA/IROS assessment and
the evidence gates that must pass before a submission claim is made.

## 3. Recommended first GPU experiment

Do not immediately scale the map or query count. The current reachability prototype performs an
iterative max-min propagation, so runtime grows with `H*W`, samples, queries, and radii. First use
the default 24x24 configuration and record wall-clock time and peak memory. Only then increase
`num_samples`, batch size, or map resolution.

The next scientific task is the P0 death test in `ROADMAP_ZH.md`: implement constant,
independent-cell, and direct-query baselines, add visible context families with different hidden
door priors, and compare task-level Brier/ECE. Do not report calibration or novelty before that
comparison is complete.

The runnable audit is now:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test
PYTHONPATH=src .venv/bin/python scripts/benchmark_merge_tree.py
# After CUDA is visible, train the neural model on the identical template-held-out protocol:
PYTHONPATH=src .venv/bin/python scripts/train_p0_neural.py --device cuda
```

It writes JSON/CSV/SVG artifacts and keeps a separate FlatLands-not-present status instead of
silently treating missing real data as a pass.

## 4. What is deliberately absent

The package does not yet contain raw RGB/LiDAR loaders, a dataset-specific BEV encoder, real-data
calibration results, SE(2) vehicle geometry, or a final scalable path/cut probability algorithm.
Those are planned milestones, not completed claims.
