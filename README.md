# ConPath: Connectivity-Calibrated Path Reliability

ConPath (the research code formerly named PathRel) is an isolated prototype for
**connectivity-calibrated stochastic occupancy**.
It does not publish ROS messages, control a robot, or depend on the imported `vi/` code.

The scientific object is

```text
q(s, g, r | X) = P(a support-valid path exists from s to g
                     for footprint radius r | observation X).
```

The first implementation deliberately starts from a rasterized BEV observation rather than
pretending that a dataset-specific RGB/LiDAR pipeline already exists:

```text
observation_bev [B, Cin, H, W]
  -> TinyBEVUNet
  -> correlated binary occupancy posterior
  -> K straight-through stochastic maps
  -> disk footprint erosion
  -> differentiable max-min connectivity propagation
  -> reachability [B, Q, R]
```

## What is implemented now

- a compact BEV encoder;
- a binary latent-world (`TRAVERSABLE`, `BLOCKED`) mean-logit head;
- low-rank global and convolutionally correlated local stochastic logits;
- straight-through relaxed categorical map samples;
- a differentiable, four-neighbor max-min reachability layer;
- discrete footprint-radius reachability curves;
- posterior-marginal NLL, variogram, and Brier/CRPS-style losses;
- exact NumPy ground-truth clearance labels;
- deterministic ambiguous-corridor synthetic data, including context families with different hidden
  doorway priors;
- a reproducible P0 death-test evaluator (`scripts/evaluate_p0.py`) with constant, independent-cell,
  direct-query, edge-connectivity, random-completion, deterministic, correlated-ablation, and
  correlated-event baselines;
- an exact NumPy merge-tree forward reference (`merge_tree_bottleneck_scores`) for many terminal
  queries on one map;
- smoke forward, smoke training, and unit tests.

## What is intentionally not claimed yet

- raw RGB-to-BEV lifting or PointPillars/SECOND integration;
- an ORFD, UnScenes3D, or WildOcc loader;
- SE(2) rectangular swept-footprint connectivity;
- the final top-K path/cut probability bounds;
- a real-data calibration or navigation result.

Those are separate milestones. The current code implements a synthetic contract harness designed
to test the hypothesis without coupling it to a large perception stack; the oracle proxy now passes
the synthetic death-test comparison, but a trained neural checkpoint has not yet been validated.

## Environment

Use Python 3.10-3.12. The repository's current default Python 3.14 PyTorch installation is not
usable, so create an isolated environment rather than modifying it:

```powershell
cd C:\Users\hairo\Desktop\G1-LioNAV\research\pathrel
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch numpy
```

Then run:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\smoke_forward.py
.venv\Scripts\python.exe scripts\train_synthetic.py --config configs\synthetic.json `
  --steps 1 --validation-size 2 --validation-samples 2
```

The last command only verifies the forward/backward/optimizer/checkpoint path. Running the full
configuration is a longer synthetic experiment and is not evidence of novelty until the constant,
independent-cell, direct-query, and event-calibration baselines are evaluated under the held-out
template protocol.

Run the P0 audit before interpreting any neural result:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test
PYTHONPATH=src .venv/bin/python scripts/benchmark_merge_tree.py
# On a CUDA-enabled machine, compare a trained neural checkpoint under the same split:
PYTHONPATH=src .venv/bin/python scripts/train_p0_neural.py --device cuda --warmup-steps 24
```

The evaluator writes `report.json`, `metrics.csv`, and `reliability.svg`. Its correlated row is an
audited synthetic posterior proxy, not a trained PathRel checkpoint; the death-test report must be
positive for both independent cells and direct query prediction before public-data work continues.

If PyTorch is unavailable, the pure NumPy label-oracle tests can still run with the system
Python; PyTorch tests skip with an explicit reason.

## Package map

```text
src/pathrel/
  labels.py               exact offline target generation
  stochastic_decoder.py  joint stochastic occupancy posterior
  reachability.py         footprint and max-min connectivity layer
  losses.py               proper task-level and map losses
  model.py                end-to-end core model
  synthetic.py            deterministic research smoke data
```

See `ALGORITHM.md` for the mathematical contract, training stages, baselines, and go/no-go
criteria.

## Versioning and publication

The local repository is branded **ConPath** while retaining the `pathrel` Python import namespace
for source compatibility. `origin` points to `https://github.com/s-team-git/ConPath.git`. After
configuring GitHub authentication, the explicitly destructive replacement requested for this
transfer can be performed from a clean tree with:

```bash
scripts/publish_conpath_remote.sh --confirm-replace
```

The script accepts the ConPath HTTPS or SSH remote and uses `git push --force-with-lease`; it therefore
refuses to run without the confirmation flag or if the working tree/remote does not match the expected
target. Renaming the GitHub repository
itself is a separate setting on GitHub; the code and distribution are already named ConPath.

## Interactive demo website

The repository also contains a static, GitHub Pages-ready walkthrough under [`site/`](site/).
It combines the tracked P0 figures, an interactive hidden-topology canvas, and the reproducible
video artifact [`conpath_p0_demo.mp4`](site/assets/conpath_p0_demo.mp4). The page deliberately labels
the correlated row as an **oracle proxy** and the current neural run as an active diagnostic; neither
is silently promoted to a real-data paper result.

**Online demo:** [https://s-team-git.github.io/ConPath/](https://s-team-git.github.io/ConPath/)

The URL is served by GitHub Pages through the tracked workflow below. If it shows a 404 initially,
open **Repository settings → Pages**, select **GitHub Actions**, and wait for the first deployment
to finish; subsequent pushes to `main` update the page automatically.

After refreshing `results/p0_death_test`, update the site snapshot and commit it with the same change:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_demo_site.py
/usr/bin/python3 scripts/build_demo_video.py
git add site
git commit -m "Refresh ConPath demo snapshot"
git push origin main
```

Every push to `main` uploads the tracked `site/` directory through
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml). Enable GitHub Pages with
**GitHub Actions** once in the repository settings to publish it.
