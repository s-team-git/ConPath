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
- a reproducible TUM RGB-D Freiburg1/desk real-data pilot (`scripts/run_tum_rgbd_pilot.py`) that
  lifts registered depth with MoCap poses into a world-frame reference raster and audits future
  start/goal events;
- an exact NumPy merge-tree forward reference (`merge_tree_bottleneck_scores`) for many terminal
  queries on one map;
- smoke forward, smoke training, and unit tests.

## What is intentionally not claimed yet

- raw RGB-to-BEV lifting or PointPillars/SECOND integration;
- an ORFD, UnScenes3D, or WildOcc loader;
- SE(2) rectangular swept-footprint connectivity;
- the final top-K path/cut probability bounds;
- a paper-grade traversability, collision, or real-robot navigation result.

Those are separate milestones. The code now has both a synthetic contract harness and a real RGB-D
geometry path. The real pilot is explicitly a reference-map audit because TUM does not provide
traversability labels; the oracle proxy passes the synthetic death-test comparison, but a trained
neural checkpoint has not yet been validated on a paper-grade public benchmark.

## Environment

Use Python 3.10-3.12. The repository's current default Python 3.14 PyTorch installation is not
usable, so create an isolated environment rather than modifying it:

```powershell
cd /path/to/ConPath
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
audited synthetic posterior proxy, not a trained PathRel checkpoint. For the real-data pilot, run
`/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --publish-site`; its report records the geometric
label construction and claim boundary.

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

## Academic project page

The repository contains a static, GitHub Pages-ready project page under [`site/`](site/). Its primary
video, teaser, depth visualization, and comparison figures are derived from the real TUM RGB-D
`freiburg1/desk` sequence. The page is styled as an academic project page (white background,
centered media, abstract/method/results sections) and labels the experiment as a **real-data
geometric pilot**.

The TUM sequence has RGB/depth and a motion-capture camera trajectory, but no traversability or
collision labels. Therefore the pilot is a reproducibility milestone, not a public navigation
benchmark or a validated neural P0 result. See [`REAL_DATA_PILOT.md`](REAL_DATA_PILOT.md) for the
protocol and exact command. The synthetic P0 media remain available only through the explicitly
opt-in development builder flag `--include-legacy`.

**Online demo:** [https://s-team-git.github.io/ConPath/](https://s-team-git.github.io/ConPath/)

The URL is served by GitHub Pages through the tracked workflow below. If it shows a 404 initially,
open **Repository settings → Pages**, select **GitHub Actions**, and wait for the first deployment
to finish; subsequent pushes to `main` update the page automatically. GitHub documents that Pages is
available for private repositories only on the applicable Pro/Team/Enterprise plans; on GitHub Free
for organizations the repository must be public. See the [Pages availability and setup guide](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

If the Pages menu is missing or the workflow cannot create a deployment, ask an organization owner
to allow Pages publication, or make this repository public if that is acceptable. For a private
Enterprise Pages site, use the **Visit site** URL shown in repository settings—the private-site URL
can differ from the public project URL above.

After refreshing the real-data pilot, update the site snapshot and commit it with the same change:

```bash
/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --publish-site
/usr/bin/python3 scripts/build_demo_site.py
git add site
git commit -m "Refresh ConPath real-data project page"
git push origin main
```

Every push to `main` uploads the tracked `site/` directory through
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml). Enable GitHub Pages with
**GitHub Actions** once in the repository settings to publish it.
