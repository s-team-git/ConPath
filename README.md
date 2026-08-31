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
- an interruption-safe neural P0 trainer with grouped repeated-world targets, visible context
  conditioning, atomic checkpoints, exact-hard/relaxed-backward event training, and held-out gates;
- a reproducible TUM RGB-D Freiburg1/desk real-data pilot (`scripts/run_tum_rgbd_pilot.py`) that
  lifts registered depth with MoCap poses into a world-frame reference raster and audits future
  start/goal events;
- read-only FlatLands ZIP integrity/split and bounded target-blind natural-query auditors, with a
  deterministic upstream-provenance manifest and replayable selected observations/queries;
- a process-safe FlatLands dataset adapter that streams the frozen 512-scene benchmark directly
  from the ZIP, filters only by the scene-disjoint provenance split, and replays query geometry
  from input-side masks before exposing targets;
- tracked machine-readable recovery state with byte/SHA verification for required ignored results;
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
traversability labels. The corrected neural model passes the synthetic P0 gate in two optimization
seeds and its matched no-reach ablation fails, but no trained checkpoint has yet been validated on a
paper-grade public benchmark.

After an interrupted session, start with:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_recovery_state.py --quick
```

Then read `CONTINUATION.md`. The full verifier (without `--quick`) rehashes the registered ignored
artifacts, including the FlatLands archive.

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
# On a CUDA-enabled machine, reproduce a trained run under the same split:
PYTHONPATH=src .venv/bin/python scripts/train_p0_neural.py \
  --device cuda --warmup-steps 24 --training-unit observation_group \
  --context-input plane --encoder-context-mode coord_global \
  --output-dir results/p0_neural_cuda_reproduction
```

The evaluator writes `report.json`, `metrics.csv`, and `reliability.svg`. Its correlated row remains
an audited oracle proxy; trained neural results are produced separately by `train_p0_neural.py`.
See `P0_DEATH_TEST.md` for the two-seed neural table, matched no-reach ablation, and exact claim
boundary. For the real-data pilot, run `scripts/run_tum_rgbd_pilot.py --publish-site` with system
Python 3; its report records the geometric label construction and claim boundary.

If PyTorch is unavailable, the pure NumPy label-oracle tests can still run with the system
Python; PyTorch tests skip with an explicit reason.

## Package map

```text
src/pathrel/
  labels.py               exact offline target generation
  flatlands.py            read-only ZIP integrity/provenance audit
  flatlands_query.py      target-blind bounded query selection/scoring
  flatlands_data.py       frozen direct-ZIP benchmark replay and collation
  flatlands_eval.py       exact-coverage scene-weighted event evaluator
  stochastic_decoder.py  joint stochastic occupancy posterior
  reachability.py         footprint and max-min connectivity layer
  losses.py               proper task-level and map losses
  model.py                end-to-end core model
  synthetic.py            deterministic research smoke data
```

See `ALGORITHM.md` for the mathematical contract, training stages, baselines, and go/no-go
criteria.

The official FlatLands observation split fails scene isolation. The upstream
`provenance.original_split` replacement is explicitly non-official but scene-disjoint, and its
512-scene direct-from-ZIP bounded mask/query audit passes the frozen data gate. The streaming
adapter is now implemented and verified across all 512 packets; this authorizes the fixed-baseline
milestone, not a trained public-data or paper result. Reproduce the audit without extraction using:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_queries.py \
  --output-dir results/p1_flatlands_query_audit_bounded --overwrite
```

Load the frozen benchmark without extraction using `FlatLandsReplayDataset`:

```python
from pathlib import Path
from pathrel.flatlands_data import FlatLandsReplayDataset

dataset = FlatLandsReplayDataset(
    Path("data/raw/flatlands/FlatLands_final_dataset.zip"),
    Path("results/p1_flatlands_query_audit_bounded/selected_observations.csv"),
    Path("results/p1_flatlands_query_audit_bounded/queries.csv"),
    split="validation",  # provenance.original_split, never the archive directory
)
sample = dataset[0]
```

See `P1_DATA_AUDIT.md` for the manifest hashes, target-blind query contract, source/radius
saturation caveat, and exact claim boundary. `P1_BASELINE_PROTOCOL.md` freezes the label-free
prediction schema, scene-weighted metrics, test-lock policy, and first three learned baselines.
`RECENT_BASELINES.md` freezes the bridge to recent 2024--2026 occupancy/completion work, including
the rule that cross-task 3-D mIoU/FID numbers are not copied into the FlatLands event table and the
final parameter-matching budget (ConPath F=16, approximately 120k parameters).

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
`freiburg1/desk` sequence. It also publishes generated FlatLands audit JSON plus source/radius and
query-outcome figures, all explicitly labelled as a bounded **data audit**, not model performance.
The page is styled as an academic project page with centered media and inspectable data tables.

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

After refreshing either tracked experiment report, rebuild the site snapshots and commit them with
the same change:

```bash
/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --publish-site
/usr/bin/python3 scripts/build_demo_site.py
git add site
git commit -m "Refresh ConPath project page data"
git push origin main
```

Once a GPU-enabled host or container exposes `/dev/nvidia0` and `nvidia-smi` succeeds, launch the
auditable public-data ConPath matrix (three seeds × full model / no-global-factor / no-event-loss)
with:

```bash
scripts/run_flatlands_conpath_matrix.sh
```

The matrix performs a CUDA preflight first and refuses to start under a host-visible-but-not-
passed-through GPU session. It keeps the FlatLands test split locked and writes one output directory
per seed/variant for later source/radius aggregation and qualitative-map rendering.

After a real ConPath checkpoint exists, render a same-scene, metadata-labelled map panel with:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_flatlands_qualitative.py \
  --checkpoint results/p1_flatlands_conpath_matrix/seed20260831_conpath/best.pt \
  --global-id <validation-global-id> --radius 10 --device cuda \
  --output site/assets/flatlands_conpath_qualitative.svg
```

The renderer can add the deterministic-completion checkpoint and direct-query manifest as optional
comparators. It refuses to invent a panel without a real checkpoint and writes a companion JSON
with the exact dataset, split, scene, query, radius, and event probabilities.

Every push to `main` uploads the tracked `site/` directory through
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml). Enable GitHub Pages with
**GitHub Actions** once in the repository settings to publish it.
