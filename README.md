# ConPath: Connectivity-Calibrated Path Reliability

[直接查看模型效果](https://s-team-git.github.io/ConPath/#effects) · [完整研究记录](https://s-team-git.github.io/ConPath/research.html)

**读取范围更正：旧训练／验证包含原发布test中的8／5条观测，本轮回放也读取了这5条验证观测。更早的512观测查询审计检查过53条物理test观测（含32条ScanNet++）。因此撤回“物理test从未读取”的说法；旧 `test_evaluated=false` 等标志不能证明物理测试未触碰。详见[读取范围更正记录](site/data/flatlands_read_scope_erratum.json)。已停止进一步读取物理test图片，最终留出集需要审核全部历史访问并排除已检查的父级地点。**

**2026-09-09 最新更正：旧 FlatLands 验证中27/160个观测与训练共享父级建筑／地点，不能据此主张新建筑泛化或超过其它论文。** 本轮已完成现有模型K=4与简单规则比较、90条论文公开成绩整理和新父级分组候选。详见[最新中文评估与改进方案](BASELINE_REVIEW_ZH.md)；[网站最新评估](https://s-team-git.github.io/ConPath/#baseline-review)提供标注图表。当前没有新增长训练，正式测试仍封存。

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
- a versioned valid-support clamp for FlatLands posterior worlds, a six-checkpoint K=128 post-hoc
  recovery audit, and completed clean three-seed ConPath/independent K=128 training with strict
  validation replay (validation-only, with all pre-correction PathRelNet-derived matrices marked
  superseded);
- a train/validation-only UnScenes3D ground-valid adapter with coordinate checks, correlated and
  independent posterior controls, a deterministic mean-map evaluator, and an S4C-inspired
  coordinate-query control; six clean target-valid-support adapters and a K=128 paired diagnostic
  now pass strict audits, while its `location_6` test site remains locked;
- tracked machine-readable recovery state with byte/SHA verification for required ignored results;
- an exact NumPy merge-tree forward reference (`merge_tree_bottleneck_scores`) for many terminal
  queries on one map;
- smoke forward, smoke training, and unit tests.

## What is intentionally not claimed yet

- raw RGB-to-BEV lifting or PointPillars/SECOND integration;
- an official ORFD or WildOcc loader, or a final public-data/cross-domain result (the current
  UnScenes3D adapter is validation-only, its pre-correction map results are superseded, and its test
  site is locked);
- SE(2) rectangular swept-footprint connectivity;
- the final top-K path/cut probability bounds;
- a paper-grade traversability, collision, or real-robot navigation result.

Those are separate milestones. The code now has both a synthetic contract harness and a real RGB-D
geometry path. The real pilot is explicitly a reference-map audit because TUM does not provide
traversability labels. The corrected neural model passes the synthetic P0 gate in two optimization
seeds and its matched no-reach ablation fails. The pre-correction FlatLands checkpoints have been
re-evaluated with invalid support hard-blocked; the correlation advantage survives, but those
checkpoints were trained under the old forward and therefore remain recovery diagnostics. Clean
K=128 retraining now passes strict replay, with ConPath Brier `0.06749 +/- 0.00936` versus
independent `0.09521 +/- 0.00703`; the physical test split and any paper-grade public-data claim
remain gated.
The same audit found that older UnScenes3D map-derived forwards omitted the `target_valid` support
clamp; their model/mean-map/qualitative artifacts are now historical only, while the non-map
coordinate-query and radius-prior controls remain usable. The clean UnScenes3D K=128 mean-map
diagnostic is `0.51142 +/- 0.00198` versus `0.51130 +/- 0.00218` on two validation scenes, so it
does not establish a correlation win. Clean training and rendering pass the support mask explicitly,
and `location_6` remains unopened.

After an interrupted session, start with:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_recovery_state.py --quick
```

Then read `WORK_PLAN.md` for current tasks and the latest entry in `CONTINUATION.md` for the hand-off.
The full verifier (without `--quick`) rehashes the registered ignored
artifacts, including the FlatLands archive.

The next paper evidence package is [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md): a nine-control
same-query table, equal-coverage risk, exact nested K replay, fixed-empirical-marginal spatial
shuffle, and an observation-conditioned error bound for UnScenes3D. The shuffle raises FlatLands
Brier from `0.06749` to `0.16139` with every empirical cell probability unchanged, while the
same-checkpoint mean map is close at `0.06957`. Equal-coverage risk improvement over the matched
independent decoder is not established. The UnScenes3D adapter imposes a validation Brier lower
bound `0.47574`; its observation model needs attention before further second-domain training.
These findings refine the working paper, not the locked-test or final-publication gate.

The clean three-seed training ablations are complete: removing event loss yields Brier
`0.20425 +/- 0.00322`; removing decoder global factors yields `0.09499 +/- 0.00157`,
versus full ConPath `0.06749 +/- 0.00936`. All six paired scene Brier intervals favor
the full model, while every equal-30%-coverage risk interval includes zero. The
[Chinese evaluation summary](EVALUATION_SUMMARY_ZH.md) includes all seeds, source/radius
strata, and 12 labelled checkpoint-derived probability/footprint images. External
LaMa/ensemble, flow and native CogniPlan comparisons are still required; tests stay locked.

## Environment

The clean September 2026 checkpoints and the latest full regression were produced with
Python 3.13.13 at `/home/hairo/miniconda3/bin/python3.13`. Use that interpreter on this machine
for checkpoint replay; the separate `.venv` is Python 3.11.15 and does not reproduce the checkpoint
serialization environment. Fresh source-only environments may use Python 3.10-3.13. For example:

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
  unscenes3d.py           label-free LiDAR adapter and ground-valid query geometry
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
and validation-candidate milestones, not a final paper result. Reproduce the audit without extraction using:

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

The current recovery hand-off is the clean candidate
`site/data/flatlands_k128_clean_candidate.json`, with its checkpoint-derived visual at
`site/assets/flatlands_k128_clean_candidate.png`. It summarizes the corrected six-checkpoint
K=128 rerun: correlated Brier `0.06749 +/- 0.00936` versus independent `0.09521 +/- 0.00703`,
over 4,224 validation events per seed, with positive paired intervals for all three seeds. The
old checkpoint directories and pre-correction snapshots remain intact as an audit trail, while
clean replacements are written to `results/p1_flatlands_conpath_k128_support_clamped_v1/` and
`results/p1_flatlands_independent_k128_support_clamped_v1/`. No test label was read.
The fail-closed `scripts/wait_and_finalize_flatlands_valid_support.sh` watcher waited for all six
reports, audited support/checkpoint/prediction/replay contracts, and wrote the clean candidate only
after every per-seed paired Brier interval had a positive lower bound. The analogous UnScenes3D
supervisor produced `site/data/unscenes3d_clean_support_k128_candidate.json` and two ground-robot
candidate panels; both packages remain validation-only and are promoted to the local page only after
manual review.

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
media now show the clean trained K=128 FlatLands correlated-versus-independent comparison, including
three-seed metrics, posterior maps, and fixed sample worlds; a separate UnScenes3D section shows the
clean ground-robot transfer diagnostic and its deliberately visible false-safe case. The real TUM
RGB-D `freiburg1/desk` video remains lower on the page as a geometry-only appendix. All model media
are explicitly validation candidates, not synthetic media or final paper/test results.

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
