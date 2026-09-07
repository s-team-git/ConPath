# ConPath project page

This directory is a static, GitHub Pages-ready academic project page. The primary visual is now a
clean-support checkpoint-derived FlatLands K=128 comparison at
`assets/flatlands_k128_clean_candidate.png`; its machine-readable provenance is
`data/flatlands_k128_clean_candidate.json`. The TUM RGB-D Freiburg1/desk video is retained lower on
the page as a geometry-only pipeline pilot. No synthetic video or fabricated model figure is
referenced by `index.html`.

The 2026-09-03 model audit found that older FlatLands neural forwards did not explicitly hard-block
cells outside `epistemic_mask`. The old checkpoints and their support-clamped post-hoc replay remain
archived for recovery, while the hero now reports the independently audited clean-support rerun:
ConPath Brier `0.06749 +/- 0.00936` versus independent-decoder `0.09521 +/- 0.00703` over three
seeds. The paired independent-minus-ConPath Brier delta is `+0.02772 +/- 0.00993`, and every
per-seed scene-bootstrap interval is positive. This is still validation-only on the non-official
provenance split; physical test labels remain unopened.

The responsive page repeats the main comparison below the hero: 29.1% lower event Brier,
3.60% versus 3.90% false-safe rate at equal 30% coverage, and positive paired Brier intervals
for 3/3 seeds. The equal-coverage risk intervals all include zero; the caption discloses that
the PNG's 20.0% fixed-threshold reduction uses unequal coverage. Mobile navigation wraps and
reproduction cards do not force horizontal page scrolling.

## Clean paper analysis

The `#paper-analysis` section contains a nine-control table generated from
`data/flatlands_clean_paper_analysis.json`, plus clean reliability, equal-coverage risk,
nested K=32/64/128 and fixed-marginal shuffle figures. The UnScenes3D section adds a rigorous
observation-conditioned error-floor diagnostic. All five new figures include standalone PDFs.
Generate the statistical report and figure/table package with the commands in
[`PAPER_EVIDENCE.md`](../PAPER_EVIDENCE.md); `scripts/build_paper_evidence.py` updates the table
between the `CLEAN_PAPER_ROWS` markers in `index.html`. It does not regenerate the original
checkpoint PNGs or archived calibration snapshots.

## Refresh the real-data page

From the repository root, after downloading and extracting the TUM sequence:

```bash
/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --publish-site
/usr/bin/python3 scripts/build_demo_site.py
# Optional torch-only integration smoke on the same real BEV hand-off:
PYTHONPATH=src .venv/bin/python scripts/run_tum_rgbd_model_smoke.py --device cpu
```

The first command writes the reproducible pilot report under
`results/tum_rgbd_freiburg1_desk_pilot/` and copies compact derived assets into `site/`. The second
command rebuilds `data/tum_rgbd_pilot.json`, `data/flatlands_audit.json`, the validation-only
`data/flatlands_baselines_validation.{json,js}`, their browser-local JS mirrors, and the FlatLands
audit/baseline SVGs. The page renders pooled, radius-stratified, and source-stratified baseline
tables from that snapshot. It expects the bounded query, provenance-audit, and first validation
baseline reports under `results/`; use `--skip-flatlands` or `--skip-flatlands-baselines` only when
intentionally rebuilding a partial page. Raw datasets and checkpoints remain under ignored paths
and are never committed.

The legacy synthetic P0 snapshot can still be regenerated for development with
`scripts/build_demo_site.py --include-legacy`, but it is intentionally excluded from the public page.

The model comparison is generated with `scripts/render_flatlands_k128_advantage.py`. It records all
six prediction hashes, both canonical checkpoint hashes, fixed visual sampling seeds, exact query
keys, the support policy, and label-derived case-selection rules. One case recovers a real path; the
other avoids an independent-decoder false-safe decision. Both show observed input, posterior mean,
the first four deterministic sample worlds, and reference support. They explain the clean-support
validation candidate and are not unbiased effect estimates or final paper claims.

The hero comparison and qualitative panels are deliberately large. The page adds a four-step visual key
(`observe → imagine → account for size → decide`) and plain-language captions so a reader can map
each color and panel to the planning event. The desk-surface pilot is explicitly marked as an
appendix-style geometry demonstration; it must be replaced by a ground-robot/floor scene before a
navigation claim is presented.

The FlatLands section retains the older three-seed independent-decoder causal-control snapshot for
audit continuity, but its unmasked PathRelNet probabilities are superseded. The clean-support
three-seed rerun and paired report are now the current validation evidence; the old post-hoc figure
is retained under its original filename as an archive record.

The earlier K=128 snapshots remain archived as reproducibility records, but their unmasked neural
event values are superseded by the support-boundary audit and must not be cited as current evidence.
The post-hoc corrected independent-minus-correlated Brier delta is `+0.01886 +/- 0.00963` across
three seeds, and each per-seed 2,000-resample paired interval is positive. The JSON hero metadata
links back to the full ignored validation manifests and paired comparison. This remains a recovery
effect-size check, not a significance, test, leaderboard, or clean-training claim.

The page also archives the deterministic PathRelNet posterior mean-map diagnostic. Its event and
map metrics predate the support clamp and are explicitly superseded; it is not current comparative
evidence.

The FlatLands reliability/selective-risk curves generated by
`scripts/evaluate_flatlands_calibration.py` are now a mixed historical view: their stochastic and
mean-map PathRelNet traces are superseded, while the direct S4C-inspired coordinate-query trace is
unaffected. The fixed baseline table above them remains valid because its completion samplers only
populate the released support-masked unknown region. No test split was opened.

The page also includes an UnScenes3D ground-vehicle validation section. A 2026-09-03 audit found the
same support-boundary class of error in the older map-derived paths: the complement of
`target_valid` was not hard-blocked before posterior sampling. Six fresh F=16 runs now use the
corrected support contract. The exact K=128 mean-map event Brier is `0.51142 +/- 0.00198` for
correlated ConPath and `0.51130 +/- 0.00218` for the matched independent decoder (paired delta
`-0.00012 +/- 0.00021`), so this second-domain adapter does not show a measurable correlation win
on its two held-out validation scenes. It is a support-consistency/transfer diagnostic, not a final
cross-domain result. The clean ground-robot panels are
`assets/unscenes3d_clean_candidate_positive.png` and
`assets/unscenes3d_clean_candidate_failure.png`, with provenance in
`data/unscenes3d_clean_support_k128_candidate.json` and
`data/unscenes3d_clean_candidate_qualitative.json`. The historical pre-correction controls and
panels remain labelled superseded; direct coordinate-query and train-fitted radius-prior controls
are unaffected. `location_6` and its test labels remain locked, and published JSON paths stay
repository-relative.

The official FlatLands implementation/weights audit is recorded in
[`OFFICIAL_FLATLANDS_CHECK.md`](../OFFICIAL_FLATLANDS_CHECK.md). The public dataset and documentation
are available, but the official repository currently marks model weights, construction code, and
additional benchmark tooling as planned; no official checkpoint is imported into the ConPath table.
The SceneSense diffusion reference is likewise documented as a 3-D pointmap/ROS contract that is not
directly comparable to the 2-D event protocol; see [`SCENESENSE_COMPATIBILITY_CHECK.md`](../SCENESENSE_COMPATIBILITY_CHECK.md).

## Preview locally

```bash
python3 -m http.server 8000 --directory site
```

Open <http://127.0.0.1:8000>. A local HTTP server exercises the same relative asset paths used by
GitHub Pages.

## GitHub Pages

`.github/workflows/deploy-pages.yml` publishes this directory on every push to `main`. Enable Pages
once in repository settings and choose **GitHub Actions** as the source. The workflow uploads tracked
files; it does not download datasets or run GPU experiments on the hosted runner.

## Attribution and claim boundary

The TUM RGB-D sequence is credited in the page and linked to the official source. Its RGB/depth and
motion-capture trajectory support a geometric reference-map pilot, not traversability or collision
ground truth. The FlatLands audit and baseline sections use only the scene-disjoint upstream
provenance split because the physical archive split leaks scenes. The baseline numbers are
validation-only diagnostics; they are not a public-data or final-paper claim. See
[`REAL_DATA_PILOT.md`](../REAL_DATA_PILOT.md), [`P1_DATA_AUDIT.md`](../P1_DATA_AUDIT.md), and
[`P1_BASELINE_PROTOCOL.md`](../P1_BASELINE_PROTOCOL.md) for exact protocols and limitations.

The September 6 final-projection audit supersedes UnScenes3D mean-map/qualitative v1: restoring
observed-free cells had reopened invalid support after sampling. Current candidate JSON and
PNG files come from v2 replay of the same six clean checkpoints. All 27,522 corrected event
predictions satisfy the pessimistic/optimistic bounds, and hidden-map metrics replay exactly.
The old versions remain in Git history and their original results directories.
