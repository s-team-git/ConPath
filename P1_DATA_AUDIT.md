# ConPath P1 data and event-identifiability audit

Status: **official split fails; non-official provenance split passes integrity plus bounded
mask/query gates; GO for a fixed streaming-baseline pilot, NO-GO for paper claims**

Updated: 2026-08-31 (America/New_York)

This is the durable gate between the passing synthetic P0 and any public-data training. P1 starts
with read-only data and query auditing. A dataset is not accepted merely because it has occupancy
maps: its labels, split, natural path-event distribution, license, and strong baselines must all be
auditable.

## Local inventory

The workspace contains about 830 MB of ignored TUM RGB-D `freiburg1/desk` data and the verified
2.055 GB FlatLands ZIP. It has no ORFD, UnScenes3D, or WildOcc assets. TUM supplies RGB/depth and
camera motion but no traversability or collision labels, so the existing geometric pilot cannot
serve as P1.

## Candidate decision

| Candidate | Primary evidence available | P1 fit | Current decision |
|---|---|---|---|
| FlatLands | aligned observed floor, complete floor, unobserved mask, valid/epistemic mask, metric metadata, official train/validation/test | closest match to completion-to-path-event calibration | **bounded provenance-split audit passed; baseline pilot next** |
| ORFD | RGB/LiDAR and traversable/non-traversable/unreachable annotations; about 30 GB | useful off-road label-semantics check, but not a multi-layout completion benchmark | secondary audit only |
| UnScenes3D | 3D occupancy, elevation, poses, and a released 14-scene mini split | relevant to final support-surface experiment but adds 3D semantics before event identifiability is established | defer to P2 |
| WildOcc | dense 3D occupancy labels derived from Rellis-3D; published label archive is about 18.08 GB plus source data | useful cross-dataset 3D occupancy test, not the cheapest first event audit | defer to P2 |

Primary sources:

- FlatLands [official repository](https://github.com/1ssb/Flat_Lands),
  [dataset card](https://huggingface.co/datasets/Rudra1ssb/FlatLands),
  [provenance](https://github.com/1ssb/Flat_Lands/blob/main/PROVENANCE.md), and
  [release notice](https://github.com/1ssb/Flat_Lands/blob/main/LICENSE);
- ORFD [official repository](https://github.com/chaytonmin/Off-Road-Freespace-Detection);
- UnScenes3D [official repository](https://github.com/ruiqi-song/UnScenes3D);
- WildOcc [official repository](https://github.com/LedKashmir/WildOcc) and
  [data instructions](https://github.com/LedKashmir/WildOcc/blob/main/docs/data.md).

## Frozen FlatLands acquisition contract

As checked on 2026-08-30, the official release advertises:

- Hugging Face dataset: `Rudra1ssb/FlatLands`;
- archive: `FlatLands_final_dataset.zip`;
- exact size: `2,054,773,316` bytes;
- SHA-256: `e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`;
- observations: 215,342 train, 26,890 validation, 28,343 test (270,575 total);
- packet files: `observed_floor.png`, `floor_map.png`, `unobserved.png`,
  `epistemic_mask.png`, and `metadata.json`;
- per-observation source dataset, scene ID, original split, metric resolution, crop, camera position,
  and provenance metadata;
- ScanNet++ only in the official test split, intended as OOD evaluation.

FlatLands is a derived research release. Its notice permits academic research/benchmarking subject
to citation and all applicable upstream terms. The source-specific metadata is authoritative; the
terms for 3RScan, ARKitScenes, Matterport3D, ScanNet, ZInD, and ScanNet++ must be reviewed before a
paper release. The repository currently says model weights, construction code, and additional
benchmark tooling are planned rather than available, so the archive alone does not satisfy the
strong-completion-baseline requirement.

Use the tracked downloader. It writes only an ignored `.part` file while transferring, resumes that
file after interruption, verifies both byte count and SHA-256, and atomically renames it only after
success. It deliberately does not unzip 270,575 five-file packets:

```bash
./scripts/download_flatlands.sh --check
./scripts/download_flatlands.sh --download
./scripts/download_flatlands.sh --check
```

The download completed and both the downloader and independent archive auditor reproduced the exact
size and SHA above. No extraction was performed.

## Full archive audit result

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_archive.py \
  --metadata-limit 0 --output-dir results/p1_flatlands_archive_audit_full
```

The archive structure is internally clean:

- 270,575 complete five-file packets and 270,575 parseable metadata objects;
- counts exactly 215,342 train, 26,890 validation, and 28,343 test;
- zero unsafe paths, duplicate names/global IDs, symlinks, encrypted files, unexpected packet files,
  incomplete packets, or missing scene identities;
- the physical directory token is `val/`, although the release page calls it validation; the audit
  normalizes both spellings.

However, the official split is observation-level, not scene-level. Full `(source, scene_id)` overlap
is:

| Split pair | Shared scenes | 3RScan | ARKitScenes | Matterport3D | ScanNet | ZInD |
|---|---:|---:|---:|---:|---:|---:|
| train / validation | 12,873 | 980 | 2,751 | 1,797 | 1,251 | 6,094 |
| train / test | 8,406 | 646 | 1,542 | 1,179 | 836 | 4,203 |
| validation / test | 6,800 | 495 | 969 | 1,010 | 681 | 3,645 |

For a concrete audit trace, ZInD scene `0371_floor_01_pano_18` contributes 16 train, 2 validation,
and 1 test observations. ScanNet++ remains test-only and is the clean OOD source. The official
in-distribution split therefore fails ConPath's scene-isolation gate and cannot be used for a
cross-scene calibration claim.

A source-stratified pixel sample found all four images to be binary 256x256 PNGs with values 0/255.
`observed_floor` was always a subset of `floor_map`; `unobserved` never overlapped observed-free
pixels. Cells outside `epistemic_mask` cannot be assumed blocked or free: sampled maps contained a
small number of floor/observed pixels outside it, so the exact boundary semantics still require a
quantified audit. Metadata confirms 0.01 m resolution, camera pixel `[128, 192]`, and a 256-cell crop
in the inspected packets.

## Scene-disjoint provenance manifest

Every packet also records `provenance.original_split`, which preserves the upstream source
dataset's split rather than FlatLands' observation-level redistribution. The complete audit found
no missing/unknown provenance split, missing global ID, duplicate global ID, scene assigned to
multiple provenance splits, or cross-split `(source_dataset, scene_id)` overlap.

| Provenance split | Observations | Scenes | Included sources |
|---|---:|---:|---|
| train | 203,373 | 13,339 | 3RScan, ARKitScenes, Matterport3D, ScanNet, ZInD |
| validation | 25,555 | 1,667 | 3RScan, ARKitScenes, Matterport3D, ScanNet, ZInD |
| test | 41,647 | 2,602 | the five sources above plus ScanNet++ |

ScanNet++ supplies 16,214 test observations and remains OOD-only. The auditor writes all 270,575
records in deterministic archive-member order to
`results/p1_flatlands_provenance_manifest/provenance_manifest.csv`; its SHA-256 is
`a5eb28123f0fa2e38cc8244e6675c1eb76bc9a534ee54f56ef9ed68c4bdbc77b`. The generating report is
bound to implementation commit `cce7703` and records
`provenance_prequery_gate_passed=true`, while the official `p1_prequery_gate_passed` remains false.

This is a non-official FlatLands split candidate: it reuses official upstream-source provenance but
does not repair or relabel the published FlatLands benchmark split. All ConPath baselines must use
this identical manifest and label it non-official.

## Bounded mask and natural-query audit

The target-blind auditor was frozen in implementation commit `472c952` and run directly against the
ZIP, without extraction:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_queries.py \
  --output-dir results/p1_flatlands_query_audit_bounded --overwrite
```

With seed `20260830`, stable SHA-256 ranks selected 32 distinct scenes per available provenance
split/source stratum and then one observation per scene: 512 observations across 16 strata. Each
observation received a 36-query metric polar stencil (0.4/0.8/1.2 m at 30-degree increments). The
start is metadata `[x,y]` converted to array `[row=y,col=x]`, using the camera cell when it is
observed/valid or the deterministic nearest observed-valid cell. Goals are retained using only
raster bounds, `unobserved`, and `epistemic_mask`. The complete `floor_map` is not read until those
queries are frozen. Exact four-neighbor disk-footprint labels use radii 0/0.1/0.2 m, which map to
0/10/20 cells at the released 0.01 m resolution.

The 512 packets contain 33,554,432 audited pixels. All PNGs are aligned binary 256x256 grayscale;
`observed_floor & ~floor_map` and `observed_floor & unobserved` are both exactly zero. Boundary
differences are real rather than silently relabeled: 5,415 observed-floor pixels and 22,539 target-
floor pixels lie outside `epistemic_mask`. The oracle always intersects the target with that mask,
so none of those cells is coerced to free. Every selected query start remained target-valid.

The 18,432 candidates yielded 6,735 target-blind selected endpoints. Of these, 2,082 were target-
invalid goals and are reported separately; 4,653 had valid endpoints. Across all splits, 121 were
disconnected at radius zero, 3,095 were reachable at radius zero but failed a larger footprint, and
1,437 remained reachable at 20 cm. Validation/test scene-weighted failure rates by source range from
0.5823 to 1.0000; every one of the 11 gated strata has at least 50 retained queries, at least eight
contributing scenes, and a failure rate above the frozen 0.10 minimum. The mask and bounded query-
balance gates therefore pass.

This is not evidence that every source/radius is well balanced. Test ARKitScenes has 115 retained
queries but zero 20 cm positives (only 3 are reachable at 10 cm), while ZInD is substantially less
saturated. Every future result must therefore be source- and radius-stratified; a pooled score could
hide this failure-mode shift. The audit establishes data/event availability, not model calibration.

Replayable artifacts are:

- `report.json`: 34,345 bytes, SHA-256
  `e210c8f30f06cf41700ec63d9e07213954a64d949d8ca4b483599b64f53156f4`;
- `selected_observations.csv`: 512 rows, SHA-256
  `4e7ae4c992cf943ab81618e3826c4748fcaaa97c3c4d7cb187518ee3fe6a9409`;
- `queries.csv`: 18,432 rows, SHA-256
  `33e7f8a0343269b0dde47b428b3be622c80effdb0f80ae34b352ca282018d60d`.

## Streaming replay adapter

`src/pathrel/flatlands_data.py` now implements the fixed first-stage benchmark hand-off. It opens
the release ZIP lazily per process, verifies the frozen selection/query hashes and archive byte
count, and filters exclusively on `provenance.original_split`; the physical archive directory is
retained only as a member locator. For every accessed packet it validates metadata identity and
mask relations, then reconstructs all 36 query coordinates and selection statuses from
`observed_floor`, `unobserved`, `epistemic_mask`, camera position, and metric resolution before the
complete target is used by the caller. Its model input is three channels—observed floor,
unobserved-region mask, and epistemic-valid mask—and its loss mask is
`unobserved & epistemic_mask`.

An all-packet replay decoded 512/512 selected observations directly from the ZIP: 160 train, 160
validation, and 192 test by provenance split, all 256×256. It reproduced 4,653 retained queries and
10,452,053 valid hidden-region cells. A two-worker PyTorch DataLoader smoke also completed. Unit
tests prove that requesting the physical archive split cannot silently substitute for provenance,
that tampered query geometry is rejected, and that padded query collation preserves its explicit
mask. The adapter does not authorize training on the leaking official split.

The project-site builder consumes the two frozen reports and generates
`site/data/flatlands_audit.json`, a browser-local mirror, and two SVG figures. The public section
shows source/radius target prevalence, selected-query outcome counts, all 11 gated strata, and the
official/provenance scene-overlap contrast. Each asset carries the same non-model claim boundary.

## Acceptance gates

All gates are per split and per source dataset, not only pooled across observations.

1. **Integrity:** exact archive size and SHA-256 match the frozen contract; unsafe ZIP paths,
   duplicate members, missing packet files, and corrupt PNG/JSON members are zero.
2. **License/provenance:** every observation has a recognized source and scene ID; upstream terms
   are recorded before publishing derived artifacts.
3. **Split isolation:** `(source_dataset, scene_id)` intersections across train, validation, and test
   are empty. Observation-level disjointness alone is insufficient.
4. **Semantics:** observed floor agrees with the complete target wherever it is observed; the
   unobserved and epistemic masks have a documented, empirically verified polarity; labels outside
   valid support are never coerced to free.
5. **Natural queries:** query coordinates are chosen from the observation, camera pose, raster
   bounds, metric scale, and validity mask only. Complete floor labels may score a fixed query but
   may not select it. Start is the camera cell or nearest observed-free cell. Goals come from a
   deterministic metric polar stencil before target inspection.
6. **Event identifiability:** report radius-zero disconnected queries, footprint-induced failures
   (reachable at radius zero but not at a larger physical radius), high-clearance positives, and
   invalid endpoints. At least 10%-15% of retained queries must be disconnected or
   footprint/bottleneck failures in validation and test; otherwise FlatLands is only a completion/OOD
   baseline, not the main event benchmark.
7. **No pseudoreplication:** a frozen scene-disjoint manifest is shared by every method, and
   confidence intervals/metrics are also scene-weighted so many observations from one room cannot
   dominate. The leaking official FlatLands split is never presented as cross-scene evidence.
8. **Baseline readiness:** deterministic completion, independent cells, direct query, and available
   official stochastic/flow completion outputs must use identical queries and masks. Missing
   official weights/tooling remains a blocker for a paper claim, even if the data audit passes.

## Query-audit protocol

- Read and validate the ZIP directly before extraction.
- Deterministically stratify observations by split, source, and scene.
- Convert physical radii to integer cells using each sample's metadata resolution; record both.
- Use the exact binary four-neighbor disk-footprint oracle for labels.
- Report counts and rates by split/source/radius plus scene-weighted bootstrap intervals.
- Save the seed, archive SHA, selected observation IDs, queries, polarity decision, and rejected
  reasons in JSON/CSV. Selection must be replayable without model outputs.
- The completed bounded sample is the fixed first baseline-evaluation manifest. Do not silently
  resample it after viewing model results.

## Current decision and next command

**NO-GO on the official split. The bounded streaming loader is complete; GO only for fixed
baselines on the explicitly non-official provenance split.** The next experiment must use the
already frozen selected-observation/query CSVs, keep ScanNet++ as OOD-only, and compare deterministic
completion, independent cells, direct query, and correlated completion under identical masks and
queries. Do not start a large neural run or make a public-data claim before those code paths and
source/radius-stratified metrics are verified.

The exact command for reproducing the bounded query result is:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_queries.py \
  --output-dir results/p1_flatlands_query_audit_bounded --overwrite
```

The next implementation milestone is a unified evaluator plus deterministic completion,
independent-cell, and direct-query baselines on this bounded manifest. Full extraction remains
unnecessary, the published
archive directories must never be described as scene-disjoint, and missing official model
weights/tooling remains a blocker for a paper claim.
