# ConPath P1 data and event-identifiability audit

Status: **archive verified; official split fails; provenance split passes pre-query integrity;
NO-GO pending query audit**

Updated: 2026-08-30 (America/New_York)

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
| FlatLands | aligned observed floor, complete floor, unobserved mask, valid/epistemic mask, metric metadata, official train/validation/test | closest match to completion-to-path-event calibration | **first audit target** |
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
does not repair or relabel the published FlatLands benchmark split. All ConPath baselines would need
to use this identical manifest. It only authorizes the bounded query audit, not extraction,
training, or a paper claim.

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

## Query-audit protocol to implement after integrity check

- Read and validate the ZIP directly before extraction.
- Deterministically stratify observations by split, source, and scene.
- Convert physical radii to integer cells using each sample's metadata resolution; record both.
- Use the exact binary four-neighbor disk-footprint oracle for labels.
- Report counts and rates by split/source/radius plus scene-weighted bootstrap intervals.
- Save the seed, archive SHA, selected observation IDs, queries, polarity decision, and rejected
  reasons in JSON/CSV. Selection must be replayable without model outputs.
- Start with a bounded audit sample; do not train a loader/model until the full metadata split check
  and the validation/test event-balance gate pass.

## Current decision and next command

**NO-GO on the official in-distribution split and NO-GO to training.** The upstream-provenance
manifest passes pre-query integrity, so a bounded salvage audit is allowed without extraction:

1. deterministically sample scenes by provenance split and source, then choose one observation per
   sampled scene without target inspection;
2. retain ScanNet++ only as OOD test;
3. quantify pixel/mask semantics and natural-query balance directly from a bounded ZIP sample;
4. label the provenance split as non-official FlatLands and require all baselines to retrain on it.

The exact next command for reproducing the current full metadata result is:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_archive.py \
  --metadata-limit 0 --write-provenance-manifest \
  --output-dir results/p1_flatlands_provenance_manifest --overwrite
```

The next implementation milestone is the bounded query audit. Do not extract or train merely to
follow the official observation split.
