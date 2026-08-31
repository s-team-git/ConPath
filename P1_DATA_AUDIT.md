# ConPath P1 data and event-identifiability audit

Status: **pre-download audit; NO-GO for public-data or paper claims**  
Updated: 2026-08-30 (America/New_York)

This is the durable gate between the passing synthetic P0 and any public-data training. P1 starts
with read-only data and query auditing. A dataset is not accepted merely because it has occupancy
maps: its labels, split, natural path-event distribution, license, and strong baselines must all be
auditable.

## Local inventory

The workspace currently contains about 830 MB of ignored TUM RGB-D `freiburg1/desk` data and no
FlatLands, ORFD, UnScenes3D, or WildOcc assets. There is no public-data loader, split manifest, or
natural-query audit script yet. TUM supplies RGB/depth and camera motion but no traversability or
collision labels, so the existing geometric pilot cannot serve as P1.

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
7. **No pseudoreplication:** official splits are retained, and confidence intervals/metrics are also
   scene-weighted so many observations from one room cannot dominate.
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

**GO to integrity-controlled FlatLands acquisition; NO-GO to extraction or training.** After the
tracked downloader and this document are committed, the exact next state-changing command is:

```bash
./scripts/download_flatlands.sh --download
```

If interrupted, rerun the same command; it resumes `data/raw/flatlands/FlatLands_final_dataset.zip.part`.
Do not delete or rename a mismatching completed file automatically. Record the verified archive and
the next audit command in `CONTINUATION.md` before opening it.
