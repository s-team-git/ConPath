# ORFD compatibility check

Status: **candidate secondary domain; semantics audited, no local run**  
Checked: 2026-09-02 (America/New_York)

This note records the contract decision for ORFD before any download, adapter, or score is
started. It prevents a pixel-wise off-road freespace result from being presented as a
same-contract ConPath result.

## What the official release contains

The official [ORFD paper](https://arxiv.org/abs/2206.09907) describes 12,198 synchronized
LiDAR/RGB pairs from 30 off-road sequences (woodland, farmland, grassland, and countryside),
with varied weather and lighting. The paper states that each sequence is about 100 m and that
the RGB images are 1280x720. The official [implementation repository](https://github.com/chaytonmin/Off-Road-Freespace-Detection)
provides a public download route (about 30 GB), calibration, sparse/dense depth, LiDAR, RGB,
and `gt_image` files under `training`, `validation`, and `testing` directories. The repository
states that the code and dataset are released under Apache 2.0.

The paper defines three image-plane classes:

* `traversable`: area that does not pose a safety threat to the vehicle;
* `non-traversable`: objects/regions that threaten safe driving;
* `unreachable`: currently distant regions (the paper gives sky as a typical example).

For the reported benchmark, `unreachable` is merged into `non-traversable` and the task is
pixel-wise RGB-image classification. The published split counts are 8,398 train, 1,245
validation, and 2,555 test pairs (about 7:1:2). The paper reports scene-stratified counts, but
does not provide a scene/site-held-out split protocol that can be assumed to satisfy our
sequence-isolation gate.

## Contract comparison

| Dimension | ConPath/FlatLands contract | ORFD release | Decision |
|---|---|---|---|
| Input | 3-channel partial support raster | RGB + 40-line LiDAR, projected depth/surface normal | Requires a new sensor adapter |
| Truth space | Metric grid with hidden-cell validity mask | Pixel-wise image-plane freespace labels | Not directly comparable |
| Target | Calibrated two-terminal footprint event, radii 0/10/20 | Pixel-wise traversable/non-traversable segmentation | Semantics can inform an audit, event cannot be copied |
| World reference | Observed/full support maps and provenance | Public README lists per-frame sensor products; no world-frame support-map/pose artifact is documented there | Metric BEV reconstruction is unverified |
| Split gate | Scene/site/sequence held out, no adjacent-frame leakage | Official paper gives pair counts and scene categories, not a held-out sequence recipe | Must construct and publish a new split before training |
| Local availability | Frozen archive is present and audited | No ORFD archive or extracted files are present in this workspace | No experiment is authorized yet |

## Allowed use

ORFD is a useful **secondary off-road semantics audit** and a credible source for a future
ground-vehicle/floor visual. A valid second-domain experiment would need to:

1. obtain the release through its official route and record the exact archive hash/license;
2. inspect calibration, timestamp/sequence identifiers, and whether ego poses support a metric
   ground-plane projection;
3. define a sequence/site-held-out split before reading test labels, with an adjacent-frame
   leakage check;
4. map only physically supported ground regions to `TRAVERSABLE`/`BLOCKED`, preserve an explicit
   observation/label-validity mask, and keep distant/sky `unreachable` regions out of the hidden
   ground target; and
5. implement a separate evaluator for the resulting metric footprint event, then compare all
   methods under that new contract.

Until those checks pass, ORFD may be cited and used to motivate the visual/label discussion, but
no ORFD F-score/IoU or OFF-Net number is entered into the FlatLands event table and no ORFD
number is described as evidence for ConPath.

## Sources

* [ORFD: A Dataset and Benchmark for Off-Road Freespace Detection (ICRA 2022)](https://arxiv.org/abs/2206.09907)
* [Official ORFD code/data repository](https://github.com/chaytonmin/Off-Road-Freespace-Detection)

