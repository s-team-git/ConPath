# ConPath continuation state

This file is the durable hand-off for interrupted Codex sessions. Update it after every material
diagnostic, code change, or experiment; do not rely on chat history or ignored `results/` alone.

## Recovery snapshot

- Updated: 2026-09-01 (America/New_York)
- Repository: `/home/hairo/pathrel_transfer/pathrel_pro6000`
- Durable checkpoint: `p1-flatlands-validation-baselines-v1` in tracked `RECOVERY_STATE.json`
- Recovery-state commit: resolve with `git log -1 --format='%h %s' -- RECOVERY_STATE.json`
- Last durable implementation commit: `883ea65` (independent-decoder control plus chunked validation to permit safe CUDA parallelism; pushed to GitHub)
- Scientific gate: **P0 GO; FlatLands bounded data gate and first validation baselines GO on a
  non-official provenance split; public-data model and paper claims not yet established**
- Active task: run multi-seed ConPath and ablations, then scalable connectivity and calibration /
  second-domain checks on the frozen bounded manifests. Do not use the leaking official split or
  extract the archive.

### GPU visibility and publication status (2026-08-31)

The full-access session exposes `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, and
`/dev/nvidia-modeset`; `nvidia-smi` sees an NVIDIA RTX PRO 6000 Blackwell (driver 580.173.02), and
the local `torch 2.13.0+cu130` venv reports `cuda_available=True, device_count=1, CC=12.0`.
Earlier no-GPU messages were a session/container passthrough failure, not physical GPU absence.
Training entry points still call `pathrel.gpu_diagnostics.cuda_unavailable_message` so future failures
remain layered and actionable.

After CUDA is visible, the reproducible six-run matrix (three seeds × ConPath / no-global / no-event-loss)
is launched with `scripts/run_flatlands_conpath_matrix.sh`. It writes an environment snapshot,
matrix manifest, per-run checkpoints, progress records, and external stdout logs; a partial run with
`latest.pt` is resumed, while ambiguous non-empty directories are refused.

`scripts/render_flatlands_qualitative.py` now renders a same-scene SVG from a real ConPath checkpoint
and the frozen FlatLands ZIP replay. It writes Observed BEV, ConPath posterior, optional deterministic
completion, and reference panels plus dataset/split/source/scene/query/radius/event metadata. It
refuses CUDA execution when the session cannot see a device and never fabricates a qualitative win.

The local `main` tree is clean and versioned; GitHub is synchronized through SSH over port 443. The
static site has been refreshed: its FlatLands section names the exact validation provenance
split/query/radii, highlights the best fixed baseline, and now lists the recent-paper bridge with
parameter matching and incompatibility status. Same-scene ConPath-versus-baseline map panels remain
reserved for a real checkpoint; no fabricated “ours is better” map is published.

`RECOVERY_STATE.json` is the machine-readable source of truth for required ignored artifacts,
byte counts, SHA-256 values, last verification, and the next action. Run the quick verifier first
after any interrupted session; run the full verifier when artifact integrity is in doubt. The
verifier is read-only and also prints the current Git status and the commit that last changed the
state file.

## Last completed work

The tracked repository already contains the TUM RGB-D geometric pilot, model hand-off smoke,
academic project page, exact NumPy merge-tree reference, and synthetic P0 baselines. The historical
neural failures below motivated the corrected two-seed result recorded later in this file.

The newest untracked/ignored run is `results/p0_neural_cuda_tuned01/` (generated 2026-08-28
06:01 local time). Its report records:

- 300 steps, 48 warm-up steps, 8 train samples, 128 validation samples;
- reachability weight 5.0 and categorical noise scale 0.1;
- event Brier `0.255772`, NLL `1.106823`, ECE `0.241109`;
- map-marginal Brier `0.007444`;
- event means by radius `[0.985555, 0.550191, 0.265706]` versus targets
  `[0.546875, 0.351562, 0.179688]`.

This is worse than the original 120-step neural run (`0.243578`) and the fixed P0 comparators:
independent Bernoulli `0.183172`, direct-query MLP `0.169888`. More generic step/weight tuning is
not the next action.

## Current diagnosis and implemented fix

The new `scripts/diagnose_p0_checkpoint.py` audit established that the tuned checkpoint is genuinely
over-open rather than merely noisy:

- true context-0/context-1 doorway-open rates are `0.21875/0.875`, while sampled rates are about
  `0.986/0.986`;
- about 30% of sampled doorway columns are fragmented, versus 0% in the targets;
- corrected unknown-region hard-map Brier is about `0.0949`; the old full-map soft score hid this;
- known visible evidence has zero violations.

Two concrete implementation faults were found and fixed in the current working tree:

1. with scaled Gumbel noise, the conditional categorical probabilities are
   `softmax(sample_logits / scale)`; the old code always used `softmax(sample_logits)`, so its
   reported/trained marginal did not match the hard-map generator when scale was not 1;
2. the old repeated straight-through extrema sent only about `2.7e-6` of an open-path closing
   gradient through the actual doorway. The model now uses exact hard events in the forward pass
   and the relaxed continuous maximum-bottleneck score only for the backward pass.

Joint training now scores only hidden cells, adds an empirical hard-map U-statistic Brier term, and
reports both conditional and empirical marginals plus doorway/context diagnostics.

## Interruption protection (verified)

`scripts/train_p0_neural.py` now writes one JSON record per step to `progress.jsonl`, atomically
saves `latest.pt` every `--checkpoint-every` steps, saves `interrupted.pt` on a caught failure or
Ctrl-C, preserves optimizer/model/Torch/CUDA/generator state, and refuses to overwrite an existing
run without `--resume`.

The recovery smoke ran steps 1-2, resumed the saved checkpoint, and completed steps 3-4 with the
loss history exactly `[1, 2, 3, 4]`. All tests currently pass: `Ran 35 tests ... OK`, `skipped=0`.

A 40-step 12x12 CPU regression completed without NaNs (event Brier `0.128662`, empirical full-map
Brier `0.020068`). It did not separate the two context priors and does not pass its reduced-protocol
direct-query baseline. This reduced setup has no positive radius-2 test events and is only a code
regression, not a scientific result.

## First official CUDA result after the gradient fix

`results/p0_neural_cuda_surrogate_v1/` completed 120 steps on the RTX PRO 6000. At 128 validation
samples it reports event Brier `0.163319`, ECE `0.017063`, and empirical full-map Brier `0.003747`.
Those aggregate scores beat independent (`0.183172`) and direct-query (`0.169888`). However, the
radius-zero predictions for context 0/1 are only `0.6027/0.6203`, versus targets `0.21875/0.875`;
the predicted gap recovers only about 2.7% of the target gap. Wall marginals, means, and factor
scales are likewise nearly context-invariant.

Therefore this result is **NO-GO despite its aggregate Brier**. The gate now additionally requires
recovering at least 50% of the held-out radius-zero context gap. The current working tree adds
coordinate channels and a globally pooled bottleneck context broadcast to the tiny P0 encoder;
the external `forward_features(...)` public-backbone path is unchanged.

The first CUDA evaluation-only resume also exposed and fixed a cross-device RNG restore bug:
serialized ByteTensor RNG states are explicitly moved back to CPU before `set_state`. The same
checkpoint then resumed successfully and completed the 128-sample evaluation.

`results/p0_neural_cuda_context_v2/` then tested that coordinate/global encoder for 120 steps. It
failed: event Brier `0.186264`, ECE `0.109511`, and context-gap ratio `0.0118`. A train-split
checkpoint diagnostic also showed almost identical context wall marginals (`0.1894/0.1909`), so
this is optimization/supervision collapse rather than held-out-template overfitting. The encoder
change alone is not a solution.

Replaying all training batch indices ruled out context sampling bias: joint-stage door rates were
`0.255/0.833` and full-train rates were `0.222/0.826`. The current working tree instead uses the
dataset's intended repeated-world structure: for each `(template, context)` it aggregates 24 worlds
into empirical map/event probability targets and applies proper scores to one shared observation.
The derived event still comes only from stochastic map samples. Legacy `--training-unit world`
remains available for old checkpoints.

`results/p0_neural_cuda_grouped_v3/` completed that grouped protocol. It reduced event Brier to
`0.169978` (direct-query is `0.169888`) and empirical hard-map Brier to `0.003525`, but the context
gap ratio remained only `0.0186`; the gate correctly failed. Grouping fixes target variance but not
the asymmetric conditional-input path.

The baseline evaluator and direct-query model receive the explicit visible `context` variable,
whereas the neural model so far had to infer it from the absolute position of an otherwise identical
2x2 landmark. The current working tree adds `--context-input plane`: a fourth, spatially broadcast
visible context-bit channel. It contains no doorway realization or event label. This gives the map
posterior the exact same conditioning variable as the baselines while retaining the original
three-channel `marker` mode for ablation/legacy checkpoints. All 35 tests pass; the official CUDA
run below is complete.

`results/p0_neural_cuda_contextplane_v4/` completed 120 steps with 128 validation samples and
**passed the tightened official gate**: event Brier `0.116377`, NLL `0.353872`, ECE `0.078647`,
empirical hard-map Brier `0.003383`, and false-safe@0.8 `0.125`. Radius-zero context predictions
were `0.4712/0.8476` versus targets `0.21875/0.875`, recovering `57.35%` of the target gap (minimum
required `50%`). This is a synthetic neural P0 result, not a public-data or paper result.

Residual weaknesses remain: context 0 is over-open and sampled doorway fragmentation is about
`13.7%/16.1%` for context 0/1 versus zero in the target.

`results/p0_neural_cuda_contextplane_noreach_v4/` is the matched ablation with only
`--reachability-weight 0` changed. It failed the gate: event Brier `0.191368`, NLL `0.795186`, ECE
`0.193570`, false-safe@0.8 `0.4258`, and context-gap ratio `0.2383`. Its empirical hard-map Brier
remained strong at `0.003098`, isolating the failure to joint/event structure rather than pixelwise
occupancy. Doorway fragmentation rose to `44.5%/71.6%`. Thus the reachability proper-score term is
material under this fixed seed: it improves event Brier by `0.074991` while the no-reach ablation
actually has slightly better marginal-map Brier.

`results/p0_neural_cuda_contextplane_seed20260828_v4/` completed the prescribed extra-seed run with
the dataset split fixed. It also passed: event Brier `0.111621`, NLL `0.346012`, ECE `0.071852`,
false-safe@0.8 `0.1125`, empirical hard-map Brier `0.002892`, and context-gap ratio `0.6957`.
Radius-zero context predictions improved to `0.3764/0.8329`. Fragmentation remained a limitation at
`16.1%/14.9%`, but it was far below the no-reach ablation's `44.5%/71.6%`.

Both full-model seeds independently pass every tightened gate; their event Brier range is
`0.111621-0.116377`, versus `0.169888` for direct-query and `0.183172` for independent Bernoulli.
The fixed-seed no-reach ablation fails. This closes the synthetic neural P0 gate, but it is not a
public-data or paper-level result.

All authoritative P0/readiness/roadmap/paper/transfer documents now record the same two-seed result,
matched ablation, synthetic-only boundary, and fragmentation limitation. The final regression run
reported `Ran 35 tests in 15.928s ... OK`, `skipped=0`; `smoke_forward.py` completed; and the 64x64
merge-tree reference matched exhaustive search at zero error for 8/64/512 queries (timing speedups
`3.09x/25.50x/161.19x`).

## P1 FlatLands archive audit

The workspace contains the ignored 830 MB TUM RGB-D pilot and the verified 2.055 GB FlatLands ZIP;
it has no ORFD, UnScenes3D, or WildOcc assets. Official-source review selected FlatLands as the first
P1 audit target because it provides aligned observed/full floor maps, unobserved and valid masks,
metric provenance, and split metadata. ORFD is a secondary off-road semantics audit;
UnScenes3D/WildOcc remain P2.

`P1_DATA_AUDIT.md` freezes the FlatLands release as `2,054,773,316` bytes with SHA-256
`e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`, defines scene-disjointness,
mask/polarity, natural-query, 10%-15% event-balance, provenance/license, and strong-baseline gates.
The official split remains NO-GO. `scripts/download_flatlands.sh` supports a
resumable `.part`, verifies size and hash, atomically finalizes the archive, and never extracts it.
`bash -n` passes and `--check` now validates the present archive.

The committed `src/pathrel/flatlands.py`, `scripts/audit_flatlands_archive.py`, and tests implement a
read-only auditor for the frozen archive hash, unsafe/duplicate/symlink/encrypted members, five-file
packet completeness, official split counts, malformed metadata, source/scene identity, duplicate
global IDs, and cross-split scene leakage without extraction. Reports and manifests are written by
fsync plus atomic rename; progress/failure logs are fsynced incrementally.

The official archive is now present at
`data/raw/flatlands/FlatLands_final_dataset.zip`. Its exact size is `2,054,773,316` bytes and both
independent checks reproduce SHA-256
`e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`. It has not been extracted.

The first bounded audit exposed the physical `val/` directory token; committed code normalizes it
to `validation`. The subsequent full scan in `results/p1_flatlands_archive_audit_full/` found all
270,575 complete packets and parseable metadata with zero unsafe/duplicate/symlink/encrypted/
unexpected members or missing identities. Official counts match exactly.

The decisive failure is scene leakage in the official observation-level split: train/validation
share 12,873 `(source, scene_id)` pairs, train/test 8,406, and validation/test 6,800. Every
in-distribution source leaks; ScanNet++ alone is test-only OOD. A concrete ZInD scene has 16/2/1
observations in train/validation/test. Therefore P1 remains **NO-GO on the official split**.

The full rerun from commit `cce7703` recovered a better split already encoded in every packet:
`provenance.original_split` preserves each upstream source's train/validation/test membership. It
contains 203,373/25,555/41,647 observations and 13,339/1,667/2,602 `(source, scene_id)` pairs, with
zero overlap, zero missing or unknown split values, and zero scenes assigned to multiple splits.
ScanNet++ contributes 16,214 observations only to provenance test. The deterministic 270,575-row
manifest is
`results/p1_flatlands_provenance_manifest/provenance_manifest.csv` (SHA-256
`a5eb28123f0fa2e38cc8244e6675c1eb76bc9a534ee54f56ef9ed68c4bdbc77b`). This is an explicitly
non-official FlatLands evaluation split, not a claim that the published archive directories pass.
Its pre-query integrity gate passes.

Implementation commits `8ee1c34` and `472c952` add a dependency-free grayscale PNG reader, stable
scene/observation sampling, target-blind metric polar queries, an exact linear-time disk-clearance
EDT, one-start multi-goal bottleneck scoring, atomic reports, and tests. The authoritative command
sampled 32 scenes per each of 16 provenance split/source strata (512 distinct scenes), one
observation per scene, and froze 36 queries per observation before reading `floor_map`. It read all
members directly from the ZIP and did not extract it.

The 512-observation mask audit found zero observed-floor/target disagreements, zero
observed/unobserved overlap, and zero invalid selected starts. It also quantified 5,415 observed and
22,539 target floor pixels outside `epistemic_mask`; the oracle explicitly excludes all such cells.
Of 18,432 query candidates, 6,735 passed target-blind selection, 2,082 then had target-invalid goals,
and 4,653 retained valid endpoints. Those contain 121 radius-zero disconnections, 3,095 larger-
footprint failures, and 1,437 20 cm positives.

Every validation/test source stratum passes the frozen minimum of 50 retained queries, eight
contributing scenes, and 0.10 scene-weighted disconnected/footprint rate; observed rates span
0.5823-1.0000. Thus the bounded mask/query data gate passes. It is not a calibration result and the
source/radius distribution is not uniformly balanced: test ARKitScenes has no 20 cm positives and
only 3/115 retained queries reachable at 10 cm. Future metrics must remain source/radius-stratified.
The reproducible result is `results/p1_flatlands_query_audit_bounded/`, generated from clean commit
`472c952`; report/selection/query SHA-256 values are registered in `RECOVERY_STATE.json`.

Implementation commit `a6ec796` completes the direct-ZIP hand-off in
`src/pathrel/flatlands_data.py`. It verifies the frozen CSV hashes and archive byte count, filters
only by `provenance.original_split`, lazily opens one ZIP handle per process, validates metadata and
masks, and reconstructs every query from input-side evidence before exposing targets. An all-packet
replay decoded 512/512 observations and reproduced split sizes 160/160/192, 4,653 retained queries,
and 10,452,053 valid hidden-region cells. All maps are 256x256; two-worker DataLoader and padded
query collation smokes pass. Tests reject archive-split fallback and tampered query geometry.

The same commit updates the project site with generated, tracked FlatLands audit data and figures:
`site/data/flatlands_audit.{json,js}`, `flatlands_reachability.svg`, and
`flatlands_query_outcomes.svg`. A real browser run populated 512 scenes, 4,653 retained queries,
11/11 gated strata, the three official overlap counts, and zero provenance overlap. The section is
labelled throughout as a bounded data audit and not a model/paper result. Full regression now reports
`Ran 52 tests in 1.104s ... OK`, `skipped=0`; `smoke_forward.py` also passes.

## P1 first validation baseline pass

The unified label-free evaluator, baseline runners, and canonical three-channel replay input are now
implemented. The evaluator joins targets only after exact prediction-key coverage is verified, uses
equal-scene primary weighting with 2,000-scene-cluster bootstrap intervals, and never reads the
locked test split during validation. The four validation-only methods and scene-weighted metrics are:

| Method | Brier | NLL | ECE | False-safe @0.8 | Coverage @0.8 |
|---|---:|---:|---:|---:|---:|
| Radius-prior control (train-only) | 0.15870 | 0.49283 | 0.01661 | 0.11156 | 0.33333 |
| Deterministic completion | **0.08556** | 1.18211 | 0.08556 | 0.08073 | 0.45450 |
| Independent-cell completion (K=32) | 0.22546 | 2.92768 | 0.24025 | **0.01454** | 0.20624 |
| Direct-query predictor | 0.09119 | **0.29788** | **0.04076** | 0.08335 | 0.37412 |

The deterministic completion has the lowest Brier; direct-query has the best NLL/ECE; independent
cells are a deliberately fragmented negative control with low false-safe coverage but poor event
calibration. These results establish evaluator plumbing and an initial difficulty baseline only. They
are not paper results: one seed, validation only, no official/public completion weights, no scalable
connectivity implementation, and no second domain yet. The direct-query run took 63.3 s (best epoch
42); marginal completion took 397.6 s including 353.4 s for K=32 event sampling on the RTX PRO 6000.

The project site now publishes `site/data/flatlands_baselines_validation.{json,js}` plus comparison
and reliability SVGs. Every public label says validation-only / test-locked / not a final paper result.

## First public-data ConPath pilot

The resumed `seed20260831_conpath` pilot completed on the RTX PRO 6000 after early stopping at
epoch 33 (best checkpoint at epoch 25). It used the historical low-capacity `F=8` encoder, 8
posterior worlds for training, 32 for validation selection, and the frozen provenance validation
split; the official archive split and test labels remained untouched. The exact-forward evaluator
produced 4,224 validation prediction rows with scene-weighted Brier `0.0836996`, NLL `0.385339`,
ECE `0.0550607`, false-safe@0.8 `0.0711530`, and zero radius-monotonicity violations. The
selection-time approximate event Brier was `0.0742096`. This is a single-seed validation pilot,
not a paper result, and is not directly capacity-matched to the existing `F=16` baselines.

The complete run manifest and predictions are retained under
`results/p1_flatlands_conpath_matrix/seed20260831_conpath/` (ignored artifacts); the next matrix
uses a separate output root with `F=16` and three fixed seeds. The matrix launcher now passes the
capacity argument into its manifest generator; recent strong methods remain required comparison
targets under the same contract before any final claim or site refresh.

## Capacity-matched F16 ConPath seeds

The first capacity-matched three-seed ConPath set is now complete under
`results/p1_flatlands_conpath_matrix_f16/`. All runs used the frozen provenance validation split,
8 training worlds, 32 validation worlds, exact-forward merge-tree evaluation, and no test labels:

| Seed | Best epoch | Selection Brier | Exact scene-weighted Brier | NLL | ECE | False-safe @0.8 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260831 | 4 | 0.08693 | 0.10809 | 0.64003 | 0.08514 | 0.04440 |
| 20260901 | 21 | 0.07559 | 0.08365 | 0.33534 | 0.05195 | 0.06595 |
| 20260902 | 10 | 0.08307 | 0.09647 | 0.39904 | 0.06902 | 0.06977 |
| **mean ± sample SD** | — | — | **0.09607 ± 0.01222** | **0.45814 ± 0.16071** | **0.06870 ± 0.01659** | **0.06004 ± 0.01367** |

These are still validation-only diagnostics, not final paper claims. The spread is material, and
the old F8 pilot (`0.08370` exact Brier) is not a fair capacity-matched comparison. The next
required step is the matched causal ablation set (no global factors, no event proper score,
independent decoder, deterministic mean map, and K convergence), followed by source/radius
uncertainty and recent strong-method comparisons under this same contract.

## F16 causal ablations completed

The two implemented causal ablations are now complete for all three fixed seeds, with exact
forward validation on the same 4,224-row-per-seed manifest:

| Variant | Exact scene-weighted Brier (20260831 / 20260901 / 20260902) | Mean ± sample SD | NLL mean ± SD | ECE mean ± SD |
|---|---:|---:|---:|---:|
| ConPath | 0.10809 / 0.08365 / 0.09647 | **0.09607 ± 0.01222** | 0.45814 ± 0.16071 | 0.06870 ± 0.01659 |
| No global factors | 0.12134 / 0.10812 / 0.11568 | 0.11505 ± 0.00663 | 0.95790 ± 0.06133 | 0.09649 ± 0.01168 |
| No reachability loss | 0.20048 / 0.20983 / 0.20635 | 0.20555 ± 0.00473 | 2.57912 ± 0.08818 | 0.21286 ± 0.01620 |

Removing global factors worsens mean Brier by `0.01898` and NLL by `0.49976`; removing the
reachability proper score worsens mean Brier by `0.10948` and NLL by `2.12098`. These are strong
mechanistic signals, not final paper claims: all rows are validation-only, and false-safe/coverage
trade-offs plus source/radius bootstrap intervals remain required. The independent decoder,
deterministic mean-map, K-convergence, and recent strong-method ports are still pending.

## Scalable connectivity checkpoint

The exact NumPy Kruskal reconstruction-tree oracle is now exposed as
`batched_merge_tree_bottleneck_scores` for `[B,K,H,W]` maps and `[B,Q,2]` query sets. It builds one
tree per map and answers all terminals with LCA lookups, so query cost is logarithmic after the map
build rather than another `H*W` relaxation per query. A strict test compares every batch/sample/query
entry to the existing single-map oracle. The synthetic CPU contract benchmark
`results/p1_flatlands_connectivity_benchmark/benchmark.json` uses 64×64 maps and K=4: 32/128/512/2048
queries take about 0.036/0.037/0.044/0.060 s, respectively. This is an exact-forward reference and
efficiency diagnostic only; a CUDA implementation and soft backward path are still pending.

`scripts/train_flatlands_conpath.py` is now the reproducible public-data neural entry point. It uses
the canonical replay channels, hidden-cell posterior NLL, variogram score, reachability U-statistic,
shared-start propagation, atomic checkpoints, and label-free validation predictions. Its final
validation path converts hard posterior worlds to exact disk-clearance maps and evaluates all queries
with the batched Kruskal merge-tree, while the bounded differentiable propagation is used only for
training/epoch selection. A one-scene CPU smoke completed end to end (including checkpoint/prediction
writing) with `paper_result=false` and validation subset evaluation disabled. Full validation runs
must still use the frozen 160-scene provenance split and multiple seeds before becoming paper results.

## Completion multi-seed controls and K convergence

Two additional completion seeds were trained under the frozen P1 protocol. They complete the
three-seed control set alongside `seed20260831`; all remain validation-only and test-locked:

| Seed | Best epoch | Map Brier | Map NLL | Deterministic event Brier | Independent K=32 event Brier |
|---:|---:|---:|---:|---:|---:|
| 20260831 | 15 | 0.11345 | — | 0.08556 | 0.22546 |
| 20260901 | 26 | 0.11640 | 0.37618 | 0.08816 | 0.23187 |
| 20260902 | 13 | 0.11450 | 0.36853 | 0.09199 | 0.23611 |

For seed `20260901`, one fixed completion checkpoint was evaluated with a shared posterior stream
at K=32/64/128. Event Brier was `0.231873/0.231068/0.230949`, NLL was
`2.992125/2.957632/2.936882`, and ECE was `0.248903/0.248603/0.248616`. The small change confirms
that K=32 is adequate for rapid diagnostics; final tables retain K=128. The evaluation script now
generates the maximum K once and reports all prefixes, so this convergence check is reproducible
without three independent expensive samplers.

## Recent-paper same-contract control

The first recent-method adaptation is a PaSCo-inspired three-subnet ensemble control. It uses the
three completion checkpoints above, the frozen three-channel FlatLands input, and a fixed total of
32 event worlds (11/11/10 per member). It is explicitly not a reproduction of PaSCo's camera/3-D
architecture. The exact scene-weighted validation result is map Brier `0.110545`, event Brier
`0.228984`, NLL `2.898173`, ECE `0.248410`, and false-safe@0.8 `0.015380`. The compact report and
member/prediction hashes are tracked in `site/data/flatlands_pasco_ensemble_validation.json`; the
full ignored artifacts remain under `results/p1_flatlands_pasco_ensemble_validation/`.

## Validation qualitative panels and website state

Two real-checkpoint FlatLands validation panels were rendered and copied to the tracked site:
`flatlands_qualitative_validation_positive.svg` (r=10, ConPath event `0.844`) and
`flatlands_qualitative_validation_uncertain.svg` (disconnected r=0 case, ConPath event `0.563`).
Each panel shows observed BEV, ConPath posterior, deterministic completion, and the reference map;
the second case is intentionally included to avoid a one-sided visual claim. The current TUM video
is still a geometry/BEV pilot. A final paper/site video must show posterior updates, radius-
conditioned reachability, baseline comparison, and at least one failure/overconfidence case.

## Independent-decoder control (completed)

The independent-cell decoder control has been added to
`scripts/train_flatlands_conpath.py` as `--decoder-variant independent`. It uses the same
three-channel replay input, optimizer, train/validation scenes, K=32 validation worlds, exact
forward evaluator, and three fixed seeds as the F16 ConPath matrix, while removing the correlated
local/global posterior structure (`local_kernel_size=1`, global factors disabled). It is a causal
control, not a recent-paper reproduction and not a final paper claim.

To use all available GPU capacity without the previous validation OOM, validation K=32 is now
processed as four sequential K=8 chunks (`--validation-sample-chunk 8`) and accumulated with the
same total sample count. The three seeds are running concurrently with atomic latest/best
checkpoints under `results/p1_flatlands_conpath_independent_f16/`; interrupted checkpoints are
resume-safe. The three exact validation manifests and reports are now complete (4,224 rows per
seed):

| Seed | Best epoch | Selection Brier | Exact scene-weighted Brier | NLL | ECE | False-safe @0.8 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260831 | 4 | 0.08522 | 0.11490 | 0.95813 | 0.10288 | 0.07114 |
| 20260901 | 10 | 0.08047 | 0.10884 | 0.90150 | 0.08816 | 0.07314 |
| 20260902 | 10 | 0.08627 | 0.11267 | 0.95242 | 0.09282 | 0.07629 |
| **mean ± sample SD** | — | — | **0.11214 ± 0.00306** | **0.93735 ± 0.03118** | **0.09462 ± 0.00752** | **0.07352 ± 0.00260** |

The matched F16 ConPath row remains lower at `0.09607 ± 0.01222` Brier, so the independent
decoder is a causal control rather than evidence that local correlation is unnecessary. The run
manifests record peak memory and wall time; all three use the same test-locked validation contract.
This control currently exports event predictions only; a separate deterministic mean-map control is
still required before any map-quality claim is made.

## Exact next actions

1. Add the deterministic mean-map control;
   report map quality and event quality together.
2. Port and evaluate recent strong uncertainty/completion methods under the same FlatLands contract;
   keep incompatible cross-task 3-D metrics as references only.
3. Implement batched exact-forward connectivity (merge-tree/MST or a validated exact-forward /
   soft-backward operator); compare against the NumPy oracle in error, latency, memory, and query
   scaling.
4. Add source/radius reliability, threshold false-safe curves, bootstrap intervals, failure cases,
   symmetry/radius-monotonicity checks, and an explicit ARKitScenes saturation analysis.
5. Audit and freeze one second domain (prefer ORFD semantics or UnScenes3D support surfaces), with
   scene/site/sequence-held-out split and no adjacent-frame leakage.
6. Only after the above, unlock the test split once, regenerate final JSON/CSV/SVG and qualitative
   figures, freeze environment/data/checkpoint hashes, and update the website.
7. Draft and internally review the ICRA/IROS paper: problem/claims, method, related work, main table/
   figures, limitations, appendix, anonymization, and venue-format/compliance checks.

## Recovery commands

```bash
cd /home/hairo/pathrel_transfer/pathrel_pro6000
git status --short --branch
PYTHONPATH=src .venv/bin/python scripts/verify_recovery_state.py --quick
# Full SHA verification (rehashes the 2.055 GB archive):
PYTHONPATH=src .venv/bin/python scripts/verify_recovery_state.py
sed -n '1,240p' CONTINUATION.md
sed -n '1,240p' results/p0_neural_cuda_contextplane_v4/report.json
sed -n '1,240p' results/p0_neural_cuda_contextplane_seed20260828_v4/report.json
sed -n '1,240p' results/p0_neural_cuda_contextplane_noreach_v4/report.json
sed -n '1,240p' results/p1_flatlands_query_audit_bounded/report.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/diagnose_p0_checkpoint.py \
  results/p0_neural_cuda_tuned01/checkpoint.pt --samples 128 --skip-events
./scripts/download_flatlands.sh --check
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_queries.py \
  --output-dir results/p1_flatlands_query_audit_bounded --overwrite
PYTHONPATH=src .venv/bin/python scripts/build_demo_site.py
```

`results/` and checkpoints are intentionally ignored. Any result used to make a research decision
must therefore be summarized here (and in the appropriate tracked research document) before a
session ends.
