# ConPath continuation state

This file is the durable hand-off for interrupted Codex sessions. Update it after every material
diagnostic, code change, or experiment; do not rely on chat history or ignored `results/` alone.

## Recovery snapshot

- Updated: 2026-08-30 (America/New_York)
- Repository: `/home/hairo/pathrel_transfer/pathrel_pro6000`
- Last durable implementation commit: `2de9103` (local `main`; not yet pushed in this continuation)
- Scientific gate: **P0 GO across two seeds; public-data/paper claims not yet established**
- Active task: acquire the frozen FlatLands archive with resume plus size/SHA verification, then
  audit it in place before extraction or public-data training.

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

## P1 pre-download audit

The workspace contains only the ignored 830 MB TUM RGB-D pilot data. It has no FlatLands, ORFD,
UnScenes3D, or WildOcc assets and no corresponding loader/split/query audit. Official-source review
selects FlatLands as the first P1 audit target because it provides aligned observed/full floor maps,
unobserved and valid masks, metric provenance, and official splits. ORFD is a secondary off-road
semantics audit; UnScenes3D/WildOcc remain P2.

`P1_DATA_AUDIT.md` freezes the FlatLands release as `2,054,773,316` bytes with SHA-256
`e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`, defines scene-disjointness,
mask/polarity, natural-query, 10%-15% event-balance, provenance/license, and strong-baseline gates,
and keeps the decision NO-GO for extraction/training. `scripts/download_flatlands.sh` supports a
resumable `.part`, verifies size and hash, atomically finalizes the archive, and never extracts it.
`bash -n` passes; `--check` correctly returns exit 2 while the archive is absent.

## Exact next actions

1. Commit `P1_DATA_AUDIT.md`, `scripts/download_flatlands.sh`, and this recovery update.
2. Run `./scripts/download_flatlands.sh --download`; after interruption, rerun the identical command
   to resume. Do not delete a mismatching file automatically.
3. Run `./scripts/download_flatlands.sh --check`, then inspect the ZIP structure/metadata without
   extracting it and implement a bounded deterministic query-balance audit.
4. Record archive integrity and audit results here before any extraction, loader training, or public
   claim. Remain NO-GO if scene leakage, semantics, licensing, or event balance fails.

## Recovery commands

```bash
cd /home/hairo/pathrel_transfer/pathrel_pro6000
git status --short --branch
sed -n '1,240p' CONTINUATION.md
sed -n '1,240p' results/p0_neural_cuda_contextplane_v4/report.json
sed -n '1,240p' results/p0_neural_cuda_contextplane_seed20260828_v4/report.json
sed -n '1,240p' results/p0_neural_cuda_contextplane_noreach_v4/report.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/diagnose_p0_checkpoint.py \
  results/p0_neural_cuda_tuned01/checkpoint.pt --samples 128 --skip-events
./scripts/download_flatlands.sh --check
```

`results/` and checkpoints are intentionally ignored. Any result used to make a research decision
must therefore be summarized here (and in the appropriate tracked research document) before a
session ends.
