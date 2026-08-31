# ConPath continuation state

This file is the durable hand-off for interrupted Codex sessions. Update it after every material
diagnostic, code change, or experiment; do not rely on chat history or ignored `results/` alone.

## Recovery snapshot

- Updated: 2026-08-30 (America/New_York)
- Repository: `/home/hairo/pathrel_transfer/pathrel_pro6000`
- Last durable hand-off commit: `74bb94b` (local `main`; not yet pushed in this continuation)
- Scientific gate: **NO-GO** for P1/public-data claims
- Active task: fix the trained neural P0 joint-event posterior under the existing held-out-template
  protocol; do not expand to a new public benchmark until it passes.

## Last completed work

The tracked repository already contains the TUM RGB-D geometric pilot, model hand-off smoke,
academic project page, exact NumPy merge-tree reference, and synthetic P0 baselines. The oracle
correlated proxy passes the death test, but the neural model does not.

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
loss history exactly `[1, 2, 3, 4]`. All tests currently pass: `Ran 30 tests ... OK`, `skipped=0`.

A 40-step 12x12 CPU regression completed without NaNs (event Brier `0.128662`, empirical full-map
Brier `0.020068`). It did not separate the two context priors and does not pass its reduced-protocol
direct-query baseline. This reduced setup has no positive radius-2 test events and is only a code
regression, not a scientific result.

## Exact next actions

1. Commit the audited implementation and run the unchanged 12/4-template CUDA protocol on the RTX
   PRO 6000 in a GPU-visible host execution context.
2. Monitor `results/p0_neural_cuda_surrogate_v1/progress.jsonl`; if interrupted, resume its
   `latest.pt` with the same immutable protocol and a larger/equal `--steps` value.
3. Compare neural event Brier/ECE and empirical hard-map Brier against the fixed death-test gates.
4. Update this file and `CONFERENCE_READINESS.md` with the exact command and result; remain NO-GO if
   independent or direct-query still wins.

## Recovery commands

```bash
cd /home/hairo/pathrel_transfer/pathrel_pro6000
git status --short --branch
sed -n '1,240p' CONTINUATION.md
sed -n '1,240p' results/p0_neural_cuda_tuned01/report.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/diagnose_p0_checkpoint.py \
  results/p0_neural_cuda_tuned01/checkpoint.pt --samples 128 --skip-events
```

`results/` and checkpoints are intentionally ignored. Any result used to make a research decision
must therefore be summarized here (and in the appropriate tracked research document) before a
session ends.
