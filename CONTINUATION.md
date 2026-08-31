# ConPath continuation state

This file is the durable hand-off for interrupted Codex sessions. Update it after every material
diagnostic, code change, or experiment; do not rely on chat history or ignored `results/` alone.

## Recovery snapshot

- Updated: 2026-08-30 (America/New_York)
- Repository: `/home/hairo/pathrel_transfer/pathrel_pro6000`
- Baseline commit: `afa5eb8` (`main`, synchronized with `origin/main` before this file was added)
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

## Current diagnosis

Map marginals are accurate while sampled path events are strongly over-open, especially at radius
zero. The current audit is checking three likely causes before changing the method:

1. reported soft posterior marginals versus empirical hard-map frequencies;
2. learned global factor strength/coherence versus local and categorical per-cell noise;
3. whether the straight-through iterative reachability gradient provides a useful topology signal.

The tuned checkpoint has non-zero factor weights, so the next diagnostic must measure their sampled
joint effect rather than merely checking that gradients exist.

## Exact next actions

1. Add checkpoint diagnostics for empirical map marginals, factor/local-scale statistics, coherent
   doorway frequency, context conditioning, and event metrics by radius.
2. Add interruption-safe JSONL progress logging, atomic latest checkpoints, and resume support to
   `scripts/train_p0_neural.py`.
3. Implement the smallest evidence-backed training/parameterization fix and add unit tests.
4. Run all unit tests with `skipped=0`, then a reduced CPU regression.
5. Run the unchanged 12/4-template CUDA protocol on the RTX PRO 6000 and compare against the fixed
   death-test thresholds. Update this file with the command, output directory, commit, and metrics.

## Recovery commands

```bash
cd /home/hairo/pathrel_transfer/pathrel_pro6000
git status --short --branch
sed -n '1,240p' CONTINUATION.md
sed -n '1,240p' results/p0_neural_cuda_tuned01/report.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`results/` and checkpoints are intentionally ignored. Any result used to make a research decision
must therefore be summarized here (and in the appropriate tracked research document) before a
session ends.
