# ConPath P0 synthetic death test (updated 2026-08-30)

This is an audit artifact, not a conference result. The experiment is a 24x24 binary latent-world
contract test with two visible context families. Their hidden doorway priors are estimated at
approximately 0.2 and 0.8 from training worlds. Four scene templates are held out from training;
there is no adjacent-frame split or test-template leakage. Every event label is produced by the
exact NumPy clearance/maximum-bottleneck oracle.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test
```

The default run uses seed `20260827`, 12 training templates, 4 test templates, 24 worlds per
template, and 64 posterior samples. The generated `results/p0_death_test/` directory contains the
full JSON/CSV/SVG artifacts: `comparison.svg` is the grouped Brier/NLL/ECE comparison and
`reliability.svg` is the reliability diagram.

| Method | Reachability Brier | NLL | ECE | False-safe @0.8 |
|---|---:|---:|---:|---:|
| constant query/radius | 0.2096 | 0.6083 | 0.0616 | — |
| independent Bernoulli | 0.1832 | 1.0316 | 0.1762 | 0.2500 |
| direct query MLP | 0.1699 | 0.5224 | 0.0891 | — |
| edge-connectivity calibrated | 0.1382 | 0.4494 | 0.1413 | — |
| random completion | 0.3115 | 2.2621 | 0.3119 | 0.4531 |
| deterministic threshold | 0.1458 | 2.0148 | 0.1458 | 0.1797 |
| correlated decoder, no reachability loss | 0.1663 | 0.4613 | 0.0219 | — |
| correlated event posterior (oracle proxy) | 0.1024 | 0.3250 | 0.0325 | 0.1474 |

The oracle-proxy death test is **PASS**: the correlated posterior beats independent cells and the
direct-query MLP on event Brier, improves ECE over both, and lowers false-safe rate versus independent
cells, while remaining within 0.02 map-marginal Brier of the independent sampler. The edge-connectivity and deterministic rows are retained as
stronger diagnostics; their deterministic/score-based outputs do not substitute for a calibrated
stochastic map because
their NLL/ECE and map uncertainty are materially different.

The proxy pass was the historical hypothesis check; by itself it did not authorize public-data
expansion. The trained-model audit below now closes the synthetic P0 gate, while retaining a strict
boundary against public-data or ICRA/IROS claims.

## Historical first CUDA checkpoint (failure baseline)

The first full 120-step CUDA run is recorded at
`results/p0_neural_cuda/report.json`. It completed without numerical or memory errors, but failed
the death-test comparison: event Brier `0.2436`, ECE `0.2146`, versus independent Brier `0.1832`
and direct-query Brier `0.1699`. The map marginal Brier was good (`0.0068`), while the radius-zero
event mean was `0.9999` for a target mean of `0.5469`. Inspection of the checkpoint shows that
unknown doorway columns are sampled as mixed per-cell noise rather than coherent open/closed doors.
This is retained as a failure baseline and motivates a posterior-sampler correction before any
public-data experiment.

## Corrected neural v4 (synthetic P0 PASS)

The corrected protocol groups all 24 repeated worlds for each `(template, context)` observation,
exposes the same visible context bit used by the baselines as a broadcast input plane, and trains
with exact hard reachability events in the forward pass plus a relaxed bottleneck surrogate only in
the backward pass. Both full runs use 120 steps (24 warm-up), 8 training samples, 128 validation
samples, reachability weight 2.0, and the same held-out dataset split. The second seed changes model,
batch, and sampling RNG only.

| Neural run | Event Brier | NLL | ECE | False-safe @0.8 | Hard-map Brier | Context-gap ratio | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| full, seed 20260827 | 0.1164 | 0.3539 | 0.0786 | 0.1250 | 0.00338 | 0.5735 | PASS |
| full, seed 20260828 | 0.1116 | 0.3460 | 0.0719 | 0.1125 | 0.00289 | 0.6957 | PASS |
| no reachability loss, seed 20260827 | 0.1914 | 0.7952 | 0.1936 | 0.4258 | 0.00310 | 0.2383 | FAIL |

Each full seed beats independent Bernoulli (`0.1832`) and direct query (`0.1699`) on event Brier,
keeps ECE within 0.02 of the better comparator, keeps empirical hard-map Brier within 0.02 of the
independent sampler, and recovers at least half of the held-out radius-zero context gap. The matched
no-reach run retains slightly better map marginals than the first full seed but loses `0.0750` event
Brier and fails context conditioning. This isolates the gain to learned joint/event structure rather
than better per-cell occupancy.

The exact reports and interruption-safe checkpoints are:

- `results/p0_neural_cuda_contextplane_v4/`;
- `results/p0_neural_cuda_contextplane_seed20260828_v4/`;
- `results/p0_neural_cuda_contextplane_noreach_v4/`.

Generated results are intentionally ignored by Git; their metrics, protocol, and recovery commands
are durably mirrored in `CONTINUATION.md`. The trainer writes `progress.jsonl`, atomically refreshes
`latest.pt` every 10 steps, and saves `interrupted.pt` on a caught interruption.

The learned synthetic P0 decision is therefore **PASS / GO to P1 data audit**, not GO to a paper
claim. This is two optimization seeds on one fixed synthetic split, not cross-dataset replication.
Doorway samples still fragment about 13.7%-16.1% in the full runs versus 0% in the target, and the
first seed overestimates context-0 open probability. P1 must audit natural query balance, label
semantics, and completion baselines before any training or public benchmark claim.

The FlatLands baseline is recorded as `not_run` because this transfer package contains no FlatLands
data. The `random_completion` row is a transparent uniform-unknown completion surrogate; it is not
an implementation of SCOPE. No real-data, SCOPE, diffusion-completion, or navigation claim is made.

## Exact-forward prototype

`src/pathrel/labels.py::merge_tree_bottleneck_scores` is an exact Kruskal merge-tree/LCA reference
for many queries on one map. On a 64x64 seeded random map it matched exhaustive bottleneck search
to zero error and measured 3.0x/22.7x/172.7x speedups for 8/64/512 queries in the latest recorded
run. It is a NumPy correctness contract, not yet a differentiable CUDA kernel.
