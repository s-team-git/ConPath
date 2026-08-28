# ConPath P0 synthetic death test (2026-08-27)

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

The overall project gate remains **NO-GO for public-data expansion** until the same margin is
reproduced by a trained neural PathRel checkpoint. CUDA is unavailable in this environment, so the
proxy pass must not be presented as a learned-model or ICRA/IROS result.

The FlatLands baseline is recorded as `not_run` because this transfer package contains no FlatLands
data. The `random_completion` row is a transparent uniform-unknown completion surrogate; it is not
an implementation of SCOPE. No real-data, SCOPE, diffusion-completion, or navigation claim is made.

## Exact-forward prototype

`src/pathrel/labels.py::merge_tree_bottleneck_scores` is an exact Kruskal merge-tree/LCA reference
for many queries on one map. On a 64x64 seeded random map it matched exhaustive bottleneck search
to zero error and measured 3.0x/22.7x/172.7x speedups for 8/64/512 queries in the latest recorded
run. It is a NumPy correctness contract, not yet a differentiable CUDA kernel.
