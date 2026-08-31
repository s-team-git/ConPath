# ConPath P1 FlatLands baseline protocol v1

Status: **frozen before any learned FlatLands validation/test result**

Frozen: 2026-08-31 (America/New_York)

This document fixes the first public-data evaluation contract. Changing a rule below after seeing
validation results requires a new protocol version and an explicit reason; the test split remains
locked until the selected configurations and seed aggregation are frozen.

## Benchmark identity and split

- Archive: `FlatLands_final_dataset.zip`, 2,054,773,316 bytes, SHA-256
  `e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`.
- Selection: 512 rows, SHA-256
  `4e7ae4c992cf943ab81618e3826c4748fcaaa97c3c4d7cb187518ee3fe6a9409`.
- Queries: 18,432 rows, SHA-256
  `33e7f8a0343269b0dde47b428b3be622c80effdb0f80ae34b352ca282018d60d`.
- Split key: only `provenance.original_split`; the physical ZIP directory is never an evaluation
  split. Train/validation/test contain 160/160/192 selected observations.
- ScanNet++ occurs only in provenance-test and is reported as OOD, never used for fitting or
  validation selection.

One report evaluates exactly one label-free prediction CSV. Its only columns are
`global_id,candidate_index,radius_cells,probability`. The evaluator joins targets after checking
exact key coverage. Prediction files containing target/label/reachable/floor-map or archive-split
columns are rejected.

## Input and target contract

The canonical three-channel model input is:

1. epistemic-valid observed floor;
2. epistemic-valid observed blocked;
3. epistemic-valid unknown.

The channels are mutually exclusive; invalid support is all-zero. Observed blocked is reconstructed
without `floor_map` as `epistemic_mask & ~unobserved & ~observed_floor`. Completion loss is evaluated
only on `unobserved & epistemic_mask`. Queries and coordinates are already frozen before targets are
read.

## Evaluation and statistical unit

The primary aggregation gives every `(source_dataset, scene_id)` equal mass, then gives every
event within that scene equal mass. Query-weighted values are secondary diagnostics. Every method
reports:

- Brier score, Bernoulli NLL, 10-bin ECE;
- false-safe rate conditional on `q >= 0.8`, plus high-confidence coverage;
- positive prevalence and mean prediction;
- overall, source, radius, and source-by-radius scopes;
- 2,000 deterministic scene-cluster bootstrap resamples with 95% intervals;
- probability monotonicity violations, where `q(s,g,r)` increases with footprint radius;
- wall time, peak GPU memory when applicable, prediction sample count, and checkpoint/config hash.

Validation is available during development. Test labels are not used until all baseline
architectures, checkpoint-selection rules, sample counts, and ConPath ablations are frozen.

## Fixed controls and baselines

### Radius-prior control

For each radius, predict the provenance-train event prevalence. This has no image input and is a
protocol/control row, not a strong baseline.

### Deterministic completion

Train the same marginal completion network used by the independent-cell baseline, then threshold
its hidden-cell free probability at 0.5. Clamp observed-free/blocked evidence exactly and mark
invalid support blocked for geometry. Exact disk-footprint connectivity produces binary event
predictions. No event loss or event-label threshold tuning is allowed.

### Independent-cell completion

Use `TinyBEVUNet` with 16 base feature channels and a one-logit free/blocked head. Train only
hidden valid cells with Bernoulli NLL; checkpoint selection uses validation hidden-cell NLL, not
event labels. Clamp observed evidence and independently sample hidden cells from the predicted
marginals. The pilot uses `K=32`; final tables use `K=128` and report K=32/64/128 convergence with
fixed per-observation RNG keys. Event probabilities are empirical exact-connectivity frequencies.

### Direct-query predictor

Use the same 16-channel `TinyBEVUNet` family on the same three input channels. The event head receives
start/goal feature samples, global pooled BEV features, normalized start/goal coordinates, metric
distance/angle, and footprint radius. It is trained directly with Bernoulli NLL on provenance-train
events; checkpoint selection uses primary validation scene-weighted Brier. It does not receive
`floor_map`, source identity, scene identity, or labels from another query.

## Training and seed policy

- Optimizer: AdamW; initial learning rate `3e-4`; weight decay `1e-4`.
- Batch size: four scenes; no geometric augmentation in v1. Direct-query training runs at most 120
  epochs with patience 20; marginal completion runs at most 80 epochs with patience 12. Both use
  minimum validation improvement `1e-5` and save/restore the selected epoch.
- Seeds: `20260831`, `20260901`, `20260902` for every learned method in final tables.
- Batch construction is scene-based. All retained query-radius events for an observation remain in
  the same batch item and split.
- Checkpoints are selected independently per seed using the rule fixed above. The test split is run
  only from selected validation checkpoints.
- ConPath and ablations later reuse the same data, three-channel input, query rows, evaluation code,
  seeds, and encoder capacity unless a capacity-matched comparison is explicitly reported.

The historical CUDA pilot was launched with `feature_channels=8` for a propagation/debug diagnosis;
its 30,428-parameter checkpoint is not a final baseline comparison. Capacity-matched paper runs use
`feature_channels=16` (120,108 ConPath parameters versus 119,921 independent completion and 127,905
direct-query parameters). The recent-method bridge and its same-contract/reference-only split are
recorded in `RECENT_BASELINES.md`.

## Go/no-go before ConPath training

The baseline stage is complete only when all three baseline prediction files pass exact-coverage
evaluation on validation, the completion model has finite hidden-region map metrics, the direct
query model beats or matches the radius-prior Brier in the primary aggregation, and runtime plus
monotonicity diagnostics are recorded. Failure triggers a baseline/data diagnosis, not immediate
large ConPath training.

This protocol does not turn the bounded audit, control row, or synthetic P0 result into a public-data
model result. Paper-grade evidence still requires multi-seed ConPath results, ablations, final locked
test evaluation, false-safe analysis, scalable connectivity, and external-domain validation.
