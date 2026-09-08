# Clean FlatLands training ablations

Frozen on 2026-09-07 before inspecting ablation outcomes. All experiments use
validation; physical tests stay locked.

## Questions and controls

| Model | Event training weight | Decoder global factors | Local kernel | Seeds |
|---|---:|---|---:|---|
| Existing clean full model | 2.0 | enabled | 5 | 20260831 / 20260901 / 20260902 |
| no_event | 0.0 | enabled | 5 | same three |
| no_global | 2.0 | disabled | 5 | same three |

The full model is reused from `results/p1_flatlands_conpath_k128_support_clamped_v1`.
Each ablation starts from its seed's random initialization and changes only its
named flag. It does not fine-tune a full-model checkpoint.

`no_event` removes event supervision from the training objective while retaining
the same validation event-based checkpoint selection. `no_global` disables only
the decoder's low-rank global random factors: encoder context, coordinate
channels, local correlation and the variogram loss remain enabled.

## Fixed contract

- Trainer/model/data implementation hashes must match the clean reference.
- Frozen scene-disjoint, non-official provenance split; 160 train / 160 validation
  packets. Final validation has 4,224 events from 142 contributing scenes.
- F=16, latent dimension 4, 120,108 parameters; batch size 1; AdamW learning rate
  0.0003, weight decay 0.0001, gradient clip 5; map weight 1 and variogram weight 0.1.
- Training K=8; validation K=128/chunk=8; bounded training/selection propagation
  256; up to eight fixed target-blind selection queries per packet.
- Maximum 40 epochs; patience 8; improvement threshold 0.00001. No seed replacement,
  query revision, validation subset, alternate stopping rule or parameter search.
- Original exact disk-clearance/merge-tree final evaluator; invalid support blocked.

## Analysis fixed before results

Primary: scene-weighted Brier, each ablation minus the full model, with paired
whole-scene bootstrap (2,000 draws, bootstrap seed 20260907) within each training
seed. Report all seeds, aggregate mean and sample SD, including reversals and
intervals containing zero. Do not promote only a favorable seed.

Secondary: NLL, ECE, source/radius strata, and false-safe risk at equal 30%
scene-weighted coverage. Ties receive label-independent fractional acceptance;
the coverage boundary is recomputed in each bootstrap draw. These exploratory
reused-validation comparisons are not adjusted confirmatory tests or deployment
thresholds. Training ablations and the earlier fixed-marginal shuffle answer
different questions and must both remain in the evidence.

## Execution and recovery

Current status: resumed again at the user's request on 2026-09-07 at 23:59 EDT
(2026-09-08 03:59 UTC). Three final-evaluation workers are active. no_event seeds 20260831/20260901/20260902
completed 12/9/17 epochs and reached patience=8; their selected best epochs are 4/1/9.
Six latest/best checkpoints passed recovery checks and were backed up under
`results/training_pause_20260908T034736Z/`. no_global is queued and starts automatically
as the current workers complete and pass their audits. The new explicit resume request
is recorded in `results/training_resume_20260908_finalization/authorization.json`.
checkpoint_action is `finalize`: exact validation has restarted with the selected
weights and saved RNG, without another training epoch. The interrupted evaluation
wrote no prediction rows; its partial work is being recomputed. Do not start duplicate workers.

Historical resumption: resumed at the user's explicit request on 2026-09-07 at 21:42 EDT
(2026-09-08 01:42 UTC). The earlier pause at 01:50 EDT is retained in the audit history.
Three no_event runs resume from complete-epoch `latest.pt` at epoch 4 and rerun
epoch 5; three no_global runs are queued. The separately retained `interrupted.pt`
is a partial-epoch artifact, not the resume source. All nine pause-time checkpoints
were backed up under `results/training_resume_20260907/paused_checkpoints/` before
resumption. The original configuration and training implementation are unchanged.
Authorization and launch receipts are in `results/training_resume_20260907/`.

```bash
PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/run_flatlands_clean_ablations.py --prepare-only
PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/run_flatlands_clean_ablations.py
cat results/paper_clean_ablation_matrix_v1/progress.json
```

The supervisor runs at most three workers and queues all six runs. An OS lock and
worker checks prevent duplicate jobs. Do not launch a second supervisor while the
recorded one is active. Progress, PIDs, logs, commands and the immutable matrix
manifest are in `results/paper_clean_ablation_matrix_v1/`.

Completed epochs save `latest.pt` and selected `best.pt`. Recovery checks every
configuration field before resuming model, optimizer and RNG state. If training
already reached its stop, recovery runs only final evaluation, without another
epoch. Ambiguous directories without checkpoints are refused.

After all six runs pass configuration, checkpoint, support, query/hash and metric
replay checks, the supervisor generates `analysis/report.json`,
`analysis/PAPER_ABLATIONS.md`, and a standalone SVG/PDF figure pair. These are
candidates for manuscript/site review; incomplete runs never enter the published
table. Historical full-model runs took about 3.6–4.5 hours per seed with concurrent
workers; this is a planning estimate, not a completion deadline.
