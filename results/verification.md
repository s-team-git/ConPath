# Verification log (updated 2026-08-30)

The project is the **ConPath** Git repository. The current P0 implementation and interruption-safe
handoff are committed on local `main`; ignored result directories are mirrored in tracked
`CONTINUATION.md`. The configured origin remains `git@github.com:s-team-git/ConPath.git`, but the
new continuation commits have not been pushed in this session.

## Environment

Command: `nvidia-smi`

Result: `nvidia-smi` reports `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, driver
`580.173.02`, CUDA `13.0`, and `97887 MiB` total memory. The GPU is visible in the hardware-enabled
execution channel.

Command: `PYTHONPATH=src .venv/bin/python scripts/check_environment.py --json`

Result: Python 3.11.15; Torch `2.13.0+cu130`; built CUDA `13.0`; `cuda_available=true`; device
count `1`; compute capability `12.0`; BF16 supported. The environment was created in a fresh
project-local `.venv`; no old environment was copied or modified. `scripts/check_environment.py
--json` records the kernel driver/GPU report separately from `nvidia-smi`.

## Tests and smoke

Command: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`

Latest result: `Ran 35 tests in 15.928s ... OK` with `skipped=0`.

Command: `PYTHONPATH=src .venv/bin/python scripts/smoke_forward.py`

Result: forward smoke passed with `[2,4,2,24,24]` map samples and `[2,2,3]` reachability output;
observed cells are now deterministic in posterior samples.

Command: `PYTHONPATH=src .venv/bin/python scripts/train_synthetic.py --config configs/synthetic.json --device cpu --steps 1 --validation-size 2 --validation-samples 2 --checkpoint checkpoints/cpu_smoke.pt`

Result: CPU forward/backward/optimizer/checkpoint smoke passed; validation Brier `0.3151041567`.

Command: `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src bash scripts/run_gpu_smoke.sh`

Historical result: complete GPU smoke passed: environment check, the then-current 27 tests
(`skipped=0`), forward smoke, and one CUDA optimizer/checkpoint step. The CUDA smoke reported
validation Brier `0.33463544`. The expanded 35-test suite is recorded by the latest CPU run above
and the full CUDA P0 runs below.

Command: `PYTHONPATH=src .venv/bin/python scripts/train_p0_neural.py --device cpu --steps 1 --batch-size 2 --train-templates 2 --test-templates 1 --worlds-per-template 4 --train-samples 2 --validation-samples 2 --output-dir results/p0_neural_cpu_smoke`

Result: neural P0 forward/backward/evaluation/checkpoint path passed on CPU; one-step event Brier
`0.48958333`. This is a code-path smoke only.

The staged trainer was also exercised on a 12x12, four-step CPU smoke (`--warmup-steps 2`); it
completed both warmup and joint stages and reported event Brier `0.13932291`.

Command: `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python scripts/train_p0_neural.py --device cuda --output-dir results/p0_neural_cuda`

Result: the full 120-step CUDA run completed and saved a checkpoint, but it did **not** pass the
P0 margin: event Brier `0.243578` (ECE `0.214613`) versus independent `0.183172` and direct-query
`0.169888`. Map marginal Brier was `0.006844`; radius-wise event means were `[0.999919, 0.465739,
0.107300]` versus targets `[0.546875, 0.351563, 0.179688]`. This is a reproducible neural
failure baseline, not a paper result.

### Corrected interruption-safe neural P0

The corrected trainer used 120 steps, 24 warm-up steps, grouped 24-world empirical targets,
8 training samples, 128 validation samples, visible context plane, coordinate/global encoder,
categorical noise scale 0.25, and 10-step atomic checkpoints. Each run wrote `progress.jsonl`,
`latest.pt`, a final checkpoint, and `report.json` in its independent output directory.

| Output / change | Event Brier | NLL | ECE | Hard-map Brier | Context-gap ratio | Gate |
|---|---:|---:|---:|---:|---:|---|
| `p0_neural_cuda_contextplane_v4`, seed 20260827 | 0.116377 | 0.353872 | 0.078647 | 0.003383 | 0.5735 | PASS |
| `p0_neural_cuda_contextplane_seed20260828_v4`, seed 20260828 | 0.111621 | 0.346012 | 0.071852 | 0.002892 | 0.6957 | PASS |
| `p0_neural_cuda_contextplane_noreach_v4`, reach weight 0 | 0.191368 | 0.795186 | 0.193570 | 0.003098 | 0.2383 | FAIL |

The two full seeds took 87.7 s and 86.5 s; the matched ablation took 87.7 s. Each reported peak CUDA
memory of 5.79 GB on the RTX PRO 6000. The no-reach run preserves strong map marginals but fails the
event and context gates, while both full seeds beat the fixed independent/direct-query Brier
comparators. This is a synthetic P0 pass and only authorizes P1 data audit.

## P0 and exact-forward artifacts

Command: `PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test`

Result: oracle correlated-posterior proxy death test passed against independent-cell and direct-
query baselines (event Brier `0.10237` vs `0.18317` and `0.16989`; ECE `0.03248`); see
`P0_DEATH_TEST.md` and `results/p0_death_test/report.json`. This row remains an oracle proxy; the
separate trained-neural evidence is recorded above.

Command: `PYTHONPATH=src .venv/bin/python scripts/benchmark_merge_tree.py --output results/merge_tree_benchmark.json`

Latest regression result: exact error `0.0`; 64x64 speedups `3.09x/25.50x/161.19x` for
`8/64/512` queries. Timing is diagnostic; zero error is the correctness contract.

## P1 FlatLands archive audit (2026-08-30)

Command: `./scripts/download_flatlands.sh --download`, followed by
`./scripts/download_flatlands.sh --check`.

Result: the official `FlatLands_final_dataset.zip` was downloaded without extraction. Its exact size
is `2,054,773,316` bytes and SHA-256 is
`e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`, matching the frozen official
release contract.

Command: `PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_archive.py --metadata-limit 0
--output-dir results/p1_flatlands_archive_audit_full`.

Result: archive structure and metadata integrity pass. All 270,575 packets are complete and all
metadata parse; official counts match; unsafe paths, duplicates, symlinks, encrypted members,
malformed metadata, and missing identities are zero. The ConPath scene-split gate fails: shared
`(source, scene_id)` counts are 12,873 for train/validation, 8,406 for train/test, and 6,800 for
validation/test. Thus the official observation split cannot support a cross-scene claim; no
extraction or public-data training was authorized.

## Real-data pilot (2026-08-28)

Command: `/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --frames 48 --queries 18 --samples 48 --publish-site`

Result: 48 synchronised TUM RGB-D Freiburg1/desk frames were lifted with the official intrinsics
and MoCap poses into a 79x105 world-frame raster. The temporal split uses 36 observed-prefix
frames and 12 future query frames, with 18 query pairs and footprint radii 0/1/2. The geometric
reference-map audit reports Brier `0.2407` for the observed-prefix baseline, `0.1908` for
independent-cell completion, and `0.1831` for the correlated-temporal completion. These are pilot
metrics on inferred geometric labels, not traversability or collision benchmark results.

Command: `PYTHONPATH=src .venv/bin/python scripts/run_tum_rgbd_model_smoke.py --device cpu --samples 4 --max-steps 96`

Result: the randomly initialised `PathRelNet` consumed the real-data BEV hand-off with input shape
`[1,3,40,56]`, produced `[1,4,40,56]` map samples and `[1,18,3]` reachability values in `0.50 s`.
This is an end-to-end integration smoke only; no trained real-data checkpoint is claimed.

The ordinary filesystem sandbox does not expose `/dev/nvidia*`, while the approved hardware-enabled
execution channel exposes the RTX PRO 6000. The three corrected P0 runs above were executed through
that channel; CPU-only invocations must not reinterpret their own `cuda_available=false` as evidence
that those reports were fabricated or rerun on CPU.
