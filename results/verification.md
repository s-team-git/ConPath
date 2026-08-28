# Verification log (2026-08-28)

The project is now the **ConPath** Git repository. The local `main` branch is clean; the latest
verification commit and complete history are available in `git log` (scientific bootstrap:
`30f3b78`). The configured origin is `git@github.com:s-team-git/ConPath.git`; SSH authentication was
verified and the current `main` branch was pushed successfully.

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

Result: `Ran 27 tests in 0.694s ... OK` with `skipped=0`.

Command: `PYTHONPATH=src .venv/bin/python scripts/smoke_forward.py`

Result: forward smoke passed with `[2,4,2,24,24]` map samples and `[2,2,3]` reachability output;
observed cells are now deterministic in posterior samples.

Command: `PYTHONPATH=src .venv/bin/python scripts/train_synthetic.py --config configs/synthetic.json --device cpu --steps 1 --validation-size 2 --validation-samples 2 --checkpoint checkpoints/cpu_smoke.pt`

Result: CPU forward/backward/optimizer/checkpoint smoke passed; validation Brier `0.3151041567`.

Command: `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src bash scripts/run_gpu_smoke.sh`

Result: complete GPU smoke passed: environment check, all 27 tests (`skipped=0`), forward smoke,
and one CUDA optimizer/checkpoint step. The CUDA smoke reported validation Brier `0.33463544`.

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

## P0 and exact-forward artifacts

Command: `PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test`

Result: oracle correlated-posterior proxy death test passed against independent-cell and direct-
query baselines (event Brier `0.10237` vs `0.18317` and `0.16989`; ECE `0.03248`); see
`P0_DEATH_TEST.md` and `results/p0_death_test/report.json`. This is not a trained neural result and
does not authorize public-data claims.

Command: `PYTHONPATH=src .venv/bin/python scripts/benchmark_merge_tree.py --output results/merge_tree_benchmark.json`

Result: exact error `0.0`; 64x64 speedups `3.0x/22.7x/172.7x` for `8/64/512` queries in the latest run.

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

The current execution session has no `/dev/nvidia*` device nodes (`torch.cuda.is_available() = false`),
so the CUDA trainer is not run here. The earlier host verification above records the separate GPU
smoke; rerun it in a GPU-passthrough session before any neural paper experiment.
