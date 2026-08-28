# Verification log (2026-08-27)

The project is now the **ConPath** Git repository. The local `main` branch is clean; the latest
verification commit and complete history are available in `git log` (scientific bootstrap:
`30f3b78`). The configured origin is `https://github.com/s-team-git/ConPath.git`; replacing the
remote contents is pending GitHub authentication (the HTTPS remote requested credentials and the
available SSH key was rejected).

## Environment

Command: `nvidia-smi`

Result: `nvidia-smi` cannot communicate with a compute device in this session. The read-only kernel
report does show `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, driver `580.173.02`, and a
loaded NVIDIA module, but no compute device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, or
`/dev/nvidia-uvm`) are visible. This is consistent with a container/session launched without GPU
passthrough; it is not evidence that the host driver package is absent.

Command: `PYTHONPATH=src .venv/bin/python scripts/check_environment.py --json`

Result: Python 3.11.15; Torch 2.13.0+cu130; built CUDA 13.0; `cuda_available=false`; device count
0; NVML unavailable. The environment was created in a fresh project-local `.venv`; no old
environment was copied or modified. `scripts/check_environment.py --json` now records the kernel
driver/GPU report separately from `nvidia-smi` so a missing passthrough is not misdiagnosed as a
missing driver installation.

## Tests and smoke

Command: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`

Result: `Ran 27 tests in 0.694s ... OK` with `skipped=0`.

Command: `PYTHONPATH=src .venv/bin/python scripts/smoke_forward.py`

Result: forward smoke passed with `[2,4,2,24,24]` map samples and `[2,2,3]` reachability output;
observed cells are now deterministic in posterior samples.

Command: `PYTHONPATH=src .venv/bin/python scripts/train_synthetic.py --config configs/synthetic.json --device cpu --steps 1 --validation-size 2 --validation-samples 2 --checkpoint checkpoints/cpu_smoke.pt`

Result: CPU forward/backward/optimizer/checkpoint smoke passed; validation Brier `0.3151041567`.

Command: `PYTHONPATH=src .venv/bin/python scripts/train_synthetic.py --config configs/synthetic.json --device cuda --steps 1 --validation-size 2 --validation-samples 2`

Result: expected explicit failure: `CUDA was requested but torch.cuda.is_available() is false`.

The bundled `scripts/run_gpu_smoke.sh` reached the same result after completing all 27 tests and
the forward smoke; its exit code is 1 solely because the requested CUDA training step cannot start.

Command: `PYTHONPATH=src .venv/bin/python scripts/train_p0_neural.py --device cpu --steps 1 --batch-size 2 --train-templates 2 --test-templates 1 --worlds-per-template 4 --train-samples 2 --validation-samples 2 --output-dir results/p0_neural_cpu_smoke`

Result: neural P0 forward/backward/evaluation/checkpoint path passed on CPU; one-step event Brier
`0.48958333`. This is a code-path smoke only.

The staged trainer was also exercised on a 12x12, four-step CPU smoke (`--warmup-steps 2`); it
completed both warmup and joint stages and reported event Brier `0.13932291`.

## P0 and exact-forward artifacts

Command: `PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test`

Result: oracle correlated-posterior proxy death test passed against independent-cell and direct-
query baselines (event Brier `0.10237` vs `0.18317` and `0.16989`; ECE `0.03248`); see
`P0_DEATH_TEST.md` and `results/p0_death_test/report.json`. This is not a trained neural result and
does not authorize public-data claims.

Command: `PYTHONPATH=src .venv/bin/python scripts/benchmark_merge_tree.py --output results/merge_tree_benchmark.json`

Result: exact error `0.0`; 64x64 speedups `3.0x/22.7x/172.7x` for `8/64/512` queries in the latest run.
