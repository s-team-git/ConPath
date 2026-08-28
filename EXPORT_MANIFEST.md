# Export manifest

This is the clean hand-off tree for the ConPath Pro 6000 machine.

Included:

- research documentation: `README.md`, `ROADMAP_ZH.md`, `ALGORITHM.md`, `CONFERENCE_READINESS.md`;
- P0 audit and paper-gate notes: `P0_DEATH_TEST.md`, `scripts/evaluate_p0.py`,
  `scripts/train_p0_neural.py`, and `scripts/benchmark_merge_tree.py`;
- transfer and GPU setup instructions: `TRANSFER_README.md`, `GPU_SETUP.md`;
- the remote-agent hand-off prompt: `PROMPT_FOR_REMOTE_CODEX.md`;
- Python package metadata and dependency hints;
- all files under `src/pathrel/`, `scripts/`, `tests/`, `configs/`, `data/`, and `results/`.

Intentionally excluded:

- `.venv/` and `.uv-cache/` from the development machine;
- `__pycache__/` and `*.pyc`;
- all checkpoints, including the obsolete `two_step_smoke.pt`;
- raw/processed datasets;
- the repository's `vi/` and ROS runtime.

The package is currently a synthetic contract prototype. A successful smoke run is not a paper
result; the remote agent must complete the P0 baseline/death-test protocol before claiming a
research contribution.
