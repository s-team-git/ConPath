#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/smoke_forward.py
python scripts/train_synthetic.py \
  --config configs/synthetic.json \
  --device cuda \
  --steps 1 --validation-size 2 --validation-samples 2 \
  --checkpoint checkpoints/pro6000_quick_smoke.pt
