#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

python scripts/evaluate_p0.py --output-dir results/p0_death_test
python scripts/benchmark_merge_tree.py --output results/merge_tree_benchmark.json
