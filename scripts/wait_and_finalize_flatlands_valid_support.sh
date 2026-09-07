#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

render_python="${PATHREL_RENDER_PYTHON:-$(command -v python)}"
python_bin="$render_python"
correlated_root="results/p1_flatlands_conpath_k128_support_clamped_v1"
independent_root="results/p1_flatlands_independent_k128_support_clamped_v1"
comparison_root="results/p1_flatlands_support_clamped_clean_k128_comparison"
seeds=(20260831 20260901 20260902)

PYTHONPATH=src "$render_python" - <<'PY'
import torch
from PIL import Image  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("clean candidate renderer requires CUDA")
PY

required_runs=()
for seed in "${seeds[@]}"; do
  required_runs+=(
    "$correlated_root/seed${seed}_conpath/run.json"
    "$independent_root/seed${seed}_independent/run.json"
  )
done

idle_checks=0
while true; do
  missing=0
  for run_path in "${required_runs[@]}"; do
    if [[ ! -s "$run_path" ]]; then
      missing=$((missing + 1))
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  printf '%s waiting for %s/6 clean run reports\n' "$(date --iso-8601=seconds)" "$((6 - missing))"
  if pgrep -f 'scripts/train_flatlands_conpath.py --output-dir results/p1_flatlands_.*support_clamped_v1' >/dev/null; then
    idle_checks=0
  else
    idle_checks=$((idle_checks + 1))
    if [[ "$idle_checks" -ge 3 ]]; then
      echo "No matching trainer has been active for three checks; refusing finalization." >&2
      exit 1
    fi
  fi
  sleep 60
done

echo "All six clean reports exist; starting strict audits."
PYTHONPATH=src "$python_bin" scripts/audit_flatlands_conpath_k128.py \
  --root "$correlated_root" \
  --run-suffix conpath \
  --decoder-variant correlated \
  --method conpath_valid_support_k128_audit
PYTHONPATH=src "$python_bin" scripts/audit_flatlands_conpath_k128.py \
  --root "$independent_root" \
  --run-suffix independent \
  --decoder-variant independent \
  --method independent_valid_support_k128_audit

PYTHONPATH=src "$python_bin" scripts/build_flatlands_k128_snapshot.py \
  --root "$correlated_root" \
  --run-suffix conpath \
  --decoder-variant correlated \
  --output site/data/flatlands_conpath_k128_support_clamped_validation.json
PYTHONPATH=src "$python_bin" scripts/build_flatlands_k128_snapshot.py \
  --root "$independent_root" \
  --run-suffix independent \
  --decoder-variant independent \
  --output site/data/flatlands_independent_k128_support_clamped_validation.json

mkdir -p "$comparison_root"
PYTHONPATH=src "$python_bin" scripts/compare_flatlands_k128_paired.py \
  --correlated-root "$correlated_root" \
  --independent-root "$independent_root" \
  --correlated-suffix conpath \
  --independent-suffix independent \
  --support-policy valid-support-clean-training \
  --output "$comparison_root/paired_comparison.json"
install -m 0644 "$comparison_root/paired_comparison.json" \
  site/data/flatlands_k128_support_clamped_paired_comparison.json

PYTHONPATH=src "$render_python" scripts/render_flatlands_k128_advantage.py \
  --evaluation-root "$comparison_root" \
  --correlated-evaluation-root "$correlated_root" \
  --independent-evaluation-root "$independent_root" \
  --paired-report "$comparison_root/paired_comparison.json" \
  --evidence-contract clean-support-training \
  --correlated-checkpoint-root "$correlated_root" \
  --independent-checkpoint-root "$independent_root" \
  --device cuda \
  --output site/assets/flatlands_k128_clean_candidate.png \
  --metadata-output site/data/flatlands_k128_clean_candidate.json

PYTHONPATH=src "$python_bin" -m unittest discover -s tests -q
"$python_bin" -m compileall -q src scripts
git diff --check

PYTHONPATH=src "$python_bin" - <<'PY'
import hashlib
import json
from pathlib import Path

metadata_path = Path("site/data/flatlands_k128_clean_candidate.json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
figure_path = Path(metadata["figure"]["path"])
actual_hash = hashlib.sha256(figure_path.read_bytes()).hexdigest()
assert metadata["clean_support_training"] is True
assert metadata["posthoc_checkpoint_evaluation"] is False
assert metadata["retraining_required"] is False
assert metadata["test_evaluated"] is False
assert actual_hash == metadata["figure"]["sha256"]
assert all(
    item["brier"]["bootstrap_95"][0] > 0.0
    for item in metadata["paired_per_seed_overall"]
)
print(
    {
        "clean_candidate": str(figure_path),
        "sha256": actual_hash,
        "all_per_seed_brier_intervals_positive": True,
        "test_evaluated": False,
    }
)
PY

echo "Clean candidate finalization passed; manual site promotion remains required."
