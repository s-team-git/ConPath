#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

gpu_python="$(command -v python)"
audit_python="$gpu_python"
manifest="results/unscenes3d_contract_manifest_ground_valid/manifest.json"
correlated_training="results/unscenes3d_ground_valid_support_clamped_f16_v1"
independent_training="results/unscenes3d_ground_valid_independent_support_clamped_f16_v1"
correlated_evaluation="results/unscenes3d_ground_valid_support_clamped_mean_map_k128_v1"
independent_evaluation="results/unscenes3d_ground_valid_independent_support_clamped_mean_map_k128_v1"
comparison_root="results/unscenes3d_ground_valid_support_clamped_k128_comparison_v1"
qualitative_root="results/unscenes3d_ground_valid_support_clamped_qualitative_v1"
log_root="results/unscenes3d_support_clamped_training_logs"
seeds=(20260831 20260901 20260902)

PYTHONPATH=src "$gpu_python" - <<'PY'
import torch
from PIL import Image  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("UnScenes3D clean-support pipeline requires CUDA")
PY

while pgrep -f 'scripts/train_flatlands_conpath.py --output-dir results/p1_flatlands_.*support_clamped_v1|scripts/wait_and_finalize_flatlands_valid_support.sh' >/dev/null; do
  printf '%s waiting for FlatLands training/finalization to release the experiment lane\n' "$(date --iso-8601=seconds)"
  sleep 60
done

mkdir -p "$log_root"

run_training_group() {
  local variant="$1"
  local output_root="$2"
  local pids=()
  local seed
  for seed in "${seeds[@]}"; do
    local output_dir="$output_root/seed${seed}"
    local log_path="$log_root/seed${seed}_${variant}.log"
    if [[ -s "$output_dir/run.json" ]]; then
      printf '%s reusing completed clean %s training seed %s\n' "$(date --iso-8601=seconds)" "$variant" "$seed"
      continue
    fi
    if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete non-empty training directory: $output_dir" >&2
      return 1
    fi
    PYTHONUNBUFFERED=1 "$gpu_python" scripts/train_unscenes3d_conpath.py \
      --manifest "$manifest" \
      --output-dir "$output_dir" \
      --seed "$seed" \
      --device cuda \
      --decoder-variant "$variant" \
      --feature-channels 16 \
      --latent-dim 4 \
      --batch-size 4 \
      --max-epochs 3 \
      --learning-rate 0.0003 \
      --weight-decay 0.0001 \
      --train-samples 4 \
      --max-reachability-steps 64 \
      --map-weight 1.0 \
      --variogram-weight 0.1 \
      --reachability-weight 0.0 \
      --posterior-samples 8 >"$log_path" 2>&1 &
    pids+=("$!")
    printf '%s started clean %s training seed %s pid %s\n' "$(date --iso-8601=seconds)" "$variant" "$seed" "$!"
  done
  local failures=0
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
  done
  if [[ "$failures" -ne 0 ]]; then
    echo "UnScenes3D training group failed: variant=$variant failures=$failures" >&2
    return 1
  fi
}

run_evaluation_group() {
  local training_root="$1"
  local output_root="$2"
  local variant="$3"
  local pids=()
  local seed
  for seed in "${seeds[@]}"; do
    local output_dir="$output_root/seed${seed}"
    local log_path="$log_root/seed${seed}_${variant}_mean_map_k128.log"
    if [[ -s "$output_dir/run.json" ]]; then
      printf '%s reusing completed clean %s K=128 evaluation seed %s\n' "$(date --iso-8601=seconds)" "$variant" "$seed"
      continue
    fi
    if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete non-empty evaluation directory: $output_dir" >&2
      return 1
    fi
    PYTHONUNBUFFERED=1 "$gpu_python" scripts/evaluate_unscenes3d_conpath_mean_map.py \
      --checkpoint "$training_root/seed${seed}/best.pt" \
      --manifest "$manifest" \
      --output-dir "$output_dir" \
      --seed "$seed" \
      --device cuda \
      --validation-samples 128 \
      --sample-chunk 16 >"$log_path" 2>&1 &
    pids+=("$!")
    printf '%s started clean %s K=128 evaluation seed %s pid %s\n' "$(date --iso-8601=seconds)" "$variant" "$seed" "$!"
  done
  local failures=0
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
  done
  if [[ "$failures" -ne 0 ]]; then
    echo "UnScenes3D K=128 evaluation group failed: variant=$variant failures=$failures" >&2
    return 1
  fi
}

echo "FlatLands experiment lane is complete; starting clean UnScenes3D support-boundary retraining."
run_training_group correlated "$correlated_training"
run_training_group independent "$independent_training"

PYTHONPATH=src "$audit_python" scripts/audit_unscenes3d_conpath_clean.py \
  --root "$correlated_training" \
  --decoder-variant correlated
PYTHONPATH=src "$audit_python" scripts/audit_unscenes3d_conpath_clean.py \
  --root "$independent_training" \
  --decoder-variant independent

run_evaluation_group "$correlated_training" "$correlated_evaluation" correlated
run_evaluation_group "$independent_training" "$independent_evaluation" independent

mkdir -p "$comparison_root"
PYTHONPATH=src "$audit_python" scripts/compare_unscenes3d_k128_paired.py \
  --correlated-root "$correlated_evaluation" \
  --independent-root "$independent_evaluation" \
  --correlated-training-root "$correlated_training" \
  --independent-training-root "$independent_training" \
  --manifest "$manifest" \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260903 \
  --output "$comparison_root/paired_comparison.json"
install -m 0644 "$comparison_root/paired_comparison.json" \
  site/data/unscenes3d_clean_support_k128_candidate.json

best_seed="$($audit_python - <<'PY'
import json
from pathlib import Path

root = Path("results/unscenes3d_ground_valid_support_clamped_mean_map_k128_v1")
records = []
for path in root.glob("seed*/run.json"):
    run = json.loads(path.read_text(encoding="utf-8"))
    records.append((float(run["event_metrics"]["scene_weighted_brier"]), int(path.parent.name[4:])))
if len(records) != 3:
    raise SystemExit("expected three correlated K=128 reports")
print(min(records)[1])
PY
)"

PYTHONPATH=src "$gpu_python" scripts/render_unscenes3d_qualitative.py \
  --checkpoint "$correlated_training/seed${best_seed}/best.pt" \
  --manifest "$manifest" \
  --output-dir "$qualitative_root/seed${best_seed}" \
  --seed "$best_seed" \
  --device cuda \
  --posterior-samples 128 \
  --sample-chunk 16
install -m 0644 "$qualitative_root/seed${best_seed}/qualitative_positive.png" \
  site/assets/unscenes3d_clean_candidate_positive.png
install -m 0644 "$qualitative_root/seed${best_seed}/qualitative_failure.png" \
  site/assets/unscenes3d_clean_candidate_failure.png
install -m 0644 "$qualitative_root/seed${best_seed}/report.json" \
  site/data/unscenes3d_clean_candidate_qualitative.json

PYTHONPATH=src "$audit_python" -m unittest discover -s tests -q
"$audit_python" -m compileall -q src scripts
git diff --check

PYTHONPATH=src "$audit_python" - <<'PY'
import hashlib
import json
from pathlib import Path

comparison_path = Path("site/data/unscenes3d_clean_support_k128_candidate.json")
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
assert comparison["posthoc_checkpoint_evaluation"] is False
assert comparison["retraining_required"] is False
assert comparison["test_evaluated"] is False
assert comparison["manifest"]["test_locked_sites"] == ["location_6"]
qualitative = json.loads(Path("site/data/unscenes3d_clean_candidate_qualitative.json").read_text(encoding="utf-8"))
assert qualitative["posthoc_checkpoint_evaluation"] is False
assert qualitative["retraining_required"] is False
assert qualitative["test_evaluated"] is False
for name in ("positive", "failure"):
    path = Path(f"site/assets/unscenes3d_clean_candidate_{name}.png")
    assert path.is_file() and path.stat().st_size > 0
    print(name, hashlib.sha256(path.read_bytes()).hexdigest())
print("UnScenes3D clean-support candidates passed; manual website promotion remains required.")
PY
