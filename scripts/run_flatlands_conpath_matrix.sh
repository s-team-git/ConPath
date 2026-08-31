#!/usr/bin/env bash
# Reproducible public-data ConPath matrix. This script refuses to start if the current
# process cannot see CUDA, so a container passthrough problem cannot silently produce a CPU
# result under a paper experiment directory.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PATHREL_PYTHON:-$project_root/.venv/bin/python}"
output_root="${PATHREL_MATRIX_OUTPUT:-$project_root/results/p1_flatlands_conpath_matrix}"
seeds_csv="${PATHREL_CONPATH_SEEDS:-20260831,20260901,20260902}"
max_epochs="${PATHREL_CONPATH_MAX_EPOCHS:-40}"
patience="${PATHREL_CONPATH_PATIENCE:-8}"
validation_samples="${PATHREL_CONPATH_VALIDATION_SAMPLES:-32}"
train_samples="${PATHREL_CONPATH_TRAIN_SAMPLES:-8}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter not found or not executable: $python_bin" >&2
  exit 2
fi

mkdir -p "$output_root"
environment_json="$output_root/environment.json"
PYTHONPATH=src "$python_bin" scripts/check_environment.py --json > "$environment_json"
PYTHONPATH=src "$python_bin" - "$environment_json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
torch_report = report.get("torch", {})
if not torch_report.get("cuda_available") or int(torch_report.get("device_count", 0)) < 1:
    kernel = report.get("nvidia_kernel", {})
    nodes = kernel.get("device_nodes", [])
    raise SystemExit(
        "CUDA preflight failed: host evidence may exist, but this process has no usable CUDA "
        f"device (device_nodes={nodes!r}, torch={torch_report.get('version')!r}). "
        "Fix GPU passthrough and rerun scripts/check_environment.py."
    )
PY

IFS=',' read -r -a seeds <<< "$seeds_csv"
if [[ "${#seeds[@]}" -lt 3 ]]; then
  echo "The paper matrix requires at least three seeds; got: $seeds_csv" >&2
  exit 2
fi

matrix_json="$output_root/matrix.json"
PYTHONPATH=src "$python_bin" - "$matrix_json" "$seeds_csv" "$max_epochs" "$patience" "$train_samples" "$validation_samples" <<'PY'
import json
import sys
from pathlib import Path

path, seeds, max_epochs, patience, train_samples, validation_samples = sys.argv[1:]
payload = {
    "schema_version": 1,
    "kind": "flatlands_conpath_experiment_matrix",
    "protocol": "P1_BASELINE_PROTOCOL.md v1 + ConPath pilot v1",
    "dataset_split": "FlatLands provenance.original_split; official archive split is forbidden",
    "seeds": [int(value) for value in seeds.split(",")],
    "variants": [
        {"id": "conpath", "flags": []},
        {"id": "no_global_factors", "flags": ["--disable-global-factors"]},
        {"id": "no_reachability_loss", "flags": ["--reachability-weight", "0"]},
    ],
    "fixed_args": {
        "device": "cuda",
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "test_evaluated": False,
    },
    "runs": [],
}
for seed in payload["seeds"]:
    for variant in payload["variants"]:
        payload["runs"].append({
            "seed": seed,
            "variant": variant["id"],
            "output_dir": f"seed{seed}_{variant['id']}",
            "status": "planned",
        })
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

log_root="$output_root/logs"
mkdir -p "$log_root"
for seed in "${seeds[@]}"; do
  for variant in conpath no_global_factors no_reachability_loss; do
    run_dir="$output_root/seed${seed}_${variant}"
    mkdir -p "$run_dir"
    resume_flags=()
    if [[ -f "$run_dir/latest.pt" ]]; then
      resume_flags+=(--resume)
    elif [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Run directory is non-empty but has no latest.pt; inspect before rerunning: $run_dir" >&2
      exit 2
    fi
    flags=()
    if [[ "$variant" == "no_global_factors" ]]; then
      flags+=(--disable-global-factors)
    elif [[ "$variant" == "no_reachability_loss" ]]; then
      flags+=(--reachability-weight 0)
    fi
    echo "[ConPath matrix] seed=$seed variant=$variant output=$run_dir" >&2
    PYTHONPATH=src "$python_bin" scripts/train_flatlands_conpath.py \
      --device cuda \
      --seed "$seed" \
      --output-dir "$run_dir" \
      --max-epochs "$max_epochs" \
      --patience "$patience" \
      --train-samples "$train_samples" \
      --validation-samples "$validation_samples" \
      "${resume_flags[@]}" \
      "${flags[@]}" \
      2>&1 | tee "$log_root/seed${seed}_${variant}.log"
  done
done

echo "Matrix complete. Test remains locked; evaluate each validation manifest with" >&2
echo "scripts/evaluate_flatlands_predictions.py and update the site only after multi-seed aggregation." >&2
