#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shared_dir="$(dirname "$project_dir")"
run_name="smoke-$(TZ=Asia/Taipei date +%Y%m%d-%H%M%S)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$shared_dir/logs"
"$shared_dir/.venv-gplasc/bin/python" -u "$project_dir/run_smoke.py" \
    --data-folder "$shared_dir/data" \
    --output-dir "$shared_dir/runs/$run_name" \
    2>&1 | tee "$shared_dir/logs/$run_name.log"
