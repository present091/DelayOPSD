#!/usr/bin/env bash
# Stage-1 teacher: bc/24 finetuned on the expert data with flow matching + masked flow matching, 8 epochs, EMA 0.999.
# The 12 levels in 4 shards of 3 (as in the paper; a level's random stream depends on its shard); one GPU,
# ~22 min per shard. The teacher is outputs/teacher/7.
# Usage: bash scripts/train_teacher.sh [shard ...]     (default: 0 1 2 3, one after another)
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
for k in ${@:-0 1 2 3}; do
  .venv/bin/python -u src/train_stage1.py --run-path assets/expert --load-dir assets/bc/24 --output-dir outputs/teacher \
    --shard $k/4 --seed 0
done
