#!/usr/bin/env bash
# The start policy executed naively asynchronously (delay 0 = synchronous), all 12 levels, delays 0..4. Uses the
# RTC release's bc/31 (the newest epoch under assets/bc), as in the paper's table.
#   -> outputs/eval_naive/results.csv
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
.venv/bin/python -u src/eval_sweep.py --run-path assets/bc --output-dir outputs/eval_naive --methods naive --levels-per-call 2
