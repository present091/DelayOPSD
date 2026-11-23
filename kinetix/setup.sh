#!/usr/bin/env bash
# Kinetix setup: the Real-Time Chunking Kinetix code at the commit we used, our OPSD files copied into it, its uv
# environment, and the released assets (base policies bc/24 and bc/31, expert data for the stage-1 teacher).
# Usage: bash kinetix/setup.sh [target dir, default kinetix/rtc] [--no-assets]
# Needs git, curl, a CUDA 12 driver, uv (or pip to install it).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=${1:-$HERE/rtc}
if [[ ! -d $DST/.git ]]; then
  git clone https://github.com/Physical-Intelligence/real-time-chunking-kinetix "$DST"
  git -C "$DST" checkout 9296f31
  git -C "$DST" submodule update --init --recursive
fi
cp "$HERE"/src/*.py "$DST/src/"
mkdir -p "$DST/scripts" "$DST/assets"
cp "$HERE"/scripts/*.sh "$DST/scripts/"
cd "$DST"
command -v uv > /dev/null || { python3 -m pip install --user uv; export PATH="$HOME/.local/bin:$PATH"; }
uv sync --python 3.12   # jaxlib 0.4.35 has no wheel for newer Pythons
.venv/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
if [[ ${2:-} != --no-assets ]]; then
  bash scripts/download_assets.sh
fi
