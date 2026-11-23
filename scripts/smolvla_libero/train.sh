#!/usr/bin/env bash
# The SmolVLA student: OPSD on the buffers from lerobot/smolvla_libero, with the frozen start policy as the teacher
# (no separate stage 1). 8 buffer pairs per update (the demonstration batch is not in the loss), LoRA r16 on the VLM's text model + the full action
# expert, 10K steps, student EMA 0.999.
# Usage: bash scripts/smolvla_libero/train.sh
set -euo pipefail
python -m delayopsd.train --config_path=configs/smolvla_libero/student.json "$@"
