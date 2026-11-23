#!/usr/bin/env bash
# Stage 1 (teacher, masked FM on the demonstrations, ~3 h on one RTX 4090) and stage 2 (student, OPSD on the buffers,
# ~17 h on one RTX 4090; batch 4, LoRA r16 on the VLM + the full action expert, fp32 masters, AdamW moments on CPU).
# Usage: bash scripts/pi05_libero/train.sh teacher|student
set -euo pipefail
stage=${1:?teacher or student}
python -m delayopsd.train --config_path=configs/pi05_libero/${stage}.json "${@:2}"
if [[ ${stage} == teacher ]]; then  # the student's config reads the teacher's EMA weights from here
  ln -sfn "$(realpath outputs/teacher/checkpoints/010000/pretrained_model_ema)" checkpoints/pi05_libero_teacher
fi
