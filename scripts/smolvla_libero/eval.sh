#!/usr/bin/env bash
# Asynchronous-execution evaluation of SmolVLA on LIBERO, the same protocol as scripts/pi05_libero/eval.sh
# (R:k = 10:k, 50 episodes per task, seed 1000, recorded init states, random flow noise) with 256 px renders.
#   method ours:   the student (EMA weights), the running chunk's next min(k, 10) actions given as a clean prefix
#   method async:  lerobot/smolvla_libero, plain switch (k = 0: synchronous execution)
# Usage: bash scripts/smolvla_libero/eval.sh ours|async K [suite ...]
#   -> outputs/eval_smolvla/<method>_10_<K>/<suite>/{episodes.csv,summary.json,report.txt}
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}
method=${1:?ours or async} K=${2:?delay k}
shift 2
suites=${@:-libero_spatial libero_object libero_goal libero_10}
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
if [[ ${method} == ours ]]; then
  STUDENT=${STUDENT:-checkpoints/smolvla_libero_student}
  LIGHT=outputs/eval_smolvla/_ours_light
  [[ -d ${LIGHT}/pretrained_model ]] || python -m delayopsd.checkpoints --out ${LIGHT}/pretrained_model ${STUDENT}
  policy=(--policy.path=${LIGHT}/pretrained_model/pretrained_model --rtc=trained --prefix_max=10)
else
  policy=(--policy.path=${POLICY:-lerobot/smolvla_libero} --rtc=off)
fi
for suite in ${suites}; do
  python -m delayopsd.eval.rollout "${policy[@]}" --policy.compile_model=false \
    --env.type=libero --env.task=${suite} --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --env.observation_height=256 --env.observation_width=256 --env.init_states=true --env.hard_reset=true \
    --eval.batch_size=1 --eval.n_episodes=50 --eval.use_async_envs=false \
    --seed=1000 --settings="[\"10:${K}\"]" --noise_mode=random --resume=true --rename_map="${RENAME}" \
    --output_dir=outputs/eval_smolvla/${method}_10_${K}/${suite}
done
