#!/usr/bin/env bash
# Asynchronous-execution evaluation on LIBERO, the paper's protocol: R:k = 10:k (replan every 10 steps, the new chunk
# takes over k steps after its observation, the running chunk executing meanwhile), 50 episodes per task
# (seed 1000 + episode, init state = episode), four suites -> 500 episodes per suite, random flow noise.
#   method ours:   the student (EMA weights), the running chunk's next min(k, 10) actions given as a clean prefix
#   method async:  the start policy, plain switch (k = 0: synchronous execution)
# Usage: bash scripts/pi05_libero/eval.sh ours|async K [suite ...]
#   -> outputs/eval/<method>_10_<K>/<suite>/{episodes.csv,summary.json,report.txt}
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}
method=${1:?ours or async} K=${2:?delay k}
shift 2
suites=${@:-libero_spatial libero_object libero_goal libero_10}
if [[ ${method} == ours ]]; then
  STUDENT=${STUDENT:-checkpoints/pi05_libero_student}
  # a LoRA checkpoint: evaluated as base + adapter (what delayopsd.checkpoints writes; the weights are not merged)
  LIGHT=outputs/eval/_ours_light
  [[ -d ${LIGHT}/pretrained_model ]] || python -m delayopsd.checkpoints --out ${LIGHT}/pretrained_model ${STUDENT}
  policy=(--policy.path=${LIGHT}/pretrained_model/pretrained_model --rtc=trained --prefix_max=10)
else
  policy=(--policy.path=${POLICY:-checkpoints/pi05_libero_finetuned} --rtc=off)
fi
for suite in ${suites}; do
  python -m delayopsd.eval.rollout "${policy[@]}" --policy.compile_model=false \
    --env.type=libero --env.task=${suite} --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --eval.batch_size=1 --eval.n_episodes=50 --eval.use_async_envs=false \
    --seed=1000 --settings="[\"10:${K}\"]" --noise_mode=random --resume=true \
    --output_dir=outputs/eval/${method}_10_${K}/${suite}
done
