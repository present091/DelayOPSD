#!/usr/bin/env bash
# Rollout buffers of the start policy (lerobot/smolvla_libero) for the SmolVLA student: one per delay K = 0..10 at
# R:K = 10:K (plain asynchronous execution, K = 0 is synchronous), 40 scenes per task (seeds 2000..2039, held out from
# the evaluation's 1000..1049), four suites, random flow noise, 256 px renders. Each buffer: 1,600 episodes.
#   -> data/buffers/smolvla_libero/10_<K>/episodes/*.npz
# Usage: bash scripts/smolvla_libero/collect_buffers.sh [K ...]     (default: all of 0..10; one GPU, CUDA_VISIBLE_DEVICES)
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}
POLICY=${POLICY:-lerobot/smolvla_libero}
OUT=${OUT:-data/buffers/smolvla_libero}
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
for K in ${@:-0 1 2 3 4 5 6 7 8 9 10}; do
  for suite in libero_spatial libero_object libero_goal libero_10; do
    dir=${OUT}/10_${K}/${suite}
    [[ -f ${dir}/episodes.csv ]] && continue
    python -m delayopsd.collect --policy.path=${POLICY} --policy.compile_model=false \
      --env.type=libero --env.task=${suite} --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' --env.init_states=false \
      --env.observation_height=256 --env.observation_width=256 \
      --eval.batch_size=1 --eval.n_episodes=40 --eval.use_async_envs=false \
      --seed=2000 --settings="[\"10:${K}\"]" --noise_mode=random --resume=true --rename_map="${RENAME}" \
      --output_dir=${dir}
  done
  mkdir -p ${OUT}/10_${K}/episodes
  for suite in libero_spatial libero_object libero_goal libero_10; do
    ln -sf "$(realpath ${OUT}/10_${K}/${suite})"/episodes/*.npz ${OUT}/10_${K}/episodes/
  done
  echo "10:${K}: $(ls ${OUT}/10_${K}/episodes | wc -l) episodes, $(ls ${OUT}/10_${K}/episodes | grep -c _S.npz) successful"
done
