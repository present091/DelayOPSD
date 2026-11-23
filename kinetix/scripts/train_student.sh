#!/usr/bin/env bash
# OPSD student for one level: 16 rollout buffers of bc/24 (seeds 2000..2015; 2,048 naive-async rollouts and 131,072
# kept pairs per delay 0..4 each, ~59 GB per level in all), the student (bc/24 + 32 epochs of OPSD against
# outputs/teacher/7, EMA, trust weighting off), and its evaluation (last epoch, delays 0..4, prefix = the committed
# actions). Done steps are skipped on a rerun; KEEP_BUFFERS=0 deletes the level's buffers after training.
#   -> outputs/student_<level>/31 (policy), outputs/eval_<level>/results.csv
# Usage: bash scripts/train_student.sh worlds/l/catapult.json     (all 12: worlds/l/{car_launch,cartpole_thrust,
#        catapult,catcher_v3,chain_lander,grasp_easy,h17_unicycle,hard_lunar_lander,mjc_half_cheetah,mjc_swimmer,
#        mjc_walker,trampoline}.json)
set -uo pipefail
cd "$(dirname "$0")/.."
LEVEL=${1:?level json} L=$(basename $1 .json) O=outputs PY=.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
[[ -e $O/teacher/7/policies/worlds_l_$L.pkl ]] || { echo "no teacher for $L: run scripts/train_teacher.sh"; exit 1; }
step() {  # <name> <command...>
  [[ -e $O/logs/$1.done ]] && return 0
  mkdir -p $O/logs; local name=$1; shift
  echo "$name started $(date '+%F %T')"
  "$@" > $O/logs/$name.log 2>&1 || { echo "$name failed (see $O/logs/$name.log)"; exit 1; }
  touch $O/logs/$name.done
}
B=()
for s in $(seq 2000 2015); do
  step buffer_s${s}_$L $PY -u src/collect_buffer.py --policy-dir assets/bc/24 --output-dir $O/buffers/s$s \
    --level-paths $LEVEL --levels-per-call 1 --seed $s
  B+=($O/buffers/s$s)
done
step student_$L $PY -u src/train_opsd.py --buffer-dir ${B[0]} --extra-buffer-dirs ${B[@]:1} --level-paths $LEVEL \
  --load-dir assets/bc/24 --teacher-dir $O/teacher/7 --output-dir $O/student_$L --seed 0 --num-epochs 32
[[ ${KEEP_BUFFERS:-1} == 0 ]] && rm -f $O/buffers/s20*/worlds_l_$L.npz
step eval_$L $PY -u src/eval_sweep.py --run-path $O/student_$L --output-dir $O/eval_$L --level-paths $LEVEL \
  --levels-per-call 1 --methods realtime --config.model.simulated-delay 5
echo "$L done $(date '+%F %T')"
