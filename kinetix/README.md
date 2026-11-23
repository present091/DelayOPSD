# DelayOPSD on Kinetix

DelayOPSD for the flow policies of the [Real-Time Chunking](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
Kinetix benchmark (12 levels, inference delays 0..4, execute horizon max(1, d)). The code is a set of files added to
that repository at commit `9296f31`; `setup.sh` clones it, copies them in and installs its environment.

```bash
bash kinetix/setup.sh                       # -> kinetix/rtc (upstream + our files, uv env, 33.5 GB of assets)
cd kinetix/rtc
bash scripts/train_teacher.sh               # stage-1 teacher, 4 shards x ~22 min
for l in worlds/l/*.json; do bash scripts/train_student.sh $l; done    # per level: 16 buffers, student, eval
bash scripts/eval_baseline.sh               # naive asynchronous execution of the base policy (delay 0 = synchronous)
```

| file | |
|---|---|
| `src/opsd.py` | masked flow matching (teacher), buffer pairs and the failed-pair rule, the DelayOPSD loss |
| `src/train_stage1.py` | stage-1 teacher: bc/24 + 8 epochs of flow matching + masked flow matching on the expert data |
| `src/collect_buffer.py` | rollout buffer: naive-async rollouts of bc/24 at delays 0..4 |
| `src/train_opsd.py` | DelayOPSD student: bc/24 trained on the buffers against the frozen teacher, 32 x 1,952 updates |
| `src/eval_sweep.py` | `eval_flow.eval` over the delay sweep, a few levels at a time |

Recipe of the paper: base policy bc/24 of the RTC release (start, buffer policy and teacher initialisation); teacher
EMA 0.999, epoch 7; per level 16 buffers (seeds 2000..2015) x 131,072 pairs per delay; failed episodes keep their
pairs before 1.5 x the median successful length and before they stall; student batch 512, AdamW 3e-4, EMA 0.999,
trust weighting off, epoch 31 evaluated with the committed actions as a clean prefix (`--methods realtime
--config.model.simulated-delay 5`). Results: `outputs/eval_<level>/results.csv` (`returned_episode_solved` per
delay); the evaluation seeds are `eval_flow.main`'s.

Cost of the paper's run on one RTX 3090: teacher 1.5 GPU-h (4 shards x 22 min); per level, buffers ~39 min and ~59 GB of
disk, student ~26 min; 14.4 GPU-h for all 12 levels.
