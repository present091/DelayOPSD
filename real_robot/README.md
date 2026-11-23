# Real robot: YAM right arm

π0.5 on a YAM arm (6 joints + gripper, top and wrist cameras at 480×640, 30 Hz), two tasks of an 8-task teleoperation
dataset: **task 6** open the drawer, put the basketball in it, close the drawer; **task 8** open the drawer, take the
baseball, place it in the basket, close the drawer.

Execution: replan every `R = 15` steps (0.5 s); asynchronous methods switch to the new chunk `k = 15` steps after its
observation while the running chunk keeps executing (`15:15`); synchronous execution holds the arm during inference
(`15:0`). The student receives the running chunk's next 15 actions as its clean prefix.

## Pipeline

1. **Dataset.** `buffer/convert_yam_to_lerobot.py` converts teleoperation recordings (mcap joints, mp4 cameras) to a
   LeRobot v3.0 dataset: state = follower joints + gripper, action = leader joints + gripper, absolute. The start
   policy normalizes with *relative-action* statistics (arm joints relative to the chunk's start state, gripper
   absolute; LeRobot's `lerobot-edit-dataset --operation.type=recompute_stats --operation.relative_action=true`).
   The YAM dataset and robot recordings are not released; the code runs on any dataset in this format.
2. **Start policy.** `lerobot/pi05_base` fine-tuned on the 8 tasks with LeRobot's trainer: relative actions with the
   gripper excluded (`--policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]'`), one empty
   camera slot, full fine-tuning in bf16, 20K steps at batch 64, lr 2.5e-5 (warmup 1000, cosine to 2.5e-6), weight
   decay 1e-10. Released as `checkpoints/pi05_yam_8task` (download link coming soon).
3. **Robot rollout buffer.** Sessions of the start policy at `15:15` on the robot (50 episodes per task), converted by
   `buffer/yam_buffer.py` (one npz per episode in the `delayopsd.collect` layout; the chunks are stored as sent, absolute
   joint targets). `buffer/verify_yam_buffer.py` and `buffer/check_yam_buffer.py` check a buffer before training.
4. **Teacher and student.** `python -m delayopsd.train --config_path=configs/yam/teacher.json`, then `configs/yam/student.json`.
   With relative actions a buffer pair spans two frames (the student's at `s`, the teacher's at `s + k`): the prefix is
   put in the student's frame, the student's flow state is shifted into the teacher's frame and the teacher's velocity
   shifted back (`delayopsd/relative.py`). `tests/test_relative.py` and `tests/test_relative_e2e.py` check this against
   closed-form values computed from the raw buffer and the checkpoint's statistics.
5. **Deployment.** `server/server.py` serves a checkpoint over HTTP to the robot's chunk-schedule client:

   ```bash
   pip install -e ".[server]"
   python -m real_robot.server.server --checkpoint checkpoints/yam_student --port 8221
   ```

   Requests without `rtc_mode` are plain sampling (synchronous `15:0` or naive asynchronous `15:15`). With
   `rtc_mode=trained` the request carries the running chunk's 15 committed actions (absolute joints), which the server
   maps into the request's model space and feeds as the clean prefix to the trainer's own sampler (`delayopsd.flow`). The
   reply is the absolute 50-step chunk; the client handles the switch, speed limits and safety.
