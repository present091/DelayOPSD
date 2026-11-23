"""Convert YAM right-arm teleop episodes to a LeRobot v3.0 dataset.

state = follower joint_pos + gripper_pos, action = leader joint_pos (6 arm + gripper). Both stay absolute and are
linearly interpolated onto the camera_top frame times; camera_right takes the nearest frame. Each episode gets one
paraphrase, chosen by random.Random(<episode dir name>). Needs the `mcap` package (--pylib: extra sys.path entry).
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import av
import numpy as np

NAMES = [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"]
HEIGHT, WIDTH, FPS = 480, 640, 30


def read_joints(path, parse):
    from mcap.reader import make_reader

    times, values = [], []
    with open(path, "rb") as f:
        for _, _, message in make_reader(f).iter_messages():
            data = json.loads(message.data)
            times.append(data["_clock"]["host_wall_time_ns"] / 1e9)
            values.append(parse(data))
    return np.asarray(times), np.asarray(values, dtype=np.float64)


def read_video(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def interpolate(times, joint_times, values):
    return np.stack([np.interp(times, joint_times, values[:, i]) for i in range(values.shape[1])], axis=1)


def load_episode(episode):
    top = read_video(episode / "camera_top-images-rgb.mp4")
    right = read_video(episode / "camera_right-images-rgb.mp4")
    top_times = np.load(episode / "camera_top-rgb-timestamp.npy")
    right_times = np.load(episode / "camera_right-rgb-timestamp.npy")
    if len(top) != len(top_times) or len(right) != len(right_times):
        raise ValueError(f"{episode}: video/timestamp length mismatch")
    state_times, state = read_joints(episode / "yam_right.mcap", lambda d: d["joint_pos"] + d["gripper_pos"])
    action_times, action = read_joints(episode / "yam_leader_right.mcap", lambda d: d["joint_pos"])
    if state.shape[1] != 7 or action.shape[1] != 7:
        raise ValueError(f"{episode}: expected 7D state/action, got {state.shape} {action.shape}")
    # nearest camera_right frame for each camera_top frame
    index = np.clip(np.searchsorted(right_times, top_times), 1, len(right_times) - 1)
    index -= top_times - right_times[index - 1] < right_times[index] - top_times
    gap = np.abs(right_times[index] - top_times).max()
    overrun = max(state_times[0] - top_times[0], top_times[-1] - state_times[-1],
                  action_times[0] - top_times[0], top_times[-1] - action_times[-1], 0.0)
    return {
        "top": top,
        "right": [right[i] for i in index],
        "state": interpolate(top_times, state_times, state).astype(np.float32),
        "action": interpolate(top_times, action_times, action).astype(np.float32),
        "right_gap": float(gap),
        "overrun": float(overrun),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo_id", default="yam_right_8task")
    parser.add_argument("--pylib", default=None)
    parser.add_argument("--limit_per_session", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.pylib:
        sys.path.insert(0, args.pylib)

    from lerobot.datasets import LeRobotDataset

    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(args.root)
        shutil.rmtree(args.root)
    video = {"dtype": "video", "shape": (HEIGHT, WIDTH, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": NAMES},
        "observation.images.camera_top": video,
        "observation.images.camera_right": dict(video),
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=features, root=args.root, robot_type="yam_right",
        use_videos=True, streaming_encoding=True,
    )
    prompts = json.loads(args.prompts.read_text())
    log = []
    for session in sorted(prompts):
        episodes = sorted(p for p in (args.src / session / "episodes").iterdir() if p.is_dir())
        if args.limit_per_session:
            episodes = episodes[: args.limit_per_session]
        for episode in episodes:
            task = random.Random(episode.name).choice(prompts[session]["paraphrases"])
            data = load_episode(episode)
            for i in range(len(data["top"])):
                dataset.add_frame({
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                    "observation.images.camera_top": data["top"][i],
                    "observation.images.camera_right": data["right"][i],
                    "task": task,
                })
            dataset.save_episode()
            log.append({"session": session, "episode": episode.name, "frames": len(data["top"]), "task": task,
                        "right_gap_s": data["right_gap"], "overrun_s": data["overrun"]})
            print(f"{len(log)} {session}/{episode.name} frames={len(data['top'])} "
                  f"right_gap={data['right_gap'] * 1e3:.1f}ms overrun={data['overrun'] * 1e3:.1f}ms", flush=True)
    dataset.finalize()
    (args.root / "meta" / "conversion_log.json").write_text(json.dumps(log, indent=1))
    print(f"done episodes={len(log)} frames={sum(row['frames'] for row in log)} root={args.root}", flush=True)


if __name__ == "__main__":
    main()
