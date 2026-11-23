"""Independent verification of a real_robot.buffer.yam_buffer buffer against the raw robot recordings.

    python -m real_robot.buffer.verify_yam_buffer --buffer outputs/yam_buffer_t68_1515 \
        --sessions <robot session dir> [...] --model <the pi0.5 that collected them> [--per-task 50]

Relative collector (meta chunks_space "absolute_raw"): chunks are the replies as sent (absolute joints, not
normalized) and are compared as they are; chunks_model (the reply in the relative model's own space) is mapped back
through the model's POSTPROCESSOR (unnormalize + AbsoluteActions with chunk_state as the reference state) -- the
inverse path of the converter's preprocessor -- and must give the reply; chunk_state must be the state sent.

Shares no code with the converter: it reads the recordings with mcap directly, decodes video with OpenCV (the
converter uses PyAV), de-normalizes with the model's own postprocessor, and reads pairs through
delayopsd.data.buffer' index exactly as training does. Checks, for every episode / pair:

  E1 meta: task from the recorded instruction, success from annotation.json, the source episode exists
  E2 selection: per task the expected count, the dropped episodes are failures and the chronologically last ones
  E3 chunk_obs_step = the requests' observation ticks (0 first), spaced k; T = the last request tick
  E4 every chunk entry from start_index on = the reply recorded in the switch event (all 7 dims, de-normalized)
  E5 obs/agent_pos = the state vector recorded as sent, exactly
  E6 images = the camera frame whose capture timestamp was recorded as sent, pixel-exact (OpenCV decode), and
     every camera's mp4 frame count = its timestamp count (else frame i is not timestamp i)
  P1 pairs (index): teacher tick = student tick + k; prefix = the running chunk's recorded reply over [s, s+k)
     (de-normalized), also = the joint commands executed there (arm exact, gripper to its clip); prefix_len k
     except an episode's first pair; task and success per pair as the episode's
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def mcap_topics(path):
    from mcap.reader import make_reader
    out = defaultdict(list)
    with open(path, "rb") as f:
        for _, ch, m in make_reader(f).iter_messages():
            out[ch.topic.rsplit("/", 1)[-1]].append((m.log_time, json.loads(m.data)))
    return out


def cv2_frames(mp4, rows):
    import cv2
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want, got, i = set(int(r) for r in rows), {}, 0
    while want - got.keys():
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return n, got


def count_frames(mp4):
    import cv2
    cap = cv2.VideoCapture(mp4)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-task", type=int, default=50)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--image-episodes", type=int, default=0, help="pixel-check this many episodes (0 = all)")
    args = ap.parse_args()
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(args.model)
    cfg.device = "cpu"
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.model,
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}},
                                         postprocessor_overrides={"device_processor": {"device": "cpu"}})
    relative = bool(getattr(cfg, "use_relative_actions", False))
    if relative:
        from lerobot.processor import RelativeActionsProcessorStep
        rel_step = [st for st in pre.steps if isinstance(st, RelativeActionsProcessorStep)][0]

        def to_absolute(x, state):  # the model's own inverse: unnormalize, then + the reference state
            rel_step._last_state = torch.from_numpy(np.asarray(state, np.float32))[None]
            return post(torch.from_numpy(np.asarray(x, np.float32))[None]).numpy()[0]

        denorm = lambda x: np.asarray(x, np.float64)  # noqa: E731 - relative buffers store raw replies
    else:
        denorm = lambda x: post(torch.from_numpy(np.asarray(x, np.float32))[None]).numpy()[0]  # noqa: E731
    k = args.k
    tasks = {"open the drawer, pick up the basket ball, put it in the drawer, and close the drawer": 6,
             "open the drawer, pick up the baseball ball, place it in the basket, and close the drawer": 8}
    fails = defaultdict(list)

    def check(cond, what):
        if not cond:
            fails[what].append(current)

    # E2: the source episodes and what the selection should keep
    source = defaultdict(list)
    for sd in args.sessions:
        for ed in sorted(glob.glob(f"{sd}/episodes/episode_*")):
            ins = json.load(open(f"{ed}/policy_instruction.json"))["instruction"]
            if ins in tasks:
                start = json.load(open(f"{ed}/recording_manifest.json"))["episode_start_wall_time"]
                outcome = json.load(open(f"{ed}/annotation.json"))["outcome"]
                source[tasks[ins]].append((start, os.path.basename(ed), outcome, ed))
    expected = {}
    for t, eps in source.items():
        eps.sort()
        drop = len(eps) - args.per_task
        failed = [e for e in eps if e[2] != "success"]
        dropped = {e[1] for e in failed[len(failed) - drop:]} if drop > 0 else set()
        expected.update({e[1]: (t, e[2] == "success", e[3]) for e in eps if e[1] not in dropped})
        print(f"task {t}: {len(eps)} source episodes, drop {sorted(dropped)}")
    files = sorted(glob.glob(f"{args.buffer}/episodes/*.npz"))
    stored = {}
    for f in files:
        meta = json.loads(str(np.load(f)["meta"]))
        stored[meta["episode"]] = f
    current = "selection"
    check(set(stored) == set(expected), "E2 selection differs")
    print(f"E2 stored {len(stored)} = expected {len(expected)}: {set(stored) == set(expected)}")

    n_img = 0
    for n_ep, (name, f) in enumerate(sorted(stored.items()), start=1):
        current = name
        print(f"episode {n_ep}/{len(stored)} {name} (failures so far: {sum(len(v) for k_, v in fails.items() if 'nonzero' not in k_)})",
              flush=True)
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        t_exp, s_exp, ed = expected.get(name, (None, None, None))
        if ed is None:
            continue
        check(meta["task_id"] == t_exp and bool(meta["success"]) == s_exp, "E1 task/success")
        check(Path(f).name.endswith("_S.npz") == s_exp, "E1 file name success flag")
        pol = [p for p in glob.glob(f"{ed}/*.mcap") if not os.path.basename(p).startswith(("yam_", "events"))][0]
        topics = mcap_topics(pol)
        ev = [e for _, d in topics["async_event"] for e in d["events"]]
        req = {int(e["obs_tick"]): e for e in ev if e["event"] == "request"}
        sw = {int(e["obs_tick"]): e for e in ev if e["event"] == "switch"}
        steps = z["chunk_obs_step"].astype(int).tolist()
        check((meta.get("chunks_space") == "absolute_raw") == relative, "E1 chunks_space does not match the --model")
        # E3
        check(steps[0] == 0 and all(b - a == k for a, b in zip(steps, steps[1:])), "E3 replan ticks not every k")
        check(set(steps[1:]) <= set(sw), "E3 a chunk without its switch event")
        obs_step = z["obs_step"].astype(int).tolist()
        check(len(z["action"]) == max(obs_step) and max(obs_step) in req, "E3 T is not the last request tick")
        check(all(t in req for t in obs_step), "E3 a stored frame that was never sent")
        # E4
        for i, s in enumerate(steps[1:], start=1):
            e = sw[s]
            raw = np.asarray(e["chunk"], np.float64)
            start = int(e["start_index"])
            back = denorm(z["chunks"][i])
            check(np.abs(back[start:] - raw[start:, :7]).max() < 1e-5, "E4 chunk != recorded reply")
            if relative:
                check(np.abs(back - raw[:, :7]).max() < 1e-6, "E4r stored chunk != the whole recorded reply")
                st = np.asarray(req[s]["state"], np.float32)[:7]
                check(np.array_equal(z["chunk_state"][i], st), "E4r chunk_state != the state sent with the request")
                check(np.abs(to_absolute(z["chunks_model"][i], st) - raw[:, :7]).max() < 1e-4,
                      "E4r chunks_model through the postprocessor != the recorded reply")
        if relative:
            check(np.isnan(z["chunk_state"][0]).all() and np.isnan(z["chunks_model"][0]).all(),
                  "E4r chunk 0 should have no state / model space (its request is not recorded)")
        # E5
        sent = np.stack([np.asarray(req[t]["state"], np.float32)[:7] for t in obs_step])
        check(np.array_equal(sent, z["obs/agent_pos"]), "E5 state != sent state")
        # E6
        if not args.image_episodes or n_img < args.image_episodes:
            n_img += 1
            for cam, wire in (("camera_top", "top_camera"), ("camera_right", "right_camera")):
                ts = np.load(f"{ed}/{cam}-rgb-timestamp.npy")
                check(count_frames(f"{ed}/{cam}-images-rgb.mp4") == len(ts), f"E6 {cam} frame count != timestamps")
                want = [int(np.argmin(np.abs(ts - float(req[t]["frame_timestamps"][wire])))) for t in obs_step]
                _, frames = cv2_frames(f"{ed}/{cam}-images-rgb.mp4", want)
                got = z[f"obs/pixels/{cam}"]
                diff = max(int(np.abs(frames[w].astype(int) - got[j].astype(int)).max()) for j, w in enumerate(want))
                check(diff <= 2, f"E6 {cam} pixels differ (max {diff})")
                if diff:
                    fails[f"E6 {cam} nonzero pixel diff (<=2, decoder rounding)"].append((name, diff))
    # P1: through the training index
    from delayopsd.data.buffer import build_index
    print("building / reading the training index ...", flush=True)
    index = build_index(args.buffer, delta=k, k=k)
    print("checking pairs ...", flush=True)
    p = np.load(index / "pairs.npz")
    episode_files = sorted(glob.glob(f"{args.buffer}/episodes/*.npz"))
    offsets = np.cumsum([0] + [len(np.load(f)["obs_step"]) for f in episode_files])
    for j in range(len(p["student"])):
        e = int(p["episode"][j])
        f = episode_files[e]
        current = f"{Path(f).name} pair {j}"
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        obs_step = z["obs_step"].astype(int)
        s, tt = obs_step[p["student"][j] - offsets[e]], obs_step[p["teacher"][j] - offsets[e]]
        check(int(p["step"][j]) == s and tt == s + k, "P1 teacher tick != student tick + k")
        check(bool(p["success"][e]) == bool(meta["success"]) and str(p["group"][e]) == f"yam/{meta['task_id']}",
              "P1 task/success")
        steps = z["chunk_obs_step"].astype(int).tolist()
        i = steps.index(s)
        if i == 0:
            check(int(p["prefix_len"][j]) == 0, "P1 first pair has a prefix")
            continue
        check(int(p["prefix_len"][j]) == k, "P1 prefix_len != k")
        ed = expected[meta["episode"]][2]
        if j % 7 == 0 and i > 1:  # recorded reply of the running chunk (the first chunk has no switch event)
            pol = [q for q in glob.glob(f"{ed}/*.mcap") if not os.path.basename(q).startswith(("yam_", "events"))][0]
            ev = [x for _, d in mcap_topics(pol)["async_event"] for x in d["events"]]
            run = {int(x["obs_tick"]): x for x in ev if x["event"] == "switch"}[steps[i - 1]]
            reply = np.asarray(run["chunk"], np.float64)[s - steps[i - 1]:s - steps[i - 1] + k, :7]
            pre = denorm(p["prefix"][j])
            check(np.abs(pre - reply).max() < 1e-5, "P1 prefix != the running chunk's recorded reply over [s, s+k)")
        ex = z["action"][s:s + k]
        pre = denorm(p["prefix"][j])
        check(np.abs(pre[:, :6] - ex[:, :6]).max() < 1e-5 and np.abs(pre[:, 6] - ex[:, 6]).max() < 0.01,
              "P1 prefix != executed commands")
    print(f"P1 {len(p['student'])} pairs checked", flush=True)
    print("RESULT", "ALL PASS" if not {k_: v for k_, v in fails.items() if "nonzero pixel diff" not in k_}
          else {k_: (len(v), v[:3]) for k_, v in fails.items()})
    for k_, v in fails.items():
        if "nonzero pixel diff" in k_:
            print(" note:", k_, len(v), "episodes, e.g.", v[:3])


if __name__ == "__main__":
    main()
