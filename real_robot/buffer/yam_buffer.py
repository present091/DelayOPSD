"""Convert YAM real-robot async recordings (robots_realtime ChunkScheduleAgent sessions) into an OPSD rollout buffer.

    python -m real_robot.buffer.yam_buffer --sessions <session dir> [...] --model <the pi0.5 that collected them> \
        --out <buffer root> [--per-task 50] [--tasks 6 8]

--model is the policy that collected the sessions (the action space follows its config).

Writes <out>/episodes/yam_<task>_<episode>_{S,F}.npz in the delayopsd.collect layout, so
delayopsd.data.buffer reads it unchanged (pair at every replan s: student frame s, prefix = the running
chunk's actions over [s, s+k), teacher frame s+k):

    obs/pixels/camera_top, obs/pixels/camera_right   (F, 480, 640, 3) uint8, the camera frame at or before the
                                                     request's observation time (what the station sent)
    obs/agent_pos      (F, 7)      follower joints + gripper at that time (the policy's observation.state)
    obs_step           (F,)        the ticks those frames belong to: every replan tick and replan tick + k
    action             (T, 7)      the joint commands executed, tick by tick (absolute)
    chunks             (C, H, 7)   each returned chunk. Absolute --model: its absolute targets normalized with the
                                   model's action quantiles (its model space). Relative --model (meta chunks_space
                                   "absolute_raw"): the reply as sent, absolute joint targets, NOT normalized -- a
                                   relative chunk's model space depends on the state it was inferred at, so the
                                   trainer puts the prefix in the student frame's space itself (relative.py)
    chunks_model       (C, H, 7)   relative --model only: the reply in the model's own space, model_space(reply,
                                   chunk_state) via its preprocessor (NaN for chunk 0, whose state was not recorded)
    chunk_state        (C, 7)      relative --model only: the state sent with each chunk's request (NaN for chunk 0)
    chunk_obs_step     (C,)        the tick whose observation produced each chunk (it takes over k ticks later)
    meta               json        group "yam", task_id, task_description, success (annotation.json), k, R, ...

Tick t of the agent is joint_pos message (offset + t); the offset comes from the switch events' timestamps. Every
executed command is checked against the chunk that owns its tick. --per-task keeps that many episodes per task,
dropping the chronologically last failures first (the success count is kept whole).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

TASKS = {6: "open the drawer, pick up the basket ball, put it in the drawer, and close the drawer",
         8: "open the drawer, pick up the baseball ball, place it in the basket, and close the drawer"}
CAMERAS = ("camera_top", "camera_right")


def read_mcap(path):
    from mcap.reader import make_reader
    out = {}
    with open(path, "rb") as f:
        for _, ch, m in make_reader(f).iter_messages():
            out.setdefault(ch.topic.rsplit("/", 1)[-1], []).append((m.log_time, json.loads(m.data)))
    return out


def decode(mp4, rows):
    import av
    want, got = set(int(r) for r in rows), {}
    with av.open(mp4) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = frame.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    return np.stack([got[int(r)] for r in rows])


def quantile_normalizer(model):
    from safetensors.numpy import load_file
    st = load_file(str(Path(model) / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    q01, q99 = st["action.q01"][:7].astype(np.float64), st["action.q99"][:7].astype(np.float64)
    return lambda x: (2.0 * (x - q01) / np.maximum(q99 - q01, 1e-8) - 1.0).astype(np.float32)


def relative_space(model):
    """None for an absolute --model; else reply (H, 7), state (7,) -> the model's own model space (H, 7)."""
    cfg = json.loads((Path(model) / "config.json").read_text())
    if not cfg.get("use_relative_actions"):
        return None
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    from delayopsd.relative import model_space
    pc = PreTrainedConfig.from_pretrained(model)
    pc.device = "cpu"
    pre, _ = make_pre_post_processors(pc, pretrained_path=model,
                                      preprocessor_overrides={"device_processor": {"device": "cpu"}},
                                      postprocessor_overrides={"device_processor": {"device": "cpu"}})
    return lambda reply, state: model_space(pre, torch.from_numpy(np.asarray(reply, np.float32))[None],
                                            torch.from_numpy(np.asarray(state, np.float32))[None]).numpy()[0]


def convert(ed, norm, k_expected, rel_space=None):
    pol = [f for f in glob.glob(f"{ed}/*.mcap") if not os.path.basename(f).startswith(("yam_", "events"))][0]
    p = read_mcap(pol)
    # Ticks and executed actions come from the chunk messages, one per tick, each holding what is left of the
    # running chunk after that tick's action: its head is the next tick's action, and at a switch it is the new
    # chunk from start_index + 1. joint_pos (what RobotNode received) is used only for the action at a switch tick
    # (new chunk index start_index), found by value: the row before the one equal to the switch message's head.
    # The two topics are not paired by index or time (an extra leading joint_pos in some episodes, and
    # ~ms jitter between the topics' log times).
    chunk_msgs = p["chunk"]
    ct = np.array([t for t, _ in chunk_msgs], np.int64)
    cmd = np.array([d["joint_pos"][:7] for _, d in p["joint_pos"]], np.float64)
    events = [e for _, d in p.get("async_event", []) for e in d.get("events", [])]
    sw = sorted((e for e in events if e["event"] == "switch"), key=lambda e: e["tick"])
    req = {e["request_id"]: e for e in events if e["event"] == "request"}
    if not sw:
        raise ValueError("no switch events")
    msgs = [np.asarray(d["right"], np.float64) for _, d in chunk_msgs]
    lens = np.array([len(m) for m in msgs])
    grow = np.flatnonzero(np.diff(lens) > 0) + 1  # the switch rows: the message grows back
    ticks = [int(e["tick"]) for e in sw]
    if len(grow) != len(ticks) or len(set((grow - np.array(ticks)).tolist())) != 1:
        raise ValueError(f"{len(grow)} chunk growths for {len(ticks)} switches, offsets "
                         f"{sorted(set((grow[:len(ticks)] - np.array(ticks[:len(grow)])).tolist()))[:6]}")
    offset = int(grow[0] - ticks[0])  # chunk message row of tick 0
    H = len(msgs[offset]) + 1
    T = len(msgs) - offset
    if int(sw[-1]["tick"]) == T - 1:  # the episode ended on a switch tick: that tick's action is unknown, drop it
        sw, T = sw[:-1], T - 1
    switch_at = {int(e["tick"]): e for e in sw}
    for t in range(1, T):  # every message is the previous one advanced by a tick, except at a switch
        r = offset + t
        if t in switch_at:
            if len(msgs[r]) != H - int(switch_at[t]["start_index"]) - 1:
                raise ValueError(f"switch at tick {t}: message of {len(msgs[r])} entries")
        elif len(msgs[r]) != len(msgs[r - 1]) - 1 or not np.array_equal(msgs[r], msgs[r - 1][1:]):
            raise ValueError(f"tick {t}: the chunk message is not the previous one advanced by one")

    def before_value(target):
        """The joint_pos row just before the unique row equal to target (the next tick's action)."""
        # arm joints exactly; the gripper within 0.01 (joint_pos clips it to [0, 1], chunks overshoot by ~1e-3)
        hit = np.flatnonzero((np.abs(cmd[:, :6] - target[:6]).max(1) < 1e-6) & (np.abs(cmd[:, 6] - target[6]) < 0.01))
        if len(hit) != 1 or hit[0] == 0:
            raise ValueError(f"{len(hit)} joint_pos rows equal a chunk head")
        return cmd[hit[0] - 1]

    a = np.zeros((T, msgs[offset].shape[1]))
    for t in range(T):
        if t == 0 or t in switch_at:
            a[t] = before_value(msgs[offset + t][0])  # this tick's action: the joint_pos before the next one's
        else:
            a[t] = msgs[offset + t - 1][0]
    chunks, steps, starts, obs_time = [], [], [], {}
    first = np.zeros((H, a.shape[1]))
    first[0], first[1:] = a[0], msgs[offset]
    chunks.append(first); steps.append(0); starts.append(0); start_idx = [0]
    k_seen = set()
    for e in sw:
        if e.get("late") or int(e.get("wait_ticks") or 0):
            raise ValueError(f"late switch at tick {e['tick']}")
        t, start = int(e["tick"]), int(e["start_index"])
        k_seen.add(t - int(e["obs_tick"]))
        full = np.zeros((H, a.shape[1]))
        full[start], full[start + 1:] = a[t], msgs[offset + t]
        if "chunk" in e:  # the reply as the server sent it (what the station keeps and sends as the next prefix);
            raw = np.asarray(e["chunk"], np.float64)[:, :full.shape[1]]  # joint_pos clips the gripper to [0, 1]
            if np.abs(raw[start + 1:] - full[start + 1:]).max() > 1e-6 or np.abs(raw[start, :6] - full[start, :6]).max() > 1e-6:
                raise ValueError(f"switch at tick {t}: the recorded reply differs from the chunk messages")
            full[start] = raw[start]
            if rel_space is not None:  # relative: the whole reply as sent (its unexecuted head too)
                if raw.shape != full.shape:
                    raise ValueError(f"switch at tick {t}: reply of shape {raw.shape}")
                full = raw.copy()
        elif rel_space is not None:
            raise ValueError(f"switch at tick {t}: no recorded reply (needed for a relative --model)")
        chunks.append(full); steps.append(int(e["obs_tick"])); starts.append(t); start_idx.append(start)
        if e["request_id"] in req:
            obs_time[int(e["obs_tick"])] = float(req[e["request_id"]]["observation_timestamp"])
    if k_seen != {k_expected}:
        raise ValueError(f"delays {k_seen}, expected {k_expected}")
    for e in req.values():
        obs_time.setdefault(int(e["obs_tick"]), float(e["observation_timestamp"]))
    # every executed action is its owner chunk's entry (switch-tick actions included, by construction of a)
    owner = np.searchsorted(np.array(starts), np.arange(T), side="right") - 1
    # the entry that runs at tick t: start_index at the switch tick, then one per tick (start_index, not
    # t - obs_tick, so server-shifted replies (FutureRTC, start_index 0) check too; identical for start_index = k)
    idx = np.arange(T) - np.array(starts)[owner] + np.array(start_idx)[owner]
    if (idx >= H).any():
        raise ValueError("a chunk ran past its end")
    ran = np.stack([chunks[o][i] for o, i in zip(owner, idx)])
    # the gripper: a switch tick's action is joint_pos (clipped to [0, 1]), every other tick's the chunk's own value
    grip = np.minimum(np.abs(ran[:, 6] - a[:, 6]), np.abs(np.clip(ran[:, 6], 0.0, 1.0) - a[:, 6]))
    err = float(max(np.abs(ran[:, :6] - a[:, :6]).max(), grip.max()))
    if err > 1e-6:
        raise ValueError(f"actions differ from their chunks by {err}")
    # the switch-tick actions from joint_pos must continue the new chunk smoothly (a wrong row would jump)
    jump = max(float(np.abs(a[t, :6] - a[t + 1, :6]).max()) for t in switch_at if t + 1 < T)
    if jump > 0.1:
        raise ValueError(f"a switch-tick action jumps {jump:.3f} rad to the next tick")
    ticks_cmd = a
    # Frames: the policy inputs exactly as sent, from the request events (state vector, the capture timestamp of
    # each camera's frame). A pair needs both its student tick s and
    # its teacher tick s+k to be requests; the episode is cut after its last request (T), so the last chunk, whose
    # teacher frame was never sent, has no pair. (Choosing frames by observation time instead took a frame
    # one capture newer than the one sent, for 98% of the top-camera frames.)
    requests = {int(e["obs_tick"]): e for e in req.values()}
    need = sorted({t for s_ in steps if s_ in requests and s_ + k_expected in requests for t in (s_, s_ + k_expected)})
    if not need:
        raise ValueError("no pair with both frames sent")
    T = max(need)
    wire = {"camera_top": "top_camera", "camera_right": "right_camera"}
    obs, residual = {}, 0.0
    for cam in CAMERAS:
        ts = np.load(f"{ed}/{cam}-rgb-timestamp.npy")
        sent = np.array([float(requests[t]["frame_timestamps"][wire[cam]]) for t in need])
        rows = np.abs(ts[None, :] - sent[:, None]).argmin(1)
        residual = max(residual, float(np.abs(ts[rows] - sent).max()))
        obs[f"obs/pixels/{cam}"] = decode(f"{ed}/{cam}-images-rgb.mp4", rows)
    if residual > 1e-3:
        raise ValueError(f"a sent frame is not in the camera recording ({residual * 1e3:.1f} ms off)")
    obs["obs/agent_pos"] = np.array([np.asarray(requests[t]["state"], np.float32)[:7] for t in need], np.float32)
    instructions = {str(requests[t].get("instruction")) for t in need}
    if rel_space is None:
        arrays = dict(obs, obs_step=np.array(need, np.int32), action=ticks_cmd[:T].astype(np.float32),
                      chunks=np.stack([norm(c[:, :7]) for c in chunks]).reshape(len(chunks), H, 7),
                      chunk_obs_step=np.array(steps, np.int32))
    else:
        state = np.full((len(chunks), 7), np.nan, np.float32)
        model = np.full((len(chunks), H, 7), np.nan, np.float32)
        for i, s_ in enumerate(steps):
            if s_ in requests:  # every chunk but the first (its tick-0 request is never recorded)
                state[i] = np.asarray(requests[s_]["state"], np.float32)[:7]
                model[i] = rel_space(chunks[i][:, :7], state[i])
            elif i:
                raise ValueError(f"chunk {i} (obs tick {s_}) has no recorded request")
        arrays = dict(obs, obs_step=np.array(need, np.int32), action=ticks_cmd[:T].astype(np.float32),
                      chunks=np.stack([c[:, :7] for c in chunks]).astype(np.float32).reshape(len(chunks), H, 7),
                      chunks_model=model, chunk_state=state, chunk_obs_step=np.array(steps, np.int32))
    # the raw reply recorded at every switch must be the chunk reconstructed from the chunk messages (index k on)
    raw_err = 0.0
    for e in sw:
        i = steps.index(int(e["obs_tick"]))
        if "chunk" in e:
            raw = np.asarray(e["chunk"], np.float64)
            raw_err = max(raw_err, float(np.abs(raw[int(e["start_index"]) + 1:, :6] - chunks[i][int(e["start_index"]) + 1:, :6]).max()))
    if raw_err > 1e-6:
        raise ValueError(f"recorded raw chunks differ from the reconstruction by {raw_err}")
    return arrays, dict(T=T, chunks=len(chunks), frames=len(need), cmd_check_rad=err, frame_residual_ms=residual * 1e3,
                        raw_chunk_check=raw_err, sent_instructions=sorted(instructions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=0)
    ap.add_argument("--tasks", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--skip-bad", action="store_true",
                    help="skip episodes the strict checks reject (e.g. a late switch) instead of stopping; analysis only")
    args = ap.parse_args()
    rel_space = relative_space(args.model)
    norm = quantile_normalizer(args.model) if rel_space is None else None
    by_task = {t: [] for t in args.tasks}
    for sd in args.sessions:
        for ed in sorted(glob.glob(f"{sd}/episodes/episode_*")):
            ins = json.load(open(f"{ed}/policy_instruction.json"))["instruction"]
            task = next((t for t, s in TASKS.items() if s == ins), None)
            if task not in by_task:
                continue
            outcome = json.load(open(f"{ed}/annotation.json")).get("outcome")
            start = json.load(open(f"{ed}/recording_manifest.json"))["episode_start_wall_time"]
            by_task[task].append((start, ed, outcome))
    out = Path(args.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for task, eps in by_task.items():
        eps.sort()
        if args.per_task:
            drop = len(eps) - args.per_task
            fails = [i for i, (_, _, o) in enumerate(eps) if o != "success"]
            dropped = set(fails[-drop:]) if drop > 0 else set()
            if drop > len(fails):
                raise SystemExit(f"task {task}: {len(eps)} episodes, cannot drop {drop} failures")
            eps = [e for i, e in enumerate(eps) if i not in dropped]
            report.append({"task": task, "dropped_failures": sorted(os.path.basename(e) for i, (_, e, _) in enumerate(
                sorted(by_task[task])) if i in dropped)})
        for start, ed, outcome in eps:
            if outcome not in ("success", "failure"):
                raise SystemExit(f"{ed}: outcome {outcome}")
            try:
                arrays, info = convert(ed, norm, args.k, rel_space)
            except ValueError as err:
                if not args.skip_bad:
                    raise
                print(json.dumps({"skipped": os.path.basename(ed), "reason": str(err)}), flush=True)
                report.append({"skipped": os.path.basename(ed), "reason": str(err)})
                continue
            name = os.path.basename(ed).removeprefix("episode_")
            meta = dict(inputs="sent", group="yam", task_id=task, task_description=TASKS[task], success=outcome == "success",
                        k=args.k, R=args.k, seed=name, state_hash=name, session=os.path.basename(os.path.dirname(
                            os.path.dirname(ed))), episode=os.path.basename(ed), start_wall=start,
                        chunks_space="model" if rel_space is None else "absolute_raw", model=str(args.model), **info)
            np.savez_compressed(out / f"yam_{task}_{name}_{'S' if meta['success'] else 'F'}.npz",
                                meta=np.array(json.dumps(meta)), **arrays)
            if meta["sent_instructions"] != [TASKS[task]]:
                raise SystemExit(f"{ed}: sent instruction {meta['sent_instructions']}")
            report.append({k: meta[k] for k in ("task_id", "episode", "success", "T", "chunks", "frames",
                                                 "cmd_check_rad", "frame_residual_ms", "raw_chunk_check")})
            print(json.dumps(report[-1]), flush=True)
    (Path(args.out) / f"conversion_t{'_'.join(map(str, args.tasks))}.json").write_text(json.dumps(report, indent=1))
    print("DONE", {t: len(v) for t, v in by_task.items()})


if __name__ == "__main__":
    main()
