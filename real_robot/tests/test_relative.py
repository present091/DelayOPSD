"""Checks of the relative-action OPSD path (delayopsd.relative).

    python -m real_robot.tests.test_relative --device cuda

A  processor   model_space == the quantile formula on (a - state) for the arm, a for the gripper, from the relative
               checkpoint's own stats; the postprocessor maps it back to a; model_space(a, s_t) ==
               model_space(a, s_s) + frame_offset(s_s, s_t)
B  buffer      the relative buffer's index stores raw replies (prefix_space absolute_raw); the prefix == the executed
               commands a[s:s+k] (arm exact, gripper to its clip); in the student frame it equals the collector's own
               model space of the running chunk (yam_buffer chunks_model, state s') + frame_offset(s', s)
C  flow        on real buffer frames with the relative base: a chunk sampled by the teacher in its own frame (o_{s+k},
               state_t) == the same Euler integration done in the student's frame through to_teacher_state /
               to_student_velocity, compared in absolute joints; and without the conversion (offset 0) it is not
"""

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REL = "checkpoints/pi05_yam_8task"  # the relative-action start policy
BUF = "data/buffers/yam/15_15"  # its robot rollout buffer (real_robot.buffer.yam_buffer)


def stats(model):
    from safetensors.numpy import load_file
    st = load_file(f"{model}/policy_preprocessor_step_3_normalizer_processor.safetensors")
    return st["action.q01"][:7].astype(np.float64), st["action.q99"][:7].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--fp32", action="store_true", help="C in float32 (separates the conversion from bf16 rounding)")
    args = ap.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import preprocess_observation
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors



    from delayopsd import relative as rel
    from delayopsd.data.buffer import BufferPairSource
    from delayopsd.flow import observation_inputs, prefix_cache, sample_with_trajectory, velocity

    dev = args.device
    cfg = PreTrainedConfig.from_pretrained(REL)
    cfg.device, cfg.compile_model = dev, False
    assert cfg.use_relative_actions and list(cfg.relative_exclude_joints) == ["gripper"]
    pre, post = make_pre_post_processors(cfg, pretrained_path=REL,
                                         preprocessor_overrides={"device_processor": {"device": dev}},
                                         postprocessor_overrides={"device_processor": {"device": "cpu"}})
    assert rel.is_relative(pre)
    g = torch.Generator().manual_seed(0)
    rng_b = np.random.default_rng(0)
    ok = True

    # ---- A
    q01, q99 = stats(REL)
    a = torch.randn(3, 50, 7, generator=g).double() * 0.5
    s_s, s_t = torch.randn(3, 7, generator=g).double() * 0.5, torch.randn(3, 7, generator=g).double() * 0.5
    mask = np.array([1, 1, 1, 1, 1, 1, 0.0])
    want = 2 * ((a.numpy() - (s_s.numpy() * mask)[:, None]) - q01) / (q99 - q01) - 1
    got = rel.model_space(pre, a.float().to(dev), s_s.float().to(dev)).cpu().double().numpy()
    e1 = np.abs(got - want).max()
    # back through the postprocessor, whose AbsoluteActions step reads the state the preprocessor's step cached
    relstep = rel._steps(pre)[0]
    relstep._last_state = s_s.float().to(dev)
    back = post(torch.from_numpy(got).float()).double().numpy()
    e2 = np.abs(back - a.numpy()).max()
    off = rel.frame_offset(pre, s_s.float().to(dev), s_t.float().to(dev), 7)
    lhs = rel.model_space(pre, a.float().to(dev), s_t.float().to(dev))
    rhs = rel.model_space(pre, a.float().to(dev), s_s.float().to(dev)) + off
    e3 = float((lhs - rhs).abs().max())
    gr = float(off[..., 6].abs().max())
    print(f"A formula {e1:.2e}  postprocessor round trip {e2:.2e}  offset identity {e3:.2e}  gripper offset {gr:.1e}")
    ok &= e1 < 1e-5 and e2 < 1e-5 and e3 < 1e-5 and gr == 0.0

    # ---- B
    from delayopsd.data.buffer import build_index, load_frames
    idx = build_index(BUF, delta=15, k=15)
    p = np.load(idx / "pairs.npz")
    space_ok = str(p["prefix_space"]) == "absolute_raw"
    fake = SimpleNamespace(student=p["student"], teacher=p["teacher"], prefix=p["prefix"], prefix_lens=p["prefix_len"],
                           episode=p["episode"], prefix_len=15, frames={k: load_frames(idx, k) for k in p["keys"]})
    BufferPairSource._relative_inputs(fake, BUF, p, idx)  # raises unless the prefix is the executed commands
    files = sorted(glob.glob(f"{BUF}/episodes/*.npz"))
    eps = [np.load(f) for f in files]
    used = np.flatnonzero(p["prefix_len"] > 0)
    ran = np.stack([eps[p["episode"][j]]["action"][p["step"][j]:p["step"][j] + 15] for j in used])
    e4 = np.abs(fake.raw_prefix[used] - ran).max(axis=(0, 1))
    print(f"B prefix_space {str(p['prefix_space'])}; raw prefix vs executed a[s:s+k], max per dim:", np.round(e4, 6),
          f"pairs {len(used)}")
    # the pair source's student-frame prefix vs the collector's own model space of the running chunk (inferred at
    # s' with state_s'), moved to frame s by the offset: model_space(reply, st_s) == chunks_model + offset(st_s', st_s)
    worst_b = 0.0
    for j in rng_b.choice(used, 200, replace=False):
        z = eps[p["episode"][j]]
        s = int(p["step"][j])
        steps = z["chunk_obs_step"].astype(int)
        i = int(np.searchsorted(steps, s))
        if i < 2:  # chunk 0 has no recorded state
            continue
        at = s - int(steps[i - 1])
        st_s = torch.as_tensor(np.asarray(fake.states[p["student"][j]])[None], device=dev)
        st_p = torch.as_tensor(z["chunk_state"][i - 1][None], device=dev)
        mine = rel.model_space(pre, torch.as_tensor(fake.raw_prefix[j][None], device=dev), st_s)
        theirs = torch.as_tensor(z["chunks_model"][i - 1][at:at + 15][None], device=dev) + rel.frame_offset(pre, st_p, st_s, 7)
        worst_b = max(worst_b, float((mine - theirs).abs().max()))
    print(f"B student-frame prefix vs collector model space + offset: max {worst_b:.2e}")
    ok &= bool(space_ok and e4[:6].max() < 1e-5 and e4[6] < 1e-2 and worst_b < 1e-4)

    # ---- C
    if args.fp32:
        cfg.dtype = "float32"
    policy = get_policy_class(cfg.type).from_pretrained(REL, config=cfg, strict=True).eval().to(dev)
    if args.fp32:
        policy = policy.float()
    model = policy.model
    H, D, N = int(cfg.chunk_size), int(cfg.max_action_dim), int(cfg.num_inference_steps)
    rng = np.random.default_rng(1)
    files = sorted(glob.glob(f"{BUF}/episodes/*.npz"))
    worst, worst0, controls = 0.0, [], []
    for f in rng.choice(files, args.pairs, replace=False):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        steps = z["chunk_obs_step"].astype(int)
        s = int(steps[int(rng.integers(2, len(steps) - 3))])
        k = int(meta["k"])

        def obs(t):
            row = int(np.searchsorted(z["obs_step"], t))
            o = preprocess_observation({"pixels": {c: z[f"obs/pixels/{c}"][row][None] for c in ("camera_top", "camera_right")},
                                        "agent_pos": z["obs/agent_pos"][row][None]})
            o["task"] = [meta["task_description"]]
            return pre(o), torch.as_tensor(z["obs/agent_pos"][row][None], device=dev)

        (o_s, st_s), (o_t, st_t) = obs(s), obs(s + k)
        noise = torch.randn(1, H, D, generator=torch.Generator().manual_seed(9)).to(dev)
        with torch.inference_mode():
            inputs_t = observation_inputs(policy, o_t)
            x_t, _, _ = sample_with_trajectory(model, inputs_t, noise.clone(), N)  # teacher, own frame
            relstep._last_state = st_t
            abs_t = post(x_t[:, :, :7].cpu()).double().numpy()
            offset = torch.zeros(1, 1, D, device=dev)
            offset[:, :, :7] = rel.frame_offset(pre, st_s, st_t, 7)
            pad, kv = prefix_cache(model, inputs_t)
            res = {}
            # control: the same manual loop in the teacher's own frame (offset 0, read in the teacher frame) -- the
            # replication error of the loop itself
            for name, c in (("converted", offset), ("unconverted", torch.zeros_like(offset)), ("control", None)):
                y, dt = noise.clone(), -1.0 / N
                cc = torch.zeros_like(offset) if c is None else c
                for i in range(N):
                    tau = torch.full((1,), 1.0 + i * dt, device=dev)
                    v = velocity(model, pad, kv, rel.to_teacher_state(y, tau, cc), tau)
                    y = y + dt * rel.to_student_velocity(v, cc)
                relstep._last_state = st_t if c is None else st_s  # the frame y is in
                res[name] = post(y[:, :, :7].cpu()).double().numpy()
        d = {n: float(np.abs(r - abs_t).max()) for n, r in res.items()}
        print(f"C {Path(f).name} s={s}: teacher own frame vs student-frame integration, max |abs diff| rad: "
              f"converted {d['converted']:.2e}, control {d['control']:.2e}, unconverted {d['unconverted']:.3f} "
              f"(|state_s - state_t| max {float((st_s - st_t)[:, :6].abs().max()):.3f})")
        worst = max(worst, d["converted"])
        worst0.append((d["unconverted"], float((st_s - st_t)[:, :6].abs().max())))
        controls.append(d["control"])
    tol = 1e-4 if args.fp32 else 5e-3  # bf16: ~1e-3 rad between paths that are equal in exact arithmetic
    # converted equals the teacher; unconverted is off by the state difference (sanity: the test can fail)
    ok &= worst < tol and all(abs(u - ds) < 0.2 * ds + tol for u, ds in worst0)
    print(f"C worst converted {worst:.2e} (tol {tol}), worst control {max(controls):.2e}")
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
