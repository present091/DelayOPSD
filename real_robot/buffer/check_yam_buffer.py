"""Silent-bug checks of a real_robot.buffer.yam_buffer buffer before training on it.

    python -m real_robot.buffer.check_yam_buffer --buffer data/buffers/yam/15_15 \
        --model checkpoints/pi05_yam_8task --dataset-root data/yam_right_8task [--device cuda] [--pairs 24]

1  structure: replans every k ticks, the pair frames stored, the prefix the training reads (running chunk, indices
   k..2k) de-normalized by the model's own postprocessor equal to the commands executed over [s, s+k)
2  observations against the demonstrations of the same tasks, after the policy's preprocessor: keys, shapes,
   state per-dimension range (gripper units), image channel means (RGB order)
3  the model on buffer frames: pi(o_s)[k:2k] against the buffer policy's own chunk over the same ticks (what ran),
   as against pi(o_s) with the two cameras swapped and with the state of another episode -- a correct encoding
   is much closer than either corruption, and close to the same model's error on demonstration frames
"""

import argparse
import glob
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pairs", type=int, default=24)
    args = ap.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.envs import preprocess_observation
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from delayopsd.data.utils import prepare_batch

    files = sorted(glob.glob(f"{args.buffer}/episodes/*.npz"))
    cfg = PreTrainedConfig.from_pretrained(args.model)
    cfg.device = args.device
    cfg.compile_model = False
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.model,
                                         preprocessor_overrides={"device_processor": {"device": args.device}},
                                         postprocessor_overrides={"device_processor": {"device": "cpu"}})
    # ---- 1 structure
    worst, n_pairs, tasks = 0.0, 0, {}
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        k = int(meta["k"])
        s, a, obs_step = z["chunk_obs_step"].astype(int), z["action"], set(z["obs_step"].tolist())
        T = len(a)
        assert s[0] == 0 and (np.diff(s) == k).all(), (f, np.unique(np.diff(s)))
        for i in range(1, len(s)):
            if s[i] + k > T:
                continue
            assert s[i] in obs_step and s[i] + k in obs_step, (f, s[i])
            prefix = post(torch.from_numpy(z["chunks"][i - 1][k:2 * k])[None]).numpy()[0, :, :7]
            worst = max(worst, float(np.abs(prefix - a[s[i]:s[i] + k]).max()))
            n_pairs += 1
        tasks.setdefault(meta["task_id"], []).append(bool(meta["success"]))
    print(f"[1] {len(files)} episodes, {n_pairs} pairs with a prefix; max |prefix - executed| = {worst:.2e} rad")
    print("    per task (episodes, success):", {t: (len(v), round(float(np.mean(v)), 2)) for t, v in tasks.items()})
    assert worst < 1e-4

    # ---- 2 observations vs demonstrations
    log = json.loads(open(f"{args.dataset_root}/meta/conversion_log.json").read())
    sess = {6: "open_and_pick_and_place_close_basket_ball", 8: "open_and_task_out_and_place_and_close"}
    eps = [i for i, e in enumerate(log) if e["session"].removeprefix("260912_") in sess.values()]
    ds = LeRobotDataset("yam_right_8task_v30", root=args.dataset_root, episodes=eps[::5])
    rng = np.random.default_rng(0)
    demo_rows = rng.choice(len(ds), args.pairs, replace=False)
    demo = [ds[int(r)] for r in demo_rows]
    d_state = np.stack([d["observation.state"].numpy() for d in demo])
    buf_obs = []
    for f in rng.choice(files, args.pairs):
        z = np.load(f)
        j = int(rng.integers(len(z["obs_step"])))
        buf_obs.append({"pixels": {c: z[f"obs/pixels/{c}"][j][None] for c in ("camera_top", "camera_right")},
                        "agent_pos": z["obs/agent_pos"][j][None], "_file": f, "_row": j})
    b_state = np.stack([o["agent_pos"][0] for o in buf_obs])
    print("[2] state per dim  demo [min,max]:", np.round(np.stack([d_state.min(0), d_state.max(0)], 1), 2).tolist())
    print("    state per dim  buffer[min,max]:", np.round(np.stack([b_state.min(0), b_state.max(0)], 1), 2).tolist())
    for cam in ("camera_top", "camera_right"):
        dm = np.stack([d[f"observation.images.{cam}"].numpy().mean((1, 2)) for d in demo]).mean(0)
        bm = np.stack([o["pixels"][cam][0].astype(np.float64).mean((0, 1)) / 255 for o in buf_obs]).mean(0)
        print(f"    {cam} RGB mean  demo {np.round(dm, 3).tolist()}  buffer {np.round(bm, 3).tolist()}")
    proc = preprocess_observation({"pixels": buf_obs[0]["pixels"], "agent_pos": buf_obs[0]["agent_pos"]})
    print("    buffer obs keys after preprocess_observation:", {k: tuple(v.shape) for k, v in proc.items()})
    dproc = demo[0]
    print("    demo  obs keys:", {k: tuple(v.shape) for k, v in dproc.items() if k.startswith("observation")})

    # ---- 3 the model on buffer frames
    policy = get_policy_class(cfg.type).from_pretrained(args.model, config=cfg, strict=True).eval().to(args.device)
    H = int(cfg.chunk_size)
    std = np.maximum(np.asarray(ds.meta.stats["action"]["std"], np.float64).reshape(-1)[:7], 1e-6)

    def predict(raw, task):
        obs = preprocess_observation(raw)
        obs["task"] = [task]
        batch = pre(obs)
        g = torch.Generator().manual_seed(7)
        noise = torch.randn(1, H, int(cfg.max_action_dim), generator=g).to(args.device)
        with torch.inference_mode():
            return post(policy.predict_action_chunk(batch, noise=noise).cpu()).float().numpy()[0, :, :7]

    errs = {"correct": [], "cameras_swapped": [], "other_state": []}
    picks = [(f, i) for f in rng.choice(files, args.pairs) for i in [None]]
    for n, (f, _) in enumerate(picks):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        k = int(meta["k"])
        s = z["chunk_obs_step"].astype(int)
        i = int(rng.integers(1, len(s) - 1))
        row = int(np.searchsorted(z["obs_step"], s[i]))
        ran = post(torch.from_numpy(z["chunks"][i][k:2 * k])[None]).numpy()[0, :, :7]  # its own chunk, ticks s+k..
        pix = {c: z[f"obs/pixels/{c}"][row][None] for c in ("camera_top", "camera_right")}
        st = z["obs/agent_pos"][row][None]
        other = np.load(files[(files.index(f) + 7) % len(files)])["obs/agent_pos"]
        variants = {"correct": {"pixels": pix, "agent_pos": st},
                    "cameras_swapped": {"pixels": {"camera_top": pix["camera_right"], "camera_right": pix["camera_top"]},
                                        "agent_pos": st},
                    "other_state": {"pixels": pix, "agent_pos": other[len(other) // 2][None]}}
        for name, raw in variants.items():
            errs[name].append(float(np.mean(np.abs(predict(raw, meta["task_description"])[k:2 * k] - ran) / std)))
    demo_err = []
    for d in demo[: args.pairs // 2]:
        item = {key: (v[None] if torch.is_tensor(v) else [v]) for key, v in d.items()}
        batch = prepare_batch(item, ds.meta.camera_keys, {}, pre)
        g = torch.Generator().manual_seed(7)
        noise = torch.randn(1, H, int(cfg.max_action_dim), generator=g).to(args.device)
        with torch.inference_mode():
            out = post(policy.predict_action_chunk(batch, noise=noise).cpu()).float().numpy()[0, :, :7]
        gt = d["action"].numpy().reshape(-1, 7)[:1]
        demo_err.append(float(np.mean(np.abs(out[:1] - gt) / std)))
    print("[3] |pi(o_s)[k:2k] - buffer policy's own chunk| / std:",
          {k: round(float(np.mean(v)), 3) for k, v in errs.items()})
    print("    (demo frames, |pi(o_t)[0] - a_t| / std:", round(float(np.mean(demo_err)), 3), ")")


if __name__ == "__main__":
    main()
