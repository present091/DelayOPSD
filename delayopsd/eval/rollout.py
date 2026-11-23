"""Simulated asynchronous chunk execution vs. synchronous replanning on LIBERO.

Schedule for one setting (R = replan_interval, k = switch_offset):

    * every R env steps, capture the observation at step t and start inference
      in a background thread while the robot keeps executing the current chunk;
    * at step t + k the new chunk takes over, starting from its index k
      (its first k actions are already covered by the previous chunk);
    * the very first chunk is waited for and starts from index 0.

A chunk therefore executes indices [k, k + R) (the first one [0, R + k)).
``k = 0`` is ordinary synchronous replanning: observe, wait, execute R actions.
Simulation time is frozen during inference, so results are exact. Real-robot
timing is estimated with dt = 1 / env.fps: synchronous settings stall for every
inference, asynchronous ones never stall but a chunk is "late" if inference
exceeds k * dt.

    python -m delayopsd.eval.rollout --policy.path=checkpoints/pi05_libero_finetuned --policy.compile_model=false \
      --env.type=libero --env.task=libero_spatial --env.task_ids='[0,1]' --eval.n_episodes=5 --seed=1000 \
      --settings='["10:0","10:10"]' --noise_mode=random --output_dir=outputs/eval/try

(scripts/pi05_libero/eval.sh runs the paper's protocol.)
"""

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from .protocol import (
    PROTOCOL_VERSION, SceneRegistry, fingerprints, prepare_reset,
)


@dataclass
class AsyncRolloutConfig(EvalPipelineConfig):
    eval: EvalConfig = field(default_factory=lambda: EvalConfig(batch_size=1, n_episodes=5, use_async_envs=False))
    # "R:k" = replan every R steps, switch to the new chunk k steps after its observation (10:0 = synchronous).
    settings: list[str] = field(default_factory=lambda: ["10:0", "10:10"])
    warmup_inferences: int = 3
    # "random": fresh initial noise per chunk (the paper's protocol). "shift": one noise field per episode indexed by
    # absolute time, so consecutive chunks share it where they overlap.
    noise_mode: str = "random"
    # Recorded state = episode + offset. The v2 fixture protocol is independent of reset
    # history; offset alone does NOT recreate legacy v1 scenes.
    init_state_offset: int = 0
    # Flow-noise draw only: the shift field is seeded with seed + ep + noise_seed_offset * 1_000_000,
    # while the scene (set_seed, init state) keeps seed + ep. 0 = the standard protocol. Used to
    # re-run one checkpoint on the same scenes with different noise.
    noise_seed_offset: int = 0
    # "off": plain sampling (synchronous at R:0, naive async at R:k); "trained": ours, see prefix_max.
    rtc: str = "off"
    # With --rtc=trained: hard clean prefix of min(k, prefix_max) steps from the running chunk (training-time
    # RTC inference), for checkpoints trained with opsd --opsd.prefix_len=prefix_max.
    prefix_max: int = 10
    # PI05 / SmolVLA, --rtc=off only: sample every chunk with the last suffix_mask action tokens removed from the expert's
    # attention keys (delayopsd.masking.suffix_mask), the input condition an OPSD teacher sees
    # (masked-FM stage1). Checks a teacher as it is used; the executed tokens must stay inside the kept H - delta.
    suffix_mask: int = 0
    # Save a GIF (all cameras side by side, red frame at chunk switches) for the first N episodes per task/setting.
    video_episodes: int = 0
    # Reuse an existing output directory and skip whatever already finished: settings that wrote
    # <out>/partial/<setting>.json.
    resume: bool = False
    # Cap this process's CUDA caching allocator. Set this below 0.5 when two standalone
    # evaluations share one GPU; 0 leaves the allocator uncapped.
    mem_fraction: float = 0.0




def parse_settings(settings, horizon):
    parsed = []
    for s in settings:
        r, k = (int(x) for x in s.split(":"))
        if r < 1 or not 0 <= k <= r:
            raise ValueError(f"{s}: need R >= 1 and 0 <= k <= R (one pending inference at a time)")
        if r + k > horizon:
            raise ValueError(f"{s}: a chunk must cover R + k <= H ({horizon}) steps")
        parsed.append((s, r, k))
    return parsed


class NoiseField:
    """Per-episode initial noise indexed by absolute physical time.

    The chunk observed at step t is denoised from ``eps[t : t + H]``, so two consecutive chunks
    carry the same noise realization wherever they cover the same physical time and only the R
    newest entries are fresh. This is the deployment-time counterpart of the ``shift_tail``
    coupling the OPSD teacher is trained with (delayopsd/masking.py): the switch
    compares old[R + k] against new[k], which are then the same noise element.

    Seeded from the episode seed alone, so every setting, checkpoint and noise mode of a run
    sees the same field (episodes are paired across models).
    """

    def __init__(self, steps, H, dim, device, seed):
        generator = torch.Generator().manual_seed(int(seed))
        self.eps = torch.randn(steps + H, dim, generator=generator).to(device)
        self.H = H

    def chunk(self, step):
        return self.eps[step: step + self.H].unsqueeze(0)


def install_suffix_mask(policy, delta):
    """Replace ``sample_actions`` (PI05 or SmolVLA) by Euler sampling with the last ``delta`` tokens masked out.

    Same steps and times as ``sample_actions`` (openpi loop), each velocity from
    ``delayopsd.flow.velocity(..., valid)`` as the OPSD teacher computes it (that module routes SmolVLA to
    delayopsd.flow_smolvla); masked tokens are returned as zeros. The first call also samples unmasked with the
    same noise and logs the difference over the kept tokens. PI05's action tokens attend bidirectionally, so there the
    mask changes the kept tokens; SmolVLA's attend causally and the difference is exactly 0.
    """
    from ..masking import suffix_mask
    from ..flow import prefix_cache, velocity

    model = policy.model
    smol = policy.config.type == "smolvla"
    H, unmasked = model.config.chunk_size, model.sample_actions
    steps = model.config.num_steps if smol else model.config.num_inference_steps
    if not 0 < delta < H:
        raise ValueError(f"--suffix_mask must be in (0, {H})")
    checked = False

    def sample_actions(*args, noise=None, **kwargs):
        nonlocal checked
        if smol:  # (images, img_masks, lang_tokens, lang_masks, state)
            observations = args[:5]
            batch, device = args[4].shape[0], args[4].device
        else:  # (images, img_masks, tokens, masks) + states= / state_masks= keywords
            observations = (*args[:4], kwargs.get("states"), kwargs.get("state_masks"))
            batch, device = args[2].shape[0], args[2].device
        if noise is None:
            noise = model.sample_noise((batch, H, model.config.max_action_dim), device)
        prefix_pad, kv = prefix_cache(model, observations)
        valid = suffix_mask(torch.full((noise.shape[0],), delta, dtype=torch.long, device=noise.device), H)

        def euler(mask):
            x, dt = noise, -1.0 / steps
            for step in range(steps):
                time = torch.full((x.shape[0],), 1.0 + step * dt, dtype=torch.float32, device=x.device)
                x = x + dt * velocity(model, prefix_pad, kv, x, time, mask)
            return x

        x = euler(valid).masked_fill(~valid[..., None], 0)
        if not checked:  # plumbing check (same loop unmasked == sample_actions) and the size of the mask's effect
            checked = True
            full = unmasked(*args, noise=noise, **kwargs)
            same = (euler(None) - full).abs().max().item()
            diff = (x - full)[:, : H - delta].abs()
            print(f"suffix_mask={delta}: unmasked loop vs sample_actions max {same:.2e} | masked vs unmasked over kept "
                  f"tokens: mean {diff.mean().item():.4f}, max {diff.max().item():.4f}, "
                  f"first 10 mean {diff[:, :10].mean().item():.4f}", flush=True)
        return x

    model.sample_actions = sample_actions


class ChunkWorker:
    """Runs preprocessing + chunk inference + postprocessing in one background thread."""

    def __init__(self, policy, pre, post, env_post, n_keep):
        self.policy, self.pre, self.post, self.env_post, self.n_keep = policy, pre, post, env_post, n_keep
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.rtc = False  # set by the caller when the policy carries an RTC processor
        self.futurertc = False  # FutureRTCWorker: replans get the committed env actions instead of an RTC prefix
        self.vlash = False  # VLASH: replan images + switch-step state, chunk index 0 at the switch
        self.prefix_max = 0  # trained RTC: cap on the hard prefix length (0 = guided / off)
        self.device = policy.config.device
        self.scene_registry = SceneRegistry()
        self.flip_video = True  # LIBERO frames; the caller clears it for upright cameras

    def _infer(self, obs, noise=None, rtc=None):
        start = time.perf_counter()
        # RTC guidance differentiates through the denoise step, and tensors created under
        # inference_mode can never require grad -- no_grad is what enable_grad can override.
        with (torch.no_grad() if self.rtc else torch.inference_mode()):
            extra = {} if noise is None else {"noise": noise}
            extra.update(rtc or {})
            chunk = self.policy.predict_action_chunk(self.pre(obs), **extra)  # (1, H, D) normalized
            env_actions = np.stack([
                self.env_post({ACTION: self.post(chunk[:, i])})[ACTION].to("cpu").numpy()[0]
                for i in range(self.n_keep)
            ])
        latency_ms = (time.perf_counter() - start) * 1e3
        return chunk[0].float().cpu().numpy(), env_actions, latency_ms

    def submit(self, obs, noise=None, rtc=None):
        return self.pool.submit(self._infer, obs, noise, rtc)

    def close(self):
        self.pool.shutdown()


def l1(a, b):
    return float(np.abs(a - b).sum(-1).mean())


def video_frame(raw, step, switched, label, flip=True):
    """One frame per camera (each camera becomes its own GIF), red border at chunk switches."""
    from PIL import Image, ImageDraw

    frames = {}
    for camera, images in raw["pixels"].items():
        frame = np.asarray(images)[0]
        frame = frame[::-1, ::-1].copy() if flip else frame.copy()  # LIBERO renders upside down
        if switched:
            frame[:3], frame[-3:], frame[:, :3], frame[:, -3:] = (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)
        image = Image.fromarray(frame)
        ImageDraw.Draw(image).text((5, 5), f"{label} t={step}", fill=(255, 255, 0))
        frames[camera.rpartition("/")[2].removesuffix("_image")] = np.asarray(image)
    return frames


def pin_init_state(env, index):
    """Pin the recorded state index only; use prepare_reset for the full v2 protocol."""
    for sub in getattr(env, "envs", []):
        sub.unwrapped.init_state_id = index


def run_episode(env, env_pre, worker, R, k, H, seed, max_steps, frames=None, label="", noise_field=None,
                init_state=None, episode_key=None):
    if init_state is None:
        raise ValueError("An explicit initial-state index is required by the v2 eval protocol")
    protocol = prepare_reset(env, init_state)
    set_seed(seed)
    raw, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    initial = fingerprints(env, raw, noise_field)
    drift = worker.scene_registry.check(episode_key, initial) if episode_key is not None else ()
    # No policy/processor memory or random draws from an earlier episode may carry over.
    worker.policy.reset()
    for processor in (worker.pre, worker.post, env_pre, worker.env_post):
        processor.reset()
    set_seed(seed)

    def observe():
        obs = preprocess_observation(raw)
        obs["task"] = list(env.call("task_description"))
        return env_pre(obs)

    chunk = env_chunk = origin = pending = stash = None
    executed, switch_steps, switches, latencies = [], [], [], []
    step, success, done = 0, False, False
    vlash = getattr(worker, "vlash", False)
    first = 0 if vlash else k  # new-chunk index that executes at the switch step

    def maybe_switch():
        # The pending chunk takes over exactly k steps after its observation (the first one immediately).
        nonlocal chunk, env_chunk, origin, pending
        if pending is None or (chunk is not None and step != pending[1] + k):
            return
        new, new_env, lat = pending[0].result()
        latencies.append(lat)
        if chunk is not None:
            idx_old = step - origin
            n = min(R, H - idx_old, H - first)
            d = chunk[idx_old] - new[first]
            # Full aligned overlap: old[idx_old + j] vs new[first + j], L2 per global step j (NaN-padded to H).
            full = min(H - idx_old, H - first)
            horizon = np.full(H, np.nan, dtype=np.float32)
            horizon[:full] = np.linalg.norm(chunk[idx_old: idx_old + full] - new[first: first + full], axis=-1)
            switches.append({
                "step": step, "latency_ms": lat, "late": bool(k > 0 and lat > k * worker.dt_ms),
                "switch_gap": float(np.linalg.norm(d)),
                "gap_pos": float(np.linalg.norm(d[0:3])), "gap_rot": float(np.linalg.norm(d[3:6])),
                "gap_grip": float(np.abs(d[6:]).sum()),
                "overlap_l1": l1(chunk[idx_old: idx_old + n], new[first: first + n]),
                "horizon_l2": horizon,
            })
            switch_steps.append(step)
        # VLASH chunks start at the switch step; the others at their observation step (k before the switch).
        chunk, env_chunk, origin, pending = new, new_env, (step if vlash else pending[1]), None

    while not done and step < max_steps:
        if vlash and stash is not None and (chunk is None or step == stash[1] + k):
            # VLASH query: the replan step's images with the robot state at execution (this switch step).
            obs = dict(stash[0])
            obs["observation.state"] = observe()["observation.state"]
            pending, stash = (worker.submit(obs, stash[2]), stash[1]), None
        maybe_switch()  # async: chunk observed k steps ago arrives
        if vlash and step % R == 0:
            stash = (observe(), step, None if noise_field is None else noise_field.chunk(step))
            if chunk is None or k == 0:
                pending, stash = (worker.submit(stash[0], stash[2]), stash[1]), None
        elif step % R == 0:  # replan: capture observation now, infer in the background
            noise = None if noise_field is None else noise_field.chunk(step)
            rtc = None
            if worker.futurertc and k > 0 and chunk is not None:
                # The k actions the old chunk executes while inference runs, in env space: what FutureRTC
                # rolls the observation forward with.
                start = step - origin
                rtc = {"committed_env": env_chunk[start: start + k]}
            elif worker.rtc and k > 0 and chunk is not None:
                # The old chunk executes indices [step - origin, step - origin + k) while inference runs;
                # those become RTC's prefix for the new chunk's first k steps (soft for guided, hard and
                # capped at the trained length for trained).
                delay = min(k, worker.prefix_max) if worker.prefix_max else k
                start = step - origin
                previous = chunk[start: start + delay]
                if len(previous) == delay:
                    rtc = {"inference_delay": delay, "execution_horizon": R,
                           "prev_chunk_left_over": torch.from_numpy(previous)[None].to(worker.device)}
            pending = (worker.submit(observe(), noise, rtc), step)
        maybe_switch()  # sync (k=0) or the very first chunk: wait for it now
        action = env_chunk[step - origin]
        executed.append(action)
        if frames is not None:
            frames.append(video_frame(raw, step, bool(switch_steps) and switch_steps[-1] == step, label,
                                      getattr(worker, "flip_video", True)))
        raw, _, terminated, truncated, info = env.step(action[None])
        done = bool(terminated[0] or truncated[0])
        if "is_success" in info:
            success = success or bool(np.asarray(info["is_success"]).reshape(-1)[0])
        step += 1
    if pending is not None:
        pending[0].result()  # drain so the next episode starts clean

    acts = np.asarray(executed)
    jumps = np.linalg.norm(np.diff(acts, axis=0), axis=-1) if len(acts) > 1 else np.zeros(0)
    boundary = [s - 1 for s in switch_steps if 0 < s < len(acts)]
    mask = np.zeros(len(jumps), dtype=bool)
    mask[boundary] = True
    return {
        "success": success, "steps": step, "n_inferences": len(latencies),
        **initial, "init_state_id": init_state, "protocol": protocol,
        "scene_drift": ",".join(drift),
        "stall_ms": float(sum(latencies)) if k == 0 else 0.0,
        "jump_within": float(jumps[~mask].mean()) if (~mask).any() else np.nan,
        "jump_boundary": float(jumps[mask].mean()) if mask.any() else np.nan,
    }, switches, latencies


def fmt(x, spec=".3f"):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def run_settings(cfg, settings, make_tasks, env_pre, worker, H, out, dt_ms, desc="async rollout"):
    """Every (setting, task, episode) of one policy; returns the per-episode and per-switch rows.

    ``make_tasks()`` builds a fresh set of environments; it is called once per setting rather than
    once per run. LIBERO appends a placement sampler to the domain on every hard reset and never
    clears it, so what a reset draws from the global NumPy stream depends on how many times that env
    has been reset before. Reusing one env across settings -- or across a queue of checkpoints --
    therefore made a score depend on the order it was measured in. Starting every setting from reset
    0 removes that, and makes episode k of a 5-episode run the same scene as episode k of a
    50-episode one. Costs ~27 s per setting against ~9 min of rollout.
    """
    episodes, switch_rows, lat_by_setting = [], [], {}
    envs, task_ids, pbar = None, None, None
    partials = out / "partial"
    partials.mkdir(parents=True, exist_ok=True)
    for name, R, k in settings:
        # A setting is 500 episodes and hours of GPU time, so it is written out the moment it ends.
        # Anything that kills the process later -- or a resumed run -- then keeps this setting.
        part = partials / f"{name.replace(':', '-')}.json"
        if part.is_file():
            saved = json.loads(part.read_text())
            episodes += saved["episodes"]
            switch_rows += [dict(row, horizon_l2=np.asarray(row["horizon_l2"], dtype=np.float32))
                            for row in saved["switches"]]
            lat_by_setting[name] = saved["latencies"]
            task_ids = task_ids or saved["task_ids"]
            if pbar is not None:
                pbar.update(len(saved["episodes"]))
            print(f"  {name}: resumed {len(saved['episodes'])} episodes from {part}")
            continue
        if envs is not None:
            close_envs(envs)
        envs, tasks = make_tasks()
        if pbar is None:
            task_ids = [t for _, t, _ in tasks]
            pbar = tqdm(total=len(settings) * len(tasks) * cfg.eval.n_episodes, desc=desc)
            pbar.update(len(episodes))  # episodes already loaded from partials
        lat_by_setting[name] = []
        for group, task_id, env in tasks:
            max_steps = env.call("_max_episode_steps")[0]
            for ep in range(cfg.eval.n_episodes):
                seed = (cfg.seed or 0) + ep  # identical seeds across settings
                wall = time.perf_counter()
                frames = [] if ep < cfg.video_episodes else None
                field = None
                if cfg.noise_mode == "shift":
                    config = worker.policy.config
                    field = NoiseField(max_steps, H, config.max_action_dim, config.device,
                                       seed + cfg.noise_seed_offset * 1_000_000)
                stats, switches, lats = run_episode(env, env_pre, worker, R, k, H, seed, max_steps,
                                                    frames, f"{name} {group}/{task_id} ep{ep}", field,
                                                    init_state=ep + cfg.init_state_offset,
                                                    episode_key=(group, task_id, ep, seed, ep + cfg.init_state_offset))
                if frames:
                    import imageio
                    from PIL import Image

                    (out / "videos").mkdir(exist_ok=True)
                    tag = "success" if stats["success"] else "fail"
                    stem = f"{name.replace(':', '-')}_{group}_{task_id}_ep{ep}_{tag}"
                    # One GIF per camera, 10 fps / 540 px: VS Code previews a GIF as an image, while its
                    # video player fails on webm/mp4. Cameras stay in separate files for side-by-side viewing.
                    for camera in frames[0]:
                        imageio.mimsave(
                            out / "videos" / f"{stem}_{camera}.gif",
                            [np.asarray(Image.fromarray(f[camera]).resize(
                                (540, 540 * f[camera].shape[0] // f[camera].shape[1])))
                             for f in frames[::2]], duration=100, loop=0)
                stats.update(setting=name, R=R, k=k, task=f"{group}/{task_id}", episode=ep, seed=seed,
                             wall_s=time.perf_counter() - wall,
                             est_real_s=(stats["steps"] * dt_ms + stats["stall_ms"]) / 1e3)
                episodes.append(stats)
                switch_rows += [dict(setting=name, task=stats["task"], episode=ep, **s) for s in switches]
                lat_by_setting[name] += lats
                pbar.update(1)
                done_eps = [e for e in episodes if e["setting"] == name]
                pbar.set_postfix(setting=name, success=f"{np.mean([e['success'] for e in done_eps]):.3f}")
        part.write_text(json.dumps({
            "episodes": [e for e in episodes if e["setting"] == name],
            "switches": [dict(row, horizon_l2=np.asarray(row["horizon_l2"]).tolist())
                         for row in switch_rows if row["setting"] == name],
            "latencies": lat_by_setting[name], "task_ids": task_ids,
        }, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    pbar.close()
    close_envs(envs)
    return episodes, switch_rows, lat_by_setting, task_ids


def build_envs(cfg, tasks=None):
    """One synchronous env per task."""
    return make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)


def task_factory(cfg):
    """A callable that builds a fresh set of one-env-per-task environments.

    ``run_settings`` calls it once per setting; see its docstring for why the LIBERO environments may not
    be shared across settings or across a queue of checkpoints."""
    def build():
        envs = build_envs(cfg)
        return envs, [(g, tid, e) for g, ts in envs.items() for tid, e in ts.items()]
    return build


def warmup_worker(worker, env_pre, cfg, n):
    """Warm the CUDA kernels on a real observation, using a throwaway env restricted to one task.

    Warmup must not touch the evaluation environments: a reset spent here would leave that task one
    placement sampler ahead of the others for the rest of the run, which is exactly the history
    dependence ``run_settings`` rebuilds the envs to avoid. A single-task env costs ~3 s."""
    env_cfg = replace(cfg.env, task_ids=[(cfg.env.task_ids or [0])[0]])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
    try:
        env = next(iter(next(iter(envs.values())).values()))
        raw, _ = env.reset(seed=[cfg.seed or 0], options={NEW_ROLLOUT_OPTION: True})
        obs = preprocess_observation(raw)
        obs["task"] = list(env.call("task_description"))
        obs = env_pre(obs)
        for _ in range(n):
            worker.submit(obs).result()
    finally:
        close_envs(envs)


def write_outputs(cfg, settings, task_ids, H, dt_ms, out, episodes, switch_rows, lat_by_setting, policy_path):
    """report.txt / summary.json / *.csv / switch_horizon_l2.npz for one policy."""
    protocol = ",".join(sorted({e.get("protocol", PROTOCOL_VERSION) for e in episodes}))
    summary = []
    for name, R, k in settings:
        eps = [e for e in episodes if e["setting"] == name]
        sw = [s for s in switch_rows if s["setting"] == name]
        lat = np.asarray(lat_by_setting[name])
        real = sum(e["est_real_s"] for e in eps)
        summary.append({
            "setting": name, "R": R, "k": k, "mode": "sync" if k == 0 else "async",
            "episodes": len(eps), "success_rate": float(np.mean([e["success"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "est_real_s": float(np.mean([e["est_real_s"] for e in eps])),
            "stall_ratio": sum(e["stall_ms"] for e in eps) / 1e3 / real if real else 0.0,
            "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p90_ms": float(np.percentile(lat, 90)),
            "late_switches": f"{sum(s['late'] for s in sw)}/{len(sw)}",
            "jump_within": float(np.nanmean([e["jump_within"] for e in eps])),
            "jump_boundary": float(np.nanmean([e["jump_boundary"] for e in eps])),
            "switch_gap": float(np.mean([s["switch_gap"] for s in sw])) if sw else np.nan,
            "overlap_l1": float(np.mean([s["overlap_l1"] for s in sw])) if sw else np.nan,
        })
    for s in summary:
        s["boundary_over_within"] = s["jump_boundary"] / s["jump_within"] if s["jump_within"] else np.nan

    header = (f"{'setting':>8} {'mode':>5} {'eps':>4} {'succ':>5} {'steps':>6} {'real_s':>7} {'stall%':>6} "
              f"{'lat50':>6} {'lat90':>6} {'late':>7} {'jmp_in':>7} {'jmp_bd':>7} {'bd/in':>6} "
              f"{'gap':>6} {'ovl_l1':>7}")
    lines = [f"PI05 {policy_path} | {cfg.env.task} tasks={task_ids} "
             f"| H={H} dt={dt_ms:.0f}ms | noise={cfg.noise_mode}+{cfg.noise_seed_offset} | rtc={cfg.rtc}{f' | suffix_mask={cfg.suffix_mask}' if cfg.suffix_mask else ''} | protocol={protocol}", header, "-" * len(header)]
    for s in summary:
        lines.append(
            f"{s['setting']:>8} {s['mode']:>5} {s['episodes']:>4} {s['success_rate']:>5.2f} "
            f"{s['mean_steps']:>6.1f} {s['est_real_s']:>7.2f} {100 * s['stall_ratio']:>6.1f} "
            f"{s['latency_p50_ms']:>6.0f} {s['latency_p90_ms']:>6.0f} {s['late_switches']:>7} "
            f"{fmt(s['jump_within']):>7} {fmt(s['jump_boundary']):>7} {fmt(s['boundary_over_within'], '.2f'):>6} "
            f"{fmt(s['switch_gap']):>6} {fmt(s['overlap_l1']):>7}"
        )
    # Self-check: one (task, ep) must be one simulator state, one fixture layout and one noise field
    # for every setting in this run. That is what makes the settings paired, and a violation is
    # fatal. The rendered image is checked too but only reported: see eval_protocol.FATAL_KEYS.
    by_episode = {}
    for e in episodes:
        by_episode.setdefault((e["task"], e["episode"]), set()).add(
            (e["state_hash"], e["fixture_hash"], e["noise_hash"]))
    varying = [key for key, seen in by_episode.items() if len(seen) > 1]
    if varying:
        raise RuntimeError(f"Simulator initial conditions differ across settings: {varying}")
    scenes = {(e["state_hash"], e["fixture_hash"]) for e in episodes}
    drifted = sorted({(e["task"], e["episode"]) for e in episodes if e.get("scene_drift")})
    lines += ["", f"scene check: {len(by_episode)} episodes, {len(scenes)} distinct scenes, "
                  f"state/fixtures/noise identical across all settings"]
    if drifted:
        lines += [f"  rendered-pixel drift on {len(drifted)} of {len(episodes)} episodes "
                  f"(same sim state, different RGB; flagged in episodes.csv as scene_drift): "
                  + ", ".join(f"{t} ep{e}" for t, e in drifted[:10])
                  + (" ..." if len(drifted) > 10 else "")]

    lines += [
        "",
        "setting=R:k  real_s: estimated real-robot seconds (steps*dt + sync stalls)  stall%: time waiting for inference",
        "late: switches where inference > k*dt  jmp_in/jmp_bd: mean ||a_t - a_t-1|| (env action space) within chunk / at switch",
        "gap: ||old[i] - new[k]|| at switch  ovl_l1: old vs new over the next R steps (normalized, sum dims, mean time)",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "summary.json").write_text(json.dumps({"config": cfg.settings, "summary": summary}, indent=2))
    (out / "eval_protocol.json").write_text(json.dumps({
        "protocol": protocol, "config": asdict(cfg), "checkpoint": str(policy_path),
        "seed_rule": "seed + episode", "init_state_rule": "offset + episode",
        "environment": "hard reset; property samplers rebuilt without accumulation",
    }, indent=2, default=str))
    if switch_rows:
        np.savez_compressed(
            out / "switch_horizon_l2.npz",
            horizon_l2=np.stack([s.pop("horizon_l2") for s in switch_rows]),
            setting=np.asarray([s["setting"] for s in switch_rows]),
            task=np.asarray([s["task"] for s in switch_rows]),
            episode=np.asarray([s["episode"] for s in switch_rows]),
            step=np.asarray([s["step"] for s in switch_rows]),
        )
    for fname, rows in (("summary.csv", summary), ("episodes.csv", episodes), ("switches.csv", switch_rows)):
        if rows:
            with open(out / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\nsaved to {out}")
    return summary


@parser.wrap()
def main(cfg: AsyncRolloutConfig):
    logging.basicConfig(level=logging.INFO)
    if cfg.policy is None or cfg.policy.type not in {"pi05", "smolvla"}:
        raise ValueError("Pass --policy.path pointing to a PI05 or SmolVLA checkpoint")
    if cfg.eval.batch_size != 1 or cfg.eval.use_async_envs:
        raise ValueError("Use --eval.batch_size=1 --eval.use_async_envs=false")
    # release: "off" (synchronous at R:0, naive async at R:k) and "trained" (ours); the baseline modes we compared
    # against (guided RTC, FutureRTC, VLASH) are not part of this release
    if cfg.rtc not in {"off", "trained"}:
        raise ValueError("--rtc must be off or trained")
    if cfg.rtc == "trained":
        if cfg.prefix_max < 1:
            raise ValueError("--rtc=trained needs --prefix_max >= 1")
        cfg.policy.rtc_config = RTCConfig(enabled=True, mode="trained")
        cfg.policy.rtc_training_max_delay = cfg.prefix_max
    if cfg.noise_mode not in {"random", "shift"}:
        raise ValueError("--noise_mode must be random or shift")
    if cfg.suffix_mask and (cfg.policy.type not in {"pi05", "smolvla"} or cfg.rtc != "off"):
        raise ValueError("--suffix_mask is for PI05 / SmolVLA with --rtc=off")
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=cfg.resume)
    set_seed(cfg.seed or 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.mem_fraction:
        if not 0 < cfg.mem_fraction <= 1:
            raise ValueError("--mem_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction)

    worker = None
    try:
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        policy.eval()
        H = policy.config.chunk_size
        settings = parse_settings(cfg.settings, H)
        if cfg.suffix_mask:
            if any(r + k > H - cfg.suffix_mask for _, r, k in settings):
                raise ValueError(f"--suffix_mask={cfg.suffix_mask}: every setting needs R + k <= {H - cfg.suffix_mask}")
            install_suffix_mask(policy, cfg.suffix_mask)
        pre, post = make_pre_post_processors(
            policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                    "rename_observations_processor": {"rename_map": cfg.rename_map}},
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        dt_ms = 1e3 / cfg.env.fps
        n_keep = max(r + k for _, r, k in settings)
        worker = ChunkWorker(policy, pre, post, env_post, n_keep=n_keep)
        worker.dt_ms = dt_ms
        worker.rtc = cfg.rtc == "trained"
        worker.vlash = False
        worker.prefix_max = cfg.prefix_max if cfg.rtc == "trained" else 0
        worker.device = policy.config.device
        worker.flip_video = True

        warmup_worker(worker, env_pre, cfg, cfg.warmup_inferences)

        episodes, switch_rows, lat_by_setting, task_ids = run_settings(
            cfg, settings, task_factory(cfg), env_pre, worker, H, out, dt_ms
        )

        write_outputs(cfg, settings, task_ids, H, dt_ms, out, episodes, switch_rows, lat_by_setting,
                      cfg.policy.pretrained_path)
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
