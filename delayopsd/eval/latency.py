"""Per-chunk inference latency of a LIBERO policy at batch 1, on the inference path delayopsd.eval.rollout
evaluates: ChunkWorker._infer (policy preprocessor + predict_action_chunk + postprocessor to env actions).

One real observation (a throwaway single-task env reset, as warmup_worker) is replayed: warm-up calls first, then
--n_iters timed calls, each CUDA-synchronized end to end. With --rtc=trained every timed call is a replan
with k = --delay (a hard prefix of min(k, prefix_max) actions from the previous chunk).
The model-only time (predict_action_chunk on an already preprocessed batch, CUDA events) is reported for --rtc=off /
trained. K_min = ceil(latency / period) at 20 Hz and 50 Hz, from the mean and from p90.

    python -m delayopsd.eval.latency --policy.path=lerobot/smolvla_libero \
      --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
      --env.observation_height=256 --env.observation_width=256 \
      --eval.batch_size=1 --eval.n_episodes=1 --eval.use_async_envs=false \
      --policy.compile_model=false --label=smolvla_base --output_dir=outputs/chunk_latency/smolvla_base_eager
"""

import json
import logging
import math
import platform
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from .rollout import AsyncRolloutConfig, ChunkWorker


@dataclass
class ChunkLatencyConfig(AsyncRolloutConfig):
    label: str = ""
    n_warmup: int = 20
    n_iters: int = 100
    # --rtc=trained: the delay k every timed replan carries (R = execution horizon for trained)
    delay: int = 10
    replan_interval: int = 10
    # merge a PEFT (LoRA) checkpoint into its base weights before timing (what a deployment would run)
    merge_lora: bool = False


def first_observation(cfg, env_pre):
    env_cfg = replace(cfg.env, task_ids=[(cfg.env.task_ids or [0])[0]])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
    try:
        env = next(iter(next(iter(envs.values())).values()))
        raw, _ = env.reset(seed=[cfg.seed or 0], options={NEW_ROLLOUT_OPTION: True})
        obs = preprocess_observation(raw)
        obs["task"] = list(env.call("task_description"))
        return env_pre(obs)
    finally:
        close_envs(envs)


def stats(ms):
    ms = np.asarray(ms)
    return {"mean": float(ms.mean()), "std": float(ms.std(ddof=1)), "median": float(np.median(ms)),
            "p90": float(np.percentile(ms, 90)), "min": float(ms.min()), "max": float(ms.max()), "n": len(ms)}


def k_min(ms):
    return {f"{hz}Hz": math.ceil(ms / (1e3 / hz)) for hz in (20, 50)}


@parser.wrap()
def main(cfg: ChunkLatencyConfig):
    logging.basicConfig(level=logging.INFO)
    if cfg.rtc not in {"off", "trained"}:
        raise ValueError("--rtc must be off or trained")
    if cfg.rtc == "trained":
        cfg.policy.rtc_config = RTCConfig(enabled=True, mode="trained")
        cfg.policy.rtc_training_max_delay = cfg.prefix_max
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # as async_rollout
    torch.manual_seed(cfg.seed or 0)

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    peft = hasattr(policy, "merge_and_unload")
    if cfg.merge_lora:
        if not peft:
            raise ValueError("--merge_lora needs a PEFT checkpoint")
        policy = policy.merge_and_unload()
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": cfg.rename_map}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    n_keep = cfg.replan_interval + cfg.delay
    worker = ChunkWorker(policy, pre, post, env_post, n_keep=n_keep)
    worker.rtc = cfg.rtc == "trained"
    worker.prefix_max = cfg.prefix_max if cfg.rtc == "trained" else 0

    obs = first_observation(cfg, env_pre)
    chunk, env_actions, _ = worker._infer(obs)  # the episode's first chunk
    replan = None
    if cfg.rtc == "trained":
        d = min(cfg.delay, cfg.prefix_max)
        replan = {"inference_delay": d, "execution_horizon": cfg.replan_interval,
                  "prev_chunk_left_over": torch.from_numpy(chunk[cfg.replan_interval:cfg.replan_interval + d])[None]
                  .to(policy.config.device)}

    def timed():
        torch.cuda.synchronize()
        return worker._infer(obs, None, replan)[2]  # perf_counter around the call; it ends on a .cpu() copy

    for _ in range(cfg.n_warmup):
        timed()
    e2e = [timed() for _ in range(cfg.n_iters)]

    batch = pre(obs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ms = []
    with (torch.no_grad() if worker.rtc else torch.inference_mode()):
        for i in range(cfg.n_warmup + cfg.n_iters):
            torch.cuda.synchronize()
            start.record()
            policy.predict_action_chunk(batch, **(replan or {}))
            end.record()
            torch.cuda.synchronize()
            if i >= cfg.n_warmup:
                ms.append(start.elapsed_time(end))
    model_only = stats(ms)

    dtypes = Counter(str(p.dtype).removeprefix("torch.") for p in policy.parameters())
    numel = Counter()
    for p in policy.parameters():
        numel[str(p.dtype).removeprefix("torch.")] += p.numel()
    config = policy.config
    result = {
        "label": cfg.label, "policy_path": str(cfg.policy.pretrained_path), "policy_type": config.type,
        "rtc": cfg.rtc, "delay": cfg.delay if cfg.rtc != "off" else 0,
        "prefix": min(cfg.delay, cfg.prefix_max) if cfg.rtc == "trained" else 0,
        "peft": peft, "merged_lora": cfg.merge_lora,
        "compile_model": bool(getattr(config, "compile_model", False)),
        "compile_mode": getattr(config, "compile_mode", None) if getattr(config, "compile_model", False) else None,
        "param_dtypes": {k: {"tensors": dtypes[k], "params": numel[k]} for k in dtypes},
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "num_denoise_steps": getattr(config, "num_steps", getattr(config, "num_inference_steps", None)),
        "chunk_size": config.chunk_size,
        "batch_size": 1, "n_warmup": cfg.n_warmup, "n_iters": cfg.n_iters,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "host": platform.node(),
        "e2e_ms": stats(e2e), "model_ms": model_only,
        "k_min_mean": k_min(stats(e2e)["mean"]), "k_min_p90": k_min(stats(e2e)["p90"]),
    }
    (out / "latency.json").write_text(json.dumps(result, indent=2))
    s = result["e2e_ms"]
    model = "-" if model_only is None else f"{model_only['mean']:.1f} ± {model_only['std']:.1f}"
    print(f"{cfg.label}: e2e {s['mean']:.1f} ± {s['std']:.1f} ms (p90 {s['p90']:.1f}) | model {model} ms | "
          f"K_min {result['k_min_mean']} | {result['gpu']} | {dict(dtypes)} | compile {result['compile_model']}",
          flush=True)


if __name__ == "__main__":
    main()
