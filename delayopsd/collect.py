"""Collect a rollout buffer for OPSD: delayopsd.eval.rollout episodes, saved step by step.

Runs delayopsd.eval.rollout unchanged (same scenes, seeding, inference and RTC path)
and records, per episode, one compressed npz in <output_dir>/episodes/:

    obs/<path>     (F, ...)    raw env observation leaves (batch dim dropped) at the steps in obs_step: obs t is
                               what the policy saw before executing action t, obs T the state after the last one
    obs_step       (F,)        the steps whose observation is stored. Default: the frames an OPSD pair can use at
                               the collection's R:k -- every replan step s, s + k and s + R, and the final frame T
                               (at 10:10 every 10th frame, a tenth of the episode). --frames=all keeps all T + 1
                               (obs_step = 0..T).
    action         (T, A)      executed env-space action
    chunks         (C, H, D)   normalized model-space chunk of every inference (what --rtc=trained takes
                               its hard prefix from)
    chunk_obs_step (C,)        step whose observation produced each chunk (it takes over k steps later)
    meta           json        task, seed, success, steps, R, k, scene fingerprints

Example (scenes held out from the evaluation; scripts/pi05_libero/collect_buffers.sh runs all of them):
    python -m delayopsd.collect --policy.path=checkpoints/pi05_libero_finetuned --env.type=libero \
        --env.task=libero_10 --env.init_states=false --eval.n_episodes=40 --seed=2000 \
        --settings='["10:10"]' --noise_mode=random --output_dir=data/buffers/pi05_libero/10_10/libero_10
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import delayopsd.eval.rollout as ar

_original = ar.run_episode
# zlib releases the GIL: compress/write an episode while the next one runs.
_writer = ThreadPoolExecutor(max_workers=2)
_pending = []


def _leaves(node, path=""):
    if isinstance(node, dict):
        for key in sorted(node):
            yield from _leaves(node[key], f"{path}/{key}" if path else key)
    else:
        yield path, np.asarray(node)[0]


def run_episode(env, env_pre, worker, R, k, H, seed, max_steps, *args, **kwargs):
    obs, actions, chunks = [], [], []
    step_count = [0]
    reset, step, submit = env.reset, env.step, worker.submit

    def reset_hook(*a, **kw):
        raw, info = reset(*a, **kw)
        obs.append(dict(_leaves(raw)))
        return raw, info

    def step_hook(action):
        result = step(action)
        actions.append(np.asarray(action)[0].copy())
        obs.append(dict(_leaves(result[0])))
        step_count[0] += 1
        return result

    def submit_hook(o, noise=None, rtc=None):
        future = submit(o, noise, rtc)
        chunks.append((step_count[0], future))
        return future

    env.reset, env.step, worker.submit = reset_hook, step_hook, submit_hook
    try:
        stats, switches, latencies = _original(env, env_pre, worker, R, k, H, seed, max_steps, *args, **kwargs)
    finally:
        del env.reset, env.step, worker.submit

    if len(obs) != len(actions) + 1:
        raise RuntimeError(f"seed {seed}: {len(obs)} observations for {len(actions)} actions")
    task = env.call("task_description")[0]
    group_task = kwargs.get("episode_key") or (None, None)
    meta = {k_: v for k_, v in stats.items() if isinstance(v, (str, int, float, bool))}
    meta.update(seed=int(seed), R=int(R), k=int(k), H=int(H), task_description=task,
                group=group_task[0], task_id=group_task[1])
    T, replans = len(actions), [s for s, _ in chunks]
    if run_episode.all_frames:
        keep = list(range(T + 1))
    else:  # student frame s, and the teacher frame of a pair with delta == k (async) or delta == R (sync buffer)
        keep = sorted({t for s in replans for t in (s, s + k, s + R) if t <= T} | {T})
    arrays = {f"obs/{key}": np.stack([obs[t][key] for t in keep]) for key in obs[0]}
    arrays["obs_step"] = np.array(keep, dtype=np.int32)
    arrays["action"] = np.stack(actions)
    arrays["chunks"] = np.stack([f.result()[0] for _, f in chunks])
    arrays["chunk_obs_step"] = np.array(replans, dtype=np.int32)
    out = Path(run_episode.out_dir)
    name = f"{meta['group']}_{meta['task_id']}_seed{seed}_{'S' if stats['success'] else 'F'}.npz"
    _pending.append(_writer.submit(np.savez_compressed, out / name, meta=np.array(json.dumps(meta)), **arrays))
    for f in [f for f in _pending if f.done()]:
        _pending.remove(f)
        f.result()  # surface write errors
    return stats, switches, latencies


def main():
    import sys

    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--output_dir=")), None)
    if out is None:
        raise SystemExit("--output_dir=<dir> is required")
    run_episode.out_dir = Path(out).resolve() / "episodes"
    frames = [a for a in sys.argv if a.startswith("--frames=")]  # ours, not one of async_rollout's arguments
    for a in frames:
        sys.argv.remove(a)
    mode = frames[-1].split("=", 1)[1] if frames else "replan"
    if mode not in {"replan", "all"}:
        raise SystemExit("--frames must be replan or all")
    run_episode.all_frames = mode == "all"
    run_episode.out_dir.mkdir(parents=True, exist_ok=True)
    ar.run_episode = run_episode
    try:
        ar.main()
    finally:
        for f in _pending:
            f.result()
        _writer.shutdown()


if __name__ == "__main__":
    main()
