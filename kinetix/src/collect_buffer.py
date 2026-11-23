"""OPSD rollout buffer: pairs from naive-async rollouts of the base policies, one set per inference delay 0..4
(opsd.collect_pairs, with its failed-pair rule). Writes <output_dir>/<level>.npz with obs, teacher_obs, prefix, ok
stacked over the delays, and <output_dir>/stats_<shard>.csv.

The paper's buffer per level: 16 of these (seeds 2000..2015), each 2,048 rollouts and 131,072 kept pairs per delay.

    uv run src/collect_buffer.py --policy-dir assets/bc/24 --output-dir outputs/buffers/s2000 \
        --level-paths worlds/l/catapult.json --levels-per-call 1 --seed 2000
"""

import csv
import dataclasses
import functools
import pathlib
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import eval_sweep
import model as _model
import opsd


@dataclasses.dataclass(frozen=True)
class Config:
    policy_dir: str  # holds policies/<level>.pkl: the buffer policy
    output_dir: str
    level_paths: Sequence[str] = eval_sweep.LEVEL_PATHS
    shard: str = "0/1"
    num_envs: int = 2048
    # Pairs kept per level and delay (a random subset of the rollouts' pairs).
    num_pairs: int = 131072
    # The delays to collect; the npz stacks them in this order and records them as `delays`.
    delays: Sequence[int] = (0, 1, 2, 3, 4)
    num_flow_steps: int = 5
    # the failed-pair rule (opsd.collect_pairs)
    fail_keep_success_quantile: float = 0.5
    fail_keep_success_scale: float = 1.5
    fail_tail_cut: float = 0.6
    fail_stall_box: float = 0.2
    fail_stall_margin: int = 0
    # Levels in one call: the delay-0 rollouts hold 256 chunks x num_envs observations per level (2.8 GB).
    levels_per_call: int = 2
    seed: int = 2000
    model: _model.ModelConfig = _model.ModelConfig()


def main(config: Config):
    level_paths = opsd.shard(config.level_paths, config.shard)
    env, env_params, _, levels, obs_dim, action_dim = opsd.setup(level_paths)
    state_dicts = opsd.load_policies(config.policy_dir, level_paths)
    # one stream per level, whatever the shard
    all_rngs = jax.random.split(jax.random.key(config.seed), len(config.level_paths))
    rngs = all_rngs[jnp.array([list(config.level_paths).index(p) for p in level_paths])]

    @functools.partial(jax.jit, static_argnums=(0,))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def collect(delay: int, rng: jax.Array, level, state_dict):
        policy = opsd.make_policy(state_dict, config.model, obs_dim, action_dim)
        return opsd.collect_pairs(
            env, env_params, jax.random.fold_in(rng, delay), level, policy, delay, config.num_envs, config.num_pairs,
            config.num_flow_steps, config.fail_keep_success_quantile, config.fail_keep_success_scale,
            config.fail_tail_cut, config.fail_stall_box, config.fail_stall_margin,
        )

    out_dir = pathlib.Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {p: [] for p in level_paths}
    rows = []
    for delay in config.delays:
        start = time.perf_counter()
        for i in range(0, len(level_paths), config.levels_per_call):
            part = jax.tree.map(lambda x: x[i : i + config.levels_per_call], (rngs, levels, state_dicts))
            out, stats = jax.device_get(collect(delay, *part))
            for j, level_path in enumerate(level_paths[i : i + config.levels_per_call]):
                pairs[level_path].append({k: v[j] for k, v in out.items()})
                rows.append({"level": level_path, "delay": delay, **{k: float(v[j]) for k, v in stats.items()}})
        print(f"delay {delay}: {time.perf_counter() - start:.0f} s", flush=True)
    for level_path, per_delay in pairs.items():
        np.savez(out_dir / f"{opsd.level_name(level_path)}.npz", delays=np.asarray(config.delays),
                 **{k: np.stack([p[k] for p in per_delay]) for k in per_delay[0]})
    with (out_dir / f"stats_{config.shard.replace('/', 'of')}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(tyro.cli(Config))
