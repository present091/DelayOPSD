"""eval_flow.py's delay sweep, a few levels at a time: the methods and the (delay, execute horizon) pairs are chosen on
the command line, and every result is appended to <output_dir>/results.csv as it comes. The rollout is eval_flow.eval,
unchanged, with eval_flow.main's seed per level (jax.random.split(key(seed), 12)), so a row is the row eval_flow
would write. The newest <epoch>/ of --run-path is evaluated.

    uv run src/eval_sweep.py --run-path outputs/student_catapult --output-dir outputs/eval_catapult \
        --level-paths worlds/l/catapult.json --levels-per-call 1 --methods realtime --config.model.simulated-delay 5
"""

import csv
import dataclasses
import functools
import inspect
import pathlib
import pickle
import time
from typing import Literal, Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import tyro

import eval_flow
import model as _model
import train_expert

LEVEL_PATHS = tuple(inspect.signature(eval_flow.main).parameters["level_paths"].default)

# naive: plain switch to the new chunk (the start policy's asynchronous baseline; delay 0 = synchronous).
# realtime with --config.model.simulated-delay 5: the policy takes the running chunk's committed actions as a clean
# prefix (how the OPSD student is run).
METHODS = {
    "naive": eval_flow.NaiveMethodConfig(),
    "realtime": eval_flow.RealtimeMethodConfig(),
}


def pairs(sweep: str, chunk_size: int) -> list[tuple[int, int]]:
    """(inference delay, execute horizon) pairs: "delay" is the papers' delay plot (horizon max(d, 1)), "horizon"
    their horizon plot (delay 1), "full" everything eval_flow.main runs."""
    full = [(d, s) for d in range(5) for s in range(max(1, d), chunk_size - d + 1)]
    if sweep == "full":
        return full
    if sweep == "delay":
        return [(d, s) for d, s in full if s == max(1, d)]
    if sweep == "horizon":
        return [(d, s) for d, s in full if d == 1]
    raise ValueError(f"unknown sweep {sweep}")


def main(
    run_path: str,
    output_dir: str,
    config: eval_flow.EvalConfig = eval_flow.EvalConfig(),
    methods: Sequence[Literal["naive", "realtime"]] = ("realtime",),
    sweep: Literal["delay", "horizon", "full"] = "delay",
    level_paths: Sequence[str] = LEVEL_PATHS,
    # Levels in one call, over all GPUs.
    levels_per_call: int = 4,
    seed: int = 0,
    # only these inference delays of the sweep (e.g. --delays 0 --sweep full: sync at every execute horizon)
    delays: Sequence[int] = (0, 1, 2, 3, 4),
):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    log_dirs = sorted((p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()),
                      key=lambda p: int(p.name))
    state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
    state_dicts = jax.tree.map(lambda *x: jnp.array(x), *state_dicts)
    print(f"policies: {log_dirs[config.step]}")

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    assert levels_per_call % jax.local_device_count() == 0 and len(level_paths) % levels_per_call == 0, (
        f"{levels_per_call=} must be a multiple of the {jax.local_device_count()} GPUs and divide {len(level_paths)}"
    )

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _eval(config: eval_flow.EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict):
        policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config.model, rngs=nnx.Rngs(rng))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        eval_info, _ = eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params)
        return eval_info

    # The seeds of eval_flow.main with the default levels; a subset of the levels keeps each level's own seed.
    all_rngs = jax.random.split(jax.random.key(seed), len(LEVEL_PATHS))
    rngs = all_rngs[jnp.array([LEVEL_PATHS.index(p) for p in level_paths])]

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.csv"
    done = set()
    if path.is_file():
        with path.open() as f:
            done = {(r["method"], int(r["delay"]), int(r["execute_horizon"]), r["level"]) for r in csv.DictReader(f)}
    for delay, horizon in [p for p in pairs(sweep, config.model.action_chunk_size) if p[0] in delays]:
        for method in methods:
            c = dataclasses.replace(config, inference_delay=delay, execute_horizon=horizon, method=METHODS[method])
            start = time.perf_counter()
            for i in range(0, len(level_paths), levels_per_call):
                names = level_paths[i : i + levels_per_call]
                if all((method, delay, horizon, name) in done for name in names):
                    continue
                part = jax.tree.map(lambda x: x[i : i + levels_per_call], (rngs, levels, state_dicts))
                out = jax.device_get(_eval(c, *part))
                new = not path.is_file()
                with path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    if new:
                        writer.writerow([*out, "delay", "method", "level", "execute_horizon"])
                    for j, name in enumerate(names):
                        writer.writerow([*(float(v[j]) for v in out.values()), delay, method, name, horizon])
            print(f"{delay=} {horizon=} {method}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
