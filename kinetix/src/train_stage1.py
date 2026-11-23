"""OPSD stage 1: finetune the base policies on the expert data with flow matching + masked flow matching
(opsd.stage1_loss). The result is the frozen teacher of train_opsd.py.

train_flow.py's data handling and optimiser, without its per-epoch evaluation and wandb. One process per GPU:
`--shard i/n` takes every n-th level from the i-th on.

    uv run src/train_stage1.py --run-path assets/expert --load-dir assets/bc/24 --output-dir outputs/teacher \
        --shard 0/4
"""

import dataclasses
import pathlib
import time
from typing import Sequence

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import eval_sweep
import model as _model
import opsd


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str  # holds data/<level>.npz (generate_data.py)
    load_dir: str  # holds policies/<level>.pkl: the base policies
    output_dir: str
    level_paths: Sequence[str] = eval_sweep.LEVEL_PATHS
    shard: str = "0/1"
    batch_size: int = 512
    num_epochs: int = 8
    seed: int = 0
    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    # EMA of the parameters, updated every step; the saved policies (<epoch>/) are the EMA, the raw ones go to
    # raw/<epoch>/. 0 = off (the raw policies are saved as <epoch>/).
    ema_decay: float = 0.999
    model: _model.ModelConfig = _model.ModelConfig()


def make_optimizer(policy: _model.FlowPolicy, config) -> nnx.Optimizer:
    return nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(config.grad_norm_clip),
            optax.adamw(
                optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps),
                weight_decay=config.weight_decay,
            ),
        ),
    )


def main(config: Config):
    level_paths = opsd.shard(config.level_paths, config.shard)
    _, _, _, _, obs_dim, action_dim = opsd.setup(level_paths)
    chunk = config.model.action_chunk_size

    data = []
    for level_path in level_paths:
        print("loading", level_path, flush=True)
        with np.load(pathlib.Path(config.run_path) / "data" / f"{opsd.level_name(level_path)}.npz") as z:
            # (steps, envs, ...) -> (envs steps, ...), as train_flow
            data.append({k: einops.rearrange(z[k], "s e ... -> (e s) ...") for k in ("obs", "action", "done")})
    valid_steps = data[0]["obs"].shape[0] - chunk + 1
    num_batches = valid_steps // config.batch_size
    # stacked on the host and copied once: a device-side stack would hold the levels twice (2.8 GB each)
    data = jax.device_put({k: np.stack([d[k][: num_batches * config.batch_size + chunk - 1] for d in data]) for k in data[0]})
    print(f"{len(level_paths)} levels, {num_batches:_} batches per epoch", flush=True)

    first = opsd.make_policy(
        jax.tree.map(lambda x: x[0], opsd.load_policies(config.load_dir, level_paths)), config.model, obs_dim, action_dim
    )
    graphdef = nnx.graphdef((first, make_optimizer(first, config)))

    @jax.jit
    @jax.vmap
    def init(state_dict):
        policy = opsd.make_policy(state_dict, config.model, obs_dim, action_dim)
        return nnx.state((policy, make_optimizer(policy, config)))

    @jax.jit
    @jax.vmap
    def init_ema(train_state):
        policy, _ = nnx.merge(graphdef, train_state)
        return nnx.state(policy, nnx.Param)

    @jax.jit
    @jax.vmap
    def train_epoch(rng: jax.Array, train_state: nnx.State, ema, data: dict):
        def train_minibatch(carry, batch_idxs):
            rng, train_state, ema = carry
            policy, optimizer = nnx.merge(graphdef, train_state)
            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model.FlowPolicy):
                window = batch_idxs[:, None] + jnp.arange(chunk)[None, :]
                action_chunks = data["action"][window]
                # zero actions after done
                done_chunks = data["done"][window]
                done_idxs = jnp.where(jnp.any(done_chunks, axis=-1), jnp.argmax(done_chunks, axis=-1), chunk)
                action_chunks = jnp.where(jnp.arange(chunk)[None, :, None] >= done_idxs[:, None, None], 0.0, action_chunks)
                return opsd.stage1_loss(policy, key, data["obs"][batch_idxs], action_chunks)

            (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
            optimizer.update(grads)
            ema = opsd.ema_update(ema, policy, config.ema_decay) if config.ema_decay > 0 else ema
            return (rng, nnx.state((policy, optimizer)), ema), {"loss": loss, **info}

        rng, key = jax.random.split(rng)
        permutation = jax.random.permutation(key, num_batches * config.batch_size).reshape(-1, config.batch_size)
        (rng, train_state, ema), info = jax.lax.scan(train_minibatch, (rng, train_state, ema), permutation)
        return rng, train_state, ema, jax.tree.map(jnp.mean, info)

    train_state = init(opsd.load_policies(config.load_dir, level_paths))
    ema = init_ema(train_state)
    rngs = jax.random.split(jax.random.key(config.seed), len(level_paths))
    out = pathlib.Path(config.output_dir)
    for epoch in range(config.num_epochs):
        start = time.perf_counter()
        rngs, train_state, ema, info = train_epoch(rngs, train_state, ema, data)
        if config.ema_decay > 0:
            opsd.save_policies(out / "raw" / str(epoch), level_paths, graphdef, train_state)
            opsd.save_policies(out / str(epoch), level_paths, graphdef, train_state, ema)
        else:
            opsd.save_policies(out / str(epoch), level_paths, graphdef, train_state)
        info = {k: np.asarray(v) for k, v in info.items()}
        print(f"epoch {epoch}: {time.perf_counter() - start:.0f} s  " +
              "  ".join(f"{k} {v.mean():.4f}" for k, v in info.items()), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
