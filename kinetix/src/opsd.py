"""On-policy self-distillation (OPSD) for the Kinetix flow policies of Real-Time Chunking.

A pair is one replan of an asynchronous rollout of the base policy with inference delay d:

    student obs = the observation at step t (what the policy saw when it started inferring)
    hard prefix = the d actions of the running chunk that execute over [t, t + d)
    teacher obs = the observation at step t + d (what those actions led to)

The student is the base policy with a clean hard prefix (training-time RTC's input: the prefix tokens at flow time
1). It is trained to give, on its tokens [d, H), the velocity a frozen teacher gives on tokens [0, H - d) from the
newer observation, on the student's own flow state shifted by d:

    L = || v_student(obs_t, x_tau, tau)[d:H] - sg v_teacher(obs_{t+d}, shift_d(x_tau), tau)[0:H-d] ||^2

x_tau is a state of the student's own Euler sampling on obs_t with that prefix. The teacher is the stage-1 model:
the base policy finetuned on the expert data with flow matching plus masked flow matching, so that it handles a
chunk whose last d tokens are missing. A missing token is a zero at flow time 0 (the MLP-Mixer mixes tokens with a
dense layer of fixed size, so it stays as an input that carries nothing) and is left out of the loss.

Failed episodes keep their pairs while the step is before 1.5 x the median length of the level's successful episodes
(success = a goal contact) and before the episode stalls (every body inside a 0.2 box until the end).

Flow time runs from 0 (noise) to 1 (data), as in the Real-Time Chunking code.
"""

import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers

import model as _model
import train_expert

MAX_DELAY = 4  # the delays 0..4 of eval_flow's sweep


def setup(level_paths: Sequence[str]):
    """(env, env_params, static_env_params, levels stacked over level_paths, obs_dim, action_dim), as eval_flow."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    return env, env_params, static_env_params, levels, obs_dim, env.action_space(env_params).shape[0]


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def shard(level_paths: Sequence[str], spec: str) -> list[str]:
    """"i/n": every n-th level from the i-th on, so n processes (one per GPU) cover the levels."""
    i, n = (int(x) for x in spec.split("/"))
    return list(level_paths)[i::n]


def load_policies(policy_dir: str, level_paths: Sequence[str]):
    """The policies of <policy_dir>/policies/<level>.pkl, stacked over the levels."""
    state_dicts = []
    for level_path in level_paths:
        with (pathlib.Path(policy_dir) / "policies" / f"{level_name(level_path)}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
    return jax.tree.map(lambda *x: jnp.array(x), *state_dicts)


def ema_update(ema, policy, decay: float):
    """EMA of the policy's parameters (an nnx.State of nnx.Param), one step."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema, nnx.state(policy, nnx.Param))


def save_policies(policy_dir: pathlib.Path, level_paths: Sequence[str], graphdef, train_state, ema=None):
    """Write the policies of a (policy, optimizer) train state stacked over the levels, as train_flow does; with
    `ema` (stacked like the train state), the EMA parameters replace the policy's."""
    (policy_dir / "policies").mkdir(parents=True, exist_ok=True)
    for i, level_path in enumerate(level_paths):
        policy, _ = nnx.merge(graphdef, jax.tree.map(lambda x: x[i], train_state))
        if ema is not None:
            nnx.update(policy, jax.tree.map(lambda x: x[i], ema))
        with (policy_dir / "policies" / f"{level_name(level_path)}.pkl").open("wb") as f:
            pickle.dump(nnx.state(policy).to_pure_dict(), f)


def make_policy(state_dict, config: _model.ModelConfig, obs_dim: int, action_dim: int) -> _model.FlowPolicy:
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def prefix_mask(delay: jax.Array, horizon: int) -> jax.Array:
    """(B,) delays -> (B, H) bool, True on the hard prefix tokens [0, d)."""
    return jnp.arange(horizon)[None, :] < delay[:, None]


def suffix_valid(delay: jax.Array, horizon: int) -> jax.Array:
    """(B,) delays -> (B, H) bool, True on the tokens [0, H - d) a chunk shifted by d still has."""
    return jnp.arange(horizon)[None, :] < horizon - delay[:, None]


def shift_tail(x: jax.Array, delay: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The student's tokens [d, H) at the teacher's positions [0, H - d); zeros beyond."""
    horizon = x.shape[1]
    index = jnp.minimum(jnp.arange(horizon)[None, :] + delay[:, None], horizon - 1)
    valid = suffix_valid(delay, horizon)
    return jnp.where(valid[:, :, None], jnp.take_along_axis(x, index[:, :, None], axis=1), 0.0), valid


def masked_velocity(policy: _model.FlowPolicy, obs: jax.Array, x: jax.Array, time: jax.Array, valid: jax.Array):
    """Velocity with the tokens outside `valid` missing: zeros at flow time 0."""
    time = jnp.broadcast_to(time[:, None] if time.ndim == 1 else time, valid.shape)
    return policy(obs, jnp.where(valid[:, :, None], x, 0.0), jnp.where(valid, time, 0.0))


def stage1_loss(policy: _model.FlowPolicy, rng: jax.Array, obs: jax.Array, action: jax.Array):
    """Flow matching + masked flow matching (the last d ~ U{1..MAX_DELAY} tokens missing) on one x_t: the sum of the
    full loss and the masked one, the latter averaged over the valid tokens of the batch."""
    noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
    time = jax.random.uniform(time_rng, (obs.shape[0],))
    noise = jax.random.normal(noise_rng, shape=action.shape)
    u_t = action - noise
    x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action
    full = jnp.mean(jnp.square(policy(obs, x_t, time) - u_t))
    delay = jax.random.randint(delay_rng, (obs.shape[0],), 1, MAX_DELAY + 1)
    valid = suffix_valid(delay, action.shape[1])
    errors = jnp.square(masked_velocity(policy, obs, x_t, time, valid) - u_t) * valid[:, :, None]
    masked = jnp.sum(errors) / (jnp.sum(valid) * action.shape[-1])
    return full + masked, {"fm": full, "masked_fm": masked}


def sample_states(policy: _model.FlowPolicy, rng: jax.Array, obs: jax.Array, prefix: jax.Array, delay: jax.Array,
                  num_steps: int):
    """Euler sampling with the clean hard prefix (FlowPolicy.realtime_action with simulated_delay), per-example
    delays; returns the state before every step, (S, B, H, D), each holding the prefix."""
    dt = 1 / num_steps
    mask = prefix_mask(delay, policy.action_chunk_size)

    def step(carry, _):
        x_t, time = carry
        x_t = jnp.where(mask[:, :, None], prefix, x_t)
        v_t = policy(obs, x_t, jnp.where(mask, 1.0, time))
        return (x_t + dt * v_t, time + dt), x_t

    noise = jax.random.normal(rng, shape=prefix.shape)
    _, states = jax.lax.scan(step, (noise, 0.0), length=num_steps)
    return states


def opsd_loss(student: _model.FlowPolicy, teacher: _model.FlowPolicy, rng: jax.Array, obs: jax.Array,
              teacher_obs: jax.Array, prefix: jax.Array, delay: jax.Array, trust_scale: float, num_steps: int):
    """The OPSD loss at a random state x_tau of the student's own sampling; tokens weighted by exp(-r / trust_scale),
    r the position in the teacher's chunk (trust_scale 0: uniform)."""
    state_rng, index_rng = jax.random.split(rng)
    batch, horizon = prefix.shape[:2]
    index = jax.random.randint(index_rng, (batch,), 0, num_steps)
    tau = index / num_steps
    states = jax.lax.stop_gradient(sample_states(student, state_rng, obs, prefix, delay, num_steps))
    x = states[index, jnp.arange(batch)]
    teacher_x, valid = shift_tail(x, delay)
    target = jax.lax.stop_gradient(masked_velocity(teacher, teacher_obs, teacher_x, tau, valid))
    pred = student(obs, x, jnp.where(prefix_mask(delay, horizon), 1.0, tau[:, None]))
    aligned, _ = shift_tail(pred, delay)
    errors = jnp.mean(jnp.square(aligned - target), axis=-1)
    weight = valid.astype(jnp.float32)
    if trust_scale > 0:
        weight = weight * jnp.exp(-jnp.arange(horizon)[None, :] / trust_scale)
    return jnp.sum(errors * weight) / jnp.sum(weight)


def kinetix_state(env_state):
    """The Kinetix EnvState inside the wrappers' states (the one with the rigid bodies)."""
    while not hasattr(env_state, "polygon"):
        env_state = env_state.env_state
    return env_state


def stall_onset(pos: jax.Array, length: jax.Array, box: float) -> jax.Array:
    """(E,) stall onsets: the earliest step t of the first episode from which every body stays in a box of side `box`
    until the episode ends (per-coordinate extent max - min over [t, length) <= box); `length` when none does.
    pos (T, E, B, 2), step t = the state before action t; bodies that never move do not matter (extent 0).
    """
    T = pos.shape[0]
    live = jnp.arange(T)[:, None] < length[None, :]
    hi = jax.lax.cummax(jnp.where(live[:, :, None, None], pos, -jnp.inf), axis=0, reverse=True)
    lo = jax.lax.cummin(jnp.where(live[:, :, None, None], pos, jnp.inf), axis=0, reverse=True)
    extent = jnp.max(jnp.where(live[:, :, None, None], hi - lo, 0.0), axis=(2, 3))
    still = (extent <= box) & live
    onset = jnp.where(jnp.any(still, axis=0), jnp.argmax(still, axis=0), length)
    return jnp.minimum(onset, length)


def collect_pairs(env, env_params, rng: jax.Array, level, policy: _model.FlowPolicy, delay: int, num_envs: int,
                  num_pairs: int, num_flow_steps: int, fail_keep_success_quantile: float = 0.5,
                  fail_keep_success_scale: float = 1.5, fail_tail_cut: float = 0.6, fail_stall_box: float = 0.2,
                  fail_stall_margin: int = 0):
    """Pairs from `num_envs` naive-async rollouts of `policy` at inference delay `delay` and execute horizon
    max(delay, 1) (eval_flow.eval's schedule and wrappers), the first episode of every rollout only.

    Returns (pairs, stats): pairs = obs, teacher_obs (P, obs_dim), prefix (P, H, action_dim; zeros from `delay`
    on), ok (P,) -- a random subset of `num_pairs` of the kept pairs, the kept ones first (ok False pads a level
    with fewer).
    Failed-pair rule: an episode succeeds when it ends with reward +1 (a goal contact on any frame-skip substep). A
    failed episode keeps its pairs with step < fail_keep_success_scale x the fail_keep_success_quantile-quantile of
    the successful episodes' lengths (a fraction 1 - fail_tail_cut of its own length when none succeeds) and
    step < its stall onset + fail_stall_margin (stall_onset over the level's dynamic bodies, box side fail_stall_box).
    """
    horizon = max(delay, 1)
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), num_envs
    )

    def execute_chunk(carry, chunk_index):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            ks = kinetix_state(env_state)
            pos = jnp.concatenate([ks.polygon.position, ks.circle.position], axis=1)
            out = (next_obs, done, info["returned_episode_solved"], info["returned_episode_lengths"], reward, pos)
            return (rng, next_obs, next_env_state), out

        rng, obs, env_state, action_chunk, ended, solved, length, final_reward = carry
        rng, key = jax.random.split(rng)
        next_action_chunk = policy.action(key, obs, num_flow_steps)
        to_execute = jnp.concatenate([action_chunk[:, :delay], next_action_chunk[:, delay:horizon]], axis=1)
        (rng, next_obs, next_env_state), (obs_seq, dones, step_solved, step_length, step_reward, pos) = jax.lax.scan(
            step, (rng, obs, env_state), to_execute.transpose(1, 0, 2)
        )
        # the frame the prefix leads to, which must still be the same episode
        teacher_obs = obs if delay == 0 else obs_seq[delay - 1]
        valid = jnp.logical_not(ended)
        if delay > 0:
            valid = valid & jnp.logical_not(jnp.any(dones[:delay], axis=0))
        prefix = jnp.where(jnp.arange(action_chunk.shape[1])[None, :, None] < delay, action_chunk, 0.0)
        # outcome of the first episode
        first = jnp.argmax(dones, axis=0)
        finishes = jnp.any(dones, axis=0) & jnp.logical_not(ended)
        solved = jnp.where(finishes, step_solved[first, jnp.arange(num_envs)] > 0, solved)
        length = jnp.where(finishes, step_length[first, jnp.arange(num_envs)], length)
        final_reward = jnp.where(finishes, step_reward[first, jnp.arange(num_envs)], final_reward)
        next_action_chunk = jnp.concatenate(
            [next_action_chunk[:, horizon:], jnp.zeros((num_envs, horizon, policy.action_dim))], axis=1
        )
        carry = (rng, next_obs, next_env_state, next_action_chunk, ended | jnp.any(dones, axis=0), solved, length,
                 final_reward)
        return carry, (obs, teacher_obs, prefix, valid, jnp.full((num_envs,), chunk_index * horizon), pos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, num_flow_steps)
    num_chunks = -(-env_params.max_timesteps // horizon)
    init = (rng, obs, env_state, action_chunk, jnp.zeros(num_envs, bool), jnp.zeros(num_envs, bool),
            jnp.full((num_envs,), env_params.max_timesteps), jnp.zeros(num_envs))
    (rng, *_, ended, solved, length, final_reward), (obs, teacher_obs, prefix, valid, step, pos) = jax.lax.scan(
        execute_chunk, init, jnp.arange(num_chunks)
    )
    # failed episodes: keep the pairs before the rule's limit
    success = final_reward > 0
    limit = jnp.where(
        jnp.any(success),
        fail_keep_success_scale * jnp.nanquantile(jnp.where(success, length, jnp.nan), fail_keep_success_quantile),
        length * (1.0 - fail_tail_cut),
    )
    dynamic = jnp.concatenate([level.polygon.active & (level.polygon.inverse_mass > 0),
                               level.circle.active & (level.circle.inverse_mass > 0)])
    # (chunks, horizon, E, B, 2) -> (T, E, B, 2); the bodies that cannot move held at 0
    pos = pos.reshape(-1, *pos.shape[2:])[: env_params.max_timesteps]
    onset = stall_onset(jnp.where(dynamic[None, None, :, None], pos, 0.0), length, fail_stall_box)
    limit = jnp.minimum(limit, onset + fail_stall_margin)
    keep = valid & (success[None, :] | (step < limit))
    rng, key = jax.random.split(rng)
    score = jnp.where(keep, jax.random.uniform(key, keep.shape), -1.0).reshape(-1)
    _, picks = jax.lax.top_k(score, num_pairs)
    flat = lambda x: x.reshape(-1, *x.shape[2:])[picks]  # noqa: E731
    pairs = {"obs": flat(obs), "teacher_obs": flat(teacher_obs), "prefix": flat(prefix), "ok": flat(keep)}
    stats = {"solved": jnp.mean(solved), "goal": jnp.mean(success), "length": jnp.mean(length),
             "pairs_valid": jnp.sum(valid), "pairs_kept": jnp.sum(keep),
             "pairs_failed_kept": jnp.sum(keep & jnp.logical_not(success[None, :])), "fail_limit": jnp.mean(limit)}
    return pairs, stats
