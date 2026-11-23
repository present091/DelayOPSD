"""Versioned initial conditions of the LIBERO evaluation, and the fingerprints that pair episodes across runs."""
import hashlib
import json
import logging

import numpy as np

LIBERO_PROTOCOL_VERSION = "libero_fresh_env_v3"
# LIBERO with init_states=false: objects are placed by LIBERO's own sampler from the global NumPy stream, which
# run_episode seeds with the episode seed right before reset. Scenes are therefore new draws from the placement
# distribution, disjoint from the 50 recorded benchmark states, and identical across checkpoints for a given seed.
LIBERO_RANDOM_PROTOCOL_VERSION = "libero_random_scene_v1"
PROTOCOL_VERSION = LIBERO_PROTOCOL_VERSION


def prepare_reset(env, index):
    """Pin the recorded initial state for this episode.

    LIBERO's placement sampler accumulation across hard resets is *not* patched out here. It is the
    upstream behaviour that the published evaluators run under, so the scene for episode k is drawn
    after k samplers have accumulated, exactly as in a reference run. What makes that reproducible
    is that ``run_settings`` gives every setting a freshly built env, so episode k always sees
    exactly k prior resets no matter which setting it is, or where the checkpoint sat in a queue."""
    subs = getattr(env, 'envs', None)
    if subs is None or len(subs) != 1:
        raise ValueError('Evaluation requires one synchronous sub-environment')
    inner = subs[0].unwrapped
    if inner.__class__.__module__.endswith('.libero') and not getattr(inner, 'init_states', True):
        if not getattr(inner, 'hard_reset', False):
            raise ValueError('Random-scene evaluation requires LIBERO hard_reset=true')
        return LIBERO_RANDOM_PROTOCOL_VERSION
    if not getattr(inner, 'init_states', False) or not getattr(inner, 'hard_reset', False):
        raise ValueError('Pinned evaluation requires LIBERO init_states=true and hard_reset=true')
    if not 0 <= index < len(inner._init_states):
        raise ValueError(f'Initial state {index} outside recorded states (no modulo wrapping permitted)')
    inner._ensure_env()
    inner.init_state_id = index
    return LIBERO_PROTOCOL_VERSION


def protocol_version(env):
    """Return the protocol without mutating the environment."""
    subs = getattr(env, 'envs', None)
    if subs is None or len(subs) != 1:
        raise ValueError('Evaluation requires one synchronous sub-environment')
    inner = subs[0].unwrapped
    if getattr(inner, 'init_states', False):
        return LIBERO_PROTOCOL_VERSION
    if inner.__class__.__module__.endswith('.libero'):
        return LIBERO_RANDOM_PROTOCOL_VERSION
    raise ValueError(f'No evaluation protocol for {inner.__class__.__module__}.{type(inner).__name__}')


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([str(a.dtype), a.shape]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def observation_hash(raw):
    leaves = {}
    def visit(node, path):
        if isinstance(node, dict):
            for key in sorted(node):
                visit(node[key], path + '/' + key)
        else:
            leaves[path] = array_hash(node)
    visit(raw, '')
    return hashlib.sha256(json.dumps(leaves, sort_keys=True).encode()).hexdigest()


def fingerprints(env, raw, noise_field):
    # Do not suppress errors: a missing fingerprint must never count as a match.
    backend = env.call('_env')[0]
    domain = getattr(backend, 'env', backend)
    sim = getattr(backend, 'sim', None)
    if sim is None:
        sim = getattr(domain, 'sim', None)
    if sim is None:
        raise RuntimeError(f"Cannot locate MuJoCo sim in {type(backend).__name__}")
    state_hash = array_hash(sim.get_state().flatten())
    fixture_hash = hashlib.sha256(
        (array_hash(sim.model.body_pos) + array_hash(sim.model.body_quat)).encode()).hexdigest()
    initializers = getattr(domain, "object_property_initializers", None)
    obs_hash = observation_hash(raw)
    return {
        'scene_hash': hashlib.sha256((state_hash + fixture_hash + obs_hash).encode()).hexdigest(),
        'state_hash': state_hash,
        'fixture_hash': fixture_hash,
        'observation_hash': obs_hash,
        'noise_hash': '' if noise_field is None else array_hash(noise_field.eps.detach().cpu().numpy()),
        'property_initializers': -1 if initializers is None else len(initializers),
    }


# A repeated (task, episode) key is only *incomparable* when the simulator itself differs. Pixels
# come out of the GPU rasteriser, which is not bit-reproducible across processes, across EGL context
# lifetimes, or under contention, so an observation-only difference is recorded and reported instead
# of raised: aborting a run that has already finished settings costs hours and protects nothing.
FATAL_KEYS = ("state_hash", "fixture_hash", "property_initializers", "noise_hash")


class SceneRegistry:
    """Compare every repeat of one (task, episode) key against its first sighting."""

    def __init__(self):
        self.reference = {}
        self.drifts = []  # observation-only differences: recorded, not fatal

    def check(self, key, fingerprint):
        """Names that differ from the first sighting. Raises only if the simulator differs."""
        expected = self.reference.setdefault(key, dict(fingerprint))
        changed = [name for name in expected if expected[name] != fingerprint[name]]
        if not changed:
            return ()
        fatal = [name for name in changed if name in FATAL_KEYS]
        if fatal:
            raise RuntimeError(f"Evaluation initial conditions changed for {key}: {fatal} "
                               f"(every difference: {changed})")
        self.drifts.append({"key": key, "changed": tuple(changed)})
        logging.warning("Rendered observation differs from the first sighting of %s (%s). Simulator "
                        "state, fixtures and noise are identical, so the episode runs and is "
                        "flagged as scene_drift.", key, ", ".join(changed))
        return tuple(changed)
