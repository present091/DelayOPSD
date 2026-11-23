"""Prefix KV caching, masked velocity and trajectory-recording sampling for SmolVLA.

Same interface as ``delayopsd.flow`` so the OPSD objective and trainer can switch on the policy type.
``velocity`` without ``valid`` and with a scalar time is exactly ``VLAFlowMatching.denoise_step`` and
``sample_with_trajectory`` without a prefix is exactly ``VLAFlowMatching.sample_actions`` (no RTC); both are checked
by ``delayopsd.flow_smolvla``.

Differences from PI05 that matter here:
- SmolVLA's action tokens attend causally (suffix att mask all ones), so token i only sees tokens <= i.
- Its expert appends suffix K/V to the prefix cache in place (self-attention layers) and the upstream sampler
  crops it back after every step. ``velocity`` works on a clone instead, so one prefix cache can serve several
  differentiable calls (teacher, student, masked) without any of them seeing another's suffix.
- Per-token flow time (training-time RTC: the clean hard prefix at time 0) needs a time embedding per action
  token; upstream ``embed_suffix`` broadcasts one time per example. ``embed_suffix`` below takes (B,) or (B, H).
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.common.vla_utils import (
    clone_past_key_values,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from delayopsd.flow import hard_prefix_mask, prefix_time  # noqa: F401  (model independent)


def observation_inputs(policy, batch):
    """Arguments of ``VLAFlowMatching.embed_prefix`` for a preprocessed batch (as SmolVLAPolicy.forward builds them)."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state


@torch.no_grad()
def prefix_cache(model, observations):
    """No-grad VLM prefill -> (prefix_pad_masks, past_key_values), as in ``sample_actions``."""
    images, img_masks, lang_tokens, lang_masks, state = observations
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad, prefix_att),
        position_ids=torch.cumsum(prefix_pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad, past_key_values


def embed_suffix(model, noisy_actions: Tensor, time: Tensor):
    """``VLAFlowMatching.embed_suffix`` with a per-token time: ``time`` is (B,) or (B, H)."""
    action_emb = model.action_in_proj(noisy_actions)
    time_emb = create_sinusoidal_pos_embedding(
        time, model.vlm_with_expert.expert_hidden_size, model.config.min_period, model.config.max_period,
        device=action_emb.device,
    ).type(dtype=action_emb.dtype)
    time_emb = time_emb[:, None, :].expand_as(action_emb) if time.ndim == 1 else time_emb
    x = model.action_time_mlp_out(F.silu(model.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))))
    batch, length = x.shape[:2]
    pad = torch.ones(batch, length, dtype=torch.bool, device=x.device)
    att = torch.ones(batch, model.config.chunk_size, dtype=x.dtype, device=x.device)
    return x, pad, att


def velocity(model, prefix_pad, past_key_values, x: Tensor, time: Tensor, valid: Tensor | None = None) -> Tensor:
    """Action-expert velocity for an H-token flow state, optionally with a suffix key mask.

    ``valid`` (B, H): tokens outside are zeroed and removed from every query's suffix keys (queries keep prefix
    access, so no softmax row is empty; their outputs are never used).
    """
    if valid is not None:
        x = x.masked_fill(~valid[..., None], 0)
    suffix_embs, suffix_pad, suffix_att = embed_suffix(model, x, time)
    batch, suffix_len = suffix_pad.shape
    prefix_len = prefix_pad.shape[1]
    prefix_2d = prefix_pad[:, None, :].expand(batch, suffix_len, prefix_len)
    suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
    if valid is not None:
        suffix_2d = suffix_2d & valid[:, None, :]
    full_2d = torch.cat([prefix_2d, suffix_2d], dim=2)
    positions = torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
    outputs, _ = model.vlm_with_expert.forward(
        attention_mask=full_2d,
        position_ids=positions,
        past_key_values=clone_past_key_values(past_key_values),
        inputs_embeds=[None, suffix_embs],
        use_cache=True,
    )
    suffix_out = outputs[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def velocity_full(model, observations, x: Tensor, time: Tensor, valid: Tensor | None = None) -> Tensor:
    """Differentiable joint VLM/expert velocity (no cache), as ``VLAFlowMatching.forward``."""
    if valid is not None:
        x = x.masked_fill(~valid[..., None], 0)
    images, img_masks, lang_tokens, lang_masks, state = observations
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    suffix_embs, suffix_pad, suffix_att = embed_suffix(model, x, time)
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    full_2d = make_att_2d_masks(pad, torch.cat([prefix_att, suffix_att], dim=1))
    if valid is not None:
        prefix_len = prefix_pad.shape[1]
        keep = torch.cat([torch.ones_like(prefix_pad), valid], dim=1)
        query_is_suffix = torch.arange(pad.shape[1], device=pad.device)[None, :, None] >= prefix_len
        full_2d = full_2d & (keep[:, None, :] | ~query_is_suffix)
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=full_2d,
        position_ids=torch.cumsum(pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


@torch.no_grad()
def sample_with_trajectory(model, observations, noise: Tensor, num_steps: int,
                           prefix: Tensor | None = None, prefix_len: Tensor | None = None, prefix_mode="clean"):
    """Euler sampling identical to ``sample_actions``; returns (actions, states[B,S,H,D], times[S]).

    Prefix handling as in ``delayopsd.flow.sample_with_trajectory``: the first prefix_len tokens are
    replaced at every step, by the prefix fed at time 0 ("clean", training-time-RTC inference) or by the prefix
    noised to the current t ("noised").
    """
    prefix_pad, kv = prefix_cache(model, observations)
    dt = -1.0 / num_steps
    hard = None
    if prefix is not None and prefix_len is not None and bool((prefix_len > 0).any()):
        if prefix_mode not in {"clean", "noised"}:
            raise ValueError(f"prefix_mode must be clean or noised, got {prefix_mode!r}")
        hard = hard_prefix_mask(prefix_len, noise.shape[1])[..., None]
        prefix = prefix.to(noise.dtype)
    eps, x, states, times = noise, noise, [], []
    for step in range(num_steps):
        t = 1.0 + step * dt
        time = torch.tensor(t, dtype=torch.float32, device=x.device).expand(x.shape[0])
        if hard is not None:
            x = torch.where(hard, prefix if prefix_mode == "clean" else t * eps + (1 - t) * prefix, x)
            if prefix_mode == "clean":
                time = prefix_time(time, prefix_len, x.shape[1])
        states.append(x)
        times.append(t)
        x = x + dt * velocity(model, prefix_pad, kv, x, time)
    if hard is not None:
        x = torch.where(hard, prefix, x)
    return x, torch.stack(states, dim=1), torch.tensor(times, dtype=torch.float32)
