"""Prefix KV caching, masked velocity and trajectory-recording sampling for PI05.

``velocity`` without ``valid`` is exactly ``PI05Pytorch.denoise_step`` and
``sample_with_trajectory`` is exactly ``PI05Pytorch.sample_actions`` (no RTC),
so on-policy flow states recorded during rollouts are those the policy produced.
With ``valid``, suffix tokens outside the mask are zeroed and removed from the
attention keys of every query (invalid queries keep prefix access so no softmax
row is empty; their outputs are never used).
"""

import torch
from torch import Tensor

from lerobot.policies.common.vla_utils import clone_past_key_values
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _smolvla(model):
    """SmolVLA's VLAFlowMatching (or its policy): route to delayopsd.flow_smolvla, same interface."""
    return hasattr(model, "vlm_with_expert") or getattr(getattr(model, "config", None), "type", None) == "smolvla"


def _smol():
    from delayopsd import flow_smolvla as flow  # lazy: that module imports hard_prefix_mask / prefix_time from here

    return flow


def observation_inputs(policy, batch):
    if _smolvla(policy):
        return _smol().observation_inputs(policy, batch)
    images, img_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], states, state_masks


@torch.no_grad()
def prefix_cache(model, observations):
    """No-grad VLM prefill -> (prefix_pad_masks, past_key_values)."""
    if _smolvla(model):
        return _smol().prefix_cache(model, observations)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(*observations)
    att_2d = make_att_2d_masks(prefix_pad, prefix_att)
    positions = torch.cumsum(prefix_pad, dim=1) - 1
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(att_2d),
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad, past_key_values


def velocity(model, prefix_pad, past_key_values, x: Tensor, time: Tensor, valid: Tensor | None = None) -> Tensor:
    """Action-expert velocity for an H-token flow state, optionally with a suffix key mask."""
    if _smolvla(model):
        return _smol().velocity(model, prefix_pad, past_key_values, x, time, valid)
    if valid is not None:
        x = x.masked_fill(~valid[..., None], 0)
    suffix_embs, suffix_pad, suffix_att, adarms_cond = model.embed_suffix(x, time)
    batch, suffix_len = suffix_pad.shape
    prefix_len = prefix_pad.shape[1]
    prefix_2d = prefix_pad[:, None, :].expand(batch, suffix_len, prefix_len)
    suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
    if valid is not None:
        suffix_2d = suffix_2d & valid[:, None, :]
    full_2d = torch.cat([prefix_2d, suffix_2d], dim=2)
    positions = torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
    outputs, _ = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(full_2d),
        position_ids=positions,
        past_key_values=clone_past_key_values(past_key_values),
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def velocity_full(
    model,
    observations,
    x: Tensor,
    time: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    """Differentiable joint VLM/expert velocity with gradient checkpointing."""
    if _smolvla(model):
        return _smol().velocity_full(model, observations, x, time, valid)
    if valid is not None:
        x = x.masked_fill(~valid[..., None], 0)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(*observations)
    suffix_embs, suffix_pad, suffix_att, adarms_cond = model.embed_suffix(x, time)
    transformer_dtype = (
        model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    )
    prefix_embs = prefix_embs.to(dtype=transformer_dtype)
    suffix_embs = suffix_embs.to(dtype=transformer_dtype)
    batch, prefix_len = prefix_pad.shape
    suffix_len = suffix_pad.shape[1]

    prefix_2d = make_att_2d_masks(prefix_pad, prefix_att)
    prefix_to_suffix = torch.zeros(
        batch, prefix_len, suffix_len, dtype=torch.bool, device=prefix_pad.device
    )
    suffix_to_prefix = prefix_pad[:, None, :].expand(batch, suffix_len, prefix_len)
    suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
    if valid is not None:
        suffix_2d = suffix_2d & valid[:, None, :]
    full_2d = torch.cat(
        [
            torch.cat([prefix_2d, prefix_to_suffix], dim=2),
            torch.cat([suffix_to_prefix, suffix_2d], dim=2),
        ],
        dim=1,
    )
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    positions = torch.cumsum(pad, dim=1) - 1
    outputs, _ = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(full_2d),
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def hard_prefix_mask(prefix_len: Tensor, horizon: int) -> Tensor:
    """(B,) hard-prefix lengths -> (B, H) bool, True on the clean prefix tokens."""
    return torch.arange(horizon, device=prefix_len.device)[None] < prefix_len[:, None]


def prefix_time(time: Tensor, prefix_len: Tensor | None, horizon: int) -> Tensor:
    """Per-token flow time with the hard prefix at time 0 (clean), as in training-time RTC.

    Returns ``time`` unchanged (shape (B,)) when there is no prefix, so the no-prefix path is exactly
    the original one.
    """
    if prefix_len is None or not bool((prefix_len > 0).any()):
        return time
    mask = hard_prefix_mask(prefix_len, horizon)
    return torch.where(mask, torch.zeros_like(time)[:, None], time[:, None].expand_as(mask))


@torch.no_grad()
def sample_with_trajectory(model, observations, noise: Tensor, num_steps: int,
                           prefix: Tensor | None = None, prefix_len: Tensor | None = None, prefix_mode="clean"):
    """Euler sampling identical to sample_actions; returns (actions, states[B,S,H,D], times[S]).

    With ``prefix`` (B, H, D) and ``prefix_len`` (B,), the first prefix_len tokens of every example are
    replaced at every step and only the rest is denoised:
      "clean":  by the prefix itself, fed at time 0 (training-time-RTC inference);
      "noised": by the prefix noised to the current t with the step's own initial noise, t * eps + (1 - t) * a,
                so every token shares one flow time -- the input distribution plain FM trains on (inpainting).
    The recorded states hold the replaced prefix, so a later forward on them sees the same input.
    """
    if _smolvla(model):
        return _smol().sample_with_trajectory(model, observations, noise, num_steps, prefix, prefix_len, prefix_mode)
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
        v = velocity(model, prefix_pad, kv, x, time)
        x = x + dt * v
    if hard is not None:
        x = torch.where(hard, prefix, x)
    return x, torch.stack(states, dim=1), torch.tensor(times, dtype=torch.float32)
