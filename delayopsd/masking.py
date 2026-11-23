import torch
from torch import Tensor


def suffix_mask(delta: Tensor, horizon: int) -> Tensor:
    """Per-example M_delta, keeping the action sequence length unchanged."""
    if delta.ndim != 1 or delta.dtype != torch.long:
        raise ValueError("delta must be an int64 vector")
    if torch.any((delta < 0) | (delta >= horizon)):
        raise ValueError("Every delay must satisfy 0 <= delta < horizon")
    return torch.arange(horizon, device=delta.device)[None] < horizon - delta[:, None]


def shift_tail(x: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    """Student tail in teacher coordinates; invalid suffix contains only zeros."""
    batch, horizon, dim = x.shape
    if delta.shape != (batch,):
        raise ValueError("Expected one delay per example")
    valid = suffix_mask(delta, horizon)
    indices = torch.arange(horizon, device=x.device)[None] + delta[:, None]
    shifted = x.gather(1, indices.clamp(max=horizon - 1)[..., None].expand(-1, -1, dim))
    return shifted.masked_fill(~valid[..., None], 0), valid


def masked_flow_loss(velocity: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Native FM mean over valid positions and actual action dimensions."""
    errors = (velocity - target).square().masked_fill(~valid[..., None], 0)
    return errors.sum() / (valid.sum() * velocity.shape[-1])
