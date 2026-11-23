"""Data-parallel OPSD training: one trainer process per GPU (batches/_train.sh with OPSD_GPUS=<a>,<b>[,...]).

The trainer backpropagates its loss terms one at a time through helper functions, not through one module forward,
so DistributedDataParallel has nothing to wrap. What it does is done by hand instead: every rank draws its own
pairs and backpropagates, then the gradients are averaged over the ranks before clipping and the optimizer step.
All ranks start from the same weights and apply the same update, so they stay equal (``spread`` measures it).

The effective batch is world size x opsd_batch_size (x batch_size for the dataset terms); the learning rate is
not rescaled. Rank 0 alone holds the envs, logs, evaluates and saves.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init():
    """(rank, world size). Joins the process group when the launcher set WORLD_SIZE > 1 (with RANK, MASTER_*)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    # rank 0 alone saves (~1 min per checkpoint on a slow disk) while the others wait in the next collective or in
    # finish()'s barrier: an hour of margin over NCCL's 10-minute default
    dist.init_process_group("nccl", timeout=timedelta(hours=1))
    return dist.get_rank(), world


def finish(world):
    """Leave the process group together: a rank that exits while rank 0 still saves the last checkpoint tears down
    the NCCL communicator under it, and rank 0 aborts mid-write."""
    if world == 1:
        return
    dist.barrier()
    dist.destroy_process_group()


@torch.no_grad()
def average_gradients(params, world, bucket=1 << 25):
    """Replace every .grad by its mean over the ranks (float32 on the wire, in buckets of ~`bucket` elements)."""
    params = [p for p in params if p.grad is not None]
    # Ranks that disagree on which parameters have a gradient would reduce buffers of different sizes.
    count = torch.tensor([len(params), -len(params)], device=params[0].device)
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    if count[0] != -count[1]:
        raise RuntimeError(f"ranks differ in parameters with a gradient ({-int(count[1])}..{int(count[0])})")
    start = 0
    while start < len(params):
        end, size = start, 0
        while end < len(params) and (size < bucket or end == start):
            size += params[end].grad.numel()
            end += 1
        flat = torch.cat([p.grad.reshape(-1).float() for p in params[start:end]])
        dist.all_reduce(flat)
        flat /= world
        offset = 0
        for p in params[start:end]:
            n = p.grad.numel()
            p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n
        start = end


@torch.no_grad()
def mean_scalars(values, world):
    """Means over the ranks of 0-dim tensors (the per-rank losses, for logging)."""
    stacked = torch.stack([v.detach().float() for v in values])
    dist.all_reduce(stacked)
    return (stacked / world).unbind()


@torch.no_grad()
def spread(tensors):
    """max - min over the ranks of the sum of all values: 0.0 as long as the ranks hold the same weights."""
    total = torch.stack([t.sum(dtype=torch.float64) for t in tensors]).sum()
    both = torch.stack([total, -total])
    dist.all_reduce(both, op=dist.ReduceOp.MAX)
    return float(both[0] + both[1])
