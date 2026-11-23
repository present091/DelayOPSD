"""fp32 master weights for the bfloat16 trainable parameters of the LeRobot PI05 port.

LeRobot's PI05 (like openpi's PyTorch path) casts the whole model to bfloat16, so AdamW updates
the bf16 weights directly. bf16 has 7 fraction bits (spacing ~0.4-0.8% of |w|), and a typical
update at lr 2.5e-5 is far below that, so most updates round away (after 4000 OPSD steps only 5.5%
of the 575M bf16 expert elements had changed). openpi's JAX trainer keeps trainable params in
float32 and only casts frozen ones to bf16; this restores that: the model keeps bf16 working
weights for forward/backward, the optimizer (and an EMA teacher) run on float32 copies.
"""

from contextlib import contextmanager

import torch


class MasterWeights:
    def __init__(self, named_params):
        self.params = []  # what the optimizer steps: fp32 params as-is, fp32 masters for the rest
        self.pairs = []  # (name, model param, fp32 master)
        for name, param in named_params:
            if param.dtype == torch.float32:
                self.params.append(param)
            else:
                master = torch.nn.Parameter(param.detach().float().clone())
                self.pairs.append((name, param, master))
                self.params.append(master)

    @torch.no_grad()
    def grads_to_master(self):
        for _, param, master in self.pairs:
            master.grad = None if param.grad is None else param.grad.float()
            param.grad = None

    @torch.no_grad()
    def master_to_model(self):
        for _, param, master in self.pairs:
            param.copy_(master)

    def by_name(self, named_params):
        """Name -> the fp32 tensor that holds the exact value of each trainable parameter."""
        masters = {name: master for name, _, master in self.pairs}
        return {name: masters.get(name, param) for name, param in named_params}


class OffloadAdamW(torch.optim.Optimizer):
    """AdamW (same update as torch.optim.AdamW, amsgrad=False) with its moments in pinned CPU memory.

    Parameters and gradients stay on the GPU. Each parameter's moments are streamed to the GPU,
    updated in float32 together with the parameter, and copied back, so the GPU only ever holds
    one parameter's moments. Frees ~8 bytes per trainable parameter of GPU memory for ~2x the
    moments' size in PCIe transfer per step.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = torch.tensor(0.0)  # tensor like torch.optim.AdamW (checkpoint format)
                    state["exp_avg"] = torch.zeros(param.shape, dtype=param.dtype, pin_memory=True)
                    state["exp_avg_sq"] = torch.zeros(param.shape, dtype=param.dtype, pin_memory=True)
                state["step"] += 1
                grad = param.grad
                exp_avg = state["exp_avg"].to(param.device, non_blocking=True)
                exp_avg_sq = state["exp_avg_sq"].to(param.device, non_blocking=True)
                param.mul_(1 - lr * wd)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step = state["step"].item()
                bias1 = 1 - beta1**step
                bias2 = 1 - beta2**step
                denom = (exp_avg_sq.sqrt() / bias2**0.5).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-lr / bias1)
                state["exp_avg"].copy_(exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)
        torch.cuda.synchronize()  # pinned host copies must land before the next step reads them
        return loss


class CpuEMA:
    """float32 EMA teacher kept in pinned CPU memory, updated on a background thread.

    update(): the exact fp32 parameter values are copied to a pinned staging buffer (queued on the
    GPU stream) and a thread lerps the EMA and refreshes a pinned copy in each parameter's model
    dtype. weights(): waits for the pending update, streams that copy to the GPU and swaps it into
    the model for the duration of the context, so the teacher always includes every finished step,
    exactly like an on-GPU EMA.
    """

    def __init__(self, named_params, decay):
        import threading

        self._threading = threading
        self.decay = decay
        self.params = list(named_params)
        self.ema = {n: p.detach().float().cpu().pin_memory() for n, p in self.params}
        self.staging = {n: torch.empty(p.shape, dtype=torch.float32, pin_memory=True) for n, p in self.params}
        self.cast = {n: torch.empty(p.shape, dtype=p.dtype, pin_memory=True) for n, p in self.params}
        for n, _ in self.params:
            self.cast[n].copy_(self.ema[n])
        self.thread = None

    def join(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    @torch.no_grad()
    def update(self, exact=None):
        self.join()
        for n, p in self.params:
            source = p if exact is None else exact[n]
            self.staging[n].copy_(source.detach().float(), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        names = [n for n, _ in self.params]

        def work():
            event.synchronize()
            torch._foreach_lerp_([self.ema[n] for n in names], [self.staging[n] for n in names], 1 - self.decay)
            for n in names:
                self.cast[n].copy_(self.ema[n])

        self.thread = self._threading.Thread(target=work, daemon=True)
        self.thread.start()

    def state_dict(self):
        self.join()
        return self.ema


@contextmanager
def swapped_weights(named_params, values):
    """Temporarily point each param at values[name] (cast to the param's device and dtype)."""
    backup = {n: p.data for n, p in named_params}
    try:
        for n, p in named_params:
            p.data = values[n].to(device=p.device, dtype=p.dtype)
        yield
    finally:
        for n, p in named_params:
            p.data = backup[n]
