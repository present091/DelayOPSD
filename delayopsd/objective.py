"""Dataset flow matching + masked flow matching + on-policy self-distillation.

Dataset batch:  L_FM   native flow matching on (obs, GT chunk)
                L_mask same x_t, the final delta tokens removed from input keys and loss
On-policy pair: L_OPSD student v(obs_t, x_tau, tau)[delta:H]  (grad)
                vs teacher v(obs_{t+delta}, shift(x_tau) with masked suffix, tau)[0:H-delta]
                (stop-grad self, or EMA weights). x_tau is a stored rollout flow state.

The action expert can train alone, or PEFT adapters can train in both the VLM and
expert. Teacher prefix caches are no-grad; student caches stay differentiable
when the VLM has trainable adapters.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lerobot.policies.pi05.modeling_pi05 import _build_flow_matching_inputs, _reduce_training_rtc_loss
from lerobot.utils.constants import ACTION

from . import relative as rel
from .masking import masked_flow_loss, shift_tail, suffix_mask
from .flow import hard_prefix_mask, observation_inputs, prefix_cache, prefix_time, velocity, velocity_full


def trust_weights(valid, delta, teacher_scale, delay_scale):
    """Trust in the teacher target: W1(r) = exp(-r / teacher_scale) over the aligned index r (the teacher's own
    chunk index: its far end extrapolates furthest from o_{t+delta}), times W2(delta) = exp(-delta / delay_scale)."""
    weight = valid.float()
    if teacher_scale > 0:
        offsets = torch.arange(valid.shape[1], device=valid.device, dtype=weight.dtype)[None]
        weight = weight * torch.exp(-offsets / teacher_scale)
    if delay_scale > 0:
        weight = weight * torch.exp(-delta.to(weight.dtype)[:, None] / delay_scale)
    return weight


def opsd_distill_loss(student, teacher, delta, teacher_scale=0.0, delay_scale=0.0):
    """Squared error between the student's velocity on the aligned region [delta, H) and the teacher's on [0, H - delta).

    Element mean over action dims, as the dataset FM / masked FM losses. With trust weights, one weighted mean over
    the whole batch (a per-example mean would cancel W2, constant within an example); without, the mean over each
    example's valid indices, then over examples.
    """
    aligned, valid = shift_tail(student, delta)
    errors = (aligned - teacher.detach()).square().mean(-1)
    trust = teacher_scale > 0 or delay_scale > 0
    weight = trust_weights(valid, delta, teacher_scale, delay_scale) if trust else valid.float()
    weight = weight.masked_fill(~valid, 0)
    errors = (errors * weight).masked_fill(~valid, 0)
    if trust:
        return errors.sum() / weight.sum().clamp(min=1e-8)
    return (errors.sum(-1) / weight.sum(-1)).mean()


class OPSDObjective(nn.Module):
    def __init__(self, policy, config, *, allow_vlm_training=False):
        super().__init__()
        pc = policy.config
        config.validate(pc.chunk_size)
        if not pc.train_expert_only and not allow_vlm_training:
            raise ValueError(
                "VLM training in this single-GPU trainer requires PEFT adapters; "
                "set policy.train_expert_only=true or enable --peft.*"
            )
        if pc.rtc_config is not None or getattr(pc, "rtc_training_max_delay", 0) or pc.compile_model:
            raise ValueError("Requires no RTC and compile_model=false")
        # Relative actions: the dataset losses work as they are (the preprocessor makes the demonstrations'
        # targets relative to their own frame's state). OPSD pairs span two frames, hence two model spaces: only
        # single-buffer pairs carry the student->teacher offset (delayopsd.relative), and opsd_loss
        # refuses a relative batch without it.
        self.relative = bool(getattr(pc, "use_relative_actions", False))
        if self.relative and config.lambda_opsd > 0 and config.buffer_md:
            raise ValueError("relative-action OPSD supports one buffer only (no buffer_md)")
        if getattr(pc, "use_visual_memory", False) or getattr(pc, "use_proprioceptive_memory", False):
            raise ValueError("Memory variants are not supported")
        self.policy, self.config = policy, config
        self.train_vlm = not pc.train_expert_only
        self.dim = pc.output_features[ACTION].shape[0]
        self.trainable = [(n, p) for n, p in policy.model.named_parameters() if p.requires_grad]
        # An EMA teacher accumulates in float32 when master weights are on (a bf16 lerp by 1e-3 rounds away).
        ema_dtype = torch.float32 if config.teacher_mode == "ema" and config.master_fp32 else None
        self.cpu_ema = None
        if config.teacher_mode == "ema" and config.ema_offload:
            from .precision import CpuEMA

            self.cpu_ema = CpuEMA(self.trainable, config.ema_decay)
        self.ema = None if self.cpu_ema is not None else (
            {n: p.detach().to(ema_dtype or p.dtype, copy=True) for n, p in self.trainable}
            if config.teacher_mode in {"ema", "frozen"} else None
        )
        if config.teacher_path:
            self._load_teacher(config.teacher_path)

    @torch.no_grad()
    def _load_teacher(self, path):
        """Frozen teacher = the trainable tensors of another checkpoint (see OPSDConfig.teacher_path).

        LoRA adapters keep their start values (B = 0, i.e. no adapter), and every other tensor of the model
        must equal the checkpoint's, so the teacher is exactly that checkpoint's model.
        """
        from safetensors import safe_open

        # The teacher must work in the student's action space (a relative student with an absolute teacher, or
        # the reverse, would distill velocities of the other space without any error)
        teacher_relative = bool(json.loads(Path(path, "config.json").read_text()).get("use_relative_actions", False))
        if teacher_relative != self.relative:
            raise ValueError(f"teacher {path} is {'relative' if teacher_relative else 'absolute'}-action, the student "
                             f"{'relative' if self.relative else 'absolute'}-action")

        def key(name):  # PEFT adapts modules in place and renames their parameters
            return "model." + name.replace(".modules_to_save.default", "").replace(".base_layer", "")

        trainable = {n for n, _ in self.trainable}
        with safe_open(f"{path}/model.safetensors", framework="pt") as f:
            keys = set(f.keys())
            for name, param in self.trainable:
                if "lora_" in name:
                    continue
                if key(name) not in keys:
                    raise KeyError(f"{key(name)} (trainable {name}) is not in the teacher checkpoint {path}")
                tensor = f.get_tensor(key(name))
                if tensor.shape != param.shape:
                    raise ValueError(f"{key(name)}: teacher {tuple(tensor.shape)} vs student {tuple(param.shape)}")
                self.ema[name] = tensor.to(device=param.device, dtype=self.ema[name].dtype)
            differ = [
                key(name) for name, param in self.policy.model.named_parameters()
                if name not in trainable and "lora_" not in name and "original_module" not in name
                and key(name) in keys and not torch.equal(f.get_tensor(key(name)).to(param.device, param.dtype), param)
            ]
        if differ:
            raise ValueError(f"teacher checkpoint differs outside the trainable tensors ({len(differ)}): {differ[:5]}")

    @torch.no_grad()
    def update_teacher(self, exact=None):
        """exact: optional name -> fp32 value of each trainable param (master weights)."""
        if self.cpu_ema is not None:
            self.cpu_ema.update(exact)
        elif self.config.teacher_mode == "ema":
            for name, param in self.trainable:
                source = param.detach() if exact is None else exact[name].detach()
                self.ema[name].lerp_(source.to(self.ema[name].dtype), 1 - self.config.ema_decay)

    @contextmanager
    def teacher_weights(self):
        if self.cpu_ema is not None:
            self.cpu_ema.join()
            teacher = {n: self.cpu_ema.cast[n].to(p.device, non_blocking=True) for n, p in self.trainable}
        elif self.ema is None:
            yield
            return
        else:
            teacher = self.ema
        backup = {n: p.data for n, p in self.trainable}
        try:
            for name, param in self.trainable:
                weight = teacher[name]
                param.data = weight if weight.dtype == param.dtype else weight.to(param.dtype)
            yield
        finally:
            for name, param in self.trainable:
                param.data = backup[name]

    def dataset_losses(self, batch):
        model = self.policy.model
        actions = self.policy.prepare_action(batch)
        noise = model.sample_noise(actions.shape, actions.device)
        time = model.sample_time(actions.shape[0], actions.device)
        hard = None
        if self.config.prefix_len and self.config.prefix_mode == "clean":
            # training-time RTC on GT: a clean GT prefix for a prefix_fm_prob share, plain FM for the rest
            use = torch.rand(actions.shape[0], device=actions.device) < self.config.prefix_fm_prob
            length = self.config.prefix_len
            low = self.config.prefix_fm_min or int(self.config.prefix_len_random)
            if low:  # multi-delay: a length per example from low..prefix_len
                length = torch.randint(low, self.config.prefix_len + 1, (actions.shape[0],), device=actions.device)
            hard = hard_prefix_mask(use.long() * length, actions.shape[1])
        x, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask=hard)
        target = (noise - actions)[:, :, : self.dim]
        observations = observation_inputs(self.policy, batch)
        if self.train_vlm:
            full_v = velocity_full(model, observations, x, model_time)
        else:
            prefix_pad, kv = prefix_cache(model, observations)
            full_v = velocity(model, prefix_pad, kv, x, model_time)
        if hard is None:
            full = F.mse_loss(full_v[:, :, : self.dim], target)
        else:  # the clean prefix is input only
            full = _reduce_training_rtc_loss(F.mse_loss(full_v[:, :, : self.dim], target, reduction="none"),
                                             hard, "mean")
        masked = full.new_zeros(())
        if self.config.lambda_mask:
            bounds = self.config.delay_bounds(actions.shape[1])
            if bounds is None:
                choices = torch.tensor(self.config.deltas, device=actions.device)
                delta = choices[torch.randint(len(choices), (actions.shape[0],), device=actions.device)]
            else:  # same mask-size distribution the teacher sees at rollout time (suffix position)
                delta = torch.randint(bounds[0], bounds[1] + 1, (actions.shape[0],), device=actions.device)
            valid = suffix_mask(delta, actions.shape[1])
            if self.train_vlm:
                masked_v = velocity_full(model, observations, x, model_time, valid)
            else:
                masked_v = velocity(model, prefix_pad, kv, x, model_time, valid)
            masked = masked_flow_loss(masked_v[:, :, : self.dim], target, valid)
        return full, masked

    def opsd_loss(self, student_batch, teacher_batch, states, times, delta, index=None, prefix_len=None):
        """index: optional (B,) flow-step indices; sampled uniformly when None (training).

        prefix_len: optional (B,) hard-prefix lengths the student chunks were sampled with; the stored
        states already hold the clean prefix, so only the student's per-token time changes. The teacher is
        untouched: shift_tail drops tokens < delta, and prefix_len <= delta.
        """
        model = self.policy.model
        device = states.device
        batch, n_steps = states.shape[:2]
        low = 1 if self.config.exclude_pure_noise and n_steps > 1 else 0
        if index is None:
            index = torch.randint(low, n_steps, (batch,), device=device)
        x = states[torch.arange(batch, device=device), index]
        tau = times.to(device)[index]
        offset = student_batch.get(rel.OFFSET_KEY)
        if self.relative and offset is None:
            raise ValueError("relative-action OPSD batch without the student->teacher offset")
        if offset is not None and not self.relative:
            raise ValueError("student->teacher offset given to an absolute-action objective")
        # Teacher first so its no-grad activations are freed before the student's graph is built.
        with torch.no_grad(), self.teacher_weights():
            teacher_x, teacher_valid = shift_tail(x, delta)
            if offset is not None:  # the student's flow state in the teacher frame's model space
                teacher_x = rel.to_teacher_state(teacher_x, tau, offset)
            teacher_observations = observation_inputs(self.policy, teacher_batch)
            if self.train_vlm:
                teacher = velocity_full(model, teacher_observations, teacher_x, tau, teacher_valid)
            else:
                t_pad, t_kv = prefix_cache(model, teacher_observations)
                teacher = velocity(model, t_pad, t_kv, teacher_x, tau, teacher_valid)
                del t_pad, t_kv
            if offset is not None:  # the teacher's velocity back in the student's model space
                teacher = rel.to_student_velocity(teacher, offset)
        student_observations = observation_inputs(self.policy, student_batch)
        clean = self.config.prefix_mode == "clean"  # a noised prefix shares the flow time of the rest
        student_tau = prefix_time(tau.expand(batch), prefix_len if clean else None, x.shape[1])
        if self.train_vlm:
            student = velocity_full(model, student_observations, x, student_tau)
        else:
            prefix_pad, kv = prefix_cache(model, student_observations)
            student = velocity(model, prefix_pad, kv, x, student_tau)
        student, teacher = student[:, :, : self.dim], teacher[:, :, : self.dim]
        return opsd_distill_loss(student, teacher, delta, self.config.trust_teacher_scale,
                                 self.config.trust_delay_scale)
