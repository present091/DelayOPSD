"""Training configuration: LeRobot's TrainPipelineConfig plus the ``--opsd.*`` group.

One trainer serves both stages of the method:

* stage 1, the teacher: masked flow matching on the demonstrations (``lambda_mask=1``, ``lambda_opsd=0``,
  ``delay_range=[0, D]``): the last delta action tokens are removed from the expert's input and loss, so the teacher
  learns to act from a later observation o_{t+delta} for the first H - delta steps.
* stage 2, the student: on-policy self-distillation (``lambda_opsd=1``, ``lambda_fm=lambda_mask=0``) on pairs
  (o_t, running-chunk prefix, o_{t+delta}) read from rollout buffers of the start policy (delayopsd.collect).

Field names and defaults are those the paper's runs were trained with (``configs/``).
"""

import math
from dataclasses import dataclass, field

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class OPSDConfig:
    # A single-buffer run's schedule R:k (empty rollout_schedules): the pair delay is k = delta.
    replan_interval: int = 10
    switch_offset: int = 7
    pair_gaps: list[int] = field(default_factory=lambda: [1])
    # Explicit pair delays (delta); empty = gap * R for every rollout schedule.
    pair_deltas: list[int] = field(default_factory=list)
    # Schedules "R:k" a single buffer was collected at (empty = replan_interval:switch_offset).
    rollout_schedules: list[str] = field(default_factory=list)
    # Masked-FM delays, delta ~ U{lo..hi} per dataset example (the stage-1 teacher: [0, D]).
    delay_range: list[int] = field(default_factory=list)

    # Loss weights: native FM and masked FM on the demonstrations, OPSD on the pairs.
    lambda_fm: float = 1.0
    lambda_mask: float = 1.0
    lambda_opsd: float = 0.1
    opsd_warmup_steps: int = 0  # linear ramp of lambda_opsd
    opsd_start_step: int = 0  # dataset-loss-only updates before the first OPSD update
    opsd_batch_size: int = 4  # pairs per update

    # Student-only clean action prefix (training-time RTC). At a delayed replan the first min(k, prefix_len) tokens
    # are the running chunk's actions for those steps -- what executes between the student's and the teacher's
    # observations -- fed clean at flow time 0 and never denoised. The teacher never sees it (prefix_len <= delta).
    prefix_len: int = 0
    # "clean": the prefix at flow time 0 (training-time RTC). "noised": noised to the current flow time (inpainting).
    prefix_mode: str = "clean"
    # Dataset FM only: share of examples given a GT prefix (lengths prefix_fm_min..prefix_len when prefix_fm_min > 0).
    prefix_fm_prob: float = 0.5
    prefix_fm_min: int = 0
    prefix_len_random: bool = False  # = prefix_fm_min 1

    # Teacher: "frozen" (the start weights, or teacher_path's trainable tensors), "ema" or "current" (stop-grad self).
    teacher_mode: str = "current"
    teacher_path: str = ""
    ema_decay: float = 0.999  # teacher_mode=ema
    ema_offload: bool = False  # EMA teacher in pinned CPU memory, updated on a background thread
    # Continue an interrupted run from one of its checkpoints (<output_dir>/checkpoints/<step>).
    resume_from: str = ""
    # EMA of the student (fp32 on CPU), saved next to every checkpoint as pretrained_model_ema. 0 = off.
    student_ema_decay: float = 0.0
    # fp32 master weights for the bf16 trainable parameters, AdamW moments in pinned CPU memory; see precision.py.
    master_fp32: bool = False
    adam_offload: bool = False
    exclude_pure_noise: bool = False  # skip tau = 1 when sampling a stored flow state

    # Trust weights on the OPSD target: exp(-r / trust_teacher_scale) over the teacher's chunk index r and
    # exp(-delta / trust_delay_scale). Non-zero switches the loss to one weighted mean over the batch. 0 = off.
    trust_teacher_scale: float = 0.0
    trust_delay_scale: float = 0.0

    # Pairs from rollout buffers (delayopsd.collect): one buffer, or with buffer_md one of its roots.
    buffer_root: str = ""
    # Multi-delay buffers "K:root" (one per delay K; delta == prefix length == K), drawn uniformly.
    buffer_md: list[str] = field(default_factory=list)
    buffer_prefetch: bool = False  # read and decode the next batch's frames on a background thread (same batches)

    # Failure rules on buffer pairs (successful episodes are never filtered): keep a failed episode's pairs until
    # fail_stall_margin steps after its end effector stalls within fail_stall_radius (m); drop a failed episode whose
    # scene also fails in the synchronous buffer fail_sync_root.
    fail_stall_radius: float = 0.0
    fail_stall_margin: int = 0
    fail_sync_root: str = ""

    @property
    def deltas(self) -> list[int]:
        return list(self.pair_deltas) or [g * self.replan_interval for g in self.pair_gaps]

    def delay_bounds(self, horizon: int) -> tuple[int, int] | None:
        if not self.delay_range:
            return None
        if len(self.delay_range) != 2 or not 0 <= self.delay_range[0] <= self.delay_range[1] < horizon:
            raise ValueError(f"delay_range must be [lo, hi] with 0 <= lo <= hi < H ({horizon})")
        return tuple(self.delay_range)

    def schedules(self, horizon: int) -> list[tuple[int, int]]:
        """(R, k) of a single buffer; the configured single schedule when unset."""
        if not self.rollout_schedules:
            return [(self.replan_interval, self.switch_offset)]
        parsed = []
        for spec in self.rollout_schedules:
            r, _, k = spec.partition(":")
            r, k = int(r), int(k)
            if r < 1 or not 0 <= k <= r or r + k > horizon:
                raise ValueError(f"{spec}: need R >= 1, 0 <= k <= R and R + k <= H ({horizon})")
            parsed.append((r, k))
        return parsed

    def slot_deltas(self, replan_interval: int) -> list[int]:
        return list(self.pair_deltas) or [g * replan_interval for g in self.pair_gaps]

    def validate(self, horizon: int) -> None:
        R, k = self.replan_interval, self.switch_offset
        if R < 1 or not 0 <= k <= R or R + k > horizon:
            raise ValueError(f"Need R >= 1, 0 <= k <= R and R + k <= H ({horizon}); got R={R}, k={k}")
        if not self.pair_gaps or len(set(self.pair_gaps)) != len(self.pair_gaps):
            raise ValueError("pair_gaps must be a nonempty list of unique integers")
        if self.pair_deltas:
            if len(set(self.pair_deltas)) != len(self.pair_deltas):
                raise ValueError("pair_deltas must be unique")
            if not all(1 <= d < horizon for d in self.pair_deltas):
                raise ValueError(f"pair_deltas must satisfy 1 <= delta < H ({horizon})")
            if self.delay_range:
                raise ValueError("pair_deltas and delay_range are alternatives; set only one")
        if any(type(g) is not int or g < 1 or g * R >= horizon for g in self.pair_gaps):
            raise ValueError(f"Every pair gap g needs g >= 1 and g * R < H ({horizon})")
        if any(not math.isfinite(x) or x < 0 for x in (self.lambda_fm, self.lambda_mask, self.lambda_opsd)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.opsd_batch_size < 1 or self.opsd_warmup_steps < 0 or self.opsd_start_step < 0:
            raise ValueError("opsd_batch_size must be positive, opsd_warmup_steps and opsd_start_step nonnegative")
        if not 0 <= self.prefix_len < horizon or not 0 <= self.prefix_fm_prob <= 1:
            raise ValueError(f"Need 0 <= prefix_len < H ({horizon}) and 0 <= prefix_fm_prob <= 1")
        if not 0 <= self.prefix_fm_min <= self.prefix_len:
            raise ValueError("prefix_fm_min must be in [0, prefix_len]")
        if self.prefix_mode not in {"clean", "noised"}:
            raise ValueError("prefix_mode must be clean or noised")
        if self.prefix_len:
            if self.delay_range:
                raise ValueError("prefix_len supports fixed rollout_schedules / pair_deltas only")
            if any(d < self.prefix_len for r, _ in self.schedules(horizon) for d in self.slot_deltas(r)):
                raise ValueError("prefix_len must not exceed any teacher delta, or the teacher would see it")
        if self.lambda_opsd > 0 and not self.buffer_root:
            raise ValueError("OPSD (lambda_opsd > 0) needs buffer_root")
        if self.buffer_md:
            roots = [entry.partition(":")[2] for entry in self.buffer_md]
            if not all(e.partition(":")[0].isdigit() and e.partition(":")[2] for e in self.buffer_md):
                raise ValueError(f"buffer_md entries must be K:root, got {self.buffer_md}")
            if self.buffer_root not in roots:
                raise ValueError("with buffer_md, buffer_root must be one of its roots")
        for r, _ in self.schedules(horizon):
            if any(g * r >= horizon for g in self.pair_gaps):
                raise ValueError(f"schedule R={r} with pair_gaps {self.pair_gaps} exceeds H ({horizon})")
        if self.teacher_mode not in {"current", "ema", "frozen"}:
            raise ValueError("teacher_mode must be current, ema or frozen")
        if self.teacher_path and (self.teacher_mode != "frozen" or self.ema_offload):
            raise ValueError("teacher_path needs teacher_mode=frozen without ema_offload")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.student_ema_decay < 1:
            raise ValueError("ema_decay and student_ema_decay must be in [0, 1)")

    @property
    def uses_dataset(self) -> bool:
        return self.lambda_fm > 0 or self.lambda_mask > 0

    def opsd_weight(self, step: int) -> float:
        if self.opsd_warmup_steps == 0:
            return self.lambda_opsd
        return self.lambda_opsd * min(1.0, step / self.opsd_warmup_steps)


@dataclass
class OPSDTrainConfig(TrainPipelineConfig):
    opsd: OPSDConfig = field(default_factory=OPSDConfig)

    def validate(self):
        if not self.opsd.resume_from:
            return super().validate()
        # resume_from continues into the existing output_dir: skip the base class's fresh-run check
        out = self.output_dir
        self.output_dir = str(out)
        try:
            return super().validate()
        finally:
            self.output_dir = type(out)(self.output_dir) if out is not None else None
