"""Relative-action pi0.5 (use_relative_actions=true) for OPSD buffer pairs.

A relative pi0.5 works in model space N(a - state): the absolute targets minus the raw observation.state of the
frame the chunk is inferred from (arm joints; the gripper, relative_exclude_joints, stays absolute), normalized
with the relative-action quantiles. Its preprocessor does exactly this to the demonstrations (RelativeActions step,
then the Normalizer), and its postprocessor undoes it at inference.

A buffer pair has two frames, student s and teacher s+k, so two model spaces:
  prefix   the running chunk's absolute commands over [s, s+k) in the STUDENT's space: model_space(a, state_s)
  offset   c = N(a - state_t) - N(a - state_s), the same for every a (the normalization is affine, without
           clipping): a chunk in the teacher's space = the same chunk in the student's space + c.
With x_tau = tau * noise + (1 - tau) * a and v = noise - a (PI05Pytorch), the student-space flow state x is
x + (1 - tau) c in the teacher's space, and a teacher velocity v_T there is v_T + c in the student's space.
delayopsd.objective applies these two (to_teacher_state / to_student_velocity).

Both are computed by the policy's own preprocessor steps, not a re-implementation of their formulas.
"""

import torch
from torch import Tensor

from lerobot.processor import (NormalizerProcessorStep, RelativeActionsProcessorStep, TransitionKey,
                               create_transition)
from lerobot.utils.constants import OBS_STATE

OFFSET_KEY = "opsd.relative_offset"  # (B, 1, max_action_dim) carried in the student batch to OPSDObjective


def _steps(preprocessor):
    rel = [s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)]
    norm = [s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)]
    if len(rel) != 1 or len(norm) != 1 or preprocessor.steps.index(rel[0]) > preprocessor.steps.index(norm[0]):
        raise ValueError("expected one RelativeActions step followed by one Normalizer step in the preprocessor")
    return rel[0], norm[0]


def is_relative(preprocessor) -> bool:
    return any(isinstance(s, RelativeActionsProcessorStep) and s.enabled for s in preprocessor.steps)


@torch.no_grad()
def model_space(preprocessor, actions: Tensor, state: Tensor) -> Tensor:
    """Absolute actions (B, T, A) of chunks inferred at raw states (B, S) -> the model's normalized actions."""
    rel, norm = _steps(preprocessor)
    if not rel.enabled:
        raise ValueError("model_space is for relative-action policies")
    cached = rel._last_state  # noqa: SLF001 - the paired postprocessor's reference state, left as it was
    try:
        out = norm(rel(create_transition(observation={OBS_STATE: state.float()}, action=actions.float())))
    finally:
        rel._last_state = cached  # noqa: SLF001
    return out[TransitionKey.ACTION]


@torch.no_grad()
def frame_offset(preprocessor, state_student: Tensor, state_teacher: Tensor, action_dim: int) -> Tensor:
    """c (B, 1, action_dim): teacher-space model action = student-space model action + c."""
    a = torch.zeros(state_student.shape[0], 1, action_dim, device=state_student.device)
    return model_space(preprocessor, a, state_teacher) - model_space(preprocessor, a, state_student)


def to_teacher_state(x: Tensor, tau: Tensor, offset: Tensor) -> Tensor:
    """Student-space flow state x (B, H, D) at flow time tau (B,) -> the teacher's space."""
    return x + (1.0 - tau.to(x.dtype)).view(-1, 1, 1) * offset.to(x.dtype)


def to_student_velocity(v: Tensor, offset: Tensor) -> Tensor:
    """Teacher-space velocity (B, H, D) -> the student's space."""
    return v + offset.to(v.dtype)
