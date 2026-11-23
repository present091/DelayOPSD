"""End-to-end check of the relative-action training path through the trainer's own objects.

    python -m real_robot.tests.test_relative_e2e --config_path=configs/yam/student.json --output_dir=/tmp/unused

Builds the policy (+ PEFT), processors, OPSDObjective and BufferPairSource exactly as delayopsd.train does,
then checks against values computed here by hand from the raw buffer and the checkpoint's own quantile stats (numpy,
not the processor code under test):

D  dataset (FM anchor / teacher targets): the actions the objective trains on (policy.prepare_action of the
   preprocessed batch) == 2 * ((a - state * arm_mask) - q01) / (q99 - q01) - 1 of the raw dataset sample
P  pairs: the clean prefix inside the recorded student trajectory == the same formula on the stored reply with the
   STUDENT frame's state; the offset carried in the student batch == 2 * (state_s - state_t) * arm_mask / (q99 - q01)
W  wiring: inside opsd_loss the teacher model is called on shift_tail(x) + (1 - tau) * offset (the valid tokens;
   the rest zero), and its velocity enters the loss as v + offset
J  joint space (independent of the offset formula): the teacher's clean-action estimate x_T - tau v_T in ITS frame,
   un-normalized + state_t, == the estimate the student is trained toward, shift(x) - tau (v_T + c) in the student
   frame, un-normalized + state_s, in absolute joints; and the noise estimates x + (1 - tau) v agree. Mutations run
   through opsd_loss (offset sign flipped, offset zeroed) must break it by about the state change
S  sign / direction: with the teacher = the student's own start weights, the OPSD loss with the offset is below the
   loss with the offset zeroed, on every batch (a wrong sign or frame would raise it)
X  refusals: an absolute teacher for this relative student, and a relative batch without the offset, raise
"""

import copy
import os
from dataclasses import asdict

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from delayopsd.masking import shift_tail
from delayopsd.data.utils import prepare_batch

from delayopsd import objective as objective_module
from delayopsd import relative as rel
from delayopsd.data.buffer import BufferPairSource
from delayopsd.config import OPSDTrainConfig
from delayopsd.objective import OPSDObjective

ABS = os.environ.get("ABS_MODEL", "")  # optional: an absolute-action checkpoint, which must be refused as teacher


@parser.wrap()
def run(cfg: OPSDTrainConfig):
    cfg.validate()
    set_seed(0)
    device = torch.device(cfg.policy.device)
    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy = policy.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))
    runtime = policy.get_base_model()
    pre, _ = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
                                      preprocessor_overrides={"device_processor": {"device": str(device)},
                                                              "rename_observations_processor": {"rename_map": cfg.rename_map}})
    assert rel.is_relative(pre) and cfg.policy.use_relative_actions
    from safetensors.numpy import load_file
    st = load_file(f"{cfg.policy.pretrained_path}/policy_preprocessor_step_3_normalizer_processor.safetensors")
    q01, q99 = st["action.q01"][:7].astype(np.float64), st["action.q99"][:7].astype(np.float64)
    mask = np.array([1, 1, 1, 1, 1, 1, 0], np.float64)  # relative_exclude_joints = [gripper]
    norm = lambda a, s: 2 * ((a - s[..., None, :] * mask) - q01) / (q99 - q01) - 1  # noqa: E731
    ok = True

    # ---- D
    worst_d = 0.0
    for i in np.random.default_rng(0).choice(len(dataset), 8, replace=False):
        raw = dataset[int(i)]
        batch = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in raw.items()}
        prepared = prepare_batch(batch, dataset.meta.camera_keys, cfg.rename_map, pre)
        got = runtime.prepare_action(prepared)[0, :, :7].float().cpu().numpy()
        want = norm(raw["action"][:, :7].numpy().astype(np.float64), raw["observation.state"][:7].numpy()[None])[0]
        worst_d = max(worst_d, float(np.abs(got - want).max()))
    print(f"D dataset targets vs hand formula (relative to the frame's state): max {worst_d:.2e}")
    ok &= worst_d < 1e-4

    objective = OPSDObjective(runtime, cfg.opsd, allow_vlm_training=True)
    source = BufferPairSource(cfg, runtime, pre, lambda x: x, seed=0)
    calls = []
    real_vfull, real_vel = objective_module.velocity_full, objective_module.velocity

    def spy_full(model, obs, x, time, valid=None):
        out = real_vfull(model, obs, x, time, valid)
        calls.append(("full", x.detach().clone(), time.detach().clone(), None if valid is None else valid.clone(), out.detach().clone()))
        return out

    objective_module.velocity_full = spy_full
    worst_p = worst_c = worst_w = worst_v = 0.0
    losses = []
    unnorm = lambda n: (n + 1.0) * (q99 - q01) / 2.0 + q01  # noqa: E731
    joint = {"correct": [], "sign flipped": [], "zeroed": []}
    eps_worst = 0.0

    def oracle(c, s_state, t_state, x, delta):
        """max |absolute joints| between the teacher's own estimate and the student-frame target, from the spy."""
        _, x_t, tau, valid, v_t = calls[0]
        shifted, ok_valid = shift_tail(x, delta)
        tau_ = tau.view(-1, 1, 1).double()
        a_t = (x_t.double() - tau_ * v_t.double())[..., :7].cpu().numpy()
        a_s = (shifted.double() - tau_ * (v_t.double() + c.double()))[..., :7].cpu().numpy()
        abs_t = unnorm(a_t) + (t_state * mask)[:, None]
        abs_s = unnorm(a_s) + (s_state * mask)[:, None]
        m = valid.cpu().numpy() & ok_valid.cpu().numpy()
        e_t = (x_t.double() + (1 - tau_) * v_t.double())[..., :7].cpu().numpy()
        e_s = (shifted.double() + (1 - tau_) * (v_t.double() + c.double()))[..., :7].cpu().numpy()
        return float(np.abs(abs_t - abs_s)[m].max()), float(np.abs(e_t - e_s)[m].max())
    for b in range(8):
        rng = copy.deepcopy(source.rng)
        picks = source.pool[rng.choice(len(source.pool), source.batch_size, p=source.weight)]
        student, teacher, states, times, delta, prefix_len = source.next_batch()
        s_state = np.stack([np.asarray(source.states[i]) for i in source.student[picks]]).astype(np.float64)
        t_state = np.stack([np.asarray(source.states[i]) for i in source.teacher[picks]]).astype(np.float64)
        # ---- P
        want_prefix = norm(source.raw_prefix[picks].astype(np.float64), s_state)
        got_prefix = states[:, 0, :15, :7].cpu().numpy()
        worst_p = max(worst_p, float(np.abs(got_prefix - want_prefix).max()))
        want_c = 2 * (s_state - t_state) * mask / (q99 - q01)
        offset = student[rel.OFFSET_KEY]
        worst_c = max(worst_c, float(np.abs(offset[:, 0, :7].cpu().numpy() - want_c).max()),
                      float(offset[:, 0, 7:].abs().max()))
        # ---- W and S
        index = torch.full((len(picks),), 4, device=device)
        with torch.no_grad():
            calls.clear()
            loss_with = float(objective.opsd_loss(student, teacher, states, times, delta, index=index, prefix_len=prefix_len))
            kind, x_in, tau, valid, v_out = calls[0]  # the teacher's call comes first
            assert valid is not None
            x = states[torch.arange(len(picks), device=device), index]
            shifted, ok_valid = shift_tail(x, delta)
            want_x = (shifted + (1 - tau).view(-1, 1, 1) * offset).masked_fill(~ok_valid[..., None], 0)
            worst_w = max(worst_w, float((x_in.masked_fill(~valid[..., None], 0) - want_x).abs().max()))
            zeroed = dict(student)
            zeroed[rel.OFFSET_KEY] = torch.zeros_like(offset)
            # the loss recomputed from the two spied velocities with the target v_teacher + c (hand-added here)
            student_v = calls[1][4]
            assert calls[1][3] is None  # the student's call (no key mask)
            d = objective.dim
            o = cfg.opsd
            manual = float(objective_module.opsd_distill_loss(
                student_v[:, :, :d], (v_out + offset)[:, :, :d], delta, o.loss_window, o.loss_window_execution,
                objective.scales, objective.freq_weight, o.trust_teacher_scale, o.trust_delay_scale))
            worst_v = max(worst_v, abs(manual - loss_with) / max(loss_with, 1e-12))
            j, e = oracle(offset, s_state, t_state, x, delta)
            joint["correct"].append(j)
            eps_worst = max(eps_worst, e)
            calls.clear()
            loss_without = float(objective.opsd_loss(zeroed, teacher, states, times, delta, index=index,
                                                     prefix_len=prefix_len))
            joint["zeroed"].append(oracle(zeroed[rel.OFFSET_KEY], s_state, t_state, x, delta)[0])
            flipped = dict(student)
            flipped[rel.OFFSET_KEY] = -offset
            calls.clear()
            objective.opsd_loss(flipped, teacher, states, times, delta, index=index, prefix_len=prefix_len)
            joint["sign flipped"].append(oracle(-offset, s_state, t_state, x, delta)[0])
            moved = float(np.abs((s_state - t_state)[:, :6]).max())
            joint.setdefault("state change", []).append(moved)
        losses.append((loss_with, loss_without, float(np.abs(want_c).max())))
        print(f"  batch {b}: loss with offset {loss_with:.4f}, offset zeroed {loss_without:.4f}, max |c| {losses[-1][2]:.3f}")
    print(f"P prefix in the student trajectory vs hand formula: max {worst_p:.2e}; offset vs hand formula: max {worst_c:.2e}")
    print(f"W teacher input vs shift_tail(x) + (1 - tau) c: max {worst_w:.2e}; loss from target v + c (relative "
          f"difference): {worst_v:.2e}")
    print("J teacher estimate vs student target in absolute joints (rad), max per batch: " + "; ".join(
        f"{k} " + " ".join(f"{v:.1e}" if k == "correct" else f"{v:.3f}" for v in vals) for k, vals in joint.items()))
    print(f"J noise estimates agree: max {eps_worst:.1e}")
    # the mutations must be off by about the state change (the frame error), the correct one by ~float error
    sens = all(z > 0.5 * m and f > 0.5 * m for z, f, m in zip(joint["zeroed"], joint["sign flipped"], joint["state change"]))
    ok &= max(joint["correct"]) < 1e-4 and eps_worst < 1e-5 and sens
    print(f"J mutations detected in every batch: {sens}")
    better = sum(w < wo for w, wo, _ in losses)
    print(f"S loss with the offset below the zeroed one in {better}/{len(losses)} batches "
          f"(mean {np.mean([w for w, _, _ in losses]):.4f} vs {np.mean([wo for _, wo, _ in losses]):.4f})")
    ok &= worst_p < 1e-4 and worst_c < 1e-5 and worst_w < 1e-5 and worst_v < 1e-5 and better == len(losses)

    # ---- X
    refused = []
    if ABS:
        try:
            objective._load_teacher(ABS)
            refused.append(False)
        except ValueError:
            refused.append(True)
    else:
        refused.append(True)  # not checked: no ABS_MODEL given
    student, teacher, states, times, delta, prefix_len = source.next_batch()
    student.pop(rel.OFFSET_KEY)
    try:
        objective.opsd_loss(student, teacher, states, times, delta, prefix_len=prefix_len)
        refused.append(False)
    except ValueError:
        refused.append(True)
    print(f"X absolute teacher refused {refused[0]}, batch without offset refused {refused[1]}")
    ok &= all(refused)
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    run()
