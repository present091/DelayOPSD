"""Trainer for both stages: the masked-FM teacher on the demonstrations and the OPSD student on rollout-buffer pairs.

One GPU, or one process per GPU with averaged gradients (ddp.py, ``torchrun``-free: OPSD_GPUS / RANK / WORLD_SIZE).

    python -m delayopsd.train --config_path=configs/pi05_libero/student.json --output_dir=outputs/student

Per update: [dataset batch -> lambda_fm * L_FM + lambda_mask * L_mask] + [pair batch -> lambda_opsd * L_OPSD], each
backpropagated on its own (the gradients add up to those of the sum), then the step on fp32 masters (master_fp32),
the teacher update (teacher_mode=ema) and the student EMA. Checkpoints hold the live weights (pretrained_model) and,
with student_ema_decay, the EMA weights (pretrained_model_ema) -- the ones evaluated.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.common.train_utils import save_checkpoint
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.envs import make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle

from . import ddp
from .config import OPSDTrainConfig
from .data.buffer import BufferPairSource, MultiBufferSource
from .data.utils import prepare_batch, validate_local_checkpoint
from .objective import OPSDObjective
from .precision import CpuEMA, MasterWeights, OffloadAdamW, swapped_weights
from .relative import is_relative


def load_adapter(policy, path, objective):
    """Load a checkpoint's PEFT adapter into the student (resume). Checks that every adapter tensor now equals the
    file and that the frozen teacher kept LoRA B = 0."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    if not hasattr(policy, "peft_config"):
        raise ValueError("a PEFT adapter needs a PEFT student (--peft.*)")
    state = load_file(f"{path}/adapter_model.safetensors")
    result = set_peft_model_state_dict(policy, state)
    if result.unexpected_keys:
        raise ValueError(f"adapter {path}: unexpected keys {result.unexpected_keys[:5]}")
    # pi0.5 adapters store the fully trained modules as "<module>.modules_to_save.default.<name>", the parameter's
    # own name: copy those directly, then check every file tensor against the live parameter
    params = dict(policy.named_parameters())
    with torch.no_grad():
        for key, value in state.items():
            if ".modules_to_save.default." in key and key in params:
                params[key].data.copy_(value.to(params[key].device, params[key].dtype))

    def live(key):  # the parameter a file key names: itself, or a LoRA weight with its ".default" adapter name
        if key in params:
            return params[key]
        for tag in (".lora_A.", ".lora_B."):
            if tag in key:
                name = key.replace(tag, tag + "default.")
                if name in params:
                    return params[name]
        return None

    wrong = [k for k in state if live(k) is None
             or not torch.equal(live(k).detach().cpu(), state[k].to(live(k).dtype))]
    if wrong:
        raise ValueError(f"adapter {path}: {len(wrong)} of {len(state)} tensors not loaded ({wrong[:3]})")
    if objective.ema is not None and any(objective.ema[n].abs().max() > 0 for n in objective.ema if "lora_B" in n):
        raise ValueError("adapter: the frozen teacher snapshot already holds a LoRA update")
    print(f"student adapter loaded from {path} ({len(state)} tensors)", flush=True)


def load_full_weights(policy, path, objective):
    """load_adapter for a student without PEFT (the stage-1 teacher): the trainable tensors of model.safetensors."""
    from safetensors.torch import load_file

    state = load_file(f"{path}/model.safetensors")
    missing = [n for n, _ in objective.trainable if f"model.{n}" not in state]
    if missing:
        raise ValueError(f"resume {path}: {len(missing)} trainable tensors missing ({missing[:3]})")
    for n, p in objective.trainable:
        p.data.copy_(state[f"model.{n}"].to(p.device, p.dtype))
    print(f"loaded {len(objective.trainable)} trainable tensors from {path}", flush=True)


@torch.no_grad()
def resume_from_checkpoint(path, policy, objective, master, optimizer, scheduler, student_ema):
    """opsd.resume_from: restore a checkpoint's training state into the freshly built run; returns its step."""
    from pathlib import Path


    from lerobot.optim.optimizers import load_optimizer_state
    from lerobot.optim.schedulers import load_scheduler_state
    from lerobot.utils.random_utils import load_rng_state

    path = Path(path)
    state_dir = path / "training_state"
    load_weights = load_adapter if hasattr(policy, "peft_config") else load_full_weights
    if student_ema is not None:
        load_weights(policy, path / "pretrained_model_ema", objective)
        ema = student_ema.state_dict()
        for n, p in objective.trainable:
            ema[n].copy_(p.detach().float())
            student_ema.cast[n].copy_(ema[n])
    load_weights(policy, path / "pretrained_model", objective)
    if master is not None:
        for _, param, fp32 in master.pairs:
            fp32.copy_(param.detach().float())
    load_optimizer_state(optimizer, state_dir)
    if scheduler is not None:
        load_scheduler_state(scheduler, state_dir)
    load_rng_state(state_dir)
    step = int(json.loads((state_dir / "training_step.json").read_text())["step"])
    print(f"resumed from {path} at step {step}", flush=True)
    return step


def reseed(source, seed, step):
    """New numpy generators for a pair source (and the sources inside a MultiBufferSource) on resume."""
    import numpy as np

    for j, part in enumerate([source, *getattr(source, "sources", [])] if source is not None else []):
        if hasattr(part, "rng"):
            part.rng = np.random.default_rng([seed, step, 0, j])


def validate_scope(cfg):
    if cfg.policy is None or cfg.policy.type not in {"pi05", "smolvla"}:
        raise ValueError("Requires policy.type=pi05 or smolvla")
    if cfg.resume or cfg.ema.enable or cfg.wandb.enable or cfg.policy.push_to_hub:
        raise ValueError("Use --opsd.resume_from and --opsd.student_ema_decay; W&B and Hub publishing are not supported")
    if cfg.dataset.streaming or cfg.dataset.eval_split:
        raise ValueError("Use a map-style dataset without eval split")
    if cfg.steps < 1 or cfg.batch_size < 1 or cfg.log_freq < 1:
        raise ValueError("steps, batch_size and log_freq must be positive")


def make_pair_source(cfg, policy, preprocessor, env_pre, seed):
    if cfg.opsd.buffer_md:
        source = MultiBufferSource(cfg, policy, preprocessor, env_pre, seed=seed)
        logging.info("multi-delay buffer pairs: %s | deltas=%s prefix_len<=%d", source.stats, source.delta,
                     source.prefix_len)
    else:
        source = BufferPairSource(cfg, policy, preprocessor, env_pre, seed=seed)
        logging.info("buffer pairs: %s | delta=%d prefix_len=%d", source.stats, source.delta, source.prefix_len)
    return source


@parser.wrap()
def train(cfg: OPSDTrainConfig):
    cfg.validate()
    validate_scope(cfg)
    validate_local_checkpoint(cfg.policy.pretrained_path)
    logging.basicConfig(level=logging.INFO)
    rank, world = ddp.init()
    main = rank == 0  # the rank that logs and saves
    set_seed(cfg.seed if cfg.seed is not None else 0)
    device = torch.device(cfg.policy.device)

    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))
        # PEFT wraps the full policy; the flow helpers operate on the wrapped policy, whose modules carry the
        # adapters in place.
        runtime_policy = policy.get_base_model()
    else:
        runtime_policy = policy
    if rank:
        # The model was built from the shared seed (same LoRA init on every rank); the flow noise and times drawn
        # from here on must differ between the ranks.
        set_seed((cfg.seed or 0) + 7919 * rank)
    source_seed = (cfg.seed or 0) + 100003 * rank  # each rank draws its own pairs

    # The start checkpoint's own processors, for the pairs and the dataset alike.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)},
                                "rename_observations_processor": {"rename_map": cfg.rename_map}},
    )
    # The action space is the checkpoint's processors' (relative or absolute); the config must agree.
    if bool(getattr(cfg.policy, "use_relative_actions", False)) != is_relative(preprocessor):
        raise ValueError(f"policy.use_relative_actions={getattr(cfg.policy, 'use_relative_actions', False)} but "
                         f"the preprocessor of {cfg.policy.pretrained_path} is "
                         f"{'relative' if is_relative(preprocessor) else 'absolute'}")
    if cfg.env is not None:  # simulation buffers store raw env observations
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    else:  # robot buffers store observations already in the policy's feature layout
        env_pre = lambda x: x  # noqa: E731
    policy.train()
    # PI05 trains its VLM only through PEFT adapters; SmolVLA (450M) trains its LM in full.
    objective = OPSDObjective(runtime_policy, cfg.opsd,
                              allow_vlm_training=cfg.peft is not None or cfg.policy.type == "smolvla")
    pairs = None
    if cfg.opsd.lambda_opsd > 0:
        pairs = make_pair_source(cfg, runtime_policy, preprocessor, env_pre, seed=source_seed + 11)
    if cfg.opsd.master_fp32:
        master = MasterWeights(objective.trainable)
        if cfg.opsd.adam_offload:
            o = cfg.optimizer
            optimizer = OffloadAdamW(master.params, lr=o.lr, betas=o.betas, eps=o.eps, weight_decay=o.weight_decay)
        else:
            optimizer = cfg.optimizer.build(master.params)
        scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
        exact = master.by_name(objective.trainable)
    else:
        master, exact = None, None
        optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    student_ema = CpuEMA(objective.trainable, cfg.opsd.student_ema_decay) if cfg.opsd.student_ema_decay else None
    start_step = 1
    if cfg.opsd.resume_from:
        start_step = 1 + resume_from_checkpoint(cfg.opsd.resume_from, policy, objective, master, optimizer,
                                                scheduler, student_ema)
        reseed(pairs, source_seed, start_step)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        pin_memory=device.type == "cuda", persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    batches = cycle(loader) if cfg.opsd.uses_dataset else None
    if main:
        cfg.output_dir.mkdir(parents=True, exist_ok=bool(cfg.opsd.resume_from))
        cfg.save_pretrained(cfg.output_dir)
    logging.info("trainable params: %.1fM", sum(p.numel() for _, p in objective.trainable) / 1e6)
    stepped = master.params if master is not None else [p for _, p in objective.trainable]
    if world > 1:
        if ddp.spread(stepped):
            raise RuntimeError("the ranks do not start from the same trainable weights")
        logging.info("rank %d of %d: effective OPSD batch %d", rank, world, world * cfg.opsd.opsd_batch_size)

    metrics_path = cfg.output_dir / "metrics.jsonl"
    if main and start_step > 1 and metrics_path.exists():  # keep the interrupted run's records up to the resume step
        lines = [line for line in metrics_path.read_text().splitlines()
                 if line.strip() and json.loads(line).get("step", 0) < start_step]
        metrics_path.write_text("".join(f"{line}\n" for line in lines))
    pair_s = 0.0
    with open(metrics_path if main else os.devnull, "a" if start_step > 1 else "w") as metrics_file:
        pbar = tqdm(range(start_step, cfg.steps + 1), desc="training", disable=not main,
                    initial=start_step - 1, total=cfg.steps)
        for step in pbar:
            opsd_active = cfg.opsd.lambda_opsd > 0 and step > cfg.opsd.opsd_start_step
            if opsd_active:
                start = time.perf_counter()
                pair_batch = pairs.next_batch()
                pair_s = time.perf_counter() - start
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # The loss terms are backpropagated one at a time, so the dataset activations are freed before the
            # OPSD graph is built.
            if cfg.opsd.uses_dataset:
                data = prepare_batch(next(batches), dataset.meta.camera_keys, cfg.rename_map, preprocessor)
                full, masked = objective.dataset_losses(data)
                (cfg.opsd.lambda_fm * full + cfg.opsd.lambda_mask * masked).backward()
                full, masked = full.detach(), masked.detach()
                del data
            else:
                full = masked = torch.zeros((), device=device)
            w_opsd = cfg.opsd.opsd_weight(step)
            distilled = torch.zeros((), device=device)
            if opsd_active:
                *pair_inputs, prefix_len = pair_batch
                distilled = objective.opsd_loss(*pair_inputs, prefix_len=prefix_len)
                (w_opsd * distilled).backward()
                distilled = distilled.detach()
            loss = cfg.opsd.lambda_fm * full + cfg.opsd.lambda_mask * masked + w_opsd * distilled
            if master is not None:
                master.grads_to_master()
            if world > 1:
                ddp.average_gradients(stepped, world)
            grad_norm = torch.nn.utils.clip_grad_norm_(stepped, cfg.optimizer.grad_clip_norm or float("inf"))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # don't hold gradients through the next forward
            if master is not None:
                master.master_to_model()
            if scheduler is not None:
                scheduler.step()
            objective.update_teacher(exact)
            if student_ema is not None:
                student_ema.update(exact)
            update_s = time.perf_counter() - start

            log_now = step % cfg.log_freq == 0 or step == 1 or step == cfg.steps
            if log_now and world > 1:
                loss, full, masked, distilled = ddp.mean_scalars([loss, full, masked, distilled], world)
            if log_now and main:
                record = {
                    "step": step, "loss": loss.item(), "loss_full": full.item(), "loss_mask": masked.item(),
                    "loss_opsd": distilled.item(), "w_opsd": w_opsd, "grad_norm": float(grad_norm),
                    "pair_s": pair_s, "update_s": update_s,
                    "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
                }
                if opsd_active:
                    record["pair_deltas"] = sorted(int(d) for d in pair_batch[4].tolist())
                    record["pairs"] = pairs.stats["pairs"]
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                pbar.set_postfix(loss=f"{record['loss']:.4f}", opsd=f"{record['loss_opsd']:.4f}")
            save_now = step == cfg.steps or (cfg.save_freq > 0 and step % cfg.save_freq == 0)
            if main and cfg.save_checkpoint and save_now:
                checkpoint = cfg.output_dir / "checkpoints" / f"{step:06d}"
                save_checkpoint(checkpoint, step, cfg, policy, optimizer, scheduler, preprocessor, postprocessor)
                if cfg.opsd.teacher_mode == "ema":
                    ema = objective.cpu_ema.state_dict() if objective.cpu_ema is not None else objective.ema
                    torch.save(ema, checkpoint / "opsd_ema_teacher.pt")
                if student_ema is not None:
                    # same files as pretrained_model, the trainable weights taken from the student EMA
                    ema_dir = checkpoint / "pretrained_model_ema"
                    shutil.copytree(checkpoint / "pretrained_model", ema_dir)
                    with swapped_weights(objective.trainable, student_ema.state_dict()):
                        policy.save_pretrained(ema_dir)
                tqdm.write(f"checkpoint saved: {checkpoint}")
    # the EMA updates run on background threads: let the last one finish before the interpreter shuts down
    if student_ema is not None:
        student_ema.join()
    if objective.cpu_ema is not None:
        objective.cpu_ema.join()
    ddp.finish(world)


if __name__ == "__main__":
    train()
