"""OPSD student: the base policies trained on the buffers' pairs to match the frozen stage-1 teacher (opsd.opsd_loss;
the only loss -- no flow matching on the expert data). A batch draws the inference delay of every pair uniformly from
0..4. train_flow.py's optimiser, EMA of the parameters (the saved <epoch>/ policies; the raw ones go to raw/<epoch>/),
no evaluation while training (evaluate with eval_sweep.py --methods realtime --config.model.simulated-delay 5: the
student takes the hard prefix as training-time RTC's checkpoints do).

    uv run src/train_opsd.py --buffer-dir outputs/buffers/s2000 --extra-buffer-dirs outputs/buffers/s20{01..15} \
        --level-paths worlds/l/catapult.json --load-dir assets/bc/24 --teacher-dir outputs/teacher/7 \
        --output-dir outputs/student_catapult --num-epochs 32
"""

import dataclasses
import os
import pathlib
import time
from typing import Sequence

# jax 0.4.35's XLA aborts compiling this train step ("reduction-layout-normalizer must run before code generation")
# unless the Triton GEMM path is off.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false").strip()

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

import eval_sweep  # noqa: E402
import model as _model  # noqa: E402
import opsd  # noqa: E402
import train_stage1  # noqa: E402

DELAYS = tuple(range(opsd.MAX_DELAY + 1))  # the training delays, drawn uniformly


@dataclasses.dataclass(frozen=True)
class Config:
    buffer_dir: str  # collect_buffer.py's output
    load_dir: str  # holds policies/<level>.pkl: the student's start (the buffer policy)
    teacher_dir: str  # holds policies/<level>.pkl: the stage-1 teacher
    output_dir: str
    level_paths: Sequence[str] = eval_sweep.LEVEL_PATHS
    shard: str = "0/1"
    # more collect_buffer.py outputs (other seeds): per delay, every buffer's kept pairs are appended
    extra_buffer_dirs: Sequence[str] = ()
    batch_size: int = 512
    # An "epoch" is steps_per_epoch updates (train_flow's epoch on the 1M-step expert data is 1952); a checkpoint
    # is written after each.
    num_epochs: int = 32
    steps_per_epoch: int = 1952
    seed: int = 0
    # exp(-r / trust_scale) on the teacher's position r; 0 = off (the Kinetix setting)
    trust_scale: float = 0.0
    ema_decay: float = 0.999
    num_flow_steps: int = 5
    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    model: _model.ModelConfig = _model.ModelConfig(simulated_delay=5)


def npz_memmap(path: pathlib.Path) -> dict:
    """The arrays of an uncompressed .npz (np.savez) as read-only memmaps, so a buffer larger than memory can be read
    a slice at a time."""
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as raw:
        for info in z.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{path}: {info.filename} is compressed"
            with z.open(info) as fp:
                version = np.lib.format.read_magic(fp)
                shape, fortran, dtype = np.lib.format._read_array_header(fp, version)
                header = fp.tell()
            assert not fortran
            raw.seek(info.header_offset)
            local = raw.read(30)
            start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
            out[info.filename.removesuffix(".npy")] = np.memmap(path, dtype, "r", start + header, shape)
    return out


def kept_slices(ok: np.ndarray, chunk: int = 65536):
    """(row, slice) over the kept pairs of a buffer, which come first in every row (collect_pairs)."""
    for r in range(ok.shape[0]):
        n = int(ok[r].sum())
        assert ok[r, :n].all(), "kept pairs are not first"
        for s in range(0, n, chunk):
            yield r, slice(s, min(n, s + chunk))


def varying_dims(paths: Sequence[pathlib.Path]) -> tuple[np.ndarray, np.ndarray]:
    """(the first kept obs, mask of the dims whose bits differ from it in any kept obs or teacher_obs of paths)."""
    ref = mask = None
    for path in paths:
        z = npz_memmap(path)
        ok = np.asarray(z["ok"])
        for k in ("obs", "teacher_obs"):
            for r, s in kept_slices(ok):
                x = np.asarray(z[k][r, s]).view(np.uint32)
                if ref is None:
                    ref, mask = x[0].copy(), np.zeros(x.shape[1], bool)
                mask |= (x != ref).any(0)
    return ref.view(np.float32), mask


def main(config: Config):
    level_paths = opsd.shard(config.level_paths, config.shard)
    _, _, _, _, obs_dim, action_dim = opsd.setup(level_paths)

    # Only the observation dims that vary somewhere in the buffers (bitwise, over obs and teacher_obs) are held; the
    # others are each level's constants and are filled back in from its first pair on the GPU. Lossless (the loss sees
    # the full float32 observations) and ~15x smaller, so the 16 buffers of a level fit on the GPU whole.
    buffer_dirs = (config.buffer_dir, *config.extra_buffer_dirs)
    npz = lambda d, level_path: pathlib.Path(d) / f"{opsd.level_name(level_path)}.npz"  # noqa: E731
    start = time.perf_counter()
    found = [varying_dims([npz(d, level_path) for d in buffer_dirs]) for level_path in level_paths]
    templates = np.stack([ref for ref, _ in found])
    obs_idx = np.flatnonzero(np.any([mask for _, mask in found], axis=0))
    print(f"compact obs: {len(obs_idx)} of {templates.shape[1]} dims vary ({time.perf_counter() - start:.0f} s)",
          flush=True)

    def load(buffer_dir: str, level_path: str) -> dict:
        z = npz_memmap(npz(buffer_dir, level_path))
        out = {k: np.asarray(z[k]) for k in z if k not in ("obs", "teacher_obs")}
        for k in ("obs", "teacher_obs"):
            x = np.zeros(z[k].shape[:2] + (len(obs_idx),), z[k].dtype)
            for r, s in kept_slices(out["ok"]):
                x[r, s] = z[k][r, s][:, obs_idx]
            out[k] = x
        del z
        out["delays"] = tuple(int(d) for d in out["delays"])
        return out

    def merge(parts: list[dict]) -> dict:
        """Rows = the delays any buffer holds, ascending; a row = every buffer's kept pairs of that delay, in
        buffer order, then padding (ok False)."""
        delays = tuple(sorted({d for p in parts for d in p["delays"]}))
        rows = [[(p, p["delays"].index(d), np.flatnonzero(p["ok"][p["delays"].index(d)]))
                 for p in parts if d in p["delays"]] for d in delays]
        sizes = [sum(len(i) for _, _, i in row) for row in rows]
        out = {"delays": delays, "ok": np.stack([np.arange(max(sizes)) < n for n in sizes])}
        for k in ("obs", "teacher_obs", "prefix"):
            x = np.zeros((len(rows), max(sizes), *parts[0][k].shape[2:]), parts[0][k].dtype)
            for r, row in enumerate(rows):
                at = 0
                for p, pr, i in row:
                    x[r, at : at + len(i)] = p[k][pr][i]
                    at += len(i)
            out[k] = x
        return out

    data = []
    start = time.perf_counter()
    for level_path in level_paths:
        parts = [load(d, level_path) for d in buffer_dirs]
        data.append(parts[0] if len(parts) == 1 else merge(parts))
        del parts
        print(f"loaded {level_path} ({time.perf_counter() - start:.0f} s)", flush=True)
    # row_of maps a delay to its row in the buffer
    delays = [d["delays"] for d in data]
    assert len(set(delays)) == 1 and set(DELAYS) <= set(delays[0]), delays
    row_of = np.full(opsd.MAX_DELAY + 1, -1)
    row_of[list(delays[0])] = np.arange(len(delays[0]))
    # the kept pairs come first (collect_pairs, merge): a pair of delay d is index < count[row_of[d]]
    counts = np.stack([d["ok"].sum(-1) for d in data])
    print(f"pairs per level and delay {delays[0]}:\n", counts, flush=True)
    assert counts[:, row_of[list(DELAYS)]].min() > 0, "a level has no pair at a training delay"
    # levels may hold different numbers of pairs once buffers are merged: pad to the largest
    size = max(d["ok"].shape[1] for d in data)
    pad = lambda x: np.pad(x, [(0, 0), (0, size - x.shape[1])] + [(0, 0)] * (x.ndim - 2))  # noqa: E731
    host = {k: np.stack([pad(d.pop(k)) for d in data]) for k in ("obs", "teacher_obs", "prefix")}
    del data
    row_of = jnp.asarray(row_of)

    def full_obs(x: jax.Array, data: dict) -> jax.Array:
        """A batch of compact observations as the policy's float32 input."""
        x = x.astype(jnp.float32)
        return jnp.broadcast_to(data["template"], (x.shape[0], data["template"].shape[-1])).at[:, obs_idx].set(x)

    students = opsd.load_policies(config.load_dir, level_paths)
    teachers = opsd.load_policies(config.teacher_dir, level_paths)
    first = opsd.make_policy(jax.tree.map(lambda x: x[0], students), config.model, obs_dim, action_dim)
    graphdef = nnx.graphdef((first, train_stage1.make_optimizer(first, config)))

    @jax.jit
    @jax.vmap
    def init(state_dict):
        policy = opsd.make_policy(state_dict, config.model, obs_dim, action_dim)
        return nnx.state((policy, train_stage1.make_optimizer(policy, config)))

    @jax.jit
    @jax.vmap
    def init_ema(train_state):
        policy, _ = nnx.merge(graphdef, train_state)
        return nnx.state(policy, nnx.Param)

    @jax.jit
    @jax.vmap
    def train_epoch(rng: jax.Array, train_state: nnx.State, ema, teacher_dict, data: dict):
        teacher = opsd.make_policy(teacher_dict, config.model, obs_dim, action_dim)

        def train_minibatch(carry, _):
            rng, train_state, ema = carry
            policy, optimizer = nnx.merge(graphdef, train_state)
            rng, delay_rng, pick_rng, key = jax.random.split(rng, 4)
            delay = jax.random.choice(delay_rng, jnp.asarray(DELAYS), (config.batch_size,))
            row = row_of[delay]
            pick = jnp.floor(jax.random.uniform(pick_rng, (config.batch_size,)) * data["count"][row]).astype(jnp.int32)

            def loss_fn(policy: _model.FlowPolicy):
                return opsd.opsd_loss(
                    policy, teacher, key, full_obs(data["obs"][row, pick], data),
                    full_obs(data["teacher_obs"][row, pick], data),
                    data["prefix"][row, pick], delay, config.trust_scale, config.num_flow_steps,
                )

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            optimizer.update(grads)
            ema = opsd.ema_update(ema, policy, config.ema_decay)
            return (rng, nnx.state((policy, optimizer)), ema), loss

        (rng, train_state, ema), losses = jax.lax.scan(
            train_minibatch, (rng, train_state, ema), None, length=config.steps_per_epoch
        )
        return rng, train_state, ema, jnp.mean(losses)

    train_state = init(students)
    ema = init_ema(train_state)
    out = pathlib.Path(config.output_dir)
    rngs = jax.random.split(jax.random.key(config.seed), len(level_paths))
    data = jax.device_put({**host, "count": counts, "template": templates})
    for epoch in range(config.num_epochs):
        start = time.perf_counter()
        rngs, train_state, ema, loss = train_epoch(rngs, train_state, ema, teachers, data)
        opsd.save_policies(out / "raw" / str(epoch), level_paths, graphdef, train_state)
        opsd.save_policies(out / str(epoch), level_paths, graphdef, train_state, ema)
        loss = np.asarray(loss)
        print(f"epoch {epoch}: {time.perf_counter() - start:.0f} s  opsd {loss.mean():.5f}  "
              f"per level {np.array2string(loss, precision=4)}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
