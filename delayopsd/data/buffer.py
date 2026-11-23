"""OPSD pairs from a rollout buffer of the start policy (delayopsd.collect).

A buffer holds whole asynchronous episodes of a fixed policy at R:k. A pair is built at every replan step s:

    student obs  = frame s
    prefix       = the running chunk's actions for [s, s+k), read from the previous chunk (inferred at s' < s,
                   executing its index s - s' at step s); none on an episode's first chunk
    teacher obs  = frame s+delta (delta == k, the frame those actions produced)

A synchronous buffer (R:0) has no running chunk: the chunk inferred at s executes [s, s+R) itself, and the prefix is
that chunk's own first k actions -- again the actions that produced frame s+delta.

The student's flow states are sampled from the current student on frame s with that prefix (delayopsd.flow), so they are
on-policy for the network being trained. Tasks are drawn equally often; a task is the episode's ``group`` in
pairs.npz (LIBERO suite and task id), so one root may hold several suites' episodes.

``build_index`` first copies the frames pairs use (every replan step and its teacher frame) to <root>/index, since a
compressed episode npz has to be decoded whole on every read: camera images as one lossless PNG per frame
(<name>.png.bin + byte offsets <name>.png.off.npy), the low-dimensional keys as uncompressed memmaps.

Failure rules (a successful episode is never filtered): ``fail_stall_radius`` / ``fail_stall_margin`` keep a failed
episode's pairs until ``margin`` steps after its end effector stalls; ``fail_sync_root`` drops a failed episode whose
scene also fails in a synchronous buffer (the failure was not caused by the delay).
"""
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from lerobot.envs import preprocess_observation

from .. import relative
from ..flow import observation_inputs, sample_with_trajectory
from .utils import collate_observations


def _is_image(key):
    return "pixels" in key.split("/")


def _encode(frame):
    import cv2

    ok, buf = cv2.imencode(".png", np.ascontiguousarray(frame))  # channel order is kept as is: lossless either way
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


class PngFrames:
    """Frames of one image key stored as <name>.png.bin + <name>.png.off.npy; frames[row] -> decoded uint8 array."""

    def __init__(self, stem):
        self.offsets = np.load(f"{stem}.png.off.npy")
        self.data = np.memmap(f"{stem}.png.bin", dtype=np.uint8, mode="r") if self.offsets[-1] else np.zeros(0, np.uint8)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, row):
        import cv2

        frame = cv2.imdecode(np.asarray(self.data[self.offsets[row]:self.offsets[row + 1]]), cv2.IMREAD_UNCHANGED)
        return frame if frame.ndim == 3 else frame[..., None]


def _stem(index_dir, key):
    """index_dir / file stem of an obs key ("obs/" stripped, as pairs.npz keys are)."""
    key = key[4:] if key.startswith("obs/") else key
    return Path(index_dir) / key.replace("/", "__")


def _frames_exist(index_dir, key):
    stem = _stem(index_dir, key)
    return Path(f"{stem}.npy").exists() or Path(f"{stem}.png.off.npy").exists()


def load_frames(index_dir, key):
    """One key's frames of an index: a PngFrames for PNG images, else the .npy memmap."""
    stem = _stem(index_dir, key)
    if Path(f"{stem}.png.off.npy").exists():
        return PngFrames(stem)
    return np.load(f"{stem}.npy", mmap_mode="r")


def _task_key(meta):
    if meta.get("group") is None:  # an episode collected without an episode key
        return meta["task_description"]
    return f"{meta['group']}/{meta['task_id']}"


def _episode_plan(path, delta, stride=0):
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        steps = z["chunk_obs_step"].astype(np.int64)
        T = int(z["action"].shape[0])
        keys = [k for k in z.files if k.startswith("obs/")]
        stored = z["obs_step"].astype(np.int64) if "obs_step" in z.files else None  # None: every frame
    students = [int(s) for s in (range(0, T, stride) if stride else steps) if s + delta <= T]
    if stored is not None and meta.get("inputs") == "sent":
        # Real-robot buffers (real_robot.buffer.yam_buffer) hold only the inputs recorded as sent; an episode's first
        # request is not recorded, so a pair is built only where both of its frames were sent. Other buffers: as before.
        have = set(stored.tolist())
        students = [s for s in students if s in have and s + delta in have]
    frames = sorted(set(students) | {s + delta for s in students})
    if stored is not None:
        missing = sorted(set(frames) - set(stored.tolist()))
        if missing:
            raise ValueError(f"{path} holds {len(stored)} of {T + 1} frames and not steps {missing[:5]}"
                             f"{' ...' if len(missing) > 5 else ''}: delta {delta} / stride {stride} need frames "
                             "the collection did not keep (collect with --frames=all)")
    return dict(path=str(path), meta=meta, T=T, students=students, frames=frames, keys=keys)


def _fill(args):
    plan, offset, index_dir, shapes, delta, k, write_frames, stride = args
    with np.load(plan["path"]) as z:
        rows = plan["frames"]
        if "obs_step" in z.files:  # the episode holds some frames only: step -> row
            rows = np.searchsorted(z["obs_step"], plan["frames"])
        png = {}  # image key -> the episode's frames as PNG bytes, appended in episode order by build_index
        for key in plan["keys"] if write_frames else []:
            if _is_image(key):
                png[key] = [_encode(frame) for frame in z[key][rows]]
                continue
            dst = np.lib.format.open_memmap(f"{_stem(index_dir, key)}.npy", mode="r+")
            dst[offset:offset + len(plan["frames"])] = z[key][rows]
            dst.flush()
            del dst
        chunks, steps = z["chunks"], z["chunk_obs_step"]
    row = {s: i for i, s in enumerate(plan["frames"])}
    by_step = {int(s): i for i, s in enumerate(steps)}
    prefix = np.zeros((len(plan["students"]), k, chunks.shape[-1]), np.float32)
    prefix_len = np.zeros(len(plan["students"]), np.int64)
    sync = int(plan["meta"]["k"]) == 0  # the chunk inferred at s executes from s on
    spliced = np.zeros(len(plan["students"]), bool)
    if stride:
        # The chunk inferred at steps[i] takes over k steps later (the first one at once) and runs until the next
        # one does: executed[t] is the model-space action that ran at step t.
        start = np.array([0] + [int(s) + int(plan["meta"]["k"]) for s in steps[1:]])
        owner = np.searchsorted(start, np.arange(plan["T"]), side="right") - 1
        executed = chunks[owner, np.arange(plan["T"]) - steps[owner]]
        for j, s in enumerate(plan["students"]):
            if s > 0:  # the first inference of an episode has no running chunk, hence no prefix
                prefix[j] = executed[s:s + k]
                prefix_len[j] = k
                spliced[j] = owner[s] != owner[s + k - 1]
    else:
        for j, s in enumerate(plan["students"]):
            i = by_step[s]
            if sync:
                prefix[j] = chunks[i][:k]
                prefix_len[j] = k
            elif i > 0:  # the first chunk of an episode has no running chunk, hence no prefix
                at = s - int(steps[i - 1])  # the running chunk's index executing at step s
                prefix[j] = chunks[i - 1][at:at + k]
                prefix_len[j] = k
    return dict(student=np.array([offset + row[s] for s in plan["students"]]),
                teacher=np.array([offset + row[s + delta] for s in plan["students"]]),
                prefix=prefix, prefix_len=prefix_len, step=np.array(plan["students"]), spliced=spliced, png=png)


def build_index(root, delta=10, k=10, workers=16, stride=0, index_name=None):
    """index_name: the index directory under root (default index / index_stride<n>), e.g. index_d<delta> when one
    root is read with several deltas (multi-delay pairs from one 10:10 --frames=all buffer)."""
    root = Path(root)
    index_dir = root / (index_name or ("index" if stride == 0 else f"index_stride{stride}"))
    if (index_dir / "pairs.npz").exists():
        return index_dir
    files = sorted((root / "episodes").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no episodes under {root / 'episodes'}")
    with ProcessPoolExecutor(workers) as pool:
        plans = list(pool.map(_episode_plan, files, [delta] * len(files), [stride] * len(files)))
    offsets = np.cumsum([0] + [len(p["frames"]) for p in plans])
    index_dir.mkdir(parents=True, exist_ok=True)
    with np.load(files[0]) as z:
        shapes = {key: (z[key].shape[1:], z[key].dtype) for key in plans[0]["keys"]}
    # Frames already in place (e.g. linked from an index of the same episodes) are reused as they are; only
    # pairs.npz is rebuilt then.
    write_frames = not all(_frames_exist(index_dir, key) for key in shapes)
    n = int(offsets[-1])
    if write_frames:
        # open_memmap makes sparse files, so a full disk only shows up midway through _fill -- and then takes down
        # every other job writing to the same disk.
        # PNG images are budgeted at 0.6 of raw (measured ~0.42 on LIBERO frames).
        need = sum(n * int(np.prod(shape)) * np.dtype(dtype).itemsize * (0.6 if _is_image(key) else 1.0)
                   for key, (shape, dtype) in shapes.items())
        free = shutil.disk_usage(index_dir).free
        margin = int(float(os.environ.get("INDEX_FREE_MARGIN_GB", 150)) * 2**30)
        if free - need < margin:
            raise OSError(f"building {index_dir} needs {need / 2**30:.0f} GiB, {free / 2**30:.0f} GiB free; refusing to "
                          f"leave less than {margin / 2**30:.0f} GiB (INDEX_FREE_MARGIN_GB)")
    for key, (shape, dtype) in shapes.items():
        if write_frames and not _is_image(key):
            np.lib.format.open_memmap(f"{_stem(index_dir, key)}.npy", mode="w+", dtype=dtype, shape=(n, *shape))
        elif not write_frames and len(frames := load_frames(index_dir, key)) != n:
            raise ValueError(f"{_stem(index_dir, key)} holds {len(frames)} frames, the episodes under "
                             f"{root / 'episodes'} {n}")
    images = [key for key in shapes if write_frames and _is_image(key)]
    sinks = {key: open(f"{_stem(index_dir, key)}.png.bin.tmp", "wb") for key in images}
    lengths = {key: [] for key in images}
    parts = []
    with ProcessPoolExecutor(workers) as pool:
        # in episode order, so each image key's frames land at the rows the memmaps use
        for part in pool.map(_fill, [(p, int(o), index_dir, shapes, delta, k, write_frames, stride)
                                     for p, o in zip(plans, offsets)]):
            for key, blobs in part.pop("png").items():
                for blob in blobs:
                    sinks[key].write(blob)
                    lengths[key].append(len(blob))
            parts.append(part)
    for key in images:
        sinks[key].close()
        if len(lengths[key]) != n:
            raise RuntimeError(f"{key}: {len(lengths[key])} PNG frames written, {n} expected")
        stem = _stem(index_dir, key)
        os.rename(f"{stem}.png.bin.tmp", f"{stem}.png.bin")
        np.save(f"{stem}.png.off.tmp.npy", np.concatenate([[0], np.cumsum(lengths[key])]).astype(np.int64))
        os.rename(f"{stem}.png.off.tmp.npy", f"{stem}.png.off.npy")  # last: its presence marks the key complete
    episode = np.concatenate([np.full(len(p["student"]), i) for i, p in enumerate(parts)])
    # the space of the stored chunks, hence of the prefix (real_robot.buffer.yam_buffer: "absolute_raw" for a
    # relative-action collector); one per buffer
    spaces = {p["meta"].get("chunks_space", "model") for p in plans}
    if len(spaces) != 1:
        raise ValueError(f"{root}: episodes with chunks in different spaces {spaces}")
    np.savez(index_dir / "pairs.tmp.npz",
             student=np.concatenate([p["student"] for p in parts]),
             teacher=np.concatenate([p["teacher"] for p in parts]),
             prefix=np.concatenate([p["prefix"] for p in parts]),
             prefix_len=np.concatenate([p["prefix_len"] for p in parts]),
             step=np.concatenate([p["step"] for p in parts]), episode=episode,
             spliced=np.concatenate([p["spliced"] for p in parts]), stride=stride,
             T=np.array([p["T"] for p in plans]), success=np.array([bool(p["meta"]["success"]) for p in plans]),
             task_id=np.array([int(p["meta"]["task_id"]) for p in plans]),
             task=np.array([p["meta"]["task_description"] for p in plans]),
             group=np.array([_task_key(p["meta"]) for p in plans]),
             keys=np.array([key[4:] for key in shapes]), delta=delta, k=k,
             prefix_from="executed_actions" if stride else "running_chunk", prefix_space=spaces.pop())
    (index_dir / "pairs.tmp.npz").rename(index_dir / "pairs.npz")
    return index_dir


def _stall_onset(args):
    """The step from which the end effector stays within `radius` (m) of the tail's centroid until the episode ends:
    the earliest stored frame i such that every frame from i on does; T (the final frame) when none before it does."""
    path, radius = args
    with np.load(path) as z:
        pos = z["obs/robot_state/eef/pos"].astype(np.float64)
        T = int(z["action"].shape[0])
        steps = z["obs_step"].astype(np.int64) if "obs_step" in z.files else np.arange(len(pos))
    onset = T
    for i in range(len(pos) - 1, -1, -1):
        tail = pos[i:]
        if np.linalg.norm(tail - tail.mean(0), axis=1).max() >= radius:
            break
        onset = int(steps[i])
    return min(onset, T)


def _scene_key(path):
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    return f"{meta['group']}/{meta['task_id']}/{meta['state_hash']}", bool(meta["success"])


def episode_scenes(root, workers=16):
    """Per episode of <root>/episodes (sorted, the index order): suite/task/initial-state key."""
    files = sorted((Path(root) / "episodes").glob("*.npz"))
    with ProcessPoolExecutor(workers) as pool:
        return [key for key, _ in pool.map(_scene_key, files, chunksize=32)]


_SOLVED = {}


def sync_solved_scenes(root, workers=16):
    """scene key -> whether any episode of that scene succeeded in a synchronous buffer (cached per process)."""
    if root not in _SOLVED:
        files = sorted((Path(root) / "episodes").glob("*.npz"))
        solved = {}
        with ProcessPoolExecutor(workers) as pool:
            for key, ok in pool.map(_scene_key, files, chunksize=32):
                solved[key] = solved.get(key, False) or ok
        _SOLVED[root] = solved
    return _SOLVED[root]


def stall_onsets(root, index_dir, radius, workers=16):
    """Per episode of the index (sorted episode files), the stall onset of _stall_onset; cached in index_dir."""
    out = Path(index_dir) / f"stall_r{radius:g}.npy"
    if out.exists():
        return np.load(out)
    files = sorted((Path(root) / "episodes").glob("*.npz"))
    with ProcessPoolExecutor(workers) as pool:
        onset = np.array(list(pool.map(_stall_onset, [(f, radius) for f in files])), dtype=np.int64)
    tmp = out.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, onset)
    tmp.rename(out)
    return onset


class BufferPairSource:
    def __init__(self, cfg, policy, preprocessor, env_pre, seed=0, root=None, k=None, index_name=None):
        """k: the pair delay (multi-delay, opsd.buffer_md) instead of the run's schedule: delta == prefix length == k.
        Usually the buffer's own R:k; with a --frames=all buffer any k (the running chunk's first k actions and the
        frame they produced), then with index_name=index_d<k> so the deltas of one root keep separate indexes."""
        opsd = cfg.opsd
        if k is not None:
            deltas = ks = [int(k)]
        else:
            schedules = opsd.schedules(policy.config.chunk_size)
            deltas = sorted({d for r, _ in schedules for d in opsd.slot_deltas(r)})
            ks = sorted({k for _, k in schedules})
        if len(ks) != 1 or deltas != ks:
            raise ValueError(f"buffer pairs need delta == k (deltas {deltas}, k {ks})")
        self.delta, self.prefix_len, self.prefix_mode = deltas[0], min(ks[0], opsd.prefix_len), opsd.prefix_mode
        index_dir = build_index(root or opsd.buffer_root, delta=self.delta, k=ks[0], stride=0,
                                index_name=index_name)
        p = np.load(index_dir / "pairs.npz")
        if int(p["delta"]) != self.delta:
            raise ValueError(f"buffer index built for delta {int(p['delta'])}, run uses {self.delta}")
        if self.prefix_len and "prefix_from" not in p.files:
            # Older indexes took the prefix from the chunk inferred at s, which is only the running chunk's
            # actions for an --rtc=trained buffer; rebuild pairs.npz (the frame memmaps are reused).
            raise ValueError(f"{index_dir / 'pairs.npz'} predates the running-chunk prefix; delete it to rebuild")
        self.frames = {key: load_frames(index_dir, key) for key in p["keys"]}
        self.student, self.teacher, self.prefix = p["student"], p["teacher"], p["prefix"]
        self.prefix_lens, self.episode, self.task = p["prefix_len"], p["episode"], p["task"]
        success, T, step = p["success"], p["T"], p["step"]
        # The unit of the per-task cut and of the task balance. Not task_id alone: it restarts at 0 in every
        # suite.
        key = p["group"] if "group" in p.files else self.task
        limit = T.astype(np.float64)  # failed episodes keep every pair unless a failure rule below applies
        if opsd.fail_stall_radius > 0:
            # a failed episode keeps its pairs until its end effector stalls (+ margin)
            onset = stall_onsets(root or opsd.buffer_root, index_dir, opsd.fail_stall_radius)
            if len(onset) != len(T):
                raise ValueError(f"{index_dir}: {len(onset)} stall onsets for {len(T)} episodes")
            limit = onset.astype(np.float64) + opsd.fail_stall_margin
        keep = success[self.episode] | (step < limit[self.episode])
        if opsd.fail_sync_root:
            # failed episodes the delay did not cause: their scene also fails synchronously
            solved = sync_solved_scenes(opsd.fail_sync_root)
            scenes = episode_scenes(root or opsd.buffer_root)
            if len(scenes) != len(T):
                raise ValueError(f"{index_dir}: {len(scenes)} episode files for {len(T)} indexed episodes")
            intrinsic = np.array([not ok and s in solved and not solved[s] for s, ok in zip(scenes, success)])
            keep &= ~intrinsic[self.episode]
            self.intrinsic_dropped = int(intrinsic.sum())
        self.pool = np.flatnonzero(keep)
        task_of = key[self.episode[self.pool]]
        tasks, counts = np.unique(task_of, return_counts=True)
        weight = 1.0 / counts[np.searchsorted(tasks, task_of)]
        weight /= weight.sum()
        failed = ~success[self.episode[self.pool]]
        natural = float(weight[failed].sum())
        self.weight = weight / weight.sum()
        self.rng = np.random.default_rng(seed)
        self.policy, self.preprocessor, self.env_pre = policy, preprocessor, env_pre
        self.relative = bool(getattr(policy.config, "use_relative_actions", False))
        if self.relative != relative.is_relative(preprocessor):
            raise ValueError("policy.use_relative_actions and the preprocessor's RelativeActions step disagree")
        space = str(p["prefix_space"]) if "prefix_space" in p.files else "model"
        if space != ("absolute_raw" if self.relative else "model"):
            raise ValueError(f"{index_dir}: prefixes stored as {space!r}, but the policy is "
                             f"{'relative' if self.relative else 'absolute'} (relative needs a buffer converted "
                             "from the relative collector, real_robot.buffer.yam_buffer, chunks_space absolute_raw)")
        if self.relative:
            self._relative_inputs(root or opsd.buffer_root, p, index_dir)
        self.batch_size = opsd.opsd_batch_size
        self.stats = {"episodes": len(T), "success": float(success.mean()), "pairs_total": len(self.student),
                      "pairs_kept": len(self.pool), "pairs_failed_kept": int((~success[self.episode[self.pool]]).sum()),
                      "pairs_spliced_kept": int(p["spliced"][self.pool].sum()) if "spliced" in p.files else 0,
                      "fail_share_natural": natural, "fail_share": float(self.weight[failed].sum()),
                      "tasks": len(tasks), "pairs": 0,
                      "intrinsic_dropped": getattr(self, "intrinsic_dropped", 0)}

    def _relative_inputs(self, root, p, index_dir):
        """Relative-action policy: the stored prefix is the running chunk's reply over [s, s+k) as sent,
        absolute joint targets (the committed actions the station sends with the next request); next_batch puts it
        in the student frame's model space with that frame's state, and gives the objective the student->teacher
        offset from both frames' states. Checked here against the executed commands a[s:s+k]: the arm joints
        equal, the gripper equal unless the robot clipped it to [0, 1]."""
        if int(p["stride"]) != 0 or str(p["prefix_from"]) != "running_chunk" or not self.prefix_len:
            raise ValueError(f"{index_dir}: relative-action pairs need replan pairs with a running-chunk prefix")
        if "agent_pos" not in self.frames:
            raise ValueError(f"{index_dir}: no agent_pos frames, needed for the relative-action states")
        files = sorted((Path(root) / "episodes").glob("*.npz"))
        if len(files) != len(p["T"]):
            raise ValueError(f"{index_dir}: {len(files)} episode files for {len(p['T'])} indexed episodes")
        k, d, used = int(p["k"]), self.prefix.shape[-1], self.prefix_lens > 0
        ran = np.zeros_like(self.prefix)
        for e, f in enumerate(files):
            with np.load(f) as z:
                a = z["action"].astype(np.float32)
            for j in np.flatnonzero((self.episode == e) & used):
                s = int(p["step"][j])
                ran[j] = a[s:s + k, :d]
        err = np.abs(self.prefix[used] - ran[used])
        grip = ran[used][..., d - 1]
        inner = (grip > 1e-3) & (grip < 1 - 1e-3)
        if err[..., :d - 1].max() > 1e-5 or err[..., d - 1][inner].max() > 1e-5 or err[..., d - 1].max() > 1e-2:
            raise ValueError(f"{index_dir}: the stored prefix is not the executed commands (arm "
                             f"{err[..., :d - 1].max():.2e}, gripper {err[..., d - 1].max():.2e})")
        self.raw_prefix = self.prefix
        self.states = self.frames["agent_pos"]

    def _observation(self, row, episode):
        raw = {}
        for key, frames in self.frames.items():
            node = raw
            *parents, leaf = key.split("/")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = np.array(frames[row])[None]  # a writable copy of the memmapped frame
        obs = preprocess_observation(raw)
        obs["task"] = [str(self.task[episode])]
        return self.env_pre(obs)

    def _pick(self, size):
        return self.pool[self.rng.choice(len(self.pool), size, p=self.weight)]

    def _load(self, picks):
        """CPU part of a batch: the student / teacher frames read and decoded (opsd.buffer_prefetch runs it ahead)."""
        return (collate_observations([self._observation(self.student[i], self.episode[i]) for i in picks]),
                collate_observations([self._observation(self.teacher[i], self.episode[i]) for i in picks]))

    @torch.no_grad()
    def next_batch(self, size=None):
        """One micro-batch in the layout the trainer builds from rollout pairs."""
        picks = self._pick(size or self.batch_size)
        return self._finish(picks, self._load(picks))

    @torch.no_grad()
    def _finish(self, picks, loaded):
        student, teacher = self.preprocessor(loaded[0]), self.preprocessor(loaded[1])
        policy, model = self.policy, self.policy.model
        device = torch.device(policy.config.device)
        batch = len(picks)
        shape = (batch, policy.config.chunk_size, policy.config.max_action_dim)
        prefix_len = torch.as_tensor(np.minimum(self.prefix_lens[picks], self.prefix_len), device=device)
        prefix = torch.zeros(shape, device=device)
        if self.relative:
            # the running chunk's absolute commands in the student frame's model space; the teacher frame's space
            # differs by a constant offset, carried to the objective in the student batch
            s_state = torch.as_tensor(np.stack([np.asarray(self.states[i]) for i in self.student[picks]]), device=device)
            t_state = torch.as_tensor(np.stack([np.asarray(self.states[i]) for i in self.teacher[picks]]), device=device)
            raw = torch.as_tensor(self.raw_prefix[picks][:, :self.prefix_len], device=device)
            stored = relative.model_space(self.preprocessor, raw, s_state)
            offset = torch.zeros(batch, 1, shape[-1], device=device)
            offset[:, :, :raw.shape[-1]] = relative.frame_offset(self.preprocessor, s_state, t_state, raw.shape[-1])
            student[relative.OFFSET_KEY] = offset
        else:
            stored = torch.as_tensor(self.prefix[picks][:, :self.prefix_len], device=device)
        prefix[:, :stored.shape[1], :stored.shape[2]] = stored  # padding dims stay 0, as in rollouts
        noise = _sample_noise(model, shape, device)
        was_training = policy.training
        policy.eval()
        try:
            _, states, times = sample_with_trajectory(
                model, observation_inputs(policy, student), noise, policy.config.num_inference_steps,
                prefix=prefix if self.prefix_len else None, prefix_len=prefix_len if self.prefix_len else None,
                prefix_mode=self.prefix_mode,
            )
        finally:
            policy.train(was_training)
        self.stats["pairs"] += batch
        delta = torch.full((batch,), self.delta, dtype=torch.long, device=device)
        return student, teacher, states.float(), times, delta, prefix_len


_ROW_NOISE = None


def _sample_noise(model, shape, device):
    """model.sample_noise, or (OPSD_TEST_ROW_NOISE=<seed>, verification only) one row per example from a fixed CPU
    generator, so a batch drawn in parts and the same batch drawn at once get the same noise."""
    global _ROW_NOISE
    seed = os.environ.get("OPSD_TEST_ROW_NOISE")
    if not seed:
        return model.sample_noise(shape, device)
    if _ROW_NOISE is None:
        _ROW_NOISE = torch.Generator().manual_seed(int(seed))
    rows = [torch.randn(shape[1:], generator=_ROW_NOISE) for _ in range(shape[0])]
    return torch.stack(rows).to(device=device, dtype=torch.float32)


@torch.no_grad()
class MultiBufferSource:
    """Multi-delay pairs (opsd.buffer_md): one BufferPairSource per "K:root", each with delta == k == K.

    Each pair of a batch draws its delay uniformly, so every delay gets the same share of the updates whatever the
    sizes of the buffers; within a delay the source's own task balancing applies. The parts are concatenated: deltas
    and prefix lengths per pair, the flow times are the same grid.
    """

    def __init__(self, cfg, policy, preprocessor, env_pre, seed=0):
        self.sources = []
        entries = [entry.partition(":")[::2] for entry in cfg.opsd.buffer_md]
        roots = [root for _, root in entries]
        for i, (k, root) in enumerate(entries):
            # one root read with several deltas (a 10:10 --frames=all buffer): an index per delta
            name = f"index_d{int(k)}" if roots.count(root) > 1 else None
            self.sources.append(BufferPairSource(cfg, policy, preprocessor, env_pre, seed=seed + 101 * (i + 1),
                                                 root=root, k=int(k), index_name=name))
        self.rng = np.random.default_rng(seed)
        self.batch_size = cfg.opsd.opsd_batch_size
        self.prefetch = bool(cfg.opsd.buffer_prefetch)
        if self.prefetch:
            from collections import deque
            from concurrent.futures import ThreadPoolExecutor
            self._next, self._executor = deque(), ThreadPoolExecutor(max_workers=1)  # 8 workers measured slower (GIL)
        self.delta = ",".join(str(s.delta) for s in self.sources)
        self.prefix_len = max(s.prefix_len for s in self.sources)

    @property
    def stats(self):
        per = {s.delta: {key: s.stats[key] for key in ("episodes", "success", "pairs_kept", "pairs")}
               for s in self.sources}
        return {"pairs": sum(v["pairs"] for v in per.values()), "per_delta": per}

    def _plan(self, size):
        counts = self.rng.multinomial(size, np.full(len(self.sources), 1.0 / len(self.sources)))
        return [(source, source._pick(int(n))) for source, n in zip(self.sources, counts) if n]

    @staticmethod
    def _load_plan(plan):
        return [source._load(picks) for source, picks in plan]

    def next_batch(self, size=None):
        size = size or self.batch_size
        if not self.prefetch:
            plan = self._plan(size)
            loaded = self._load_plan(plan)
        else:
            # opsd.buffer_prefetch: the next batch's picks are drawn now -- the same generators in the
            # same order as without prefetch, so the batches are identical -- and its frames are read and decoded on
            # a worker thread while this step trains; the student sampling stays here, with the current weights.
            # one batch ahead on one worker (two ahead on 8 workers was slower)
            while len(self._next) < 2:
                ahead = self._plan(size)
                self._next.append((ahead, [self._executor.submit(source._load, picks) for source, picks in ahead]))
            plan, futures = self._next.popleft()
            loaded = [future.result() for future in futures]
        parts = [source._finish(picks, part) for (source, picks), part in zip(plan, loaded)]
        if len(parts) == 1:
            return parts[0]
        return (collate_observations([p[0] for p in parts]), collate_observations([p[1] for p in parts]),
                torch.cat([p[2] for p in parts]), parts[0][3], torch.cat([p[4] for p in parts]),
                torch.cat([p[5] for p in parts]))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    print(build_index(sys.argv[1]))
