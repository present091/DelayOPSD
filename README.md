# DelayOPSD: On-Policy Self-Distillation from a Synchronous Teacher for Real-time Asynchronous Policies

Hyeonjae Kim<sup>1*</sup>, Jeongsol Kim<sup>1*</sup>, Sunwoo Park<sup>1</sup>, Chanyu Moon<sup>1</sup>, Yoojin Oh<sup>1</sup>, Youngjun Jun<sup>1</sup>, Kyumin Choi<sup>2</sup>, Jong Chul Ye<sup>1†</sup>

<sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Sungkyunkwan University &nbsp;&nbsp; <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author

**Paper:** arXiv link coming soon

<p align="center"><img src="assets/teaser.png" width="100%"></p>

Vision-language-action policies that predict action chunks are run asynchronously in practice: while the robot executes
the current chunk, the next one is inferred from an observation that is already `k` steps old, and the new chunk takes
over `k` steps later. The policy was never trained for that gap. DelayOPSD trains it on its own asynchronous rollouts:

1. **Rollout buffers**: the start policy executed asynchronously at every delay `k = 0..10` (naive switch, `k = 0` is
   synchronous execution), collected once before training.
2. **Teacher**: the same policy run synchronously, i.e. on the *later* observation `o_{t+k}` with the last `k` action
   tokens removed, so it sees the state the chunk will actually start in. SmolVLA's action tokens attend causally, so
   the frozen start policy already is this teacher. π0.5's attend bidirectionally, so its teacher is the start policy
   fine-tuned with masked flow matching on the demonstrations (stage 1), then frozen.
3. **Student**: at each replan state `o_t` of the buffers, the student samples its chunk given the running chunk's next
   `k` actions as a clean prefix (training-time action conditioning), and every state of its own denoising trajectory
   is distilled towards the teacher's velocity at `o_{t+k}`.

At deployment the student takes the running chunk's committed actions as its prefix; the inference cost is that of the
base policy (74.0 vs 70.9 ms per chunk for π0.5 on an RTX 4090, LoRA merged).

This repository builds on [LeRobot](https://github.com/huggingface/lerobot), pinned at `2774d9b` plus one commit that adds training-time RTC to SmolVLA ([`present091/lerobot@delayopsd`](https://github.com/present091/lerobot/tree/delayopsd)); π0.5 runs on unmodified LeRobot code.

| setting | teacher | configs, scripts | details |
|---|---|---|---|
| LIBERO, π0.5 | stage 1 (masked flow matching) | `configs/pi05_libero`, `scripts/pi05_libero` | [below](#libero-π05) |
| LIBERO, SmolVLA | frozen start policy | `configs/smolvla_libero`, `scripts/smolvla_libero` | [below](#libero-smolvla) |
| Kinetix (JAX) | stage 1 (masked flow matching) | `kinetix/` | [`kinetix/README.md`](kinetix/README.md) |
| Real robot (YAM, π0.5) | stage 1 (masked flow matching) | `configs/yam`, `real_robot/` | [`real_robot/README.md`](real_robot/README.md) |

## Install

```bash
conda create -n delayopsd python=3.12 && conda activate delayopsd
pip install -e .              # LeRobot (pinned, see above) with pi0.5, SmolVLA, PEFT, LIBERO; MuJoCo 3.2.3
pip install -e ".[server]"    # only for the real-robot inference server
```

LIBERO results use **MuJoCo 3.2.3**: from 3.4.0 on, the bowl of libero_spatial task 5 no longer settles within
LIBERO's wait steps ([lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)), which changes that task
by more than 30 points.

## Data and checkpoints

Our checkpoints: coming soon (one folder per model, to be placed under `checkpoints/`).

| | source | path used by the configs |
|---|---|---|
| π0.5 LIBERO start policy | [`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned) @ `8e17415` | `checkpoints/pi05_libero_finetuned` |
| π0.5 LIBERO demonstrations (v3.0, 1,693 episodes) | [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) @ `a1aaacb` | `data/libero` |
| π0.5 LIBERO teacher (8.8 GB) | ours | `checkpoints/pi05_libero_teacher` |
| π0.5 LIBERO student, EMA (1.7 GB, LoRA + action expert on the start policy) | ours | `checkpoints/pi05_libero_student` |
| SmolVLA LIBERO start policy (also the teacher) | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) @ `a24f1f3` | (hub id) |
| SmolVLA LIBERO demonstrations | [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero) @ `8695891` | `data/libero_smolvla` |
| SmolVLA LIBERO student, EMA (217 MB, LoRA + action expert) | ours | `checkpoints/smolvla_libero_student` |
| YAM start policy, teacher (8.8 GB each), student (1.7 GB) | ours | `checkpoints/pi05_yam_8task`, `checkpoints/yam_teacher`, `checkpoints/yam_student` |
| YAM teleoperation dataset, robot rollouts | not released | |

## LIBERO evaluation protocol

Both LIBERO settings use the asynchronous protocol of the paper: replan every `R = 10` steps, the new chunk takes over
`k` steps after its observation while the running chunk keeps executing (`R:k = 10:k`); four suites, 50 episodes per
task (seed `1000 + episode`, recorded initial states), 500 per suite. Each `eval.sh` takes a method and `k`:
`ours` (the student), `async` (the start policy with a naive switch; `async 0` is synchronous execution).
Results land in `outputs/eval/` (π0.5) or `outputs/eval_smolvla/` (SmolVLA), under `<method>_10_<k>/<suite>/`
(`summary.json`, per-episode `episodes.csv` with scene and state hashes, so two methods can be compared episode by
episode). Per-chunk latency: `python -m delayopsd.eval.latency --help`.

## LIBERO, π0.5

```bash
huggingface-cli download lerobot/pi05_libero_finetuned --revision 8e174154ef5f6c60a8da12ae99c303d8963138c1 \
  --local-dir checkpoints/pi05_libero_finetuned
huggingface-cli download lerobot/libero --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 --local-dir data/libero

bash scripts/pi05_libero/collect_buffers.sh   # 11 rollout buffers of the start policy, 10:0 .. 10:10 (~18 GPU-h)
bash scripts/pi05_libero/train.sh teacher     # stage 1, ~3 h on one RTX 4090
bash scripts/pi05_libero/train.sh student     # stage 2, ~17 h on one RTX 4090

bash scripts/pi05_libero/eval.sh ours 10      # DelayOPSD student at 10:10
bash scripts/pi05_libero/eval.sh async 10     # start policy, naive asynchronous switch at 10:10
bash scripts/pi05_libero/eval.sh async 0      # start policy, synchronous execution (10:0)
```

## LIBERO, SmolVLA

Same pipeline on 256 px renders, with the frozen start policy as the teacher (no stage 1). Training-time action
conditioning for SmolVLA is the one commit of our LeRobot pin.

```bash
huggingface-cli download HuggingFaceVLA/libero --repo-type dataset \
  --revision 86958911c0f959db2bbbdb107eb3e17c5f9c798e --local-dir data/libero_smolvla

bash scripts/smolvla_libero/collect_buffers.sh   # 11 rollout buffers of lerobot/smolvla_libero, 10:0 .. 10:10
bash scripts/smolvla_libero/train.sh             # student, 10K updates

bash scripts/smolvla_libero/eval.sh ours 10      # or: async 10, async 0
```

## Configurations

`configs/<setting>/{teacher,student}.json` are the configurations of the paper's runs (LeRobot `TrainPipelineConfig`
plus the `opsd` group, documented in `delayopsd/config.py`). Any field can be overridden on the command line, e.g.
`--steps=1000 --output_dir=outputs/try`. The student's main settings, shared by both LIBERO models: 11 buffers drawn
uniformly over the delay, clean prefix of `min(k, 10)` actions, frozen teacher, trust weights `exp(-r/30)` over the
teacher's chunk index, student EMA 0.999 (evaluated), failed episodes kept until 30 steps after the arm stalls and
dropped when their scene also fails synchronously. The losses use only the buffer pairs (`lambda_fm = lambda_mask = 0`);
the demonstration dataset is loaded for its metadata.

## Kinetix

`kinetix/` holds the JAX implementation on the [Real-Time Chunking](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
Kinetix benchmark (12 levels, delays 0..4): files added to that repository, which `kinetix/setup.sh` clones at the
commit we used. See [`kinetix/README.md`](kinetix/README.md).

## Real robot (YAM arm)

`real_robot/` holds the YAM pipeline: converting teleoperation recordings to a LeRobot dataset, building the rollout
buffer from asynchronous robot sessions (relative joint actions: the prefix and the teacher's state are re-expressed
in each frame, `delayopsd/relative.py`), the configs `configs/yam/{teacher,student}.json`, the inference server the robot
station queries (`real_robot/server`, modes: synchronous, naive asynchronous, ours), and tests. See
[`real_robot/README.md`](real_robot/README.md).

## Repository

```
delayopsd/
  flow.py                    prefix KV cache, masked velocity, Euler sampling with a clean prefix (π0.5);
                             routes SmolVLA to flow_smolvla.py, same interface
  flow_smolvla.py            the same for SmolVLA (causal action tokens)
  objective.py               masked flow matching and the DelayOPSD loss (trust weights)
  data/buffer.py             pairs from rollout buffers: index, failure rules, delay mixing
  train.py, config.py        trainer for both stages and both models
  collect.py                 rollout-buffer collection
  eval/                      asynchronous-execution evaluation, latency
  checkpoints.py             LoRA checkpoint -> evaluable copy (base + adapter)
configs/, scripts/   pi05_libero, smolvla_libero, yam
real_robot/            YAM data conversion, robot buffers, inference server
kinetix/               Kinetix (JAX, on the Real-Time Chunking code)
```

## Citation

BibTeX will be added with the arXiv release.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) and the π0.5 checkpoints of
[openpi](https://github.com/Physical-Intelligence/openpi) ported to it. The asynchronous protocol and the Kinetix
benchmark are those of [Real-Time Chunking](https://github.com/Physical-Intelligence/real-time-chunking-kinetix).
