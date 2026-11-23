"""Serve one LeRobot pi0.5 YAM right-arm checkpoint over HTTP to the robot's chunk-schedule client.

    python -m real_robot.server.server --checkpoint checkpoints/yam_student --port 8221

The client (the robot station) observes every R steps, sends the observation, keeps executing the running chunk, and
switches to the reply k steps later; with ``rtc_mode=trained`` it also sends the running chunk's next k actions::

    GET  /healthz -> plain text ``OK`` (only after load and warm-up)
    GET  /act     -> checkpoint metadata
    POST /act     -> json_numpy {"state": (7,) float32, "instruction": str,
                                 "top_cam": (480, 640, 3) uint8 RGB, "right_cam": ...}
                  -> {"actions": (50, 7) float32 absolute joint pose, "dt_ms": float}
    POST /reset   -> clear the policy's per-episode state

The checkpoint's own saved processor pipelines are used unmodified: normalization, the
relative-joint conversion (``use_relative_actions``) and its inverse, the tokenizer and the
224x224 letterbox all come from training. The relative conversion caches the state of the
last preprocessed observation for the postprocessor, so requests are serialized by a lock.

The profile's ``prefix_conditioning`` selects the method per request (``rtc_mode``):

* ``off`` (no ``rtc_mode``): plain sampling. Sync (R:0) and naive async (R:k).
* ``trained``: the OPSD student (clean prefix). The k committed actions
  (``prev_chunk_left_over``, absolute, the running chunk's actions for [t, t+k)) are put in this request's
  model space -- relative checkpoints: minus the request's state, then the checkpoint's normalizer, as
  delayopsd.relative.model_space does for the training prefixes -- and fed clean at flow time 0
  while the rest is denoised, by the trainer's own sampler (``delayopsd.flow``). The reply
  starts at the observation tick; its first k actions are the prefix itself. Only with exactly
  TRAINED_PREFIX actions (the length the students were trained with; the station's prefix_max default of 10
  would otherwise cut it silently).
Nothing here can move the arm; the speed cap, ramp and handoff guards live in the runtime
profile and RobotNode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import json_numpy
import numpy as np
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("opsd.real_robot.server")

CAMERAS = {"top_cam": "observation.images.camera_top", "right_cam": "observation.images.camera_right"}
IMAGE_HW = (480, 640)
STATE_DIM = ACTION_DIM = 7
REQUIRED_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
# Sent by ChunkScheduleAgent alongside the observation.
SCHEDULE_FIELDS = {"episode_generation", "episode_step", "replan_interval", "inference_delay", "switch_step"}
# Extra fields each prefix_conditioning mode carries (rtc_mode itself is added by the agent).
MODE_FIELDS = {
    "off": set(),
    "trained": {"prev_chunk_left_over", "execution_horizon"},
}
# The clean-prefix length the YAM student is trained with (configs/yam/student.json: prefix_len 15 at 15:15).
TRAINED_PREFIX = int(os.environ.get("HYEONJAE_PI05_TRAINED_PREFIX", "15"))


class RequestError(ValueError):
    """A request that does not match the wire contract (HTTP 400)."""


def check_checkpoint(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    # A LoRA student (PEFT adapter + modules_to_save) is served on top of its base checkpoint, exactly as
    # lerobot.policies.factory.make_policy loads it for the LIBERO evals (never merged: merging LoRA into the bf16
    # base erases most of its delta, see delayopsd/checkpoints.py).
    required = REQUIRED_FILES
    if (path / "adapter_config.json").is_file():
        required = tuple("adapter_model.safetensors" if n == "model.safetensors" else n for n in REQUIRED_FILES)
        base = Path(json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"])
        if not (base / "model.safetensors").is_file():
            raise SystemExit(f"{path}: the adapter's base checkpoint {base} is not here")
    missing = [name for name in required if not (path / name).is_file()]
    if not path.is_dir() or missing:
        raise SystemExit(f"not a LeRobot checkpoint directory: {path} (missing {missing})")
    config = json.loads((path / "config.json").read_text())
    if config.get("type") != "pi05":
        raise SystemExit(f"{path} is a {config.get('type')!r} policy, not pi05")
    # empty_camera_* are the padding slots pi0.5 fills itself (empty_cameras=1)
    names = [k for k in config.get("input_features", {})
             if k.startswith("observation.images.") and not k.startswith("observation.images.empty_camera_")]
    if sorted(names) != sorted(CAMERAS.values()):
        raise SystemExit(f"{path} expects cameras {names}, the wire carries {sorted(CAMERAS.values())}")
    if config["output_features"]["action"]["shape"] != [ACTION_DIM]:
        raise SystemExit(f"{path} has action shape {config['output_features']['action']['shape']}")
    return path


def _actions(payload: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(payload[key], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or not len(value) or not np.isfinite(value).all():
        raise RequestError(f"{key} must be (k, {ACTION_DIM}) finite absolute joint targets, got {value.shape}")
    return value


def validate_request(payload: Any, horizon: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError(f"request must be a JSON object, got {type(payload).__name__}")
    mode = payload.get("rtc_mode", "off")
    if mode not in MODE_FIELDS:
        raise RequestError(f"rtc_mode {mode!r} is not served (served: {sorted(MODE_FIELDS)})")
    other_modes = set().union(*MODE_FIELDS.values()) - MODE_FIELDS[mode]
    stray = other_modes & payload.keys()
    if stray:
        raise RequestError(f"prefix fields {sorted(stray)} do not belong to rtc_mode {mode!r}; "
                           "set prefix_conditioning in the profile")
    unknown = set(payload) - {"state", "instruction", "rtc_mode", *CAMERAS} - SCHEDULE_FIELDS - MODE_FIELDS[mode]
    missing = {"state", "instruction", *CAMERAS} - payload.keys()
    if missing or unknown:
        raise RequestError(f"wire mismatch: missing {sorted(missing)}, unknown {sorted(unknown)}")
    state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise RequestError(f"state must be {STATE_DIM} finite values, got shape {state.shape}")
    instruction = payload["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise RequestError("instruction must be a non-empty string")
    images = {}
    for wire in CAMERAS:
        image = np.asarray(payload[wire])
        if image.shape != (*IMAGE_HW, 3) or image.dtype != np.uint8:
            raise RequestError(f"{wire} must be {(*IMAGE_HW, 3)} uint8 RGB, got {image.shape} {image.dtype}")
        images[wire] = image
    request = {"state": state, "instruction": instruction.strip(), "images": images, "mode": mode,
               "generation": payload.get("episode_generation"), "step": payload.get("episode_step"),
               "delay": 0, "prefix": None}
    if mode == "off":
        return request

    delay = int(payload.get("inference_delay", -1))
    if request["step"] is None:
        raise RequestError(f"rtc_mode {mode!r} needs episode_step")
    request["step"] = int(request["step"])
    key = "prev_chunk_left_over"
    if key not in payload:
        raise RequestError(f"rtc_mode {mode!r} needs {key}")
    prefix = _actions(payload, key)
    if delay != len(prefix) or not 0 < delay < horizon:
        raise RequestError(f"inference_delay {delay} must equal the {len(prefix)} rows of {key} and be in [1, {horizon})")
    request.update(delay=delay, prefix=prefix)
    if mode == "trained" and delay != TRAINED_PREFIX:
        raise RequestError(f"rtc_mode 'trained' needs exactly {TRAINED_PREFIX} committed actions (the trained prefix "
                           f"length), got {delay}: set prefix_max: {TRAINED_PREFIX} in the profile")
    return request


class Runner:
    """One loaded policy with its saved pipelines; inference is serialized by a lock."""

    def __init__(self, checkpoint: Path, *, checkpoint_name: str, device: str, num_steps: int | None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(f"{device} requested but torch sees no CUDA device")
        started = time.perf_counter()
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.device = device
        if num_steps is not None:
            config.num_inference_steps = int(num_steps)
        # strict: a tensor dropped or left at random init is a different model, not a warning
        if (checkpoint / "adapter_config.json").is_file():
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(str(checkpoint))
            base = get_policy_class(config.type).from_pretrained(peft_config.base_model_name_or_path, config=config,
                                                                 strict=True)
            peft_model = PeftModel.from_pretrained(base, str(checkpoint), config=peft_config)
            self.policy = peft_model.base_model.model  # the policy, its LoRA layers and saved modules in place
            self.adapter = str(checkpoint)
        else:
            self.policy = get_policy_class(config.type).from_pretrained(str(checkpoint), config=config, strict=True)
            self.adapter = None
        self.policy.to(device).eval()
        self.policy.requires_grad_(False)
        if "embed_prefix" in vars(self.policy.model):
            raise SystemExit("policy.model.embed_prefix is already patched")
        self.pre, self.post = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.relative_step = next((s for s in self.pre.steps
                                   if isinstance(s, RelativeActionsProcessorStep) and s.enabled), None)
        self.normalizer_step = next(s for s in self.pre.steps if isinstance(s, NormalizerProcessorStep))
        self.config, self.checkpoint, self.checkpoint_name, self.device = config, checkpoint, checkpoint_name, device
        self.horizon = int(config.chunk_size)
        self.load_s = time.perf_counter() - started
        self.warmup_ms: dict[str, float] = {}
        self.requests = {mode: 0 for mode in MODE_FIELDS}
        self._generation: Any = None
        self._last: tuple[np.ndarray, int | None] | None = None  # (returned chunk, its episode_step)
        self._lock = threading.Lock()
        self._torch = torch

    @property
    def modes(self) -> list[str]:
        return list(MODE_FIELDS)

    def _reset_episode(self) -> None:
        self._last = None

    def _model_space(self, actions_abs: np.ndarray) -> Any:
        """Absolute joint targets -> the model's normalized action space at the cached state."""
        torch = self._torch
        from lerobot.policies.rtc import reanchor_relative_rtc_prefix
        from lerobot.processor import TransitionKey, create_transition

        actions = torch.from_numpy(np.ascontiguousarray(actions_abs))[None]
        if self.relative_step is not None:
            return reanchor_relative_rtc_prefix(actions, self.relative_step.get_cached_state(),
                                                self.relative_step, self.normalizer_step, self.device)
        return self.normalizer_step(create_transition(action=actions))[TransitionKey.ACTION].to(self.device)

    def predict(self, payload: Any) -> np.ndarray:
        request = validate_request(payload, self.horizon)
        mode, delay, prefix = request["mode"], request["delay"], request["prefix"]
        torch = self._torch
        with self._lock:
            if request["generation"] is not None and request["generation"] != self._generation:
                self._reset_episode()
                self._generation = request["generation"]
            state = request["state"]
            batch = {"observation.state": torch.from_numpy(np.asarray(state, np.float32))[None],
                     "task": [request["instruction"]]}
            for wire, key in CAMERAS.items():
                image = torch.from_numpy(np.ascontiguousarray(request["images"][wire]))
                batch[key] = image.permute(2, 0, 1).float().div(255.0)[None]
            self.requests[mode] += 1
            with torch.inference_mode():
                observation = self.pre(batch)
                if mode == "trained":
                    # Relative checkpoints too: _model_space puts the committed absolute actions in this
                    # request's model space (relative: minus the state just cached by self.pre, then normalized) --
                    # the student-frame prefix of delayopsd.relative.model_space -- and self.post adds
                    # that state back to the whole chunk, prefix included.
                    from delayopsd import flow as opsd_flow  # the trainer's own sampler

                    shape = (1, self.horizon, int(self.config.max_action_dim))
                    model_prefix = torch.zeros(shape, device=self.device)
                    model_prefix[:, :delay, :ACTION_DIM] = self._model_space(prefix)[..., :ACTION_DIM].float()
                    noise = self.policy.model.sample_noise(shape, self.device)
                    x, _, _ = opsd_flow.sample_with_trajectory(
                        self.policy.model, opsd_flow.observation_inputs(self.policy, observation), noise,
                        int(self.config.num_inference_steps), prefix=model_prefix,
                        prefix_len=torch.tensor([delay], device=self.device), prefix_mode="clean")
                    chunk = x[:, :, :ACTION_DIM]
                else:
                    chunk = self.policy.predict_action_chunk(observation)
                # The absolute-actions step adds back the state cached by the preprocessor call above.
                actions = self.post(chunk)
            actions = actions.float().cpu().numpy()[0, : self.horizon, :ACTION_DIM]
            if actions.shape != (self.horizon, ACTION_DIM) or not np.isfinite(actions).all():
                raise RuntimeError(f"model returned an invalid chunk: shape {actions.shape}, finite {np.isfinite(actions).all()}")
            actions = actions.astype(np.float32)
            self._last = (actions.copy(), request["step"])
        return actions

    def warmup(self) -> None:
        """One request per served mode, as one short episode, so no first request pays for setup."""
        image = {wire: np.full((*IMAGE_HW, 3), 128, np.uint8) for wire in CAMERAS}
        instruction = "warm up"
        base = {"state": np.zeros(STATE_DIM, np.float32), "instruction": instruction, **image,
                "episode_generation": -1, "episode_step": 0}
        started = time.perf_counter()
        first = self.predict(base)
        self.warmup_ms["off"] = (time.perf_counter() - started) * 1000.0
        k = TRAINED_PREFIX
        requests = {}
        if self.horizon > 2 * TRAINED_PREFIX:
            requests["trained"] = {"rtc_mode": "trained", "prev_chunk_left_over": first[TRAINED_PREFIX:2 * TRAINED_PREFIX],
                                   "inference_delay": TRAINED_PREFIX, "execution_horizon": TRAINED_PREFIX}
        for mode, extra in requests.items():
            started = time.perf_counter()
            self.predict({**base, **extra, "episode_step": k})
            self.warmup_ms[mode] = (time.perf_counter() - started) * 1000.0
            self._last = (first, 0)
        self.reset()
        self.requests = {mode: 0 for mode in MODE_FIELDS}

    def reset(self) -> None:
        with self._lock:
            self.policy.reset()
            self._reset_episode()
            self._generation = None

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "lerobot_pi05",
            "checkpoint": str(self.checkpoint),
            "checkpoint_name": self.checkpoint_name,
            "use_relative_actions": bool(getattr(self.config, "use_relative_actions", False)),
            "action_horizon": self.horizon,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "action_representation": "absolute joint pose (6 joints + gripper, gripper 1 = open)",
            "cameras": CAMERAS,
            "image_hw": list(IMAGE_HW),
            "control_fps": 30,
            "num_inference_steps": int(self.config.num_inference_steps),
            "dtype": str(self.config.dtype),
            "prefix_conditioning": self.modes,
            "device": self.device,
            "load_s": self.load_s,
            "warmup_ms": self.warmup_ms,
            "requests_served": self.requests,
        }


def build_app(runner: Runner) -> FastAPI:
    app = FastAPI(title="OPSD pi0.5 YAM right arm", version="1.0")

    def error(status: int, message: str) -> Response:
        return Response(content=json_numpy.dumps({"error": message}), status_code=status, media_type="application/json")

    @app.get("/healthz")
    async def healthz() -> Response:
        return Response(content="OK", media_type="text/plain")

    @app.get("/act")
    async def metadata() -> JSONResponse:
        return JSONResponse(runner.metadata())

    @app.post("/reset")
    async def reset() -> JSONResponse:
        runner.reset()
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        try:
            payload = json_numpy.loads((await request.body()).decode("utf-8"))
            started = time.perf_counter()
            # in a worker thread, so a forward pass never blocks /healthz or /act metadata
            actions = await run_in_threadpool(runner.predict, payload)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        except (RequestError, UnicodeDecodeError, ValueError, TypeError) as exc:
            return error(400, str(exc))
        except Exception as exc:  # noqa: BLE001 -- a serving failure must not kill the process
            log.exception("inference failed")
            return error(500, f"inference failed: {exc}")
        return Response(content=json_numpy.dumps({"actions": actions, "dt_ms": elapsed_ms}), media_type="application/json")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-name", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=None, help="denoising steps; default is the checkpoint's")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    checkpoint = check_checkpoint(args.checkpoint)
    runner = Runner(checkpoint, checkpoint_name=args.checkpoint_name or checkpoint.name,
                    device=args.device, num_steps=args.num_steps)
    log.info("Loaded %s in %.1f s (relative=%s, horizon %d, %d denoising steps, modes %s)", checkpoint, runner.load_s,
             runner.metadata()["use_relative_actions"], runner.horizon, runner.config.num_inference_steps, runner.modes)
    if not args.no_warmup:
        runner.warmup()
        log.info("Warm-up inference %s ms", {m: round(v) for m, v in runner.warmup_ms.items()})

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(build_app(runner), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
