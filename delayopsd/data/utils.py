"""Small helpers shared by the trainer, the pair sources and the evaluation."""

from pathlib import Path

import torch

from lerobot.processor.rename_processor import rename_batch_keys


def collate_observations(observations):
    """Concatenate single-example observation dicts (tensors along dim 0, lists extended)."""
    batch = {}
    for key, value in observations[0].items():
        if isinstance(value, torch.Tensor):
            batch[key] = torch.cat([o[key] for o in observations], dim=0)
        elif isinstance(value, list):
            batch[key] = [item for o in observations for item in o[key]]
        else:
            batch[key] = value
    return batch


def prepare_batch(batch, camera_keys, rename_map, preprocessor):
    """A LeRobot dataset batch -> the policy's preprocessed input (uint8 images to [0, 1] first)."""
    for key in camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float() / 255.0
    return preprocessor(rename_batch_keys(batch, rename_map))


def validate_local_checkpoint(pretrained_path):
    """Reject a missing local checkpoint before dataset loading or model allocation."""
    if pretrained_path is None:
        return
    source = str(pretrained_path)
    path = Path(source).expanduser()
    if not (path.is_absolute() or source.startswith(".") or path.exists()):
        return  # Hub repository IDs are resolved by the native loader.
    if not path.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {path}")
    required = ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is missing {missing}")
