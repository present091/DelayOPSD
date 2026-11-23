"""A PEFT (LoRA + modules_to_save) checkpoint -> an evaluable copy: base model + LoRA-only adapter.

The student trains LoRA on the VLM and the full action expert (PEFT modules_to_save). Merging LoRA into the bf16 base
weights is lossy (the rounding erases 30-80% of the LoRA delta), so the copy is written exactly instead:

* the fully trained modules are written into a copy of the base model.safetensors, in the base tensor's dtype -- the
  same cast PEFT applies when it copies the adapter's fp32 values into the loaded bf16 module;
* the LoRA factors stay an adapter on top of that base, which also avoids PEFT holding a second copy of the action
  expert at evaluation time.

    python -m delayopsd.checkpoints --out outputs/eval/student_light/pretrained_model checkpoints/pi05_libero_student
    -> <out>/base (full model), <out>/pretrained_model (adapter; pass this as --policy.path)
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."
ADAPTER = "adapter_model.safetensors"


def light_copy(checkpoint, out):
    checkpoint, out = Path(checkpoint), Path(out)
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    base = Path(config["base_model_name_or_path"])
    if not base.is_dir():  # a hub repo id (e.g. lerobot/smolvla_libero): use its cached snapshot
        base = Path(snapshot_download(str(base)))
    if config.get("use_dora") or config.get("use_rslora") or config.get("rank_pattern"):
        raise ValueError("only plain LoRA with a single rank is supported")

    lora, full = {}, {}
    with safe_open(checkpoint / ADAPTER, "pt") as adapter:
        for key in adapter.keys():
            tensor = adapter.get_tensor(key).float().contiguous()
            if ".lora_A." in key or ".lora_B." in key:
                lora[key] = tensor
            else:
                full[key.removeprefix(PREFIX)] = tensor

    base_out, adapter_out = out / "base", out / "pretrained_model"
    base_out.mkdir(parents=True, exist_ok=False)
    adapter_out.mkdir(parents=True)
    with safe_open(base / "model.safetensors", "pt") as f:
        missing = set(full) - set(f.keys())
        if missing:
            raise ValueError(f"trained modules not found in base: {sorted(missing)[:3]}")
        state = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            state[key] = full[key].to(tensor.dtype) if key in full else tensor
        metadata = f.metadata()
    save_file(state, base_out / "model.safetensors", metadata=metadata)
    for item in base.iterdir():
        if item.name != "model.safetensors":
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, base_out / item.name)

    save_file(lora, adapter_out / ADAPTER, metadata={"format": "pt"})
    for item in checkpoint.iterdir():
        if item.name not in (ADAPTER, "adapter_config.json"):
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, adapter_out / item.name)
    config.update(base_model_name_or_path=str(base_out), modules_to_save=None)
    (adapter_out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    print(f"{checkpoint} -> {adapter_out} (LoRA r={config['r']}, base {base_out})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    light_copy(args.checkpoint, args.out)
