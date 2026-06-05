import json
from collections import OrderedDict
from pathlib import Path
from safetensors.torch import load_file

import torch


def load_checkpoint(checkpoint_path, map_location="cpu", config_path=None):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return load_safetensors_checkpoint(checkpoint_path, config_path)
    return torch.load(checkpoint_path, map_location=map_location)


def load_safetensors_checkpoint(checkpoint_path, config_path=None):
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path) if config_path is not None else checkpoint_path.with_suffix(".config.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing safetensors config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    groups = config.get("checkpoint_groups")
    if not groups:
        raise KeyError(f"`checkpoint_groups` not found in config: {config_path}")

    tensors = load_file(str(checkpoint_path), device="cpu")
    checkpoint = {}

    for group_name, group_info in groups.items():
        prefix = group_info.get("prefix", "")
        strip_prefix = group_info.get("strip_prefix", True)
        state_dict = OrderedDict()

        for tensor_name, tensor in tensors.items():
            if not tensor_name.startswith(prefix):
                continue
            key = tensor_name[len(prefix):] if strip_prefix else tensor_name
            state_dict[key] = tensor

        expected_count = group_info.get("tensor_count")
        if expected_count is not None and len(state_dict) != expected_count:
            raise ValueError(
                f"{group_name} tensor count mismatch: "
                f"expected {expected_count}, got {len(state_dict)}"
            )

        checkpoint[group_name] = state_dict

    return checkpoint


def checkpoint_result_name(checkpoint_path):
    return Path(checkpoint_path).stem
