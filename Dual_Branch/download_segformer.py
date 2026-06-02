from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


repo_dir = snapshot_download(
    repo_id="nvidia/mit-b1",
    allow_patterns=["pytorch_model.bin"],
)

hf_state = torch.load(Path(repo_dir) / "pytorch_model.bin", map_location="cpu")

if "state_dict" in hf_state:
    hf_state = hf_state["state_dict"]
elif "model" in hf_state:
    hf_state = hf_state["model"]


def find_prefix(state_dict):
    for prefix in ("segformer.encoder", "encoder", ""):
        key = f"{prefix}.patch_embeddings.0.proj.weight" if prefix else "patch_embeddings.0.proj.weight"
        if key in state_dict:
            return prefix
    raise KeyError("Cannot find SegFormer encoder keys in pytorch_model.bin")


def key(prefix, name):
    return f"{prefix}.{name}" if prefix else name


def get(state_dict, name):
    if name not in state_dict:
        raise KeyError(f"Missing key: {name}")
    return state_dict[name]


prefix = find_prefix(hf_state)
local_state = OrderedDict()

for stage in range(4):
    local_stage = stage + 1

    hf_patch = key(prefix, f"patch_embeddings.{stage}")
    local_state[f"patch_embed{local_stage}.proj.weight"] = get(hf_state, f"{hf_patch}.proj.weight")
    local_state[f"patch_embed{local_stage}.proj.bias"] = get(hf_state, f"{hf_patch}.proj.bias")
    local_state[f"patch_embed{local_stage}.norm.weight"] = get(hf_state, f"{hf_patch}.layer_norm.weight")
    local_state[f"patch_embed{local_stage}.norm.bias"] = get(hf_state, f"{hf_patch}.layer_norm.bias")

    hf_norm = key(prefix, f"layer_norm.{stage}")
    local_state[f"norm{local_stage}.weight"] = get(hf_state, f"{hf_norm}.weight")
    local_state[f"norm{local_stage}.bias"] = get(hf_state, f"{hf_norm}.bias")

    block_idx = 0
    while key(prefix, f"block.{stage}.{block_idx}.layer_norm_1.weight") in hf_state:
        hf_block = key(prefix, f"block.{stage}.{block_idx}")
        local_block = f"block{local_stage}.{block_idx}"

        local_state[f"{local_block}.norm1.weight"] = get(hf_state, f"{hf_block}.layer_norm_1.weight")
        local_state[f"{local_block}.norm1.bias"] = get(hf_state, f"{hf_block}.layer_norm_1.bias")
        local_state[f"{local_block}.norm2.weight"] = get(hf_state, f"{hf_block}.layer_norm_2.weight")
        local_state[f"{local_block}.norm2.bias"] = get(hf_state, f"{hf_block}.layer_norm_2.bias")

        hf_attn = f"{hf_block}.attention"
        local_state[f"{local_block}.attn.q.weight"] = get(hf_state, f"{hf_attn}.self.query.weight")
        local_state[f"{local_block}.attn.q.bias"] = get(hf_state, f"{hf_attn}.self.query.bias")
        local_state[f"{local_block}.attn.kv.weight"] = torch.cat(
            [
                get(hf_state, f"{hf_attn}.self.key.weight"),
                get(hf_state, f"{hf_attn}.self.value.weight"),
            ],
            dim=0,
        )
        local_state[f"{local_block}.attn.kv.bias"] = torch.cat(
            [
                get(hf_state, f"{hf_attn}.self.key.bias"),
                get(hf_state, f"{hf_attn}.self.value.bias"),
            ],
            dim=0,
        )
        local_state[f"{local_block}.attn.proj.weight"] = get(hf_state, f"{hf_attn}.output.dense.weight")
        local_state[f"{local_block}.attn.proj.bias"] = get(hf_state, f"{hf_attn}.output.dense.bias")

        sr_weight = f"{hf_attn}.self.sr.weight"
        if sr_weight in hf_state:
            local_state[f"{local_block}.attn.sr.weight"] = hf_state[sr_weight]
            local_state[f"{local_block}.attn.sr.bias"] = get(hf_state, f"{hf_attn}.self.sr.bias")
            local_state[f"{local_block}.attn.norm.weight"] = get(hf_state, f"{hf_attn}.self.layer_norm.weight")
            local_state[f"{local_block}.attn.norm.bias"] = get(hf_state, f"{hf_attn}.self.layer_norm.bias")

        hf_mlp = f"{hf_block}.mlp"
        local_state[f"{local_block}.mlp.fc1.weight"] = get(hf_state, f"{hf_mlp}.dense1.weight")
        local_state[f"{local_block}.mlp.fc1.bias"] = get(hf_state, f"{hf_mlp}.dense1.bias")
        local_state[f"{local_block}.mlp.dwconv.dwconv.weight"] = get(hf_state, f"{hf_mlp}.dwconv.dwconv.weight")
        local_state[f"{local_block}.mlp.dwconv.dwconv.bias"] = get(hf_state, f"{hf_mlp}.dwconv.dwconv.bias")
        local_state[f"{local_block}.mlp.fc2.weight"] = get(hf_state, f"{hf_mlp}.dense2.weight")
        local_state[f"{local_block}.mlp.fc2.bias"] = get(hf_state, f"{hf_mlp}.dense2.bias")

        block_idx += 1

out_path = Path(__file__).resolve().parent / "model" / "pretrained" / "mit_b1.pth"
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(local_state, out_path)

print("Downloaded from:", repo_dir)
print("Saved to:", out_path)
