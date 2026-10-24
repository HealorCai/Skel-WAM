#!/usr/bin/env python3
"""Interpolate a Wan Video DiT into the narrower Skel-WAM Keypoint Expert.

The script reads sharded safetensors lazily. It never instantiates the 5B Video
Expert and therefore only retains the resulting Keypoint Expert state in RAM.
New keypoint input/output/role modules are intentionally absent from the output
and are initialized by ``SkelWamBackboneModel``.

python utils/preprocess_keypoint_expert.py \
  --wan_checkpoint_dir checkpoints/Wan-AI/Wan2.2-TI2V-5B \
  --output checkpoints/skel_wam/keypoint_expert_1024.pt \
  --dtype bfloat16

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--ffn_dim", type=int, default=4096)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    return parser.parse_args()


def _resize_vector(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[0] == size:
        return tensor.clone()
    result = F.interpolate(
        tensor.float().reshape(1, 1, -1),
        size=size,
        mode="linear",
        align_corners=True,
    ).reshape(size)
    return result.to(dtype=tensor.dtype)


def _resize_matrix(
    tensor: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    if tuple(tensor.shape) == (rows, columns):
        return tensor.clone()
    result = F.interpolate(
        tensor.float().reshape(1, 1, *tensor.shape),
        size=(rows, columns),
        mode="bilinear",
        align_corners=True,
    ).reshape(rows, columns)
    return result.to(dtype=tensor.dtype)


def _resize_modulation_projection(
    tensor: torch.Tensor, hidden_dim: int
) -> torch.Tensor:
    source_hidden = tensor.shape[1]
    if tensor.shape[0] != source_hidden * 6:
        raise ValueError(
            f"Expected time projection [6D,D], got {tuple(tensor.shape)}"
        )
    chunks = tensor.reshape(6, source_hidden, source_hidden)
    resized = [
        _resize_matrix(chunk, hidden_dim, hidden_dim) for chunk in chunks
    ]
    return torch.stack(resized).reshape(6 * hidden_dim, hidden_dim)


def _target_shape(
    key: str,
    source_shape: tuple[int, ...],
    *,
    hidden_dim: int,
    ffn_dim: int,
    attention_dim: int,
) -> tuple[int, ...] | None:
    if key.startswith("patch_embedding.") or key.startswith("head."):
        return None
    if key == "text_embedding.0.weight":
        return (hidden_dim, source_shape[1])
    if key == "text_embedding.0.bias":
        return (hidden_dim,)
    if key == "text_embedding.2.weight":
        return (hidden_dim, hidden_dim)
    if key == "text_embedding.2.bias":
        return (hidden_dim,)
    if key == "time_embedding.0.weight":
        return (hidden_dim, source_shape[1])
    if key == "time_embedding.0.bias":
        return (hidden_dim,)
    if key == "time_embedding.2.weight":
        return (hidden_dim, hidden_dim)
    if key == "time_embedding.2.bias":
        return (hidden_dim,)
    if key == "time_projection.1.weight":
        return (hidden_dim * 6, hidden_dim)
    if key == "time_projection.1.bias":
        return (hidden_dim * 6,)
    if not key.startswith("blocks."):
        return None
    if key.endswith(".modulation"):
        return (1, 6, hidden_dim)
    if ".norm3." in key:
        return (hidden_dim,)
    if ".self_attn." in key or ".cross_attn." in key:
        leaf = key.rsplit(".", 2)[-2:]
        projection, parameter = leaf
        if projection in {"q", "k", "v"}:
            return (
                (attention_dim, hidden_dim)
                if parameter == "weight"
                else (attention_dim,)
            )
        if projection == "o":
            return (
                (hidden_dim, attention_dim)
                if parameter == "weight"
                else (hidden_dim,)
            )
        if projection in {"norm_q", "norm_k"}:
            return (attention_dim,)
    if ".ffn.0." in key:
        return (ffn_dim, hidden_dim) if key.endswith("weight") else (ffn_dim,)
    if ".ffn.2." in key:
        return (hidden_dim, ffn_dim) if key.endswith("weight") else (hidden_dim,)
    return None


def _convert_tensor(
    key: str, tensor: torch.Tensor, target_shape: tuple[int, ...], hidden_dim: int
) -> torch.Tensor:
    if tuple(tensor.shape) == target_shape:
        return tensor.clone()
    if key == "time_projection.1.weight":
        return _resize_modulation_projection(tensor, hidden_dim)
    if key == "time_projection.1.bias":
        source_hidden = tensor.shape[0] // 6
        return torch.stack(
            [
                _resize_vector(chunk, hidden_dim)
                for chunk in tensor.reshape(6, source_hidden)
            ]
        ).reshape(-1)
    if key.endswith(".modulation"):
        return torch.stack(
            [
                _resize_vector(chunk, hidden_dim)
                for chunk in tensor.reshape(6, tensor.shape[-1])
            ],
            dim=0,
        ).reshape(1, 6, hidden_dim)
    if len(target_shape) == 1:
        return _resize_vector(tensor, target_shape[0])
    if len(target_shape) == 2:
        return _resize_matrix(tensor, target_shape[0], target_shape[1])
    raise ValueError(
        f"Unsupported interpolation for {key}: {tuple(tensor.shape)} -> {target_shape}"
    )


def _checkpoint_index(checkpoint_dir: Path) -> tuple[dict[str, str], Iterable[Path]]:
    index_candidates = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if index_candidates:
        payload = json.loads(index_candidates[0].read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_candidates[0]}")
        return weight_map, []
    shards = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {checkpoint_dir}")
    return {}, shards


def main() -> None:
    args = parse_args()
    config_path = args.wan_checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text())
    num_heads = int(config["num_heads"])
    num_layers = int(config["num_layers"])
    video_hidden_dim = int(config["dim"])
    video_head_dim = video_hidden_dim // num_heads
    if video_head_dim != args.head_dim:
        raise ValueError(
            f"Wan head_dim={video_head_dim}, requested head_dim={args.head_dim}"
        )
    attention_dim = num_heads * args.head_dim
    output_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    weight_map, unindexed_shards = _checkpoint_index(args.wan_checkpoint_dir)
    shard_names = sorted(set(weight_map.values())) if weight_map else [
        path.name for path in unindexed_shards
    ]
    output_state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = args.wan_checkpoint_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                target_shape = _target_shape(
                    key,
                    tuple(handle.get_slice(key).get_shape()),
                    hidden_dim=args.hidden_dim,
                    ffn_dim=args.ffn_dim,
                    attention_dim=attention_dim,
                )
                if target_shape is None:
                    continue
                output_state[key] = _convert_tensor(
                    key, handle.get_tensor(key), target_shape, args.hidden_dim
                ).to(dtype=output_dtype)
    expected_blocks = {
        int(key.split(".")[1])
        for key in output_state
        if key.startswith("blocks.")
    }
    if expected_blocks != set(range(num_layers)):
        raise ValueError(
            f"Converted block coverage mismatch: expected {num_layers}, got "
            f"{sorted(expected_blocks)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "format": "skel_wam_keypoint_expert_v1",
                "source": str(args.wan_checkpoint_dir),
                "hidden_dim": args.hidden_dim,
                "ffn_dim": args.ffn_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": args.head_dim,
                "text_dim": int(config.get("text_dim", 4096)),
                "freq_dim": int(config.get("freq_dim", 256)),
                "dtype": args.dtype,
                "interpolation": "align_corners_linear_bilinear",
            },
            "state_dict": output_state,
        },
        args.output,
    )
    print(f"Saved {len(output_state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
