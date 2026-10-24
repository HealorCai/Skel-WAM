from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from skel_wam.models.wan.modules.model import (
    WanLayerNorm,
    WanModel,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)

KEYPOINT_VALIDITY_MEAN = 0.5
KEYPOINT_VALIDITY_STD = 1.0

ACTION_STATE_PREFIXES = (
    "action_patch_embedding.",
    "action_latent_memory.",
    "action_keypoint_memory.",
    "action_fm_head.",
)


def clone_video_patch_embedding(module: nn.Conv3d) -> nn.Conv3d:
    """Clone Wan's patch projection without constructing the full Video Expert."""
    cloned = nn.Conv3d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    cloned.load_state_dict(module.state_dict(), strict=True)
    return cloned


def load_component_state_dict(
    model: nn.Module,
    backbone_state: dict[str, torch.Tensor],
    *,
    action_state: dict[str, torch.Tensor] | None = None,
    allow_legacy_shared_action_patch: bool = True,
) -> None:
    """Load a expert backbone and optional portable Action checkpoint strictly."""
    expected = set(model.state_dict())
    expected_action = {
        key for key in expected if key.startswith(ACTION_STATE_PREFIXES)
    }
    expected_backbone = expected - expected_action
    provided_backbone = {
        str(key): value
        for key, value in backbone_state.items()
        if not str(key).startswith(ACTION_STATE_PREFIXES)
    }
    missing_backbone = sorted(expected_backbone - set(provided_backbone))
    unexpected_backbone = sorted(set(provided_backbone) - expected_backbone)
    if missing_backbone or unexpected_backbone:
        raise RuntimeError(
            "Incompatible expert backbone state: "
            f"missing={missing_backbone[:8]}, "
            f"unexpected={unexpected_backbone[:8]}"
        )
    model.load_state_dict(provided_backbone, strict=False)

    if not expected_action:
        return
    source = action_state if action_state is not None else backbone_state
    provided_action = {
        str(key): value
        for key, value in source.items()
        if str(key).startswith(ACTION_STATE_PREFIXES)
    }
    missing_action = expected_action - set(provided_action)
    legacy_patch_keys = {
        key
        for key in expected_action
        if key.startswith("action_patch_embedding.")
    }
    if (
        allow_legacy_shared_action_patch
        and missing_action
        and missing_action == legacy_patch_keys
    ):
        action_patch = getattr(model, "action_patch_embedding", None)
        video_expert = getattr(model, "video_expert", None)
        if action_patch is None or video_expert is None:
            raise RuntimeError("Cannot restore legacy shared Action patch embedding")
        action_patch.load_state_dict(
            video_expert.patch_embedding.state_dict(),
            strict=True,
        )
        missing_action = set()
    unexpected_action = set(provided_action) - expected_action
    if missing_action or unexpected_action:
        raise RuntimeError(
            "Incompatible expert Action state: "
            f"missing={sorted(missing_action)[:8]}, "
            f"unexpected={sorted(unexpected_action)[:8]}"
        )
    model.load_state_dict(provided_action, strict=False)


def infer_action_patch_embedding_mode(
    backbone_state: dict[str, torch.Tensor],
    *,
    action_state: dict[str, torch.Tensor] | None = None,
) -> Literal["shared", "portable_frozen"]:
    """Infer Action patch ownership from a checkpoint state dict."""
    source = action_state if action_state is not None else backbone_state
    has_portable_patch = any(
        str(key).startswith("action_patch_embedding.") for key in source
    )
    return "portable_frozen" if has_portable_patch else "shared"


def normalize_keypoint_validity(
    validity: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Map invalid/valid targets from 0/1 to the joint-FM space."""
    if dtype is None:
        dtype = validity.dtype if torch.is_floating_point(validity) else torch.float32
    return (
        validity.to(dtype=dtype) - KEYPOINT_VALIDITY_MEAN
    ) / KEYPOINT_VALIDITY_STD


def decode_keypoint_validity(
    validity_state: torch.Tensor,
) -> torch.Tensor:
    """Decode a clean normalized validity estimate to a boolean mask."""
    threshold = (0.5 - KEYPOINT_VALIDITY_MEAN) / KEYPOINT_VALIDITY_STD
    return validity_state >= threshold


def _as_batch_timestep(timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.reshape(-1)
    if timestep.numel() == 1 and batch_size > 1:
        timestep = timestep.expand(batch_size)
    if timestep.numel() != batch_size:
        raise ValueError(
            f"Expected one timestep per sample ({batch_size}), got {tuple(timestep.shape)}"
        )
    return timestep


def _pad_text(text_emb: Sequence[torch.Tensor], text_len: int) -> torch.Tensor:
    if not text_emb:
        raise ValueError("text_emb cannot be empty")
    padded = []
    for item in text_emb:
        if item.ndim != 2:
            raise ValueError(f"Expected text item [L,D], got {tuple(item.shape)}")
        if item.shape[0] > text_len:
            item = item[:text_len]
        padded.append(
            torch.cat([item, item.new_zeros(text_len - item.shape[0], item.shape[1])])
        )
    return torch.stack(padded)


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    # Tensors created inside torch.inference_mode() deliberately have no
    # version counter.  Runtime inference caches also key on data_ptr, shape,
    # dtype and device, so a sentinel is sufficient for those immutable
    # inference tensors while retaining mutation detection for regular ones.
    try:
        version: int | None = int(tensor._version)
    except RuntimeError as exc:
        if "Inference tensors do not track version counter" not in str(exc):
            raise
        version = None
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        int(tensor.data_ptr()),
        version,
    )


def _text_signature(text_emb: Sequence[torch.Tensor]) -> tuple[tuple[Any, ...], ...]:
    return tuple(_tensor_signature(item) for item in text_emb)


def _sincos_position_embedding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).reshape(length, 1)
    half_dim = dim // 2
    div = torch.exp(
        torch.arange(half_dim, dtype=torch.float32)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    ).reshape(1, half_dim)
    embedding = torch.cat(
        [torch.sin(position * div), torch.cos(position * div)], dim=-1
    )
    if embedding.shape[-1] < dim:
        embedding = torch.cat(
            [
                embedding,
                embedding.new_zeros(length, dim - embedding.shape[-1]),
            ],
            dim=-1,
        )
    return embedding[:, :dim]


def _sequence_times_on_action_axis(
    length: int, action_frames: int, device: torch.device
) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"Sequence length must be positive, got {length}")
    if length == 1:
        return torch.zeros(1, device=device, dtype=torch.float32)
    return (
        torch.arange(length, device=device, dtype=torch.float32)
        * float(action_frames - 1)
        / float(length - 1)
    )


def _video_position_ids(
    grid_size: torch.Tensor, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    f, h, w = [int(value) for value in grid_size.tolist()]
    ff = torch.arange(f, device=device).view(f, 1, 1).expand(f, h, w)
    hh = torch.arange(h, device=device).view(1, h, 1).expand(f, h, w)
    ww = torch.arange(w, device=device).view(1, 1, w).expand(f, h, w)
    position_ids = torch.stack([ff, hh, ww], dim=-1).reshape(1, f * h * w, 3)
    frame_ids = ff.reshape(1, f * h * w)
    return (
        position_ids.expand(batch_size, -1, -1).contiguous(),
        frame_ids.expand(batch_size, -1).contiguous(),
    )


@dataclass(frozen=True)
class _CausalTimeMask:
    query_time: torch.Tensor
    key_time: torch.Tensor


def _causal_by_time(
    query_time: torch.Tensor, key_time: torch.Tensor
) -> _CausalTimeMask:
    return _CausalTimeMask(query_time=query_time, key_time=key_time)


def _attention_mask_block(
    mask: torch.Tensor | _CausalTimeMask | None,
    *,
    query_slice: slice,
    key_slice: slice,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if isinstance(mask, _CausalTimeMask):
        return (
            mask.query_time[:, query_slice, None]
            >= mask.key_time[:, None, key_slice]
        )
    edge_mask = mask.to(device=device, dtype=torch.bool)
    if edge_mask.ndim == 2:
        edge_mask = edge_mask.unsqueeze(0)
    if edge_mask.ndim != 3:
        raise ValueError(
            f"Expected edge mask [B,Q,K] or [Q,K], got {tuple(mask.shape)}"
        )
    if edge_mask.shape[0] == 1 and batch_size > 1:
        edge_mask = edge_mask.expand(batch_size, -1, -1)
    return edge_mask[:, query_slice, key_slice]


def _apply_rope_3d(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan-compatible 3D RoPE to unrotated [B,S,H,D] tensors."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    complex_dim = x.shape[-1] // 2
    split = [
        complex_dim - 2 * (complex_dim // 3),
        complex_dim // 3,
        complex_dim // 3,
    ]
    freq_parts = freqs.split(split, dim=1)
    pos = position_ids.to(device=x.device, dtype=torch.long)
    phases = torch.cat(
        [
            freq_parts[axis].index_select(0, pos[..., axis].reshape(-1)).reshape(
                *pos.shape[:2], split[axis]
            )
            for axis in range(3)
        ],
        dim=-1,
    ).unsqueeze(2)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], complex_dim, 2)
    )
    return torch.view_as_real(x_complex * phases).flatten(-2).float()


def _apply_rope_temporal(
    x: torch.Tensor,
    temporal_ids: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply only Wan's temporal RoPE partition and keep spatial phase neutral."""
    complex_dim = x.shape[-1] // 2
    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    ids = temporal_ids.to(device=x.device, dtype=torch.long)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= freqs.shape[0]):
        raise ValueError(
            f"Temporal position ids must be in [0,{freqs.shape[0] - 1}], "
            f"got [{int(ids.min())},{int(ids.max())}]"
        )
    temporal = freqs[:, :temporal_dim].index_select(0, ids.reshape(-1)).reshape(
        *ids.shape, temporal_dim
    )
    neutral = torch.ones(
        *ids.shape,
        complex_dim - temporal_dim,
        device=x.device,
        dtype=temporal.dtype,
    )
    phases = torch.cat([temporal, neutral], dim=-1).unsqueeze(2)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], complex_dim, 2)
    )
    return torch.view_as_real(x_complex * phases).flatten(-2).float()


def _apply_rope_1d(
    x: torch.Tensor,
    temporal_ids: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply a full-head 1D RoPE for keypoint-to-keypoint attention."""
    complex_dim = x.shape[-1] // 2
    if freqs.shape[1] != complex_dim:
        raise ValueError(
            f"Expected full 1D RoPE complex dim {complex_dim}, got {freqs.shape[1]}"
        )
    ids = temporal_ids.to(device=x.device, dtype=torch.long)
    phases = freqs.index_select(0, ids.reshape(-1)).reshape(
        *ids.shape, complex_dim
    ).unsqueeze(2)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], complex_dim, 2)
    )
    return torch.view_as_real(x_complex * phases).flatten(-2).float()


def _segment_attention(
    query_views: Sequence[torch.Tensor],
    key_views: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor | _CausalTimeMask | None],
    *,
    query_chunk_size: int = 256,
    key_chunk_size: int = 2048,
    direct_score_limit: int = 16_000_000,
) -> torch.Tensor:
    """Attend over several key segments with one normalization over all keys."""
    if not (len(query_views) == len(key_views) == len(values) == len(masks)):
        raise ValueError("Segment attention inputs must have equal lengths")
    if not query_views:
        raise ValueError("At least one attention edge is required")
    batch_size, query_len, num_heads, head_dim = query_views[0].shape
    total_scores = sum(
        batch_size * num_heads * query_len * key.shape[1] for key in key_views
    )
    if total_scores > direct_score_limit:
        output_chunks = []
        for query_start in range(0, query_len, query_chunk_size):
            query_stop = min(query_start + query_chunk_size, query_len)
            q_slice = slice(query_start, query_stop)
            running_max = torch.full(
                (batch_size, num_heads, query_stop - query_start, 1),
                float("-inf"),
                device=query_views[0].device,
                dtype=torch.float32,
            )
            running_sum = torch.zeros_like(running_max)
            running_value = torch.zeros(
                (
                    batch_size,
                    num_heads,
                    query_stop - query_start,
                    head_dim,
                ),
                device=query_views[0].device,
                dtype=torch.float32,
            )
            for query, key, value, mask in zip(
                query_views, key_views, values, masks
            ):
                query_block = query[:, q_slice].float().transpose(1, 2)
                for key_start in range(0, key.shape[1], key_chunk_size):
                    key_stop = min(key_start + key_chunk_size, key.shape[1])
                    k_slice = slice(key_start, key_stop)
                    logits = torch.matmul(
                        query_block,
                        key[:, k_slice].float().transpose(1, 2).transpose(-2, -1),
                    ) * (head_dim**-0.5)
                    block_mask = _attention_mask_block(
                        mask,
                        query_slice=q_slice,
                        key_slice=k_slice,
                        batch_size=batch_size,
                        device=query.device,
                    )
                    if block_mask is not None:
                        logits = logits.masked_fill(
                            ~block_mask[:, None], float("-inf")
                        )
                    block_max = logits.amax(dim=-1, keepdim=True)
                    new_max = torch.maximum(running_max, block_max)
                    previous_scale = torch.where(
                        torch.isfinite(running_max),
                        torch.exp(running_max - new_max),
                        torch.zeros_like(new_max),
                    )
                    probabilities = torch.where(
                        torch.isfinite(logits),
                        torch.exp(logits - new_max),
                        torch.zeros_like(logits),
                    )
                    running_sum = (
                        running_sum * previous_scale
                        + probabilities.sum(dim=-1, keepdim=True)
                    )
                    running_value = (
                        running_value * previous_scale
                        + torch.matmul(
                            probabilities,
                            value[:, k_slice].float().transpose(1, 2),
                        )
                    )
                    running_max = new_max
            if (running_sum == 0).any():
                bad_inputs = []
                for edge_index, (query, key, value) in enumerate(
                    zip(query_views, key_views, values)
                ):
                    for tensor_name, tensor in (
                        ("query", query),
                        ("key", key),
                        ("value", value),
                    ):
                        nonfinite = int((~torch.isfinite(tensor)).sum().item())
                        if nonfinite:
                            bad_inputs.append(
                                f"edge={edge_index} {tensor_name}={nonfinite}/"
                                f"{tensor.numel()} non-finite"
                            )
                detail = "; ".join(bad_inputs)
                if detail:
                    raise FloatingPointError(
                        "Segment attention received non-finite Q/K/V: " + detail
                    )
                raise ValueError(
                    "An attention query has no visible key despite finite Q/K/V"
                )
            output_chunks.append(
                (running_value / running_sum)
                .transpose(1, 2)
                .to(dtype=values[0].dtype)
            )
        return torch.cat(output_chunks, dim=1)

    logits = []
    for query, key, mask in zip(query_views, key_views, masks):
        if query.shape[:1] + query.shape[2:] != key.shape[:1] + key.shape[2:]:
            raise ValueError(
                f"Incompatible Q/K shapes: {tuple(query.shape)} and {tuple(key.shape)}"
            )
        edge_logits = torch.matmul(
            query.float().transpose(1, 2),
            key.float().transpose(1, 2).transpose(-2, -1),
        ) * (query.shape[-1] ** -0.5)
        edge_mask = _attention_mask_block(
            mask,
            query_slice=slice(0, query.shape[1]),
            key_slice=slice(0, key.shape[1]),
            batch_size=query.shape[0],
            device=query.device,
        )
        if edge_mask is not None:
            edge_logits = edge_logits.masked_fill(
                ~edge_mask[:, None], float("-inf")
            )
        logits.append(edge_logits)
    all_logits = torch.cat(logits, dim=-1)
    if not torch.isfinite(all_logits).any(dim=-1).all():
        if torch.isnan(all_logits).any() or torch.isposinf(all_logits).any():
            raise FloatingPointError(
                "Segment attention produced non-finite logits from Q/K"
            )
        raise ValueError(
            "An attention query has no visible key despite finite Q/K/V"
        )
    probabilities = torch.softmax(all_logits, dim=-1).to(dtype=values[0].dtype)
    all_values = torch.cat([value.transpose(1, 2) for value in values], dim=2)
    return torch.matmul(probabilities, all_values).transpose(1, 2)


def _segment_attention_probabilities(
    query_views: Sequence[torch.Tensor],
    key_views: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor | _CausalTimeMask | None],
    *,
    segment_index: int,
) -> torch.Tensor:
    """Return one key segment's probabilities under the shared softmax."""
    if not (len(query_views) == len(key_views) == len(masks)):
        raise ValueError("Segment attention inputs must have equal lengths")
    if not 0 <= segment_index < len(key_views):
        raise IndexError(
            f"segment_index={segment_index} for {len(key_views)} segments"
        )
    logits = []
    for query, key, mask in zip(query_views, key_views, masks):
        if query.shape[:1] + query.shape[2:] != key.shape[:1] + key.shape[2:]:
            raise ValueError(
                f"Incompatible Q/K shapes: {tuple(query.shape)} and {tuple(key.shape)}"
            )
        edge_logits = torch.matmul(
            query.float().transpose(1, 2),
            key.float().transpose(1, 2).transpose(-2, -1),
        ) * (query.shape[-1] ** -0.5)
        edge_mask = _attention_mask_block(
            mask,
            query_slice=slice(0, query.shape[1]),
            key_slice=slice(0, key.shape[1]),
            batch_size=query.shape[0],
            device=query.device,
        )
        if edge_mask is not None:
            edge_logits = edge_logits.masked_fill(
                ~edge_mask[:, None], float("-inf")
            )
        logits.append(edge_logits)
    all_logits = torch.cat(logits, dim=-1)
    if torch.isneginf(all_logits).all(dim=-1).any():
        raise ValueError("An attention query has no visible key")
    probabilities = torch.softmax(all_logits, dim=-1)
    start = sum(key.shape[1] for key in key_views[:segment_index])
    stop = start + key_views[segment_index].shape[1]
    return probabilities[..., start:stop]


class FlexibleSelfAttention(nn.Module):
    """Self-attention whose residual and QKV dimensions are independent."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_dim = self.num_heads * self.head_dim
        self.q = nn.Linear(self.dim, self.attention_dim)
        self.k = nn.Linear(self.dim, self.attention_dim)
        self.v = nn.Linear(self.dim, self.attention_dim)
        self.o = nn.Linear(self.attention_dim, self.dim)
        self.norm_q = WanRMSNorm(self.attention_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.attention_dim, eps=eps) if qk_norm else nn.Identity()

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[:2]
        q = self.norm_q(self.q(x)).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.norm_k(self.k(x)).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return q, k, v


class ExpertCrossAttention(FlexibleSelfAttention):
    """Expert-owned text cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            eps=eps,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, query_len = x.shape[:2]
        q = self.norm_q(self.q(x)).reshape(
            batch_size, query_len, self.num_heads, self.head_dim
        )
        batch_size, context_len = context.shape[:2]
        k = self.norm_k(self.k(context)).reshape(
            batch_size,
            context_len,
            self.num_heads,
            self.head_dim,
        )
        v = self.v(context).reshape(
            batch_size,
            context_len,
            self.num_heads,
            self.head_dim,
        )
        attended = _segment_attention([q], [k], [v], [None])
        return self.o(attended.flatten(2))


class ActionLatentMemory(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self, patch_features: torch.Tensor, *, action_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_features.ndim != 5:
            raise ValueError(
                f"Expected patch features [B,D,T,H,W], got {tuple(patch_features.shape)}"
            )
        batch_size, dim, frames, height, width = patch_features.shape
        if dim != self.hidden_dim:
            raise ValueError(f"Expected latent dim={self.hidden_dim}, got {dim}")
        x = patch_features.permute(0, 2, 3, 4, 1).contiguous()
        x = self.proj(self.norm(x))
        axis_position = _sincos_position_embedding(
            frames + height + width, self.hidden_dim
        ).to(device=x.device, dtype=x.dtype)
        x = (
            x
            + axis_position[:frames][None, :, None, None]
            + axis_position[frames : frames + height][None, None, :, None]
            + axis_position[frames + height :][None, None, None]
        )
        memory = x.reshape(
            batch_size, frames * height * width, self.hidden_dim
        )
        frame_times = _sequence_times_on_action_axis(
            frames, action_frames, x.device
        )
        memory_times = (
            frame_times[:, None, None]
            .expand(frames, height, width)
            .reshape(-1)
        )
        return memory, memory_times


class ActionKeypointMemory(nn.Module):
    def __init__(
        self,
        *,
        keypoint_dims: int,
        num_hands: int,
        num_keypoints: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.keypoint_dims = int(keypoint_dims)
        self.num_hands = int(num_hands)
        self.num_keypoints = int(num_keypoints)
        self.hidden_dim = int(hidden_dim)
        self.coord_embed = nn.Sequential(
            nn.Linear(self.keypoint_dims, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.hand_embed = nn.Embedding(self.num_hands, self.hidden_dim)
        self.joint_embed = nn.Embedding(self.num_keypoints, self.hidden_dim)
        self.modality_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        self.null_token = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(
        self,
        keypoints: torch.Tensor,
        valid: torch.Tensor | None,
        *,
        action_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.num_hands, self.num_keypoints, self.keypoint_dims)
        if keypoints.ndim != 5 or tuple(keypoints.shape[-3:]) != expected:
            raise ValueError(
                f"Expected keypoints [B,T,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(keypoints.shape)}"
            )
        batch_size, frames, hands, joints, _ = keypoints.shape
        if valid is None:
            valid = torch.ones(
                (batch_size, frames, hands, joints),
                device=keypoints.device,
                dtype=torch.bool,
            )
        elif tuple(valid.shape) != (batch_size, frames, hands, joints):
            raise ValueError(f"Unexpected keypoint validity shape: {tuple(valid.shape)}")
        x = self.coord_embed(keypoints)
        hand_ids = torch.arange(hands, device=x.device)
        joint_ids = torch.arange(joints, device=x.device)
        temporal_position = _sincos_position_embedding(
            frames, self.hidden_dim
        ).to(device=x.device, dtype=x.dtype)
        x = (
            x
            + self.hand_embed(hand_ids)[None, None, :, None]
            + self.joint_embed(joint_ids)[None, None, None]
            + temporal_position[None, :, None, None]
            + self.modality_embed.to(x)[None, None, None, None]
        )
        memory = x.reshape(
            batch_size, frames * hands * joints, self.hidden_dim
        )
        memory_valid = valid.reshape(batch_size, frames * hands * joints)
        frame_times = _sequence_times_on_action_axis(
            frames, action_frames, x.device
        )
        memory_times = (
            frame_times[:, None, None]
            .expand(frames, hands, joints)
            .reshape(-1)
        )
        null = self.null_token.to(x)[None, None].expand(batch_size, 1, -1)
        memory = torch.cat([memory, null], dim=1)
        memory_valid = torch.cat(
            [
                memory_valid,
                torch.ones(
                    (batch_size, 1), device=x.device, dtype=torch.bool
                ),
            ],
            dim=1,
        )
        memory_times = torch.cat(
            [memory_times, memory_times.new_zeros(1)], dim=0
        )
        return memory, memory_valid, memory_times


class ActionCausalConditionLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_keypoints: bool,
    ) -> None:
        super().__init__()
        self.use_keypoints = bool(use_keypoints)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.latent_norm = nn.LayerNorm(dim)
        self.latent_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        if self.use_keypoints:
            self.keypoint_norm = nn.LayerNorm(dim)
            self.keypoint_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
            self.keypoint_gate = nn.Parameter(torch.tensor(0.0))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        action_mask: torch.Tensor,
        latent_memory: torch.Tensor,
        latent_mask: torch.Tensor,
        keypoint_memory: torch.Tensor | None,
        keypoint_mask: torch.Tensor | None,
        keypoint_padding_mask: torch.Tensor | None,
        keypoint_keep: torch.Tensor | None,
        capture_modality_contributions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized = self.self_norm(x)
        x = x + self.dropout(
            self.self_attn(
                normalized,
                normalized,
                normalized,
                attn_mask=action_mask,
                need_weights=False,
            )[0]
        )
        normalized = self.latent_norm(x)
        latent_delta = self.dropout(
            self.latent_attn(
                normalized,
                latent_memory,
                latent_memory,
                attn_mask=latent_mask,
                need_weights=False,
            )[0]
        )
        x = x + latent_delta
        keypoint_delta = None
        if self.use_keypoints:
            if (
                keypoint_memory is None
                or keypoint_mask is None
                or keypoint_padding_mask is None
            ):
                raise ValueError(
                    "Keypoint-conditioned Action layer requires keypoint memory and masks"
                )
            normalized = self.keypoint_norm(x)
            delta = self.keypoint_attn(
                normalized,
                keypoint_memory,
                keypoint_memory,
                attn_mask=keypoint_mask,
                key_padding_mask=keypoint_padding_mask,
                need_weights=False,
            )[0]
            if keypoint_keep is not None:
                delta = delta * keypoint_keep[:, None, None].to(delta)
            keypoint_delta = self.dropout(
                torch.sigmoid(self.keypoint_gate).to(delta) * delta
            )
            x = x + keypoint_delta
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if not capture_modality_contributions:
            return x
        if keypoint_delta is None:
            raise RuntimeError(
                "Action modality contribution capture requires keypoint conditioning"
            )
        # Both deltas are measured after their output projection, dropout, and
        # (for keypoints) learned gate, in the shared Action residual space.
        # Float32 RMS is stable under bf16/fp16 inference and independent of
        # the hidden width.
        return x, {
            "video_rms": latent_delta.float().square().mean(dim=-1).sqrt(),
            "keypoint_rms": keypoint_delta.float().square().mean(dim=-1).sqrt(),
            "keypoint_gate": torch.sigmoid(self.keypoint_gate.float()),
        }


class ActionCausalFMHead(nn.Module):
    """Causal Action FM decoder with latent and optional keypoint memory."""

    def __init__(
        self,
        *,
        action_frames: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        use_keypoints: bool,
    ) -> None:
        super().__init__()
        self.action_frames = int(action_frames)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_keypoints = bool(use_keypoints)
        self.action_embedder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                ActionCausalConditionLayer(
                    dim=self.hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_keypoints=self.use_keypoints,
                )
                for _ in range(num_layers)
            ]
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.register_buffer(
            "pos_embed",
            _sincos_position_embedding(
                self.action_frames, self.hidden_dim
            ).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        latent_memory: torch.Tensor,
        latent_times: torch.Tensor,
        keypoint_memory: torch.Tensor | None = None,
        keypoint_times: torch.Tensor | None = None,
        keypoint_valid: torch.Tensor | None = None,
        capture_modality_contributions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (self.action_frames, self.action_dim)
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != expected:
            raise ValueError(
                f"Expected noisy_action [B,{expected[0]},{expected[1]}], "
                f"got {tuple(noisy_action.shape)}"
            )
        x = self.action_embedder(noisy_action)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        timestep = _as_batch_timestep(timestep, x.shape[0])
        time_embedding = sinusoidal_embedding_1d(
            self.hidden_dim, timestep.to(device=x.device)
        ).to(dtype=x.dtype)
        x = x + self.time_mlp(time_embedding).unsqueeze(1)
        action_times = torch.arange(
            self.action_frames, device=x.device, dtype=torch.float32
        )
        action_mask = action_times[:, None] < action_times[None, :]
        latent_mask = (
            action_times[:, None]
            < latent_times.to(device=x.device)[None, :]
        )
        keypoint_mask = None
        keypoint_padding_mask = None
        keypoint_keep = None
        if self.use_keypoints:
            if (
                keypoint_memory is None
                or keypoint_times is None
                or keypoint_valid is None
            ):
                raise ValueError(
                    "Keypoint-conditioned Action head requires keypoint memory"
                )
            keypoint_mask = (
                action_times[:, None]
                < keypoint_times.to(device=x.device)[None, :]
            )
            keypoint_padding_mask = ~keypoint_valid.bool()
            keypoint_keep = torch.ones(
                x.shape[0], device=x.device, dtype=torch.bool
            )
        layer_contributions: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            layer_output = layer(
                x,
                action_mask=action_mask,
                latent_memory=latent_memory,
                latent_mask=latent_mask,
                keypoint_memory=keypoint_memory,
                keypoint_mask=keypoint_mask,
                keypoint_padding_mask=keypoint_padding_mask,
                keypoint_keep=keypoint_keep,
                capture_modality_contributions=capture_modality_contributions,
            )
            if capture_modality_contributions:
                if not isinstance(layer_output, tuple):
                    raise RuntimeError("Missing Action modality contributions")
                x, contribution = layer_output
                layer_contributions.append(contribution)
            else:
                if isinstance(layer_output, tuple):
                    raise RuntimeError("Unexpected Action modality contributions")
                x = layer_output
        velocity = self.velocity_head(x)
        if not capture_modality_contributions:
            return velocity
        video_rms = torch.stack(
            [item["video_rms"] for item in layer_contributions], dim=1
        )
        keypoint_rms = torch.stack(
            [item["keypoint_rms"] for item in layer_contributions], dim=1
        )
        denominator = (video_rms + keypoint_rms).clamp_min(1.0e-12)
        return velocity, {
            "video_rms": video_rms,
            "keypoint_rms": keypoint_rms,
            "video_share": video_rms / denominator,
            "keypoint_share": keypoint_rms / denominator,
            "keypoint_gate": torch.stack(
                [item["keypoint_gate"] for item in layer_contributions]
            ),
        }


class SkelWamActionModel(nn.Module):
    """Portable Action-only model without Video/Keypoint Transformer blocks."""

    def __init__(
        self,
        *,
        patch_weight: torch.Tensor,
        patch_bias: torch.Tensor,
        action_frames: int,
        action_dim: int,
        keypoint_dims: int = 3,
        num_hands: int = 2,
        num_keypoints: int = 21,
        num_heads: int = 24,
        action_condition_mode: Literal[
            "latent_only", "latent_keypoint"
        ] = "latent_keypoint",
    ) -> None:
        super().__init__()
        if patch_weight.ndim != 5 or patch_bias.ndim != 1:
            raise ValueError("Invalid Wan patch embedding tensors")
        if patch_weight.shape[0] != patch_bias.shape[0]:
            raise ValueError("Patch embedding weight/bias dimensions differ")
        patch_size = tuple(int(value) for value in patch_weight.shape[2:])
        self.action_patch_embedding = nn.Conv3d(
            in_channels=int(patch_weight.shape[1]),
            out_channels=int(patch_weight.shape[0]),
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=patch_weight.device,
            dtype=patch_weight.dtype,
        )
        self.action_patch_embedding.load_state_dict(
            {"weight": patch_weight, "bias": patch_bias},
            strict=True,
        )
        self.action_patch_embedding.requires_grad_(False)
        hidden_dim = int(patch_weight.shape[0])
        self.action_frames = int(action_frames)
        self.action_dim = int(action_dim)
        self.action_condition_mode = action_condition_mode
        self.action_latent_memory = ActionLatentMemory(hidden_dim)
        use_keypoints = action_condition_mode == "latent_keypoint"
        self.action_keypoint_memory = (
            ActionKeypointMemory(
                keypoint_dims=keypoint_dims,
                num_hands=num_hands,
                num_keypoints=num_keypoints,
                hidden_dim=hidden_dim,
            )
            if use_keypoints
            else None
        )
        self.action_fm_head = ActionCausalFMHead(
            action_frames=self.action_frames,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            ffn_dim=hidden_dim * 4,
            dropout=0.0,
            use_keypoints=use_keypoints,
        )

    def forward(
        self,
        *,
        condition_latents: torch.Tensor,
        noisy_action: torch.Tensor,
        action_timestep: torch.Tensor,
        condition_keypoints: torch.Tensor | None = None,
        condition_keypoints_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        patch_features = self.action_patch_embedding(condition_latents)
        latent_memory, latent_times = self.action_latent_memory(
            patch_features,
            action_frames=self.action_frames,
        )
        keypoint_memory = None
        keypoint_valid = None
        keypoint_times = None
        if self.action_condition_mode == "latent_keypoint":
            if self.action_keypoint_memory is None or condition_keypoints is None:
                raise ValueError("latent_keypoint mode requires keypoints")
            keypoint_memory, keypoint_valid, keypoint_times = (
                self.action_keypoint_memory(
                    condition_keypoints,
                    condition_keypoints_valid,
                    action_frames=self.action_frames,
                )
            )
        velocity = self.action_fm_head(
            noisy_action=noisy_action,
            timestep=action_timestep,
            latent_memory=latent_memory,
            latent_times=latent_times,
            keypoint_memory=keypoint_memory,
            keypoint_times=keypoint_times,
            keypoint_valid=keypoint_valid,
        )
        return {"action_velocity": velocity}


class KeypointDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.norm1 = WanLayerNorm(hidden_dim, eps)
        self.self_attn = FlexibleSelfAttention(
            hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm3 = (
            WanLayerNorm(hidden_dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = ExpertCrossAttention(
            hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm2 = WanLayerNorm(hidden_dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / hidden_dim**0.5
        )


class KeypointExpert(nn.Module):
    def __init__(
        self,
        *,
        keypoint_frames: int,
        keypoint_dims: int = 3,
        num_hands: int = 2,
        num_keypoints: int = 21,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 24,
        head_dim: int = 128,
        num_layers: int = 30,
        text_len: int = 512,
        text_dim: int = 4096,
        freq_dim: int = 256,
        use_text_cross_attention: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.keypoint_frames = int(keypoint_frames)
        self.keypoint_dims = int(keypoint_dims)
        self.num_hands = int(num_hands)
        self.num_keypoints = int(num_keypoints)
        self.keypoint_dim = self.num_hands * self.num_keypoints * self.keypoint_dims
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_layers = int(num_layers)
        self.text_len = int(text_len)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.use_text_cross_attention = bool(use_text_cross_attention)

        self.keypoint_input_dim = (
            self.num_keypoints * (self.keypoint_dims + 1)
        )
        self.keypoint_embedder = nn.Linear(
            self.keypoint_input_dim, self.hidden_dim
        )
        self.hand_embedding = nn.Embedding(self.num_hands, self.hidden_dim)
        self.role_embedding = nn.Embedding(2, self.hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                KeypointDiTBlock(
                    hidden_dim=self.hidden_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    eps=eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_norm = WanLayerNorm(self.hidden_dim, eps)
        self.velocity_head = nn.Linear(
            self.hidden_dim,
            self.num_keypoints * (self.keypoint_dims + 1),
        )
        complex_dim = self.head_dim // 2
        self.freqs = torch.cat(
            [
                rope_params(1024, self.head_dim - 4 * (self.head_dim // 6)),
                rope_params(1024, 2 * (self.head_dim // 6)),
                rope_params(1024, 2 * (self.head_dim // 6)),
            ],
            dim=1,
        )
        if self.freqs.shape[1] != complex_dim:
            raise RuntimeError("Unexpected Keypoint Expert RoPE dimension")
        self.temporal_freqs = rope_params(1024, self.head_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def flatten(self, keypoints: torch.Tensor) -> torch.Tensor:
        expected = (self.num_hands, self.num_keypoints, self.keypoint_dims)
        if keypoints.ndim != 5 or tuple(keypoints.shape[-3:]) != expected:
            raise ValueError(
                f"Expected keypoints [B,T,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(keypoints.shape)}"
            )
        return keypoints.reshape(keypoints.shape[0], keypoints.shape[1], -1)

    def keypoint_tokens(
        self,
        keypoints: torch.Tensor,
        normalized_validity: torch.Tensor | None,
    ) -> torch.Tensor:
        self.flatten(keypoints)
        if normalized_validity is None:
            normalized_validity = normalize_keypoint_validity(
                torch.ones(
                    keypoints.shape[:-1],
                    device=keypoints.device,
                    dtype=torch.bool,
                ),
                dtype=keypoints.dtype,
            )
        if not torch.is_floating_point(normalized_validity):
            raise TypeError(
                "keypoint_tokens expects normalized floating-point validity"
            )
        if tuple(normalized_validity.shape) != tuple(keypoints.shape[:-1]):
            raise ValueError(
                f"Expected validity {tuple(keypoints.shape[:-1])}, "
                f"got {tuple(normalized_validity.shape)}"
            )
        keypoints_with_validity = torch.cat(
            [
                keypoints,
                normalized_validity.to(
                    device=keypoints.device, dtype=keypoints.dtype
                ).unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(
            keypoints.shape[0],
            keypoints.shape[1],
            self.num_hands,
            self.keypoint_input_dim,
        )
        tokens = self.keypoint_embedder(keypoints_with_validity)
        hand_ids = torch.arange(self.num_hands, device=keypoints.device)
        tokens = tokens + self.hand_embedding(hand_ids)[None, None].to(tokens)
        return tokens.flatten(1, 2)

    def output(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, token_count = hidden.shape[:2]
        if token_count % self.num_hands:
            raise ValueError(
                f"Keypoint token count {token_count} is not divisible by "
                f"num_hands={self.num_hands}"
            )
        frames = token_count // self.num_hands
        hidden = self.out_norm(hidden).reshape(
            batch_size, frames, self.num_hands, self.hidden_dim
        )
        joint_velocity = self.velocity_head(hidden).reshape(
            batch_size,
            frames,
            self.num_hands,
            self.num_keypoints,
            self.keypoint_dims + 1,
        )
        # Frame 0 is a fixed clean condition, not a denoising target.
        joint_velocity = torch.cat(
            [torch.zeros_like(joint_velocity[:, :1]), joint_velocity[:, 1:]],
            dim=1,
        )
        return {
            "keypoint_flow_velocity": joint_velocity,
            "keypoint_velocity": joint_velocity[..., : self.keypoint_dims],
            "keypoint_validity_velocity": joint_velocity[..., self.keypoint_dims],
        }


@dataclass(frozen=True)
class ConditionVideoCache:
    layers: tuple[dict[str, torch.Tensor], ...]
    batch_size: int
    grid_size: tuple[int, int, int]
    dtype: torch.dtype
    device: torch.device
    condition_signature: tuple[Any, ...]
    text_signature: tuple[tuple[Any, ...], ...]
    model_id: int


@dataclass(frozen=True)
class HistoryKeypointCache:
    layers: tuple[dict[str, torch.Tensor], ...]
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    history_signature: tuple[Any, ...]
    current_signature: tuple[Any, ...]
    history_valid_signature: tuple[Any, ...] | None
    current_valid_signature: tuple[Any, ...] | None
    text_signature: tuple[tuple[Any, ...], ...]
    history_length: int
    model_id: int


class ExpertLayerScheduler(nn.Module):
    """Parameter-free layer scheduler for edge-specific-RoPE mixed attention."""

    def __init__(self, owner: "SkelWamBackboneModel") -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    @property
    def owner(self) -> "SkelWamBackboneModel":
        return object.__getattribute__(self, "_owner")

    @staticmethod
    def _prepare_block(
        block: nn.Module, x: torch.Tensor, modulation: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        with torch.amp.autocast(x.device.type, enabled=False):
            parts = (block.modulation.unsqueeze(0) + modulation.float()).chunk(6, dim=2)
        attention_input = (
            block.norm1(x).float() * (1 + parts[1].squeeze(2))
            + parts[0].squeeze(2)
        ).to(dtype=x.dtype)
        q, k, v = block.self_attn.project(attention_input) if isinstance(
            block.self_attn, FlexibleSelfAttention
        ) else ExpertLayerScheduler._project_wan_attention(block.self_attn, attention_input)
        return (
            q,
            k,
            v,
            x,
            parts[2].squeeze(2),
            parts[3].squeeze(2),
            parts[4].squeeze(2),
            parts[5].squeeze(2),
        )

    @staticmethod
    def _project_wan_attention(
        attention: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[:2]
        q = attention.norm_q(attention.q(x)).reshape(
            batch_size, seq_len, attention.num_heads, attention.head_dim
        )
        k = attention.norm_k(attention.k(x)).reshape(
            batch_size, seq_len, attention.num_heads, attention.head_dim
        )
        v = attention.v(x).reshape(
            batch_size, seq_len, attention.num_heads, attention.head_dim
        )
        return q, k, v

    @staticmethod
    def _finish_block(
        block: nn.Module,
        prepared: tuple[torch.Tensor, ...],
        attended: torch.Tensor,
        context: torch.Tensor | None,
        *,
        use_text_cross_attention: bool,
    ) -> torch.Tensor:
        _, _, _, residual, gate_attn, shift_ffn, scale_ffn, gate_ffn = prepared
        projected = block.self_attn.o(attended.flatten(2))
        with torch.amp.autocast(residual.device.type, enabled=False):
            x = residual.float() + projected.float() * gate_attn.float()
        x = x.to(dtype=residual.dtype)
        if use_text_cross_attention:
            if context is None:
                raise ValueError("Text context is required when cross-attention is enabled")
            cross_attention = block.cross_attn
            cross_input = block.norm3(x)
            if isinstance(cross_attention, ExpertCrossAttention):
                cross_out = cross_attention(cross_input, context)
            elif x.device.type == "cuda":
                # Video text attention has one positional view, so the native
                # Wan FlashAttention path is both valid and substantially faster.
                cross_out = cross_attention(cross_input, context, None)
            else:
                batch_size, query_len = cross_input.shape[:2]
                context_len = context.shape[1]
                q = cross_attention.norm_q(cross_attention.q(cross_input)).reshape(
                    batch_size,
                    query_len,
                    cross_attention.num_heads,
                    cross_attention.head_dim,
                )
                k = cross_attention.norm_k(cross_attention.k(context)).reshape(
                    batch_size,
                    context_len,
                    cross_attention.num_heads,
                    cross_attention.head_dim,
                )
                v = cross_attention.v(context).reshape(
                    batch_size,
                    context_len,
                    cross_attention.num_heads,
                    cross_attention.head_dim,
                )
                cross_out = cross_attention.o(
                    _segment_attention([q], [k], [v], [None]).flatten(2)
                )
            x = x + cross_out
        scale = scale_ffn.squeeze(2) if scale_ffn.ndim == 4 else scale_ffn
        shift = shift_ffn.squeeze(2) if shift_ffn.ndim == 4 else shift_ffn
        ffn_input = (
            block.norm2(x).float() * (1 + scale)
            + shift
        ).to(dtype=x.dtype)
        ffn_out = block.ffn(ffn_input)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float() + ffn_out.float() * gate_ffn.float()
        return x.to(dtype=residual.dtype)


class SkelWamBackboneModel(nn.Module):
    """Two-expert Skel-WAM with edge-specific RoPE and inference K/V caches."""

    def __init__(
        self,
        *,
        transformer_path: str | Path | None,
        keypoint_frames: int,
        keypoint_dims: int = 3,
        num_hands: int = 2,
        num_keypoints: int = 21,
        keypoint_hidden_dim: int = 1024,
        keypoint_ffn_dim: int = 4096,
        keypoint_expert_checkpoint: str | Path | None = None,
        keypoint_cond_video_mode: Literal["full", "causal"] = "full",
        keypoint_use_text_cross_attention: bool = False,
        action_frames: int = 49,
        action_dim: int = 0,
        predict_action: bool = False,
        action_condition_mode: Literal["latent_only", "latent_keypoint"] = "latent_keypoint",
        action_patch_embedding_mode: Literal[
            "shared", "portable_frozen"
        ] = "shared",
        use_keypoint: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        use_gradient_checkpointing: bool = False,
        checkpoint_full_layer: bool = True,
        load_pretrained_transformer: bool = True,
        video_transformer: WanModel | None = None,
    ) -> None:
        super().__init__()
        if keypoint_cond_video_mode not in {"full", "causal"}:
            raise ValueError(
                f"Unsupported keypoint_cond_video_mode: {keypoint_cond_video_mode}"
            )
        if action_condition_mode not in {"latent_only", "latent_keypoint"}:
            raise ValueError(f"Unsupported action_condition_mode: {action_condition_mode}")
        if action_patch_embedding_mode not in {"shared", "portable_frozen"}:
            raise ValueError(
                "Unsupported action_patch_embedding_mode: "
                f"{action_patch_embedding_mode}"
            )
        if video_transformer is not None:
            self.video_expert = video_transformer
        elif transformer_path is None:
            raise ValueError("transformer_path is required without video_transformer")
        elif load_pretrained_transformer:
            self.video_expert = WanModel.from_pretrained(
                str(transformer_path), torch_dtype=torch_dtype
            )
        else:
            config = WanModel.load_config(str(transformer_path))
            original_init = WanModel.init_weights
            try:
                WanModel.init_weights = lambda _: None
                self.video_expert = WanModel.from_config(config)
            finally:
                WanModel.init_weights = original_init

        self.keypoint_frames = int(keypoint_frames)
        self.keypoint_cond_video_mode = keypoint_cond_video_mode
        self.keypoint_use_text_cross_attention = bool(
            keypoint_use_text_cross_attention
        )
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.checkpoint_full_layer = bool(checkpoint_full_layer)
        self.action_frames = int(action_frames)
        self.action_dim = int(action_dim)
        self.predict_action = bool(predict_action)
        self.action_condition_mode = action_condition_mode
        self.action_patch_embedding_mode = action_patch_embedding_mode
        self.use_keypoint = bool(use_keypoint)
        video_head_dim = int(self.video_expert.dim // self.video_expert.num_heads)
        self.keypoint_expert = KeypointExpert(
            keypoint_frames=self.keypoint_frames,
            keypoint_dims=keypoint_dims,
            num_hands=num_hands,
            num_keypoints=num_keypoints,
            hidden_dim=keypoint_hidden_dim,
            ffn_dim=keypoint_ffn_dim,
            num_heads=int(self.video_expert.num_heads),
            head_dim=video_head_dim,
            num_layers=len(self.video_expert.blocks),
            text_len=int(self.video_expert.text_len),
            text_dim=int(self.video_expert.text_dim),
            freq_dim=int(self.video_expert.freq_dim),
            use_text_cross_attention=self.keypoint_use_text_cross_attention,
            eps=float(self.video_expert.eps),
        )
        if not self.keypoint_use_text_cross_attention:
            self.keypoint_expert.text_embedding.requires_grad_(False)
            for block in self.keypoint_expert.blocks:
                block.norm3.requires_grad_(False)
                block.cross_attn.requires_grad_(False)
        if not self.use_keypoint:
            # Do not retain Keypoint Expert parameters in a strict video-only
            # ablation model, optimizer, or checkpoint.
            self.keypoint_expert = None
        self.video_role_embedding = nn.Embedding(2, int(self.video_expert.dim))
        self.action_patch_embedding = None
        self.action_latent_memory = None
        self.action_keypoint_memory = None
        self.action_fm_head = None
        if self.predict_action:
            if self.action_dim <= 0:
                raise ValueError("action_dim must be positive when predict_action=True")
            if self.action_patch_embedding_mode == "portable_frozen":
                self.action_patch_embedding = clone_video_patch_embedding(
                    self.video_expert.patch_embedding
                )
                self.action_patch_embedding.requires_grad_(False)
            self.action_latent_memory = ActionLatentMemory(
                int(self.video_expert.dim)
            )
            use_keypoints = self.action_condition_mode == "latent_keypoint"
            if use_keypoints:
                self.action_keypoint_memory = ActionKeypointMemory(
                    keypoint_dims=keypoint_dims,
                    num_hands=num_hands,
                    num_keypoints=num_keypoints,
                    hidden_dim=int(self.video_expert.dim),
                )
            self.action_fm_head = ActionCausalFMHead(
                action_frames=self.action_frames,
                action_dim=self.action_dim,
                hidden_dim=int(self.video_expert.dim),
                num_heads=int(self.video_expert.num_heads),
                num_layers=2,
                ffn_dim=int(self.video_expert.dim) * 4,
                dropout=0.0,
                use_keypoints=use_keypoints,
            )
        self.expert_scheduler = ExpertLayerScheduler(self)
        if self.use_keypoint and keypoint_expert_checkpoint is not None:
            self.load_keypoint_expert_checkpoint(keypoint_expert_checkpoint)
        if device is not None:
            self.to(device=device, dtype=torch_dtype)

    def load_keypoint_expert_checkpoint(self, path: str | Path) -> None:
        if self.keypoint_expert is None:
            raise RuntimeError("Cannot load a Keypoint Expert when use_keypoint=False")
        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid Keypoint Expert checkpoint: {type(payload)}")
        metadata = payload.get("metadata")
        state_dict = payload.get("state_dict")
        if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
            raise ValueError("Keypoint Expert checkpoint requires metadata and state_dict")
        expected = {
            "hidden_dim": self.keypoint_expert.hidden_dim,
            "ffn_dim": self.keypoint_expert.ffn_dim,
            "num_layers": self.keypoint_expert.num_layers,
            "num_heads": self.keypoint_expert.num_heads,
            "head_dim": self.keypoint_expert.head_dim,
        }
        mismatch = {
            name: (expected_value, metadata.get(name))
            for name, expected_value in expected.items()
            if metadata.get(name) != expected_value
        }
        if mismatch:
            raise ValueError(f"Keypoint Expert metadata mismatch: {mismatch}")
        missing, unexpected = self.keypoint_expert.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            name
            for name in missing
            if name.startswith(
                (
                    "keypoint_embedder.",
                    "hand_embedding.",
                    "role_embedding.",
                    "out_norm.",
                    "velocity_head.",
                )
            )
        }
        real_missing = sorted(set(missing) - allowed_missing)
        if real_missing or unexpected:
            raise ValueError(
                f"Invalid Keypoint Expert state: missing={real_missing}, unexpected={unexpected}"
            )

    def forward(self, *, method: str, **kwargs):
        dispatch = {
            "forward_joint": self.forward_joint,
            "forward_video_only": self.forward_video_only,
            "forward_latent_step": self.forward_latent_step,
            "forward_keypoint_step": self.forward_keypoint_step,
            "forward_action_fm": self.forward_action_fm,
            "forward_robot": self.forward_robot,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown SkelWamBackboneModel method: {method}")
        return dispatch[method](**kwargs)

    def forward_action_fm(
        self,
        *,
        condition_latents: torch.Tensor,
        noisy_action: torch.Tensor,
        action_timestep: torch.Tensor,
        condition_keypoints: torch.Tensor | None = None,
        condition_keypoints_valid: torch.Tensor | None = None,
        capture_modality_contributions: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if self.action_fm_head is None or self.action_latent_memory is None:
            raise RuntimeError("Action FM is not initialized")
        if self.action_patch_embedding_mode == "shared":
            patch_features = self.video_expert.patch_embedding(
                condition_latents
            )
        else:
            if self.action_patch_embedding is None:
                raise RuntimeError("Portable Action patch is not initialized")
            patch_features = self.action_patch_embedding(condition_latents)
        # Video/Keypoint and Action are optimized together but remain separate
        # gradient domains.  Action may consume the current video patch features,
        # while its loss must not update the Video Expert patch embedding.
        patch_features = patch_features.detach()
        latent_memory, latent_times = self.action_latent_memory(
            patch_features,
            action_frames=self.action_frames,
        )
        keypoint_memory = None
        keypoint_valid = None
        keypoint_times = None
        if self.action_condition_mode == "latent_keypoint":
            if self.action_keypoint_memory is None or condition_keypoints is None:
                raise ValueError("latent_keypoint mode requires condition_keypoints")
            keypoint_memory, keypoint_valid, keypoint_times = (
                self.action_keypoint_memory(
                    condition_keypoints,
                    condition_keypoints_valid,
                    action_frames=self.action_frames,
                )
            )
        head_output = self.action_fm_head(
            noisy_action=noisy_action,
            timestep=action_timestep,
            latent_memory=latent_memory,
            latent_times=latent_times,
            keypoint_memory=keypoint_memory,
            keypoint_times=keypoint_times,
            keypoint_valid=keypoint_valid,
            capture_modality_contributions=capture_modality_contributions,
        )
        if not capture_modality_contributions:
            if isinstance(head_output, tuple):
                raise RuntimeError("Unexpected Action modality contributions")
            return {"action_velocity": head_output}
        if not isinstance(head_output, tuple):
            raise RuntimeError("Missing Action modality contributions")
        velocity, contributions = head_output
        return {
            "action_velocity": velocity,
            "action_modality_contributions": contributions,
        }

    def forward_robot(
        self,
        *,
        action_condition_latents: torch.Tensor,
        action_condition_keypoints: torch.Tensor | None,
        noisy_action: torch.Tensor,
        action_timestep: torch.Tensor,
        action_condition_keypoints_valid: torch.Tensor | None = None,
        **joint_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward_joint(**joint_kwargs)
        outputs.update(
            self.forward_action_fm(
                condition_latents=action_condition_latents,
                condition_keypoints=action_condition_keypoints,
                condition_keypoints_valid=action_condition_keypoints_valid,
                noisy_action=noisy_action,
                action_timestep=action_timestep,
            )
        )
        return outputs

    def _ensure_freqs(self, device: torch.device) -> None:
        if self.video_expert.freqs.device != device:
            self.video_expert.freqs = self.video_expert.freqs.to(device)
        if self.keypoint_expert is None:
            return
        if self.keypoint_expert.freqs.device != device:
            self.keypoint_expert.freqs = self.keypoint_expert.freqs.to(device)
        if self.keypoint_expert.temporal_freqs.device != device:
            self.keypoint_expert.temporal_freqs = (
                self.keypoint_expert.temporal_freqs.to(device)
            )

    def _patchify(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_5d = self.video_expert.patch_embedding(latents)
        grid_size = torch.tensor(
            tokens_5d.shape[2:], device=latents.device, dtype=torch.long
        )
        tokens = tokens_5d.flatten(2).transpose(1, 2)
        position_ids, frame_ids = _video_position_ids(
            grid_size, latents.shape[0], latents.device
        )
        return tokens, grid_size, position_ids, frame_ids

    def forward_video_only(
        self,
        *,
        noisy_latents: torch.Tensor,
        condition_latents: torch.Tensor,
        latent_timestep: torch.Tensor,
        text_emb: Sequence[torch.Tensor],
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Video FM without constructing or consuming any keypoint tokens."""
        self._ensure_freqs(noisy_latents.device)
        if noisy_latents.shape != condition_latents.shape:
            raise ValueError(
                "noisy_latents and condition_latents must have identical shapes"
            )
        model_latents = noisy_latents.clone()
        model_latents[:, :, 0:1] = condition_latents[:, :, 0:1]
        hidden, grid_size, position_ids, frame_ids = self._patchify(model_latents)
        hidden = hidden + self.video_role_embedding.weight[0].to(hidden)
        batch_size = hidden.shape[0]
        latent_timestep = _as_batch_timestep(latent_timestep, batch_size)
        token_timestep = latent_timestep[:, None].expand(
            batch_size, hidden.shape[1]
        ).clone()
        token_timestep = token_timestep.masked_fill(frame_ids == 0, 0)
        _, modulation = self._time_modulation(
            time_embedding=self.video_expert.time_embedding,
            time_projection=self.video_expert.time_projection,
            freq_dim=int(self.video_expert.freq_dim),
            hidden_dim=int(self.video_expert.dim),
            timesteps=token_timestep,
        )
        context = self._video_context(text_emb)
        causal_mask = _causal_by_time(frame_ids, frame_ids)

        for video_block in self.video_expert.blocks:
            def layer_fn(
                layer_hidden: torch.Tensor,
                block: nn.Module = video_block,
            ) -> torch.Tensor:
                prepared = self.expert_scheduler._prepare_block(block, layer_hidden, modulation)
                query, key, value = prepared[:3]
                query = _apply_rope_3d(query, position_ids, self.video_expert.freqs)
                key = _apply_rope_3d(key, position_ids, self.video_expert.freqs)
                attended = self._mixed_attention(
                    [query], [key], [value], [causal_mask]
                )
                return self.expert_scheduler._finish_block(
                    block,
                    prepared,
                    attended,
                    context,
                    use_text_cross_attention=True,
                )

            if self.training and self.use_gradient_checkpointing:
                hidden = checkpoint(layer_fn, hidden, use_reentrant=False)
            else:
                hidden = layer_fn(hidden)

        return {
            "latent_velocity": self._video_output(
                hidden, token_timestep, grid_size, noisy_latents.dtype
            )
        }

    def _video_context(self, text_emb: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.video_expert.text_embedding(
            _pad_text(text_emb, int(self.video_expert.text_len))
        )

    def _keypoint_context(self, text_emb: Sequence[torch.Tensor]) -> torch.Tensor | None:
        if not self.keypoint_use_text_cross_attention:
            return None
        return self.keypoint_expert.text_embedding(
            _pad_text(text_emb, self.keypoint_expert.text_len)
        )

    @staticmethod
    def _time_modulation(
        *,
        time_embedding: nn.Module,
        time_projection: nn.Module,
        freq_dim: int,
        hidden_dim: int,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = timesteps.shape
        parameter = next(time_embedding.parameters())
        sinusoidal = sinusoidal_embedding_1d(
            freq_dim, timesteps.flatten()
        ).unflatten(0, (batch_size, seq_len))
        embedded = time_embedding(
            sinusoidal.to(device=parameter.device, dtype=parameter.dtype)
        )
        modulation = time_projection(embedded).unflatten(2, (6, hidden_dim))
        return embedded.float(), modulation.float()

    def _aligned_video_time(
        self, frame_ids: torch.Tensor, video_frames: int, origin: int
    ) -> torch.Tensor:
        if video_frames <= 1:
            return torch.full_like(frame_ids, origin)
        scale = float(max(self.keypoint_frames - 1, 0)) / float(video_frames - 1)
        return torch.round(frame_ids.float() * scale).long() + origin

    def _mixed_attention(
        self,
        query_views: Sequence[torch.Tensor],
        key_views: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor | _CausalTimeMask | None],
    ) -> torch.Tensor:
        if (
            self.training
            and self.use_gradient_checkpointing
            and not self.checkpoint_full_layer
        ):
            num_edges = len(query_views)

            def attention_fn(*tensors: torch.Tensor) -> torch.Tensor:
                return _segment_attention(
                    tensors[:num_edges],
                    tensors[num_edges : 2 * num_edges],
                    tensors[2 * num_edges :],
                    masks,
                )

            return checkpoint(
                attention_fn,
                *query_views,
                *key_views,
                *values,
                use_reentrant=False,
            )
        return _segment_attention(query_views, key_views, values, masks)

    def _build_inputs(
        self,
        *,
        noisy_latents: torch.Tensor,
        condition_latents: torch.Tensor,
        noisy_keypoints: torch.Tensor,
        noisy_keypoint_validity: torch.Tensor,
        current_keypoints: torch.Tensor,
        history_keypoints: torch.Tensor,
        latent_timestep: torch.Tensor,
        keypoint_timestep: torch.Tensor,
        current_keypoints_valid: torch.Tensor | None = None,
        history_keypoints_valid: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if noisy_latents.shape != condition_latents.shape:
            raise ValueError("noisy_latents and condition_latents must have identical shapes")
        batch_size = noisy_latents.shape[0]
        model_latents = noisy_latents.clone()
        model_latents[:, :, 0:1] = condition_latents[:, :, 0:1]
        vn, grid_size, video_pos, video_frame = self._patchify(model_latents)
        vc, cond_grid, cond_pos, cond_frame = self._patchify(condition_latents)
        if not torch.equal(grid_size, cond_grid):
            raise ValueError("Noisy and condition video grids must match")
        vn = vn + self.video_role_embedding.weight[0].to(vn)
        vc = vc + self.video_role_embedding.weight[1].to(vc)

        model_keypoints = noisy_keypoints.clone()
        model_keypoints[:, 0:1] = current_keypoints
        if not torch.is_floating_point(noisy_keypoint_validity):
            raise TypeError("noisy_keypoint_validity must be floating point")
        if tuple(noisy_keypoint_validity.shape) != tuple(model_keypoints.shape[:-1]):
            raise ValueError(
                "Expected noisy_keypoint_validity "
                f"{tuple(model_keypoints.shape[:-1])}, "
                f"got {tuple(noisy_keypoint_validity.shape)}"
            )
        model_keypoint_validity = noisy_keypoint_validity.to(
            device=model_keypoints.device, dtype=model_keypoints.dtype
        ).clone()
        current_valid_for_kn = current_keypoints_valid
        if current_valid_for_kn is None:
            current_valid_for_kn = torch.ones(
                current_keypoints.shape[:-1],
                device=current_keypoints.device,
                dtype=torch.bool,
            )
        model_keypoint_validity[:, 0:1] = normalize_keypoint_validity(
            current_valid_for_kn,
            dtype=model_keypoints.dtype,
        )
        kn = self.keypoint_expert.keypoint_tokens(
            model_keypoints, model_keypoint_validity
        )
        kh_points = torch.cat([history_keypoints, current_keypoints], dim=1)
        kh_valid = None
        if history_keypoints_valid is not None or current_keypoints_valid is not None:
            if history_keypoints_valid is None or current_keypoints_valid is None:
                raise ValueError("History and current validity must be provided together")
            kh_valid = normalize_keypoint_validity(
                torch.cat(
                    [history_keypoints_valid, current_keypoints_valid], dim=1
                ),
                dtype=kh_points.dtype,
            )
        kh = self.keypoint_expert.keypoint_tokens(kh_points, kh_valid)
        kh = kh + self.keypoint_expert.role_embedding.weight[0].to(kh)
        kn = kn + self.keypoint_expert.role_embedding.weight[1].to(kn)

        latent_timestep = _as_batch_timestep(latent_timestep, batch_size)
        keypoint_timestep = _as_batch_timestep(keypoint_timestep, batch_size)
        vn_t = latent_timestep[:, None].expand(batch_size, vn.shape[1]).clone()
        vn_t = vn_t.masked_fill(video_frame == 0, 0)
        vc_t = latent_timestep.new_zeros((batch_size, vc.shape[1]))
        kh_t = keypoint_timestep.new_zeros((batch_size, kh.shape[1]))
        kn_frame_t = keypoint_timestep[:, None].expand(
            batch_size, model_keypoints.shape[1]
        ).clone()
        kn_frame_t[:, 0] = 0
        kn_t = kn_frame_t.repeat_interleave(
            self.keypoint_expert.num_hands, dim=1
        )

        history_length = history_keypoints.shape[1]
        origin = history_length
        kh_time = torch.arange(
            kh_points.shape[1], device=kh.device, dtype=torch.long
        ).repeat_interleave(self.keypoint_expert.num_hands)[None].expand(
            batch_size, -1
        )
        kn_time = (
            (
                torch.arange(
                    model_keypoints.shape[1],
                    device=kn.device,
                    dtype=torch.long,
                )
                + origin
            )
            .repeat_interleave(self.keypoint_expert.num_hands)[None]
            .expand(batch_size, -1)
        )
        video_frames = int(grid_size[0])
        vn_cross_time = self._aligned_video_time(video_frame, video_frames, origin)
        vc_cross_time = self._aligned_video_time(cond_frame, video_frames, origin)
        return {
            "vn": vn,
            "vc": vc,
            "kh": kh,
            "kn": kn,
            "grid_size": grid_size,
            "video_pos": video_pos,
            "cond_pos": cond_pos,
            "video_frame": video_frame,
            "cond_frame": cond_frame,
            "vn_cross_time": vn_cross_time,
            "vc_cross_time": vc_cross_time,
            "kh_time": kh_time,
            "kn_time": kn_time,
            "vn_t": vn_t,
            "vc_t": vc_t,
            "kh_t": kh_t,
            "kn_t": kn_t,
        }

    def _run_joint_layer(
        self,
        vn: torch.Tensor,
        vc: torch.Tensor,
        kh: torch.Tensor,
        kn: torch.Tensor,
        *,
        video_block: nn.Module,
        keypoint_block: nn.Module,
        video_modulation: torch.Tensor,
        keypoint_modulation: torch.Tensor,
        state: dict[str, Any],
        video_context: torch.Tensor,
        keypoint_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one complete Video/Keypoint expert layer."""
        vn_len, kh_len = vn.shape[1], kh.shape[1]
        video_x = torch.cat([vn, vc], dim=1)
        keypoint_x = torch.cat([kh, kn], dim=1)
        video_prepared = self.expert_scheduler._prepare_block(
            video_block, video_x, video_modulation
        )
        keypoint_prepared = self.expert_scheduler._prepare_block(
            keypoint_block, keypoint_x, keypoint_modulation
        )
        qv, kv, vv = video_prepared[:3]
        qk, kk, vk = keypoint_prepared[:3]
        q_vn, q_vc = qv[:, :vn_len], qv[:, vn_len:]
        k_vn, k_vc = kv[:, :vn_len], kv[:, vn_len:]
        v_vn, v_vc = vv[:, :vn_len], vv[:, vn_len:]
        q_kh, q_kn = qk[:, :kh_len], qk[:, kh_len:]
        k_kh, k_kn = kk[:, :kh_len], kk[:, kh_len:]
        v_kh, v_kn = vk[:, :kh_len], vk[:, kh_len:]

        video_freqs = self.video_expert.freqs
        keypoint_freqs = self.keypoint_expert.freqs
        vn_3d_q = _apply_rope_3d(q_vn, state["video_pos"], video_freqs)
        vn_3d_k = _apply_rope_3d(k_vn, state["video_pos"], video_freqs)
        vc_3d_q = _apply_rope_3d(q_vc, state["cond_pos"], video_freqs)
        vc_3d_k = _apply_rope_3d(k_vc, state["cond_pos"], video_freqs)
        kh_t_q = _apply_rope_1d(
            q_kh, state["kh_time"], self.keypoint_expert.temporal_freqs
        )
        kh_t_k = _apply_rope_1d(
            k_kh, state["kh_time"], self.keypoint_expert.temporal_freqs
        )
        kn_t_q = _apply_rope_1d(
            q_kn, state["kn_time"], self.keypoint_expert.temporal_freqs
        )
        kn_t_k = _apply_rope_1d(
            k_kn, state["kn_time"], self.keypoint_expert.temporal_freqs
        )
        vn_cross_q = _apply_rope_temporal(
            q_vn, state["vn_cross_time"], video_freqs
        )
        kh_cross_k = _apply_rope_temporal(
            k_kh, state["kh_time"], keypoint_freqs
        )
        kn_cross_q = _apply_rope_temporal(
            q_kn, state["kn_time"], keypoint_freqs
        )
        vc_cross_k = _apply_rope_temporal(
            k_vc, state["vc_cross_time"], video_freqs
        )

        mask_vn = _causal_by_time(state["video_frame"], state["video_frame"])
        mask_vc = _causal_by_time(state["cond_frame"], state["cond_frame"])
        mask_kh = _causal_by_time(state["kh_time"], state["kh_time"])
        mask_kn = _causal_by_time(state["kn_time"], state["kn_time"])
        if self.keypoint_cond_video_mode == "full":
            mask_kn_vc = None
        else:
            mask_kn_vc = _causal_by_time(
                state["kn_time"], state["vc_cross_time"]
            )
        out_vn = self._mixed_attention(
            [vn_3d_q, vn_cross_q],
            [vn_3d_k, kh_cross_k],
            [v_vn, v_kh],
            [mask_vn, None],
        )
        out_vc = self._mixed_attention(
            [vc_3d_q], [vc_3d_k], [v_vc], [mask_vc]
        )
        out_kh = self._mixed_attention(
            [kh_t_q], [kh_t_k], [v_kh], [mask_kh]
        )
        out_kn = self._mixed_attention(
            [kn_cross_q, kn_t_q],
            [vc_cross_k, kn_t_k],
            [v_vc, v_kn],
            [mask_kn_vc, mask_kn],
        )
        video_x = self.expert_scheduler._finish_block(
            video_block,
            video_prepared,
            torch.cat([out_vn, out_vc], dim=1),
            video_context,
            use_text_cross_attention=True,
        )
        keypoint_x = self.expert_scheduler._finish_block(
            keypoint_block,
            keypoint_prepared,
            torch.cat([out_kh, out_kn], dim=1),
            keypoint_context,
            use_text_cross_attention=self.keypoint_use_text_cross_attention,
        )
        return (
            video_x[:, :vn_len],
            video_x[:, vn_len:],
            keypoint_x[:, :kh_len],
            keypoint_x[:, kh_len:],
        )

    def _run_layers(
        self,
        state: dict[str, Any],
        *,
        video_context: torch.Tensor,
        keypoint_context: torch.Tensor | None,
        capture_layers: frozenset[int] | None = None,
        feature_callback: Callable[
            [int, dict[str, torch.Tensor], dict[str, Any]], None
        ]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        vn, vc, kh, kn = state["vn"], state["vc"], state["kh"], state["kn"]
        if (capture_layers is None) != (feature_callback is None):
            raise ValueError(
                "capture_layers and feature_callback must be provided together"
            )
        if capture_layers is not None:
            invalid_layers = sorted(
                layer
                for layer in capture_layers
                if layer < -1 or layer >= len(self.video_expert.blocks)
            )
            if invalid_layers:
                raise ValueError(
                    "Feature capture layer indices must be -1 (pre-expert) or in "
                    f"[0,{len(self.video_expert.blocks) - 1}], got {invalid_layers}"
                )

            if -1 in capture_layers:
                feature_callback(
                    -1,
                    {"vn": vn, "vc": vc, "kh": kh, "kn": kn},
                    state,
                )
        video_time = torch.cat([state["vn_t"], state["vc_t"]], dim=1)
        keypoint_time = torch.cat([state["kh_t"], state["kn_t"]], dim=1)
        _, video_mod = self._time_modulation(
            time_embedding=self.video_expert.time_embedding,
            time_projection=self.video_expert.time_projection,
            freq_dim=int(self.video_expert.freq_dim),
            hidden_dim=int(self.video_expert.dim),
            timesteps=video_time,
        )
        _, kp_mod = self._time_modulation(
            time_embedding=self.keypoint_expert.time_embedding,
            time_projection=self.keypoint_expert.time_projection,
            freq_dim=self.keypoint_expert.freq_dim,
            hidden_dim=self.keypoint_expert.hidden_dim,
            timesteps=keypoint_time,
        )

        def tensor_debug(name: str, tensor: torch.Tensor) -> str:
            finite = torch.isfinite(tensor)
            finite_count = int(finite.sum().item())
            if finite_count:
                finite_values = tensor[finite].float()
                value_range = (
                    f"min={finite_values.min().item():.6g},"
                    f"max={finite_values.max().item():.6g},"
                    f"absmax={finite_values.abs().max().item():.6g}"
                )
            else:
                value_range = "no-finite-values"
            return (
                f"{name}[shape={tuple(tensor.shape)},dtype={tensor.dtype},"
                f"finite={finite_count}/{tensor.numel()},{value_range}]"
            )

        def layer_failure_context(
            video_block: nn.Module,
            keypoint_block: nn.Module,
        ) -> str:
            tensors = [
                ("vn", vn),
                ("vc", vc),
                ("kh", kh),
                ("kn", kn),
                ("video_mod", video_mod),
                ("keypoint_mod", kp_mod),
                ("video_v_weight", video_block.self_attn.v.weight),
                ("keypoint_v_weight", keypoint_block.self_attn.v.weight),
            ]
            return "; ".join(tensor_debug(name, tensor) for name, tensor in tensors)
        for layer_index, (video_block, keypoint_block) in enumerate(zip(
            self.video_expert.blocks, self.keypoint_expert.blocks
        )):
            def layer_fn(
                vn_input: torch.Tensor,
                vc_input: torch.Tensor,
                kh_input: torch.Tensor,
                kn_input: torch.Tensor,
                video_block_for_layer: nn.Module = video_block,
                keypoint_block_for_layer: nn.Module = keypoint_block,
            ) -> tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
                return self._run_joint_layer(
                    vn_input,
                    vc_input,
                    kh_input,
                    kn_input,
                    video_block=video_block_for_layer,
                    keypoint_block=keypoint_block_for_layer,
                    video_modulation=video_mod,
                    keypoint_modulation=kp_mod,
                    state=state,
                    video_context=video_context,
                    keypoint_context=keypoint_context,
                )

            if (
                self.training
                and self.use_gradient_checkpointing
                and self.checkpoint_full_layer
            ):
                try:
                    vn, vc, kh, kn = checkpoint(
                        layer_fn,
                        vn,
                        vc,
                        kh,
                        kn,
                        use_reentrant=False,
                    )
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"expert layer {layer_index} failed: {exc}; "
                        + layer_failure_context(video_block, keypoint_block)
                    ) from exc
            else:
                try:
                    vn, vc, kh, kn = layer_fn(vn, vc, kh, kn)
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"expert layer {layer_index} failed: {exc}; "
                        + layer_failure_context(video_block, keypoint_block)
                    ) from exc
            if capture_layers is not None and layer_index in capture_layers:
                feature_callback(
                    layer_index,
                    {"vn": vn, "vc": vc, "kh": kh, "kn": kn},
                    state,
                )
        return {"vn": vn, "vc": vc, "kh": kh, "kn": kn}

    def _video_output(
        self,
        hidden: torch.Tensor,
        timesteps: torch.Tensor,
        grid_size: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        embedded, _ = self._time_modulation(
            time_embedding=self.video_expert.time_embedding,
            time_projection=self.video_expert.time_projection,
            freq_dim=int(self.video_expert.freq_dim),
            hidden_dim=int(self.video_expert.dim),
            timesteps=timesteps,
        )
        patches = self.video_expert.head(hidden, embedded)
        grid_sizes = grid_size[None].expand(hidden.shape[0], -1).contiguous()
        return self.video_expert.unpatchify(patches, grid_sizes).to(dtype=output_dtype)

    def forward_joint(
        self,
        *,
        noisy_latents: torch.Tensor,
        condition_latents: torch.Tensor,
        noisy_keypoints: torch.Tensor,
        noisy_keypoint_validity: torch.Tensor,
        current_keypoints: torch.Tensor,
        history_keypoints: torch.Tensor,
        latent_timestep: torch.Tensor,
        keypoint_timestep: torch.Tensor,
        text_emb: Sequence[torch.Tensor],
        current_keypoints_valid: torch.Tensor | None = None,
        history_keypoints_valid: torch.Tensor | None = None,
        capture_layers: frozenset[int] | None = None,
        feature_callback: Callable[
            [int, dict[str, torch.Tensor], dict[str, Any]], None
        ]
        | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        self._ensure_freqs(noisy_latents.device)
        state = self._build_inputs(
            noisy_latents=noisy_latents,
            condition_latents=condition_latents,
            noisy_keypoints=noisy_keypoints,
            noisy_keypoint_validity=noisy_keypoint_validity,
            current_keypoints=current_keypoints,
            history_keypoints=history_keypoints,
            latent_timestep=latent_timestep,
            keypoint_timestep=keypoint_timestep,
            current_keypoints_valid=current_keypoints_valid,
            history_keypoints_valid=history_keypoints_valid,
        )
        outputs = self._run_layers(
            state,
            video_context=self._video_context(text_emb),
            keypoint_context=self._keypoint_context(text_emb),
            capture_layers=capture_layers,
            feature_callback=feature_callback,
        )
        result = self.keypoint_expert.output(outputs["kn"])
        result["latent_velocity"] = self._video_output(
            outputs["vn"], state["vn_t"], state["grid_size"], noisy_latents.dtype
        )
        return result

    @torch.no_grad()
    def prefill_condition_video_cache(
        self,
        *,
        condition_latents: torch.Tensor,
        text_emb: Sequence[torch.Tensor],
    ) -> ConditionVideoCache:
        """Encode fixed condition video once for repeated keypoint denoising."""
        self._ensure_freqs(condition_latents.device)
        vc, grid_size, position_ids, frame_ids = self._patchify(condition_latents)
        vc = vc + self.video_role_embedding.weight[1].to(vc)
        batch_size = vc.shape[0]
        timesteps = vc.new_zeros((batch_size, vc.shape[1]))
        _, modulation = self._time_modulation(
            time_embedding=self.video_expert.time_embedding,
            time_projection=self.video_expert.time_projection,
            freq_dim=int(self.video_expert.freq_dim),
            hidden_dim=int(self.video_expert.dim),
            timesteps=timesteps,
        )
        context = self._video_context(text_emb)
        video_frames = int(grid_size[0])
        temporal_ids = self._aligned_video_time(frame_ids, video_frames, 0)
        self_mask = _causal_by_time(frame_ids, frame_ids)
        layers: list[dict[str, torch.Tensor]] = []
        for block in self.video_expert.blocks:
            prepared = self.expert_scheduler._prepare_block(block, vc, modulation)
            q, k, v = prepared[:3]
            layers.append(
                {
                    "k_temporal": _apply_rope_temporal(
                        k, temporal_ids, self.video_expert.freqs
                    ).detach(),
                    "v": v.detach(),
                    "temporal_ids": temporal_ids.detach(),
                }
            )
            out = _segment_attention(
                [_apply_rope_3d(q, position_ids, self.video_expert.freqs)],
                [_apply_rope_3d(k, position_ids, self.video_expert.freqs)],
                [v],
                [self_mask],
            )
            vc = self.expert_scheduler._finish_block(
                block,
                prepared,
                out,
                context,
                use_text_cross_attention=True,
            )
        return ConditionVideoCache(
            layers=tuple(layers),
            batch_size=batch_size,
            grid_size=tuple(int(value) for value in grid_size.tolist()),
            dtype=condition_latents.dtype,
            device=condition_latents.device,
            condition_signature=_tensor_signature(condition_latents),
            text_signature=_text_signature(text_emb),
            model_id=id(self),
        )

    @torch.no_grad()
    def prefill_history_keypoint_cache(
        self,
        *,
        history_keypoints: torch.Tensor,
        current_keypoints: torch.Tensor,
        text_emb: Sequence[torch.Tensor],
        history_keypoints_valid: torch.Tensor | None = None,
        current_keypoints_valid: torch.Tensor | None = None,
    ) -> HistoryKeypointCache:
        """Encode fixed history/current keypoints once for repeated video denoising."""
        self._ensure_freqs(history_keypoints.device)
        points = torch.cat([history_keypoints, current_keypoints], dim=1)
        valid = None
        if history_keypoints_valid is not None or current_keypoints_valid is not None:
            if history_keypoints_valid is None or current_keypoints_valid is None:
                raise ValueError("History and current validity must be provided together")
            valid = normalize_keypoint_validity(
                torch.cat(
                    [history_keypoints_valid, current_keypoints_valid], dim=1
                ),
                dtype=points.dtype,
            )
        kh = self.keypoint_expert.keypoint_tokens(points, valid)
        kh = kh + self.keypoint_expert.role_embedding.weight[0].to(kh)
        batch_size = kh.shape[0]
        timesteps = kh.new_zeros((batch_size, kh.shape[1]))
        _, modulation = self._time_modulation(
            time_embedding=self.keypoint_expert.time_embedding,
            time_projection=self.keypoint_expert.time_projection,
            freq_dim=self.keypoint_expert.freq_dim,
            hidden_dim=self.keypoint_expert.hidden_dim,
            timesteps=timesteps,
        )
        temporal_ids = torch.arange(
            points.shape[1], device=kh.device, dtype=torch.long
        ).repeat_interleave(self.keypoint_expert.num_hands)[None].expand(
            batch_size, -1
        )
        self_mask = _causal_by_time(temporal_ids, temporal_ids)
        context = self._keypoint_context(text_emb)
        layers: list[dict[str, torch.Tensor]] = []
        for block in self.keypoint_expert.blocks:
            prepared = self.expert_scheduler._prepare_block(block, kh, modulation)
            q, k, v = prepared[:3]
            layers.append(
                {
                    "k_temporal": _apply_rope_temporal(
                        k, temporal_ids, self.keypoint_expert.freqs
                    ).detach(),
                    "v": v.detach(),
                    "temporal_ids": temporal_ids.detach(),
                }
            )
            out = _segment_attention(
                [_apply_rope_1d(q, temporal_ids, self.keypoint_expert.temporal_freqs)],
                [_apply_rope_1d(k, temporal_ids, self.keypoint_expert.temporal_freqs)],
                [v],
                [self_mask],
            )
            kh = self.expert_scheduler._finish_block(
                block,
                prepared,
                out,
                context,
                use_text_cross_attention=self.keypoint_use_text_cross_attention,
            )
        return HistoryKeypointCache(
            layers=tuple(layers),
            batch_size=batch_size,
            dtype=history_keypoints.dtype,
            device=history_keypoints.device,
            history_signature=_tensor_signature(history_keypoints),
            current_signature=_tensor_signature(current_keypoints),
            history_valid_signature=(
                _tensor_signature(history_keypoints_valid)
                if history_keypoints_valid is not None
                else None
            ),
            current_valid_signature=(
                _tensor_signature(current_keypoints_valid)
                if current_keypoints_valid is not None
                else None
            ),
            text_signature=(
                _text_signature(text_emb)
                if self.keypoint_use_text_cross_attention
                else ()
            ),
            history_length=int(history_keypoints.shape[1]),
            model_id=id(self),
        )

    def _validate_cache_common(
        self,
        *,
        cache: ConditionVideoCache | HistoryKeypointCache,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        text_emb: Sequence[torch.Tensor],
    ) -> None:
        if len(cache.layers) != len(self.video_expert.blocks):
            raise ValueError(
                f"Cache layer count mismatch: expected {len(self.video_expert.blocks)}, "
                f"got {len(cache.layers)}"
            )
        if cache.model_id != id(self):
            raise ValueError("Cache was created by a different model instance")
        if cache.batch_size != batch_size:
            raise ValueError(
                f"Cache batch mismatch: expected {cache.batch_size}, got {batch_size}"
            )
        if cache.dtype != dtype or cache.device != device:
            raise ValueError(
                f"Cache dtype/device mismatch: cache={cache.dtype}/{cache.device}, "
                f"input={dtype}/{device}"
            )
        if cache.text_signature and cache.text_signature != _text_signature(text_emb):
            raise ValueError("Text condition changed; rebuild the inference cache")

    def forward_keypoint_step(
        self,
        *,
        noisy_keypoints: torch.Tensor,
        noisy_keypoint_validity: torch.Tensor,
        current_keypoints: torch.Tensor,
        keypoint_timestep: torch.Tensor,
        text_emb: Sequence[torch.Tensor],
        condition_video_cache: ConditionVideoCache,
        current_keypoints_valid: torch.Tensor | None = None,
        capture_attention: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Denoise keypoints while reusing fixed condition-video layer K/V."""
        batch_size = noisy_keypoints.shape[0]
        self._validate_cache_common(
            cache=condition_video_cache,
            batch_size=batch_size,
            dtype=noisy_keypoints.dtype,
            device=noisy_keypoints.device,
            text_emb=text_emb,
        )
        self._ensure_freqs(noisy_keypoints.device)
        model_keypoints = noisy_keypoints.clone()
        model_keypoints[:, 0:1] = current_keypoints
        if not torch.is_floating_point(noisy_keypoint_validity):
            raise TypeError("noisy_keypoint_validity must be floating point")
        if tuple(noisy_keypoint_validity.shape) != tuple(model_keypoints.shape[:-1]):
            raise ValueError(
                "Expected noisy_keypoint_validity "
                f"{tuple(model_keypoints.shape[:-1])}, "
                f"got {tuple(noisy_keypoint_validity.shape)}"
            )
        model_keypoint_validity = noisy_keypoint_validity.to(
            device=model_keypoints.device, dtype=model_keypoints.dtype
        ).clone()
        if current_keypoints_valid is None:
            current_keypoints_valid = torch.ones(
                current_keypoints.shape[:-1],
                device=current_keypoints.device,
                dtype=torch.bool,
            )
        model_keypoint_validity[:, 0:1] = normalize_keypoint_validity(
            current_keypoints_valid,
            dtype=model_keypoints.dtype,
        )
        kn = self.keypoint_expert.keypoint_tokens(
            model_keypoints, model_keypoint_validity
        )
        kn = kn + self.keypoint_expert.role_embedding.weight[1].to(kn)
        timestep = _as_batch_timestep(keypoint_timestep, batch_size)
        kn_frame_timestep = timestep[:, None].expand(
            batch_size, model_keypoints.shape[1]
        ).clone()
        kn_frame_timestep[:, 0] = 0
        kn_timestep = kn_frame_timestep.repeat_interleave(
            self.keypoint_expert.num_hands, dim=1
        )
        _, modulation = self._time_modulation(
            time_embedding=self.keypoint_expert.time_embedding,
            time_projection=self.keypoint_expert.time_projection,
            freq_dim=self.keypoint_expert.freq_dim,
            hidden_dim=self.keypoint_expert.hidden_dim,
            timesteps=kn_timestep,
        )
        temporal_ids = torch.arange(
            model_keypoints.shape[1], device=kn.device, dtype=torch.long
        ).repeat_interleave(self.keypoint_expert.num_hands)[None].expand(
            batch_size, -1
        )
        self_mask = _causal_by_time(temporal_ids, temporal_ids)
        context = self._keypoint_context(text_emb)
        captured_attention: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.keypoint_expert.blocks):
            prepared = self.expert_scheduler._prepare_block(block, kn, modulation)
            q, k, v = prepared[:3]
            cached = condition_video_cache.layers[layer_index]
            if self.keypoint_cond_video_mode == "full":
                cross_mask = None
            else:
                cross_mask = _causal_by_time(
                    temporal_ids, cached["temporal_ids"]
                )
            cross_query = _apply_rope_temporal(
                q, temporal_ids, self.keypoint_expert.freqs
            )
            self_query = _apply_rope_1d(
                q, temporal_ids, self.keypoint_expert.temporal_freqs
            )
            self_key = _apply_rope_1d(
                k, temporal_ids, self.keypoint_expert.temporal_freqs
            )
            query_views = [cross_query, self_query]
            key_views = [cached["k_temporal"], self_key]
            masks = [cross_mask, self_mask]
            out = _segment_attention(
                query_views,
                key_views,
                [cached["v"], v],
                masks,
            )
            if capture_attention:
                # Keep only the actual Kn->Vc edge, normalized jointly with
                # the Kn self-attention keys. Head averaging and CPU transfer
                # keep diagnostics small and do not affect the model output.
                captured_attention.append(
                    _segment_attention_probabilities(
                        query_views,
                        key_views,
                        masks,
                        segment_index=0,
                    )
                    .mean(dim=1)
                    .detach()
                    .float()
                    .cpu()
                )
            kn = self.expert_scheduler._finish_block(
                block,
                prepared,
                out,
                context,
                use_text_cross_attention=self.keypoint_use_text_cross_attention,
            )
        result = self.keypoint_expert.output(kn)
        if capture_attention:
            result["kn_vc_attention"] = torch.stack(
                captured_attention, dim=1
            )
            result["kn_vc_attention_grid_size"] = torch.tensor(
                condition_video_cache.grid_size,
                dtype=torch.long,
            )
        return result

    def forward_latent_step(
        self,
        *,
        noisy_latents: torch.Tensor,
        condition_latents: torch.Tensor,
        latent_timestep: torch.Tensor,
        text_emb: Sequence[torch.Tensor],
        history_keypoint_cache: HistoryKeypointCache,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Denoise video while reusing fixed history/current-keypoint layer K/V."""
        batch_size = noisy_latents.shape[0]
        self._validate_cache_common(
            cache=history_keypoint_cache,
            batch_size=batch_size,
            dtype=noisy_latents.dtype,
            device=noisy_latents.device,
            text_emb=text_emb,
        )
        self._ensure_freqs(noisy_latents.device)
        model_latents = noisy_latents.clone()
        model_latents[:, :, 0:1] = condition_latents[:, :, 0:1]
        vn, grid_size, position_ids, frame_ids = self._patchify(model_latents)
        vn = vn + self.video_role_embedding.weight[0].to(vn)
        timestep = _as_batch_timestep(latent_timestep, batch_size)
        token_timestep = timestep[:, None].expand(batch_size, vn.shape[1]).clone()
        token_timestep = token_timestep.masked_fill(frame_ids == 0, 0)
        _, modulation = self._time_modulation(
            time_embedding=self.video_expert.time_embedding,
            time_projection=self.video_expert.time_projection,
            freq_dim=int(self.video_expert.freq_dim),
            hidden_dim=int(self.video_expert.dim),
            timesteps=token_timestep,
        )
        origin = history_keypoint_cache.history_length
        cross_time = self._aligned_video_time(frame_ids, int(grid_size[0]), origin)
        self_mask = _causal_by_time(frame_ids, frame_ids)
        cross_mask = None
        context = self._video_context(text_emb)
        for layer_index, block in enumerate(self.video_expert.blocks):
            prepared = self.expert_scheduler._prepare_block(block, vn, modulation)
            q, k, v = prepared[:3]
            cached = history_keypoint_cache.layers[layer_index]
            out = _segment_attention(
                [
                    _apply_rope_3d(q, position_ids, self.video_expert.freqs),
                    _apply_rope_temporal(q, cross_time, self.video_expert.freqs),
                ],
                [
                    _apply_rope_3d(k, position_ids, self.video_expert.freqs),
                    cached["k_temporal"],
                ],
                [v, cached["v"]],
                [self_mask, cross_mask],
            )
            vn = self.expert_scheduler._finish_block(
                block,
                prepared,
                out,
                context,
                use_text_cross_attention=True,
            )
        return {
            "latent_velocity": self._video_output(
                vn, token_timestep, grid_size, noisy_latents.dtype
            )
        }
