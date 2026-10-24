"""Condition corruption shared by Skel-WAM training."""
from __future__ import annotations

import torch


def corrupt_condition_tensor(
    clean: torch.Tensor,
    *,
    noise_prob: float = 0.5,
    min_timestep: float = 0.0,
    max_timestep: float = 0.5,
    preserve_first_frame_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt a clean condition under the Skel-WAM FM convention.

    Skel-WAM follows the Wan FM convention ``x_t = (1 - t) * clean + t * noise``;
    therefore ``t=0`` is the ground-truth condition and larger ``t`` is noisier.
    With the default interval [0.0, 0.5], noisy conditions are at most half-noised.
    If ``preserve_first_frame_dim`` is set, frame 0 along that dimension remains clean
    because the current observation is known. Returned values are the corrupted tensor
    and the condition timestep.
    """
    if not 0.0 <= min_timestep <= max_timestep <= 1.0:
        raise ValueError(
            f"condition timestep range must satisfy 0 <= min <= max <= 1, got {min_timestep}, {max_timestep}"
        )
    batch_size = int(clean.shape[0])
    timesteps = clean.new_zeros((batch_size,), dtype=clean.dtype)
    if noise_prob <= 0.0 or max_timestep <= 0.0:
        return clean.detach(), timesteps.detach().float()
    apply_noise = torch.rand(batch_size, device=clean.device) < float(noise_prob)
    sampled = torch.empty(batch_size, device=clean.device, dtype=clean.dtype).uniform_(
        float(min_timestep), float(max_timestep)
    )
    timesteps = torch.where(apply_noise, sampled, timesteps)
    t_view = timesteps.reshape(batch_size, *([1] * (clean.ndim - 1)))
    noisy = (1.0 - t_view) * clean + t_view * torch.randn_like(clean)
    if preserve_first_frame_dim is not None:
        frame_dim = int(preserve_first_frame_dim)
        if frame_dim < 0:
            frame_dim += clean.ndim
        if frame_dim <= 0 or frame_dim >= clean.ndim:
            raise ValueError(f"preserve_first_frame_dim must refer to a non-batch tensor dimension, got {preserve_first_frame_dim}")
        index = [slice(None)] * clean.ndim
        index[frame_dim] = slice(0, 1)
        noisy[tuple(index)] = clean[tuple(index)]
    return noisy.detach(), timesteps.detach().float()
