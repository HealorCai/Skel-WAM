from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class FlowMatchingBatch:
    sample: torch.Tensor
    noise: torch.Tensor
    timestep: torch.Tensor
    velocity_target: torch.Tensor


class FlowMatchingScheduler:
    """Straight-line flow matching scheduler for training and Euler inference.

    The model timestep follows the Wan flow-matching convention where ``t=0``
    is clean data and ``t=1`` is noise: ``x_t = (1 - t) * target + t * noise``.
    The velocity target is ``noise - target``. ``snr_shift`` uses the same
    sigma-shift form as Wan/Lingbot-VA; larger values bias training timesteps
    toward noisier, larger-``t`` points while keeping the same endpoints.
    """

    def __init__(
        self,
        *,
        noise_std: float = 1.0,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
        snr_shift: float = 1.0,
        eps: float = 0.0,
    ) -> None:
        if noise_std <= 0:
            raise ValueError(f"`noise_std` must be positive, got {noise_std}")
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        if snr_shift <= 0:
            raise ValueError(f"`snr_shift` must be positive, got {snr_shift}")
        if eps < 0 or eps >= 0.5:
            raise ValueError(f"`eps` must be in [0, 0.5), got {eps}")
        self.noise_std = float(noise_std)
        self.num_train_timesteps = int(num_train_timesteps)
        self.snr_shift = float(snr_shift)
        self.eps = float(eps)
        self.sigmas: torch.Tensor
        self.linear_timesteps_weights: torch.Tensor
        self.timesteps: torch.Tensor
        self.deltas: torch.Tensor
        self.num_inference_steps = 0
        self.set_timesteps(num_inference_steps)

    def _shift_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.snr_shift * sigma / (1.0 + (self.snr_shift - 1.0) * sigma)

    def _shift_progress(self, progress: torch.Tensor) -> torch.Tensor:
        return self._shift_sigma(progress)

    def set_training_weight(self) -> None:
        steps = 1000
        self.sigmas = self.timesteps.to(dtype=torch.float32)
        x = self.sigmas * float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((x - steps / 2.0) / steps).pow(2))
        y_shifted = y - y.min()
        if y_shifted.sum() <= 0:
            bsmntw_weighing = torch.ones_like(y_shifted)
        else:
            bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != steps:
            # This follows the empirical rescale used by the Wan training-weight table.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            if len(self.timesteps) > 1:
                bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        self.num_inference_steps = int(num_inference_steps)
        base_grid = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=dtype)
        grid = self._shift_progress(base_grid)
        if self.eps > 0:
            grid = grid.clamp(self.eps, 1.0 - self.eps)
        self.timesteps = grid[:-1]
        self.deltas = grid[1:] - grid[:-1]
        self.set_training_weight()

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        indices = torch.randint(len(self.timesteps), (batch_size,), device=device)
        table_timesteps = self.timesteps.to(device=device, dtype=dtype)
        return table_timesteps.index_select(0, indices)

    @staticmethod
    def expand_timesteps(timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps [B], got {tuple(timesteps.shape)}")
        if target.shape[0] != timesteps.shape[0]:
            raise ValueError(
                f"Batch mismatch between target {tuple(target.shape)} and timesteps {tuple(timesteps.shape)}"
            )
        return timesteps.reshape(timesteps.shape[0], *([1] * (target.ndim - 1)))

    def add_noise(
        self,
        target: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t = self.expand_timesteps(timesteps.to(device=target.device, dtype=target.dtype), target)
        noise = noise.to(device=target.device, dtype=target.dtype)
        return (1.0 - t) * target + t * noise

    @staticmethod
    def training_target(target: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise.to(device=target.device, dtype=target.dtype) - target

    def training_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        original_shape = timesteps.shape
        query = timesteps.to(dtype=torch.float32).reshape(-1)
        table_timesteps = self.timesteps.to(device=query.device, dtype=torch.float32)
        table_weights = self.linear_timesteps_weights.to(device=query.device)
        nearest = (query[:, None] - table_timesteps[None, :]).abs().argmin(dim=1)
        weights = table_weights.index_select(0, nearest).reshape(original_shape)
        return weights.to(device=timesteps.device, dtype=timesteps.dtype)

    def sample_training_batch(self, target: torch.Tensor) -> FlowMatchingBatch:
        timesteps = self.sample_timesteps(
            int(target.shape[0]),
            device=target.device,
            dtype=target.dtype,
        )
        noise = torch.randn_like(target) * self.noise_std
        sample = self.add_noise(target, noise, timesteps)
        velocity_target = self.training_target(target, noise)
        return FlowMatchingBatch(
            sample=sample,
            noise=noise,
            timestep=timesteps,
            velocity_target=velocity_target,
        )

    def step(self, velocity: torch.Tensor, timestep: torch.Tensor | float, sample: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "timesteps"):
            raise RuntimeError("Call set_timesteps before inference")
        step_t = torch.as_tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
        step_index = int(torch.argmin((self.timesteps - step_t).abs()).item())
        delta = self.deltas[step_index].to(device=sample.device, dtype=sample.dtype)
        return sample + delta * velocity.to(device=sample.device, dtype=sample.dtype)

    @torch.no_grad()
    def sample(
        self,
        initial_sample: torch.Tensor,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        if num_inference_steps is not None or self.timesteps.device != initial_sample.device:
            self.set_timesteps(num_inference_steps, device=initial_sample.device, dtype=initial_sample.dtype)
        sample = initial_sample
        batch_size = int(sample.shape[0])
        for timestep in self.timesteps.to(device=sample.device, dtype=sample.dtype):
            timestep_batch = timestep.expand(batch_size)
            velocity = model_fn(sample, timestep_batch)
            sample = self.step(velocity, timestep, sample)
        return sample

    def estimate_x1(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t = self.expand_timesteps(timesteps.to(device=sample.device, dtype=sample.dtype), sample)
        return sample - t * velocity.to(device=sample.device, dtype=sample.dtype)
