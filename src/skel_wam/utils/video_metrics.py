from __future__ import annotations

import torch
import torch.nn.functional as F


def video_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1.0e-8,
) -> float:
    """Average frame-wise PSNR for videos shaped [C,T,H,W]."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Video shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError(f"Expected video [C,T,H,W], got {tuple(prediction.shape)}")
    error = (prediction.float() - target.float()).square().mean(dim=(0, 2, 3))
    values = 10.0 * torch.log10((data_range * data_range) / (error + eps))
    return float(values.mean().item())


def video_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Average frame/channel SSIM for videos shaped [C,T,H,W]."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Video shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError(f"Expected video [C,T,H,W], got {tuple(prediction.shape)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    prediction = prediction.float().permute(1, 0, 2, 3).contiguous()
    target = target.float().permute(1, 0, 2, 3).contiguous()
    channels = prediction.shape[1]
    coords = (
        torch.arange(kernel_size, device=prediction.device, dtype=prediction.dtype)
        - (kernel_size - 1) / 2.0
    )
    gaussian = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    kernel = (kernel / kernel.sum()).reshape(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2

    mean_prediction = F.conv2d(
        prediction, kernel, padding=padding, groups=channels
    )
    mean_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_cross = mean_prediction * mean_target
    variance_prediction = (
        F.conv2d(prediction.square(), kernel, padding=padding, groups=channels)
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(target.square(), kernel, padding=padding, groups=channels)
        - mean_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels)
        - mean_cross
    )
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mean_cross + c1) * (2.0 * covariance + c2)
    denominator = (
        (mean_prediction_sq + mean_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return float((numerator / (denominator + 1.0e-12)).mean().item())
