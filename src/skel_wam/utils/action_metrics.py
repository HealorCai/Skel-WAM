"""Physical component errors for real-robot actions (right 21-D, left 21-D)."""
from __future__ import annotations

import math
import torch


def _rotation_from_columns(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gram-Schmidt decode two concatenated matrix columns; flag degeneracy."""
    first, second = values[..., :3], values[..., 3:6]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    b1 = first / first_norm.clamp_min(1e-8)
    orthogonal = second - (b1 * second).sum(dim=-1, keepdim=True) * b1
    second_norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    b2 = orthogonal / second_norm.clamp_min(1e-8)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    valid = (first_norm[..., 0] > 1e-8) & (second_norm[..., 0] > 1e-8)
    return torch.stack((b1, b2, b3), dim=-1), valid


def real_robot_action_metrics(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
) -> dict[str, float | int | None]:
    """Report errors on selected frames, pooling both hands.

    Inputs are denormalized [T,42]: each hand has rotation columns (6),
    translation (3, original dataset units), and joint angles (12, radians).
    Degenerate rotations are counted and excluded only from rotation metrics.
    """
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[-1] != 42:
        raise ValueError('Real-robot metrics require matching [T,42] actions.')
    if valid.shape != target.shape[:1]:
        raise ValueError('Action validity must have shape [T].')
    selected = valid.to(device=target.device, dtype=torch.bool)
    gt = target.detach()[selected].to(dtype=torch.float64)
    pred = prediction.detach().to(device=target.device)[selected].to(dtype=torch.float64)
    if not gt.numel():
        raise ValueError('No valid action frames to evaluate.')
    if not torch.isfinite(gt).all() or not torch.isfinite(pred).all():
        raise ValueError('Non-finite action values in evaluated frames.')
    pred, gt = pred.reshape(-1, 2, 21), gt.reshape(-1, 2, 21)
    pred_rotation, pred_ok = _rotation_from_columns(pred[..., :6])
    gt_rotation, gt_ok = _rotation_from_columns(gt[..., :6])
    rotation_valid = pred_ok & gt_ok
    relative = pred_rotation.transpose(-1, -2) @ gt_rotation
    # atan2 is stable at both zero and pi, unlike an unclamped acos(trace).
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angles = torch.atan2(sine, cosine)[rotation_valid]
    translation_error = pred[..., 6:9] - gt[..., 6:9]
    joint_error = pred[..., 9:21] - gt[..., 9:21]
    result = {
        'action_valid_frames': int(gt.shape[0]),
        'action_rotation_valid_count': int(rotation_valid.sum()),
        'action_rotation_invalid_prediction_count': int((~pred_ok).sum()),
        'action_rotation_invalid_target_count': int((~gt_ok).sum()),
        'action_translation_mae': translation_error.abs().mean().item(),
        'action_translation_rmse': translation_error.square().mean().sqrt().item(),
        'action_translation_l2_mean': torch.linalg.vector_norm(translation_error, dim=-1).mean().item(),
    }
    for statistic, value in (
        ('mean', angles.mean().item() if angles.numel() else None),
        ('rmse', angles.square().mean().sqrt().item() if angles.numel() else None),
    ):
        result[f'action_rotation_{statistic}(rad)'] = value
        result[f'action_rotation_{statistic}(deg)'] = None if value is None else math.degrees(value)
    for statistic, value in (
        ('mae', joint_error.abs().mean().item()),
        ('rmse', joint_error.square().mean().sqrt().item()),
    ):
        result[f'action_hand_joint_{statistic}(rad)'] = value
        result[f'action_hand_joint_{statistic}(deg)'] = math.degrees(value)
    return result
