from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict, cast

import torch


ValidationMode = Literal["none", "first", "always"]


class GeneralVideoSample(TypedDict):
    """Fields shared by keypoint and strict video-only training sources."""

    video: torch.Tensor
    language_prompts: str


class GeneralSourceSample(GeneralVideoSample, total=False):
    """Canonical sample; keypoint fields are present together or all absent."""

    keypoints: torch.Tensor
    keypoints_valid: torch.Tensor
    history_keypoints: torch.Tensor
    history_keypoints_valid: torch.Tensor
    keypoint_mean: torch.Tensor
    keypoint_std: torch.Tensor


class GeneralMixtureSample(GeneralSourceSample):
    """Canonical sample after a source has been selected by a mixture sampler."""

    source_id: torch.Tensor
    source_index: torch.Tensor
    source_name: str


GENERAL_SOURCE_REQUIRED_KEYS = frozenset(GeneralSourceSample.__required_keys__)
GENERAL_KEYPOINT_SAMPLE_KEYS = frozenset(GeneralSourceSample.__optional_keys__)
GENERAL_SOURCE_SAMPLE_KEYS = (
    GENERAL_SOURCE_REQUIRED_KEYS | GENERAL_KEYPOINT_SAMPLE_KEYS
)
GENERAL_MIXTURE_REQUIRED_KEYS = frozenset(GeneralMixtureSample.__required_keys__)
GENERAL_MIXTURE_SAMPLE_KEYS = (
    GENERAL_MIXTURE_REQUIRED_KEYS | GENERAL_KEYPOINT_SAMPLE_KEYS
)


def _require_tensor(
    sample: Mapping[str, object],
    key: str,
    *,
    source_name: str,
) -> torch.Tensor:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Source {source_name!r} field {key!r} must be a torch.Tensor, "
            f"got {type(value).__name__}"
        )
    return value


def validate_general_source_sample(
    sample: Mapping[str, object],
    *,
    source_name: str,
) -> None:
    """Validate the exact schema and shape contract consumed by general training."""

    actual_keys = set(sample)
    keypoint_keys = actual_keys & GENERAL_KEYPOINT_SAMPLE_KEYS
    valid_schema = (
        GENERAL_SOURCE_REQUIRED_KEYS
        if not keypoint_keys
        else GENERAL_SOURCE_SAMPLE_KEYS
    )
    if actual_keys != valid_schema:
        missing = sorted(valid_schema - actual_keys)
        extra = sorted(actual_keys - valid_schema)
        raise ValueError(
            f"Source {source_name!r} returned an invalid general sample schema: "
            f"missing={missing}, extra={extra}"
        )

    if not keypoint_keys:
        video = _require_tensor(sample, "video", source_name=source_name)
        if video.ndim != 4:
            raise ValueError(
                f"Source {source_name!r} video must have shape [C,T,H,W], "
                f"got {tuple(video.shape)}"
            )
        if (int(video.shape[1]) - 1) % 4 != 0:
            raise ValueError(
                f"Source {source_name!r} video frames must satisfy "
                f"(frames - 1) % 4 == 0, got {video.shape[1]}"
            )
        if not torch.is_floating_point(video):
            raise TypeError(f"Source {source_name!r} video must be floating point")
        prompt = sample["language_prompts"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Source {source_name!r} language_prompts must be a non-empty str"
            )
        return

    video = _require_tensor(sample, "video", source_name=source_name)
    keypoints = _require_tensor(sample, "keypoints", source_name=source_name)
    keypoints_valid = _require_tensor(sample, "keypoints_valid", source_name=source_name)
    history = _require_tensor(sample, "history_keypoints", source_name=source_name)
    history_valid = _require_tensor(
        sample, "history_keypoints_valid", source_name=source_name
    )
    mean = _require_tensor(sample, "keypoint_mean", source_name=source_name)
    std = _require_tensor(sample, "keypoint_std", source_name=source_name)

    if video.ndim != 4:
        raise ValueError(
            f"Source {source_name!r} video must have shape [C,T,H,W], "
            f"got {tuple(video.shape)}"
        )
    video_frames = int(video.shape[1])
    if (video_frames - 1) % 4 != 0:
        raise ValueError(
            f"Source {source_name!r} video frames must satisfy "
            f"(frames - 1) % 4 == 0, got {video_frames}"
        )
    if not torch.is_floating_point(video):
        raise TypeError(f"Source {source_name!r} video must be floating point")

    if keypoints.ndim < 3:
        raise ValueError(
            f"Source {source_name!r} keypoints must have at least 3 dimensions, "
            f"got {tuple(keypoints.shape)}"
        )
    if int(keypoints.shape[0]) != video_frames:
        raise ValueError(
            f"Source {source_name!r} video/keypoint frame counts differ: "
            f"{video_frames} vs {keypoints.shape[0]}"
        )
    if keypoints.shape[-1] not in (2, 3):
        raise ValueError(
            f"Source {source_name!r} keypoint coordinate dimension must be 2 or 3, "
            f"got {keypoints.shape[-1]}"
        )
    if keypoints_valid.shape != keypoints.shape[:-1]:
        raise ValueError(
            f"Source {source_name!r} keypoints_valid shape "
            f"{tuple(keypoints_valid.shape)} does not match keypoints "
            f"{tuple(keypoints.shape)}"
        )
    if keypoints_valid.dtype != torch.bool:
        raise TypeError(f"Source {source_name!r} keypoints_valid must be bool")

    if history.ndim != keypoints.ndim:
        raise ValueError(
            f"Source {source_name!r} history/keypoint ranks differ: "
            f"{history.ndim} vs {keypoints.ndim}"
        )
    if history.shape[1:] != keypoints.shape[1:]:
        raise ValueError(
            f"Source {source_name!r} history shape {tuple(history.shape)} is "
            f"incompatible with keypoints {tuple(keypoints.shape)}"
        )
    if history_valid.shape != history.shape[:-1]:
        raise ValueError(
            f"Source {source_name!r} history_keypoints_valid shape "
            f"{tuple(history_valid.shape)} does not match history "
            f"{tuple(history.shape)}"
        )
    if history_valid.dtype != torch.bool:
        raise TypeError(
            f"Source {source_name!r} history_keypoints_valid must be bool"
        )

    coordinate_dims = keypoints.shape[-1]
    if mean.shape != (coordinate_dims,) or std.shape != (coordinate_dims,):
        raise ValueError(
            f"Source {source_name!r} keypoint stats must have shape "
            f"({coordinate_dims},), got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Source {source_name!r} keypoint stats must be finite")
    if not (std > 0).all():
        raise ValueError(f"Source {source_name!r} keypoint std must be positive")

    prompt = sample["language_prompts"]
    if not isinstance(prompt, str):
        raise TypeError(
            f"Source {source_name!r} language_prompts must be str, "
            f"got {type(prompt).__name__}"
        )
    if not prompt.strip():
        raise ValueError(f"Source {source_name!r} language_prompts must not be empty")


def validate_general_mixture_sample(
    sample: Mapping[str, object],
    *,
    source_name: str,
) -> None:
    actual_keys = set(sample)
    keypoint_keys = actual_keys & GENERAL_KEYPOINT_SAMPLE_KEYS
    valid_schema = (
        GENERAL_MIXTURE_REQUIRED_KEYS
        if not keypoint_keys
        else GENERAL_MIXTURE_SAMPLE_KEYS
    )
    if actual_keys != valid_schema:
        missing = sorted(valid_schema - actual_keys)
        extra = sorted(actual_keys - valid_schema)
        raise ValueError(
            f"Source {source_name!r} returned an invalid mixture sample schema: "
            f"missing={missing}, extra={extra}"
        )

    validate_general_source_sample(
        {
            key: sample[key]
            for key in GENERAL_SOURCE_SAMPLE_KEYS
            if key in sample
        },
        source_name=source_name,
    )
    source_id = _require_tensor(sample, "source_id", source_name=source_name)
    source_index = _require_tensor(sample, "source_index", source_name=source_name)
    if source_id.ndim != 0 or source_id.dtype != torch.long:
        raise TypeError(
            f"Source {source_name!r} source_id must be a scalar torch.long tensor"
        )
    if source_index.ndim != 0 or source_index.dtype != torch.long:
        raise TypeError(
            f"Source {source_name!r} source_index must be a scalar torch.long tensor"
        )
    if sample["source_name"] != source_name:
        raise ValueError(
            f"source_name mismatch: expected {source_name!r}, "
            f"got {sample['source_name']!r}"
        )


def as_general_source_sample(
    sample: Mapping[str, object],
    *,
    source_name: str,
    validation: ValidationMode,
    already_validated: bool,
) -> tuple[GeneralSourceSample, bool]:
    """Copy only canonical keys and optionally validate the result."""

    if validation not in ("none", "first", "always"):
        raise ValueError(f"Unknown sample validation mode: {validation!r}")
    canonical = cast(
        GeneralSourceSample,
        {
            key: sample[key]
            for key in GENERAL_SOURCE_SAMPLE_KEYS
            if key in sample
        },
    )
    should_validate = validation == "always" or (
        validation == "first" and not already_validated
    )
    if should_validate:
        validate_general_source_sample(canonical, source_name=source_name)
    return canonical, already_validated or should_validate
