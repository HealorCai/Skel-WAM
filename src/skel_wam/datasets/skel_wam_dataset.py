from __future__ import annotations


from pathlib import Path
from typing import Sequence

import torch

from skel_wam.datasets.video_dataset import (
    EpisodeVideoDataset,
    EpisodeVideoKeypointDataset,
    EpisodeVideoKeypointQposDataset,
    VideoEpisode,
)


class _SkelWamKeypointMixin:
    """Shared keypoint handling for Skel-WAM.

    Raw skeleton keypoints use -1 as the invalid sentinel. This mixin computes a
    valid mask before statistics normalization, normalizes valid coordinates, and
    writes invalid coordinates to neutral 0 in normalized space. The separate
    valid masks preserve missingness without leaking a sentinel value to models.

    Non-finite coordinates are invalid, and
    boundary-clamped video/history frames retain the validity of the copied frame.
    """

    def _normalize_keypoints_with_valid(self, keypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints = keypoints[..., : self.keypoint_dims].to(torch.float32)
        valid = (keypoints != -1).all(dim=-1)
        valid = valid & torch.isfinite(keypoints).all(dim=-1)
        if self.keypoint_mean is not None:
            mean = self.keypoint_mean[: self.keypoint_dims].to(keypoints)
            std = self.keypoint_std[: self.keypoint_dims].to(keypoints)
            normalized = (keypoints - mean) / std
            keypoints = torch.where(valid.unsqueeze(-1), normalized, keypoints.new_zeros(()))
        else:
            keypoints = torch.where(valid.unsqueeze(-1), keypoints, keypoints.new_zeros(()))
        return keypoints, valid

    def _read_raw_keypoints(self, episode: VideoEpisode, frame_indices: torch.Tensor) -> torch.Tensor:
        try:
            import h5py
            import numpy as np
        except ImportError as exc:
            raise ImportError("SkelWam keypoint loading requires h5py and numpy.") from exc

        keypoint_path = self._keypoint_path_for_episode(episode)
        if not keypoint_path.is_file():
            raise FileNotFoundError(f"Missing keypoint file: {keypoint_path}")
        indices = frame_indices.detach().cpu().numpy().astype("int64")
        with h5py.File(keypoint_path, "r") as f:
            if self.keypoint_dataset_key not in f:
                raise KeyError(f"Dataset {self.keypoint_dataset_key!r} not found in {keypoint_path}")
            dataset = f[self.keypoint_dataset_key]
            keypoints = [dataset[int(index)] for index in indices]
        return torch.from_numpy(np.stack(keypoints, axis=0)).to(torch.float32)

    def _read_history_keypoints(
        self,
        episode: VideoEpisode,
        local_start: int,
        *,
        history_keypoint_offsets: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = (
            self.history_keypoint_offsets
            if history_keypoint_offsets is None
            else tuple(int(value) for value in history_keypoint_offsets)
        )
        raw_indices = torch.tensor(
            [int(local_start) + offset for offset in offsets],
            dtype=torch.long,
        )
        clamped = raw_indices.clamp(0, int(episode.num_frames) - 1)
        is_pad = raw_indices < 0
        history_raw = self._read_raw_keypoints(episode, clamped)
        history, history_valid = self._normalize_keypoints_with_valid(history_raw)
        return history, history_valid, clamped, is_pad

    def _history_offsets_for_sample(
        self,
        sample: dict[str, object],
        episode: VideoEpisode,
    ) -> tuple[int, ...]:
        del sample, episode
        return self.history_keypoint_offsets

    def _add_skel_wam_keypoints(self, sample: dict[str, object]) -> dict[str, object]:
        episode = self.episodes[int(sample["episode_index"])]
        keypoints_raw = self._read_raw_keypoints(episode, sample["frame_indices"])
        keypoints, keypoints_valid = self._normalize_keypoints_with_valid(keypoints_raw)
        keypoints = torch.where(keypoints_valid.unsqueeze(-1), keypoints, keypoints.new_zeros(()))
        history, history_valid, history_indices, history_is_pad = self._read_history_keypoints(
            episode,
            int(sample["local_start_index"]),
            history_keypoint_offsets=self._history_offsets_for_sample(
                sample, episode
            ),
        )
        sample["keypoints"] = keypoints
        sample["keypoints_valid"] = keypoints_valid
        sample["history_keypoints"] = history
        sample["history_keypoints_valid"] = history_valid
        sample["history_frame_indices"] = history_indices
        sample["history_is_pad"] = history_is_pad
        if self.keypoint_mean is not None:
            sample["keypoint_mean"] = self.keypoint_mean
            sample["keypoint_std"] = self.keypoint_std
        return sample


class SkelWamVideoKeypointDataset(_SkelWamKeypointMixin, EpisodeVideoKeypointDataset):
    """Video/keypoint windows with extra downsampled keypoint history."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        history_keypoint_offsets: Sequence[int] = (-16, -12, -8, -4),
        keypoint_dims: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(dataset_root, **kwargs)
        if int(keypoint_dims) not in (2, 3):
            raise ValueError(f"keypoint_dims must be 2 or 3, got {keypoint_dims}")
        self.keypoint_dims = int(keypoint_dims)
        self.history_keypoint_offsets = tuple(int(v) for v in history_keypoint_offsets)
        if not self.history_keypoint_offsets:
            raise ValueError("history_keypoint_offsets must not be empty")
        if any(v >= 0 for v in self.history_keypoint_offsets):
            raise ValueError("history_keypoint_offsets must be negative raw-frame offsets")
        self.history_keypoint_offsets = tuple(sorted(self.history_keypoint_offsets))

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = EpisodeVideoDataset.__getitem__(self, idx)
        return self._add_skel_wam_keypoints(sample)




class SkelWamVideoKeypointQposDataset(_SkelWamKeypointMixin, EpisodeVideoKeypointQposDataset):
    """Robot-data variant with qpos targets and keypoint history."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        history_keypoint_offsets: Sequence[int] = (-16, -12, -8, -4),
        keypoint_dims: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(dataset_root, **kwargs)
        if int(keypoint_dims) not in (2, 3):
            raise ValueError(f"keypoint_dims must be 2 or 3, got {keypoint_dims}")
        self.keypoint_dims = int(keypoint_dims)
        self.history_keypoint_offsets = tuple(int(v) for v in history_keypoint_offsets)
        if not self.history_keypoint_offsets:
            raise ValueError("history_keypoint_offsets must not be empty")
        if any(v >= 0 for v in self.history_keypoint_offsets):
            raise ValueError("history_keypoint_offsets must be negative raw-frame offsets")
        self.history_keypoint_offsets = tuple(sorted(self.history_keypoint_offsets))

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = EpisodeVideoDataset.__getitem__(self, idx)
        sample = self._add_skel_wam_keypoints(sample)
        return self._add_qpos_to_sample(sample)



 
