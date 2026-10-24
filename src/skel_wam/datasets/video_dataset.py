from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2


try:
    import decord
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    decord = None


@dataclass(frozen=True)
class VideoEpisode:
    path: str
    task: str
    num_frames: int
    fps: float | None
    start: int
    end: int


def _natural_key(path: Path) -> list[int | str]:
    parts: list[int | str] = []
    text = str(path)
    token = ""
    for ch in text:
        if ch.isdigit():
            token += ch
            continue
        if token:
            parts.append(int(token))
            token = ""
        parts.append(ch)
    if token:
        parts.append(int(token))
    return parts


def _count_video_frames(path: Path) -> tuple[int, float | None]:
    if decord is None:
        raise ImportError(
            "EpisodeVideoDataset requires decord for mp4 loading. "
            "Install it with `pip install decord` in the training environment."
        )
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    num_frames = len(reader)
    if num_frames <= 0:
        raise ValueError(f"Video has no frames: {path}")
    try:
        fps = float(reader.get_avg_fps())
    except Exception:
        fps = None
    return num_frames, fps


class EpisodeVideoDataset(Dataset):
    """Load fixed video windows from a directory of episode-level mp4 files.

    The dataset builds one global frame-index space by concatenating all episode
    videos. ``idx`` is interpreted as the first frame of a window inside its own
    episode; windows never cross episode boundaries. Out-of-range frames are
    clamped to the episode's last valid frame and marked in ``video_is_pad``.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        num_frames: int = 33,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 4,
        task_include: Sequence[str] | str | None = None,
        task_exclude: Sequence[str] | str | None = None,
        image_size: tuple[int, int] | None = (256, 256),
        normalize: bool = True,
        video_extensions: Sequence[str] = (".mp4",),
        metadata_cache_path: str | Path | None = None,
        use_metadata_cache: bool = True,
        decord_num_threads: int = 1,
        resize_size = None,
    ):
        self.resize_size = resize_size
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.num_frames = int(num_frames)
        self.global_sample_stride = int(global_sample_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.task_include = self._normalize_task_filter(task_include)
        self.task_exclude = self._normalize_task_filter(task_exclude)
        if self.num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}")
        if self.global_sample_stride <= 0:
            raise ValueError(
                f"`global_sample_stride` must be positive, got {global_sample_stride}"
            )
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {action_video_freq_ratio}"
            )

        self.image_size = None if image_size is None else tuple(map(int, image_size))
        if self.image_size is not None and len(self.image_size) != 2:
            raise ValueError(f"`image_size` must be (height, width) or None, got {image_size}")
        self.normalize = bool(normalize)
        self.video_extensions = tuple(ext.lower() for ext in video_extensions)
        self.decord_num_threads = max(int(decord_num_threads), 1)

        if metadata_cache_path is None:
            cache_key = hashlib.sha1(str(self.dataset_root).encode("utf-8")).hexdigest()[:16]
            metadata_cache_path = (
                Path.cwd() / ".cache" / "skel_wam_video_index" / f"{cache_key}.json"
            )
        self.metadata_cache_path = Path(metadata_cache_path).expanduser()

        self.episodes = self._filter_and_reindex_episodes(
            self._load_or_build_index(use_metadata_cache=use_metadata_cache)
        )
        if not self.episodes:
            raise FileNotFoundError(
                f"No video files matched task filters under {self.dataset_root}"
            )
        self._starts = [ep.start for ep in self.episodes]
        self.total_frames = self.episodes[-1].end

    @property
    def video_window_size(self) -> int:
        return len(range(0, self.num_frames, self.action_video_freq_ratio))

    def _discover_videos(self) -> list[Path]:
        videos = [
            p
            for p in self.dataset_root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.video_extensions
        ]
        return sorted(videos, key=_natural_key)

    def _task_from_path(self, path: Path) -> str:
        rel_parent = path.parent.relative_to(self.dataset_root)
        return str(rel_parent) if str(rel_parent) != "." else ""


    def _normalize_task_filter(self, tasks: Sequence[str] | str | None) -> tuple[str, ...] | None:
        if tasks is None:
            return None
        if isinstance(tasks, str):
            raw_items = tasks.split(",")
        else:
            raw_items = []
            for task in tasks:
                raw_items.extend(str(task).split(","))
        normalized = tuple(item.strip() for item in raw_items if str(item).strip())
        return normalized or None

    def _filter_and_reindex_episodes(self, episodes: Sequence[VideoEpisode]) -> list[VideoEpisode]:
        if self.task_include:
            allowed = set(self.task_include)
            filtered = [ep for ep in episodes if ep.task in allowed]
        elif self.task_exclude:
            blocked = set(self.task_exclude)
            filtered = [ep for ep in episodes if ep.task not in blocked]
        else:
            filtered = list(episodes)

        cursor = 0
        reindexed: list[VideoEpisode] = []
        for ep in filtered:
            reindexed.append(
                VideoEpisode(
                    path=ep.path,
                    task=ep.task,
                    num_frames=ep.num_frames,
                    fps=ep.fps,
                    start=cursor,
                    end=cursor + ep.num_frames,
                )
            )
            cursor += ep.num_frames
        return reindexed

    def _load_or_build_index(self, *, use_metadata_cache: bool) -> list[VideoEpisode]:
        videos: list[Path] | None = None
        if use_metadata_cache and self.metadata_cache_path.exists():
            with self.metadata_cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if (
                payload.get("dataset_root") == str(self.dataset_root)
                and payload.get("backend") == "decord"
                and payload.get("video_extensions") == list(self.video_extensions)
            ):
                cached_episodes = [VideoEpisode(**item) for item in payload["episodes"]]
                videos = self._discover_videos()
                if [episode.path for episode in cached_episodes] == [
                    str(path) for path in videos
                ]:
                    return cached_episodes

        episodes: list[VideoEpisode] = []
        cursor = 0
        if videos is None:
            videos = self._discover_videos()
        for path in videos:
            num_frames, fps = _count_video_frames(path)
            episode = VideoEpisode(
                path=str(path),
                task=self._task_from_path(path),
                num_frames=num_frames,
                fps=fps,
                start=cursor,
                end=cursor + num_frames,
            )
            episodes.append(episode)
            cursor = episode.end

        try:
            self.metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metadata_cache_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_root": str(self.dataset_root),
                        "video_extensions": list(self.video_extensions),
                        "backend": "decord",
                        "episodes": [asdict(ep) for ep in episodes],
                    },
                    f,
                    ensure_ascii=True,
                    indent=2,
                )
        except OSError:
            # Dataset construction should not fail just because the cache path is read-only.
            pass
        return episodes

    def __len__(self) -> int:
        return self.total_frames

    def _resolve_episode(self, idx: int) -> tuple[int, VideoEpisode, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        episode_idx = bisect.bisect_right(self._starts, idx) - 1
        episode = self.episodes[episode_idx]
        local_start = idx - episode.start
        return episode_idx, episode, local_start

    def _frame_offsets_for_sample(
        self,
        dataset_index: int,
        episode_idx: int,
        episode: VideoEpisode,
        local_start: int,
    ) -> tuple[int, ...]:
        del dataset_index, episode_idx, episode, local_start
        return tuple(
            t * self.global_sample_stride
            for t in range(0, self.num_frames, self.action_video_freq_ratio)
        )

    def _make_frame_indices(
        self,
        local_start: int,
        episode_len: int,
        *,
        frame_offsets: Sequence[int] | None = None,
    ):
        if frame_offsets is None:
            frame_offsets = tuple(
                t * self.global_sample_stride
                for t in range(0, self.num_frames, self.action_video_freq_ratio)
            )
        requested = [local_start + int(offset) for offset in frame_offsets]
        clamped = [min(max(i, 0), episode_len - 1) for i in requested]
        is_pad = [i >= episode_len for i in requested]
        return requested, clamped, is_pad

    def _read_frames(self, video_path: str, frame_indices: Sequence[int]) -> torch.Tensor:
        if decord is None:
            raise ImportError(
                "EpisodeVideoDataset requires decord for mp4 loading. "
                "Install it with `pip install decord` in the training environment."
            )
        reader = decord.VideoReader(
            video_path,
            ctx=decord.cpu(0),
            num_threads=self.decord_num_threads,
        )
        video = torch.from_numpy(reader.get_batch(list(map(int, frame_indices))).asnumpy())

        # --- add cv2.resize ---
        if hasattr(self, 'resize_size') and self.resize_size:
            try:
                target_w, target_h = map(int, self.resize_size.lower().split('x'))
            except ValueError:
                raise ValueError(f"resize_size 格式错误，应为 '宽x高' (如 '1280x704'), 当前为: {self.resize_size}")
            
            # 此时 video 的形状为 [T, H, W, C]，转为 numpy 进行逐帧 resize
            video_np = video.numpy() if isinstance(video, torch.Tensor) else video
            resized_frames = []
            for frame in video_np:
                # cv2.resize 接收的尺寸顺序是 (width, height)
                resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_frames = resized_frame if len(video_np) == 1 else resized_frames + [resized_frame]
                
            # 如果只有一帧，需要保持 T 维度
            if len(video_np) == 1:
                import numpy as np
                video = torch.from_numpy(np.expand_dims(resized_frames, axis=0))
            else:
                import numpy as np
                video = torch.from_numpy(np.array(resized_frames))
            # ---------------------------

        video = video.to(torch.float32)  # [T, H, W, C], RGB
        video = video.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        if self.image_size is not None and tuple(video.shape[-2:]) != self.image_size:
            video = F.interpolate(
                video,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.normalize:
            video = video / 127.5 - 1.0
        else:
            video = video / 255.0
        return video.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]

    def __getitem__(self, idx: int) -> dict[str, object]:
        dataset_index = int(idx)
        episode_idx, episode, local_start = self._resolve_episode(dataset_index)
        frame_offsets = self._frame_offsets_for_sample(
            dataset_index,
            episode_idx,
            episode,
            local_start,
        )
        requested, clamped, is_pad = self._make_frame_indices(
            local_start,
            episode.num_frames,
            frame_offsets=frame_offsets,
        )
        video = self._read_frames(episode.path, clamped)

        return {
            "video": video,
            "video_is_pad": torch.tensor(is_pad, dtype=torch.bool),
            "global_index": torch.tensor(idx, dtype=torch.long),
            "episode_index": torch.tensor(episode_idx, dtype=torch.long),
            "local_start_index": torch.tensor(local_start, dtype=torch.long),
            "requested_frame_indices": torch.tensor(requested, dtype=torch.long),
            "frame_indices": torch.tensor(clamped, dtype=torch.long),
            "episode_path": episode.path,
            "task": episode.task,
            "fps": torch.tensor(-1.0 if episode.fps is None else episode.fps, dtype=torch.float32),
        }


class EpisodeVideoKeypointDataset(EpisodeVideoDataset):
    """Episode video windows with aligned per-frame hand keypoints."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        keypoint_root: str | Path,
        keypoint_dataset_key: str = "keypoints",
        keypoint_stats_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(dataset_root, **kwargs)
        self.keypoint_root = Path(keypoint_root).expanduser().resolve()
        if not self.keypoint_root.is_dir():
            raise FileNotFoundError(f"Keypoint root does not exist: {self.keypoint_root}")
        self.keypoint_dataset_key = keypoint_dataset_key
        self.keypoint_stats_path = None
        self.keypoint_mean = None
        self.keypoint_std = None
        if keypoint_stats_path is not None:
            self.keypoint_stats_path = Path(keypoint_stats_path).expanduser().resolve()
            if not self.keypoint_stats_path.is_file():
                raise FileNotFoundError(f"Keypoint stats file does not exist: {self.keypoint_stats_path}")
            stats = json.loads(self.keypoint_stats_path.read_text(encoding="utf-8"))
            mean = torch.tensor(stats.get("mean"), dtype=torch.float32)
            std = torch.tensor(stats.get("std"), dtype=torch.float32)
            if mean.shape != (3,) or std.shape != (3,):
                raise ValueError("Keypoint stats must contain length-3 'mean' and 'std' arrays")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError("Keypoint stats 'mean' and 'std' must be finite")
            if not (std > 0).all():
                raise ValueError("Keypoint stats 'std' values must be positive")
            self.keypoint_mean = mean
            self.keypoint_std = std

    def _keypoint_path_for_episode(self, episode: VideoEpisode) -> Path:
        video_path = Path(episode.path)
        rel_parent = video_path.parent.relative_to(self.dataset_root)
        return self.keypoint_root / rel_parent / f"{video_path.stem}.hdf5"

    def _read_keypoints(self, episode: VideoEpisode, frame_indices: torch.Tensor) -> torch.Tensor:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "EpisodeVideoKeypointDataset requires h5py for HDF5 keypoint loading."
            ) from exc

        keypoint_path = self._keypoint_path_for_episode(episode)
        if not keypoint_path.is_file():
            raise FileNotFoundError(f"Missing keypoint file: {keypoint_path}")

        indices = frame_indices.detach().cpu().numpy().astype("int64")
        with h5py.File(keypoint_path, "r") as f:
            if self.keypoint_dataset_key not in f:
                raise KeyError(
                    f"Dataset {self.keypoint_dataset_key!r} not found in {keypoint_path}"
                )
            dataset = f[self.keypoint_dataset_key]
            keypoints = [dataset[int(index)] for index in indices]
        import numpy as np
        keypoints = torch.from_numpy(np.stack(keypoints, axis=0)).to(torch.float32)
        if self.keypoint_mean is not None:
            keypoints = (keypoints - self.keypoint_mean) / self.keypoint_std
        return keypoints

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = super().__getitem__(idx)
        episode_idx = int(sample["episode_index"])
        episode = self.episodes[episode_idx]
        sample["keypoints"] = self._read_keypoints(episode, sample["frame_indices"])
        if self.keypoint_mean is not None:
            sample["keypoint_mean"] = self.keypoint_mean
            sample["keypoint_std"] = self.keypoint_std
        return sample


class _QposDatasetMixin:
    def _init_qpos(
        self,
        *,
        qpos_root: str | Path,
        qpos_dataset_key: str = "observations/qpos",
        qpos_target_frames: int = 49,
        qpos_stats_path: str | Path | None = None,
        qpos_exclude_indices: Sequence[int] | None = None,
    ) -> None:
        self.qpos_root = Path(qpos_root).expanduser().resolve()
        if not self.qpos_root.is_dir():
            raise FileNotFoundError(f"Qpos root does not exist: {self.qpos_root}")
        self.qpos_dataset_key = qpos_dataset_key
        self.qpos_target_frames = int(qpos_target_frames)
        if self.qpos_target_frames <= 0:
            raise ValueError(f"`qpos_target_frames` must be positive, got {qpos_target_frames}")
        self._qpos_cache = {}
        first_qpos = self._load_qpos(self._qpos_path_for_episode(self.episodes[0]))
        self.raw_qpos_dim = int(first_qpos.shape[-1])
        excluded = tuple(int(index) for index in (qpos_exclude_indices or ()))
        if len(set(excluded)) != len(excluded):
            raise ValueError(f"qpos_exclude_indices contains duplicates: {excluded}")
        invalid = sorted(
            index for index in excluded if index < 0 or index >= self.raw_qpos_dim
        )
        if invalid:
            raise ValueError(
                "qpos_exclude_indices must be in "
                f"[0, {self.raw_qpos_dim - 1}], got {invalid}"
            )
        excluded_set = set(excluded)
        self.qpos_exclude_indices = tuple(sorted(excluded_set))
        self.qpos_indices = tuple(
            index
            for index in range(self.raw_qpos_dim)
            if index not in excluded_set
        )
        if not self.qpos_indices:
            raise ValueError("qpos_exclude_indices removes every qpos dimension")
        self.qpos_dim = len(self.qpos_indices)
        self._qpos_index_tensor = torch.tensor(self.qpos_indices, dtype=torch.long)
        self.qpos_stats_path = None
        self.qpos_mean = None
        self.qpos_std = None
        if qpos_stats_path is not None:
            self.qpos_stats_path = Path(qpos_stats_path).expanduser().resolve()
            if not self.qpos_stats_path.is_file():
                raise FileNotFoundError(f"Qpos stats file does not exist: {self.qpos_stats_path}")
            stats = json.loads(self.qpos_stats_path.read_text(encoding="utf-8"))
            mean = torch.tensor(stats.get("mean"), dtype=torch.float32)
            std = torch.tensor(stats.get("std"), dtype=torch.float32)
            if mean.shape != (self.raw_qpos_dim,) or std.shape != (self.raw_qpos_dim,):
                raise ValueError(
                    "Qpos stats must describe the raw qpos tensor before dimension "
                    f"selection: expected length {self.raw_qpos_dim}, got "
                    f"mean={tuple(mean.shape)} std={tuple(std.shape)}"
                )
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError("Qpos stats 'mean' and 'std' must be finite")
            if not (std > 0).all():
                raise ValueError("Qpos stats 'std' values must be positive")
            self.qpos_mean = mean.index_select(0, self._qpos_index_tensor)
            self.qpos_std = std.index_select(0, self._qpos_index_tensor)

    def _qpos_path_for_episode(self, episode: VideoEpisode) -> str:
        stem = Path(episode.path).stem
        npy_path = Path(self.qpos_root) / episode.task / f"{stem}.npy"
        if npy_path.is_file():
            return str(npy_path)
        return str(Path(self.qpos_root) / episode.task / f"{stem}.hdf5")

    def _load_qpos(self, qpos_path: str) -> torch.Tensor:
        if qpos_path in self._qpos_cache:
            return self._qpos_cache[qpos_path]

        qpos_file = Path(qpos_path)
        if not qpos_file.is_file():
            raise FileNotFoundError(f"Missing qpos file: {qpos_path}")
        if qpos_file.suffix == ".npy":
            import numpy as np

            qpos = np.load(qpos_file)
        else:
            import h5py

            with h5py.File(qpos_file, "r") as f:
                if self.qpos_dataset_key not in f:
                    raise KeyError(f"Dataset {self.qpos_dataset_key!r} not found in {qpos_path}")
                qpos = f[self.qpos_dataset_key][:]

        import numpy as np
        qpos = torch.from_numpy(np.asarray(qpos, dtype=np.float32))
        self._qpos_cache[qpos_path] = qpos
        return qpos

    def _read_qpos_window(self, episode: VideoEpisode, local_start: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        qpos_path = self._qpos_path_for_episode(episode)
        qpos = self._load_qpos(qpos_path)
        qpos_end = int(local_start) + self.qpos_target_frames
        qpos_window = qpos[int(local_start):qpos_end]
        valid_len = int(qpos_window.shape[0])
        if valid_len <= 0:
            raise ValueError(f"Empty qpos window for {qpos_path} at idx={local_start}")
        if valid_len < self.qpos_target_frames:
            pad_count = self.qpos_target_frames - valid_len
            pad_value = qpos_window[-1:].repeat(pad_count, 1)
            qpos_window = torch.cat([qpos_window, pad_value], dim=0)
        qpos_window = qpos_window.index_select(
            -1, self._qpos_index_tensor.to(device=qpos_window.device)
        )
        qpos_is_pad = torch.zeros(self.qpos_target_frames, dtype=torch.bool)
        if valid_len < self.qpos_target_frames:
            qpos_is_pad[valid_len:] = True
        if self.qpos_mean is not None:
            qpos_window = (qpos_window - self.qpos_mean) / self.qpos_std
        return qpos_window, qpos_is_pad, qpos_path

    def _add_qpos_to_sample(self, sample: dict[str, object]) -> dict[str, object]:
        episode_idx = int(sample["episode_index"])
        episode = self.episodes[episode_idx]
        qpos, qpos_is_pad, qpos_path = self._read_qpos_window(
            episode, int(sample["local_start_index"])
        )
        sample["qpos"] = qpos
        sample["qpos_is_pad"] = qpos_is_pad
        sample["qpos_path"] = qpos_path
        if self.qpos_mean is not None:
            sample["qpos_mean"] = self.qpos_mean
            sample["qpos_std"] = self.qpos_std
        return sample


class EpisodeVideoQposDataset(_QposDatasetMixin, EpisodeVideoDataset):
    """Episode video windows with aligned qpos windows."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        qpos_root: str | Path,
        qpos_dataset_key: str = "observations/qpos",
        qpos_target_frames: int = 49,
        qpos_stats_path: str | Path | None = None,
        qpos_exclude_indices: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__(dataset_root, **kwargs)
        self._init_qpos(
            qpos_root=qpos_root,
            qpos_dataset_key=qpos_dataset_key,
            qpos_target_frames=qpos_target_frames,
            qpos_stats_path=qpos_stats_path,
            qpos_exclude_indices=qpos_exclude_indices,
        )

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = super().__getitem__(idx)
        return self._add_qpos_to_sample(sample)


class EpisodeVideoKeypointQposDataset(_QposDatasetMixin, EpisodeVideoKeypointDataset):
    """Episode video/keypoint windows with aligned qpos windows."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        qpos_root: str | Path,
        qpos_dataset_key: str = "observations/qpos",
        qpos_target_frames: int = 49,
        qpos_stats_path: str | Path | None = None,
        qpos_exclude_indices: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__(dataset_root, **kwargs)
        self._init_qpos(
            qpos_root=qpos_root,
            qpos_dataset_key=qpos_dataset_key,
            qpos_target_frames=qpos_target_frames,
            qpos_stats_path=qpos_stats_path,
            qpos_exclude_indices=qpos_exclude_indices,
        )

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = super().__getitem__(idx)
        return self._add_qpos_to_sample(sample)





def parse_task_filter(tasks: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if tasks is None:
        return None
    if isinstance(tasks, str):
        raw_items = tasks.split(",")
    else:
        raw_items = []
        for task in tasks:
            raw_items.extend(str(task).split(","))
    normalized = tuple(item.strip() for item in raw_items if item.strip())
    return normalized or None


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def build_video_dataloader(
    dataset_root: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    dataset = EpisodeVideoDataset(dataset_root, **dataset_kwargs)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Inspect EpisodeVideoDataset samples.")
    parser.add_argument("dataset_root")
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--global-sample-stride", type=int, default=1)
    parser.add_argument("--action-video-freq-ratio", type=int, default=4)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--decord-num-threads", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset = EpisodeVideoDataset(
        args.dataset_root,
        num_frames=args.num_frames,
        global_sample_stride=args.global_sample_stride,
        action_video_freq_ratio=args.action_video_freq_ratio,
        decord_num_threads=args.decord_num_threads,
        use_metadata_cache=not args.no_cache,
    )
    sample = dataset[args.idx]
    print(f"episodes: {len(dataset.episodes)}")
    print(f"total_frames: {len(dataset)}")
    print(f"video_window_size: {dataset.video_window_size}")
    print(f"sample.video: {tuple(sample['video'].shape)}")
    print(f"sample.video_is_pad: {sample['video_is_pad'].tolist()}")
    print(f"sample.task: {sample['task']}")
    print(f"sample.episode_path: {sample['episode_path']}")
    print(f"sample.requested_frame_indices: {sample['requested_frame_indices'].tolist()}")
    print(f"sample.frame_indices: {sample['frame_indices'].tolist()}")


if __name__ == "__main__":
    main()
