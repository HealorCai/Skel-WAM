"""
Compute x/y/z mean and standard deviation for skeleton keypoints.

Run from the repository root:
    python3 utils/egovla_utils/compute_keypoint_stats.py

The defaults scan the seven EgoVLA training tasks under
``data/EgoVLA_SIM/skeleton_keypoints`` and write
``utils/egovla_utils/keypoint_stats_egovla_7tasks.json``.

Equivalent explicit command:
    python3 utils/egovla_utils/compute_keypoint_stats.py \
        --root data/EgoVLA_SIM/skeleton_keypoints \
        --task-include Close-Drawer Flip-Mug Open-Drawer Open-Laptop \
            Pour-Balls Push-Box Stack-Can \
        --output utils/egovla_utils/keypoint_stats_egovla_7tasks.json

Validity follows the fixed Skel-WAM dataset rule: a
point is valid only when all coordinates are finite and none equals ``-1``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data/EgoVLA_SIM/skeleton_keypoints"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "keypoint_stats_egovla_7tasks.json"
DEFAULT_TASKS = (
    "Close-Drawer",
    "Flip-Mug",
    "Open-Drawer",
    "Open-Laptop",
    "Pour-Balls",
    "Push-Box",
    "Stack-Can",
)
AXES = ("x", "y", "z")


def parse_task_filter(
    tasks: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
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


def task_from_path(root: Path, path: Path) -> str:
    rel_parent = path.parent.relative_to(root)
    return str(rel_parent) if str(rel_parent) != "." else ""


def discover_paths(
    root: Path,
    *,
    task_include: tuple[str, ...] | None,
    task_exclude: tuple[str, ...] | None,
) -> list[Path]:
    paths = sorted(root.rglob("*.hdf5"))
    if task_include:
        allowed = set(task_include)
        paths = [path for path in paths if task_from_path(root, path) in allowed]
    elif task_exclude:
        blocked = set(task_exclude)
        paths = [path for path in paths if task_from_path(root, path) not in blocked]
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for skeleton keypoints."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--task-include",
        nargs="*",
        default=list(DEFAULT_TASKS),
        help=(
            "Only include these task directories. Comma-separated entries are "
            "also accepted. Defaults to the seven EgoVLA training tasks."
        ),
    )
    parser.add_argument(
        "--task-exclude",
        nargs="*",
        default=None,
        help="Exclude these task directories. Ignored when --task-include is set.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=4096,
        help="Number of frames to read from each HDF5 file at a time.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many files. Set to 0 to disable.",
    )
    return parser.parse_args()


def merge_batch(
    count: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
    min_value: np.ndarray,
    max_value: np.ndarray,
    min_location: list,
    max_location: list,
    values: np.ndarray,
    file_path: Path,
    frame_offset: int,
) -> None:
    """Merge one batch into per-axis Welford accumulators and track min/max locations."""
    original_shape = values.shape  # e.g. [chunk_T, 2, 21, 3]
    flat = values.reshape(-1, 3).astype(np.float64, copy=False)

    # Match the fixed _SkelWamKeypointMixin validity rule:
    # every coordinate must be finite and every coordinate must differ from -1.
    valid = np.isfinite(flat).all(axis=-1)
    valid &= (flat != -1.0).all(axis=-1)

    valid_indices = np.nonzero(valid)[0]
    flat_valid = flat[valid]

    if flat_valid.shape[0] == 0:
        return

    # update min / max and record location
    for axis_idx, axis_name in enumerate(AXES):
        local_min_idx = np.argmin(flat_valid[:, axis_idx])
        local_max_idx = np.argmax(flat_valid[:, axis_idx])

        candidate_min = flat_valid[local_min_idx, axis_idx]
        candidate_max = flat_valid[local_max_idx, axis_idx]

        raw_min_flat_idx = valid_indices[local_min_idx]
        raw_max_flat_idx = valid_indices[local_max_idx]

        min_multi_index = np.unravel_index(raw_min_flat_idx, original_shape[:-1])
        max_multi_index = np.unravel_index(raw_max_flat_idx, original_shape[:-1])

        # min_multi_index[0] 是 batch 内部帧号
        # 加上 frame_offset 后变成 hdf5 文件内的全局帧号
        min_frame = int(frame_offset + min_multi_index[0])
        max_frame = int(frame_offset + max_multi_index[0])

        # 剩下的维度表示这一帧里的 keypoint 位置
        # 如果 shape 是 [T, 2, 21, 3]，这里就是 [hand_idx, joint_idx]
        # 如果 shape 是 [T, 42, 3]，这里就是 [point_idx]
        min_point_index = [int(x) for x in min_multi_index[1:]]
        max_point_index = [int(x) for x in max_multi_index[1:]]

        if candidate_min < min_value[axis_idx]:
            min_value[axis_idx] = candidate_min
            min_location[axis_idx] = {
                "axis": axis_name,
                "value": float(candidate_min),
                "file": Path(os.path.relpath(file_path, PROJECT_ROOT)).as_posix(),
                "frame": min_frame,
                "point_index": min_point_index,
                "xyz": flat[raw_min_flat_idx].tolist(),
            }

        if candidate_max > max_value[axis_idx]:
            max_value[axis_idx] = candidate_max
            max_location[axis_idx] = {
                "axis": axis_name,
                "value": float(candidate_max),
                "file": Path(os.path.relpath(file_path, PROJECT_ROOT)).as_posix(),
                "frame": max_frame,
                "point_index": max_point_index,
                "xyz": flat[raw_max_flat_idx].tolist(),
            }

    # Welford statistics
    batch_count = np.full(3, flat_valid.shape[0], dtype=np.int64)
    batch_mean = flat_valid.mean(axis=0)
    batch_m2 = np.square(flat_valid - batch_mean).sum(axis=0)

    total_count = count + batch_count
    delta = batch_mean - mean

    mean += delta * batch_count / total_count
    m2 += batch_m2 + np.square(delta) * count * batch_count / total_count

    count[:] = total_count

def compute_stats(
    root: Path,
    chunk_frames: int,
    progress_every: int,
    *,
    task_include: tuple[str, ...] | None,
    task_exclude: tuple[str, ...] | None,
) -> dict[str, object]:
    paths = discover_paths(
        root,
        task_include=task_include,
        task_exclude=task_exclude,
    )
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files found under: {root}")

    count = np.zeros(3, dtype=np.int64)
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)

    min_value = np.full(3, np.inf, dtype=np.float64)
    max_value = np.full(3, -np.inf, dtype=np.float64)

    min_location = [None, None, None]
    max_location = [None, None, None]

    for file_index, path in enumerate(paths, start=1):
        with h5py.File(path, "r") as h5_file:
            if "keypoints" not in h5_file:
                raise KeyError(f"Missing 'keypoints' dataset: {path}")
            keypoints = h5_file["keypoints"]
            if keypoints.ndim < 2 or keypoints.shape[-1] != 3:
                raise ValueError(
                    f"Expected keypoints shape (..., 3), got {keypoints.shape}: {path}"
                )
            for start in range(0, keypoints.shape[0], chunk_frames):
                merge_batch(
                    count=count,
                    mean=mean,
                    m2=m2,
                    min_value=min_value,
                    max_value=max_value,
                    min_location=min_location,
                    max_location=max_location,
                    values=keypoints[start : start + chunk_frames],
                    file_path=path,
                    frame_offset=start,
                )

        if progress_every and file_index % progress_every == 0:
            print(f"processed_files: {file_index}/{len(paths)}")

    if np.any(count == 0):
        raise ValueError(f"No finite values found for axes: {count.tolist()}")

    std = np.sqrt(m2 / count)

    return {
        "root": Path(os.path.relpath(root, PROJECT_ROOT)).as_posix(),
        "num_files": len(paths),
        "task_include": list(task_include) if task_include else None,
        "task_exclude": list(task_exclude) if task_exclude else None,
        "validity_logic": "all coordinates finite and every coordinate != -1",
        "axes": list(AXES),
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": min_value.tolist(),
        "max": max_value.tolist(),
        "min_location": min_location,
        "max_location": max_location,
    }


def main() -> None:
    args = parse_args()
    # Preserve repository-relative paths through dataset symlinks.
    root = Path(os.path.abspath(Path(args.root).expanduser()))
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    task_include = parse_task_filter(args.task_include)
    task_exclude = parse_task_filter(args.task_exclude)
    stats = compute_stats(
        root,
        args.chunk_frames,
        args.progress_every,
        task_include=task_include,
        task_exclude=task_exclude,
    )
    text = json.dumps(stats, indent=2)
    print(text)

    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{text}\n", encoding="utf-8")
        print(f"saved_to: {output}")


if __name__ == "__main__":
    main()
