#!/usr/bin/env python3
"""Compute real-data keypoint normalization statistics.

Two modes are supported:

* ``robot``: use ``data/real_robot/skeleton_keypoints`` only.
* ``human_robot``: pool valid points from both real robot and real human data.

The mixed mode performs one sample-count-weighted Welford accumulation over all
valid points; it does not average the two domains' final means/stds. A point is
valid exactly when all three coordinates are finite and none equals ``-1``.

Examples:

    cd /path/to/Skel-WAM
    python utils/real_utils/compute_keypoint_stats.py --mode robot --task-include put_something_into_drawer
    python utils/real_utils/compute_keypoint_stats.py --mode human_robot
    python utils/real_utils/compute_keypoint_stats.py \
      --mode human_robot --task-include stack_cups

Default outputs are written beside this script as
``keypoint_stats_real_robot.json`` and
``keypoint_stats_real_human_robot.json`` respectively.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_ROOT = PROJECT_ROOT / "data/real_robot/skeleton_keypoints"
DEFAULT_HUMAN_ROOT = PROJECT_ROOT / "data/real_human/skeleton_keypoints"
SCRIPT_DIR = Path(__file__).resolve().parent
AXES = ("x", "y", "z")


def _require_dependencies():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Statistics require numpy and h5py") from exc
    return h5py, np


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
    normalized = tuple(item.strip() for item in raw_items if item.strip())
    return normalized or None


def task_from_path(root: Path, path: Path) -> str:
    relative = path.parent.relative_to(root)
    return relative.as_posix() if relative.as_posix() != "." else ""


def discover_paths(
    root: Path,
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


def merge_batch(
    *,
    values,
    count,
    mean,
    m2,
    minimum,
    maximum,
    min_location: list,
    max_location: list,
    source_name: str,
    file_path: Path,
    frame_offset: int,
    np,
) -> int:
    original_shape = values.shape
    flat = values.reshape(-1, 3).astype(np.float64, copy=False)
    valid = np.isfinite(flat).all(axis=-1)
    valid &= (flat != -1.0).all(axis=-1)
    valid_indices = np.flatnonzero(valid)
    batch = flat[valid]
    if batch.shape[0] == 0:
        return 0

    for axis_index, axis in enumerate(AXES):
        local_min = int(np.argmin(batch[:, axis_index]))
        local_max = int(np.argmax(batch[:, axis_index]))
        for local_index, extrema, locations, better in (
            (local_min, minimum, min_location, lambda a, b: a < b),
            (local_max, maximum, max_location, lambda a, b: a > b),
        ):
            candidate = float(batch[local_index, axis_index])
            if not better(candidate, extrema[axis_index]):
                continue
            raw_flat_index = int(valid_indices[local_index])
            multi_index = np.unravel_index(raw_flat_index, original_shape[:-1])
            extrema[axis_index] = candidate
            locations[axis_index] = {
                "source": source_name,
                "axis": axis,
                "value": candidate,
                "file": os.path.relpath(file_path, PROJECT_ROOT),
                "frame": int(frame_offset + multi_index[0]),
                "point_index": [int(value) for value in multi_index[1:]],
                "xyz": flat[raw_flat_index].tolist(),
            }

    batch_count = int(batch.shape[0])
    batch_mean = batch.mean(axis=0)
    batch_m2 = np.square(batch - batch_mean).sum(axis=0)
    total_count = count + batch_count
    delta = batch_mean - mean
    mean += delta * batch_count / total_count
    m2 += batch_m2 + np.square(delta) * count * batch_count / total_count
    count[:] = total_count
    return batch_count


def compute_stats(
    sources: list[tuple[str, Path]],
    *,
    chunk_frames: int,
    progress_every: int,
    task_include: tuple[str, ...] | None,
    task_exclude: tuple[str, ...] | None,
) -> dict[str, object]:
    h5py, np = _require_dependencies()
    count = np.zeros(3, dtype=np.int64)
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    min_location = [None, None, None]
    max_location = [None, None, None]
    files_by_source: dict[str, int] = {}
    valid_points_by_source: dict[str, int] = {}

    work_items: list[tuple[str, Path]] = []
    for source_name, root in sources:
        paths = discover_paths(root, task_include, task_exclude)
        if not paths:
            raise FileNotFoundError(f"No matching HDF5 files under {root}")
        files_by_source[source_name] = len(paths)
        valid_points_by_source[source_name] = 0
        work_items.extend((source_name, path) for path in paths)

    for file_index, (source_name, path) in enumerate(work_items, start=1):
        with h5py.File(path, "r") as source:
            if "keypoints" not in source:
                raise KeyError(f"Missing /keypoints dataset: {path}")
            keypoints = source["keypoints"]
            if keypoints.ndim != 4 or keypoints.shape[1:] != (2, 21, 3):
                raise ValueError(
                    f"Expected keypoints [T,2,21,3], got {keypoints.shape}: {path}"
                )
            mode = source.attrs.get("mode", "")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8")
            if str(mode) != "norm":
                raise ValueError(f"Expected mode='norm', got {mode!r}: {path}")
            for start in range(0, keypoints.shape[0], chunk_frames):
                valid_points_by_source[source_name] += merge_batch(
                    values=keypoints[start : start + chunk_frames],
                    count=count,
                    mean=mean,
                    m2=m2,
                    minimum=minimum,
                    maximum=maximum,
                    min_location=min_location,
                    max_location=max_location,
                    source_name=source_name,
                    file_path=path,
                    frame_offset=start,
                    np=np,
                )
        if progress_every and file_index % progress_every == 0:
            print(f"processed_files: {file_index}/{len(work_items)}")

    if np.any(count == 0):
        raise ValueError(f"No valid keypoint values found: count={count.tolist()}")
    std = np.sqrt(m2 / count)
    return {
        "roots": {name: os.path.relpath(root, PROJECT_ROOT) for name, root in sources},
        "num_files": len(work_items),
        "num_files_by_source": files_by_source,
        "valid_points_by_source": valid_points_by_source,
        "task_include": list(task_include) if task_include else None,
        "task_exclude": list(task_exclude) if task_exclude else None,
        "validity_logic": "all coordinates finite and every coordinate != -1",
        "axes": list(AXES),
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "min_location": min_location,
        "max_location": max_location,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("robot", "human_robot"), required=True)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-include", nargs="*")
    parser.add_argument("--task-exclude", nargs="*")
    parser.add_argument("--chunk-frames", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    robot_root = args.robot_root.expanduser().absolute()
    human_root = args.human_root.expanduser().absolute()
    sources = [("robot", robot_root)]
    if args.mode == "human_robot":
        sources.append(("human", human_root))
    for _, root in sources:
        if not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")

    task_include = parse_task_filter(args.task_include)
    task_exclude = parse_task_filter(args.task_exclude)
    if args.dry_run:
        for source_name, root in sources:
            paths = discover_paths(root, task_include, task_exclude)
            print(f"{source_name}: root={root}, files={len(paths)}")
        return

    stats = compute_stats(
        sources,
        chunk_frames=args.chunk_frames,
        progress_every=args.progress_every,
        task_include=task_include,
        task_exclude=task_exclude,
    )
    stats["mode"] = args.mode
    text = json.dumps(stats, indent=2)
    print(text)
    output = args.output
    if output is None:
        name = (
            "keypoint_stats_real_robot.json"
            if args.mode == "robot"
            else "keypoint_stats_real_human_robot.json"
        )
        output = SCRIPT_DIR / name
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(f"{text}\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"saved_to: {output}")


if __name__ == "__main__":
    main()
