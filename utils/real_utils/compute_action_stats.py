#!/usr/bin/env python3
"""Compute per-dimension statistics for processed real-robot actions.

The default input is ``data/real_robot/action``. Every file must contain a
float-compatible ``[T, 42]`` NumPy array ordered as right 21-D then left 21-D.

Example:

    cd /path/to/Skel-WAM
    python utils/real_utils/compute_action_stats.py
    python utils/real_utils/compute_action_stats.py --task-include put_something_into_drawer

The default output is ``utils/real_utils/action_stats_real_robot.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data/real_robot/action"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "action_stats_real_robot.json"
ACTION_DIM = 42


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Action statistics require numpy") from exc
    return np


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
    paths = sorted(root.rglob("*.npy"))
    if task_include:
        allowed = set(task_include)
        paths = [path for path in paths if task_from_path(root, path) in allowed]
    elif task_exclude:
        blocked = set(task_exclude)
        paths = [path for path in paths if task_from_path(root, path) not in blocked]
    return paths


def dimension_names() -> list[str]:
    names = []
    for side in ("right", "left"):
        names.extend(f"{side}_rotation_6d_{index}" for index in range(6))
        names.extend(f"{side}_translation_{axis}" for axis in ("x", "y", "z"))
        names.extend(f"{side}_xhand_{index}" for index in range(12))
    return names


def merge_batch(values, count, mean, m2, minimum, maximum, np) -> None:
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action [T,{ACTION_DIM}], got {values.shape}")
    batch = values.astype(np.float64, copy=False)
    finite = np.isfinite(batch)
    batch_count = finite.sum(axis=0)
    safe = np.where(finite, batch, 0.0)
    batch_mean = safe.sum(axis=0) / np.maximum(batch_count, 1)
    centered = np.where(finite, batch - batch_mean, 0.0)
    batch_m2 = np.square(centered).sum(axis=0)
    total_count = count + batch_count
    delta = batch_mean - mean
    mean += delta * batch_count / np.maximum(total_count, 1)
    m2 += batch_m2 + np.square(delta) * count * batch_count / np.maximum(
        total_count, 1
    )
    count[:] = total_count
    batch_min = np.min(np.where(finite, batch, np.inf), axis=0)
    batch_max = np.max(np.where(finite, batch, -np.inf), axis=0)
    np.minimum(minimum, batch_min, out=minimum)
    np.maximum(maximum, batch_max, out=maximum)


def compute_stats(
    root: Path,
    *,
    paths: list[Path],
    chunk_frames: int,
    progress_every: int,
    task_include: tuple[str, ...] | None,
    task_exclude: tuple[str, ...] | None,
) -> dict[str, object]:
    np = _require_numpy()
    count = np.zeros(ACTION_DIM, dtype=np.int64)
    mean = np.zeros(ACTION_DIM, dtype=np.float64)
    m2 = np.zeros(ACTION_DIM, dtype=np.float64)
    minimum = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    total_frames = 0

    for file_index, path in enumerate(paths, start=1):
        action = np.load(path, mmap_mode="r")
        if action.ndim != 2 or action.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Expected action [T,{ACTION_DIM}], got {action.shape}: {path}"
            )
        total_frames += int(action.shape[0])
        for start in range(0, action.shape[0], chunk_frames):
            merge_batch(
                action[start : start + chunk_frames],
                count,
                mean,
                m2,
                minimum,
                maximum,
                np,
            )
        if progress_every and file_index % progress_every == 0:
            print(f"processed_files: {file_index}/{len(paths)}")

    if np.any(count == 0):
        missing = np.flatnonzero(count == 0).tolist()
        raise ValueError(f"No finite values found for action dims: {missing}")
    return {
        "root": os.path.relpath(root, PROJECT_ROOT),
        "num_files": len(paths),
        "num_frames": total_frames,
        "action_dim": ACTION_DIM,
        "action_layout": "right_21_then_left_21",
        "side_layout": "rotation_6d_columns_then_translation_xyz_then_xhand_12",
        "dimension_names": dimension_names(),
        "task_include": list(task_include) if task_include else None,
        "task_exclude": list(task_exclude) if task_exclude else None,
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(m2 / count).tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    root = args.root.expanduser().absolute()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    task_include = parse_task_filter(args.task_include)
    task_exclude = parse_task_filter(args.task_exclude)
    paths = discover_paths(root, task_include, task_exclude)
    if not paths:
        raise FileNotFoundError(f"No matching .npy files under {root}")
    if args.dry_run:
        print(f"root={root}, files={len(paths)}")
        return

    stats = compute_stats(
        root,
        paths=paths,
        chunk_frames=args.chunk_frames,
        progress_every=args.progress_every,
        task_include=task_include,
        task_exclude=task_exclude,
    )
    text = json.dumps(stats, indent=2)
    print(text)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(f"{text}\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"saved_to: {output}")


if __name__ == "__main__":
    main()
