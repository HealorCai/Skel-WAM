"""
Compute per-dimension mean and standard deviation for qpos trajectories.

Run from the repository root:
    python3 utils/egovla_utils/compute_qpos_stats.py

The defaults scan the seven EgoVLA training tasks under
``data/EgoVLA_SIM/qpos`` and write
``utils/egovla_utils/qpos_stats_egovla_7tasks.json``.

Equivalent explicit command:
    python3 utils/egovla_utils/compute_qpos_stats.py \
        --root data/EgoVLA_SIM/qpos \
        --dataset-key observations/qpos \
        --task-include Close-Drawer Flip-Mug Open-Drawer Open-Laptop \
            Pour-Balls Push-Box Stack-Can \
        --output utils/egovla_utils/qpos_stats_egovla_7tasks.json
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
DEFAULT_ROOT = PROJECT_ROOT / "data/EgoVLA_SIM/qpos"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "qpos_stats_egovla_7tasks.json"
DEFAULT_TASKS = (
    "Close-Drawer",
    "Flip-Mug",
    "Open-Drawer",
    "Open-Laptop",
    "Pour-Balls",
    "Push-Box",
    "Stack-Can",
)


def parse_task_filter(tasks: Sequence[str] | str | None) -> tuple[str, ...] | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for qpos trajectories."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--dataset-key", default="observations/qpos")
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
        "--extensions",
        nargs="+",
        default=[".hdf5", ".npy"],
        help="File extensions to scan.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=4096,
        help="Number of frames to read from each HDF5 dataset at a time.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many files. Set to 0 to disable.",
    )
    return parser.parse_args()


def task_from_path(root: Path, path: Path) -> str:
    rel_parent = path.parent.relative_to(root)
    return str(rel_parent) if str(rel_parent) != "." else ""


def discover_paths(
    root: Path,
    *,
    extensions: Sequence[str],
    task_include: tuple[str, ...] | None,
    task_exclude: tuple[str, ...] | None,
) -> list[Path]:
    normalized_exts = tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    )
    if task_include:
        allowed = set(task_include)
        paths = [path for path in paths if task_from_path(root, path) in allowed]
    elif task_exclude:
        blocked = set(task_exclude)
        paths = [path for path in paths if task_from_path(root, path) not in blocked]
    return paths


def init_accumulators(qpos_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if qpos_dim <= 0:
        raise ValueError(f"`qpos_dim` must be positive, got {qpos_dim}")
    count = np.zeros(qpos_dim, dtype=np.int64)
    mean = np.zeros(qpos_dim, dtype=np.float64)
    m2 = np.zeros(qpos_dim, dtype=np.float64)
    return count, mean, m2


def merge_batch(
    count: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> None:
    """Merge one batch into per-dimension Welford accumulators."""
    if values.ndim < 2:
        raise ValueError(f"Expected qpos shape (..., D), got {values.shape}")
    qpos_dim = int(values.shape[-1])
    if qpos_dim != mean.shape[0]:
        raise ValueError(f"Qpos dim mismatch: got {qpos_dim}, expected {mean.shape[0]}")

    flat = values.reshape(-1, qpos_dim).astype(np.float64, copy=False)
    finite = np.isfinite(flat)
    batch_count = finite.sum(axis=0)
    if not np.any(batch_count):
        return

    safe_values = np.where(finite, flat, 0.0)
    batch_mean = safe_values.sum(axis=0) / np.maximum(batch_count, 1)
    centered = np.where(finite, flat - batch_mean, 0.0)
    batch_m2 = np.square(centered).sum(axis=0)

    total_count = count + batch_count
    delta = batch_mean - mean
    mean += delta * batch_count / np.maximum(total_count, 1)
    m2 += batch_m2 + np.square(delta) * count * batch_count / np.maximum(
        total_count, 1
    )
    count[:] = total_count


def read_qpos_dim(path: Path, dataset_key: str) -> int:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
        if array.ndim < 2:
            raise ValueError(f"Expected qpos shape (..., D), got {array.shape}: {path}")
        return int(array.shape[-1])
    with h5py.File(path, "r") as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"Missing dataset {dataset_key!r}: {path}")
        dataset = h5_file[dataset_key]
        if dataset.ndim < 2:
            raise ValueError(f"Expected qpos shape (..., D), got {dataset.shape}: {path}")
        return int(dataset.shape[-1])


def merge_file(
    path: Path,
    *,
    dataset_key: str,
    chunk_frames: int,
    count: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
) -> None:
    if path.suffix.lower() == ".npy":
        merge_batch(count, mean, m2, np.load(path, mmap_mode="r"))
        return

    with h5py.File(path, "r") as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"Missing dataset {dataset_key!r}: {path}")
        qpos = h5_file[dataset_key]
        if qpos.ndim < 2:
            raise ValueError(f"Expected qpos shape (..., D), got {qpos.shape}: {path}")
        for start in range(0, qpos.shape[0], chunk_frames):
            merge_batch(count, mean, m2, qpos[start : start + chunk_frames])


def compute_stats(
    root: Path,
    *,
    dataset_key: str,
    chunk_frames: int,
    progress_every: int,
    task_include: tuple[str, ...] | None,
    task_exclude: tuple[str, ...] | None,
    extensions: Sequence[str],
) -> dict[str, object]:
    paths = discover_paths(
        root,
        extensions=extensions,
        task_include=task_include,
        task_exclude=task_exclude,
    )
    if not paths:
        raise FileNotFoundError(f"No qpos files found under: {root}")

    qpos_dim = read_qpos_dim(paths[0], dataset_key)
    count, mean, m2 = init_accumulators(qpos_dim)

    for file_index, path in enumerate(paths, start=1):
        merge_file(
            path,
            dataset_key=dataset_key,
            chunk_frames=chunk_frames,
            count=count,
            mean=mean,
            m2=m2,
        )
        if progress_every and file_index % progress_every == 0:
            print(f"processed_files: {file_index}/{len(paths)}")

    if np.any(count == 0):
        missing = np.flatnonzero(count == 0).tolist()
        raise ValueError(f"No finite values found for qpos dims: {missing}")

    std = np.sqrt(m2 / count)
    return {
        "root": Path(os.path.relpath(root, PROJECT_ROOT)).as_posix(),
        "dataset_key": dataset_key,
        "num_files": len(paths),
        "qpos_dim": qpos_dim,
        "task_include": list(task_include) if task_include else None,
        "task_exclude": list(task_exclude) if task_exclude else None,
        "extensions": list(extensions),
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
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

    stats = compute_stats(
        root,
        dataset_key=args.dataset_key,
        chunk_frames=args.chunk_frames,
        progress_every=args.progress_every,
        task_include=parse_task_filter(args.task_include),
        task_exclude=parse_task_filter(args.task_exclude),
        extensions=args.extensions,
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
