#!/usr/bin/env python3
"""EgoVLA skeleton dataset preprocessing. See docs/skeleton_preprocessing.md.

Ported from the benchmark test tools; runs without starting Isaac Sim.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from tqdm import tqdm

from project_robot_hand_skeleton_egovla import run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch process EgoVLA_SIM HDF5 files.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of EgoVLA_SIM.")
    parser.add_argument("--out_root", type=str, default=None, help="Output root for skeleton videos (required unless --no_video).")
    parser.add_argument("--no_video", action="store_true", help="Do not write overlay skeleton video.")
    parser.add_argument("--norm_keypoints", action="store_true", help="If set, save normalized keypoints (x/W, y/H) instead of pixels.")
    parser.add_argument("--keypoints_out_root", type=str, default=None, help="If set, save per-frame keypoints (.hdf5) mirroring the HDF5 tree.")
    parser.add_argument("--keypoints_mode", type=str, default="pixel", choices=["pixel", "norm"], help="Keypoints export mode (pixel or normalized).")
    parser.add_argument("--keypoints_include_z", action="store_true", help="Also export projected depth z (camera frame).")
    parser.add_argument("--pattern", type=str, default="*.hdf5", help="Glob pattern for HDF5 files.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output mp4 exists.")

    # Pass-through options used by single-file runner.
    parser.add_argument("--hdf5_images", type=str, default="observations/images/main")
    parser.add_argument("--hdf5_left_tips", type=str, default="observations/left_finger_tip_pos")
    parser.add_argument("--hdf5_right_tips", type=str, default="observations/right_finger_tip_pos")
    parser.add_argument("--hdf5_qpos", type=str, default="observations/qpos")
    parser.add_argument("--hdf5_left_ee_pose", type=str, default="observations/left_ee_pose")
    parser.add_argument("--hdf5_right_ee_pose", type=str, default="observations/right_ee_pose")
    parser.add_argument("--joint_names_file", type=str, default=None)
    parser.add_argument("--usd_path", type=str, default=None)
    parser.add_argument("--use_fk_hand", action="store_true")
    parser.add_argument("--skeleton_cfg", type=str, default=None)
    parser.add_argument("--ee_quat_order", type=str, default="wxyz", choices=["wxyz", "xyzw"])
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug_save_dir", type=str, default=None)
    parser.add_argument("--debug_draw_every", type=int, default=1)
    parser.add_argument("--debug_draw_scale", type=float, default=0.1)
    return parser


def _make_args(base: argparse.Namespace, hdf5_path: Path, out_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        hdf5=str(hdf5_path),
        hdf5_images=base.hdf5_images,
        hdf5_left_tips=base.hdf5_left_tips,
        hdf5_right_tips=base.hdf5_right_tips,
        hdf5_qpos=base.hdf5_qpos,
        hdf5_left_ee_pose=base.hdf5_left_ee_pose,
        hdf5_right_ee_pose=base.hdf5_right_ee_pose,
        joint_names_file=base.joint_names_file,
        usd_path=base.usd_path,
        use_fk_hand=base.use_fk_hand,
        skeleton_cfg=base.skeleton_cfg,
        ee_quat_order=base.ee_quat_order,
        video_out=(None if getattr(base, 'no_video', False) or out_path is None else str(out_path)),
        keypoints_out=(str(getattr(base, 'keypoints_out_path', None)) if hasattr(base, 'keypoints_out_path') else None),
        keypoints_mode=('norm' if getattr(base, 'norm_keypoints', False) else getattr(base, 'keypoints_mode', 'pixel')) ,
        keypoints_include_z=getattr(base, 'keypoints_include_z', False),
        max_frames=base.max_frames,
        fps=base.fps,
        debug_save_dir=base.debug_save_dir,
        debug_draw_every=base.debug_draw_every,
        debug_draw_scale=base.debug_draw_scale,
    )


def _iter_hdf5(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.no_video and not args.out_root:
        raise ValueError('--out_root is required unless --no_video is set.')

    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else None
    hdf5_files = _iter_hdf5(root, args.pattern)
    if not hdf5_files:
        print(f"[warn] No hdf5 files found under {root} with pattern {args.pattern}")
        return

    if out_root is not None:
        os.makedirs(out_root, exist_ok=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for hdf5_path in hdf5_files:
            rel = hdf5_path.relative_to(root)
            out_path = None
            if out_root is not None:
                out_path = (out_root / rel).with_suffix(".mp4")
            keypoints_out_path = None
            if args.keypoints_out_root:
                kp_root = Path(args.keypoints_out_root)
                keypoints_out_path = (kp_root / rel).with_suffix('.hdf5')
                keypoints_out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and ((out_path is None) or out_path.exists()) and (not keypoints_out_path or keypoints_out_path.exists()):
                continue
            jobs.append((hdf5_path, out_path))

        futures = []
        for hdf5_path, out_path in jobs:
            # Stash per-job keypoints path onto the base args so _make_args can pick it up.
            if args.keypoints_out_root:
                args.keypoints_out_path = str((Path(args.keypoints_out_root) / hdf5_path.relative_to(root)).with_suffix('.hdf5'))
            else:
                args.keypoints_out_path = None
            job_args = _make_args(args, hdf5_path, out_path)
            futures.append(executor.submit(run, job_args))

        errors = []
        with tqdm(total=len(futures), desc="Processing HDF5") as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    errors.append(exc)
                    print(f"[error] {exc}")
                pbar.update(1)
        if errors:
            raise RuntimeError(f"{len(errors)} preprocessing jobs failed; see errors above.")


if __name__ == "__main__":
    main()
