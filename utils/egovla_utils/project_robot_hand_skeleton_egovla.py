#!/usr/bin/env python3
"""EgoVLA skeleton dataset preprocessing. See docs/skeleton_preprocessing.md.

Ported from the benchmark test tools; runs without starting Isaac Sim.
"""

import argparse
import os

import h5py
import imageio.v2 as imageio
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed camera intrinsics/extrinsics (world->camera).
K_DEFAULT = np.array(
    [
        [488.6662, 0.0, 640.0],
        [0.0, 488.6662, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

EXTR_DEFAULT = np.array(
    [
        [-1.1089e-05, -1.0000e+00, -4.0362e-06, 7.8595e-06],
        [-7.6605e-01, 1.1089e-05, -6.4278e-01, 1.1617e+00],
        [6.4278e-01, -4.0362e-06, -7.6605e-01, 1.2444e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00],
    ],
    dtype=np.float32,
)


def _project_points(points_w: torch.Tensor, K: torch.Tensor, extr: torch.Tensor) -> torch.Tensor:
    # points_w: (N, 3), K: (3, 3), extr: (4, 4) world->camera
    ones = torch.ones((points_w.shape[0], 1), device=points_w.device, dtype=points_w.dtype)
    pw_h = torch.cat([points_w, ones], dim=-1)  # (N, 4)
    pc_h = (extr @ pw_h.t()).t()  # (N, 4)
    pc = pc_h[:, :3]
    z = pc[:, 2].clamp(min=1e-6)
    x = pc[:, 0] / z
    y = pc[:, 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return torch.stack([u, v, z], dim=-1)


def _save_debug_plot(save_path: str, tips_w: torch.Tensor, axis_len: float) -> None:
    pts = tips_w.detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="g", s=10, label="fingertips")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Fingertips (World)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def _pose_to_mat(pos: np.ndarray, quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        qw, qx, qy, qz = quat
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
    elif quat_order == "xyzw":
        qx, qy, qz, qw = quat
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")
    R_mat = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = pos.astype(np.float32)
    return T


def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / norm
    return R.from_rotvec(axis * float(angle)).as_matrix().astype(np.float32)


class _JointInfo:
    def __init__(self, parent_link: str, child_link: str, axis: np.ndarray, T_p0: np.ndarray, T_c0: np.ndarray):
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis = axis
        self.T_p0 = T_p0
        self.T_c0 = T_c0


def _load_usd_hand_kinematics(usd_path: str, side_prefix: str) -> Tuple[str, Dict[str, _JointInfo], Dict[str, List[str]]]:
    try:
        from pxr import Usd, UsdPhysics
    except Exception as exc:  # pragma: no cover - depends on Isaac/pxr
        raise RuntimeError("pxr is required for USD kinematics. Please run in Isaac/Omniverse env.") from exc

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    base_link = f"{side_prefix}_hand_base_link"
    joints: Dict[str, _JointInfo] = {}
    children: Dict[str, List[str]] = {}

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue
        link0 = str(body0[0]).split("/")[-1]
        link1 = str(body1[0]).split("/")[-1]
        if not (link0.startswith(side_prefix) and link1.startswith(side_prefix)):
            continue

        axis_attr = joint.GetAxisAttr().Get()
        if axis_attr is None:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif isinstance(axis_attr, str):
            axis_token = axis_attr.upper()
            if axis_token == "X":
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif axis_token == "Y":
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif axis_token == "Z":
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown joint axis token: {axis_attr}")
        else:
            axis = np.array([axis_attr[0], axis_attr[1], axis_attr[2]], dtype=np.float32)

        pos0 = joint.GetLocalPos0Attr().Get()
        rot0 = joint.GetLocalRot0Attr().Get()
        pos1 = joint.GetLocalPos1Attr().Get()
        rot1 = joint.GetLocalRot1Attr().Get()

        def _gf_quat_to_wxyz(q) -> np.ndarray:
            if q is None:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            try:
                qw = q.GetReal()
                qx, qy, qz = q.GetImaginary()
            except Exception:
                qw = q[0]
                qx, qy, qz = q[1], q[2], q[3]
            return np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float32)

        def _gf_vec_to_np(v) -> np.ndarray:
            if v is None:
                return np.zeros(3, dtype=np.float32)
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)

        T_p0 = _pose_to_mat(_gf_vec_to_np(pos0), _gf_quat_to_wxyz(rot0), "wxyz")
        T_c0 = _pose_to_mat(_gf_vec_to_np(pos1), _gf_quat_to_wxyz(rot1), "wxyz")

        joint_name = prim.GetName()
        joints[joint_name] = _JointInfo(link0, link1, axis, T_p0, T_c0)
        children.setdefault(link0, []).append(joint_name)

    if base_link not in children:
        print(f"[warn] No joints found with base {base_link}. Check USD naming.")
    return base_link, joints, children


def _compute_hand_link_transforms(
    base_pose: np.ndarray,
    base_link: str,
    joints: Dict[str, _JointInfo],
    children: Dict[str, List[str]],
    joint_angles: Dict[str, float],
) -> Dict[str, np.ndarray]:
    T_world: Dict[str, np.ndarray] = {base_link: base_pose}
    stack = [base_link]
    while stack:
        parent = stack.pop()
        for joint_name in children.get(parent, []):
            info = joints[joint_name]
            angle = float(joint_angles.get(joint_name, 0.0))
            R = _axis_angle_to_mat(info.axis, angle)
            T_rot = np.eye(4, dtype=np.float32)
            T_rot[:3, :3] = R
            T_parent = T_world[parent]
            T_child = T_parent @ info.T_p0 @ T_rot @ np.linalg.inv(info.T_c0)
            if info.child_link not in T_world:
                T_world[info.child_link] = T_child
                stack.append(info.child_link)
    return T_world


def _compute_hand_link_positions(
    base_pose: np.ndarray,
    base_link: str,
    joints: Dict[str, _JointInfo],
    children: Dict[str, List[str]],
    joint_angles: Dict[str, float],
    target_links: List[str],
) -> Dict[str, np.ndarray]:
    T_world = _compute_hand_link_transforms(base_pose, base_link, joints, children, joint_angles)
    out: Dict[str, np.ndarray] = {}
    for link in target_links:
        if link in T_world:
            out[link] = T_world[link][:3, 3].copy()
    return out


def _load_joint_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def _load_skeleton_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _project_point(point_w: np.ndarray, K: np.ndarray, extr: np.ndarray) -> Tuple[float, float, float]:
    pw_h = np.array([point_w[0], point_w[1], point_w[2], 1.0], dtype=np.float32)
    pc_h = extr @ pw_h
    pc = pc_h[:3]
    z = max(float(pc[2]), 1e-6)
    x = float(pc[0]) / z
    y = float(pc[1]) / z
    u = float(K[0, 0]) * x + float(K[0, 2])
    v = float(K[1, 1]) * y + float(K[1, 2])
    return u, v, z


def run(args: argparse.Namespace) -> None:
    K = torch.from_numpy(K_DEFAULT)
    extr = torch.from_numpy(EXTR_DEFAULT)

    writer = None
    frame_count = 0
    use_fk_hand = args.use_fk_hand
    skel_cfg = _load_skeleton_cfg(args.skeleton_cfg) if args.skeleton_cfg else None
    keypoints_out = getattr(args, 'keypoints_out', None)
    keypoints_mode = getattr(args, 'keypoints_mode', 'pixel')
    keypoints_include_z = bool(getattr(args, 'keypoints_include_z', False))
    if keypoints_out and not skel_cfg:
        raise ValueError('--keypoints_out requires --skeleton_cfg (to define point_names ordering).')
    joint_name_to_index: Dict[str, int] = {}
    left_base_link = "L_hand_base_link"
    right_base_link = "R_hand_base_link"
    left_kin = right_kin = None
    left_children = right_children = None
    left_target_links = right_target_links = None
    left_joint_names = right_joint_names = None

    if use_fk_hand:
        if args.joint_names_file is None:
            raise ValueError("--joint_names_file is required when --use_fk_hand is set.")
        if args.usd_path is None:
            raise ValueError("--usd_path is required when --use_fk_hand is set.")

        joint_names = _load_joint_names(args.joint_names_file)
        joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
        left_base_link, left_kin, left_children = _load_usd_hand_kinematics(args.usd_path, "L")
        right_base_link, right_kin, right_children = _load_usd_hand_kinematics(args.usd_path, "R")

        left_target_links = [
            "L_index_proximal",
            "L_index_intermediate",
            "L_index_tip",
            "L_middle_proximal",
            "L_middle_intermediate",
            "L_middle_tip",
            "L_ring_proximal",
            "L_ring_intermediate",
            "L_ring_tip",
            "L_pinky_proximal",
            "L_pinky_intermediate",
            "L_pinky_tip",
            "L_thumb_proximal",
            "L_thumb_intermediate",
            "L_thumb_distal",
            "L_thumb_tip",
        ]
        right_target_links = [name.replace("L_", "R_") for name in left_target_links]

        left_joint_names = [
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
        ]
        right_joint_names = [name.replace("L_", "R_") for name in left_joint_names]

    with h5py.File(args.hdf5, "r") as f:
        if args.hdf5_images not in f:
            raise KeyError(f"HDF5 images dataset not found: {args.hdf5_images} in {args.hdf5}")
        if args.hdf5_left_tips not in f:
            raise KeyError(f"HDF5 left tips dataset not found: {args.hdf5_left_tips} in {args.hdf5}")
        if args.hdf5_right_tips not in f:
            raise KeyError(f"HDF5 right tips dataset not found: {args.hdf5_right_tips} in {args.hdf5}")
        if args.hdf5_left_ee_pose not in f:
            args.hdf5_left_ee_pose = "observations/left_curr_ee_pose"
            if args.hdf5_left_ee_pose not in f:
                raise KeyError(f"HDF5 left ee pose dataset not found: {args.hdf5_left_ee_pose} in {args.hdf5}")
        if args.hdf5_right_ee_pose not in f:
            args.hdf5_right_ee_pose = "observations/right_curr_ee_pose"
            if args.hdf5_right_ee_pose not in f:
                raise KeyError(f"HDF5 right ee pose dataset not found: {args.hdf5_right_ee_pose} in {args.hdf5}")
        if use_fk_hand and args.hdf5_qpos not in f:
            raise KeyError(f"HDF5 qpos dataset not found: {args.hdf5_qpos} in {args.hdf5}")

        img_ds = f[args.hdf5_images]
        left_tips_ds = f[args.hdf5_left_tips]
        right_tips_ds = f[args.hdf5_right_tips]
        left_ee_pose_ds = f[args.hdf5_left_ee_pose]
        right_ee_pose_ds = f[args.hdf5_right_ee_pose]
        qpos_ds = f[args.hdf5_qpos] if use_fk_hand else None
        total_frames = min(
            img_ds.shape[0],
            left_tips_ds.shape[0],
            right_tips_ds.shape[0],
            left_ee_pose_ds.shape[0],
            right_ee_pose_ds.shape[0],
            qpos_ds.shape[0] if qpos_ds is not None else img_ds.shape[0],
        )
        if args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames)
        keypoints_point_names = None
        keypoints_kpz = None
        if keypoints_out:
            keypoints_point_names = list((skel_cfg or {}).get('point_names', []))
            if not keypoints_point_names:
                raise ValueError("skeleton_cfg missing 'point_names'; cannot export keypoints.")
            # (T, N, 3): (x, y, z). Missing/out-of-frame stays -1.
            keypoint_dim = 3 if keypoints_include_z else 2

            keypoints_kpz = np.full((total_frames, len(keypoints_point_names), keypoint_dim), -1.0, dtype=np.float32)

        with tqdm(total=total_frames, desc="Overlay frames") as pbar:
            while frame_count < total_frames:
                left_tips = torch.from_numpy(left_tips_ds[frame_count])
                right_tips = torch.from_numpy(right_tips_ds[frame_count])
                left_ee_pose = left_ee_pose_ds[frame_count]
                right_ee_pose = right_ee_pose_ds[frame_count]
                left_wrist_w = torch.from_numpy(left_ee_pose[:3]).unsqueeze(0)
                right_wrist_w = torch.from_numpy(right_ee_pose[:3]).unsqueeze(0)
                tips_w = torch.cat([left_tips, right_tips], dim=0)  # (10, 3)

                points_w_list = [left_wrist_w, right_wrist_w, tips_w]
                point_colors: List[Tuple[int, int, int]] = []
                point_colors.extend([(255, 255, 0), (0, 255, 255)])  # wrists
                point_colors.extend([(0, 255, 0)] * 5 + [(0, 0, 255)] * 5)  # tips

                points_map: Dict[str, np.ndarray] = {}
                # Apply -0.04 offset along each hand base local Y axis.
                left_base_T = _pose_to_mat(left_ee_pose[:3], left_ee_pose[3:7], args.ee_quat_order)
                right_base_T = _pose_to_mat(right_ee_pose[:3], right_ee_pose[3:7], args.ee_quat_order)
                base_offset_local = np.array([0.0, -0.04, 0.0, 1.0], dtype=np.float32)
                points_map["L_hand_base_link"] = (left_base_T @ base_offset_local)[:3].copy()
                points_map["R_hand_base_link"] = (right_base_T @ base_offset_local)[:3].copy()
                # Map fingertip positions from dataset to named tips.
                left_tip_names = ["L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"]
                right_tip_names = ["R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"]
                for i, name in enumerate(left_tip_names):
                    if i < left_tips.shape[0]:
                        points_map[name] = left_tips[i].cpu().numpy().copy()
                for i, name in enumerate(right_tip_names):
                    if i < right_tips.shape[0]:
                        points_map[name] = right_tips[i].cpu().numpy().copy()

                if use_fk_hand:
                    qpos = qpos_ds[frame_count]
                    left_joint_angles = {}
                    right_joint_angles = {}
                    for name in left_joint_names:
                        if name in joint_name_to_index:
                            left_joint_angles[name] = float(qpos[joint_name_to_index[name]])
                    for name in right_joint_names:
                        if name in joint_name_to_index:
                            right_joint_angles[name] = float(qpos[joint_name_to_index[name]])

                    left_base_T = _pose_to_mat(left_ee_pose[:3], left_ee_pose[3:7], args.ee_quat_order)
                    right_base_T = _pose_to_mat(right_ee_pose[:3], right_ee_pose[3:7], args.ee_quat_order)

                    left_T_world = _compute_hand_link_transforms(
                        left_base_T, left_base_link, left_kin, left_children, left_joint_angles
                    )
                    right_T_world = _compute_hand_link_transforms(
                        right_base_T, right_base_link, right_kin, right_children, right_joint_angles
                    )

                    left_link_pos = {
                        k: left_T_world[k][:3, 3].copy()
                        for k in left_target_links
                        if k in left_T_world
                    }
                    right_link_pos = {
                        k: right_T_world[k][:3, 3].copy()
                        for k in right_target_links
                        if k in right_T_world
                    }

                    left_fk_pts = [left_link_pos[k] for k in left_target_links if k in left_link_pos]
                    right_fk_pts = [right_link_pos[k] for k in right_target_links if k in right_link_pos]
                    if left_fk_pts:
                        points_w_list.append(torch.from_numpy(np.stack(left_fk_pts, axis=0)))
                        point_colors.extend([(0, 200, 0)] * len(left_fk_pts))
                    if right_fk_pts:
                        points_w_list.append(torch.from_numpy(np.stack(right_fk_pts, axis=0)))
                        point_colors.extend([(0, 0, 200)] * len(right_fk_pts))

                    # Virtual points on intermediate links for index/middle/ring/pinky (exclude thumb):
                    # offset 2cm along local Y axis.
                    virtual_pts = []
                    for side_prefix, T_world in [("L", left_T_world), ("R", right_T_world)]:
                        for finger in ["index", "middle", "ring", "pinky"]:
                            link_name = f"{side_prefix}_{finger}_intermediate"
                            if link_name in T_world:
                                y_offset = -0.02 if side_prefix == "L" else 0.02
                                offset_local = np.array([0.0, y_offset, 0.0, 1.0], dtype=np.float32)
                                p = T_world[link_name] @ offset_local
                                virtual_pts.append(p[:3].copy())
                                points_map[f"{side_prefix}_{finger}_intermediate_tip"] = p[:3].copy()
                    if virtual_pts:
                        points_w_list.append(torch.from_numpy(np.stack(virtual_pts, axis=0)))
                        point_colors.extend([(255, 128, 0)] * len(virtual_pts))

                    # Add FK link points to map for skeleton drawing.
                    for name, pos in left_link_pos.items():
                        points_map[name] = pos.copy()
                    for name, pos in right_link_pos.items():
                        points_map[name] = pos.copy()

                points_w = torch.cat(points_w_list, dim=0)
                proj = _project_points(points_w, K, extr)

                need_frame_image = bool(args.video_out) or bool(args.debug_save_dir)
                if need_frame_image:
                    img = img_ds[frame_count].copy()
                    img_h, img_w = img.shape[:2]
                else:
                    # Avoid cv2 drawing work when not writing video / debug images.
                    img = None
                    img_h = int(img_ds.shape[1])
                    img_w = int(img_ds.shape[2])
                sim_w, sim_h = 1280, 720
                sx = img_w / float(sim_w)
                sy = img_h / float(sim_h)

                if skel_cfg:
                    point_names = skel_cfg.get("point_names", [])
                    style = skel_cfg.get("style", {})
                    point_style = style.get("point", {})
                    line_style = style.get("line", {})
                    point_radius = int(point_style.get("radius", 2))
                    point_thickness = int(point_style.get("thickness", -1))
                    joint_type_colors = point_style.get("joint_type_colors", {})
                    point_types = skel_cfg.get("point_types", {})
                    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                    finger_colors = style.get(
                        "finger_colors",
                        {
                            "thumb": [255, 0, 0],
                            "index": [255, 255, 0],
                            "middle": [0, 255, 0],
                            "ring": [255, 128, 0],
                            "pinky": [0, 0, 255],
                        },
                    )
                    palm_color = point_style.get("palm_color", joint_type_colors.get("palm", [0, 140, 255]))

                    def _rgb_to_draw(color) -> Tuple[int, int, int]:
                        return tuple(int(c) for c in color)

                    def _get_point_side(name: str) -> str | None:
                        if name.startswith("L_"):
                            return "L"
                        if name.startswith("R_"):
                            return "R"
                        return None

                    def _get_point_finger(name: str) -> str | None:
                        parts = name.split("_")
                        if len(parts) >= 3 and parts[1] == "hand" and parts[2] == "base":
                            return "palm"
                        if len(parts) >= 2 and parts[1] in finger_names:
                            return parts[1]
                        return None

                    def _get_point_color(name: str) -> Tuple[int, int, int]:
                        finger = _get_point_finger(name)
                        if finger == "palm":
                            return _rgb_to_draw(palm_color)
                        if finger in finger_colors:
                            return _rgb_to_draw(finger_colors[finger])
                        ptype = point_types.get(name)
                        if ptype and ptype in joint_type_colors:
                            return _rgb_to_draw(joint_type_colors[ptype])
                        if name.startswith("L_"):
                            return _rgb_to_draw(point_style.get("color_left", [255, 0, 0]))
                        if name.startswith("R_"):
                            return _rgb_to_draw(point_style.get("color_right", [255, 0, 0]))
                        return (255, 255, 255)

                    # pre-project points used in skeleton
                    proj_map: Dict[str, Tuple[int, int, float]] = {}
                    for name in point_names:
                        if name not in points_map:
                            continue
                        u, v, z = _project_point(points_map[name], K_DEFAULT, EXTR_DEFAULT)
                        if z <= 0:
                            continue
                        ui = int(round(u * sx))
                        vi = int(round(v * sy))
                        if 0 <= ui < img_w and 0 <= vi < img_h:
                            proj_map[name] = (ui, vi, z)
                    if keypoints_out and keypoints_kpz is not None and keypoints_point_names is not None:
                        if keypoints_mode not in ('pixel', 'norm'):
                            raise ValueError(f'Unsupported --keypoints_mode: {keypoints_mode}')
                        denom = np.array([float(img_w), float(img_h)], dtype=np.float32)
                        for j, name in enumerate(keypoints_point_names):
                            if name not in proj_map:
                                continue
                            ui, vi, z = proj_map[name]
                            xy = np.array([float(ui), float(vi)], dtype=np.float32)
                            if keypoints_mode == 'norm':
                                xy = xy / denom
                            keypoints_kpz[frame_count, j, 0:2] = xy
                            if keypoints_include_z and keypoints_kpz.shape[2] == 3:
                                keypoints_kpz[frame_count, j, 2] = float(z)

                    # Draw fingers globally by depth. Larger camera z is farther away, so draw it first.
                    edges = skel_cfg.get("edges", {})
                    line_thickness = int(line_style.get("thickness", 1))
                    if img is not None:
                        finger_groups: Dict[Tuple[str, str], Dict] = {}
                        for side_key in ("left", "right"):
                            for a, b in edges.get(side_key, []):
                                if a not in proj_map or b not in proj_map:
                                    continue
                                side = _get_point_side(a) or _get_point_side(b)
                                finger_a = _get_point_finger(a)
                                finger_b = _get_point_finger(b)
                                finger = finger_a if finger_a != "palm" else finger_b
                                if side is None or finger not in finger_names:
                                    continue
                                group = finger_groups.setdefault(
                                    (side, finger),
                                    {"side": side, "finger": finger, "edges": [], "points": set()},
                                )
                                group["edges"].append((a, b))
                                if finger_a == finger:
                                    group["points"].add(a)
                                if finger_b == finger:
                                    group["points"].add(b)

                        for name in point_names:
                            if name not in proj_map:
                                continue
                            side = _get_point_side(name)
                            finger = _get_point_finger(name)
                            if side is None or finger not in finger_names:
                                continue
                            group = finger_groups.setdefault(
                                (side, finger),
                                {"side": side, "finger": finger, "edges": [], "points": set()},
                            )
                            group["points"].add(name)

                        drawable_groups = []
                        for group in finger_groups.values():
                            depths = [proj_map[name][2] for name in group["points"] if name in proj_map]
                            if not depths:
                                continue
                            group["depth"] = float(np.mean(depths))
                            drawable_groups.append(group)

                        drawable_groups.sort(key=lambda group: group["depth"], reverse=True)
                        for group in drawable_groups:
                            color = _rgb_to_draw(finger_colors.get(group["finger"], [255, 255, 255]))
                            for a, b in group["edges"]:
                                cv2.line(img, proj_map[a][:2], proj_map[b][:2], color, line_thickness)
                            for name in sorted(group["points"]):
                                ui, vi, _z = proj_map[name]
                                cv2.circle(img, (ui, vi), point_radius, color, point_thickness)

                        # Draw palm/base points last so the shared root remains visible.
                        for name, (ui, vi, _z) in proj_map.items():
                            if _get_point_finger(name) != "palm":
                                continue
                            color = _get_point_color(name)
                            cv2.circle(img, (ui, vi), point_radius, color, point_thickness)

                    if args.video_out and img is not None:
                        if writer is None:
                            writer = imageio.get_writer(args.video_out, fps=args.fps)
                        writer.append_data(img)
                    if args.debug_save_dir and (frame_count % max(1, args.debug_draw_every) == 0):
                        os.makedirs(args.debug_save_dir, exist_ok=True)
                        save_path = f"{args.debug_save_dir}/debug_{frame_count:06d}.png"
                        _save_debug_plot(save_path, tips_w, args.debug_draw_scale)
                    frame_count += 1
                    pbar.update(1)
                    continue

                if img is not None:
                    for i in range(proj.shape[0]):
                        u, v, z = proj[i].tolist()
                        if z <= 0:
                            continue
                        ui = int(round(u * sx))
                        vi = int(round(v * sy))
                        if 0 <= ui < img_w and 0 <= vi < img_h:
                            if i == 0:
                                color = (255, 255, 0)  # left wrist
                                radius = 4
                            elif i == 1:
                                color = (0, 255, 255)  # right wrist
                                radius = 4
                            else:
                                color = point_colors[i] if i < len(point_colors) else (255, 255, 255)
                                radius = 3
                            cv2.circle(img, (ui, vi), radius, color, -1)

                if args.video_out and img is not None:
                    if writer is None:
                        writer = imageio.get_writer(args.video_out, fps=args.fps)
                    writer.append_data(img)

                if args.debug_save_dir and (frame_count % max(1, args.debug_draw_every) == 0):
                    os.makedirs(args.debug_save_dir, exist_ok=True)
                    save_path = f"{args.debug_save_dir}/debug_{frame_count:06d}.png"
                    _save_debug_plot(save_path, tips_w, args.debug_draw_scale)

                frame_count += 1
                pbar.update(1)

    if writer is not None:
        writer.close()
    if keypoints_out and keypoints_kpz is not None and keypoints_point_names is not None:
        os.makedirs(os.path.dirname(keypoints_out) or '.', exist_ok=True)
        if keypoints_mode not in ('pixel', 'norm'):
            raise ValueError(f'Unsupported --keypoints_mode: {keypoints_mode}')
        if len(keypoints_point_names) % 2 != 0:
            raise ValueError(f'Expected even number of point_names, got {len(keypoints_point_names)}')
        half = len(keypoints_point_names) // 2
        kps = keypoints_kpz.reshape((total_frames, 2, half, 3 if keypoints_include_z else 2))
        ext = os.path.splitext(keypoints_out)[1].lower()
        if ext in ('.h5', '.hdf5'):
            with h5py.File(keypoints_out, 'w') as out_f:
                out_f.create_dataset('keypoints', data=kps, compression='gzip', compression_opts=4)
                out_f.attrs['mode'] = keypoints_mode
                out_f.attrs['point_names'] = np.array(keypoints_point_names, dtype='S')
                out_f.attrs['image_width'] = int(img_w)
                out_f.attrs['image_height'] = int(img_h)
        else:
            # Fallback: npz without visible mask; missing points encoded as -1.
            np.savez_compressed(
                keypoints_out,
                keypoints=kps,
                point_names=np.array(keypoints_point_names, dtype=object),
                mode=np.array([keypoints_mode], dtype=object),
                image_width=np.int32(img_w),
                image_height=np.int32(img_h),
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project fingertip skeleton onto dataset images (no sim).")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 dataset.")
    parser.add_argument("--hdf5_images", type=str, default="observations/images/main", help="Image dataset path.")
    parser.add_argument("--hdf5_left_tips", type=str, default="observations/left_finger_tip_pos", help="Left tips path.")
    parser.add_argument("--hdf5_right_tips", type=str, default="observations/right_finger_tip_pos", help="Right tips path.")
    parser.add_argument("--hdf5_qpos", type=str, default="observations/qpos", help="Qpos dataset path.")
    parser.add_argument("--hdf5_left_ee_pose", type=str, default="observations/left_ee_pose", help="Left EE pose path.")
    parser.add_argument("--hdf5_right_ee_pose", type=str, default="observations/right_ee_pose", help="Right EE pose path.")
    parser.add_argument("--joint_names_file", type=str, default=None, help="Path to joint names file (one per line).")
    parser.add_argument("--usd_path", type=str, default=None, help="USD path for robot.")
    parser.add_argument("--use_fk_hand", action="store_true", help="Use qpos + USD to FK hand keypoints.")
    parser.add_argument("--skeleton_cfg", type=str, default=None, help="YAML config for skeleton points/edges/colors.")
    parser.add_argument("--ee_quat_order", type=str, default="wxyz", choices=["wxyz", "xyzw"], help="Quaternion order in ee_pose.")
    parser.add_argument("--video_out", type=str, default=None, help="Path to save overlay video (mp4).")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to record. <=0 means no limit.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--debug_save_dir", type=str, default=None, help="Save matplotlib 3D debug images to dir.")
    parser.add_argument("--debug_draw_every", type=int, default=1, help="Save debug every N frames.")
    parser.add_argument("--debug_draw_scale", type=float, default=0.1, help="Unused placeholder for compatibility.")
    parser.add_argument("--keypoints_out", type=str, default=None, help="Save per-frame hand keypoints to .hdf5/.h5 (preferred) or .npz.")
    parser.add_argument("--keypoints_mode", type=str, default="pixel", choices=["pixel", "norm"], help="Export keypoints in pixel coords or normalized [0,1] coords.")
    parser.add_argument("--keypoints_include_z", action="store_true", help="Also export projected depth z (camera frame).")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
