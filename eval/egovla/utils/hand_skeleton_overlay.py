from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
        [-1.1089e-05, -1.0000e00, -4.0362e-06, 7.8595e-06],
        [-7.6605e-01, 1.1089e-05, -6.4278e-01, 1.1617e00],
        [6.4278e-01, -4.0362e-06, -7.6605e-01, 1.2444e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
    ],
    dtype=np.float32,
)


def _quat_to_rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in quat_wxyz]
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def pose_to_mat(pos: np.ndarray, quat: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    if quat_order == "wxyz":
        quat_wxyz = quat
    elif quat_order == "xyzw":
        qx, qy, qz, qw = quat
        quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")
    R_mat = _quat_to_rotmat_wxyz(quat_wxyz)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = pos
    return T


def axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    angle = float(angle)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = (axis / norm).tolist()
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def project_point(point_w: np.ndarray, K: np.ndarray = K_DEFAULT, extr: np.ndarray = EXTR_DEFAULT) -> tuple[float, float, float]:
    pw_h = np.array([float(point_w[0]), float(point_w[1]), float(point_w[2]), 1.0], dtype=np.float32)
    pc_h = extr @ pw_h
    pc = pc_h[:3]
    z = max(float(pc[2]), 1e-6)
    u = float(K[0, 0]) * (float(pc[0]) / z) + float(K[0, 2])
    v = float(K[1, 1]) * (float(pc[1]) / z) + float(K[1, 2])
    return u, v, z


def load_joint_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_skeleton_cfg(path: str) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for skeleton overlay config.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class JointInfo:
    parent_link: str
    child_link: str
    axis: np.ndarray
    T_p0: np.ndarray
    T_c0: np.ndarray


def load_usd_hand_kinematics(usd_path: str, side_prefix: str):
    try:
        from pxr import Usd, UsdPhysics
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pxr (USD) is required for FK hand skeleton overlay.") from exc

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    base_link = f"{side_prefix}_hand_base_link"
    joints: dict[str, JointInfo] = {}
    children: dict[str, list[str]] = {}

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
            tok = axis_attr.upper()
            if tok == "X":
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif tok == "Y":
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif tok == "Z":
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown joint axis token: {axis_attr}")
        else:
            axis = np.array([axis_attr[0], axis_attr[1], axis_attr[2]], dtype=np.float32)

        pos0 = joint.GetLocalPos0Attr().Get()
        rot0 = joint.GetLocalRot0Attr().Get()
        pos1 = joint.GetLocalPos1Attr().Get()
        rot1 = joint.GetLocalRot1Attr().Get()

        def gf_quat_to_wxyz(q) -> np.ndarray:
            if q is None:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            try:
                qw = q.GetReal()
                qx, qy, qz = q.GetImaginary()
            except Exception:
                qw = q[0]
                qx, qy, qz = q[1], q[2], q[3]
            return np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float32)

        def gf_vec_to_np(v) -> np.ndarray:
            if v is None:
                return np.zeros(3, dtype=np.float32)
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)

        T_p0 = pose_to_mat(gf_vec_to_np(pos0), gf_quat_to_wxyz(rot0), "wxyz")
        T_c0 = pose_to_mat(gf_vec_to_np(pos1), gf_quat_to_wxyz(rot1), "wxyz")

        joint_name = prim.GetName()
        joints[joint_name] = JointInfo(link0, link1, axis, T_p0, T_c0)
        children.setdefault(link0, []).append(joint_name)

    return base_link, joints, children


def compute_hand_link_transforms(
    base_pose: np.ndarray,
    base_link: str,
    joints: dict[str, JointInfo],
    children: dict[str, list[str]],
    joint_angles: dict[str, float],
) -> dict[str, np.ndarray]:
    T_world: dict[str, np.ndarray] = {base_link: base_pose}
    stack = [base_link]
    while stack:
        parent = stack.pop()
        for joint_name in children.get(parent, []):
            info = joints[joint_name]
            angle = float(joint_angles.get(joint_name, 0.0))
            Rm = axis_angle_to_mat(info.axis, angle)
            T_rot = np.eye(4, dtype=np.float32)
            T_rot[:3, :3] = Rm
            T_parent = T_world[parent]
            T_child = T_parent @ info.T_p0 @ T_rot @ np.linalg.inv(info.T_c0)
            if info.child_link not in T_world:
                T_world[info.child_link] = T_child
                stack.append(info.child_link)
    return T_world


def compute_hand_link_positions(
    base_pose: np.ndarray,
    base_link: str,
    joints: dict[str, JointInfo],
    children: dict[str, list[str]],
    joint_angles: dict[str, float],
    target_links: list[str],
) -> dict[str, np.ndarray]:
    T_world = compute_hand_link_transforms(base_pose, base_link, joints, children, joint_angles)
    out: dict[str, np.ndarray] = {}
    for link in target_links:
        if link in T_world:
            out[link] = T_world[link][:3, 3].copy()
    return out, T_world


class HandSkeletonOverlay:
    def __init__(
        self,
        *,
        use_fk_hand: bool,
        joint_names_file: str | None,
        usd_path: str | None,
        skeleton_cfg: str,
        ee_quat_order: str = "wxyz",
    ) -> None:
        self.use_fk_hand = bool(use_fk_hand)
        self.ee_quat_order = ee_quat_order

        cfg = load_skeleton_cfg(skeleton_cfg)
        self.point_names: list[str] = list(cfg.get("point_names", []))
        self.edges: dict[str, list[list[str]]] = cfg.get("edges", {}) or {}
        self.style: dict[str, Any] = cfg.get("style", {}) or {}
        self.point_style: dict[str, Any] = self.style.get("point", {}) or {}
        self.line_style: dict[str, Any] = self.style.get("line", {}) or {}
        self.point_types: dict[str, str] = cfg.get("point_types", {}) or {}
        self.joint_type_colors: dict[str, list[int]] = self.point_style.get("joint_type_colors", {}) or {}
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        self.finger_colors: dict[str, list[int]] = self.style.get(
            "finger_colors",
            {
                "thumb": [255, 0, 0],
                "index": [255, 255, 0],
                "middle": [0, 255, 0],
                "ring": [255, 128, 0],
                "pinky": [0, 0, 255],
            },
        )
        self.palm_color: list[int] = self.point_style.get(
            "palm_color", self.joint_type_colors.get("palm", [0, 140, 255])
        )

        self.joint_name_to_index: dict[str, int] = {}
        self.left_base_link = "L_hand_base_link"
        self.right_base_link = "R_hand_base_link"
        self.left_joints = self.right_joints = None
        self.left_children = self.right_children = None

        if self.use_fk_hand:
            if joint_names_file is None:
                raise ValueError("--joint_names_file is required when --use_fk_hand is set.")
            if usd_path is None:
                raise ValueError("--usd_path is required when --use_fk_hand is set.")
            joint_names = load_joint_names(joint_names_file)
            self.joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
            self.left_base_link, self.left_joints, self.left_children = load_usd_hand_kinematics(usd_path, "L")
            self.right_base_link, self.right_joints, self.right_children = load_usd_hand_kinematics(usd_path, "R")

    def _rgb_to_bgr(self, rgb: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
        return int(rgb[2]), int(rgb[1]), int(rgb[0])

    def _get_point_side(self, name: str) -> str | None:
        if name.startswith("L_"):
            return "L"
        if name.startswith("R_"):
            return "R"
        return None

    def _get_point_finger(self, name: str) -> str | None:
        parts = name.split("_")
        if len(parts) >= 3 and parts[1] == "hand" and parts[2] == "base":
            return "palm"
        if len(parts) >= 2 and parts[1] in self.finger_names:
            return parts[1]
        return None

    def _get_point_color_bgr(self, name: str) -> tuple[int, int, int]:
        finger = self._get_point_finger(name)
        if finger == "palm":
            return self._rgb_to_bgr(self.palm_color)
        if finger in self.finger_colors:
            return self._rgb_to_bgr(self.finger_colors[finger])
        ptype = self.point_types.get(name)
        if ptype and ptype in self.joint_type_colors:
            return self._rgb_to_bgr(self.joint_type_colors[ptype])
        rgb = self.point_style.get("color_left", [255, 0, 0])
        return self._rgb_to_bgr(rgb)

    def _compute_points_map(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        points_map: dict[str, np.ndarray] = {}

        left_ee_pose = obs["left_ee_pose"][0].detach().cpu().numpy()
        right_ee_pose = obs["right_ee_pose"][0].detach().cpu().numpy()
        left_base_T = pose_to_mat(left_ee_pose[:3], left_ee_pose[3:7], self.ee_quat_order)
        right_base_T = pose_to_mat(right_ee_pose[:3], right_ee_pose[3:7], self.ee_quat_order)

        left_tips = obs["left_finger_tip_pos"][0].detach().cpu().numpy()
        right_tips = obs["right_finger_tip_pos"][0].detach().cpu().numpy()
        left_tip_names = ["L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"]
        right_tip_names = ["R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"]
        for i, name in enumerate(left_tip_names):
            if i < left_tips.shape[0]:
                points_map[name] = left_tips[i].copy()
        for i, name in enumerate(right_tip_names):
            if i < right_tips.shape[0]:
                points_map[name] = right_tips[i].copy()

        if self.use_fk_hand and self.point_names:
            qpos = obs["qpos"][0].detach().cpu().numpy()
            left_joint_angles: dict[str, float] = {}
            right_joint_angles: dict[str, float] = {}
            for jname, jidx in self.joint_name_to_index.items():
                if jidx >= qpos.shape[0]:
                    continue
                if jname.startswith("L_"):
                    left_joint_angles[jname] = float(qpos[jidx])
                elif jname.startswith("R_"):
                    right_joint_angles[jname] = float(qpos[jidx])

            left_pos, left_T_world = compute_hand_link_positions(
                base_pose=left_base_T,
                base_link=self.left_base_link,
                joints=self.left_joints,
                children=self.left_children,
                joint_angles=left_joint_angles,
                target_links=[n for n in self.point_names if n.startswith("L_")],
            )
            right_pos, right_T_world = compute_hand_link_positions(
                base_pose=right_base_T,
                base_link=self.right_base_link,
                joints=self.right_joints,
                children=self.right_children,
                joint_angles=right_joint_angles,
                target_links=[n for n in self.point_names if n.startswith("R_")],
            )
            points_map.update(left_pos)
            points_map.update(right_pos)

            base_offset_local = np.array([0.0, -0.04, 0.0, 1.0], dtype=np.float32)
            points_map["L_hand_base_link"] = (left_base_T @ base_offset_local)[:3].copy()
            points_map["R_hand_base_link"] = (right_base_T @ base_offset_local)[:3].copy()

            for side_prefix, T_world in [("L", left_T_world), ("R", right_T_world)]:
                for finger in ["index", "middle", "ring", "pinky"]:
                    link_name = f"{side_prefix}_{finger}_intermediate"
                    if link_name in T_world:
                        y_offset = -0.02 if side_prefix == "L" else 0.02
                        offset_local = np.array([0.0, y_offset, 0.0, 1.0], dtype=np.float32)
                        p = T_world[link_name] @ offset_local
                        points_map[f"{side_prefix}_{finger}_intermediate_tip"] = p[:3].copy()
        return points_map

    def extract_keypoints(self, obs: dict[str, Any]) -> np.ndarray:
        """Return Skel-WAM raw keypoints as [2, 21, 3] in right/left order."""
        points_map = self._compute_points_map(obs)
        keypoints = np.full((2, 21, 3), -1.0, dtype=np.float32)
        side_offsets = {"R": 0, "L": 21}
        for side, offset in side_offsets.items():
            hand_idx = 0 if side == "R" else 1
            for idx, name in enumerate(self.point_names[offset : offset + 21]):
                if name not in points_map:
                    continue
                u, v, z = project_point(points_map[name], K_DEFAULT, EXTR_DEFAULT)
                if z <= 0:
                    continue
                keypoints[hand_idx, idx] = np.array([u / 1280.0, v / 720.0, z], dtype=np.float32)
        return keypoints

    def overlay_on_rgb(self, rgb: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        import cv2

        img_rgb = np.asarray(rgb)
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image HxWx3, got {img_rgb.shape}")

        img_bgr = img_rgb[..., ::-1].copy()
        img_h, img_w = img_bgr.shape[:2]

        sx = img_w / 1280.0
        sy = img_h / 720.0
        points_map = self._compute_points_map(obs)

        proj_map: dict[str, tuple[int, int, float]] = {}
        for name in self.point_names:
            if name not in points_map:
                continue
            u, v, z = project_point(points_map[name], K_DEFAULT, EXTR_DEFAULT)
            if z <= 0:
                continue
            ui = int(round(u * sx))
            vi = int(round(v * sy))
            if 0 <= ui < img_w and 0 <= vi < img_h:
                proj_map[name] = (ui, vi, z)

        point_radius = int(self.point_style.get("radius", 2))
        point_thickness = int(self.point_style.get("thickness", -1))
        line_thickness = int(self.line_style.get("thickness", 1))

        finger_groups: dict[tuple[str, str], dict[str, Any]] = {}
        for side_key in ("left", "right"):
            for a, b in (self.edges.get(side_key, []) or []):
                if a not in proj_map or b not in proj_map:
                    continue
                side = self._get_point_side(a) or self._get_point_side(b)
                finger_a = self._get_point_finger(a)
                finger_b = self._get_point_finger(b)
                finger = finger_a if finger_a != "palm" else finger_b
                if side is None or finger not in self.finger_names:
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

        for name in self.point_names:
            if name not in proj_map:
                continue
            side = self._get_point_side(name)
            finger = self._get_point_finger(name)
            if side is None or finger not in self.finger_names:
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

        # Larger camera z is farther away; draw far fingers first so near fingers cover them.
        drawable_groups.sort(key=lambda group: group["depth"], reverse=True)
        for group in drawable_groups:
            color = self._rgb_to_bgr(self.finger_colors.get(group["finger"], [255, 255, 255]))
            for a, b in group["edges"]:
                cv2.line(img_bgr, proj_map[a][:2], proj_map[b][:2], color, line_thickness)
            for name in sorted(group["points"]):
                ui, vi, _z = proj_map[name]
                cv2.circle(img_bgr, (ui, vi), point_radius, color, point_thickness)

        # Draw palm/base points last so the shared root remains visible.
        for name, (ui, vi, _z) in proj_map.items():
            if self._get_point_finger(name) != "palm":
                continue
            cv2.circle(img_bgr, (ui, vi), point_radius, self._get_point_color_bgr(name), point_thickness)

        return img_bgr[..., ::-1].copy()

