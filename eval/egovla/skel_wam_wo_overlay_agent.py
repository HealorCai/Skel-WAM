"""Standalone EgoVLA evaluator for the w/o-skeleton-overlay ablation.

This file intentionally owns the complete evaluation loop: checkpoint
construction, cached Video and Keypoint sampling, Action flow matching,
simulator reset, and receding-horizon execution are all implemented here. Its
defaults match ``scripts/train_skel_wam_multi_node.sh``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def force_conda_aws():
    sys.path = [
        p for p in sys.path
        if "omni.kit.pip_archive" not in p and "pip_prebundle" not in p
    ]

    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", "") or ""
        if (
            name == "botocore" or name.startswith("botocore.")
            or name == "boto3" or name.startswith("boto3.")
            or name == "s3transfer" or name.startswith("s3transfer.")
        ) and ("omni.kit.pip_archive" in file or "pip_prebundle" in file):
            del sys.modules[name]

    print("[force_conda_aws]")
    import botocore
    print("botocore   :", botocore.__file__, botocore.__version__)
    import boto3
    print("boto3      :", boto3.__file__, boto3.__version__)
    import s3transfer
    print("s3transfer :", s3transfer.__file__, s3transfer.__version__)

force_conda_aws()

import os
import tqdm

from omni.isaac.lab.app import AppLauncher

seed_map = {
    "Humanoid-Close-Drawer-v0": 0,
    "Humanoid-Sort-Cans-v0": 1,
    "Humanoid-Flip-Mug-v0": 2,
    "Humanoid-Insert-Cans-v0": 2,
    "Humanoid-Push-Box-v0": 4,
    "Humanoid-Stack-Can-v0": 4,
    "Humanoid-Unload-Cans-v0": 5,
    "Humanoid-Insert-And-Unload-Cans-v0": 6,
    "Humanoid-Open-Laptop-v0": 7,
    "Humanoid-Stack-Can-Into-Drawer-v0": 8,
    "Humanoid-Open-Drawer-v0": 9,
    "Humanoid-Pour-Balls-v0": 12,
}

DEFAULT_PROMPT_TEMPLATE = (
    "An egocentric video of a robot executing the following task: {task}"
)

import argparse
parser = argparse.ArgumentParser(
    description="Evaluate the two-expert Skel-WAM MoT model on EgoVLA."
)
# add egpvla argparse arguments
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--room_idx", type=int, default=None, help="Room Idx")
parser.add_argument("--table_idx", type=int, default=None, help="Table Idx")
parser.add_argument("--num_episodes", type=int, default=None, help="episode_label")
parser.add_argument("--num_trials", type=int, default=1, help="trial label")
parser.add_argument("--saving_path", type=str, default=None, help="result saving path")
parser.add_argument("--save_frames", type=int, default=0, help="result saving path")
parser.add_argument("--additional_label", type=str, default=None, help="additional_label")

# MoT window and flow-matching arguments.
parser.add_argument("--num_frames", type=int, default=49)
parser.add_argument("--action_video_freq_ratio", type=int, default=6)
parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
parser.add_argument("--prompt_template", default=None)
parser.add_argument("--empty_task_name", default=None)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--latent_num_inference_steps", type=int, default=None)
parser.add_argument("--keypoint_num_inference_steps", type=int, default=None)
parser.add_argument("--latent_snr_shift", type=float, default=None)
parser.add_argument("--keypoint_snr_shift", type=float, default=None)
parser.add_argument("--seed", type=int, default=42)
# add Skel-WAM argparse arguments
parser.add_argument("--checkpoint", required=True, help="Path to Skel-WAM checkpoint.")
parser.add_argument(
    "--action_checkpoint",
    default=None,
    help=(
        "Optional portable Action-only checkpoint. When omitted, Action weights "
        "are loaded from --checkpoint."
    ),
)
parser.add_argument("--vae_checkpoint_dir", default="checkpoints/Wan-AI/Wan2.2-TI2V-5B")
parser.add_argument("--exec_action_steps", type=int, default=20)
parser.add_argument(
    "--action_chunk_weighting",
    choices=["none", "exponential"],
    default="none",
    help=(
        "How to combine time-aligned predictions from overlapping action chunks. "
        "'none' preserves the original single-chunk execution."
    ),
)
parser.add_argument(
    "--action_chunk_weight_decay", type=float, default=0.7,
    help="Exponential decay applied per chunk age; the newest chunk has weight 1.",
)
parser.add_argument(
    "--action_chunk_max_history", type=int, default=3,
    help="Maximum number of recent action chunks used by temporal ensembling.",
)

# Optional: draw hand skeleton overlay on camera RGB (like EgoVLA dataset visualization).
parser.add_argument("--overlay_hand_skeleton", action="store_true")
parser.add_argument(
    "--extract_hand_keypoints",
    action="store_true",
    help="Initialize the hand helper for numerical keypoint extraction only.",
)
parser.add_argument("--use_fk_hand", action="store_true")
parser.add_argument("--joint_names_file", type=str, default=None)
parser.add_argument("--usd_path", type=str, default=None)
parser.add_argument("--skeleton_cfg", type=str, default=None)
parser.add_argument("--ee_quat_order", type=str, default="wxyz", choices=["wxyz", "xyzw"])

parser.add_argument("--action_target_frames", type=int, default=49)
parser.add_argument(
    "--action_exclude_indices",
    type=int,
    nargs="*",
    default=[0, 1, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19],
    help=(
        "Zero-based environment action dimensions excluded during training. "
        "The Action head predicts only retained dimensions; excluded dimensions "
        "are restored with their raw-statistics mean before env.step()."
    ),
)
parser.add_argument("--keypoint_dims", type=int, choices=[2, 3], default=3)
parser.add_argument("--history_keypoint_offsets", type=int, nargs="+", default=[-18, -12, -6])
parser.add_argument(
    "--keypoint_stats_path",
    default="utils/egovla_utils/keypoint_stats_egovla_7tasks.json",
)
parser.add_argument(
    "--action_stats_path",
    default="utils/egovla_utils/qpos_stats_egovla_7tasks.json",
)
parser.add_argument("--action_num_inference_steps", type=int, default=None)
parser.add_argument("--action_snr_shift", type=float, default=None)
parser.add_argument("--precomputed_text_embeddings_path", default=None)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.num_frames - 1) % args.action_video_freq_ratio != 0:
    parser.error(
        "(--num_frames - 1) must be divisible by "
        "--action_video_freq_ratio"
    )
if args.exec_action_steps <= 0:
    parser.error("--exec_action_steps must be positive")
if args.action_chunk_weight_decay < 0:
    parser.error("--action_chunk_weight_decay must be non-negative")
if args.action_chunk_max_history <= 0:
    parser.error("--action_chunk_max_history must be positive")

# launch omniverse app
app_launcher = AppLauncher(enable_cameras=True, device=args.device, headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
import humanoid.tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg
import torch

from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

import cv2
import time

from skel_wam.utils.compatibility import validate_fixed_training_settings
from skel_wam.models.skel_wam import (
    infer_action_patch_embedding_mode,
    SkelWamBackboneModel,
    decode_keypoint_validity,
    load_component_state_dict,
    normalize_keypoint_validity,
)
from skel_wam.models.wan.modules.t5 import T5EncoderModel
from skel_wam.models.wan.modules.vae2_2 import Wan2_2_VAE
from skel_wam.utils.flow_matching import FlowMatchingScheduler
from skel_wam.utils.text_embeddings import (
    PrecomputedPromptEmbeddingStore,
    load_precomputed_text_embeddings,
)
import imageio
import numpy as np
import json

from utils.hand_skeleton_overlay import HandSkeletonOverlay

TASK_MAX_HORIZON = {
  "Humanoid-Push-Box-v0": 400,
  "Humanoid-Open-Drawer-v0": 400,
  "Humanoid-Close-Drawer-v0": 200,
  "Humanoid-Flip-Mug-v0": 400,
  "Humanoid-Pour-Balls-v0": 400,

  "Humanoid-Sort-Cans-v0": 900,
  "Humanoid-Insert-Cans-v0": 500,
  "Humanoid-Unload-Cans-v0": 500,

  "Humanoid-Insert-And-Unload-Cans-v0": 900,
  "Humanoid-Orient-Pour-Balls-v0": 600,

  "Humanoid-Stack-Can-v0": 300,
  "Humanoid-Open-Laptop-v0": 300,

  "Humanoid-Stack-Can-Into-Drawer-v0": 400
}

TASK_INIT_EPISODE = {
    "Close-Drawer": [
        ("episode_17.hdf5", 0.008808),
        ("episode_35.hdf5", 0.010483),
        ("episode_39.hdf5", 0.011340),
        ("episode_40.hdf5", 0.011753),
        ("episode_14.hdf5", 0.011799),
        ("episode_3.hdf5", 0.011995),
        ("episode_6.hdf5", 0.012234),
        ("episode_27.hdf5", 0.012507),
        ("episode_41.hdf5", 0.012627),
        ("episode_29.hdf5", 0.012791),
    ],
    "Flip-Mug": [
        ("episode_49.hdf5", 0.002252),
        ("episode_47.hdf5", 0.002291),
        ("episode_15.hdf5", 0.002464),
        ("episode_26.hdf5", 0.002471),
        ("episode_22.hdf5", 0.002494),
        ("episode_48.hdf5", 0.002598),
        ("episode_20.hdf5", 0.002886),
        ("episode_17.hdf5", 0.002915),
        ("episode_24.hdf5", 0.003137),
        ("episode_40.hdf5", 0.003161),
    ],
    "Insert-And-Unload-Cans": [
        ("episode_98.hdf5", 0.002737),
        ("episode_62.hdf5", 0.003789),
        ("episode_64.hdf5", 0.004229),
        ("episode_93.hdf5", 0.004603),
        ("episode_74.hdf5", 0.005061),
        ("episode_61.hdf5", 0.005299),
        ("episode_65.hdf5", 0.005371),
        ("episode_57.hdf5", 0.005518),
        ("episode_66.hdf5", 0.005821),
        ("episode_81.hdf5", 0.005871),
    ],
    "Insert-Cans": [
        ("episode_56.hdf5", 0.006791),
        ("episode_62.hdf5", 0.007331),
        ("episode_68.hdf5", 0.007532),
        ("episode_51.hdf5", 0.007645),
        ("episode_57.hdf5", 0.007861),
        ("episode_60.hdf5", 0.007894),
        ("episode_67.hdf5", 0.007988),
        ("episode_64.hdf5", 0.008021),
        ("episode_66.hdf5", 0.008172),
        ("episode_24.hdf5", 0.008201),
    ],
    "Open-Drawer": [
        ("episode_35.hdf5", 0.008118),
        ("episode_0.hdf5", 0.010806),
        ("episode_42.hdf5", 0.010956),
        ("episode_11.hdf5", 0.011135),
        ("episode_14.hdf5", 0.011418),
        ("episode_32.hdf5", 0.011524),
        ("episode_4.hdf5", 0.012320),
        ("episode_2.hdf5", 0.012502),
        ("episode_37.hdf5", 0.012656),
        ("episode_23.hdf5", 0.012837),
    ],
    "Open-Laptop": [
        ("episode_8.hdf5", 0.008424),
        ("episode_35.hdf5", 0.008433),
        ("episode_25.hdf5", 0.009747),
        ("episode_90.hdf5", 0.009884),
        ("episode_5.hdf5", 0.010008),
        ("episode_72.hdf5", 0.010378),
        ("episode_45.hdf5", 0.010485),
        ("episode_6.hdf5", 0.010558),
        ("episode_41.hdf5", 0.010697),
        ("episode_9.hdf5", 0.010750),
    ],
    "Pour-Balls": [
        ("episode_55.hdf5", 0.005932),
        ("episode_52.hdf5", 0.006337),
        ("episode_63.hdf5", 0.006806),
        ("episode_66.hdf5", 0.006951),
        ("episode_65.hdf5", 0.007088),
        ("episode_54.hdf5", 0.007339),
        ("episode_61.hdf5", 0.007532),
        ("episode_74.hdf5", 0.007791),
        ("episode_60.hdf5", 0.008137),
        ("episode_64.hdf5", 0.008280),
    ],
    "Push-Box": [
        ("episode_50.hdf5", 0.010576),
        ("episode_51.hdf5", 0.011270),
        ("episode_52.hdf5", 0.011941),
        ("episode_98.hdf5", 0.012541),
        ("episode_66.hdf5", 0.012898),
        ("episode_70.hdf5", 0.012963),
        ("episode_53.hdf5", 0.013595),
        ("episode_92.hdf5", 0.013633),
        ("episode_97.hdf5", 0.013849),
        ("episode_6.hdf5", 0.013854),
    ],
    "Sort-Cans": [
        ("episode_1.hdf5", 0.011536),
        ("episode_58.hdf5", 0.012234),
        ("episode_28.hdf5", 0.012709),
        ("episode_16.hdf5", 0.012919),
        ("episode_30.hdf5", 0.012944),
        ("episode_21.hdf5", 0.013026),
        ("episode_29.hdf5", 0.013076),
        ("episode_47.hdf5", 0.013148),
        ("episode_27.hdf5", 0.013332),
        ("episode_26.hdf5", 0.013394),
    ],
    "Stack-Can": [
        ("episode_55.hdf5", 0.006748),
        ("episode_32.hdf5", 0.007932),
        ("episode_43.hdf5", 0.007939),
        ("episode_44.hdf5", 0.007952),
        ("episode_91.hdf5", 0.008753),
        ("episode_20.hdf5", 0.008928),
        ("episode_47.hdf5", 0.009045),
        ("episode_22.hdf5", 0.009379),
        ("episode_12.hdf5", 0.009754),
        ("episode_23.hdf5", 0.009801),
    ],
    "Stack-Can-Into-Drawer": [
        ("episode_28.hdf5", 0.037548),
        ("episode_39.hdf5", 0.042628),
        ("episode_38.hdf5", 0.051440),
        ("episode_29.hdf5", 0.052506),
        ("episode_43.hdf5", 0.056320),
        ("episode_30.hdf5", 0.061978),
        ("episode_35.hdf5", 0.062655),
        ("episode_15.hdf5", 0.064243),
        ("episode_36.hdf5", 0.067350),
        ("episode_13.hdf5", 0.068193),
    ],
    "Unload-Cans": [
        ("episode_47.hdf5", 0.003218),
        ("episode_37.hdf5", 0.004854),
        ("episode_24.hdf5", 0.005575),
        ("episode_30.hdf5", 0.006169),
        ("episode_29.hdf5", 0.006214),
        ("episode_39.hdf5", 0.006806),
        ("episode_23.hdf5", 0.006823),
        ("episode_2.hdf5", 0.006844),
        ("episode_32.hdf5", 0.006960),
        ("episode_33.hdf5", 0.007112),
    ],
}

LANGUAGE_MAPPING = {
  "Pour-Balls": "pour balls in cup into bowl",
  "Pick-Place-Safe-Drawer": "Pick up object, place it in drawer and close drawer",
  "Stack-Cubes": "stack one cube on another cube",
  "Dump-Apple": "dump aplle into bowl",
  "Push-Box": "push box to the marker",
  "Sort-Cans": "Put sprite cans to the left box, and orange cans to the right box",
  "Sort-Cans-Old": "Put sprite cans to the left box, and orange cans to the right box",
  "Orient-Cube": "Orient the cube to the rotation as show in the observation",
  "Insert-Cans": "Insert cans into the boxes",
  "Close-Drawer": "Close the opened drawer",
  "Open-Drawer": "Open the closed drawer",
  "Stack-Cubes-From-Drawer": "Open the closed drawer, and stack on cube on another cube on the desk",
  "Insert-And-Unload-Cans": "Insert the left can into the slot and insert the right can into the slot, unload the left cans andd then unload the right cans",
  "Insert-And-Unload-Cans-Old": "Insert the right can into the slot and insert the left can into the slot, unload the right cans andd then unload the left cans",
  "Orient-Pour-Balls": "Reorient the mug and pour balls in cup into bowl",
  "Press-Gamepad-Blue": "Press the blue button on the gamepad",
  "Press-Gamepad-Red": "Press the red button on the gamepad",
  "Press-Gamepad-Blue-Red": "Press the blue button on the gamepad then press the red button",
  "Flip-Mug": "Flip the mug",
  "Unload-Cans": "unload the right cans and then unload the left cans",
  "Press-Gamepad-Red-Blue": "Press the red button on the gamepad then press the blue button",
  "Stack-Single-Cube": "stack right cube on the middle cube",
  "Stack-Single-Cube-From-Drawer": "Open the closed drawer, and stack the cube on the cube in front of the drawer",
  "Stack-Can": "put can on the saucer",
  "Stack-Can-From-Drawer": "Put can on the saucer, and Close the drawer",
  "Stack-Can-Into-Drawer": "Open the drawer, and Put can on the saucer",
  "Open-Laptop": "open the laptop",
}

TASK_LANG_MAP_OPTIMIZED = {
    "Close-Drawer": "Extend the left hand behind the opened wooden door, and push it in an arc motion inward until the door is fully closed and flush against the cabinet structure.",
    "Flip-Mug": "Reach out with the right hand, grab the overturned mug, gently lift it, give it a twist to right it, and then set it back down steadily.",
    "Insert-And-Unload-Cans": "First, pick up the left can and insert it into the left slot, then pick up the right can and insert it into the right slot. Next, reverse the action by unloading the left can from its slot.",
    "Insert-Cans/Left_Right_Deprecated": "Use the left hand to pick up the left can and insert it into the left slot, then use the other hand to pick up the right can and insert it into the right slot.",
    "Insert-Cans": "Use the right hand to pick up the right can and insert it into the right slot, then use the left hand to pick up the left can and insert it into the left slot.",
    "Open-Drawer": "Approach the closed wooden door, firmly grasp its vertical handle, and pull it outward in an arc motion to swing the door open.",
    "Open-Laptop": "Position the right hand at the edge of the laptop screen lid, and lift it upward and backward until the screen is fully open at a viewable angle.",
    "Pour-Balls": "Use the right hand to grasp the cup containing the balls, lift it above the target bowl, and tilt the cup until all the balls pour out completely into the bowl.",
    "Push-Box": "Place the hand against the side of the box and apply steady pressure, sliding the box across the surface until it perfectly aligns with the marker on the table.",
    "Sort-Cans/Single_hand_first_deprecated": "Identify and pick up the Sprite cans to place them into the left box, then pick up the orange soda cans and place them into the right box for sorting.",
    "Sort-Cans": "Sort the cans by picking up the green Sprite cans and placing them into the left box, and picking up the orange cans to place them into the right box.",
    "Stack-Can-Into-Drawer": "Use the right hand to pick up the can and place it carefully onto the saucer in the open cabinet. then use the left hand to push the cabinet door inward until it is fully closed.",
    "Stack-Can": "Use the right hand to pick up the designated can, move it above the saucer, and lower it carefully to stack it right in the center of the saucer.",
    "Unload-Cans": "Reach into the slots to remove the can on the right side first, and then proceed to remove the remaining can from the left side.",
}

def ik_step(
    env,
    left_ik_controller,
    right_ik_controller,

    left_ik_commands_world,
    right_ik_commands_world,

    left_ik_commands_robot,
    right_ik_commands_robot,

    left_ee_goal, right_ee_goal,
    left_hand_dof, right_hand_dof,

    action
):
  left_jacobin_idx = env.left_ee_idx-1
  right_jacobin_idx = env.right_ee_idx-1

  robot_pose_w = env.robot.data.root_state_w[:, 0:7]
  left_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, left_jacobin_idx, :, env.cfg.left_arm_cfg.joint_ids]
  left_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.left_arm_cfg.body_ids[0], 0:7]
  left_joint_pos = env.robot.data.joint_pos[:, env.cfg.left_arm_cfg.joint_ids]
  right_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, right_jacobin_idx, :, env.cfg.right_arm_cfg.joint_ids]
  right_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.right_arm_cfg.body_ids[0], 0:7]
  right_joint_pos = env.robot.data.joint_pos[:, env.cfg.right_arm_cfg.joint_ids]
  # prepare IK
  left_ee_curr_pose_robot, left_ee_curr_quat_robot = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ee_curr_pose_world[:, 0:3], left_ee_curr_pose_world[:, 3:7]
  )
  right_ee_curr_pos_robot, right_ee_curr_quat_robot = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ee_curr_pose_world[:, 0:3], right_ee_curr_pose_world[:, 3:7]
  )
  left_ik_commands_world[:, 0:7] = torch.FloatTensor(left_ee_goal[0:7]).to(left_ik_commands_world.device)
  left_ik_commands_robot[:, 0:3], left_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ik_commands_world[:, 0:3], left_ik_commands_world[:, 3:7]
  )
  right_ik_commands_world[:, 0:7] = torch.FloatTensor(right_ee_goal[0:7]).to(right_ik_commands_world.device)
  right_ik_commands_robot[:, 0:3], right_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ik_commands_world[:, 0:3], right_ik_commands_world[:, 3:7]
  )
  left_ik_controller.set_command(left_ik_commands_robot, left_ee_curr_pose_robot, left_ee_curr_quat_robot)
  right_ik_controller.set_command(right_ik_commands_robot, right_ee_curr_pos_robot, right_ee_curr_quat_robot)
  # compute the joint commands
  left_joint_pos_des = left_ik_controller.compute(left_ee_curr_pose_robot, left_ee_curr_quat_robot, left_arm_jacobian, left_joint_pos)
  right_joint_pos_des = right_ik_controller.compute(right_ee_curr_pos_robot, right_ee_curr_quat_robot, right_arm_jacobian, right_joint_pos)

  action[:, :] = 0
  action[:, env.cfg.left_arm_cfg.joint_ids] = left_joint_pos_des
  action[:, env.cfg.right_arm_cfg.joint_ids] = right_joint_pos_des

  action[:, env.cfg.left_hand_cfg.joint_ids] =  torch.FloatTensor(left_hand_dof).to(action.device)# set hand joint commands
  action[:, env.cfg.right_hand_cfg.joint_ids] =  torch.FloatTensor(right_hand_dof).to(action.device)# set hand joint commands

def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]

def build_prompt(task: str, template: str, empty_task_name: str, task_lang_map: bool = False) -> str:
    if task_lang_map:
        # task_text = LANGUAGE_MAPPING[str(task).strip()]
        task_text = TASK_LANG_MAP_OPTIMIZED[str(task).strip()]
    else:
        task_text = str(task).strip() or empty_task_name
    return template.format(task=task_text)

def _state_dict(payload: dict) -> dict:
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state dict or model state dict")
    return state

def denormalize_action(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    y = x
    y = y * std + mean
    return y


def restore_action_dimensions(
    reduced_action: torch.Tensor,
    *,
    retained_indices: torch.Tensor,
    full_action_mean: torch.Tensor,
) -> torch.Tensor:
    """Scatter a reduced Action prediction back to the environment layout."""
    if reduced_action.shape[-1] != retained_indices.numel():
        raise ValueError(
            "Reduced Action dimension does not match retained indices: "
            f"{reduced_action.shape[-1]} vs {retained_indices.numel()}"
        )
    full_shape = (*reduced_action.shape[:-1], full_action_mean.numel())
    full_action = full_action_mean.to(reduced_action).expand(full_shape).clone()
    full_action[..., retained_indices] = reduced_action
    return full_action


def ensemble_action_chunks(
    action_chunks: list[tuple[int, torch.Tensor]],
    *,
    control_step: int,
    weight_decay: float,
) -> torch.Tensor:
    """Combine time-aligned absolute-qpos predictions for one control step."""
    candidates = []
    weights = []
    newest_chunk_index = len(action_chunks) - 1
    for chunk_index, (start_step, chunk) in enumerate(action_chunks):
        # Frame zero is the qpos at inference time; execution starts at frame one.
        action_index = int(control_step) - int(start_step) + 1
        if not 0 <= action_index < chunk.shape[0]:
            continue
        chunk_age = newest_chunk_index - chunk_index
        candidates.append(chunk[action_index])
        weights.append(np.exp(-float(weight_decay) * chunk_age))
    if not candidates:
        raise RuntimeError(f"No action chunk covers control step {control_step}")
    weight_tensor = candidates[0].new_tensor(weights)
    stacked = torch.stack(candidates, dim=0)
    return (stacked * weight_tensor[:, None]).sum(dim=0) / weight_tensor.sum()


def autocast_for_tensor(tensor: torch.Tensor):
    enabled = tensor.is_cuda and tensor.dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=tensor.dtype, enabled=enabled)

def load_skel_wam_model(
    checkpoint: str,
    *,
    args: argparse.Namespace,
    action_dim: int,
    video_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SkelWamBackboneModel, dict]:
    backbone_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    backbone_args = backbone_payload.get("args", {}) or {}
    if not isinstance(backbone_args, dict):
        raise TypeError("MoT backbone checkpoint args must be a dict")
    if not bool(backbone_args.get("use_keypoint", True)):
        raise ValueError(
            "The w/o-overlay evaluator requires use_keypoint=True; the given "
            "checkpoint is a w/o-keypoint checkpoint."
        )
    backbone_state = _state_dict(backbone_payload)

    action_payload = (
        torch.load(
            args.action_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if args.action_checkpoint
        else None
    )
    action_state = (
        _state_dict(action_payload)
        if action_payload is not None
        else None
    )
    action_args = (
        action_payload.get("args", {}) or {}
        if action_payload is not None
        else backbone_args
    )
    if not isinstance(action_args, dict):
        raise TypeError("Action checkpoint args must be a dict")
    if str(action_args.get("action_condition_mode", "latent_keypoint")) != "latent_keypoint":
        raise ValueError(
            "The w/o-overlay evaluator requires an Action head conditioned on "
            "latent_keypoint."
        )
    checkpoint_exclude_indices = action_args.get("action_exclude_indices")
    if checkpoint_exclude_indices is not None:
        checkpoint_exclude_indices = sorted(
            int(index) for index in checkpoint_exclude_indices
        )
        eval_exclude_indices = sorted(
            int(index) for index in (args.action_exclude_indices or [])
        )
        if checkpoint_exclude_indices != eval_exclude_indices:
            raise ValueError(
                "Action excluded dimensions do not match the checkpoint: "
                f"checkpoint={checkpoint_exclude_indices}, "
                f"eval={eval_exclude_indices}"
            )
    validate_fixed_training_settings(backbone_args, context="Backbone checkpoint")
    validate_fixed_training_settings(action_args, context="Action checkpoint")

    if backbone_args.get("dataset_type") == "cotrain":
        cotrain_payload = backbone_args.get("cotrain_sources", {})
        source_configs = (
            cotrain_payload.get("sources", [])
            if isinstance(cotrain_payload, dict)
            else []
        )
        source_frame_counts = set()
        for source in source_configs:
            source_kwargs = source.get("kwargs", {}) if isinstance(source, dict) else {}
            if not isinstance(source_kwargs, dict):
                continue
            source_num_frames = int(source_kwargs.get("num_frames", 0))
            source_ratio = int(source_kwargs.get("action_video_freq_ratio", 1))
            if source_num_frames > 0 and source_ratio > 0:
                source_frame_counts.add(len(range(0, source_num_frames, source_ratio)))
        if len(source_frame_counts) > 1:
            raise ValueError(
                "Cotrain backbone must describe one common Video/Keypoint frame "
                f"count, got {sorted(source_frame_counts)}"
            )
        # Older cotrain checkpoints store the source config beside the
        # checkpoint rather than inside checkpoint args. In that case the
        # top-level num_frames=49 is an unused parser default, so the explicit
        # eval geometry is the only valid value.
        checkpoint_video_frames = (
            source_frame_counts.pop()
            if source_frame_counts
            else int(video_frames)
        )
    else:
        checkpoint_video_frames = len(
            range(
                0,
                int(backbone_args.get("num_frames", args.num_frames)),
                int(
                    backbone_args.get(
                        "action_video_freq_ratio",
                        args.action_video_freq_ratio,
                    )
                ),
            )
        )
    if checkpoint_video_frames != int(video_frames):
        raise ValueError(
            "Video/keypoint frame mismatch between MoT checkpoint and eval: "
            f"{checkpoint_video_frames} vs {video_frames}"
        )
    effective_action_state = action_state or backbone_state
    action_embedder = effective_action_state.get(
        "action_fm_head.action_embedder.weight"
    )
    if not isinstance(action_embedder, torch.Tensor):
        source_label = "--action_checkpoint" if args.action_checkpoint else "--checkpoint"
        raise RuntimeError(f"{source_label} does not contain an Action FM head")
    checkpoint_action_dim = int(action_embedder.shape[1])
    if checkpoint_action_dim != int(action_dim):
        raise ValueError(
            "Action dimension mismatch between MoT checkpoint and environment: "
            f"{checkpoint_action_dim} vs {action_dim}"
        )
    action_frames = int(
        action_args.get(
            "action_target_frames",
            args.action_target_frames or args.num_frames,
        )
    )
    if (
        args.action_target_frames is not None
        and action_frames != int(args.action_target_frames)
    ):
        raise ValueError(
            "Action frame mismatch between checkpoint and eval: "
            f"{action_frames} vs {int(args.action_target_frames)}"
        )
    model = SkelWamBackboneModel(
        transformer_path=args.vae_checkpoint_dir,
        keypoint_frames=video_frames,
        keypoint_dims=int(backbone_args.get("keypoint_dims", args.keypoint_dims)),
        keypoint_hidden_dim=int(backbone_args.get("keypoint_hidden_dim", 1024)),
        keypoint_ffn_dim=int(backbone_args.get("keypoint_ffn_dim", 4096)),
        keypoint_expert_checkpoint=None,
        keypoint_cond_video_mode=str(
            backbone_args.get("keypoint_cond_video_mode", "full")
        ),
        keypoint_use_text_cross_attention=bool(
            backbone_args.get("keypoint_use_text_cross_attention", False)
        ),

        action_frames=action_frames,
        action_dim=checkpoint_action_dim,
        predict_action=True,
        action_patch_embedding_mode=infer_action_patch_embedding_mode(backbone_state, action_state=action_state),
        action_condition_mode=str(
            action_args.get("action_condition_mode", "latent_keypoint")
        ),
        torch_dtype=dtype,
        device=device,
        use_gradient_checkpointing=False,
        load_pretrained_transformer=False,
    )
    load_component_state_dict(
        model,
        backbone_state,
        action_state=action_state,
    )
    runtime_args = dict(backbone_args)
    for name in (
        "action_target_frames",
        "action_condition_mode",
        "action_snr_shift",
    ):
        if name in action_args:
            runtime_args[name] = action_args[name]
    del backbone_payload, backbone_state, action_payload, action_state
    model.eval().requires_grad_(False)
    return model, runtime_args


@torch.no_grad()
def encode_video_latents(vae: Wan2_2_VAE, video: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latents = vae.encode([video.to(device=device, dtype=dtype)])
    if latents is None:
        raise RuntimeError("Wan VAE encode returned None")
    return latents[0].unsqueeze(0).to(device=device, dtype=dtype)


@torch.no_grad()
def sample_video_mot(
    model: SkelWamBackboneModel,
    *,
    condition_latents: torch.Tensor,
    current_keypoints: torch.Tensor,
    history_keypoints: torch.Tensor,
    current_keypoints_valid: torch.Tensor,
    history_keypoints_valid: torch.Tensor,
    text_emb: list[torch.Tensor],
    num_inference_steps: int,
    snr_shift: float,
) -> torch.Tensor:
    scheduler = FlowMatchingScheduler(
        num_inference_steps=num_inference_steps,
        snr_shift=snr_shift,
    )
    scheduler.set_timesteps(
        num_inference_steps,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    latents = torch.randn_like(condition_latents)
    latents[:, :, 0:1] = condition_latents[:, :, 0:1]
    with autocast_for_tensor(condition_latents):
        history_cache = model.prefill_history_keypoint_cache(
            history_keypoints=history_keypoints,
            current_keypoints=current_keypoints,
            text_emb=text_emb,
            history_keypoints_valid=history_keypoints_valid,
            current_keypoints_valid=current_keypoints_valid,
        )
    for timestep in scheduler.timesteps.to(
        condition_latents.device, condition_latents.dtype
    ):
        with autocast_for_tensor(latents):
            velocity = model.forward_latent_step(
                noisy_latents=latents,
                condition_latents=condition_latents,
                latent_timestep=timestep.expand(latents.shape[0]),
                text_emb=text_emb,
                history_keypoint_cache=history_cache,
            )["latent_velocity"]
        latents = scheduler.step(velocity, timestep, latents)
        latents[:, :, 0:1] = condition_latents[:, :, 0:1]
    return latents


@torch.no_grad()
def sample_keypoint_mot(
    model: SkelWamBackboneModel,
    *,
    condition_latents: torch.Tensor,
    current_keypoints: torch.Tensor,
    current_keypoints_valid: torch.Tensor,
    text_emb: list[torch.Tensor],
    num_inference_steps: int,
    snr_shift: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler = FlowMatchingScheduler(
        num_inference_steps=num_inference_steps,
        snr_shift=snr_shift,
    )
    scheduler.set_timesteps(
        num_inference_steps,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    expert = model.keypoint_expert
    state = torch.randn(
        condition_latents.shape[0],
        model.keypoint_frames,
        expert.num_hands,
        expert.num_keypoints,
        expert.keypoint_dims + 1,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    current_state = torch.cat(
        [
            current_keypoints,
            normalize_keypoint_validity(
                current_keypoints_valid,
                dtype=current_keypoints.dtype,

            ).unsqueeze(-1),
        ],
        dim=-1,
    )
    state[:, 0:1] = current_state
    with autocast_for_tensor(condition_latents):
        video_cache = model.prefill_condition_video_cache(
            condition_latents=condition_latents,
            text_emb=text_emb,
        )
    for timestep in scheduler.timesteps.to(
        condition_latents.device, condition_latents.dtype
    ):
        with autocast_for_tensor(state):
            output = model.forward_keypoint_step(
                noisy_keypoints=state[..., : expert.keypoint_dims],
                noisy_keypoint_validity=state[..., expert.keypoint_dims],
                current_keypoints=current_keypoints,
                current_keypoints_valid=current_keypoints_valid,
                keypoint_timestep=timestep.expand(state.shape[0]),
                text_emb=text_emb,
                condition_video_cache=video_cache,
            )
        state = scheduler.step(
            output["keypoint_flow_velocity"], timestep, state
        )
        state[:, 0:1] = current_state
    validity = decode_keypoint_validity(
        state[..., expert.keypoint_dims],

    )
    validity[:, 0:1] = current_keypoints_valid
    return state[..., : expert.keypoint_dims], validity


@torch.no_grad()
def sample_action_mot(
    model: SkelWamBackboneModel,
    *,
    condition_latents: torch.Tensor,
    condition_keypoints: torch.Tensor | None,
    condition_keypoints_valid: torch.Tensor | None,
    num_inference_steps: int,
    snr_shift: float,
) -> torch.Tensor:
    scheduler = FlowMatchingScheduler(
        num_inference_steps=num_inference_steps,
        snr_shift=snr_shift,
    )
    scheduler.set_timesteps(
        num_inference_steps,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    action_sample = torch.randn(
        condition_latents.shape[0],
        model.action_frames,
        model.action_dim,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    for timestep in scheduler.timesteps.to(
        condition_latents.device, condition_latents.dtype
    ):
        with autocast_for_tensor(action_sample):
            velocity = model.forward_action_fm(
                condition_latents=condition_latents,
                condition_keypoints=condition_keypoints,
                condition_keypoints_valid=condition_keypoints_valid,
                noisy_action=action_sample,
                action_timestep=timestep.expand(action_sample.shape[0]),
            )["action_velocity"]
        action_sample = scheduler.step(
            velocity, timestep, action_sample
        )
    return action_sample


def normalize_keypoints(
    raw_keypoints: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    keypoints = torch.from_numpy(raw_keypoints[..., : mean.numel()]).to(device=device, dtype=torch.float32)
    valid = (keypoints != -1).all(dim=-1) & torch.isfinite(keypoints).all(dim=-1)
    normalized = (keypoints - mean.to(device=device)) / std.to(device=device)
    keypoints = torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    return keypoints.to(dtype=dtype), valid


def denormalize_keypoints(
    keypoints: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    dims = keypoints.shape[-1]
    if mean.numel() < dims or std.numel() < dims:
        raise ValueError(
            f"Keypoint stats have fewer dimensions than predictions: "
            f"pred={dims}, mean={mean.numel()}, std={std.numel()}"
        )
    mean = mean[:dims].to(device=keypoints.device, dtype=torch.float32)
    std = std[:dims].to(device=keypoints.device, dtype=torch.float32)
    raw = keypoints.detach().to(dtype=torch.float32) * std + mean
    return raw.cpu().numpy()


def overlay_predicted_keypoints(
    rgb: np.ndarray,
    keypoints: np.ndarray,
    valid: np.ndarray,
    overlay: HandSkeletonOverlay,
) -> np.ndarray:
    """Draw raw SOMA keypoints [2, 21, D] in right/left order on an RGB frame."""
    img_rgb = np.asarray(rgb)
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {img_rgb.shape}")

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[:2] != (2, 21) or keypoints.shape[2] < 2:
        raise ValueError(f"Expected keypoints with shape [2, 21, D>=2], got {keypoints.shape}")
    valid = np.asarray(valid, dtype=np.bool_)
    if valid.shape != (2, 21):
        raise ValueError(f"Expected valid mask with shape [2, 21], got {valid.shape}")


    img_bgr = img_rgb[..., ::-1].copy()
    img_h, img_w = img_bgr.shape[:2]
    projected: dict[str, tuple[int, int, float]] = {}
    for hand_idx, offset in ((0, 0), (1, 21)):
        for joint_idx, name in enumerate(overlay.point_names[offset : offset + 21]):
            if not valid[hand_idx, joint_idx]:
                continue
            point = keypoints[hand_idx, joint_idx]
            x_norm, y_norm = float(point[0]), float(point[1])
            z = float(point[2]) if point.shape[0] >= 3 else 0.0
            if not np.isfinite(x_norm) or not np.isfinite(y_norm) or not np.isfinite(z):
                continue
            if not (0.0 <= x_norm < 1.0 and 0.0 <= y_norm < 1.0):
                continue
            if point.shape[0] >= 3 and z <= 0.0:
                continue
            projected[name] = (
                int(round(x_norm * (img_w - 1))),
                int(round(y_norm * (img_h - 1))),
                z,
            )

    line_thickness = int(overlay.line_style.get("thickness", 1))
    for side_key in ("left", "right"):
        for a, b in (overlay.edges.get(side_key, []) or []):
            if a not in projected or b not in projected:
                continue
            line_name = b if overlay._get_point_finger(a) == "palm" else a
            cv2.line(
                img_bgr,
                projected[a][:2],
                projected[b][:2],
                overlay._get_point_color_bgr(line_name),
                line_thickness,
            )

    point_radius = int(overlay.point_style.get("radius", 2))
    point_thickness = int(overlay.point_style.get("thickness", -1))
    for name, (u, v, _z) in projected.items():
        cv2.circle(
            img_bgr,
            (u, v),
            point_radius,
            overlay._get_point_color_bgr(name),
            point_thickness,
        )
    return img_bgr[..., ::-1].copy()


def make_history_keypoints(
    keypoint_history: list[np.ndarray],
    offsets: list[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not keypoint_history:
        raise RuntimeError("No keypoint history is available for Skel-WAM inference")
    frames = []
    valid_frames = []
    n = len(keypoint_history)
    for offset in offsets:
        idx = max(0, n - 1 + int(offset))
        idx = min(idx, n - 1)
        keypoints, valid = normalize_keypoints(keypoint_history[idx], mean, std, device, dtype)
        frames.append(keypoints)
        valid_frames.append(valid)
    return (
        torch.stack(frames, dim=0).unsqueeze(0),
        torch.stack(valid_frames, dim=0).unsqueeze(0),
    )


def make_condition_video(
    rgb_obs: np.ndarray,
    video_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rgb_tensor = torch.from_numpy(rgb_obs).to(torch.float32).permute(2, 0, 1).contiguous()
    rgb_tensor = rgb_tensor / 127.5 - 1.0
    # The dataset first samples 33 raw frames to 9 aligned video/keypoint
    # frames. Wan's temporal VAE then maps those 9 video frames to 3 latents.
    video = torch.zeros(
        (
            3,
            video_frames,
            rgb_tensor.shape[-2],
            rgb_tensor.shape[-1],
        ),
        dtype=torch.float32,
    )
    video[:, 0] = rgb_tensor
    return video.to(device=device, dtype=dtype)


def tensor_video_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    video = video.detach().float().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().to(torch.uint8)
    return [video[:, i].permute(1, 2, 0).contiguous().numpy() for i in range(video.shape[1])]


@torch.no_grad()
def decode_video_latents(vae: Wan2_2_VAE, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode([latents[0].detach().float()])
    if decoded is None:
        raise RuntimeError("Wan VAE decode returned None")
    return decoded[0].detach().cpu()

def main():

    args = parser.parse_args()

    import random
    assert args.task in seed_map
    print(f"Setting seed to {seed_map[args.task]}")

    # random.seed(seed_map[args.task])

    seed = seed_map[args.task]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    randomize_idxes = list(range(10000))
    random.shuffle(randomize_idxes)

    # train data are using idxes 0-49, test start from 50

    # We used the first 100 random idxes for training
    # Starting from 101th random ides for evaluation
    curr_random_idx = 100 + (
      args.room_idx * 5 + args.table_idx
    )  * args.num_trials * args.num_episodes

    # parse configuration
    env_cfg = parse_env_cfg(
        args.task,
        num_envs=1,
    )

    env_cfg.episode_length_s = 60 # 60 seconds episode length -> For long horizon tasks
    env_cfg.randomize = True
    # create environment
    env_cfg.spawn_background =True
    # select background
    room_idx = args.room_idx
    table_idx = args.table_idx
    env_cfg.room_idx = room_idx
    env_cfg.table_idx = table_idx
    env = gym.make(
        args.task,
        cfg=env_cfg
    )
    env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
    env.reset()

    # IK controllers
    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="pinv")
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)

    # Create buffers to store actions
    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)

    save_path = os.path.join(
      args.saving_path,
      args.additional_label,
    )
    result_saving_path = os.path.join(
      save_path,
      "result_log.txt",
    )

    from pathlib import Path
    Path(save_path).mkdir(exist_ok=True, parents=True)

    import pickle
    with open("eval/egovla/init_poses_fixed_set_100traj.pkl", "rb") as f:
       init_poses = pickle.load(f)

    task_name = args.task[9:-3]
    load_name = task_name

    # Collect Initial Hand and EE poses from data -> Only set the arm and hand for start
    episode_list = TASK_INIT_EPISODE[task_name][:args.num_episodes]

    padding = 0

    hand_overlay = None
    if args.overlay_hand_skeleton:
      raise ValueError(
          "The w/o-overlay evaluator refuses --overlay_hand_skeleton; use "
          "--extract_hand_keypoints instead."
      )
    if args.extract_hand_keypoints:
      if args.skeleton_cfg is None:
        raise ValueError("--skeleton_cfg is required with --extract_hand_keypoints.")
      hand_overlay = HandSkeletonOverlay(
        use_fk_hand=args.use_fk_hand,
        joint_names_file=args.joint_names_file,
        usd_path=args.usd_path,
        skeleton_cfg=args.skeleton_cfg,
        ee_quat_order=args.ee_quat_order,
      )
    if hand_overlay is None:
      raise ValueError(
          "The w/o-overlay model still uses numerical keypoints. Pass "
          "--extract_hand_keypoints to initialize the keypoint extractor; "
          "camera pixels remain unmodified in this evaluator."
      )

    ######## load model ########
    print(f"========== Loading Skel-WAM MoT model ==========")
    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    video_keypoint_frames = int((args.num_frames - 1) / args.action_video_freq_ratio + 1)
    if video_keypoint_frames <= 0:
        raise ValueError(f"Invalid video/keypoint frame count from num_frames={args.num_frames}")

    vae = Wan2_2_VAE(
        vae_pth=str(Path(args.vae_checkpoint_dir) / "Wan2.2_VAE.pth"),
        dtype=torch_dtype,
        device=str(device),
    )
    action_exclude_indices = tuple(
        sorted(int(index) for index in (args.action_exclude_indices or ()))
    )
    if len(set(action_exclude_indices)) != len(action_exclude_indices):
        raise ValueError(
            f"--action_exclude_indices contains duplicates: {action_exclude_indices}"
        )
    invalid_action_indices = [
        index
        for index in action_exclude_indices
        if index < 0 or index >= env.num_actions
    ]
    if invalid_action_indices:
        raise ValueError(
            "--action_exclude_indices must be in "
            f"[0, {env.num_actions - 1}], got {invalid_action_indices}"
        )
    action_exclude_set = set(action_exclude_indices)
    action_retained_indices = tuple(
        index for index in range(env.num_actions) if index not in action_exclude_set
    )
    action_retained_index_tensor = torch.tensor(
        action_retained_indices, dtype=torch.long, device=device
    )

    skel_model, checkpoint_args = load_skel_wam_model(
        args.checkpoint,
        args=args,
        action_dim=len(action_retained_indices),
        video_frames=video_keypoint_frames,
        device=device,
        dtype=torch_dtype,
    )

    prompt_template = args.prompt_template or checkpoint_args.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    empty_task_name = args.empty_task_name or checkpoint_args.get("empty_task_name", "robot manipulation")
    latent_snr_shift = args.latent_snr_shift if args.latent_snr_shift is not None else float(checkpoint_args.get("latent_snr_shift", 1.0))
    keypoint_snr_shift = args.keypoint_snr_shift if args.keypoint_snr_shift is not None else float(checkpoint_args.get("keypoint_snr_shift", 1.0))
    action_snr_shift = (
        args.action_snr_shift
        if args.action_snr_shift is not None
        else float(checkpoint_args.get("action_snr_shift", 1.0))
    )
    latent_num_inference_steps = args.latent_num_inference_steps if args.latent_num_inference_steps is not None else args.num_inference_steps
    keypoint_num_inference_steps = args.keypoint_num_inference_steps if args.keypoint_num_inference_steps is not None else args.num_inference_steps
    action_num_inference_steps = (
        args.action_num_inference_steps
        if args.action_num_inference_steps is not None
        else args.num_inference_steps
    )

    wan_checkpoint_dir = Path(args.vae_checkpoint_dir)
    prompt = build_prompt(task_name, prompt_template, empty_task_name, task_lang_map=True)
    if args.precomputed_text_embeddings_path:
        prompt_embeddings, _ = load_precomputed_text_embeddings(
            args.precomputed_text_embeddings_path
        )
        prompt_ids = PrecomputedPromptEmbeddingStore(
            prompt_embeddings,
            expected_text_dim=4096,
        ).encode(
            [prompt],
            output_device=device,
            output_dtype=torch_dtype,
        )
    else:
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(wan_checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(wan_checkpoint_dir / "google/umt5-xxl"),
            shard_fn=None,
        )
        text_encoder.model.eval().requires_grad_(False)
        text_encoder.model.to(device)
        with torch.no_grad():
            prompt_ids = text_encoder([prompt], device)
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    keypoint_stats_path = Path(args.keypoint_stats_path).expanduser().resolve()
    if not keypoint_stats_path.is_file():
        raise FileNotFoundError(f"Keypoint stats file does not exist: {keypoint_stats_path}")
    keypoint_stats = json.loads(keypoint_stats_path.read_text(encoding="utf-8"))
    if not isinstance(keypoint_stats, dict):
        raise TypeError("Keypoint stats must be a JSON object")
    expected_normalization = "xy_unit_to_minus_one_one_shared_zscore"
    stats_use_new_logic = (
        keypoint_stats.get("normalization") == expected_normalization
    )
    if stats_use_new_logic:
        raise ValueError(
            "The checkpoint uses legacy keypoint normalization but eval was "
            f"given new-normalization stats: {keypoint_stats_path}. Use the "
            "legacy EgoVLA stats or select a checkpoint trained with "
            "new_normalize_logic."
        )
    kp_mean = torch.tensor(keypoint_stats.get("mean"), dtype=torch.float32)
    kp_std = torch.tensor(keypoint_stats.get("std"), dtype=torch.float32)
    checkpoint_keypoint_dims = skel_model.keypoint_expert.keypoint_dims
    if (
        kp_mean.numel() < checkpoint_keypoint_dims
        or kp_std.numel() < checkpoint_keypoint_dims
    ):
        raise ValueError(
            "Keypoint stats have fewer dimensions than the checkpoint: "
            f"mean={kp_mean.numel()}, std={kp_std.numel()}, "
            f"checkpoint={checkpoint_keypoint_dims}"
        )

    action_stats_path = Path(args.action_stats_path).expanduser().resolve()
    if not action_stats_path.is_file():
        raise FileNotFoundError(
            f"Action stats file does not exist: {action_stats_path}"
        )
    action_stats = json.loads(action_stats_path.read_text(encoding="utf-8"))
    full_action_mean = torch.tensor(
        action_stats.get("mean"), dtype=torch.float32, device=device
    )
    full_action_std = torch.tensor(
        action_stats.get("std"), dtype=torch.float32, device=device
    )
    if (
        full_action_mean.numel() != env.num_actions
        or full_action_std.numel() != env.num_actions
    ):
        raise ValueError(
            "Action stats dimension mismatch: "
            f"mean={full_action_mean.numel()}, std={full_action_std.numel()}, "
            f"environment={env.num_actions}"
        )
    action_mean = full_action_mean[action_retained_index_tensor]
    action_std = full_action_std[action_retained_index_tensor]
    print(f"Loaded Skel-WAM MoT checkpoint: {args.checkpoint}")
    print(f"frames: action/raw={args.num_frames}, video/keypoint={video_keypoint_frames}, ratio={args.action_video_freq_ratio}")
    print(
        "latent/keypoint/action inference steps="
        f"{latent_num_inference_steps}/{keypoint_num_inference_steps}/"
        f"{action_num_inference_steps}"
    )
    print(
        f"action_condition_mode={skel_model.action_condition_mode}, "
        f"keypoint_cond_video_mode={skel_model.keypoint_cond_video_mode}, "
        "validity=joint_flow_matching, "
        "normalization=zscore"
    )
    print(
        f"action_chunk_weighting={args.action_chunk_weighting}, "
        f"decay={args.action_chunk_weight_decay}, "
        f"max_history={args.action_chunk_max_history}"
    )
    print(
        f"action dimensions: model={len(action_retained_indices)}, "
        f"environment={env.num_actions}, excluded={list(action_exclude_indices)}"
    )
    print(f"========== Skel-WAM MoT model Loaded==========")

    ######## load model ########

    exec_action_steps = args.exec_action_steps

    print(f"========== Finished initialization ==========")

    # with torch.inference_mode():
    for episode_idx in episode_list:
      for trial_idx in range(args.num_trials):
        # seq_name = f"episode_{episode_idx}.hdf5"
        seq_name = episode_idx[0]

        # 30 Hz
        # original video is 15fps and env is 30 fps

        seq_save_path = os.path.join(
          save_path,
          f"{task_name}-room_{room_idx}-table_{table_idx}"
        )
        from pathlib import Path
        Path(seq_save_path).mkdir(exist_ok=True, parents=True)
        output_path = os.path.join(
          seq_save_path,
          f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}.mp4"
        )
        pred_path = os.path.join(
          seq_save_path,
          f"video_pred_{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}.mp4"
        )
        if args.save_frames:
          frames_output_path = os.path.join(
            seq_save_path,
            f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}"
          )
          Path(frames_output_path).mkdir(exist_ok=True, parents=True)
        fps = 30
        writer = imageio.get_writer(output_path, fps=fps)
        pred_writer = imageio.get_writer(pred_path, fps=fps / float(args.action_video_freq_ratio))

        print(f"========== Start reset ==========")
        # reset
        if True:
          curr_random_idx += 1
          env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
          env_results = env.reset()
          left_ik_controller.reset()
          right_ik_controller.reset()

          left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
          right_dof = init_poses[load_name][seq_name][padding]["right_dof"]

          for idx in range(100):
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]

            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]

            left_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["left_ee"]
            right_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["right_ee"]

            ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world,
              right_ik_commands_world,

              left_ik_commands_robot,
              right_ik_commands_robot,

              left_ee_pose_traj_gt, right_ee_pose_traj_gt,
              left_dof, right_dof,
              action
            )
            env_results = env.step(action)
            # rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
            # rgb_obs = cv2.resize(rgb_obs, (256, 256))
        print(f"========== Down reset ==========")

        count = 0

        result = False
        max_horizon = TASK_MAX_HORIZON[args.task]

        exec_step = 1
        keypoint_history = []
        action_chunk_pred = None
        action_chunks = []
        for i in tqdm.tqdm(range(max_horizon)):
          # run everything in inference mode
          # obtain quantities from simulation
          rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]

          rgb_obs = cv2.resize(rgb_obs, (384, 384))

          # Ablation contract: feed the raw video_384x384 pixels to the model.
          # hand_overlay is retained only as the numerical keypoint extractor.

          # rgb_obs = cv2.resize(rgb_obs, (256, 256))

          ######## add model infer here ########
          raw_keypoints = hand_overlay.extract_keypoints(env_results[0])
          keypoint_history.append(raw_keypoints)
          if exec_step == 1:
            t_s = time.time()
            condition_video = make_condition_video(
                rgb_obs, video_keypoint_frames, device, torch_dtype
            )
            condition_latents = encode_video_latents(vae, condition_video, device, torch_dtype)
            current_keypoints, current_keypoints_valid = normalize_keypoints(
                raw_keypoints, kp_mean, kp_std, device, torch_dtype
            )
            current_keypoints = current_keypoints.unsqueeze(0).unsqueeze(0)
            current_keypoints_valid = current_keypoints_valid.unsqueeze(0).unsqueeze(0)
            history_keypoints, history_keypoints_valid = make_history_keypoints(
                keypoint_history,
                args.history_keypoint_offsets,
                kp_mean,
                kp_std,
                device,
                torch_dtype,
            )

            pred_latents = sample_video_mot(
                skel_model,
                condition_latents=condition_latents,
                current_keypoints=current_keypoints,
                history_keypoints=history_keypoints,
                current_keypoints_valid=current_keypoints_valid,
                history_keypoints_valid=history_keypoints_valid,
                text_emb=prompt_ids,
                num_inference_steps=latent_num_inference_steps,
                snr_shift=latent_snr_shift,
            )
            t_1 = time.time()
            raw_pred_keypoints, pred_keypoints_valid = sample_keypoint_mot(
                skel_model,
                condition_latents=pred_latents,
                current_keypoints=current_keypoints,
                current_keypoints_valid=current_keypoints_valid,
                text_emb=prompt_ids,
                num_inference_steps=keypoint_num_inference_steps,
                snr_shift=keypoint_snr_shift,
            )
            t_2 = time.time()
            pred_keypoints = torch.where(
                pred_keypoints_valid.unsqueeze(-1),
                raw_pred_keypoints,
                raw_pred_keypoints.new_zeros(()),
            )
            action_keypoints = pred_keypoints
            action_keypoints_valid = pred_keypoints_valid
            if skel_model.action_condition_mode == "latent_only":
                action_keypoints = None
                action_keypoints_valid = None
            pred_action = sample_action_mot(
                skel_model,
                condition_latents=pred_latents,
                condition_keypoints=action_keypoints,
                condition_keypoints_valid=action_keypoints_valid,
                num_inference_steps=action_num_inference_steps,
                snr_shift=action_snr_shift,
            )
            reduced_action_chunk_pred = denormalize_action(
                pred_action[0], action_mean, action_std
            ).to(device=device, dtype=action.dtype)
            action_chunk_pred = restore_action_dimensions(
                reduced_action_chunk_pred,
                retained_indices=action_retained_index_tensor,
                full_action_mean=full_action_mean,
            )
            if args.action_chunk_weighting == "exponential":
              action_chunks.append((i, action_chunk_pred))
              action_chunks = action_chunks[-args.action_chunk_max_history:]
            t_3 = time.time()

            try:
                pred_video = decode_video_latents(vae, pred_latents)
                pred_frames = tensor_video_to_uint8(pred_video)

                pred_frames_exec = pred_frames[: max(1, int(len(pred_frames) * args.exec_action_steps / args.num_frames))]
                for pred_frame in pred_frames_exec:
                  pred_writer.append_data(pred_frame)
            except Exception as exc:
                tqdm.tqdm.write(f"Warning: failed to decode/write predicted video: {exc}")

            tqdm.tqdm.write(
                f"generate action in {t_3 - t_s} seconds >>> "
                f"latent: {t_1 - t_s} seconds, "
                f"keypoint: {t_2 - t_1} seconds, "
                f"action: {t_3 - t_2} seconds"
            )
            tqdm.tqdm.write(f"action_chunk_pred shape: {action_chunk_pred.shape}")

          if args.action_chunk_weighting == "exponential":
            action = ensemble_action_chunks(
                action_chunks, control_step=i,
                weight_decay=args.action_chunk_weight_decay,
            ).unsqueeze(0)
          else:
            action_idx = min(exec_step, action_chunk_pred.shape[0] - 1)
            action = action_chunk_pred[action_idx, :].unsqueeze(0)

          ######## add model infer here ########
          exec_step += 1
          if exec_step >= exec_action_steps + 1:
            exec_step = 1


          env_results = env.step(action)

          # Success
          if env_results[0]["success"].sum().item() == 1:
            result = True
            break

          result_img_3d = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :].copy()
          # Save raw simulator RGB as well; no skeleton is rendered anywhere in
          # the w/o-overlay evaluation data path.

          writer.append_data(result_img_3d)

          if args.save_frames:
            cv2.imwrite(
              os.path.join(frames_output_path, f"{i}.jpg"),
              result_img_3d
            )

          count += 1

        with open(result_saving_path, "a") as f:
          f.write(f"Task: {task_name}, Room Idx: {room_idx}, Table Idx: {table_idx}, Episode Label: {episode_idx[0]}, Trial Label: {trial_idx}, Result: {result}, Steps: {count} \n")
          subtask_string = ""
          for key in env_results[0].keys():
            if "success" in key:
              subtask_string += f"{key}: {env_results[0][key].sum().item()} "
          subtask_string += "\n"
          f.write(subtask_string)

        writer.close()
        pred_writer.close()
        # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
