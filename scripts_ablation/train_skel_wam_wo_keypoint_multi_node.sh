#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ABLATION_NAME=wo_keypoint
export ABLATION_DATASET_ROOT=data/EgoVLA_SIM/skeleton_video_384x384
export ABLATION_PROMPT_TEMPLATE="An egocentric video of a robot executing the following task, with a skeletal overlay on the hands: {task}"
export ABLATION_TEXT_EMBEDDINGS_PATH=checkpoints/skel_wam/training_assets/text_embeddings_egovla.pt
export ABLATION_USE_KEYPOINT=0
export ABLATION_ACTION_CONDITION_MODE=latent_only
export ABLATION_KEYPOINT_LOSS_WEIGHT=0.0
exec bash "${project_root}/scripts_ablation/train_skel_wam_wo_overlay_multi_node.sh" "$@"
