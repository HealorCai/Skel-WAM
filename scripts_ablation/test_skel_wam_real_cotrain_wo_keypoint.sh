#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIDEO_SUBDIR=skeleton_video_384x384
# Accept both real-data cache filename conventions.
default_text_embeddings="${project_root}/checkpoints/skel_wam/training_assets/text_embeddings.pt"
if [[ ! -f "$default_text_embeddings" && -f "${project_root}/checkpoints/skel_wam/training_assets/text_embeddings_real.pt" ]]; then
    default_text_embeddings="${project_root}/checkpoints/skel_wam/training_assets/text_embeddings_real.pt"
fi
export TEST_TEXT_EMBEDDINGS_PATH="${TEST_TEXT_EMBEDDINGS_PATH:-$default_text_embeddings}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-output/real_robot_cotrain_ablations/wo_keypoint/test}"
exec bash "${project_root}/scripts/test_skel_wam_real_cotrain.sh" --prompt_template "An egocentric video of a robot executing the following task, with a skeletal overlay on the hands: {task}" "$@"
