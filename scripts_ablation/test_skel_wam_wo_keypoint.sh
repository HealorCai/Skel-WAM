#!/usr/bin/env bash
# TASK_INCLUDE=Close-Drawer \
# SAMPLE_IDX=30 \
# bash scripts_ablation/test_skel_wam_wo_keypoint.sh
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "Set CHECKPOINT to an ablation_wo_keypoint weights checkpoint." >&2
    exit 1
fi
export TEST_DATASET_ROOT="${TEST_DATASET_ROOT:-data/EgoVLA_SIM/skeleton_video_384x384}"
export TEST_TEXT_EMBEDDINGS_PATH="${TEST_TEXT_EMBEDDINGS_PATH:-checkpoints/skel_wam/training_assets/text_embeddings_egovla.pt}"
export TEST_OUTPUT_ROOT="${TEST_OUTPUT_ROOT:-${OUTPUT_ROOT:-output/egovla_sim_ablations/wo_keypoint/test}}"
export VIS_KEYPOINT_VIDEO_ATTN=0
exec bash "${project_root}/scripts/test_skel_wam.sh" "$@"
