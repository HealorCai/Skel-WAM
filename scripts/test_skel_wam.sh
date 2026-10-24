#!/usr/bin/env bash
set -euo pipefail
# CHECKPOINT=/path/to/weights.pt bash scripts/test_skel_wam.sh

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

# Keep these paths aligned with train_skel_wam_multi_node.sh.
# Use a test-specific override; DATASET_ROOT may be inherited from training.
dataset_root="${TEST_DATASET_ROOT:-data/EgoVLA_SIM/skeleton_video_384x384}"
keypoint_root="${KEYPOINT_ROOT:-data/EgoVLA_SIM/skeleton_keypoints}"
action_root="${ACTION_ROOT:-data/EgoVLA_SIM/qpos}"
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/egovla_utils/keypoint_stats_egovla_7tasks.json}"
action_stats_path="${ACTION_STATS_PATH:-utils/egovla_utils/qpos_stats_egovla_7tasks.json}"
wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
precomputed_text_embeddings_path="${TEST_TEXT_EMBEDDINGS_PATH-${TEXT_EMBEDDINGS_PATH-checkpoints/skel_wam/training_assets/text_embeddings_egovla.pt}}"

sample_idx="${SAMPLE_IDX:-0}"
task_include=${TASK_INCLUDE:-"Close-Drawer"}
task_exclude=
metadata_cache_path=
output_root="${TEST_OUTPUT_ROOT:-${OUTPUT_ROOT:-output/egovla_sim/test}}"
output_dir=${OUTPUT_DIR:-}

# Default: 37 raw/action frames, ratio 3, 13 video/keypoint frames.

checkpoint="${CHECKPOINT:?Set CHECKPOINT to an existing Skel-WAM checkpoint}"
action_checkpoint="${ACTION_CHECKPOINT:-}"
num_frames="${NUM_FRAMES:-37}"
action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO:-3}"
history_keypoint_offsets_str="${HISTORY_KEYPOINT_OFFSETS:--12 -9 -6 -3}"

global_sample_stride=1
image_size=384x384
action_target_frames=${num_frames}
action_exclude_indices_str="0 1 2 3 6 7 10 11 14 15 18 19"

dtype=bfloat16
latent_num_inference_steps=${LATENT_NUM_INFERENCE_STEPS:-10}
keypoint_num_inference_steps=${KEYPOINT_NUM_INFERENCE_STEPS:-10}
action_num_inference_steps=${ACTION_NUM_INFERENCE_STEPS:-5}
keypoint_video_condition=${KEYPOINT_VIDEO_CONDITION:-pred}  # pred: end-to-end; gt: isolate keypoint expert
action_condition=${ACTION_CONDITION:-pred}                  # pred: end-to-end; gt: isolate action head
run_action=${RUN_ACTION:-1}
vis_keypoint_video_attn=${VIS_KEYPOINT_VIDEO_ATTN:-0}
vis_action_modality_contribution=0
seed=42
fps=$((30 / action_video_freq_ratio))
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

if [ -z "${checkpoint}" ]; then
    echo "Set checkpoint to a trained Skel-WAM checkpoint." >&2
    exit 1
fi

image_size_normalized="${image_size//x/ }"
image_size_normalized="${image_size_normalized//\*/ }"
read -r image_h image_w image_extra <<< "${image_size_normalized}"
if [ -n "${image_extra:-}" ] || ! [[ "${image_h:-}" =~ ^[0-9]+$ ]] || ! [[ "${image_w:-}" =~ ^[0-9]+$ ]]; then
    echo "image_size must look like 384x384 or 384*384, got: ${image_size}" >&2
    exit 1
fi
if (( (num_frames - 1) % action_video_freq_ratio != 0 )); then
    echo "num_frames must satisfy (num_frames - 1) % action_video_freq_ratio == 0" >&2
    exit 1
fi
video_keypoint_frames=$(( (num_frames - 1) / action_video_freq_ratio + 1 ))
read -r -a history_keypoint_offsets <<< "${history_keypoint_offsets_str}"
read -r -a action_exclude_indices <<< "${action_exclude_indices_str}"

if [ -z "${output_dir}" ]; then
    timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
    checkpoint_name="$(basename "${checkpoint}")"
    run_name="$(basename "$(dirname "$(dirname "${checkpoint}")")")"
    output_dir="${output_root}/${run_name}/${checkpoint_name}_${task_include:-all}_s${sample_idx}_${timestamp}"
fi

optional_args=()
if [ -n "${action_checkpoint}" ]; then optional_args+=(--action_checkpoint "${action_checkpoint}"); fi
if [ -n "${task_include}" ]; then optional_args+=(--task_include "${task_include}"); fi
if [ -n "${task_exclude}" ]; then optional_args+=(--task_exclude "${task_exclude}"); fi
if [ -n "${metadata_cache_path}" ]; then optional_args+=(--metadata_cache_path "${metadata_cache_path}"); fi
if [ -n "${precomputed_text_embeddings_path}" ]; then
    optional_args+=(--precomputed_text_embeddings_path "${precomputed_text_embeddings_path}")
fi
if [ "${run_action}" = "1" ]; then optional_args+=(--run_action); else optional_args+=(--no-run_action); fi
if [ "${vis_keypoint_video_attn}" = "1" ]; then
    optional_args+=(--vis_keypoint_video_attn)
else
    optional_args+=(--no-vis_keypoint_video_attn)
fi
if [ "${vis_action_modality_contribution}" = "1" ]; then
    optional_args+=(--vis_action_modality_contribution)
else
    optional_args+=(--no-vis_action_modality_contribution)
fi

echo "Testing Skel-WAM on one GPU"
echo "  checkpoint=${checkpoint}"
echo "  action_checkpoint=${action_checkpoint:-embedded}"
echo "  sample_idx=${sample_idx} task_include=${task_include:-all}"
echo "  frames: raw/action=${num_frames}, video/keypoint=${video_keypoint_frames}, ratio=${action_video_freq_ratio}"
echo "  action dimensions: retained=$((50 - ${#action_exclude_indices[@]})), excluded=${action_exclude_indices[*]}"
echo "  inference steps: video=${latent_num_inference_steps}, keypoint=${keypoint_num_inference_steps}, action=${action_num_inference_steps}"
echo "  conditions: keypoint_video=${keypoint_video_condition}, action=${action_condition}"
echo "  visualize Kn->Vc attention=${vis_keypoint_video_attn}"
echo "  visualize Action modality contribution=${vis_action_modality_contribution}"
echo "  output_dir=${output_dir}"

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" python3 main/test_skel_wam.py \
    --checkpoint "${checkpoint}" \
    --dataset_root "${dataset_root}" \
    --keypoint_root "${keypoint_root}" \
    --action_root "${action_root}" \
    --keypoint_stats_path "${keypoint_stats_path}" \
    --action_stats_path "${action_stats_path}" \
    --wan_checkpoint_dir "${wan_checkpoint_dir}" \
    --sample_idx "${sample_idx}" \
    --output_dir "${output_dir}" \
    --num_frames "${num_frames}" \
    --global_sample_stride "${global_sample_stride}" \
    --action_video_freq_ratio "${action_video_freq_ratio}" \
    --image_size "${image_h}" "${image_w}" \
    --history_keypoint_offsets "${history_keypoint_offsets[@]}" \
    --action_target_frames "${action_target_frames}" \
    --action_exclude_indices "${action_exclude_indices[@]}" \
    --dtype "${dtype}" \
    --latent_num_inference_steps "${latent_num_inference_steps}" \
    --keypoint_num_inference_steps "${keypoint_num_inference_steps}" \
    --action_num_inference_steps "${action_num_inference_steps}" \
    --keypoint_video_condition "${keypoint_video_condition}" \
    --action_condition "${action_condition}" \
    --seed "${seed}" \
    --fps "${fps}" \
    "${optional_args[@]}" \
    "$@"
