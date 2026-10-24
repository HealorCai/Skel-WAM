#!/usr/bin/env bash
# Test a real-robot checkpoint (video + keypoint + 42-D Action).
# Example:
#   CHECKPOINT=checkpoints/skel_wam_real_robot/<run>/checkpoints/step_050000_weights.pt \
#     bash scripts/test_skel_wam_real_robot.sh
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

checkpoint="${CHECKPOINT:-}"
action_checkpoint="${ACTION_CHECKPOINT:-}"
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/real_utils/keypoint_stats_real_robot.json}"

real_root="${REAL_ROBOT_ROOT:-data/real_robot}"
dataset_root=${real_root}/skeleton_video_384x384
keypoint_root=${real_root}/skeleton_keypoints
action_root=${real_root}/action
action_stats_path="${ACTION_STATS_PATH:-utils/real_utils/action_stats_real_robot.json}"
lang_map_path=configs/real/lang_map.json
# Accept both real-data cache filename conventions.
default_text_embeddings="${project_root}/checkpoints/skel_wam/training_assets/text_embeddings.pt"
if [[ ! -f "$default_text_embeddings" && -f "${project_root}/checkpoints/skel_wam/training_assets/text_embeddings_real.pt" ]]; then
    default_text_embeddings="${project_root}/checkpoints/skel_wam/training_assets/text_embeddings_real.pt"
fi
precomputed_text_embeddings_path="${TEST_TEXT_EMBEDDINGS_PATH:-${TEXT_EMBEDDINGS_PATH:-$default_text_embeddings}}"
wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"

# task_include=place_fruit_into_box
# task_include=pour_water
# task_include=put_something_into_drawer
task_include="${TASK_INCLUDE:-stack_cups}"
sample_idx="${SAMPLE_IDX:-300}"
output_root="${OUTPUT_ROOT:-output/real_robot/test}"
num_frames=73
action_video_freq_ratio=6
history_keypoint_offsets=(-18 -12 -6)
image_size=(384 384)

if [[ -z "${checkpoint}" ]]; then
    echo "Set CHECKPOINT to a real-robot Skel-WAM checkpoint." >&2
    exit 1
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint does not exist: ${checkpoint}" >&2
    exit 1
fi
if [[ -n "${action_checkpoint}" && ! -f "${action_checkpoint}" ]]; then
    echo "Action checkpoint does not exist: ${action_checkpoint}" >&2
    exit 1
fi
for path in "${dataset_root}" "${keypoint_root}" "${action_root}"; do
    [[ -d "${path}" ]] || { echo "Required directory does not exist: ${path}" >&2; exit 1; }
done
for path in "${keypoint_stats_path}" "${action_stats_path}" "${lang_map_path}" "${precomputed_text_embeddings_path}"; do
    [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 1; }
done

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
checkpoint_name=$(basename "${checkpoint}")
run_name=$(basename "$(dirname "$(dirname "${checkpoint}")")")
output_dir=${OUTPUT_DIR:-${output_root}/${run_name}/${checkpoint_name}_${task_include}_s${sample_idx}_${timestamp}}

optional_args=()
[[ -n "${action_checkpoint}" ]] && optional_args+=(--action_checkpoint "${action_checkpoint}")

echo "Testing real_robot checkpoint: ${checkpoint}"
echo "  action_checkpoint=${action_checkpoint:-embedded}"
echo "  task=${task_include}, sample=${sample_idx}, output=${output_dir}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 main/test_skel_wam.py \
    --checkpoint "${checkpoint}" \
    --dataset_type skel_wam \
    --dataset_root "${dataset_root}" \
    --keypoint_root "${keypoint_root}" \
    --action_root "${action_root}" \
    --keypoint_stats_path "${keypoint_stats_path}" \
    --action_stats_path "${action_stats_path}" \
    --task_lang_map_path "${lang_map_path}" \
    --precomputed_text_embeddings_path "${precomputed_text_embeddings_path}" \
    --wan_checkpoint_dir "${wan_checkpoint_dir}" \
    --task_include "${task_include}" \
    --sample_idx "${sample_idx}" \
    --output_dir "${output_dir}" \
    --num_frames "${num_frames}" \
    --global_sample_stride 1 \
    --action_video_freq_ratio "${action_video_freq_ratio}" \
    --image_size "${image_size[@]}" \
    --history_keypoint_offsets "${history_keypoint_offsets[@]}" \
    --action_target_frames "${num_frames}" \
    --dtype bfloat16 \
    --latent_num_inference_steps 3 \
    --keypoint_num_inference_steps 3 \
    --action_num_inference_steps 5 \
    --keypoint_video_condition pred \
    --action_condition pred \
    --run_action \
    --no-vis_keypoint_video_attn \
    --no-vis_keypoint_video_same_frame_attn \
    --attn_vis_keep_height_ratio 0.83 \
    --text_encoder_device cuda \
    --seed 42 \
    --fps 8 \
    "${optional_args[@]}" \
  "$@"
