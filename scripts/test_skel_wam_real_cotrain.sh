#!/usr/bin/env bash
# Test a real_robot + real_human cotrain checkpoint on either or both domains.
# By default this evaluates the cotrain objectives (video + keypoint) only.
# Set RUN_ROBOT_ACTION=1 to also evaluate a preserved real-robot Action head.
# Example:
#   CHECKPOINT=checkpoints/skel_wam_real_cotrain/<run>/checkpoints/step_080000_weights.pt \
#     TEST_DOMAIN=both bash scripts/test_skel_wam_real_cotrain.sh
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

video_subdir="${VIDEO_SUBDIR:-skeleton_video_384x384}"
action_checkpoint="${ACTION_CHECKPOINT:-}"
checkpoint="${CHECKPOINT:-}"
test_domain=${TEST_DOMAIN:-both} # robot, human, both
run_robot_action=${RUN_ROBOT_ACTION:-0}
data_parent="${REAL_DATA_PARENT:-data}"
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/real_utils/keypoint_stats_real_human_robot.json}"
action_stats_path="${ACTION_STATS_PATH:-utils/real_utils/action_stats_real_robot.json}"
lang_map_path=configs/real/lang_map.json
# Accept both real-data cache filename conventions.
default_text_embeddings="${project_root}/checkpoints/skel_wam/training_assets/text_embeddings.pt"
if [[ ! -f "$default_text_embeddings" && -f "${project_root}/checkpoints/skel_wam/training_assets/text_embeddings_real.pt" ]]; then
    default_text_embeddings="${project_root}/checkpoints/skel_wam/training_assets/text_embeddings_real.pt"
fi
precomputed_text_embeddings_path="${TEST_TEXT_EMBEDDINGS_PATH:-${TEXT_EMBEDDINGS_PATH:-$default_text_embeddings}}"
wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
output_root="${OUTPUT_ROOT:-output/real_robot_cotrain/test}"
task_include="${TASK_INCLUDE:-stack_cups}"
robot_sample_idx="${SAMPLE_IDX:-300}"
human_sample_idx="${SAMPLE_IDX:-200}"
num_frames=73
action_video_freq_ratio=6
history_keypoint_offsets=(-18 -12 -6)

if [[ -z "${checkpoint}" ]]; then
    echo "Set CHECKPOINT to a real cotrain checkpoint." >&2
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
case "${test_domain}" in robot|human|both) ;; *) echo "TEST_DOMAIN must be robot, human, or both." >&2; exit 1 ;; esac
case "${run_robot_action}" in 0|1) ;; *) echo "RUN_ROBOT_ACTION must be 0 or 1." >&2; exit 1 ;; esac
for path in "${keypoint_stats_path}" "${lang_map_path}" "${precomputed_text_embeddings_path}"; do
    [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 1; }
done
if [[ "${run_robot_action}" == 1 && ! -f "${action_stats_path}" ]]; then
    echo "Action stats do not exist: ${action_stats_path}" >&2
    exit 1
fi

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
checkpoint_name=$(basename "${checkpoint}")
run_name=$(basename "$(dirname "$(dirname "${checkpoint}")")")
output_dir=${OUTPUT_DIR:-${output_root}/${run_name}/${checkpoint_name}_${timestamp}}
declare -a commands=()
extra_args=("$@")

add_domain() {
    local domain=$1
    local root=${data_parent}/real_${domain}
    local sample_idx
    [[ "${domain}" == robot ]] && sample_idx=${robot_sample_idx} || sample_idx=${human_sample_idx}
    for path in "${root}/${video_subdir}" "${root}/skeleton_keypoints"; do
        [[ -d "${path}" ]] || { echo "Required directory does not exist: ${path}" >&2; exit 1; }
    done

    local -a command=(
        --checkpoint "${checkpoint}"
        --dataset_type skel_wam
        --dataset_root "${root}/${video_subdir}"
        --keypoint_root "${root}/skeleton_keypoints"
        --keypoint_stats_path "${keypoint_stats_path}"
        --task_lang_map_path "${lang_map_path}"
        --precomputed_text_embeddings_path "${precomputed_text_embeddings_path}"
        --wan_checkpoint_dir "${wan_checkpoint_dir}"
        --task_include "${task_include}"
        --sample_idx "${sample_idx}"
        --output_dir "${output_dir}/${domain}_${task_include}_s${sample_idx}"
        --num_frames "${num_frames}"
        --global_sample_stride 1
        --action_video_freq_ratio "${action_video_freq_ratio}"
        --image_size 384 384
        --history_keypoint_offsets "${history_keypoint_offsets[@]}"
        --dtype bfloat16
        --latent_num_inference_steps 3
        --keypoint_num_inference_steps 3
        --keypoint_video_condition pred
        --no-vis_keypoint_video_attn
        --no-vis_keypoint_video_same_frame_attn
        --attn_vis_keep_height_ratio 0.83
        --text_encoder_device cuda
        --seed 42
        --fps 8
    )
    if [[ -n "${action_checkpoint}" && "${domain}" == robot && "${run_robot_action}" == 1 ]]; then
        command+=(--action_checkpoint "${action_checkpoint}")
    fi
    if [[ "${domain}" == robot && "${run_robot_action}" == 1 ]]; then
        [[ -d "${root}/action" ]] || { echo "Action root does not exist: ${root}/action" >&2; exit 1; }
        command+=(
            --action_root "${root}/action"
            --action_stats_path "${action_stats_path}"
            --action_target_frames "${num_frames}"
            --action_num_inference_steps 5
            --action_condition pred
            --run_action
        )
    else
        command+=(--no-run_action)
    fi
    (( ${#commands[@]} == 0 )) || commands+=(--next)
    commands+=("${command[@]}" "${extra_args[@]}")
}

[[ "${test_domain}" == robot || "${test_domain}" == both ]] && add_domain robot
[[ "${test_domain}" == human || "${test_domain}" == both ]] && add_domain human

echo "Testing real cotrain checkpoint: ${checkpoint}"
echo "  domains=${test_domain}, robot_action=${run_robot_action}, output=${output_dir}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    python3 main/run_skel_wam_inprocess.py "${commands[@]}"
