#!/usr/bin/env bash
# Run the Skel-WAM model over all 25 EgoVLA room/table settings.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

DRY_RUN="${DRY_RUN:-0}"

# Evaluation task and output.
TASK_NAME="${TASK_NAME:-Push-Box}"
TASK="Humanoid-${TASK_NAME}-v0"
SAVE_FRAMES=0

SAVING_PATH="${EVAL_OUTPUT_ROOT:-output/egovla/skel_wam_wo_overlay}"
BASE_LABEL="${EVAL_LABEL:-wo_overlay_${TASK_NAME}}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to an existing Skel-WAM checkpoint}"
num_frames="${NUM_FRAMES:-37}"
action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO:-3}"
read -r -a history_keypoint_offsets <<< "${HISTORY_KEYPOINT_OFFSETS:--12 -9 -6 -3}"

action_checkpoint="${ACTION_CHECKPOINT:-}"

NUM_TRIALS="${NUM_TRIALS:-1}"

GPU_IDS="${GPU_IDS:-0}"

wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
precomputed_text_embeddings_path="${TEXT_EMBEDDINGS_PATH-${TEST_TEXT_EMBEDDINGS_PATH-checkpoints/skel_wam/training_assets/text_embeddings_egovla_wo_overlay.pt}}"

# Window geometry. 37 raw/action frames produce 13 video/keypoint frames.
action_target_frames=${num_frames}
action_exclude_indices=(0 1 2 3 6 7 10 11 14 15 18 19)
exec_action_steps=30

# Overlapping action-chunk temporal ensemble. Set ACTION_CHUNK_WEIGHTING=none
# to preserve the original single-chunk execution.
action_chunk_weighting=none # exponential / none
action_chunk_weight_decay=0.8
action_chunk_max_history=4

# Flow-matching inference.
dtype="${DTYPE:-bfloat16}"
num_inference_steps=10
latent_num_inference_steps=10
keypoint_num_inference_steps=10
action_num_inference_steps=5
latent_snr_shift="${LATENT_SNR_SHIFT:-}"
keypoint_snr_shift="${KEYPOINT_SNR_SHIFT:-}"
action_snr_shift="${ACTION_SNR_SHIFT:-}"

# Normalization and skeleton extraction.
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/egovla_utils/keypoint_stats_egovla_7tasks.json}"
action_stats_path="${ACTION_STATS_PATH:-utils/egovla_utils/qpos_stats_egovla_7tasks.json}"
joint_names_file="eval/egovla/utils/joint_names.txt"
usd_path="eval/egovla/utils/h1_inspire_convex_decomp.usd"
skeleton_cfg="eval/egovla/utils/h1_inspire_both_hands.yaml"

read -r -a GPU_LIST <<< "${GPU_IDS}"

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi
if [[ -z "${IsaacLab_PATH:-}" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: IsaacLab_PATH is not set." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
  echo "ERROR: checkpoint does not exist: ${checkpoint}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && -n "${action_checkpoint}" && ! -f "${action_checkpoint}" ]]; then
  echo "ERROR: action checkpoint does not exist: ${action_checkpoint}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${wan_checkpoint_dir}/Wan2.2_VAE.pth" ]]; then
  echo "ERROR: Wan VAE does not exist under: ${wan_checkpoint_dir}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && -n "${precomputed_text_embeddings_path}" && ! -f "${precomputed_text_embeddings_path}" ]]; then
  echo "ERROR: precomputed text embeddings do not exist: ${precomputed_text_embeddings_path}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${keypoint_stats_path}" ]]; then
  echo "ERROR: keypoint stats do not exist: ${keypoint_stats_path}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${action_stats_path}" ]]; then
  echo "ERROR: action stats do not exist: ${action_stats_path}" >&2
  exit 2
fi
if (( (num_frames - 1) % action_video_freq_ratio != 0 )); then
  echo "ERROR: num_frames must satisfy (num_frames - 1) % action_video_freq_ratio == 0." >&2
  exit 2
fi

run_setting() {
  local room_idx="$1"
  local table_idx="$2"
  local num_episodes="$3"
  local gpu_id="$4"
  local additional_label="${BASE_LABEL}"
  local optional_args=()

  if [[ -n "${precomputed_text_embeddings_path}" ]]; then
    optional_args+=(--precomputed_text_embeddings_path "${precomputed_text_embeddings_path}")
  fi
  if [[ -n "${action_checkpoint}" ]]; then
    optional_args+=(--action_checkpoint "${action_checkpoint}")
  fi
  if [[ -n "${latent_snr_shift}" ]]; then optional_args+=(--latent_snr_shift "${latent_snr_shift}"); fi
  if [[ -n "${keypoint_snr_shift}" ]]; then optional_args+=(--keypoint_snr_shift "${keypoint_snr_shift}"); fi
  if [[ -n "${action_snr_shift}" ]]; then optional_args+=(--action_snr_shift "${action_snr_shift}"); fi

  echo "[GPU ${gpu_id}] task=${TASK_NAME} room=${room_idx} table=${table_idx} episodes=${num_episodes} label=${additional_label}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${IsaacLab_PATH}/isaaclab.sh" -p eval/egovla/skel_wam_wo_overlay_agent.py \
    --task "${TASK}" \
    --room_idx "${room_idx}" \
    --table_idx "${table_idx}" \
    --num_episodes "${num_episodes}" \
    --num_trials "${NUM_TRIALS}" \
    --saving_path "${SAVING_PATH}" \
    --save_frames "${SAVE_FRAMES}" \
    --additional_label "${additional_label}" \
    --checkpoint "${checkpoint}" \
    --vae_checkpoint_dir "${wan_checkpoint_dir}" \
    --num_frames "${num_frames}" \
    --action_video_freq_ratio "${action_video_freq_ratio}" \
    --history_keypoint_offsets "${history_keypoint_offsets[@]}" \
    --action_target_frames "${action_target_frames}" \
    --action_exclude_indices "${action_exclude_indices[@]}" \
    --dtype "${dtype}" \
    --num_inference_steps "${num_inference_steps}" \
    --latent_num_inference_steps "${latent_num_inference_steps}" \
    --keypoint_num_inference_steps "${keypoint_num_inference_steps}" \
    --action_num_inference_steps "${action_num_inference_steps}" \
    --keypoint_stats_path "${keypoint_stats_path}" \
    --action_stats_path "${action_stats_path}" \
    --exec_action_steps "${exec_action_steps}" \
    --action_chunk_weighting "${action_chunk_weighting}" \
    --action_chunk_weight_decay "${action_chunk_weight_decay}" \
    --action_chunk_max_history "${action_chunk_max_history}" \
    --overlay_hand_skeleton \
    --use_fk_hand \
    --joint_names_file "${joint_names_file}" \
    --usd_path "${usd_path}" \
    --skeleton_cfg "${skeleton_cfg}" \
    "${optional_args[@]}"
}

settings=()
for room_idx in 1 2 3 4 5; do
  for table_idx in 1 2 3 4 5; do
    if [[ "${table_idx}" -eq 1 && "${room_idx}" -le 3 ]]; then
      settings+=("${room_idx},${table_idx},9")
    else
      settings+=("${room_idx},${table_idx},3")
    fi
  done
done

echo "Evaluating Skel-WAM over ${#settings[@]} settings"
echo "task=${TASK_NAME} GPUs=${GPU_LIST[*]} label=${BASE_LABEL}"
echo "checkpoint=${checkpoint}"
echo "action_checkpoint=${action_checkpoint:-embedded}"
echo "frames: raw/action=${num_frames}, video/keypoint=$(( (num_frames - 1) / action_video_freq_ratio + 1 ))"
echo "action dimensions: model=$((50 - ${#action_exclude_indices[@]})), environment=50, excluded=${action_exclude_indices[*]}"
echo "inference steps: video=${latent_num_inference_steps}, keypoint=${keypoint_num_inference_steps}, action=${action_num_inference_steps}"

gpu_count="${#GPU_LIST[@]}"
declare -a gpu_jobs
for ((slot = 0; slot < gpu_count; slot++)); do
  gpu_jobs[slot]=""
done
for ((index = 0; index < ${#settings[@]}; index++)); do
  slot=$((index % gpu_count))
  gpu_jobs[slot]+="${settings[index]} "
done

pids=()
for ((slot = 0; slot < gpu_count; slot++)); do
  gpu_id="${GPU_LIST[slot]}"
  job_list="${gpu_jobs[slot]}"
  if [[ -z "${job_list// /}" ]]; then
    continue
  fi
  (
    for spec in ${job_list}; do
      IFS=',' read -r room_idx table_idx num_episodes <<< "${spec}"
      run_setting "${room_idx}" "${table_idx}" "${num_episodes}" "${gpu_id}"
    done
  ) &
  pids+=("$!")
done

failure=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failure=1
  fi
done
exit "${failure}"
