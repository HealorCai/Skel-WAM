#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

# Data and model artifacts.
real_root="${REAL_ROBOT_ROOT:-data/real_robot}"
dataset_root=${real_root}/skeleton_video_384x384
keypoint_root=${real_root}/skeleton_keypoints
action_root=${real_root}/action
lang_map_path=configs/real/lang_map.json
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/real_utils/keypoint_stats_real_robot.json}"
action_stats_path="${ACTION_STATS_PATH:-utils/real_utils/action_stats_real_robot.json}"
# Accept both real-data cache filename conventions, preserving explicit overrides.
default_text_embeddings="checkpoints/skel_wam/training_assets/text_embeddings.pt"
if [[ ! -f "$default_text_embeddings" && -f "checkpoints/skel_wam/training_assets/text_embeddings_real.pt" ]]; then
    default_text_embeddings="checkpoints/skel_wam/training_assets/text_embeddings_real.pt"
fi
precomputed_text_embeddings_path="${TEXT_EMBEDDINGS_PATH:-$default_text_embeddings}"

wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
transformer_checkpoint_dir=${wan_checkpoint_dir}
keypoint_expert_checkpoint="${KEYPOINT_EXPERT_CHECKPOINT:-checkpoints/skel_wam/training_assets/keypoint_expert_1024.pt}"
t5_checkpoint_path=
t5_tokenizer_path=

train_mode=robot
output_root="${OUTPUT_ROOT:-checkpoints/skel_wam/real_robot}"
task_include="${TASK_INCLUDE:-}"
# task_include=put_something_into_drawer
task_exclude=
metadata_cache_path=
resume="${RESUME:-}"
resume_weights_only="${RESUME_WEIGHTS_ONLY:-0}"

prompt_template="An egocentric video of a robot executing the following task, with a skeletal overlay on the hands: {task}"
empty_task_name="perform the task"

num_frames=73
action_video_freq_ratio=6
history_keypoint_offsets_str="-18 -12 -6"

global_sample_stride=1
image_size=384x384
action_target_frames=${num_frames}
action_exclude_indices_str=""
raw_action_dim=42

# Runtime/training hyperparameters. Tune max_steps after the first data audit.
ngpu_per_node="${GPUS_PER_NODE:-8}"
batch_size="${BATCH_SIZE:-2}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
num_workers="${NUM_WORKERS:-4}"
max_steps="${MAX_STEPS:-80000}"
save_step_every="${SAVE_STEP_EVERY:-2000}"
learning_rate=1.0e-4
keypoint_lr_multiplier=1.0
action_lr_multiplier=1.0
min_learning_rate=1.0e-5
lr_scheduler=warmup_cosine
warmup_steps=5000
dtype=bfloat16
vae_dtype=bfloat16
use_gradient_checkpointing=1
checkpoint_full_layer=0
detect_anomaly=${DETECT_ANOMALY:-0}
check_finite=${CHECK_FINITE:-0}
gradient_as_bucket_view=1
encode_prompts_with_t5=0
cache_text_embeddings=1

keypoint_hidden_dim=1024
keypoint_ffn_dim=4096
keypoint_cond_video_mode=full
keypoint_use_text_cross_attention=0
action_condition_mode=latent_keypoint
action_patch_embedding_mode=shared

fm_num_train_timesteps=1000
latent_fm_num_train_timesteps=${fm_num_train_timesteps}
keypoint_fm_num_train_timesteps=${fm_num_train_timesteps}
action_fm_num_train_timesteps=${fm_num_train_timesteps}
latent_snr_shift=5.0
keypoint_snr_shift=1.0
action_snr_shift=1.0
condition_noise_prob=0.5
condition_min_timestep=0.0
condition_max_timestep=0.2
keypoint_condition_noise_prob=0.0
keypoint_condition_min_timestep=0.0
keypoint_condition_max_timestep=0.0
latent_loss_weight=1.0
keypoint_loss_weight=2.0
action_loss_weight=2.0

wandb_project=skel-wam-real-robot
wandb_mode="${WANDB_MODE:-offline}"
wandb_name=

# Distributed configuration.
num_machines=${NUM_MACHINES:-${SLURM_NNODES:-1}}
machine_rank=${MACHINE_RANK:-${SLURM_NODEID:-0}}
num_processes=$((ngpu_per_node * num_machines))
if [[ -n "${MASTER_ADDR:-}" ]]; then
    master_addr=${MASTER_ADDR}
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
else
    master_addr=127.0.0.1
fi
master_port=${MASTER_PORT:-29500}

if ! [[ "${ngpu_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ngpu_per_node must be positive: ${ngpu_per_node}" >&2
    exit 1
fi
if ! [[ "${num_machines}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_machines must be positive: ${num_machines}" >&2
    exit 1
fi
if ! [[ "${machine_rank}" =~ ^[0-9]+$ ]] || ((machine_rank >= num_machines)); then
    echo "Invalid machine_rank=${machine_rank} for num_machines=${num_machines}" >&2
    exit 1
fi
if ((num_machines > 1)) && [[ "${master_addr}" == "127.0.0.1" ]]; then
    echo "Multi-node training requires MASTER_ADDR or SLURM_JOB_NODELIST" >&2
    exit 1
fi
if [[ "${resume_weights_only}" != "0" && "${resume_weights_only}" != "1" ]]; then
    echo "resume_weights_only must be 0 or 1" >&2
    exit 1
fi
if [[ "${resume_weights_only}" == "1" && -z "${resume}" ]]; then
    echo "resume_weights_only=1 requires resume" >&2
    exit 1
fi

run_id=${RUN_ID:-${SLURM_JOB_ID:-$(date "+%Y-%m-%d_%H-%M-%S")}}
if [[ -z "${wandb_name}" ]]; then
    wandb_name="skel_wam_real_robot_${task_include:-all}_${num_frames}f_${image_size}_${run_id}"
fi
output_dir="${output_root}/${wandb_name}"

image_size_normalized=${image_size//x/ }
image_size_normalized=${image_size_normalized//\*/ }
read -r image_h image_w image_extra <<< "${image_size_normalized}"
if [[ -n "${image_extra:-}" ]] || ! [[ "${image_h:-}" =~ ^[0-9]+$ ]] || ! [[ "${image_w:-}" =~ ^[0-9]+$ ]]; then
    echo "Invalid image_size=${image_size}" >&2
    exit 1
fi
if (((num_frames - 1) % action_video_freq_ratio != 0)); then
    echo "(num_frames - 1) must be divisible by action_video_freq_ratio" >&2
    exit 1
fi
video_keypoint_frames=$(((num_frames - 1) / action_video_freq_ratio + 1))
if (((video_keypoint_frames - 1) % 4 != 0)); then
    echo "video_keypoint_frames must satisfy (frames - 1) % 4 == 0" >&2
    exit 1
fi

required_paths=(
    "${dataset_root}"
    "${keypoint_root}"
    "${action_root}"
    "${lang_map_path}"
    "${keypoint_stats_path}"
    "${action_stats_path}"
    "${wan_checkpoint_dir}"
    "${keypoint_expert_checkpoint}"
)
if [[ "${encode_prompts_with_t5}" == "0" ]]; then
    required_paths+=("${precomputed_text_embeddings_path}")
fi
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

read -r -a history_keypoint_offsets <<< "${history_keypoint_offsets_str}"
read -r -a action_exclude_indices <<< "${action_exclude_indices_str}"
optional_args=()
[[ -n "${task_include}" ]] && optional_args+=(--task_include "${task_include}")
[[ -n "${task_exclude}" ]] && optional_args+=(--task_exclude "${task_exclude}")
[[ -n "${metadata_cache_path}" ]] && optional_args+=(--metadata_cache_path "${metadata_cache_path}")
[[ -n "${resume}" ]] && optional_args+=(--resume "${resume}")
[[ "${resume_weights_only}" == "1" ]] && optional_args+=(--resume_weights_only)
[[ ${#action_exclude_indices[@]} -gt 0 ]] && optional_args+=(--action_exclude_indices "${action_exclude_indices[@]}")
[[ "${use_gradient_checkpointing}" == "1" ]] && optional_args+=(--use_gradient_checkpointing)
[[ "${gradient_as_bucket_view}" == "1" ]] && optional_args+=(--gradient_as_bucket_view) || optional_args+=(--no-gradient_as_bucket_view)
[[ "${checkpoint_full_layer}" == "1" ]] && optional_args+=(--checkpoint_full_layer) || optional_args+=(--no-checkpoint_full_layer)
[[ "${detect_anomaly}" == "1" ]] && optional_args+=(--detect_anomaly)
[[ "${check_finite}" == "1" ]] && optional_args+=(--check_finite)
if [[ "${encode_prompts_with_t5}" == "1" ]]; then
    optional_args+=(--encode_prompts_with_t5)
else
    optional_args+=(--no-encode_prompts_with_t5 --precomputed_text_embeddings_path "${precomputed_text_embeddings_path}")
fi
[[ "${cache_text_embeddings}" == "1" ]] && optional_args+=(--cache_text_embeddings) || optional_args+=(--no-cache_text_embeddings)
[[ "${keypoint_use_text_cross_attention}" == "1" ]] && optional_args+=(--keypoint_use_text_cross_attention) || optional_args+=(--no-keypoint_use_text_cross_attention)
[[ -n "${t5_checkpoint_path}" ]] && optional_args+=(--t5_checkpoint_path "${t5_checkpoint_path}")
[[ -n "${t5_tokenizer_path}" ]] && optional_args+=(--t5_tokenizer_path "${t5_tokenizer_path}")

accelerate_args=(
    --num_machines "${num_machines}"
    --num_processes "${num_processes}"
    --machine_rank "${machine_rank}"
    --main_process_ip "${master_addr}"
    --main_process_port "${master_port}"
)
if ((num_processes > 1)); then accelerate_args=(--multi_gpu "${accelerate_args[@]}"); fi
effective_global_batch=$((num_processes * batch_size * gradient_accumulation_steps))

echo "Training real-robot Skel-WAM MoT-v2"
echo "  data=${real_root} task=${task_include:-all}"
echo "  language_map=${lang_map_path}"
echo "  prompt_template=${prompt_template}"
echo "  frames: action=${num_frames} video/keypoint=${video_keypoint_frames} ratio=${action_video_freq_ratio}"
echo "  action_dims=${raw_action_dim} excluded=${#action_exclude_indices[@]} retained=$((raw_action_dim - ${#action_exclude_indices[@]}))"
echo "  processes=${num_processes} effective_global_batch=${effective_global_batch}"
echo "  text_embeddings=${precomputed_text_embeddings_path}"
echo "  output_dir=${output_dir}"

accelerate launch \
    "${accelerate_args[@]}" \
    main/train_skel_wam.py \
    --dataset_root "${dataset_root}" \
    --keypoint_root "${keypoint_root}" \
    --action_root "${action_root}" \
    --vae_checkpoint_dir "${wan_checkpoint_dir}" \
    --transformer_checkpoint_dir "${transformer_checkpoint_dir}" \
    --keypoint_expert_checkpoint "${keypoint_expert_checkpoint}" \
    --keypoint_hidden_dim "${keypoint_hidden_dim}" \
    --keypoint_ffn_dim "${keypoint_ffn_dim}" \
    --keypoint_cond_video_mode "${keypoint_cond_video_mode}" \
    --train_mode "${train_mode}" \
    --output_dir "${output_dir}" \
    --num_frames "${num_frames}" \
    --global_sample_stride "${global_sample_stride}" \
    --action_video_freq_ratio "${action_video_freq_ratio}" \
    --image_size "${image_h}" "${image_w}" \
    --history_keypoint_offsets "${history_keypoint_offsets[@]}" \
    --action_target_frames "${action_target_frames}" \
    --prompt_template "${prompt_template}" \
    --empty_task_name "${empty_task_name}" \
    --task_lang_map_path "${lang_map_path}" \
    --batch_size "${batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --num_workers "${num_workers}" \
    --max_steps "${max_steps}" \
    --save_step_every "${save_step_every}" \
    --learning_rate "${learning_rate}" \
    --keypoint_lr_multiplier "${keypoint_lr_multiplier}" \
    --action_lr_multiplier "${action_lr_multiplier}" \
    --min_learning_rate "${min_learning_rate}" \
    --lr_scheduler "${lr_scheduler}" \
    --warmup_steps "${warmup_steps}" \
    --keypoint_stats_path "${keypoint_stats_path}" \
    --action_stats_path "${action_stats_path}" \
    --dtype "${dtype}" \
    --vae_dtype "${vae_dtype}" \
    --fm_num_train_timesteps "${fm_num_train_timesteps}" \
    --latent_fm_num_train_timesteps "${latent_fm_num_train_timesteps}" \
    --keypoint_fm_num_train_timesteps "${keypoint_fm_num_train_timesteps}" \
    --action_fm_num_train_timesteps "${action_fm_num_train_timesteps}" \
    --latent_snr_shift "${latent_snr_shift}" \
    --keypoint_snr_shift "${keypoint_snr_shift}" \
    --action_snr_shift "${action_snr_shift}" \
    --condition_noise_prob "${condition_noise_prob}" \
    --condition_min_timestep "${condition_min_timestep}" \
    --condition_max_timestep "${condition_max_timestep}" \
    --keypoint_condition_noise_prob "${keypoint_condition_noise_prob}" \
    --keypoint_condition_min_timestep "${keypoint_condition_min_timestep}" \
    --keypoint_condition_max_timestep "${keypoint_condition_max_timestep}" \
    --action_condition_mode "${action_condition_mode}" \
    --action_patch_embedding_mode "${action_patch_embedding_mode}" \
    --latent_loss_weight "${latent_loss_weight}" \
    --keypoint_loss_weight "${keypoint_loss_weight}" \
    --action_loss_weight "${action_loss_weight}" \
    --wandb_project "${wandb_project}" \
    --wandb_name "${wandb_name}" \
    --wandb_mode "${wandb_mode}" \
    "${optional_args[@]}" \
  "$@"
