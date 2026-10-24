#!/usr/bin/env bash
# Real robot + human cotrain ablation: wo_both (sampling ratio 2:1).
set -euo pipefail
DRY_RUN=${DRY_RUN:-0}

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

# Data and text. Source weights are defined in the JSON as robot:human = 2:1.
dataset_type=cotrain
train_mode=general

cotrain_config="${COTRAIN_CONFIG:-configs/real/cotrain_real_robot_human_73f_wo_both.json}"
action_target_frames=73

cotrain_validation=first
cotrain_samples_per_rank=${COTRAIN_SAMPLES_PER_RANK:-}
# Accept both real-data cache filename conventions, preserving explicit overrides.
default_text_embeddings="checkpoints/skel_wam/training_assets/text_embeddings_wo_overlay.pt"
if [[ ! -f "$default_text_embeddings" && -f "checkpoints/skel_wam/training_assets/text_embeddings_real_wo_overlay.pt" ]]; then
    default_text_embeddings="checkpoints/skel_wam/training_assets/text_embeddings_real_wo_overlay.pt"
fi
precomputed_text_embeddings_path="${TEXT_EMBEDDINGS_PATH:-${PRECOMPUTED_TEXT_EMBEDDINGS_PATH:-$default_text_embeddings}}"

# Model. RESUME is optional. With RESUME, the 42-D Action head is restored,
# frozen, omitted from the optimizer, and retained in every new checkpoint.
wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
transformer_checkpoint_dir=${wan_checkpoint_dir}
keypoint_expert_checkpoint="${KEYPOINT_EXPERT_CHECKPOINT:-}"
resume="${RESUME:-${RESUME:-}}"
resume_weights_only="${RESUME_WEIGHTS_ONLY:-0}"
preserve_action_head=0
freeze_action_head=0
action_dim=42
action_condition_mode=latent_only
action_patch_embedding_mode=shared
if [[ -n "${resume}" ]]; then
    resume_weights_only=1
    preserve_action_head=1
    freeze_action_head=1
fi

keypoint_hidden_dim=1024
keypoint_ffn_dim=4096
keypoint_cond_video_mode=full
keypoint_use_text_cross_attention=0
keypoint_dims=3
use_keypoint=0
image_size=384x384

# Direct joint training: no staged video/keypoint freezing.
freeze_video_expert=0
freeze_keypoint_expert=0
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
use_gradient_checkpointing=1
checkpoint_full_layer=0
gradient_as_bucket_view=1

# Prompts are fixed and precomputed for both robot and human templates.
encode_prompts_with_t5=0
cache_text_embeddings=1
t5_checkpoint_path=
t5_tokenizer_path=

fm_num_train_timesteps=1000
latent_fm_num_train_timesteps=${fm_num_train_timesteps}
keypoint_fm_num_train_timesteps=${fm_num_train_timesteps}
latent_snr_shift=5.0
keypoint_snr_shift=1.0
condition_noise_prob=0.5
condition_min_timestep=0.0
condition_max_timestep=0.2
latent_loss_weight=1.0
keypoint_loss_weight=0.0

wandb_project=skel-wam-real-cotrain-ablation
wandb_mode="${WANDB_MODE:-offline}"
wandb_name=
output_root="${OUTPUT_ROOT:-checkpoints/skel_wam/real_robot_cotrain_ablations/wo_both}"

# Distributed setup.
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
    echo "Invalid machine_rank=${machine_rank}/${num_machines}" >&2
    exit 1
fi
if ((num_machines > 1)) && [[ "${master_addr}" == "127.0.0.1" ]]; then
    echo "Multi-node training requires MASTER_ADDR or SLURM_JOB_NODELIST" >&2
    exit 1
fi

required_paths=(
    "${cotrain_config}"
    "${wan_checkpoint_dir}/Wan2.2_VAE.pth"
    "${precomputed_text_embeddings_path}"
)
if [[ "${use_keypoint}" == "1" ]]; then
    required_paths+=(
        "${keypoint_expert_checkpoint}"
        "utils/real_utils/keypoint_stats_real_human_robot.json"
    )
fi
if [[ -n "${resume}" ]]; then required_paths+=("${resume}"); fi
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

image_size_normalized=${image_size//x/ }
image_size_normalized=${image_size_normalized//\*/ }
read -r image_h image_w image_extra <<< "${image_size_normalized}"
if [[ -n "${image_extra:-}" ]] || ! [[ "${image_h:-}" =~ ^[0-9]+$ ]] || ! [[ "${image_w:-}" =~ ^[0-9]+$ ]]; then
    echo "Invalid image_size=${image_size}" >&2
    exit 1
fi

run_id=${RUN_ID:-${SLURM_JOB_ID:-$(date "+%Y-%m-%d_%H-%M-%S")}}
if [[ -z "${wandb_name}" ]]; then
    init_name=scratch
    [[ -n "${resume}" ]] && init_name=real_robot_resume
    wandb_name="skel_wam_real_cotrain_wo_both_2to1_${init_name}_${run_id}"
fi
output_dir="${output_root}/${wandb_name}"

optional_args=()
[[ -n "${cotrain_samples_per_rank}" ]] && optional_args+=(--cotrain_samples_per_rank "${cotrain_samples_per_rank}")
if [[ -n "${resume}" ]]; then
    optional_args+=(--resume "${resume}" --resume_weights_only)
    optional_args+=(--preserve_action_head --freeze_action_head --action_dim_override "${action_dim}")
fi
[[ "${use_keypoint}" == "1" ]] && optional_args+=(--use_keypoint) || optional_args+=(--no-use_keypoint)
[[ -n "${keypoint_expert_checkpoint}" ]] && optional_args+=(--keypoint_expert_checkpoint "${keypoint_expert_checkpoint}")
[[ "${freeze_video_expert}" == "1" ]] && optional_args+=(--freeze_video_expert)
[[ "${freeze_keypoint_expert}" == "1" ]] && optional_args+=(--freeze_keypoint_expert)
[[ "${use_gradient_checkpointing}" == "1" ]] && optional_args+=(--use_gradient_checkpointing)
[[ "${checkpoint_full_layer}" == "1" ]] && optional_args+=(--checkpoint_full_layer) || optional_args+=(--no-checkpoint_full_layer)
[[ "${gradient_as_bucket_view}" == "1" ]] && optional_args+=(--gradient_as_bucket_view) || optional_args+=(--no-gradient_as_bucket_view)
[[ "${encode_prompts_with_t5}" == "1" ]] && optional_args+=(--encode_prompts_with_t5) || optional_args+=(--no-encode_prompts_with_t5 --precomputed_text_embeddings_path "${precomputed_text_embeddings_path}")
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

echo "Training real robot/human cotrain ablation: wo_both (sampling 2:1)"
echo "  config=${cotrain_config} validation=${cotrain_validation}"
echo "  initialization=$([[ -n "${resume}" ]] && echo real_robot_resume || echo scratch)"
echo "  preserve_action=${preserve_action_head} freeze_action=${freeze_action_head} action_loss=disabled"
echo "  use_keypoint=${use_keypoint} action_condition_mode=${action_condition_mode}"
echo "  freeze_video=${freeze_video_expert} freeze_keypoint=${freeze_keypoint_expert}"
echo "  processes=${num_processes} effective_global_batch=${effective_global_batch}"
echo "  output=${output_dir}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

accelerate launch \
    "${accelerate_args[@]}" \
    main/train_skel_wam.py \
    --dataset_type "${dataset_type}" \
    --cotrain_config "${cotrain_config}" \
    --cotrain_validation "${cotrain_validation}" \
    --train_mode "${train_mode}" \
    --vae_checkpoint_dir "${wan_checkpoint_dir}" \
    --transformer_checkpoint_dir "${transformer_checkpoint_dir}" \
    --keypoint_hidden_dim "${keypoint_hidden_dim}" \
    --keypoint_ffn_dim "${keypoint_ffn_dim}" \
    --keypoint_cond_video_mode "${keypoint_cond_video_mode}" \
    --keypoint_dims "${keypoint_dims}" \
    --image_size "${image_h}" "${image_w}" \
    --action_target_frames "${action_target_frames}" \
    --action_condition_mode "${action_condition_mode}" \
    --action_patch_embedding_mode "${action_patch_embedding_mode}" \
    --output_dir "${output_dir}" \
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
    --dtype "${dtype}" \
    --fm_num_train_timesteps "${fm_num_train_timesteps}" \
    --latent_fm_num_train_timesteps "${latent_fm_num_train_timesteps}" \
    --keypoint_fm_num_train_timesteps "${keypoint_fm_num_train_timesteps}" \
    --latent_snr_shift "${latent_snr_shift}" \
    --keypoint_snr_shift "${keypoint_snr_shift}" \
    --condition_noise_prob "${condition_noise_prob}" \
    --condition_min_timestep "${condition_min_timestep}" \
    --condition_max_timestep "${condition_max_timestep}" \
    --latent_loss_weight "${latent_loss_weight}" \
    --keypoint_loss_weight "${keypoint_loss_weight}" \
    --wandb_project "${wandb_project}" \
    --wandb_name "${wandb_name}" \
    --wandb_mode "${wandb_mode}" \
    "${optional_args[@]}" \
  "$@"
