#!/usr/bin/env bash
set -euo pipefail

# =========================
# Frequently edited config
# =========================
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

dataset_root="${DATASET_ROOT:-data/EgoVLA_SIM/skeleton_video_384x384}"
keypoint_root="${KEYPOINT_ROOT:-data/EgoVLA_SIM/skeleton_keypoints}"
action_root="${ACTION_ROOT:-data/EgoVLA_SIM/qpos}"
wan_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
transformer_checkpoint_dir=${wan_checkpoint_dir}
keypoint_expert_checkpoint="${KEYPOINT_EXPERT_CHECKPOINT:-checkpoints/skel_wam/training_assets/keypoint_expert_1024.pt}"
t5_checkpoint_path=
t5_tokenizer_path=
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/egovla_utils/keypoint_stats_egovla_7tasks.json}"
action_stats_path="${ACTION_STATS_PATH:-utils/egovla_utils/qpos_stats_egovla_7tasks.json}"
action_exclude_indices_str="0 1 2 3 6 7 10 11 14 15 18 19"

train_mode=robot
output_root="${OUTPUT_ROOT:-checkpoints/skel_wam/egovla_sim}"
task_include="${TASK_INCLUDE:-}"
task_exclude="Insert-And-Unload-Cans,Insert-Cans/Left_Right_Deprecated,Insert-Cans,Sort-Cans/Single_hand_first_deprecated,Sort-Cans,Stack-Can-Into-Drawer,Unload-Cans"
metadata_cache_path=
resume="${RESUME:-}"
resume_weights_only="${RESUME_WEIGHTS_ONLY:-0}"


prompt_template="An egocentric video of a robot executing the following task, with a skeletal overlay on the hands: {task}"
empty_task_name="perform the task"
disable_task_lang_map=0

# Data/window geometry. IMAGE_SIZE accepts 384x384 or 384*384.

num_frames="${NUM_FRAMES:-37}"
action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO:-3}"
history_keypoint_offsets_str="${HISTORY_KEYPOINT_OFFSETS:--12 -9 -6 -3}"

global_sample_stride=1
image_size=384x384
action_target_frames=${num_frames}

# Runtime/training hyperparameters.
# These are per-node/per-process settings. Accelerate's num_processes is global.
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

lr_scheduler=warmup_cosine # warmup_cosine for scratch, constant for resume
warmup_steps=5000
dtype=bfloat16
use_gradient_checkpointing=1
checkpoint_full_layer=0
gradient_as_bucket_view=1
encode_prompts_with_t5="${ENCODE_PROMPTS_WITH_T5:-0}"
cache_text_embeddings=1
precomputed_text_embeddings_path="${TEXT_EMBEDDINGS_PATH-checkpoints/skel_wam/training_assets/text_embeddings_egovla.pt}"

keypoint_hidden_dim=1024
keypoint_ffn_dim=4096
keypoint_cond_video_mode=full
keypoint_use_text_cross_attention=0
action_condition_mode=latent_keypoint
action_patch_embedding_mode=shared

# Flow-matching and condition corruption. FM noise is fixed to standard Gaussian.
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

# Logging. RUN_ID may be supplied explicitly when launching multiple runs in one job.
wandb_project=skel-wam
wandb_mode="${WANDB_MODE:-offline}"
wandb_name=

# =========================
# Distributed configuration
# =========================
num_machines="${NUM_MACHINES:-${SLURM_NNODES:-1}}"
machine_rank="${MACHINE_RANK:-${SLURM_NODEID:-0}}"
num_processes=$((ngpu_per_node * num_machines))

if [ -n "${MASTER_ADDR:-}" ]; then
    master_addr=${MASTER_ADDR}
elif [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
else
    master_addr=127.0.0.1
fi
master_port=${MASTER_PORT:-29500}

if ! [[ "${ngpu_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer, got: ${ngpu_per_node}" >&2
    exit 1
fi
if ! [[ "${num_machines}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_NNODES must be a positive integer, got: ${num_machines}" >&2
    exit 1
fi
if ! [[ "${machine_rank}" =~ ^[0-9]+$ ]] || (( machine_rank >= num_machines )); then
    echo "Invalid machine rank ${machine_rank} for ${num_machines} machines." >&2
    exit 1
fi
if (( num_machines > 1 )) && [ "${master_addr}" = "127.0.0.1" ]; then
    echo "Multi-node training requires a reachable MASTER_ADDR." >&2
    exit 1
fi
if [[ "${resume_weights_only}" != "0" && "${resume_weights_only}" != "1" ]]; then
    echo "resume_weights_only must be 0 or 1, got: ${resume_weights_only}" >&2
    exit 1
fi
if [[ "${resume_weights_only}" = "1" && -z "${resume}" ]]; then
    echo "resume_weights_only=1 requires a non-empty resume checkpoint." >&2
    exit 1
fi

# SLURM_JOB_ID is shared by every node, unlike a per-node wall-clock timestamp.
run_id=${RUN_ID:-${SLURM_JOB_ID:-$(date "+%Y-%m-%d_%H-%M-%S")}}

# =========================
# Derived config and checks
# =========================
if [ -z "${wandb_name}" ]; then
    wandb_name="skel_wam_${train_mode}_${task_include:-all}_${num_frames}f_${image_size}_${run_id}"
fi

output_dir="${output_root}/${wandb_name}"

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

optional_args=()
if [ -n "${task_include}" ]; then optional_args+=(--task_include "${task_include}"); fi
if [ -n "${task_exclude}" ]; then optional_args+=(--task_exclude "${task_exclude}"); fi
if [ -n "${metadata_cache_path}" ]; then optional_args+=(--metadata_cache_path "${metadata_cache_path}"); fi
if [ -n "${resume}" ]; then optional_args+=(--resume "${resume}"); fi
if [ "${resume_weights_only}" = "1" ]; then optional_args+=(--resume_weights_only); fi
if [ "${disable_task_lang_map}" = "1" ]; then optional_args+=(--disable_task_lang_map); fi
if [ "${use_gradient_checkpointing}" = "1" ]; then optional_args+=(--use_gradient_checkpointing); fi
if [ "${gradient_as_bucket_view}" = "1" ]; then optional_args+=(--gradient_as_bucket_view); fi
if [ "${gradient_as_bucket_view}" = "0" ]; then optional_args+=(--no-gradient_as_bucket_view); fi
if [ "${checkpoint_full_layer}" = "1" ]; then optional_args+=(--checkpoint_full_layer); fi
if [ "${checkpoint_full_layer}" = "0" ]; then optional_args+=(--no-checkpoint_full_layer); fi
if [ "${encode_prompts_with_t5}" = "1" ]; then optional_args+=(--encode_prompts_with_t5); fi
if [ "${encode_prompts_with_t5}" = "0" ]; then
    optional_args+=(--no-encode_prompts_with_t5 --precomputed_text_embeddings_path "${precomputed_text_embeddings_path}")
fi
if [ "${cache_text_embeddings}" = "1" ]; then optional_args+=(--cache_text_embeddings); fi
if [ "${cache_text_embeddings}" = "0" ]; then optional_args+=(--no-cache_text_embeddings); fi
if [ "${keypoint_use_text_cross_attention}" = "1" ]; then optional_args+=(--keypoint_use_text_cross_attention); fi
if [ "${keypoint_use_text_cross_attention}" = "0" ]; then optional_args+=(--no-keypoint_use_text_cross_attention); fi
optional_args+=(--action_condition_mode "${action_condition_mode}" --action_patch_embedding_mode "${action_patch_embedding_mode}")
if [ -n "${t5_checkpoint_path}" ]; then optional_args+=(--t5_checkpoint_path "${t5_checkpoint_path}"); fi
if [ -n "${t5_tokenizer_path}" ]; then optional_args+=(--t5_tokenizer_path "${t5_tokenizer_path}"); fi

accelerate_args=(
    --num_machines "${num_machines}"
    --num_processes "${num_processes}"
    --machine_rank "${machine_rank}"
    --main_process_ip "${master_addr}"
    --main_process_port "${master_port}"
)
if (( num_processes > 1 )); then
    accelerate_args=(--multi_gpu "${accelerate_args[@]}")
fi

effective_global_batch=$(( num_processes * batch_size * gradient_accumulation_steps ))

echo "Training Skel-WAM (multi-node)"
echo "  slurm_job_id=${SLURM_JOB_ID:-none} node=${HOSTNAME:-unknown}"
echo "  num_machines=${num_machines} machine_rank=${machine_rank} ngpu_per_node=${ngpu_per_node} num_processes=${num_processes}"
echo "  master_addr=${master_addr} master_port=${master_port}"
echo "  wan_checkpoint_dir=${wan_checkpoint_dir}"
echo "  frames: action/raw=${num_frames}, video/keypoint=${video_keypoint_frames}, ratio=${action_video_freq_ratio}, image=${image_h}x${image_w}"
echo "  history_keypoint_offsets=${history_keypoint_offsets[*]}"
echo "  action_exclude_indices=${action_exclude_indices[*]} retained_action_dims=$((50 - ${#action_exclude_indices[@]}))"
echo "  latent_fm: snr_shift=${latent_snr_shift}, train_timesteps=${latent_fm_num_train_timesteps}"
echo "  keypoint_fm: snr_shift=${keypoint_snr_shift}, train_timesteps=${keypoint_fm_num_train_timesteps}"
echo "  keypoint_stats=${keypoint_stats_path}"
echo "  keypoint validity: joint FM channel, invalid=-0.5 valid=+0.5 threshold=0"
echo "  keypoint_expert=${keypoint_expert_checkpoint} hidden=${keypoint_hidden_dim} ffn=${keypoint_ffn_dim}"
echo "  keypoint_cond_video_mode=${keypoint_cond_video_mode} keypoint_text_cross_attention=${keypoint_use_text_cross_attention}"
echo "  action_patch_embedding_mode=${action_patch_embedding_mode}"
echo "  latent_condition: prob=${condition_noise_prob}, timestep=${condition_min_timestep}-${condition_max_timestep}"
echo "  keypoint_condition: prob=${keypoint_condition_noise_prob}, timestep=${keypoint_condition_min_timestep}-${keypoint_condition_max_timestep}"
echo "  batch_size=${batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} effective_global_batch=${effective_global_batch}"
echo "  num_workers_per_process=${num_workers} total_dataloader_workers=$((num_processes * num_workers))"
echo "  lr: video=${learning_rate} keypoint_multiplier=${keypoint_lr_multiplier} action_multiplier=${action_lr_multiplier}"
echo "  gradient_as_bucket_view=${gradient_as_bucket_view}"
echo "  gradient_checkpointing=${use_gradient_checkpointing} full_layer=${checkpoint_full_layer}"
echo "  encode_prompts_with_t5=${encode_prompts_with_t5} cache_text_embeddings=${cache_text_embeddings} precomputed=${precomputed_text_embeddings_path}"
echo "  resume=${resume:-none} weights_only=${resume_weights_only}"
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
    --action_exclude_indices "${action_exclude_indices[@]}" \
    --action_target_frames "${action_target_frames}" \
    --prompt_template "${prompt_template}" \
    --empty_task_name "${empty_task_name}" \
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
    --latent_loss_weight "${latent_loss_weight}" \
    --keypoint_loss_weight "${keypoint_loss_weight}" \
    --action_loss_weight "${action_loss_weight}" \
    --wandb_project "${wandb_project}" \
    --wandb_name "${wandb_name}" \
    --wandb_mode "${wandb_mode}" \
    "${optional_args[@]}" \
    "$@"
