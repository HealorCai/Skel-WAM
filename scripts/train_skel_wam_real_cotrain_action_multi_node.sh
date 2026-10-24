#!/usr/bin/env bash
# Train a portable 42-D real-robot Action head on top of a real robot/human
# cotrain backbone. The Video/Keypoint Transformer and T5 are not loaded here.
set -euo pipefail
DRY_RUN="${DRY_RUN:-0}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

backbone_checkpoint="${BACKBONE_CHECKPOINT:-}"
resume="${RESUME:-}"

real_root="${REAL_ROBOT_ROOT:-${REAL_ROBOT_ROOT:-data/real_robot}}"
dataset_root="${real_root}/skeleton_video_384x384"
keypoint_root="${real_root}/skeleton_keypoints"
action_root="${real_root}/action"
keypoint_stats_path="${KEYPOINT_STATS_PATH:-utils/real_utils/keypoint_stats_real_human_robot.json}"
action_stats_path="${ACTION_STATS_PATH:-utils/real_utils/action_stats_real_robot.json}"
vae_checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
output_root="${OUTPUT_ROOT:-checkpoints/skel_wam/real_robot_cotrain/action}"
task_include="${TASK_INCLUDE:-}"
# task_include=stack_cups
task_exclude=

num_frames=73
action_video_freq_ratio=6
global_sample_stride=1
action_target_frames=${num_frames}
action_dim=42
image_size=(384 384)
action_condition_mode=latent_keypoint
condition_noise_prob=0.5
condition_min_timestep=0.0
condition_max_timestep=0.2
keypoint_condition_noise_prob=0.0
keypoint_condition_min_timestep=0.0
keypoint_condition_max_timestep=0.0

batch_size="${BATCH_SIZE:-8}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
num_workers="${NUM_WORKERS:-4}"
max_steps="${MAX_STEPS:-50000}"
save_step_every="${SAVE_STEP_EVERY:-2000}"
learning_rate=1.0e-4
min_learning_rate=1.0e-5
lr_scheduler=warmup_cosine
warmup_steps=2000
dtype=bfloat16
gradient_as_bucket_view=1

wandb_project=skel-wam-real-cotrain-action
wandb_mode="${WANDB_MODE:-offline}"
wandb_name=

ngpu_per_node="${GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
num_machines="${NUM_MACHINES:-${SLURM_NNODES:-1}}"
machine_rank="${MACHINE_RANK:-${SLURM_NODEID:-0}}"
num_processes=$((ngpu_per_node * num_machines))
if [[ -n "${MASTER_ADDR:-}" ]]; then
  master_addr="${MASTER_ADDR}"
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  master_addr=127.0.0.1
fi
master_port="${MASTER_PORT:-29500}"
run_id="${RUN_ID:-${SLURM_JOB_ID:-$(date "+%Y-%m-%d_%H-%M-%S")}}"
if [[ -z "${wandb_name}" ]]; then
  wandb_name="skel_wam_real_cotrain_action_${task_include}_${num_frames}f_${run_id}"
fi
output_dir="${output_root}/${wandb_name}"

if ! [[ "${ngpu_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer: ${ngpu_per_node}" >&2
  exit 1
fi
if ! [[ "${num_machines}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_NNODES must be a positive integer: ${num_machines}" >&2
  exit 1
fi
if ! [[ "${machine_rank}" =~ ^[0-9]+$ ]] || ((machine_rank >= num_machines)); then
  echo "Invalid machine rank ${machine_rank}/${num_machines}" >&2
  exit 1
fi
if ((num_machines > 1)) && [[ "${master_addr}" == "127.0.0.1" ]]; then
  echo "Multi-node training requires a reachable MASTER_ADDR." >&2
  exit 1
fi

required_paths=(
  "${backbone_checkpoint}"
  "${dataset_root}"
  "${keypoint_root}"
  "${action_root}"
  "${vae_checkpoint_dir}/Wan2.2_VAE.pth"
  "${keypoint_stats_path}"
  "${action_stats_path}"
)
[[ -n "${resume}" ]] && required_paths+=("${resume}")
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

optional_args=()
[[ -n "${resume}" ]] && optional_args+=(--resume "${resume}")
[[ -n "${task_include}" ]] && optional_args+=(--task_include "${task_include}")
[[ -n "${task_exclude}" ]] && optional_args+=(--task_exclude "${task_exclude}")
[[ "${gradient_as_bucket_view}" == "1" ]] && optional_args+=(--gradient_as_bucket_view) || optional_args+=(--no-gradient_as_bucket_view)

accelerate_args=(
  --num_machines "${num_machines}"
  --num_processes "${num_processes}"
  --machine_rank "${machine_rank}"
  --main_process_ip "${master_addr}"
  --main_process_port "${master_port}"
)
if ((num_processes > 1)); then
  accelerate_args=(--multi_gpu "${accelerate_args[@]}")
fi

effective_global_batch=$((num_processes * batch_size * gradient_accumulation_steps))
echo "Training portable real-cotrain Action head"
echo "  backbone=${backbone_checkpoint}"
echo "  data=${real_root} task=${task_include:-all}"
echo "  frames: action/raw=${num_frames}, video/keypoint=$(((num_frames - 1) / action_video_freq_ratio + 1)), ratio=${action_video_freq_ratio}"
echo "  action_dim=${action_dim} keypoint_stats=${keypoint_stats_path}"
echo "  processes=${num_processes} effective_global_batch=${effective_global_batch}"
echo "  lr=${learning_rate} scheduler=${lr_scheduler} warmup=${warmup_steps} min_lr=${min_learning_rate}"
echo "  output=${output_dir}"
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

accelerate launch \
  "${accelerate_args[@]}" \
  main/train_skel_wam_action.py \
  --backbone_checkpoint "${backbone_checkpoint}" \
  --dataset_root "${dataset_root}" \
  --keypoint_root "${keypoint_root}" \
  --action_root "${action_root}" \
  --keypoint_stats_path "${keypoint_stats_path}" \
  --action_stats_path "${action_stats_path}" \
  --vae_checkpoint_dir "${vae_checkpoint_dir}" \
  --output_dir "${output_dir}" \
  --wandb_project "${wandb_project}" \
  --wandb_name "${wandb_name}" \
  --wandb_mode "${wandb_mode}" \
  --num_frames "${num_frames}" \
  --global_sample_stride "${global_sample_stride}" \
  --action_video_freq_ratio "${action_video_freq_ratio}" \
  --action_target_frames "${action_target_frames}" \
  --action_dim "${action_dim}" \
  --image_size "${image_size[@]}" \
  --action_condition_mode "${action_condition_mode}" \
  --condition_noise_prob "${condition_noise_prob}" \
  --condition_min_timestep "${condition_min_timestep}" \
  --condition_max_timestep "${condition_max_timestep}" \
  --keypoint_condition_noise_prob "${keypoint_condition_noise_prob}" \
  --keypoint_condition_min_timestep "${keypoint_condition_min_timestep}" \
  --keypoint_condition_max_timestep "${keypoint_condition_max_timestep}" \
  --batch_size "${batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --num_workers "${num_workers}" \
  --max_steps "${max_steps}" \
  --save_step_every "${save_step_every}" \
  --learning_rate "${learning_rate}" \
  --min_learning_rate "${min_learning_rate}" \
  --lr_scheduler "${lr_scheduler}" \
  --warmup_steps "${warmup_steps}" \
  --dtype "${dtype}" \
  "${optional_args[@]}" \
  "$@"
