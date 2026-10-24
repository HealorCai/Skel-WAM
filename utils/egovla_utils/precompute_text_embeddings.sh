#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

checkpoint_dir="${WAN_CHECKPOINT_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
output_path="${TEXT_EMBEDDINGS_PATH:-checkpoints/skel_wam/training_assets/text_embeddings_egovla.pt}"
device=cuda
batch_size=1

tasks=(
    "Close-Drawer"
    "Flip-Mug"
    "Insert-And-Unload-Cans"
    "Insert-Cans/Left_Right_Deprecated"
    "Insert-Cans"
    "Open-Drawer"
    "Open-Laptop"
    "Pour-Balls"
    "Push-Box"
    "Sort-Cans/Single_hand_first_deprecated"
    "Sort-Cans"
    "Stack-Can-Into-Drawer"
    "Stack-Can"
    "Unload-Cans"
)

python utils/egovla_utils/precompute_text_embeddings.py \
    --checkpoint_dir "${checkpoint_dir}" \
    --output_path "${output_path}" \
    --device "${device}" \
    --batch_size "${batch_size}" \
    --tasks "${tasks[@]}" \
    "$@"
