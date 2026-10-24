# Quick Test

Complete [Installation](Installation.md) and [Data and Training Assets Preparation](data_and_training_assets.md) first. Run commands from the repository root with the training environment activated. These scripts run offline inference on dataset samples using one GPU (GPU 0 by default).

## Download checkpoints

Download the model checkpoints from [Healor/Skel-WAM](https://huggingface.co/Healor/Skel-WAM), using the same scenario and experiment directories as the training scripts:

```bash
hf download Healor/Skel-WAM \
  --repo-type model \
  --include "egovla_sim/**/step_*_weights.pt" \
  --include "egovla_sim_ablations/**/step_*_weights.pt" \
  --include "real_robot/**/step_*_weights.pt" \
  --include "real_robot_cotrain/**/step_*_weights.pt" \
  --include "real_robot_cotrain_ablations/**/step_*_weights.pt" \
  --local-dir checkpoints/skel_wam

CKPT_ROOT=checkpoints/skel_wam
```

This downloads only `step_*_weights.pt` checkpoint files (including `step_*_action_weights.pt`), excluding training assets and other files. Download `training_assets/` separately using the preparation guide. 

To download only one checkpoint, pass its repository-relative path:

```bash
hf download Healor/Skel-WAM \
  "egovla_sim/skel_wam_robot_all_37f_384x384_20260810-180053/checkpoints/step_080000_weights.pt" \
  --repo-type model --local-dir checkpoints/skel_wam
```

## 1. EgoVLA-Sim

```bash
CHECKPOINT="$CKPT_ROOT/egovla_sim/skel_wam_robot_all_37f_384x384_20260810-180053/checkpoints/step_080000_weights.pt" \
  bash scripts/test_skel_wam.sh
```

## 2. EgoVLA-Sim ablations

```bash
# wo_overlay
CHECKPOINT="$CKPT_ROOT/egovla_sim_ablations/wo_overlay/skel_wam_ablation_wo_overlay_robot_all_37f_384x384_20260818-092110/checkpoints/step_080000_weights.pt" \
  bash scripts_ablation/test_skel_wam_wo_overlay.sh
```

```bash
# wo_keypoint
CHECKPOINT="$CKPT_ROOT/egovla_sim_ablations/wo_keypoint/skel_wam_ablation_wo_keypoint_robot_all_37f_384x384_20260818-083518/checkpoints/step_080000_weights.pt" \
  bash scripts_ablation/test_skel_wam_wo_keypoint.sh
```

```bash
# wo_both
CHECKPOINT="$CKPT_ROOT/egovla_sim_ablations/wo_both/skel_wam_ablation_wo_both_robot_all_37f_384x384_20260818-083522/checkpoints/step_080000_weights.pt" \
  bash scripts_ablation/test_skel_wam_wo_both.sh
```

## 3. Real robot

```bash
CHECKPOINT="$CKPT_ROOT/real_robot/skel_wam_real_robot_all_73f_384x384_20260819-160827/checkpoints/step_080000_weights.pt" \
  bash scripts/test_skel_wam_real_robot.sh
```

## 4. Real robot-human cotraining

Test the backbone on both robot and human samples:

```bash
CHECKPOINT="$CKPT_ROOT/real_robot_cotrain/skel_wam_real_cotrain_2to1_scratch_20260820-121000/checkpoints/step_080000_weights.pt" \
TEST_DOMAIN=both \
  bash scripts/test_skel_wam_real_cotrain.sh
```

To include robot action prediction, pair the backbone with its Action checkpoint:

```bash
CHECKPOINT="$CKPT_ROOT/real_robot_cotrain/skel_wam_real_cotrain_2to1_scratch_20260820-121000/checkpoints/step_080000_weights.pt" \
ACTION_CHECKPOINT="$CKPT_ROOT/real_robot_cotrain/action/skel_wam_real_cotrain_action__73f_20260824-033921/step_050000_action_weights.pt" \
TEST_DOMAIN=robot RUN_ROBOT_ACTION=1 \
  bash scripts/test_skel_wam_real_cotrain.sh
```

## 5. Real cotrain ablations

Use the backbone and Action checkpoint from the same variant:

```bash
# wo_overlay
CHECKPOINT="$CKPT_ROOT/real_robot_cotrain_ablations/wo_overlay/skel_wam_real_cotrain_wo_overlay_2to1_scratch_20260821-061648/checkpoints/step_080000_weights.pt" \
ACTION_CHECKPOINT="$CKPT_ROOT/real_robot_cotrain_ablations/wo_overlay/action/skel_wam_real_cotrain_action_wo_overlay__73f_20260824-044615/step_050000_action_weights.pt" \
TEST_DOMAIN=robot RUN_ROBOT_ACTION=1 \
  bash scripts_ablation/test_skel_wam_real_cotrain_wo_overlay.sh
```

```bash
# wo_keypoint
CHECKPOINT="$CKPT_ROOT/real_robot_cotrain_ablations/wo_keypoint/skel_wam_real_cotrain_wo_keypoint_2to1_scratch_20260821-013914/checkpoints/step_080000_weights.pt" \
ACTION_CHECKPOINT="$CKPT_ROOT/real_robot_cotrain_ablations/wo_keypoint/action/skel_wam_real_cotrain_action_wo_keypoint__73f_20260824-044619/step_050000_action_weights.pt" \
TEST_DOMAIN=robot RUN_ROBOT_ACTION=1 \
  bash scripts_ablation/test_skel_wam_real_cotrain_wo_keypoint.sh
```

```bash
# wo_both
CHECKPOINT="$CKPT_ROOT/real_robot_cotrain_ablations/wo_both/skel_wam_real_cotrain_wo_both_2to1_scratch_20260821-013922/checkpoints/step_080000_weights.pt" \
ACTION_CHECKPOINT="$CKPT_ROOT/real_robot_cotrain_ablations/wo_both/action/skel_wam_real_cotrain_action_wo_both__73f_20260824-044607/step_050000_action_weights.pt" \
TEST_DOMAIN=robot RUN_ROBOT_ACTION=1 \
  bash scripts_ablation/test_skel_wam_real_cotrain_wo_both.sh
```

For backbone-only evaluation, omit `ACTION_CHECKPOINT` and set `TEST_DOMAIN=both RUN_ROBOT_ACTION=0`.

## Test your own trained checkpoints

Use the same scenario-specific scripts above, replacing `CHECKPOINT` with your local `step_*_weights.pt` and, for cotrain action prediction, `ACTION_CHECKPOINT` with the matching `step_*_action_weights.pt`. Keep the training normalization statistics, temporal settings, and ablation variant consistent.

```bash
# Example: your own EgoVLA-Sim training run
CHECKPOINT="checkpoints/skel_wam/egovla_sim/<run>/checkpoints/step_080000_weights.pt" \
  bash scripts/test_skel_wam.sh

# Example: your own cotrain backbone and Action runs
CHECKPOINT="checkpoints/skel_wam/real_robot_cotrain/<backbone_run>/checkpoints/step_080000_weights.pt" \
ACTION_CHECKPOINT="checkpoints/skel_wam/real_robot_cotrain/action/<action_run>/step_050000_action_weights.pt" \
TEST_DOMAIN=robot RUN_ROBOT_ACTION=1 \
  bash scripts/test_skel_wam_real_cotrain.sh
```

Replace the `<...>` placeholders with your run directories and select the saved step you want to test. Custom `OUTPUT_ROOT` training runs may use different paths.

## Test outputs

Results are saved under the corresponding scenario directory:

- `output/egovla_sim/test/`
- `output/egovla_sim_ablations/<variant>/test/`
- `output/real_robot/test/`
- `output/real_robot_cotrain/test/`
- `output/real_robot_cotrain_ablations/<variant>/test/`

Set `CUDA_VISIBLE_DEVICES` to select a GPU, `TASK_INCLUDE` to select a task, and `OUTPUT_DIR` for an explicit result directory. All test scripts accept `SAMPLE_IDX` to select a sample.

For closed-loop simulator evaluation, see [Evaluation in EgoVLA-Sim](egovla_sim_evaluation.md).
