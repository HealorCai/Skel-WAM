# Training

Complete [Installation](Installation.md) and [Data and Training Assets Preparation](data_and_training_assets.md), then run the commands below from the repository root with the training environment activated.

## 1. EgoVLA-Sim

```bash
bash scripts/train_skel_wam_multi_node.sh
```

## 2. EgoVLA-Sim ablations

```bash
# Without skeleton overlay
bash scripts_ablation/train_skel_wam_wo_overlay_multi_node.sh

# Without keypoints
bash scripts_ablation/train_skel_wam_wo_keypoint_multi_node.sh

# Without both
bash scripts_ablation/train_skel_wam_wo_both_multi_node.sh
```

## 3. Real robot

```bash
bash scripts/train_skel_wam_real_robot_multi_node.sh
```

## 4. Real robot-human cotraining

Train the backbone first, then use its checkpoint to train the Action Expert.

```bash
# Stage 1: backbone
bash scripts/train_skel_wam_real_cotrain_multi_node.sh

# Stage 2: Action Expert
BACKBONE_CHECKPOINT=/path/to/backbone.pt \
  bash scripts/train_skel_wam_real_cotrain_action_multi_node.sh
```

## 5. Real cotrain ablations

For each variant, train the backbone first and use its matching checkpoint for the Action Expert.

### Without skeleton overlay

```bash
bash scripts_ablation/train_skel_wam_real_cotrain_wo_overlay_multi_node.sh

BACKBONE_CHECKPOINT=/path/to/wo_overlay/backbone.pt \
  bash scripts_ablation/train_skel_wam_real_cotrain_action_wo_overlay_multi_node.sh
```

### Without keypoints

```bash
bash scripts_ablation/train_skel_wam_real_cotrain_wo_keypoint_multi_node.sh

BACKBONE_CHECKPOINT=/path/to/wo_keypoint/backbone.pt \
  bash scripts_ablation/train_skel_wam_real_cotrain_action_wo_keypoint_multi_node.sh
```

### Without both

```bash
bash scripts_ablation/train_skel_wam_real_cotrain_wo_both_multi_node.sh

BACKBONE_CHECKPOINT=/path/to/wo_both/backbone.pt \
  bash scripts_ablation/train_skel_wam_real_cotrain_action_wo_both_multi_node.sh
```
