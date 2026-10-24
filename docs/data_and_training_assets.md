# Data and Training Assets Preparation

## 1. Create Local Directories

```bash
mkdir -p data
mkdir -p checkpoints
mkdir -p output
```

## 2. Download EgoVLA-Sim Dataset

The original dataset is [EgoVLA/EgoVLA-Humanoid-Sim](https://huggingface.co/datasets/EgoVLA/EgoVLA-Humanoid-Sim). We provide a processed version for Skel-WAM below.

Download the `EgoVLA_SIM` folder from [Healor/Skel-WAM_EgoVLA-sim](https://huggingface.co/datasets/Healor/Skel-WAM_EgoVLA-sim):

```bash
hf download Healor/Skel-WAM_EgoVLA-sim \
  --repo-type dataset \
  --include "EgoVLA_SIM/**" \
  --local-dir data
```

## 3. Download Real Dataset

Download the `real_human` and `real_robot` folders from [Healor/Skel-WAM_real](https://huggingface.co/datasets/Healor/Skel-WAM_real):

```bash
hf download Healor/Skel-WAM_real \
  --repo-type dataset \
  --include "real_human/**" \
  --include "real_robot/**" \
  --local-dir data
```

The datasets are saved under `data/real_human/` and `data/real_robot/`.

## 4. Download Backbone Checkpoints

Download [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B):

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B
```

<!-- Download [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B):

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
``` -->

## 5. Download Training Assets

Download the five training assets from [Healor/Skel-WAM](https://huggingface.co/Healor/Skel-WAM/tree/main/training_assets):

```bash
hf download Healor/Skel-WAM \
  --repo-type model \
  --include "training_assets/*.pt" \
  --local-dir checkpoints/skel_wam
```

The files are saved under `checkpoints/skel_wam/training_assets/`:

| File | Purpose |
| --- | --- |
| `keypoint_expert_1024.pt` | Keypoint Expert initialization weights |
| `text_embeddings_egovla.pt` | EgoVLA-Sim text embeddings with skeleton overlay prompts |
| `text_embeddings_egovla_wo_overlay.pt` | EgoVLA-Sim text embeddings without overlay prompts |
| `text_embeddings.pt` | Real robot text embeddings with skeleton overlay prompts |
| `text_embeddings_wo_overlay.pt` | Real human–robot cotraining text embeddings without overlay prompts |

## Directory Layout

```text
Skel-WAM/
├── data/
│   ├── EgoVLA_SIM/
│   │   ├── qpos/
│   │   ├── skeleton_keypoints/
│   │   ├── skeleton_video_384x384/
│   │   └── video_384x384/
│   ├── real_human/
│   │   ├── skeleton_keypoints/
│   │   ├── skeleton_video_384x384/
│   │   └── video_384x384/
│   └── real_robot/
│       ├── action/
│       ├── skeleton_keypoints/
│       ├── skeleton_video_384x384/
│       └── video_384x384/
├── checkpoints/
│   ├── Wan-AI/
│   │   ├── Wan2.2-TI2V-5B/
│   │   └── Wan2.1-T2V-1.3B/
│   └── skel_wam/
│       └── training_assets/
│           ├── keypoint_expert_1024.pt
│           ├── text_embeddings_egovla.pt
│           ├── text_embeddings_egovla_wo_overlay.pt
│           ├── text_embeddings.pt
│           └── text_embeddings_wo_overlay.pt
└── output/
```

## Regenerate text embeddings (optional)

Run from the repository root after preparing the environment, datasets, and Wan2.2 backbone. These commands overwrite caches in `checkpoints/skel_wam/training_assets/`.

### EgoVLA-Sim

```bash
# With skeleton overlay
bash utils/egovla_utils/precompute_text_embeddings.sh

# Without skeleton overlay
TEXT_EMBEDDINGS_PATH=checkpoints/skel_wam/training_assets/text_embeddings_egovla_wo_overlay.pt \
  bash utils/egovla_utils/precompute_text_embeddings.sh \
  --prompt_template "An egocentric video of a robot executing the following task: {task}"
```

### Real robot and robot-human cotraining

Uses `data/real_robot/video_384x384` and `data/real_human/video_384x384` to discover tasks.

```bash
# Robot + human, with skeleton overlay
python utils/real_utils/precompute_text_embeddings.py --mode human_robot

# Robot + human, without skeleton overlay
python utils/real_utils/precompute_text_embeddings.py --mode human_robot --wo-overlay
```

Outputs are `text_embeddings.pt` and `text_embeddings_wo_overlay.pt`, respectively. Add `--print-prompts-only` to preview prompts without loading T5.

## Recompute normalization statistics (optional)

Run from the repository root. These commands overwrite the corresponding JSON files in `utils/egovla_utils/` or `utils/real_utils/`; use `--output` to save elsewhere. For existing checkpoints, keep the statistics used during training.

### EgoVLA-Sim

```bash
python utils/egovla_utils/compute_qpos_stats.py
python utils/egovla_utils/compute_keypoint_stats.py
```

### Real robot and robot-human cotraining

```bash
python utils/real_utils/compute_action_stats.py
python utils/real_utils/compute_keypoint_stats.py --mode robot
python utils/real_utils/compute_keypoint_stats.py --mode human_robot
```
