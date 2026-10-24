# Evaluation in EgoVLA-Sim

Complete [environment installation](egovla_sim_installation.md), then run from the repository root. For recorded data, see [Quick Test](quick_test.md) and [Skeleton preprocessing](skeleton_preprocessing.md).

## 1. Prepare the checkpoint

Use a released or newly trained EgoVLA-Sim checkpoint:

```bash
conda activate skelwam
export IsaacLab_PATH=/path/to/IsaacLab
cd /path/to/Skel-WAM

export CHECKPOINT="$PWD/checkpoints/skel_wam/egovla_sim/skel_wam_robot_all_37f_384x384_20260810-180053/checkpoints/step_080000_weights.pt"
```

Defaults: Wan backbone at `checkpoints/Wan-AI/Wan2.2-TI2V-5B`, text cache at `checkpoints/skel_wam/training_assets/text_embeddings_egovla.pt`. Statistics and evaluation resources are included; download benchmark assets during environment installation.

Match the launcher to the checkpoint variant. Real-robot checkpoints are incompatible.

```bash
# Set CHECKPOINT to the corresponding ablation weights before each command.
bash eval/egovla/skel_wam_wo_overlay_eval.sh
bash eval/egovla/skel_wam_wo_keypoint_eval.sh
bash eval/egovla/skel_wam_wo_both_eval.sh
```

Ablations select the matching text cache and save to `output/egovla/skel_wam_<variant>`.

## 2. Evaluate one task

```bash
TASK_NAME=Close-Drawer GPU_IDS="0" \
EVAL_LABEL=skel_wam_close_drawer \
  bash eval/egovla/skel_wam_eval.sh
```

Runs headlessly with cameras: **25 room/table settings, 93 rollouts per task** (27 seen + 66 unseen). `NUM_TRIALS` repeats each episode.

To distribute settings across GPUs:

```bash
TASK_NAME=Close-Drawer GPU_IDS="0 1 2 3" \
EVAL_LABEL=skel_wam_close_drawer \
  bash eval/egovla/skel_wam_eval.sh
```

Use space-separated GPU IDs; each GPU runs an independent simulator and model.

## 3. Evaluate all seven tasks

```bash
for task in Close-Drawer Flip-Mug Open-Drawer Open-Laptop Pour-Balls Push-Box Stack-Can; do
  TASK_NAME="$task" GPU_IDS="0 1 2 3" \
  EVAL_LABEL="skel_wam_${task}" \
    bash eval/egovla/skel_wam_eval.sh || break
done
```

## 4. Outputs

By default, results are saved under:

```text
output/egovla/skel_wam/<EVAL_LABEL>/
  result_log.txt
  <task>-room_<room>-table_<table>/
    <task>_room_<room>_table_<table>_episode_*.mp4
    video_pred_<task>_room_<room>_table_<table>_episode_*.mp4
```

`result_log.txt` appends rollout results and subtask success flags. Use a new `EVAL_LABEL` for each run. Repeated trials can overwrite episode videos because filenames omit the trial index.

## 5. Overrides and scheduling preview

| Variable | Default / purpose |
| --- | --- |
| `CHECKPOINT` | Required main-model checkpoint |
| `IsaacLab_PATH` | Required absolute IsaacLab checkout path |
| `TASK_NAME` | `Push-Box` |
| `GPU_IDS` | `0`; space-separated IDs |
| `NUM_TRIALS` | `1` |
| `EVAL_OUTPUT_ROOT` | `output/egovla/skel_wam` |
| `EVAL_LABEL` | `skel_wam_<TASK_NAME>` for the main model |
| `WAN_CHECKPOINT_DIR` | Wan backbone directory |
| `TEXT_EMBEDDINGS_PATH` | Precomputed prompt cache |
| `KEYPOINT_STATS_PATH`, `ACTION_STATS_PATH` | Training-compatible normalization statistics |

Keep inference settings consistent with the checkpoint. Launchers accept environment overrides, not appended Python arguments.

Preview scheduling without starting the simulator:

```bash
DRY_RUN=1 TASK_NAME=Push-Box GPU_IDS="0 1" \
  bash eval/egovla/skel_wam_eval.sh
```

Set `CHECKPOINT` before previewing. Dry run checks scheduling only; end-to-end simulator validation remains pending.

## Success-rate statistics

```bash
python eval/egovla/stat_success_rate.py output/egovla/skel_wam
python eval/egovla/stat_success_rate.py output/egovla/skel_wam_wo_overlay
python eval/egovla/stat_success_rate.py output/egovla/skel_wam_wo_keypoint
python eval/egovla/stat_success_rate.py output/egovla/skel_wam_wo_both
```

Reports seen (rooms 1–3, table 1), unseen, and overall SR/PSR. PSR aggregates partial-success stages. Keep each model/run in a separate input directory.
