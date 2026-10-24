# EgoVLA-Sim evaluation

See [installation](../../docs/egovla_sim_installation.md) and [evaluation commands](../../docs/egovla_sim_evaluation.md).

| Model | Launcher | Agent |
| --- | --- | --- |
| Skel-WAM | `skel_wam_eval.sh` | `skel_wam_agent.py` |
| Without overlay | `skel_wam_wo_overlay_eval.sh` | `skel_wam_wo_overlay_agent.py` |
| Without keypoints | `skel_wam_wo_keypoint_eval.sh` | `skel_wam_wo_keypoint_agent.py` |
| Without both | `skel_wam_wo_both_eval.sh` | `skel_wam_wo_both_agent.py` |

Compute seen/unseen/overall SR and PSR (use a separate results directory for each model/run):

```bash
python eval/egovla/stat_success_rate.py output/egovla/skel_wam
```

For offline dataset processing, see [Skeleton preprocessing](../../docs/skeleton_preprocessing.md).
