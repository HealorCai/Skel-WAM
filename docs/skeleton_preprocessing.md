# EgoVLA skeleton dataset preprocessing

Generate skeleton-overlay videos and keypoints from recorded HDF5 episodes (`<task>/episode_*.hdf5`). Install Skel-WAM dependencies, including `pxr`, and run from the repository root:

```bash
python utils/egovla_utils/project_robot_hand_skeleton_egovla_ds.py \
  --root data/EgoVLA_SIM/raw_hdf5 \
  --out_root data/EgoVLA_SIM/skeleton_video_384x384 \
  --keypoints_out_root data/EgoVLA_SIM/skeleton_keypoints \
  --norm_keypoints --keypoints_include_z \
  --workers 8 --use_fk_hand \
  --joint_names_file eval/egovla/utils/joint_names.txt \
  --usd_path eval/egovla/utils/h1_inspire_convex_decomp.usd \
  --skeleton_cfg utils/egovla_utils/config/h1_inspire_both_hands_v2.yaml \
  --skip_existing
```

Set `--root` to your raw HDF5 dataset. Outputs preserve the task/episode structure; keypoints contain normalized x/y and camera-space depth z. Video resolution follows the input; the command does not resize to 384×384.

Use `--no_video` to export keypoints only, or `--help` for other options.
