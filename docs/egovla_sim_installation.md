# EgoVLA-Sim Environment Installation

First follow [Installation](Installation.md) to create the `skelwam` environment and install Skel-WAM dependencies. Activate it with `conda activate skelwam` before running the commands below.

The benchmark specifies **IsaacLab 1.2.0 and Isaac Sim 4.2.0**; later versions can change physics behavior.

## 1. Install IsaacLab

```bash
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

git clone --branch v1.2.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
export IsaacLab_PATH="$PWD"

apt install cmake build-essential

./isaaclab.sh --install
```

## 2. Clone Ego_Humanoid_Manipulation_Benchmark

Clone the forked benchmark repository, then download simulation assets directly from [EgoVLA/Assets](https://huggingface.co/datasets/EgoVLA/Assets).

```bash
cd ..
git clone https://github.com/HealorCai/Ego_Humanoid_Manipulation_Benchmark.git
cd Ego_Humanoid_Manipulation_Benchmark

hf download EgoVLA/Assets \
  --repo-type dataset \
  --local-dir source/extensions/humanoid.tasks/humanoid/tasks/data
```

The download path is relative to the benchmark repository root. These simulation assets are separate from the Skel-WAM training dataset.

## 3. Install the Humanoid Tasks Extension

Activate the environment configured for IsaacLab and run from the benchmark repository root:

```bash
"$IsaacLab_PATH/isaaclab.sh" -p -m pip install -e source/extensions/humanoid.tasks
```

## 4. Adjust Rendering and Physics (Recommended)

Replace these files in your IsaacLab installation with the corresponding versions supplied by the benchmark repository:

- `source/apps/isaaclab.python.kit`
- `source/apps/isaaclab.python.rendering.kit`

Apply the same changes to headless app configurations if needed. In newer IsaacLab versions, these files are under `apps/` rather than `source/apps/` and may not be directly replaceable; update the corresponding settings manually in that case.

## 5. Check the simulator

From the benchmark root, run its random-action example:

```bash
"$IsaacLab_PATH/isaaclab.sh" -p scripts/random_agent.py \
  --task Humanoid-Push-Box-v0 --num_envs 1 --enable_cameras --headless
```

Stop the example after confirming that the environment and cameras initialize. A successful `pip install` alone does not verify rendering, assets, or task registration.

Then follow [Evaluation in EgoVLA-Sim](egovla_sim_evaluation.md).
