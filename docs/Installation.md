# Installation

This guide covers environment setup for Skel-WAM training and offline evaluation. Run all project commands from the repository root. The Python package name is `skel_wam`.

## 1. Prerequisites

The instructions below use the following versions from the existing development environment:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.6.0+cu118 |
| torchvision | 0.21.0+cu118 |
| flash-attn | 2.7.3 |

## 2. Create a Conda Environment

```bash
conda create -n skelwam python=3.10 -y
conda activate skelwam
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

Enter your local `Skel-WAM` repository:

```bash
cd /path/to/Skel-WAM
pip install -e .
```

Install FlashAttention:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

or

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
