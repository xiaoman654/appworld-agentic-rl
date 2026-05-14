# AutoDL Setup

AutoDL images often configure pip to use the Aliyun mirror by default. Some
newer or niche packages, including AppWorld, may not be available there. Use
the official PyPI index when installing this project's requirements.

This project treats AutoDL as a reproducible experiment base rather than a
single fixed development machine. Keep code, datasets, checkpoints, and logs in
stable locations so instances can be stopped, cloned, or replaced cheaply.

## 0. Recommended Layout

Use `/root/autodl-fs` for things that should survive instance changes and
`/root/autodl-tmp` for high-IO temporary caches.

```text
/root/autodl-fs/
  repos/
    appworld-agentic-rl/
  datasets/
    appworld/
  outputs/
    logs/
    reports/
    checkpoints/
    rollouts/

/root/autodl-tmp/
  hf_home/
  appworld_tmp/
  scratch/
```

Recommended environment variables:

```bash
export PROJECT_ROOT=/root/autodl-fs/repos/appworld-agentic-rl
export HF_HOME=/root/autodl-tmp/hf_home
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_home
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_home/datasets
export WANDB_DIR=/root/autodl-fs/outputs/wandb
```

Create directories:

```bash
mkdir -p /root/autodl-fs/repos /root/autodl-fs/datasets/appworld
mkdir -p /root/autodl-fs/outputs/{logs,reports,checkpoints,rollouts,wandb}
mkdir -p /root/autodl-tmp/{hf_home,appworld_tmp,scratch}
```

## 1. Enter Repo

```bash
cd /root/autodl-fs/repos
git clone https://github.com/xiaoman654/appworld-agentic-rl.git
cd appworld-agentic-rl
```

## 2. Create Environment

Use Python 3.11. AppWorld currently requires Python `>=3.11,<4.0`, and Python
3.11 is also safer than 3.12 for the broader training stack.

```bash
conda create -n appworld_rl python=3.11 -y
conda activate appworld_rl
pip install --upgrade pip setuptools wheel
```

If conda reports a plugin/libmamba solver error on AutoDL:

```bash
CONDA_NO_PLUGINS=true conda create -n appworld_rl python=3.11 -y --solver=classic
```

## 3. Install PyTorch

For CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install Project Requirements

Use official PyPI, not the default AutoDL mirror:

```bash
pip install -r requirements.txt -i https://pypi.org/simple
```

If official PyPI is slow, install AppWorld from official PyPI first, then use a
mirror for the rest:

```bash
pip install appworld -i https://pypi.org/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 5. Install AppWorld Data

```bash
appworld install
appworld download data
```

If the CLI is unavailable:

```bash
python -m appworld.cli install
python -m appworld.cli download data
```

## 6. Run C0 Audit

```bash
python scripts/00_install_check.py
python scripts/01_appworld_audit.py --split train --num-tasks 3
```

For a more complete first audit:

```bash
python scripts/00_install_check.py --run-verify-tests
python scripts/01_appworld_audit.py --split train --num-tasks 3 --run-verify-tasks
```

Generated reports:

```text
reports/install_check.md
reports/env_audit.md
```

Copy important reports to persistent output storage:

```bash
cp -r reports /root/autodl-fs/outputs/reports/c0_$(date +%Y%m%d_%H%M%S)
```

## 7. Long-Running Jobs

Use `tmux` for anything that may run for more than a few minutes:

```bash
tmux new -s appworld
conda activate appworld_rl
python scripts/00_install_check.py --run-verify-tests 2>&1 | tee /root/autodl-fs/outputs/logs/c0_install_check.log
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t appworld
```

For expensive GPU jobs, add shutdown after successful completion:

```bash
python train_or_eval.py 2>&1 | tee /root/autodl-fs/outputs/logs/job.log && /usr/bin/shutdown
```

## 8. Environment Summary

Recommended baseline:

```text
Base image: PyTorch official image or Miniconda image
OS: Ubuntu 22.04
CUDA: 12.4 if using your current PyTorch/2.5.1 image
Python env: conda appworld_rl, Python 3.11
PyTorch: install with pip cu124 wheel
AppWorld: install from official PyPI
Pydantic: v1, because AppWorld 0.1.x depends on pydantic>=1.9,<2
Main install command: pip install -r requirements.txt -i https://pypi.org/simple
```
