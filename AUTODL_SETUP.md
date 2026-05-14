# AutoDL Setup

AutoDL images often configure pip to use the Aliyun mirror by default. Some
newer or niche packages, including AppWorld, may not be available there. Prefer
the Tsinghua mirror for speed, and install AppWorld from official PyPI if the
mirror cannot find it.

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

Use the Tsinghua mirror for most packages:

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If the mirror cannot find AppWorld, install AppWorld from official PyPI first,
then install the rest through Tsinghua:

```bash
python -m pip install appworld -i https://pypi.org/simple
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If official PyPI is slow from your instance, use the official index only for
AppWorld with a longer timeout:

```bash
python -m pip install appworld -i https://pypi.org/simple --timeout 120
```

## 5. Install AppWorld Data

Put AppWorld data under persistent storage:

```bash
export APPWORLD_ROOT=/root/autodl-fs/datasets/appworld_root
mkdir -p "$APPWORLD_ROOT"
echo "APPWORLD_ROOT=$APPWORLD_ROOT" > .env
```

```bash
appworld install
appworld download data
```

If the CLI is unavailable:

```bash
python -m appworld.cli install
python -m appworld.cli download data
```

If the data download is slow, interrupted, or fails without a clear message,
run it in `tmux` and save the log:

```bash
tmux new -s appworld_download
conda activate appworld_rl
cd /root/autodl-fs/repos/appworld-agentic-rl
export APPWORLD_ROOT=/root/autodl-fs/datasets/appworld_root
python -m appworld.cli download data 2>&1 | tee /root/autodl-fs/outputs/logs/appworld_download.log
```

If it still fails, inspect the command and package path:

```bash
python -m appworld.cli download --help
python - <<'PY'
import appworld
print(appworld.__file__)
PY
```

Then send the last 50 lines of the log:

```bash
tail -n 50 /root/autodl-fs/outputs/logs/appworld_download.log
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
