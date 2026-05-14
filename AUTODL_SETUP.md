# AutoDL Setup

AutoDL images often configure pip to use the Aliyun mirror by default. Some
newer or niche packages, including AppWorld, may not be available there. Use
the official PyPI index when installing this project's requirements.

## 1. Enter Repo

```bash
cd ~/autodl-tmp/appworld-agentic-rl
```

## 2. Create Environment

```bash
conda create -n appworld_rl python=3.11 -y
conda activate appworld_rl
pip install --upgrade pip setuptools wheel
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
