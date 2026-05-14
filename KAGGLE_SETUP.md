# Kaggle Setup

Kaggle's base notebook image preinstalls many libraries that require
Pydantic v2. AppWorld 0.1.x requires Pydantic v1, so installing AppWorld into
the global Kaggle environment will trigger dependency warnings and may break
unrelated Kaggle packages.

Use an isolated virtual environment and run project scripts through that
environment's Python.

## 1. Create Environment

```bash
cd /kaggle/working/appworld-curriculum-agent-rl
python -m venv /kaggle/working/appworld_venv
source /kaggle/working/appworld_venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 2. Install AppWorld Data

```bash
source /kaggle/working/appworld_venv/bin/activate
appworld install
appworld download data
```

If the CLI is not visible:

```bash
python -m appworld.cli install
python -m appworld.cli download data
```

## 3. Run C0 Audit

```bash
source /kaggle/working/appworld_venv/bin/activate
python scripts/00_install_check.py --run-verify-tests
python scripts/01_appworld_audit.py --split train --num-tasks 3 --run-verify-tasks
```

The generated reports are:

```text
reports/install_check.md
reports/env_audit.md
```

## 4. Notebook Usage

In a Kaggle notebook cell, use shell commands with the virtual environment
explicitly:

```bash
!/kaggle/working/appworld_venv/bin/python scripts/00_install_check.py
!/kaggle/working/appworld_venv/bin/python scripts/01_appworld_audit.py --split train --num-tasks 3
```

Avoid running `pip install -r requirements.txt` in the global notebook kernel.
