# Kaggle Setup

Kaggle's base notebook image preinstalls many libraries that require
Pydantic v2. AppWorld 0.1.x requires Pydantic v1, so installing AppWorld into
the global Kaggle environment will trigger dependency warnings and may break
unrelated Kaggle packages.

Use an isolated virtual environment and run project scripts through that
environment's Python.

## 1. Create Environment

```bash
cd /kaggle/working/<your-repo-dir>
PROJECT_DIR="$(pwd)"
python -m pip install --user virtualenv
python -m virtualenv /kaggle/working/appworld_venv
/kaggle/working/appworld_venv/bin/python -m pip install --upgrade pip setuptools wheel
/kaggle/working/appworld_venv/bin/python -m pip install wrapt
/kaggle/working/appworld_venv/bin/python -m pip install -r "$PROJECT_DIR/requirements.txt"
```

Use `virtualenv` instead of `python -m venv` on Kaggle because some Kaggle
images fail while running `ensurepip` during standard-library venv creation.

## 2. Install AppWorld Data

```bash
/kaggle/working/appworld_venv/bin/appworld install
/kaggle/working/appworld_venv/bin/appworld download data
```

If the CLI is not visible:

```bash
/kaggle/working/appworld_venv/bin/python -m appworld.cli install
/kaggle/working/appworld_venv/bin/python -m appworld.cli download data
```

## 3. Run C0 Audit

```bash
cd /kaggle/working/<your-repo-dir>
/kaggle/working/appworld_venv/bin/python scripts/00_install_check.py --run-verify-tests
/kaggle/working/appworld_venv/bin/python scripts/01_appworld_audit.py --split train --num-tasks 3 --run-verify-tasks
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
%cd /kaggle/working/<your-repo-dir>
!/kaggle/working/appworld_venv/bin/python scripts/00_install_check.py
!/kaggle/working/appworld_venv/bin/python scripts/01_appworld_audit.py --split train --num-tasks 3
```

Avoid running `pip install -r requirements.txt` in the global notebook kernel.
