"""Check whether the AppWorld project environment is ready.

This script is intentionally lightweight: it does not train anything and does
not require a GPU. Run it after installing requirements and downloading
AppWorld data.

Example:
    python scripts/00_install_check.py
    python scripts/00_install_check.py --run-verify-tests
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


PACKAGES = [
    "appworld",
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "peft",
    "datasets",
    "bitsandbytes",
    "pydantic",
]


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report the real import failure.
        return {"ok": False, "error": repr(exc)}
    return {"ok": True, "module_file": getattr(module, "__file__", None)}


def run_command(command: list[str], timeout: int = 600) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - command availability is part of the check.
        return {
            "command": command,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "started_at": started_at,
        }
    return {
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "started_at": started_at,
    }


def load_appworld_dataset_summary() -> dict[str, Any]:
    try:
        from appworld import load_task_ids
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}

    summary: dict[str, Any] = {"ok": True, "splits": {}}
    for split in ["train", "dev", "test_normal", "test_challenge"]:
        try:
            task_ids = load_task_ids(split)
        except Exception as exc:  # noqa: BLE001
            summary["splits"][split] = {"ok": False, "error": repr(exc)}
            continue
        summary["splits"][split] = {
            "ok": True,
            "count": len(task_ids),
            "first_task_ids": task_ids[:5],
        }
    return summary


def torch_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(index).total_memory / (1024**3), 2
                    ),
                }
            )
    return {
        "ok": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": cuda_devices,
    }


def write_reports(results: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / "install_check.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# AppWorld Install Check",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Python: `{results['python']['version']}`",
        f"- Platform: `{results['python']['platform']}`",
        f"- AppWorld CLI: `{results['appworld_cli'] or 'NOT FOUND'}`",
        "",
        "## Package Versions",
        "",
    ]
    for package_name, package_info in results["packages"].items():
        import_ok = package_info["import"]["ok"]
        lines.append(f"- `{package_name}`: `{package_info['version']}` import_ok=`{import_ok}`")

    lines.extend(["", "## Torch", ""])
    torch_info = results["torch"]
    lines.append(f"- CUDA available: `{torch_info.get('cuda_available')}`")
    lines.append(f"- CUDA version: `{torch_info.get('cuda_version')}`")
    lines.append(f"- GPU count: `{torch_info.get('device_count')}`")
    for device in torch_info.get("devices", []):
        lines.append(
            "- GPU "
            f"{device['index']}: `{device['name']}`, "
            f"capability `{device['capability']}`, "
            f"memory `{device['total_memory_gb']} GB`"
        )

    lines.extend(["", "## Dataset Splits", ""])
    dataset = results["appworld_dataset"]
    if not dataset["ok"]:
        lines.append(f"- Could not load task ids: `{dataset['error']}`")
    else:
        for split, split_info in dataset["splits"].items():
            if split_info["ok"]:
                lines.append(
                    f"- `{split}`: {split_info['count']} tasks; "
                    f"first ids: `{split_info['first_task_ids']}`"
                )
            else:
                lines.append(f"- `{split}`: failed, `{split_info['error']}`")

    if results.get("verify_tests"):
        verify = results["verify_tests"]
        lines.extend(["", "## AppWorld Verify Tests", ""])
        lines.append(f"- Command: `{' '.join(verify['command'])}`")
        lines.append(f"- Return code: `{verify['returncode']}`")
        lines.append(f"- OK: `{verify['ok']}`")
        if verify["stderr"]:
            lines.append("")
            lines.append("```text")
            lines.append(verify["stderr"])
            lines.append("```")

    md_path = REPORT_DIR / "install_check.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-verify-tests",
        action="store_true",
        help="Also run `appworld verify tests` after import checks. This can take a few minutes.",
    )
    parser.add_argument("--verify-timeout", type=int, default=900)
    args = parser.parse_args()

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "appworld_cli": shutil.which("appworld"),
        "packages": {},
    }

    for package_name in PACKAGES:
        results["packages"][package_name] = {
            "version": package_version(package_name),
            "import": import_status(package_name),
        }

    results["torch"] = torch_summary()
    results["appworld_dataset"] = load_appworld_dataset_summary()

    if args.run_verify_tests:
        results["verify_tests"] = run_command(
            ["appworld", "verify", "tests"],
            timeout=args.verify_timeout,
        )

    write_reports(results)

    failed_imports = [
        name for name, info in results["packages"].items() if not info["import"]["ok"]
    ]
    dataset_ok = results["appworld_dataset"]["ok"] and any(
        split.get("ok") for split in results["appworld_dataset"].get("splits", {}).values()
    )

    print(f"Wrote {REPORT_DIR / 'install_check.md'}")
    print(f"Wrote {REPORT_DIR / 'install_check.json'}")
    if failed_imports:
        print(f"Failed imports: {failed_imports}")
    if not dataset_ok:
        print("AppWorld task ids could not be loaded. Run `appworld download data` and retry.")
    return 1 if failed_imports or not dataset_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
