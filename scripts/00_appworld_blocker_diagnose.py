"""Collect diagnostics when `appworld install/download data` is blocked.

This script does not require AppWorld data to be present. It checks package
versions, CLI help output, relevant environment variables, common writable
paths, and basic connectivity to likely package/data hosts.

Example:
    python scripts/00_appworld_blocker_diagnose.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"


URLS = [
    "https://pypi.org/simple/appworld/",
    "https://github.com/StonyBrookNLP/appworld",
    "https://huggingface.co/",
]


HOSTS = [
    "pypi.org",
    "github.com",
    "huggingface.co",
]


COMMANDS = [
    ["python", "-m", "pip", "show", "appworld"],
    ["python", "-m", "appworld.cli", "--help"],
    ["python", "-m", "appworld.cli", "download", "--help"],
    ["python", "-m", "appworld.cli", "install", "--help"],
]


def get_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def run_command(command: list[str], timeout: int = 60) -> dict[str, object]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "command": command,
            "ok": False,
            "returncode": None,
            "seconds": round(time.time() - started, 2),
            "stdout": "",
            "stderr": repr(exc),
        }
    return {
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 2),
        "stdout": completed.stdout[-6000:],
        "stderr": completed.stderr[-6000:],
    }


def check_url(url: str, timeout: int = 15) -> dict[str, object]:
    started = time.time()
    request = Request(url, headers={"User-Agent": "appworld-agentic-rl-diagnose/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - diagnostic CLI.
            status = getattr(response, "status", None)
            final_url = response.geturl()
            sample = response.read(256)
    except Exception as exc:  # noqa: BLE001
        return {
            "url": url,
            "ok": False,
            "seconds": round(time.time() - started, 2),
            "error": repr(exc),
        }
    return {
        "url": url,
        "ok": True,
        "status": status,
        "final_url": final_url,
        "seconds": round(time.time() - started, 2),
        "sample_bytes": len(sample),
    }


def check_dns(host: str) -> dict[str, object]:
    started = time.time()
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except Exception as exc:  # noqa: BLE001
        return {
            "host": host,
            "ok": False,
            "seconds": round(time.time() - started, 2),
            "error": repr(exc),
        }
    ips = sorted({item[4][0] for item in addresses})
    return {
        "host": host,
        "ok": True,
        "seconds": round(time.time() - started, 2),
        "ips": ips[:8],
    }


def path_info(path: str) -> dict[str, object]:
    expanded = Path(path).expanduser()
    info: dict[str, object] = {
        "path": str(expanded),
        "exists": expanded.exists(),
        "is_dir": expanded.is_dir(),
    }
    try:
        expanded.mkdir(parents=True, exist_ok=True)
        test_file = expanded / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        info["writable"] = True
    except Exception as exc:  # noqa: BLE001
        info["writable"] = False
        info["write_error"] = repr(exc)
    return info


def markdown(results: dict[str, object]) -> str:
    lines = [
        "# AppWorld Download Blocker Diagnosis",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Python: `{results['python']}`",
        f"- AppWorld version: `{results['versions']['appworld']}`",
        f"- Pydantic version: `{results['versions']['pydantic']}`",
        "",
        "## Environment",
        "",
    ]
    env = results["env"]
    for key, value in env.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Paths", ""])
    for item in results["paths"]:
        lines.append(
            f"- `{item['path']}` exists=`{item['exists']}` "
            f"is_dir=`{item['is_dir']}` writable=`{item.get('writable')}`"
        )

    lines.extend(["", "## DNS", ""])
    for item in results["dns"]:
        lines.append(f"- `{item['host']}` ok=`{item['ok']}` seconds=`{item['seconds']}`")
        if item.get("error"):
            lines.append(f"  error: `{item['error']}`")

    lines.extend(["", "## URL Checks", ""])
    for item in results["urls"]:
        lines.append(f"- `{item['url']}` ok=`{item['ok']}` seconds=`{item['seconds']}`")
        if item.get("status"):
            lines.append(f"  status: `{item['status']}`")
        if item.get("error"):
            lines.append(f"  error: `{item['error']}`")

    lines.extend(["", "## CLI Commands", ""])
    for item in results["commands"]:
        lines.append(f"### `{' '.join(item['command'])}`")
        lines.append("")
        lines.append(f"- ok: `{item['ok']}`")
        lines.append(f"- returncode: `{item['returncode']}`")
        lines.append(f"- seconds: `{item['seconds']}`")
        if item["stdout"]:
            lines.append("")
            lines.append("```text")
            lines.append(str(item["stdout"]))
            lines.append("```")
        if item["stderr"]:
            lines.append("")
            lines.append("```text")
            lines.append(str(item["stderr"]))
            lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "versions": {
            "appworld": get_version("appworld"),
            "pydantic": get_version("pydantic"),
            "torch": get_version("torch"),
        },
        "env": {
            key: os.environ.get(key)
            for key in [
                "APPWORLD_ROOT",
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "HF_DATASETS_CACHE",
                "HTTP_PROXY",
                "HTTPS_PROXY",
            ]
        },
        "paths": [
            path_info("/root/autodl-fs/datasets/appworld_root"),
            path_info("/root/autodl-fs/outputs/logs"),
            path_info("/root/autodl-tmp"),
        ],
        "dns": [check_dns(host) for host in HOSTS],
        "urls": [check_url(url) for url in URLS],
        "commands": [run_command(command) for command in COMMANDS],
    }

    json_path = REPORT_DIR / "appworld_blocker_diagnose.json"
    md_path = REPORT_DIR / "appworld_blocker_diagnose.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown(results), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
