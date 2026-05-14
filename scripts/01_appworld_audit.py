"""Run the first AppWorld environment audit.

The audit opens a few train/dev task worlds, executes a harmless code snippet,
checks task completion behavior, calls `complete_task()` prematurely, and runs
the evaluator. This validates reset, execution, completion detection, and
failure scoring before any model training begins.

Example:
    python scripts/01_appworld_audit.py --split train --num-tasks 3
    python scripts/01_appworld_audit.py --run-verify-tasks
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
RAW_AUDIT_DIR = PROJECT_ROOT / "data" / "raw" / "appworld_audit"


def serialize_object(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [serialize_object(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_object(item) for key, item in value.items()}
    return repr(value)


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001 - protected AppWorld objects may compute attributes lazily.
        return default


def evaluate_report(world: Any) -> dict[str, Any]:
    try:
        evaluation = world.evaluate()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}

    result: dict[str, Any] = {
        "ok": True,
        "type": type(evaluation).__name__,
        "repr": repr(evaluation),
    }
    for attr_name in ["success", "score", "num_tests_passed", "num_tests", "test_results"]:
        attr_value = safe_getattr(evaluation, attr_name)
        if attr_value is not None:
            result[attr_name] = serialize_object(attr_value)
    try:
        result["report"] = evaluation.report()
    except Exception as exc:  # noqa: BLE001
        result["report_error"] = repr(exc)
    return result


def collect_ground_truth_summary(task: Any) -> dict[str, Any]:
    ground_truth = safe_getattr(task, "ground_truth")
    if ground_truth is None:
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "type": type(ground_truth).__name__,
        "metadata": serialize_object(safe_getattr(ground_truth, "metadata")),
    }
    for attr_name in [
        "required_apps",
        "required_apis",
        "answer",
        "api_calls",
        "evaluation_code",
        "compiled_solution_module",
    ]:
        attr_value = safe_getattr(ground_truth, attr_name)
        if attr_value is not None:
            summary[attr_name] = serialize_object(attr_value)
    return summary


def run_task_audit(task_id: str, experiment_name: str, ground_truth_mode: str) -> dict[str, Any]:
    from appworld import AppWorld

    task_result: dict[str, Any] = {
        "task_id": task_id,
        "ok": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    init_kwargs = {
        "task_id": task_id,
        "experiment_name": experiment_name,
    }
    if ground_truth_mode:
        init_kwargs["ground_truth_mode"] = ground_truth_mode

    try:
        with AppWorld(**init_kwargs) as world:
            task = world.task
            task_result["instruction"] = safe_getattr(task, "instruction")
            task_result["ground_truth"] = collect_ground_truth_summary(task)

            try:
                task_result["initial_task_completed"] = world.task_completed()
            except Exception as exc:  # noqa: BLE001
                task_result["initial_task_completed_error"] = repr(exc)

            harmless_code = "print('appworld_audit: execution_ok')"
            task_result["harmless_code"] = harmless_code
            try:
                task_result["harmless_output"] = world.execute(harmless_code)
                task_result["harmless_execution_ok"] = True
            except Exception as exc:  # noqa: BLE001
                task_result["harmless_output"] = repr(exc)
                task_result["harmless_execution_ok"] = False

            premature_complete_code = "apis.supervisor.complete_task()"
            task_result["premature_complete_code"] = premature_complete_code
            try:
                task_result["premature_complete_output"] = world.execute(premature_complete_code)
                task_result["premature_complete_execution_ok"] = True
            except Exception as exc:  # noqa: BLE001
                task_result["premature_complete_output"] = repr(exc)
                task_result["premature_complete_execution_ok"] = False

            try:
                task_result["after_complete_task_completed"] = world.task_completed()
            except Exception as exc:  # noqa: BLE001
                task_result["after_complete_task_completed_error"] = repr(exc)

            task_result["evaluation_after_premature_complete"] = evaluate_report(world)
            task_result["output_directory"] = serialize_object(safe_getattr(world, "output_directory"))
            task_result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        task_result["error"] = repr(exc)

    task_result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return task_result


def run_verify_tasks(timeout: int) -> dict[str, Any]:
    command = ["appworld", "verify", "tasks"]
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
            "stdout": "",
            "stderr": repr(exc),
        }
    return {
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }


def make_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# AppWorld Environment Audit",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Split: `{results['split']}`",
        f"- Requested tasks: `{results['num_tasks']}`",
        f"- Experiment name: `{results['experiment_name']}`",
        f"- Ground truth mode: `{results['ground_truth_mode']}`",
        "",
        "## Summary",
        "",
    ]

    tasks = results["tasks"]
    ok_count = sum(1 for item in tasks if item["ok"])
    harmless_ok = sum(1 for item in tasks if item.get("harmless_execution_ok"))
    evaluator_ok = sum(
        1
        for item in tasks
        if item.get("evaluation_after_premature_complete", {}).get("ok")
    )
    lines.append(f"- Worlds opened: `{ok_count}/{len(tasks)}`")
    lines.append(f"- Harmless execution succeeded: `{harmless_ok}/{len(tasks)}`")
    lines.append(f"- Evaluator callable: `{evaluator_ok}/{len(tasks)}`")
    lines.append("")

    lines.extend(["## Task Samples", ""])
    for item in tasks:
        lines.append(f"### `{item['task_id']}`")
        lines.append("")
        lines.append(f"- World opened: `{item['ok']}`")
        if item.get("error"):
            lines.append(f"- Error: `{item['error']}`")
            lines.append("")
            continue
        instruction = item.get("instruction") or ""
        lines.append(f"- Instruction: {instruction}")
        metadata = item.get("ground_truth", {}).get("metadata")
        lines.append(f"- Metadata: `{metadata}`")
        lines.append(f"- Initial task_completed: `{item.get('initial_task_completed')}`")
        lines.append(f"- Harmless execution OK: `{item.get('harmless_execution_ok')}`")
        lines.append(f"- After premature complete task_completed: `{item.get('after_complete_task_completed')}`")
        evaluation = item.get("evaluation_after_premature_complete", {})
        lines.append(f"- Evaluator OK: `{evaluation.get('ok')}`")
        if "success" in evaluation:
            lines.append(f"- Evaluator success: `{evaluation['success']}`")
        if "score" in evaluation:
            lines.append(f"- Evaluator score: `{evaluation['score']}`")
        lines.append("")

    if results.get("verify_tasks"):
        verify = results["verify_tasks"]
        lines.extend(["## Reference Solution Verification", ""])
        lines.append("This section uses AppWorld's official task verifier.")
        lines.append("")
        lines.append(f"- Command: `{' '.join(verify['command'])}`")
        lines.append(f"- Return code: `{verify['returncode']}`")
        lines.append(f"- OK: `{verify['ok']}`")
        if verify.get("stdout"):
            lines.append("")
            lines.append("```text")
            lines.append(verify["stdout"])
            lines.append("```")
        if verify.get("stderr"):
            lines.append("")
            lines.append("```text")
            lines.append(verify["stderr"])
            lines.append("```")

    lines.extend(
        [
            "## Next Decision",
            "",
            "Proceed to C1 base model audit only if worlds open, code execution works, "
            "and evaluator calls return structured results. If `appworld verify tasks` fails, "
            "fix the AppWorld installation or data download before training.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test_normal", "test_challenge"])
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--experiment-name", default="c0_env_audit")
    parser.add_argument(
        "--ground-truth-mode",
        default="full",
        help="Use `full` for train/dev audits. If AppWorld rejects it, retry with an empty value.",
    )
    parser.add_argument(
        "--run-verify-tasks",
        action="store_true",
        help="Also run `appworld verify tasks` to validate reference solutions. Takes a few minutes.",
    )
    parser.add_argument("--verify-timeout", type=int, default=1200)
    args = parser.parse_args()

    from appworld import load_task_ids

    task_ids = load_task_ids(args.split)[: args.num_tasks]
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "num_tasks": args.num_tasks,
        "experiment_name": args.experiment_name,
        "ground_truth_mode": args.ground_truth_mode,
        "task_ids": task_ids,
        "tasks": [],
    }

    RAW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        print(f"Auditing task {task_id}...")
        task_result = run_task_audit(
            task_id=task_id,
            experiment_name=args.experiment_name,
            ground_truth_mode=args.ground_truth_mode,
        )
        results["tasks"].append(task_result)
        (RAW_AUDIT_DIR / f"{task_id}.json").write_text(
            json.dumps(task_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.run_verify_tasks:
        print("Running `appworld verify tasks`...")
        results["verify_tasks"] = run_verify_tasks(timeout=args.verify_timeout)

    (REPORT_DIR / "env_audit.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REPORT_DIR / "env_audit.md").write_text(make_markdown(results), encoding="utf-8")

    print(f"Wrote {REPORT_DIR / 'env_audit.md'}")
    print(f"Wrote {REPORT_DIR / 'env_audit.json'}")
    print(f"Wrote raw task logs to {RAW_AUDIT_DIR}")

    all_opened = all(item["ok"] for item in results["tasks"])
    evaluator_ok = all(
        item.get("evaluation_after_premature_complete", {}).get("ok")
        for item in results["tasks"]
        if item["ok"]
    )
    return 0 if all_opened and evaluator_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
