#!/usr/bin/env python3
"""Automated headless regression runner for RobAuditor MuJoCo recovery.

Run from the repository root:
  PYTHONPATH=python/src/robauditor/src \
    python python/src/robauditor/src/recovery/automated_recovery_regression.py

The runner starts a fresh simulator process for the baseline and each failure
scenario, captures a full log per run, and writes CSV/JSON summaries.
It does not modify project source files or execute real-robot commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCENARIOS = [
    {
        "name": "GOB_grasp",
        "stage": "grasp",
        "strategy": "GOB",
        "action": "return_to_pre_grasp",
        "success_markers": [
            "[recovery-execution] postconditions: PASS",
            "[recovery-backend] success=True",
        ],
    },
    {
        "name": "GBB_transfer",
        "stage": "transfer",
        "strategy": "GBB",
        "action": "return_to_last_safe_waypoint",
        "success_markers": [
            "[recovery-execution] GBB postconditions: PASS",
            "[recovery-backend] success=True",
        ],
    },
    {
        "name": "INA_pre_grasp",
        "stage": "pre_grasp",
        "strategy": "INA",
        "action": "replan_pre_grasp",
        "success_markers": [
            "[recovery-execution] INA postconditions: PASS",
            "[recovery-backend] success=True",
        ],
    },
    {
        "name": "ESS_lift",
        "stage": "lift",
        "strategy": "ESS",
        "action": "stop_robot",
        "success_markers": [
            "[recovery-execution] ESS safe-hold checks PASS",
            "[recovery-backend] success=True",
        ],
    },
    {
        "name": "INA_insertion",
        "stage": "insertion",
        "strategy": "INA",
        "action": "replan_insertion",
        "success_markers": [
            "[recovery-execution] INA insertion postconditions: PASS",
            "[recovery-backend] success=True",
        ],
    },
    {
        "name": "GOB_placement",
        "stage": "placement",
        "strategy": "GOB",
        "action": "return_to_placement_waypoint",
        "success_markers": [
            "[recovery-execution] GOB placement postconditions: PASS",
            "[recovery-backend] success=True",
        ],
    },
    {
        "name": "INO_object_unavailable",
        "stage": "object_unavailable",
        "strategy": "INO",
        "action": "introduce_replacement_canister",
        "success_markers": [
            "[recovery-execution] INO recovery: SUCCEEDED",
            "[recovery-backend] success=True",
        ],
        "additional_markers": [
            "[success] canister is placed within the selected tray opening tolerance",
        ],
    },
]


def run_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
) -> tuple[int | None, str, float, str | None]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return completed.returncode, completed.stdout, elapsed, None
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return None, output, elapsed, f"Timed out after {timeout_seconds:g} seconds"
    except Exception as exc:  # record launch errors in the report
        elapsed = time.perf_counter() - start
        return None, "", elapsed, f"{type(exc).__name__}: {exc}"


def evaluate_run(
    output: str,
    return_code: int | None,
    launch_error: str | None,
    required_markers: list[str],
    expected_action: str | None = None,
) -> tuple[bool, list[str]]:
    missing = [marker for marker in required_markers if marker not in output]
    if expected_action and f"[recovery-backend] executed actions: {expected_action}" not in output:
        missing.append(f"executed action: {expected_action}")
    if launch_error:
        missing.append(f"process error: {launch_error}")
    if return_code != 0:
        missing.append(f"process exit code: {return_code}")
    return not missing, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current working directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/recovery_experiments/automated_regression"),
        help="Directory for logs and reports (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=240.0,
                        help="Maximum seconds per simulation process (default: 240)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for child processes")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the no-failure baseline simulation")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    main_script = project_root / "python/src/robauditor/src/canister_pick_and_place.py"
    if not main_script.is_file():
        print(f"[error] Main simulation script not found: {main_script}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    package_src = str(project_root / "python/src/robauditor/src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        package_src + os.pathsep + existing_pythonpath
        if existing_pythonpath else package_src
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    runs: list[dict[str, Any]] = []
    if not args.skip_baseline:
        runs.append({
            "name": "baseline",
            "stage": "",
            "strategy": "",
            "action": "",
            "required_markers": [
                "[success] canister is placed within the selected tray opening tolerance",
            ],
            "additional_markers": [],
            "failure_stage": None,
        })

    for scenario in SCENARIOS:
        runs.append({
            **scenario,
            "required_markers": scenario["success_markers"],
            "failure_stage": scenario["stage"],
        })

    print(f"[regression] project root: {project_root}")
    print(f"[regression] output directory: {output_dir}")
    print(f"[regression] runs scheduled: {len(runs)}")
    print("[regression] mode: MuJoCo simulation, headless; no hardware commands")

    for index, run in enumerate(runs, start=1):
        command = [
            args.python,
            str(main_script),
            "--mode", "sim",
            "--no-viewer",
        ]
        if run["failure_stage"]:
            command.extend(["--inject-failure", run["failure_stage"]])

        print(f"\n[{index}/{len(runs)}] Running {run['name']} ...")
        return_code, output, elapsed, launch_error = run_process(
            command, project_root, env, args.timeout
        )

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run["name"])
        log_path = output_dir / f"{safe_name}.log"
        log_path.write_text(output, encoding="utf-8")

        ok, missing = evaluate_run(
            output,
            return_code,
            launch_error,
            run["required_markers"] + run.get("additional_markers", []),
            run["action"] or None,
        )
        # Preserve a clear distinction between a failed recovery and a process
        # that did not shut down cleanly, even if success markers appeared.
        row = {
            "scenario": run["name"],
            "failure_stage": run["stage"],
            "expected_strategy": run["strategy"],
            "expected_action": run["action"],
            "passed": ok,
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 4),
            "missing_checks": "; ".join(missing),
            "log_file": str(log_path.relative_to(project_root))
            if log_path.is_relative_to(project_root) else str(log_path),
        }
        rows.append(row)
        print(f"[{ 'PASS' if ok else 'FAIL' }] {run['name']} "
              f"({elapsed:.2f}s, exit={return_code})")
        if missing:
            for item in missing:
                print(f"  - {item}")
        print(f"  log: {log_path}")

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    fieldnames = [
        "scenario", "failure_stage", "expected_strategy", "expected_action",
        "passed", "return_code", "elapsed_seconds", "missing_checks", "log_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed_count = sum(bool(row["passed"]) for row in rows)
    summary = {
        "created_at_utc": timestamp,
        "project_root": str(project_root),
        "execution_mode": "simulation",
        "headless": True,
        "hardware_commands_sent": False,
        "total_runs": len(rows),
        "passed": passed_count,
        "failed": len(rows) - passed_count,
        "all_passed": passed_count == len(rows),
        "results": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[regression] SUMMARY")
    print(f"  passed: {passed_count}/{len(rows)}")
    print(f"  failed: {len(rows) - passed_count}/{len(rows)}")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Logs: {output_dir}")

    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
