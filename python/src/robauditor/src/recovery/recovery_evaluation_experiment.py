#!/usr/bin/env python3
"""Repeated robustness evaluation for the MuJoCo recovery scenarios.

Run from the repository root with the project virtual environment active:
  PYTHONPATH=python/src/robauditor/src python \
    python/src/robauditor/src/recovery/recovery_evaluation_experiment.py --repeat 20

Each trial is launched in a fresh process. Results (per-trial CSV, aggregate
JSON, and raw logs) are written beneath results/recovery_evaluation/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = ("grasp", "transfer", "insertion", "placement")
DEFAULT_SCRIPT = Path("python/src/robauditor/src/canister_pick_and_place.py")
DEFAULT_OUTPUT = Path("results/recovery_evaluation")

REGEX = {
    "strategy": r"^\[recovery\] test strategy: (\S+)",
    "manager_status": r"^\[recovery\] manager status: (\S+)",
    "backend_status": r"^\[recovery-backend\] status=(\S+)",
    "attempts": r"^\[recovery-backend\] attempts=(\d+)",
    "postconditions_verified": r"^\[recovery-backend\] postconditions_verified=(True|False|None)",
    "executed_actions": r"^\[recovery-backend\] executed actions: (.+)$",
    "waypoint_error_mm": (
        r"(?:pre-grasp position error|last-safe waypoint position error|"
        r"insertion tool-position error|safe-waypoint error):\s*([0-9.]+)\s*mm"
    ),
    "object_displacement_mm": (
        r"(?:canister displacement|canister movement during return|"
        r"canister motion during approach):\s*([0-9.]+)\s*mm"
    ),
    "attachment_error_mm": r"mount-relative attachment error:\s*([0-9.]+)\s*mm",
    "contact_count": r"'contact_count':\s*(\d+)",
    "environment_contact_count": r"'canister_environment_contact_count':\s*(\d+)",
    "gripper_contact_count": r"'canister_gripper_contact_count':\s*(\d+)",
}

CASTS = {
    "attempts": int,
    "waypoint_error_mm": float,
    "object_displacement_mm": float,
    "attachment_error_mm": float,
    "contact_count": int,
    "environment_contact_count": int,
    "gripper_contact_count": int,
}


def _last(pattern: str, output: str, cast: Any = str) -> Any:
    matches = re.findall(pattern, output, re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if cast is bool:
        return value == "True"
    return cast(value)


def parse_run_output(stage: str, repeat_index: int, returncode: int,
                     elapsed_s: float, output: str) -> dict[str, Any]:
    """Convert one simulation process's output into a structured trial row."""
    row: dict[str, Any] = {
        "stage": stage,
        "repeat_index": repeat_index,
        "process_returncode": returncode,
        "process_elapsed_s": round(elapsed_s, 6),
    }
    for key, pattern in REGEX.items():
        cast = bool if key == "postconditions_verified" else CASTS.get(key, str)
        row[key] = _last(pattern, output, cast)

    row["run_success"] = (
        returncode == 0
        and row["manager_status"] == "VALIDATED"
        and row["backend_status"] == "SUCCEEDED"
        and row["postconditions_verified"] is True
    )
    row["failure_reason"] = None
    if not row["run_success"]:
        if returncode != 0:
            row["failure_reason"] = f"simulation process exited with code {returncode}"
        elif row["backend_status"] != "SUCCEEDED":
            row["failure_reason"] = "backend did not report SUCCEEDED"
        elif row["postconditions_verified"] is not True:
            row["failure_reason"] = "postconditions were not verified as true"
        else:
            row["failure_reason"] = "recovery plan was not validated"
    return row


def run_scenario(stage: str, repeat_index: int, simulation_script: Path,
                 python_executable: str, env: dict[str, str],
                 timeout_s: float) -> tuple[dict[str, Any], str]:
    command = [
        python_executable, str(simulation_script), "--mode", "sim",
        "--no-viewer", "--inject-failure", stage,
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, env=env,
            timeout=timeout_s, check=False,
        )
        elapsed = time.perf_counter() - started
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return parse_run_output(stage, repeat_index, result.returncode,
                                elapsed, output), output
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        output = stdout + "\n" + stderr
        row = parse_run_output(stage, repeat_index, 124, elapsed, output)
        row["failure_reason"] = f"scenario timed out after {timeout_s} seconds"
        return row, output


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(row[key]) for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None,
                "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "stddev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all trials overall and separately for each failure stage."""
    successful = sum(bool(row["run_success"]) for row in rows)
    by_stage: dict[str, Any] = {}
    for stage in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage]
        if not stage_rows:
            continue
        stage_successes = sum(bool(row["run_success"]) for row in stage_rows)
        by_stage[stage] = {
            "trials": len(stage_rows),
            "successful_trials": stage_successes,
            "failed_trials": len(stage_rows) - stage_successes,
            "success_rate": round(stage_successes / len(stage_rows), 6),
            "process_elapsed_s": _metric_summary(stage_rows, "process_elapsed_s"),
            "waypoint_error_mm": _metric_summary(stage_rows, "waypoint_error_mm"),
            "object_displacement_mm": _metric_summary(stage_rows, "object_displacement_mm"),
            "attachment_error_mm": _metric_summary(stage_rows, "attachment_error_mm"),
        }
    total = len(rows)
    return {
        "total_trials": total,
        "successful_trials": successful,
        "failed_trials": total - successful,
        "overall_success_rate": round(successful / total, 6) if total else 0.0,
        "by_stage": by_stage,
        "interpretation": (
            "Repeated simulation trials measure repeatability under the configured "
            "simulator and failure-injection conditions. They do not establish hardware "
            "reliability, physical grasp performance, or collision safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--repeat", type=int, default=20,
                        help="number of fresh-process trials per stage (default: 20)")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    repo_root = Path.cwd()
    simulation_script = (
        args.script if args.script.is_absolute() else repo_root / args.script
    )
    if not simulation_script.is_file():
        parser.error(f"simulation script not found: {simulation_script}")
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    source_path = str(repo_root / "python/src/robauditor/src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = output_dir / f"repeat_logs_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    total_expected = len(args.stages) * args.repeat
    trial_number = 0
    for stage in args.stages:
        for repeat_index in range(1, args.repeat + 1):
            trial_number += 1
            print(
                f"\n[{trial_number}/{total_expected}] stage={stage} "
                f"trial={repeat_index}/{args.repeat}",
                flush=True,
            )
            row, log = run_scenario(
                stage, repeat_index, simulation_script, args.python, env, args.timeout
            )
            rows.append(row)
            log_name = f"{stage}_trial_{repeat_index:02d}.log"
            (log_dir / log_name).write_text(log, encoding="utf-8")
            print(
                f"[{stage} #{repeat_index}] "
                f"{'PASS' if row['run_success'] else 'FAIL'} "
                f"strategy={row['strategy']} backend={row['backend_status']} "
                f"attempts={row['attempts']} "
                f"postconditions={row['postconditions_verified']} "
                f"elapsed={row['process_elapsed_s']:.3f}s"
            )
            if not row["run_success"] and args.stop_on_failure:
                print("Stopping early because --stop-on-failure was set.")
                break
        if args.stop_on_failure and rows and not rows[-1]["run_success"]:
            break

    summary = summarize(rows)
    csv_path = output_dir / f"repeated_recovery_trials_{timestamp}.csv"
    json_path = output_dir / f"repeated_recovery_summary_{timestamp}.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_repetitions_per_stage": args.repeat,
        "requested_stages": args.stages,
        "completed_trials": len(rows),
        "simulation_script": str(simulation_script),
        "summary": summary,
        "trials": rows,
        "raw_log_directory": str(log_dir),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Repeated robustness evaluation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Raw logs: {log_dir}")
    return 0 if summary["failed_trials"] == 0 and len(rows) == total_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
