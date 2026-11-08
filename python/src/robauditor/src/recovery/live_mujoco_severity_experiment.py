#!/usr/bin/env python3
"""Run staged-severity, seeded initial-state perturbation trials in MuJoCo.

Each trial launches a fresh simulation process. Results are written as one
combined CSV and JSON report, with separate logs for every trial.

Scope: kinematic task-level recovery in MuJoCo. This is not physical grasp
or dynamics validation, sensor-noise testing, collision-free planning, or
hardware validation.
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

STAGES = ("grasp", "transfer", "insertion", "placement")
LEVELS = {
    "baseline": {
        "xy_m": 0.005, "yaw_deg": 3.0,
        "joint_deg": 0.25, "joint_velocity": 0.03,
    },
    "moderate": {
        "xy_m": 0.010, "yaw_deg": 5.0,
        "joint_deg": 0.5, "joint_velocity": 0.1,
    },
    "high": {
        "xy_m": 0.015, "yaw_deg": 8.0,
        "joint_deg": 1.0, "joint_velocity": 0.2,
    },
}


def extract(pattern: str, text: str, cast=str, default=None):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return default
    try:
        return cast(match.group(1))
    except (TypeError, ValueError):
        return default


def run_trial(repo: Path, sim_script: Path, level: str, stage: str,
              seed: int, bounds: dict, args, log_dir: Path) -> dict:
    # Draw actual perturbation magnitudes uniformly from a non-zero interval
    # up to the selected level's maximum. Signs/axes are randomized by the
    # simulation's seeded perturbation function.
    import random
    rng = random.Random(seed)
    xy = rng.uniform(0.001, bounds["xy_m"])
    yaw = rng.uniform(0.25, bounds["yaw_deg"])
    joint = rng.uniform(0.05, bounds["joint_deg"])
    velocity = rng.uniform(0.005, bounds["joint_velocity"])
    command = [
        sys.executable, str(sim_script),
        "--mode", "sim", "--no-viewer",
        "--inject-failure", stage,
        "--perturb-seed", str(seed),
        "--canister-xy-perturb-m", f"{xy:.8f}",
        "--canister-yaw-perturb-deg", f"{yaw:.6f}",
        "--joint-position-perturb-deg", f"{joint:.6f}",
        "--joint-velocity-perturb-rad-s", f"{velocity:.6f}",
        "--save-perception", str(log_dir / f"perception_{level}_{stage}_{seed}.png"),
        "--settle-seconds", str(args.settle_seconds),
    ]
    started = time.perf_counter()
    log_path = log_dir / f"{level}_{stage}_{seed}.log"
    try:
        completed = subprocess.run(
            command, cwd=repo, capture_output=True, text=True,
            timeout=args.timeout_seconds, check=False,
            env={**os.environ, "PYTHONPATH": str(repo / "python/src/robauditor/src")},
        )
        elapsed = time.perf_counter() - started
        output = completed.stdout + "\n" + completed.stderr
        log_path.write_text(output, encoding="utf-8")
        backend_status = extract(r"\[recovery-backend\] status=\s*(\S+)", output)
        postconditions = extract(
            r"\[recovery-backend\] postconditions_verified=\s*(True|False)",
            output, lambda value: value == "True",
        )
        manager_status = extract(r"\[recovery\] manager status:\s*(\S+)", output)
        strategy = extract(r"\[recovery\] selected strategies:\s*(.*)", output, str, "")
        return {
            "level": level, "stage": stage, "seed": seed,
            "return_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 6),
            "canister_xy_bound_m": round(xy, 8),
            "canister_yaw_bound_deg": round(yaw, 6),
            "joint_position_bound_deg": round(joint, 6),
            "joint_velocity_bound_rad_s": round(velocity, 6),
            "manager_status": manager_status,
            "backend_status": backend_status,
            "postconditions_verified": postconditions,
            "strategy": strategy,
            "trial_pass": bool(
                completed.returncode == 0
                and backend_status == "SUCCEEDED"
                and postconditions is True
            ),
            "error": "" if completed.returncode == 0 else output[-2400:],
            "log_file": str(log_path),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        detail = (exc.stdout or "")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        detail += "\nTIMEOUT: simulation exceeded configured timeout\n"
        log_path.write_text(detail, encoding="utf-8")
        return {
            "level": level, "stage": stage, "seed": seed,
            "return_code": None, "elapsed_seconds": round(elapsed, 6),
            "canister_xy_bound_m": round(xy, 8),
            "canister_yaw_bound_deg": round(yaw, 6),
            "joint_position_bound_deg": round(joint, 6),
            "joint_velocity_bound_rad_s": round(velocity, 6),
            "manager_status": None, "backend_status": None,
            "postconditions_verified": None, "strategy": "",
            "trial_pass": False, "error": "subprocess timeout",
            "log_file": str(log_path),
        }


def summarize(rows: list[dict], levels: list[str], stages: list[str]) -> dict:
    passed = sum(bool(row["trial_pass"]) for row in rows)
    by_level = {}
    for level in levels:
        group = [row for row in rows if row["level"] == level]
        ok = sum(bool(row["trial_pass"]) for row in group)
        by_stage = {}
        for stage in stages:
            subset = [row for row in group if row["stage"] == stage]
            sub_ok = sum(bool(row["trial_pass"]) for row in subset)
            times = [row["elapsed_seconds"] for row in subset]
            by_stage[stage] = {
                "trials": len(subset), "passed": sub_ok,
                "failed": len(subset) - sub_ok,
                "success_rate": round(sub_ok / len(subset), 6) if subset else None,
                "mean_elapsed_seconds": round(sum(times) / len(times), 6) if times else None,
            }
        by_level[level] = {
            "bounds": LEVELS[level], "trials": len(group),
            "passed": ok, "failed": len(group) - ok,
            "success_rate": round(ok / len(group), 6) if group else None,
            "mean_elapsed_seconds": round(
                sum(row["elapsed_seconds"] for row in group) / len(group), 6
            ) if group else None,
            "by_stage": by_stage,
        }
    return {
        "trials": len(rows), "passed": passed,
        "failed": len(rows) - passed,
        "success_rate": round(passed / len(rows), 6) if rows else 0.0,
        "by_level": by_level,
        "scope": (
            "Live MuJoCo kinematic task-level recovery under bounded seeded "
            "initial-state perturbations. Not physical grasp/dynamics, sensor-noise, "
            "collision-free planning, or hardware validation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5, help="Trials per stage per severity level")
    parser.add_argument("--levels", nargs="+", choices=tuple(LEVELS), default=["moderate"],
                        help="Severity levels to run, in order (default: moderate)")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--settle-seconds", type=float, default=0.1)
    parser.add_argument("--sim-script", type=Path,
                        default=Path("python/src/robauditor/src/canister_pick_and_place.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/live_mujoco_severity"))
    args = parser.parse_args()
    if args.trials < 1 or args.timeout_seconds <= 0 or args.settle_seconds < 0:
        parser.error("trials and timeout must be positive; settle time cannot be negative")
    if len(set(args.levels)) != len(args.levels):
        parser.error("Do not specify a severity level more than once")
    repo = Path.cwd()
    sim_script = args.sim_script if args.sim_script.is_absolute() else repo / args.sim_script
    if not sim_script.is_file():
        parser.error(f"Simulation script not found: {sim_script}")
    outdir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = outdir / f"logs_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = args.trials * len(args.stages) * len(args.levels)
    for level_index, level in enumerate(args.levels):
        bounds = LEVELS[level]
        for stage_index, stage in enumerate(args.stages):
            for trial_index in range(args.trials):
                seed = args.seed + level_index * 1_000_000 + stage_index * 100_000 + trial_index
                row = run_trial(repo, sim_script, level, stage, seed, bounds, args, log_dir)
                rows.append(row)
                print(
                    f"[{len(rows)}/{total}] {level}/{stage} seed={seed}: "
                    f"{'PASS' if row['trial_pass'] else 'FAIL'} "
                    f"backend={row['backend_status']} elapsed={row['elapsed_seconds']:.3f}s",
                    flush=True,
                )

    summary = summarize(rows, args.levels, args.stages)
    csv_path = outdir / f"live_mujoco_severity_{stamp}.csv"
    json_path = outdir / f"live_mujoco_severity_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed, "trials_per_stage_per_level": args.trials,
        "levels": args.levels, "stages": args.stages,
        "summary": summary, "trials": rows,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== LIVE MUJOCO SEVERITY SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}\nLogs: {log_dir}")
    return 0 if rows and summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
