#!/usr/bin/env python3
"""Repeated live-MuJoCo recovery trials with seeded initial-state perturbations.

This runs the actual pick-and-place simulation as a fresh process for each
trial. It is an exploratory simulation experiment, not physical robot testing.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("grasp", "transfer", "insertion", "placement")


def extract(pattern: str, text: str, cast=str, default=None):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return default
    try:
        return cast(match.group(1))
    except (TypeError, ValueError):
        return default


def run_trial(repo: Path, sim_script: Path, stage: str, seed: int,
              args, log_dir: Path) -> dict:
    rng = random.Random(seed)
    xy = rng.uniform(0.001, args.max_xy_m)
    yaw = rng.uniform(0.25, args.max_yaw_deg)
    joint = rng.uniform(0.05, args.max_joint_deg)
    velocity = rng.uniform(0.005, args.max_joint_velocity)
    command = [
        sys.executable, str(sim_script),
        "--mode", "sim", "--no-viewer",
        "--inject-failure", stage,
        "--perturb-seed", str(seed),
        "--canister-xy-perturb-m", f"{xy:.8f}",
        "--canister-yaw-perturb-deg", f"{yaw:.6f}",
        "--joint-position-perturb-deg", f"{joint:.6f}",
        "--joint-velocity-perturb-rad-s", f"{velocity:.6f}",
        "--save-perception", str(log_dir / f"perception_{stage}_{seed}.png"),
        "--settle-seconds", str(args.settle_seconds),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=repo, capture_output=True, text=True,
            timeout=args.timeout_seconds, check=False,
            env={**__import__("os").environ,
                 "PYTHONPATH": str(repo / "python/src/robauditor/src")},
        )
        elapsed = time.perf_counter() - started
        output = completed.stdout + "\n" + completed.stderr
        (log_dir / f"{stage}_{seed}.log").write_text(output, encoding="utf-8")
        manager_status = extract(r"\[recovery\] manager status:\s*(\S+)", output)
        backend_status = extract(r"\[recovery-backend\] status=\s*(\S+)", output)
        backend_success = None
        postconditions = extract(r"\[recovery-backend\] postconditions_verified=\s*(True|False)", output, lambda x: x == "True")
        strategy = extract(r"\[recovery\] selected strategies:\s*(.*)", output, str, "")
        waypoint_error = extract(r"(?:waypoint|pre-grasp).*?error:\s*([0-9.eE+-]+)\s*mm", output, float)
        object_displacement = extract(r"(?:canister|object).*?displacement:\s*([0-9.eE+-]+)\s*mm", output, float)
        contact_count = extract(r"contact_count['\"]?\s*[:=]\s*(\d+)", output, int)
        return {
            "stage": stage, "seed": seed, "return_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 6),
            "canister_xy_bound_m": round(xy, 6),
            "canister_yaw_bound_deg": round(yaw, 4),
            "joint_position_bound_deg": round(joint, 4),
            "joint_velocity_bound_rad_s": round(velocity, 5),
            "manager_status": manager_status,
            "backend_status": backend_status,
            "backend_success": backend_success,
            "postconditions_verified": postconditions,
            "strategy": strategy,
            "waypoint_error_mm": waypoint_error,
            "object_displacement_mm": object_displacement,
            "contact_count": contact_count,
            "trial_pass": bool(completed.returncode == 0 and backend_status == "SUCCEEDED" and postconditions is True),
            "error": "" if completed.returncode == 0 else output[-1800:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        (log_dir / f"{stage}_{seed}_timeout.log").write_text(
            str(exc), encoding="utf-8"
        )
        return {
            "stage": stage, "seed": seed, "return_code": None,
            "elapsed_seconds": round(elapsed, 6),
            "canister_xy_bound_m": xy, "canister_yaw_bound_deg": yaw,
            "joint_position_bound_deg": joint,
            "joint_velocity_bound_rad_s": velocity,
            "manager_status": None, "backend_status": None,
            "backend_success": None, "postconditions_verified": None,
            "strategy": "", "waypoint_error_mm": None,
            "object_displacement_mm": None, "contact_count": None,
            "trial_pass": False, "error": "subprocess timeout",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per stage (default 5; 20 stages total)")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--settle-seconds", type=float, default=0.1)
    parser.add_argument("--max-xy-m", type=float, default=0.005)
    parser.add_argument("--max-yaw-deg", type=float, default=3.0)
    parser.add_argument("--max-joint-deg", type=float, default=0.25)
    parser.add_argument("--max-joint-velocity", type=float, default=0.03)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--sim-script", type=Path,
                        default=Path("python/src/robauditor/src/canister_pick_and_place.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/live_mujoco_perturbation"))
    args = parser.parse_args()
    if args.trials < 1 or args.timeout_seconds <= 0 or args.settle_seconds < 0:
        parser.error("trials and timeout must be positive; settle time cannot be negative")
    if not (0 < args.max_xy_m <= 0.015 and 0 < args.max_yaw_deg <= 10
            and 0 < args.max_joint_deg <= 2 and 0 < args.max_joint_velocity <= 0.5):
        parser.error("Perturbation bounds exceed safe experimental limits")

    repo = Path.cwd()
    sim_script = args.sim_script if args.sim_script.is_absolute() else repo / args.sim_script
    if not sim_script.is_file():
        parser.error(f"Simulation script not found: {sim_script}")
    outdir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = outdir / f"logs_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = args.trials * len(args.stages)
    for stage_index, stage in enumerate(args.stages):
        for trial in range(args.trials):
            seed = args.seed + stage_index * 100_000 + trial
            row = run_trial(repo, sim_script, stage, seed, args, log_dir)
            rows.append(row)
            print(
                f"[{len(rows)}/{total}] {stage} seed={seed}: "
                f"{'PASS' if row['trial_pass'] else 'FAIL'} "
                f"backend={row['backend_status']} "
                f"elapsed={row['elapsed_seconds']:.3f}s"
            )

    passed = sum(row["trial_pass"] for row in rows)
    summary = {
        "trials": len(rows), "passed": passed, "failed": len(rows) - passed,
        "success_rate": round(passed / len(rows), 6) if rows else 0.0,
        "by_stage": {},
        "scope": (
            "Live MuJoCo kinematic task-level recovery under bounded seeded "
            "initial-state perturbations. This is not physical grasp/dynamics, "
            "sensor-noise, collision-free planning, or hardware validation."
        ),
        "perturbation_bounds": {
            "canister_xy_m": args.max_xy_m,
            "canister_yaw_deg": args.max_yaw_deg,
            "joint_position_deg": args.max_joint_deg,
            "joint_velocity_rad_s": args.max_joint_velocity,
        },
    }
    for stage in args.stages:
        group = [row for row in rows if row["stage"] == stage]
        ok = sum(row["trial_pass"] for row in group)
        summary["by_stage"][stage] = {
            "trials": len(group), "passed": ok, "failed": len(group) - ok,
            "success_rate": round(ok / len(group), 6) if group else 0.0,
            "mean_elapsed_seconds": round(
                sum(row["elapsed_seconds"] for row in group) / len(group), 6
            ) if group else None,
        }
    csv_path = outdir / f"live_mujoco_perturbation_{stamp}.csv"
    json_path = outdir / f"live_mujoco_perturbation_{stamp}.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed, "trials_per_stage": args.trials,
        "summary": summary, "trials": rows,
    }, indent=2), encoding="utf-8")
    print("\n=== LIVE MUJOCO PERTURBATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}\nLogs: {log_dir}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
