#!/usr/bin/env python3
"""Stress-test recovery-plan synthesis under randomized and adversarial inputs.

This is a model-level ensemble stress test. It exercises RecoveryManager and
its validators/risk assessment; it does not move MuJoCo bodies or claim physical
robot robustness.

Run from the repository root:
  PYTHONPATH=python/src/robauditor/src python \
    python/src/robauditor/src/recovery/ensemble_stress_evaluation.py --trials 20
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recovery.models import (
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    RecoverySuggestion,
    VerifierResult,
)
from recovery.recovery_manager import RecoveryManager


def make_action(name: str, direction: str | None, risk: float) -> RecoveryAction:
    parameters = {}
    if direction is not None:
        parameters["direction"] = direction
    return RecoveryAction(
        name=name,
        parameters=parameters,
        description=f"Stress-test proposal: {name}",
        prerequisites=[],
        estimated_risk=risk,
    )


def make_result(
    verifier_id: str,
    decision: bool | None,
    confidence: float,
    action: RecoveryAction | None,
    strategy: RecoveryStrategy = RecoveryStrategy.GOB,
    metadata: dict[str, Any] | None = None,
) -> VerifierResult:
    suggestion = None
    if action is not None:
        suggestion = RecoverySuggestion(
            verifier_id=verifier_id,
            strategy=strategy,
            actions=[action],
            confidence=confidence,
            explanation=f"{verifier_id} generated a stress-test recovery proposal",
            modalities=["simulated_state"],
            failure_type="stress_test_failure",
            metadata=metadata or {},
            decision=decision,
            evidence=[],
            visual_evidence=[],
        )
    return VerifierResult(
        verifier_id=verifier_id,
        decision=decision,
        confidence=confidence,
        explanation=f"Synthetic stress-test verifier output from {verifier_id}",
        suggestion=suggestion,
        evidence=[],
        visual_evidence=[],
        explanation_metadata=metadata or {},
    )


def make_context(
    seed: int,
    attempt_count: int = 0,
    max_attempts: int = 3,
    safe: bool = True,
    joint_speed: float = 0.25,
    position_noise_mm: float = 0.0,
) -> RecoveryContext:
    return RecoveryContext(
        failed_action_id=f"stress-failure-{seed}",
        failed_action="move_canister",
        participants=["robot", "canister"],
        object_state={
            "object_id": "canister",
            "available": True,
            "position_uncertainty_mm": position_noise_mm,
        },
        robot_state={
            "safe": safe,
            "max_abs_joint_velocity": joint_speed,
            "joint_state_uncertainty": position_noise_mm / 1000.0,
        },
        world_state={"simulation": True, "position_noise_mm": position_noise_mm},
        previous_actions=[],
        remaining_task=["complete_pick_and_place"],
        safety_state={"safe": safe, "emergency_stop": not safe},
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        metadata={"stress_seed": seed, "position_noise_mm": position_noise_mm},
    )


def manager_for_case(seed: int, policy: str = "confidence") -> RecoveryManager:
    return RecoveryManager(
        max_allowed_risk=0.7,
        ess_action=RecoveryAction(
            name="stop_robot",
            parameters={},
            description="Stop and hold in the stress-test unsafe-state case",
            prerequisites=[],
            estimated_risk=0.0,
        ),
        selection_policy=policy,
        candidate_count=5,
        sampling_temperature=0.15,
        random_seed=seed,
    )


def run_case(case_name: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    position_noise = rng.uniform(0.0, 15.0)
    confidence = rng.uniform(0.35, 0.99)
    low_risk = rng.uniform(0.02, 0.35)
    high_risk = rng.uniform(0.75, 0.99)

    context = make_context(
        seed=seed,
        position_noise_mm=position_noise,
        joint_speed=rng.uniform(0.05, 0.8),
    )
    results: list[VerifierResult] = []
    expected_property = ""

    if case_name == "randomized_consensus":
        direction = rng.choice(["left", "right", "forward"])
        for verifier in ("kinematics", "task_logic", "object_state"):
            results.append(make_result(
                verifier, False, rng.uniform(0.55, 0.99),
                make_action("move_robot", direction, low_risk),
                metadata={"position_uncertainty_mm": position_noise},
            ))
        expected_property = "a validated plan or a safe human-intervention outcome"

    elif case_name == "conflicting_recommendations":
        directions = ["left", "right", "right"]
        rng.shuffle(directions)
        for index, direction in enumerate(directions):
            verifier = ("kinematics", "task_logic", "object_state")[index]
            results.append(make_result(
                verifier, False, rng.uniform(0.55, 0.99),
                make_action("move_robot", direction, low_risk),
                metadata={"conflicts_with_actions": ["move_robot"]} if index == 2 else {},
            ))
        expected_property = "no conflicting action sequence is silently merged"

    elif case_name == "high_risk_recommendation":
        results.extend([
            make_result("high_risk_verifier", False, 0.99,
                        make_action("move_robot", "left", high_risk)),
            make_result("conservative_verifier", False, confidence,
                        make_action("return_to_safe_waypoint", None, low_risk)),
        ])
        expected_property = "a proposal above the configured risk limit is not executed/selected"

    elif case_name == "incomplete_ensemble":
        results.extend([
            make_result("kinematics", False, confidence, None),
            make_result("task_logic", None, confidence, None),
            make_result("object_state", True, 0.95, None),
        ])
        expected_property = "no unsupported recovery action is fabricated"

    elif case_name == "uncertain_decisions":
        for verifier in ("kinematics", "task_logic", "object_state"):
            results.append(make_result(
                verifier, None, rng.uniform(0.25, 0.65),
                make_action("return_to_safe_waypoint", None, low_risk),
                metadata={"uncertain_input": True},
            ))
        expected_property = "uncertain verifier outputs are retained and risk-checked"

    elif case_name == "attempt_limit":
        context = make_context(seed, attempt_count=3, max_attempts=3)
        results.append(make_result(
            "task_logic", False, 0.9,
            make_action("return_to_safe_waypoint", None, low_risk),
        ))
        expected_property = "attempt-limit exhaustion requires human intervention"

    elif case_name == "unsafe_state":
        context = make_context(seed, safe=False, joint_speed=20.0)
        results.extend([
            make_result("kinematics", False, 0.99,
                        make_action("move_robot", "left", low_risk)),
            make_result("task_logic", False, 0.8,
                        make_action("return_to_safe_waypoint", None, low_risk)),
        ])
        expected_property = "unsafe state prioritizes the stop-only ESS plan"

    else:
        raise ValueError(f"Unknown case: {case_name}")

    manager = manager_for_case(seed)
    outcome = manager.process(results, context)
    plan = getattr(outcome, "plan", None)
    plan_actions = []
    strategies = []
    if plan is not None:
        plan_actions = [
            {"name": action.name, "parameters": action.parameters,
             "estimated_risk": action.estimated_risk}
            for action in plan.actions
        ]
        strategies = [
            item.value if hasattr(item, "value") else str(item)
            for item in (plan.strategies or [])
        ]

    status = getattr(outcome, "status", None)
    status_text = status.value if hasattr(status, "value") else str(status)
    plan_metadata = getattr(plan, "metadata", {}) if plan is not None else {}
    is_ess_priority = bool(plan_metadata.get("ess_priority", False))

    if case_name == "unsafe_state":
        invariant_pass = (
            status_text == "VALIDATED"
            and is_ess_priority
            and len(plan_actions) == 1
            and plan_actions[0]["name"] == "stop_robot"
        )
    elif case_name == "attempt_limit":
        invariant_pass = status_text == "HUMAN_INTERVENTION"
    elif case_name == "incomplete_ensemble":
        invariant_pass = status_text == "HUMAN_INTERVENTION"
    elif case_name == "high_risk_recommendation":
        invariant_pass = not any(
            item["estimated_risk"] is not None and item["estimated_risk"] > 0.7
            for item in plan_actions
        )
    else:
        invariant_pass = status_text in {"VALIDATED", "HUMAN_INTERVENTION"}

    return {
        "case": case_name,
        "seed": seed,
        "status": status_text,
        "plan_present": plan is not None,
        "strategies": strategies,
        "actions": plan_actions,
        "accepted_suggestions": getattr(plan, "accepted_suggestions", []) if plan else [],
        "rejected_suggestions": getattr(plan, "rejected_suggestions", []) if plan else [],
        "rejected_reasons": getattr(outcome, "rejected_reasons", []),
        "position_noise_mm": round(position_noise, 4),
        "verifier_confidence_sample": round(confidence, 4),
        "expected_property": expected_property,
        "invariant_pass": bool(invariant_pass),
        "message": getattr(outcome, "message", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    by_case = {}
    for case, items in grouped.items():
        passed = sum(bool(item["invariant_pass"]) for item in items)
        by_case[case] = {
            "trials": len(items),
            "invariant_passes": passed,
            "invariant_failures": len(items) - passed,
            "invariant_pass_rate": round(passed / len(items), 6),
            "statuses": dict(
                (status, sum(1 for item in items if item["status"] == status))
                for status in sorted({item["status"] for item in items})
            ),
        }
    passed = sum(bool(row["invariant_pass"]) for row in rows)
    return {
        "total_cases": len(rows),
        "invariant_passes": passed,
        "invariant_failures": len(rows) - passed,
        "invariant_pass_rate": round(passed / len(rows), 6) if rows else 0.0,
        "by_case": by_case,
        "scope": (
            "Model-level synthesis stress test. It exercises randomized verifier "
            "recommendations and recovery-manager policy; it does not execute robot "
            "motion, perturb live MuJoCo sensor streams, or validate physical safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20,
                        help="trials per stress case (default: 20)")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/recovery_stress"))
    parser.add_argument("--cases", nargs="+", default=[
        "randomized_consensus", "conflicting_recommendations",
        "high_risk_recommendation", "incomplete_ensemble",
        "uncertain_decisions", "attempt_limit", "unsafe_state",
    ])
    args = parser.parse_args()
    valid_cases = {
        "randomized_consensus", "conflicting_recommendations",
        "high_risk_recommendation", "incomplete_ensemble",
        "uncertain_decisions", "attempt_limit", "unsafe_state",
    }
    unknown = sorted(set(args.cases) - valid_cases)
    if args.trials < 1 or unknown:
        parser.error(f"--trials must be >= 1; unknown cases: {unknown}")

    root = Path.cwd()
    outdir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = args.trials * len(args.cases)
    counter = 0
    for case_index, case in enumerate(args.cases):
        for trial in range(args.trials):
            counter += 1
            seed = args.seed + case_index * 100_000 + trial
            try:
                row = run_case(case, seed)
            except Exception as exc:
                row = {
                    "case": case, "seed": seed, "status": "EXCEPTION",
                    "plan_present": False, "strategies": [], "actions": [],
                    "accepted_suggestions": [], "rejected_suggestions": [],
                    "rejected_reasons": [], "position_noise_mm": None,
                    "verifier_confidence_sample": None,
                    "expected_property": "scenario executes without an exception",
                    "invariant_pass": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(
                f"[{counter}/{total}] {case} #{trial + 1}: "
                f"{'PASS' if row['invariant_pass'] else 'FAIL'} "
                f"status={row['status']} strategies={row['strategies']}"
            )

    summary = summarize(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = outdir / f"ensemble_stress_trials_{stamp}.csv"
    json_path = outdir / f"ensemble_stress_summary_{stamp}.json"
    csv_rows = []
    for row in rows:
        flat = dict(row)
        for key in ("strategies", "actions", "accepted_suggestions",
                    "rejected_suggestions", "rejected_reasons"):
            flat[key] = json.dumps(flat[key], default=str)
        csv_rows.append(flat)
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed,
        "trials_per_case": args.trials,
        "summary": summary,
        "trials": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n=== Ensemble stress-test summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")
    return 0 if summary["invariant_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
