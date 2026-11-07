#!/usr/bin/env python3
"""Randomized stress test for RobAuditor recovery planning.

This is a software-level experiment using synthetic verifier fixtures.
It does not execute robot actions or represent sensor-backed verifier data.

E1: one designated kinematics verifier through RecoveryManager.
E2: risk screening followed by naive action concatenation and validation.
E3: the existing RecoveryManager ensemble pipeline.

The generator records a scenario-level expected safe subset for auditing.
It does not claim that this synthetic oracle is ground truth for real robots.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

from recovery.models import (
    PlanStatus, RecoveryAction, RecoveryContext, RecoveryPlan,
    RecoveryStrategy, RecoverySuggestion, VerifierResult,
)
from recovery.recovery_manager import RecoveryManager
from recovery.risk_assessor import RiskDecision, assess_suggestion
from recovery.suggestion_normalizer import normalize_suggestions
from recovery.plan_validator import validate_plan


MAX_RISK = 0.70
VERIFIER_IDS = (
    "kinematics_verifier",
    "task_logic_verifier",
    "safety_verifier",
)
ACTIONS = (
    "return_to_waypoint",
    "replan_approach",
    "hold_robot",
    "verify_clearance",
    "replan_insertion",
    "disable_collision_monitor",
    "direct_transfer",
)
STRATEGIES = (
    RecoveryStrategy.GOB,
    RecoveryStrategy.GBB,
    RecoveryStrategy.INA,
    RecoveryStrategy.ESS,
    RecoveryStrategy.HIC,
    RecoveryStrategy.INO,
)


def make_context(case_id: str, unsafe: bool = False,
                 attempt_limit: bool = False) -> RecoveryContext:
    state = {
        "safe": not unsafe,
        "status": "unsafe" if unsafe else "nominal",
        "collision_detected": unsafe,
    }
    return RecoveryContext(
        failed_action_id=case_id,
        failed_action="canister_pick_and_place",
        participants={"Actor": "UR10", "Patient": "canister"},
        object_state="synthetic_object_state",
        robot_state=dict(state),
        world_state={"simulation": "MuJoCo", "case_id": case_id},
        previous_actions=["perceive_canister", "approach_canister"],
        remaining_task=["recover", "complete_task"],
        safety_state=dict(state),
        attempt_count=2 if attempt_limit else 0,
        max_attempts=2,
        metadata={"stress_case": case_id},
    )


def make_result(verifier_id: str, strategy: RecoveryStrategy,
                action_name: str, parameters: dict, confidence: float,
                risk: float | None, *, missing_suggestion: bool = False,
                decision: bool = False,
                conflicts_with: list[str] | None = None) -> VerifierResult:
    suggestion = None
    if not missing_suggestion and not decision:
        action = RecoveryAction(
            name=action_name,
            parameters=parameters,
            description=f"Randomized synthetic action {action_name}",
            prerequisites=[],
            estimated_risk=risk,
        )
        suggestion = RecoverySuggestion(
            verifier_id=verifier_id,
            strategy=strategy,
            actions=[action],
            confidence=confidence,
            explanation="Randomized synthetic stress-test recommendation.",
            modalities=["synthetic_stress_fixture"],
            failure_type="randomized_test_failure",
            metadata={"conflicts_with": conflicts_with or []},
        )
    return VerifierResult(
        verifier_id=verifier_id,
        decision=decision,
        confidence=confidence,
        explanation="Synthetic randomized verifier result.",
        suggestion=suggestion,
        evidence={"source": "randomized_synthetic_fixture"},
    )


def generate_case(rng: random.Random, index: int) -> dict:
    """Generate three verifier outputs with controlled edge cases."""
    case_id = f"V3_{index:05d}"
    mode = rng.choice([
        "ordinary", "multi_conflict", "unsafe_suggestion",
        "missing_suggestion", "unsafe_context", "attempt_limit",
        "incomplete_parameters", "forbidden_action",
    ])
    results = []
    expected_safe = set()
    forbidden = []

    for verifier_id in VERIFIER_IDS:
        confidence = round(rng.uniform(0.05, 1.0), 3)
        risk = round(rng.uniform(0.02, 0.98), 3)
        strategy = rng.choice(STRATEGIES)
        action_name = rng.choice(ACTIONS[:5])
        params = {"target": rng.choice(["last_safe", "pre_grasp", "home"])}
        conflicts = []

        if mode == "multi_conflict":
            action_name = "return_to_waypoint"
            params = {"waypoint": rng.choice(["home", "pre_grasp", "last_safe"])}
        elif mode == "incomplete_parameters":
            params = {}
        elif mode == "forbidden_action" and verifier_id == "task_logic_verifier":
            action_name = "disable_collision_monitor"
            forbidden = ["disable_collision_monitor"]
            risk = round(rng.uniform(0.02, 0.30), 3)
        elif mode == "unsafe_suggestion" and verifier_id == "task_logic_verifier":
            action_name = "direct_transfer"
            risk = round(rng.uniform(0.71, 0.99), 3)
        elif mode == "multi_conflict" and verifier_id == "safety_verifier":
            conflicts = ["kinematics_verifier"]

        missing = mode == "missing_suggestion" and verifier_id == "task_logic_verifier"
        decision = missing and rng.random() < 0.5
        if missing and not decision:
            # Explicitly no suggestion, but the verifier reports failure.
            results.append(make_result(
                verifier_id, strategy, action_name, params, confidence, risk,
                missing_suggestion=True,
            ))
            continue

        results.append(make_result(
            verifier_id, strategy, action_name, params, confidence, risk,
            decision=decision, conflicts_with=conflicts,
        ))

        if not decision and risk is not None and risk <= MAX_RISK \
                and action_name not in forbidden and mode not in {"unsafe_context", "attempt_limit"}:
            expected_safe.add(verifier_id)

    unsafe_context = mode == "unsafe_context"
    attempt_limit = mode == "attempt_limit"
    return {
        "id": case_id,
        "mode": mode,
        "results": results,
        "unsafe_context": unsafe_context,
        "attempt_limit": attempt_limit,
        "forbidden": forbidden,
        "expected_safe": expected_safe,
    }


def status_text(status):
    return status.value if hasattr(status, "value") else str(status)


def plan_metrics(condition: str, case: dict, plan, status: str,
                 elapsed_ms: float, reason: str = "") -> dict:
    accepted = set(plan.accepted_suggestions) if plan else set()
    rejected = set(plan.rejected_suggestions) if plan else set()
    actions = set(a.name for a in plan.actions) if plan else set()
    expected = case["expected_safe"]

    precision = len(accepted & expected) / len(accepted) if accepted else (
        1.0 if not expected else 0.0
    )
    recall = len(accepted & expected) / len(expected) if expected else (
        1.0 if not accepted else 0.0
    )
    intervention = status == PlanStatus.HUMAN_INTERVENTION.value
    return {
        "case_id": case["id"],
        "mode": case["mode"],
        "condition": condition,
        "status": status,
        "validated": status == PlanStatus.VALIDATED.value,
        "human_intervention": intervention,
        "expected_safe_verifiers": "|".join(sorted(expected)),
        "accepted_verifiers": "|".join(sorted(accepted)),
        "rejected_verifiers": "|".join(sorted(rejected)),
        "verifier_selection_precision": round(precision, 4),
        "verifier_selection_recall": round(recall, 4),
        "correct_safe_subset": accepted == expected,
        "action_count": len(plan.actions) if plan else 0,
        "actions": "|".join(sorted(actions)),
        "risk_score": (plan.metadata or {}).get("risk_score", "") if plan else "",
        "planning_time_ms": round(elapsed_ms, 5),
        "message": reason,
    }


def run_e1(case, context, manager):
    selected = next(
        (r for r in case["results"] if r.verifier_id == "kinematics_verifier"),
        case["results"][0],
    )
    start = time.perf_counter()
    result = manager.process([selected], context)
    return result.plan, status_text(result.status), (
        time.perf_counter() - start
    ) * 1000.0, result.message


def run_e2(case, context, constraints):
    start = time.perf_counter()
    suggestions = normalize_suggestions(case["results"])
    assessments = {
        s.verifier_id: assess_suggestion(
            s, context=context, max_allowed_risk=MAX_RISK,
            safety_constraints=constraints,
        )
        for s in suggestions
    }
    accepted = [
        s for s in suggestions
        if assessments[s.verifier_id].decision == RiskDecision.ACCEPT
    ]
    rejected = {
        s.verifier_id: (
            f"{assessments[s.verifier_id].decision.value}: "
            + "; ".join(assessments[s.verifier_id].reasons)
        )
        for s in suggestions
        if assessments[s.verifier_id].decision != RiskDecision.ACCEPT
    }
    merged, seen = [], set()
    for suggestion in accepted:
        for action in suggestion.actions:
            signature = (action.name.lower(), json.dumps(action.parameters, sort_keys=True))
            if signature not in seen:
                merged.append(action)
                seen.add(signature)
    plan = RecoveryPlan(
        recovery_id=f"naive_{case['id']}",
        failed_action_id=context.failed_action_id,
        actions=merged,
        strategies=list(dict.fromkeys(s.strategy for s in accepted)),
        status=PlanStatus.PROPOSED if merged else PlanStatus.HUMAN_INTERVENTION,
        explanation="Risk-screened naive merge without conflict resolution.",
        accepted_suggestions=[s.verifier_id for s in accepted],
        rejected_suggestions=rejected,
        max_attempts=context.max_attempts,
        metadata={
            "confidence_score": statistics.mean(
                [s.confidence or 0.0 for s in accepted]
            ) if accepted else 0.0,
            "risk_score": max(
                [assessments[s.verifier_id].risk_score for s in accepted],
                default=1.0,
            ),
        },
    )
    reason = ""
    if merged:
        validation = validate_plan(plan, context, safety_constraints=constraints)
        if validation.valid:
            plan.status = PlanStatus.VALIDATED
        else:
            plan.status = PlanStatus.HUMAN_INTERVENTION
            reason = "; ".join(validation.reasons)
    else:
        reason = "No risk-accepted suggestions."
    return plan, status_text(plan.status), (
        time.perf_counter() - start
    ) * 1000.0, reason


def run_e3(case, context, manager):
    start = time.perf_counter()
    result = manager.process(case["results"], context)
    return result.plan, status_text(result.status), (
        time.perf_counter() - start
    ) * 1000.0, result.message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument(
        "--output-dir",
        default="results/recovery_experiments/ensemble_stress_v3",
    )
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be at least 1")

    rng = random.Random(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for index in range(1, args.cases + 1):
        case = generate_case(rng, index)
        constraints = {
            "max_allowed_risk": MAX_RISK,
            "forbidden_actions": case["forbidden"],
        }
        manager = RecoveryManager(
            max_allowed_risk=MAX_RISK,
            safety_constraints=constraints,
        )
        for condition in ("E1_single", "E2_naive", "E3_ensemble"):
            context = make_context(
                case["id"], unsafe=case["unsafe_context"],
                attempt_limit=case["attempt_limit"],
            )
            if condition == "E1_single":
                plan, status, elapsed, reason = run_e1(case, context, manager)
            elif condition == "E2_naive":
                plan, status, elapsed, reason = run_e2(case, context, constraints)
            else:
                plan, status, elapsed, reason = run_e3(case, context, manager)
            rows.append(plan_metrics(
                condition, case, plan, status, elapsed, reason
            ))

    csv_path = output / "stress_results_v3.csv"
    json_path = output / "stress_results_v3.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== RobAuditor randomized stress test V3 ===")
    print(f"Seed: {args.seed}; cases: {args.cases}; "
          f"trials/condition: {args.cases}; total trials: {len(rows)}")
    print("Synthetic software stress test only; no robot execution.\n")
    print(f"{'Condition':14} {'Validated':10} {'Human handoff':14} "
          f"{'Safe subset':12} {'Mean actions':13} {'Mean ms':10}")
    for condition in ("E1_single", "E2_naive", "E3_ensemble"):
        subset = [r for r in rows if r["condition"] == condition]
        n = len(subset)
        print(
            f"{condition:14} "
            f"{sum(r['validated'] for r in subset)/n:>8.1%} "
            f"{sum(r['human_intervention'] for r in subset)/n:>12.1%} "
            f"{sum(r['correct_safe_subset'] for r in subset)/n:>10.1%} "
            f"{statistics.mean(r['action_count'] for r in subset):>12.2f} "
            f"{statistics.mean(r['planning_time_ms'] for r in subset):>9.4f}"
        )

    print("\nBy generated case type (E3):")
    for mode in sorted({r["mode"] for r in rows}):
        subset = [r for r in rows if r["condition"] == "E3_ensemble"
                  and r["mode"] == mode]
        if subset:
            print(
                f"  {mode:22} n={len(subset):4d} "
                f"validated={sum(r['validated'] for r in subset)/len(subset):.1%} "
                f"handoff={sum(r['human_intervention'] for r in subset)/len(subset):.1%}"
            )

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
