#!/usr/bin/env python3
"""Scenario-based planning evaluation for RobAuditor recovery synthesis.

E1_single: designated kinematics verifier only.
E2_naive: all suggestions are risk-screened, then merged without conflict
          resolution.
E3_ensemble: RecoveryManager normalization, risk screening, conflict
             resolution, ranking and validation.

All verifier outputs are synthetic fixtures. This is a planning-level
experiment: no robot actions are executed and no sensor-backed verifier
performance is claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from recovery.models import (
    PlanStatus, RecoveryAction, RecoveryContext, RecoveryPlan,
    RecoveryStrategy, RecoverySuggestion, VerifierResult,
)
from recovery.recovery_manager import RecoveryManager
from recovery.risk_assessor import RiskDecision, assess_suggestion
from recovery.suggestion_normalizer import normalize_suggestions
from recovery.plan_validator import validate_plan


MAX_RISK = 0.70
BASE_CONSTRAINTS = {"max_allowed_risk": MAX_RISK, "forbidden_actions": []}


def make_action(name: str, parameters: dict | None = None,
                risk: float = 0.2) -> RecoveryAction:
    return RecoveryAction(
        name=name,
        parameters=parameters or {},
        description=f"Synthetic experiment action: {name}",
        prerequisites=[],
        estimated_risk=risk,
    )


def make_verifier(verifier_id: str, strategy: RecoveryStrategy,
                  actions: List[RecoveryAction], confidence: float,
                  explanation: str, metadata: dict | None = None,
                  decision: bool = False) -> VerifierResult:
    suggestion = None if decision else RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=strategy,
        actions=actions,
        confidence=confidence,
        explanation=explanation,
        modalities=["synthetic_simulation_context"],
        failure_type="injected_test_failure",
        metadata=metadata or {},
    )
    return VerifierResult(
        verifier_id=verifier_id,
        decision=decision,
        confidence=confidence,
        explanation=explanation,
        suggestion=suggestion,
        evidence={"source": "synthetic_experiment_fixture"},
    )


def context_for(scenario: dict, repeat: int) -> RecoveryContext:
    unsafe = bool(scenario.get("unsafe_start", False))
    state = {"safe": not unsafe, "status": "unsafe" if unsafe else "nominal"}
    return RecoveryContext(
        failed_action_id=f"{scenario['id']}_rep{repeat:03d}",
        failed_action="canister_pick_and_place",
        participants={"Actor": "UR10", "Patient": "canister"},
        object_state="canister_at_known_pose",
        robot_state=dict(state),
        world_state={"simulation": "MuJoCo", "scenario_id": scenario["id"]},
        previous_actions=["perceive_canister", "approach_canister"],
        remaining_task=["recover", "complete_pick_and_place"],
        safety_state=dict(state),
        attempt_count=0,
        max_attempts=2,
        metadata={"experiment_scenario": scenario["id"], "repeat": repeat},
    )


def v_kin(strategy, name, params, conf=0.9, risk=0.2, metadata=None):
    return make_verifier(
        "kinematics_verifier", strategy, [make_action(name, params, risk)],
        conf, "Synthetic kinematics-domain recommendation.", metadata,
    )


def v_task(strategy, name, params, conf=0.8, risk=0.2, metadata=None):
    return make_verifier(
        "task_logic_verifier", strategy, [make_action(name, params, risk)],
        conf, "Synthetic task-logic-domain recommendation.", metadata,
    )


def v_safety(strategy, name, params, conf=0.9, risk=0.1, metadata=None):
    return make_verifier(
        "safety_verifier", strategy, [make_action(name, params, risk)],
        conf, "Synthetic safety-domain recommendation.", metadata,
    )


def scenarios() -> List[dict]:
    return [
        {
            "id": "S1_waypoint_parameter_conflict",
            "description": "Same action, contradictory waypoint parameter.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "pre_grasp"}, 0.93),
                v_task(RecoveryStrategy.GOB, "return_to_waypoint",
                       {"waypoint": "home"}, 0.76),
                v_safety(RecoveryStrategy.ESS, "hold_robot",
                         {"mode": "controlled_hold"}),
            ],
            "expected_selected": {"kinematics_verifier", "safety_verifier"},
            "expected_rejected": {"task_logic_verifier"},
            "expected_actions": {"return_to_waypoint", "hold_robot"},
            "forbidden": [],
        },
        {
            "id": "S2_explicit_verifier_conflict",
            "description": "Explicit conflict declared in suggestion metadata.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "last_safe"}, 0.91,
                      metadata={"conflicts_with": ["task_logic_verifier"]}),
                v_task(RecoveryStrategy.INA, "replan_approach",
                       {"route": "raised"}, 0.82),
                v_safety(RecoveryStrategy.ESS, "hold_robot",
                         {"mode": "controlled_hold"}, 0.90),
            ],
            "expected_selected": {"kinematics_verifier", "safety_verifier"},
            "expected_rejected": {"task_logic_verifier"},
            "expected_actions": {"return_to_waypoint", "hold_robot"},
            "forbidden": [],
        },
        {
            "id": "S3_compatible_suggestions",
            "description": "Three complementary recommendations.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "transfer_start"}, 0.88),
                v_task(RecoveryStrategy.INA, "replan_insertion",
                       {"approach": "centered"}, 0.84),
                v_safety(RecoveryStrategy.ESS, "verify_clearance",
                         {"minimum_clearance_m": 0.05}, 0.92),
            ],
            "expected_selected": {"kinematics_verifier", "task_logic_verifier",
                                  "safety_verifier"},
            "expected_rejected": set(),
            "expected_actions": {"return_to_waypoint", "replan_insertion",
                                 "verify_clearance"},
            "forbidden": [],
        },
        {
            "id": "S4_high_risk_recommendation",
            "description": "One recommendation exceeds the risk threshold.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "last_safe"}, 0.88, 0.2),
                v_task(RecoveryStrategy.INA, "direct_transfer",
                       {"route": "unverified_direct"}, 0.99, 0.94),
                v_safety(RecoveryStrategy.ESS, "hold_robot",
                         {"mode": "controlled_hold"}, 0.90, 0.1),
            ],
            "expected_selected": {"kinematics_verifier", "safety_verifier"},
            "expected_rejected": {"task_logic_verifier"},
            "expected_actions": {"return_to_waypoint", "hold_robot"},
            "forbidden": [],
        },
        {
            "id": "S5_forbidden_action",
            "description": "A verifier proposes an explicitly forbidden action.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "last_safe"}, 0.87),
                v_task(RecoveryStrategy.INA, "disable_collision_monitor",
                       {"enabled": True}, 0.98, 0.1),
                v_safety(RecoveryStrategy.ESS, "hold_robot",
                         {"mode": "controlled_hold"}, 0.91),
            ],
            "expected_selected": {"kinematics_verifier", "safety_verifier"},
            "expected_rejected": {"task_logic_verifier"},
            "expected_actions": {"return_to_waypoint", "hold_robot"},
            "forbidden": ["disable_collision_monitor"],
        },
        {
            "id": "S6_missing_verifier_suggestion",
            "description": "Task-logic verifier reports success/no suggestion.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "pre_grasp"}, 0.90),
                make_verifier("task_logic_verifier", RecoveryStrategy.INA, [],
                              0.80, "No recovery suggestion.", decision=True),
                v_safety(RecoveryStrategy.ESS, "hold_robot",
                         {"mode": "controlled_hold"}, 0.89),
            ],
            "expected_selected": {"kinematics_verifier", "safety_verifier"},
            "expected_rejected": set(),
            "expected_actions": {"return_to_waypoint", "hold_robot"},
            "forbidden": [],
        },
        {
            "id": "S7_low_confidence_conflict",
            "description": "Conflicting waypoint recommendations with unequal confidence.",
            "verifiers": [
                v_kin(RecoveryStrategy.GOB, "return_to_waypoint",
                      {"waypoint": "last_safe"}, 0.97),
                v_task(RecoveryStrategy.GOB, "return_to_waypoint",
                       {"waypoint": "home"}, 0.31),
                v_safety(RecoveryStrategy.ESS, "hold_robot",
                         {"mode": "controlled_hold"}, 0.88),
            ],
            "expected_selected": {"kinematics_verifier", "safety_verifier"},
            "expected_rejected": {"task_logic_verifier"},
            "expected_actions": {"return_to_waypoint", "hold_robot"},
            "forbidden": [],
        },
        {
            "id": "S8_no_recovery_suggestions",
            "description": "No verifier supplies a recovery suggestion.",
            "verifiers": [
                make_verifier("kinematics_verifier", RecoveryStrategy.GOB, [],
                              0.8, "No suggestion.", decision=True),
                make_verifier("task_logic_verifier", RecoveryStrategy.INA, [],
                              0.8, "No suggestion.", decision=True),
                make_verifier("safety_verifier", RecoveryStrategy.ESS, [],
                              0.8, "No suggestion.", decision=True),
            ],
            "expected_selected": set(),
            "expected_rejected": set(),
            "expected_actions": set(),
            "forbidden": [],
        },
    ]


def status_text(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def evaluate_row(condition: str, scenario: dict, repeat: int,
                 status: str, plan, elapsed_ms: float, reason: str = "") -> dict:
    accepted = set(plan.accepted_suggestions) if plan else set()
    rejected_map = plan.rejected_suggestions if plan else {}
    rejected = set(rejected_map)
    actions = {a.name for a in plan.actions} if plan else set()
    expected_selected = scenario["expected_selected"]
    expected_rejected = scenario["expected_rejected"]
    expected_actions = scenario["expected_actions"]

    # Strategy correctness is measured against fixture annotations, not inferred
    # from the algorithm's own output.
    selected_precision = (
        len(accepted & expected_selected) / len(accepted) if accepted else
        (1.0 if not expected_selected else 0.0)
    )
    selected_recall = (
        len(accepted & expected_selected) / len(expected_selected)
        if expected_selected else (1.0 if not accepted else 0.0)
    )
    action_precision = (
        len(actions & expected_actions) / len(actions) if actions else
        (1.0 if not expected_actions else 0.0)
    )
    action_recall = (
        len(actions & expected_actions) / len(expected_actions)
        if expected_actions else (1.0 if not actions else 0.0)
    )
    return {
        "scenario": scenario["id"],
        "condition": condition,
        "repeat": repeat,
        "status": status,
        "validated": status == PlanStatus.VALIDATED.value,
        "expected_verifiers": "|".join(sorted(expected_selected)),
        "accepted_verifiers": "|".join(sorted(accepted)),
        "rejected_verifiers": "|".join(sorted(rejected)),
        "correct_verifier_selection": accepted == expected_selected,
        "verifier_selection_precision": round(selected_precision, 4),
        "verifier_selection_recall": round(selected_recall, 4),
        "expected_actions": "|".join(sorted(expected_actions)),
        "generated_actions": "|".join(sorted(actions)),
        "action_set_exact_match": actions == expected_actions,
        "action_precision": round(action_precision, 4),
        "action_recall": round(action_recall, 4),
        "action_count": len(plan.actions) if plan else 0,
        "rejected_verifier_count": len(rejected),
        "confidence_score": (plan.metadata or {}).get("confidence_score", "") if plan else "",
        "risk_score": (plan.metadata or {}).get("risk_score", "") if plan else "",
        "planning_time_ms": round(elapsed_ms, 5),
        "message": reason,
        "scenario_description": scenario["description"],
    }


def run_e1(results, context, manager):
    chosen = next((r for r in results if r.verifier_id == "kinematics_verifier"), results[0])
    start = time.perf_counter()
    out = manager.process([chosen], context)
    return out, (time.perf_counter() - start) * 1000.0


def run_e2(results, context, constraints):
    start = time.perf_counter()
    suggestions = normalize_suggestions(results)
    assessments = {
        s.verifier_id: assess_suggestion(
            s, context=context, max_allowed_risk=MAX_RISK,
            safety_constraints=constraints,
        )
        for s in suggestions
    }
    accepted = [s for s in suggestions
                if assessments[s.verifier_id].decision == RiskDecision.ACCEPT]
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
        for item in suggestion.actions:
            signature = (item.name.lower(), json.dumps(item.parameters, sort_keys=True))
            if signature not in seen:
                merged.append(item)
                seen.add(signature)

    plan = RecoveryPlan(
        recovery_id=f"naive_{context.failed_action_id}",
        failed_action_id=context.failed_action_id,
        actions=merged,
        strategies=list(dict.fromkeys(s.strategy for s in accepted)),
        status=PlanStatus.PROPOSED if merged else PlanStatus.HUMAN_INTERVENTION,
        explanation="Risk-screened naive merge; no conflict resolution.",
        accepted_suggestions=[s.verifier_id for s in accepted],
        rejected_suggestions=rejected,
        max_attempts=context.max_attempts,
        metadata={
            "confidence_score": (
                statistics.mean([s.confidence or 0.0 for s in accepted])
                if accepted else 0.0
            ),
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
    return plan, (time.perf_counter() - start) * 1000.0, reason


def run_e3(results, context, manager):
    start = time.perf_counter()
    out = manager.process(results, context)
    return out, (time.perf_counter() - start) * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/recovery_experiments/ensemble_planning_v2")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in scenarios():
        constraints = {
            "max_allowed_risk": MAX_RISK,
            "forbidden_actions": list(scenario["forbidden"]),
        }
        manager = RecoveryManager(
            max_allowed_risk=MAX_RISK,
            safety_constraints=constraints,
        )
        for repeat in range(1, args.repeats + 1):
            for condition in ("E1_single", "E2_naive", "E3_ensemble"):
                # Fresh context per method prevents one method from mutating
                # the next method's context.
                context = context_for(scenario, repeat)
                if condition == "E1_single":
                    out, elapsed = run_e1(scenario["verifiers"], context, manager)
                    plan, status, reason = out.plan, status_text(out.status), out.message
                elif condition == "E2_naive":
                    plan, elapsed, reason = run_e2(
                        scenario["verifiers"], context, constraints
                    )
                    status = status_text(plan.status)
                else:
                    out, elapsed = run_e3(scenario["verifiers"], context, manager)
                    plan, status, reason = out.plan, status_text(out.status), out.message
                rows.append(evaluate_row(
                    condition, scenario, repeat, status, plan, elapsed, reason
                ))

    csv_path = output_dir / "planning_comparison_v2.csv"
    json_path = output_dir / "planning_comparison_v2.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== RobAuditor scenario-based planning evaluation ===")
    print(f"Scenarios: {len(scenarios())}; repeats/scenario: {args.repeats}; "
          f"trials/condition: {len(scenarios()) * args.repeats}")
    print("Synthetic verifier fixtures; planning only; no robot execution.\n")
    print(f"{'Condition':14} {'Validated':12} {'Exact selection':16} "
          f"{'Exact actions':14} {'Mean actions':12} {'Mean ms':10}")
    for condition in ("E1_single", "E2_naive", "E3_ensemble"):
        subset = [r for r in rows if r["condition"] == condition]
        n = len(subset)
        validated = sum(r["validated"] for r in subset)
        selection = sum(r["correct_verifier_selection"] for r in subset)
        exact_actions = sum(r["action_set_exact_match"] for r in subset)
        print(
            f"{condition:14} {validated/n:>10.1%} "
            f"{selection/n:>14.1%} {exact_actions/n:>12.1%} "
            f"{statistics.mean(r['action_count'] for r in subset):>11.2f} "
            f"{statistics.mean(r['planning_time_ms'] for r in subset):>9.4f}"
        )

    print("\nScenario-level E3 results:")
    for scenario in scenarios():
        subset = [r for r in rows if r["scenario"] == scenario["id"]
                  and r["condition"] == "E3_ensemble"]
        print(
            f"  {scenario['id']}: validated="
            f"{sum(r['validated'] for r in subset)}/{len(subset)}, "
            f"exact_selection="
            f"{sum(r['correct_verifier_selection'] for r in subset)}/{len(subset)}, "
            f"exact_actions="
            f"{sum(r['action_set_exact_match'] for r in subset)}/{len(subset)}"
        )

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
