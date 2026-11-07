"""Controlled comparison of single, naive-merge, and risk-aware recovery.

Run from the repository root:
PYTHONPATH=python/src/robauditor/src python \
  python/src/robauditor/src/recovery/ensemble_synthesis_experiment.py

This is a deterministic software experiment using synthetic verifier results.
It does not connect to physical sensors or command robot hardware.
"""

import csv
import json
from pathlib import Path

from recovery.conflict_resolver import ConflictType, classify_pair
from recovery.models import (
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryPlan,
    RecoveryStrategy,
    RecoverySuggestion,
    VerifierResult,
)
from recovery.recovery_manager import RecoveryManager


OUTPUT_DIR = Path("results/recovery_experiments/ensemble_synthesis")


def make_result(
    verifier_id,
    action_name=None,
    confidence=0.8,
    risk=0.1,
    parameters=None,
    decision=False,
):
    """Create one synthetic verifier result; None action means no suggestion."""

    suggestion = None
    if action_name is not None:
        suggestion = RecoverySuggestion(
            verifier_id=verifier_id,
            strategy=RecoveryStrategy.GOB,
            actions=[
                RecoveryAction(
                    name=action_name,
                    parameters=parameters or {},
                    description=f"Synthetic proposal from {verifier_id}",
                    estimated_risk=risk,
                )
            ],
            confidence=confidence,
            explanation=f"Synthetic recommendation from {verifier_id}",
            failure_type="controlled_ensemble_test",
            metadata={"synthetic": True},
        )

    return VerifierResult(
        verifier_id=verifier_id,
        decision=decision,
        confidence=confidence,
        explanation=f"Synthetic test result from {verifier_id}",
        suggestion=suggestion,
        evidence={"synthetic_fixture": True},
    )


def make_context(case_name):
    return RecoveryContext(
        failed_action_id=f"ensemble_test_{case_name}",
        failed_action="test_action",
        participants=["test_robot"],
        object_state={"available": True},
        robot_state={"safe": True, "status": "stopped"},
        world_state={"simulation": "synthetic_test"},
        previous_actions=[],
        remaining_task=[],
        safety_state={"safe": True},
        attempt_count=0,
        max_attempts=3,
        metadata={"experiment_case": case_name},
    )


def manager_for_tests():
    return RecoveryManager(
        max_allowed_risk=0.7,
        safety_constraints={},
        ess_action=RecoveryAction(
            name="stop_robot",
            description="Synthetic safe-state action",
            estimated_risk=0.1,
        ),
    )


def naive_merge(results, context):
    """Intentionally naive baseline: concatenate every proposed action.

    It does not apply risk assessment or conflict resolution. It is measured
    as a proposal baseline, not treated as an executable validated plan.
    """

    actions = []
    proposed = []
    for result in results:
        if result.decision or result.suggestion is None:
            continue
        proposed.append(result.suggestion)
        actions.extend(result.suggestion.actions)

    conflict_pairs = []
    for index, left in enumerate(proposed):
        for right in proposed[index + 1:]:
            report = classify_pair(left, right)
            if report.classification == ConflictType.CONFLICTING:
                conflict_pairs.append(
                    f"{left.verifier_id}<->{right.verifier_id}"
                )

    return {
        "action_count": len(actions),
        "action_names": [action.name for action in actions],
        "conflict_pair_count": len(conflict_pairs),
        "conflict_pairs": conflict_pairs,
        "risk_values": [action.estimated_risk for action in actions],
        "contains_over_threshold_risk": any(
            action.estimated_risk is None
            or action.estimated_risk > 0.7
            for action in actions
        ),
    }


def describe_outcome(outcome):
    plan = outcome.plan
    return {
        "status": outcome.status.value,
        "actions": [action.name for action in plan.actions] if plan else [],
        "action_parameters": [
            action.parameters for action in plan.actions
        ] if plan else [],
        "accepted_verifiers": list(plan.accepted_suggestions) if plan else [],
        "rejected_verifiers": dict(plan.rejected_suggestions) if plan else {},
        "risk_score": (
            plan.metadata.get("risk_score") if plan else None
        ),
        "confidence_score": (
            plan.metadata.get("confidence_score") if plan else None
        ),
        "message": outcome.message,
    }


def run_case(case_name, results):
    context = make_context(case_name)

    # E1: one verifier, chosen deterministically as the first proposer.
    proposers = [
        result for result in results
        if not result.decision and result.suggestion is not None
    ]
    single_results = proposers[:1] if proposers else results[:1]
    e1 = manager_for_tests().process(single_results, context)

    # E2: naive concatenation, without risk/conflict filtering.
    e2 = naive_merge(results, context)

    # E3: full existing manager pipeline over the ensemble.
    e3 = manager_for_tests().process(results, context)

    return {
        "case": case_name,
        "E1_single_verifier": describe_outcome(e1),
        "E2_naive_merge": e2,
        "E3_risk_aware_ensemble": describe_outcome(e3),
    }


def main():
    cases = {
        "consensus": [
            make_result("kinematics", "return_to_pre_grasp", 0.94, 0.2),
            make_result("task_logic", "return_to_pre_grasp", 0.90, 0.2),
            make_result("object_state", "return_to_pre_grasp", 0.87, 0.2),
        ],
        "conflicting_actions": [
            make_result(
                "kinematics", "move_robot", 0.94, 0.2,
                {"direction": "left"},
            ),
            make_result(
                "task_logic", "move_robot", 0.89, 0.2,
                {"direction": "right"},
            ),
            make_result(
                "object_state", "hold_position", 0.82, 0.2,
            ),
        ],
        "high_risk_vs_low_risk": [
            make_result("high_confidence", "aggressive_recovery", 0.99, 0.95),
            make_result("lower_confidence", "conservative_recovery", 0.72, 0.1),
        ],
        "incomplete_recommendations": [
            make_result("kinematics", decision=True),
            make_result("task_logic", decision=False),
            make_result("object_state", decision=True),
        ],
    }

    results = [
        run_case(case_name, verifier_results)
        for case_name, verifier_results in cases.items()
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "ensemble_synthesis_results.json"
    csv_path = OUTPUT_DIR / "ensemble_synthesis_summary.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for item in results:
        e1 = item["E1_single_verifier"]
        e2 = item["E2_naive_merge"]
        e3 = item["E3_risk_aware_ensemble"]
        rows.append({
            "case": item["case"],
            "E1_status": e1["status"],
            "E1_actions": ";".join(e1["actions"]),
            "E2_action_count": e2["action_count"],
            "E2_conflict_pair_count": e2["conflict_pair_count"],
            "E2_over_threshold_risk": e2["contains_over_threshold_risk"],
            "E3_status": e3["status"],
            "E3_actions": ";".join(e3["actions"]),
            "E3_accepted_verifiers": len(e3["accepted_verifiers"]),
            "E3_rejected_verifiers": len(e3["rejected_verifiers"]),
            "E3_risk_score": e3["risk_score"],
        })

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("[ensemble experiment] synthetic controlled comparison")
    print("[ensemble experiment] no physical sensors or robot hardware used")
    for row in rows:
        print(f"\n[case] {row['case']}")
        print(
            f"  E1 single: {row['E1_status']} "
            f"actions={row['E1_actions'] or 'none'}"
        )
        print(
            f"  E2 naive: actions={row['E2_action_count']} "
            f"conflicts={row['E2_conflict_pair_count']} "
            f"over_threshold_risk={row['E2_over_threshold_risk']}"
        )
        print(
            f"  E3 risk-aware: {row['E3_status']} "
            f"actions={row['E3_actions'] or 'none'} "
            f"accepted={row['E3_accepted_verifiers']} "
            f"rejected={row['E3_rejected_verifiers']}"
        )

    print("\n[ensemble experiment] JSON:", json_path.resolve())
    print("[ensemble experiment] CSV:", csv_path.resolve())


if __name__ == "__main__":
    main()
