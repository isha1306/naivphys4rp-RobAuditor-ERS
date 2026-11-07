"""Compare confidence-first and consensus-aware plan synthesis.

Run from the repository root:
PYTHONPATH=python/src/robauditor/src python \
  python/src/robauditor/src/recovery/consensus_policy_experiment.py

Synthetic, deterministic software experiment only; no robot hardware or
physical sensor verifiers are involved.
"""

import json
from pathlib import Path

from recovery.models import (
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    RecoverySuggestion,
)
from recovery.risk_assessor import RiskAssessment, RiskDecision
from recovery.plan_generator import generate_recovery_plan


OUTPUT_DIR = Path("results/recovery_experiments/consensus_policy")


def make_suggestion(verifier_id, direction, confidence, risk):
    return RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=RecoveryStrategy.GOB,
        actions=[
            RecoveryAction(
                name="move_robot",
                parameters={"direction": direction},
                description=f"Synthetic move {direction} proposal",
                estimated_risk=risk,
            )
        ],
        confidence=confidence,
        explanation=f"Synthetic {verifier_id} recommendation",
        failure_type="controlled_conflict",
        metadata={"synthetic": True},
    )


def main():
    suggestions = [
        make_suggestion("kinematics_verifier", "left", 0.95, 0.20),
        make_suggestion("task_logic_verifier", "right", 0.90, 0.20),
        make_suggestion("object_state_verifier", "right", 0.85, 0.10),
    ]

    assessments = {
        suggestion.verifier_id: RiskAssessment(
            decision=RiskDecision.ACCEPT,
            risk_score=suggestion.actions[0].estimated_risk,
            reasons=["Within configured risk threshold."],
        )
        for suggestion in suggestions
    }

    context = RecoveryContext(
        failed_action_id="consensus_policy_test",
        failed_action="test_move",
        participants=["test_robot"],
        object_state={"available": True},
        robot_state={"safe": True, "status": "stopped"},
        world_state={"simulation": "synthetic_test"},
        previous_actions=[],
        remaining_task=[],
        safety_state={"safe": True},
        attempt_count=0,
        max_attempts=3,
        metadata={"experiment": "confidence_vs_consensus"},
    )

    confidence_plan = generate_recovery_plan(
        suggestions,
        assessments,
        context,
        selection_policy="confidence",
    )
    consensus_plan = generate_recovery_plan(
        suggestions,
        assessments,
        context,
        selection_policy="consensus",
    )

    results = {
        "experiment": "confidence_vs_consensus",
        "synthetic_verifier_inputs": [
            {
                "verifier": s.verifier_id,
                "direction": s.actions[0].parameters["direction"],
                "confidence": s.confidence,
                "risk": s.actions[0].estimated_risk,
            }
            for s in suggestions
        ],
        "confidence_policy": {
            "actions": [
                {
                    "name": a.name,
                    "parameters": a.parameters,
                }
                for a in confidence_plan.actions
            ],
            "accepted_verifiers": confidence_plan.accepted_suggestions,
            "rejected_verifiers": confidence_plan.rejected_suggestions,
            "metadata": confidence_plan.metadata,
        },
        "consensus_policy": {
            "actions": [
                {
                    "name": a.name,
                    "parameters": a.parameters,
                }
                for a in consensus_plan.actions
            ],
            "accepted_verifiers": consensus_plan.accepted_suggestions,
            "rejected_verifiers": consensus_plan.rejected_suggestions,
            "metadata": consensus_plan.metadata,
        },
        "expected": {
            "confidence_direction": "left",
            "consensus_direction": "right",
            "reason": (
                "Two verifiers support right; one verifier with the highest "
                "individual confidence supports left. All proposals pass "
                "the risk assessment."
            ),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "consensus_policy_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    confidence_direction = (
        confidence_plan.actions[0].parameters.get("direction")
        if confidence_plan.actions else None
    )
    consensus_direction = (
        consensus_plan.actions[0].parameters.get("direction")
        if consensus_plan.actions else None
    )

    print("[consensus experiment] synthetic verifier comparison")
    print("[consensus experiment] no physical sensors or hardware used")
    print("\n[inputs]")
    for item in results["synthetic_verifier_inputs"]:
        print(
            f"  {item['verifier']}: direction={item['direction']}, "
            f"confidence={item['confidence']:.2f}, risk={item['risk']:.2f}"
        )

    print("\n[confidence policy]")
    print("  selected direction:", confidence_direction)
    print("  accepted:", confidence_plan.accepted_suggestions)
    print("  rejected:", confidence_plan.rejected_suggestions)

    print("\n[consensus policy]")
    print("  selected direction:", consensus_direction)
    print("  accepted:", consensus_plan.accepted_suggestions)
    print("  rejected:", consensus_plan.rejected_suggestions)

    if confidence_direction != "left":
        raise AssertionError("Confidence policy did not select the expected left proposal.")
    if consensus_direction != "right":
        raise AssertionError("Consensus policy did not select the expected right proposal.")

    print("\n[consensus experiment] PASS: expected policy difference observed")
    print("[consensus experiment] JSON:", output_path.resolve())


if __name__ == "__main__":
    main()
