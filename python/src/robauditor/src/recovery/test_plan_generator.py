from .models import (
    RecoveryAction,
    RecoveryStrategy,
    RecoverySuggestion,
    RecoveryContext,
    PlanStatus,
)

from .risk_assessor import RiskAssessment, RiskDecision
from .plan_generator import generate_recovery_plan


def make_suggestion(verifier_id, action_name, confidence=0.8, risk=0.2):
    return RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=RecoveryStrategy.GOB,
        actions=[
            RecoveryAction(
                name=action_name,
                estimated_risk=risk,
            )
        ],
        confidence=confidence,
        explanation="Test recovery suggestion",
    )


def make_context():
    return RecoveryContext(
        failed_action_id="grasp_001",
        failed_action="grasp_canister",
        max_attempts=3,
    )


def make_assessment(decision=RiskDecision.ACCEPT):
    return RiskAssessment(
        decision=decision,
        risk_score=0.2,
        reasons=["Test assessment"],
    )


def test_compatible_suggestions_are_combined():
    suggestions = [
        make_suggestion("grasp_verifier", "retry_grasp"),
        make_suggestion("safety_verifier", "move_to_safe_position"),
    ]

    assessments = {
        "grasp_verifier": make_assessment(),
        "safety_verifier": make_assessment(),
    }

    plan = generate_recovery_plan(
        suggestions,
        assessments,
        make_context(),
    )

    assert len(plan.actions) == 2
    assert len(plan.accepted_suggestions) == 2
    assert plan.status == PlanStatus.PROPOSED


def test_duplicate_actions_are_removed():
    suggestions = [
        make_suggestion("verifier_a", "retry_grasp"),
        make_suggestion("verifier_b", "retry_grasp"),
    ]

    assessments = {
        "verifier_a": make_assessment(),
        "verifier_b": make_assessment(),
    }

    plan = generate_recovery_plan(
        suggestions,
        assessments,
        make_context(),
    )

    assert len(plan.actions) == 1
    assert len(plan.accepted_suggestions) == 2


def test_unsafe_suggestion_is_excluded():
    suggestions = [
        make_suggestion("safe_verifier", "retry_grasp"),
        make_suggestion("unsafe_verifier", "move_robot"),
    ]

    assessments = {
        "safe_verifier": make_assessment(RiskDecision.ACCEPT),
        "unsafe_verifier": make_assessment(RiskDecision.REJECT),
    }

    plan = generate_recovery_plan(
        suggestions,
        assessments,
        make_context(),
    )

    assert len(plan.actions) == 1
    assert "unsafe_verifier" in plan.rejected_suggestions
    assert "safe_verifier" in plan.accepted_suggestions


def test_conflicting_suggestions():
    suggestions = [
        make_suggestion(
            "left_verifier",
            "move_robot",
            confidence=0.9,
        ),
        make_suggestion(
            "right_verifier",
            "move_robot",
            confidence=0.7,
        ),
    ]

    # Different values for the same parameter create a conflict.
    suggestions[0].actions[0].parameters = {"direction": "left"}
    suggestions[1].actions[0].parameters = {"direction": "right"}

    assessments = {
        "left_verifier": make_assessment(),
        "right_verifier": make_assessment(),
    }

    plan = generate_recovery_plan(
        suggestions,
        assessments,
        make_context(),
    )

    assert "left_verifier" in plan.accepted_suggestions
    assert "right_verifier" in plan.rejected_suggestions


def test_no_safe_suggestions_requires_human():
    suggestions = [
        make_suggestion("unsafe_verifier", "retry_grasp")
    ]

    assessments = {
        "unsafe_verifier": make_assessment(RiskDecision.REJECT)
    }

    plan = generate_recovery_plan(
        suggestions,
        assessments,
        make_context(),
    )

    assert len(plan.actions) == 0
    assert plan.status == PlanStatus.HUMAN_INTERVENTION


if __name__ == "__main__":
    test_compatible_suggestions_are_combined()
    test_duplicate_actions_are_removed()
    test_unsafe_suggestion_is_excluded()
    test_conflicting_suggestions()
    test_no_safe_suggestions_requires_human()

    print("All plan generator tests passed!")