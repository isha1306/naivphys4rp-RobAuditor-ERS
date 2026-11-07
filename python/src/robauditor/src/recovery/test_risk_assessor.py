from .models import (
    RecoveryAction,
    RecoveryStrategy,
    RecoverySuggestion,
    RecoveryContext,
)

from .risk_assessor import (
    RiskDecision,
    assess_suggestion,
)


def make_suggestion(action_name, risk=0.2):
    return RecoverySuggestion(
        verifier_id="test_verifier",
        strategy=RecoveryStrategy.GOB,
        actions=[
            RecoveryAction(
                name=action_name,
                estimated_risk=risk,
            )
        ],
        confidence=0.95,
        explanation="Test recovery suggestion",
    )


def make_context(
    attempt_count=0,
    max_attempts=3,
    safety_state=None,
):
    return RecoveryContext(
        failed_action_id="grasp_001",
        failed_action="grasp_canister",
        participants=[],
        object_state={},
        robot_state={},
        world_state={},
        previous_actions=[],
        remaining_task=[],
        safety_state=safety_state or {},
        attempt_count=attempt_count,
        max_attempts=max_attempts,
    )


def test_safe_suggestion():
    suggestion = make_suggestion("retry_grasp", risk=0.2)

    result = assess_suggestion(suggestion)

    assert result.decision == RiskDecision.ACCEPT


def test_high_risk_suggestion():
    suggestion = make_suggestion("move_robot", risk=0.9)

    result = assess_suggestion(
        suggestion,
        max_allowed_risk=0.7,
    )

    assert result.decision == RiskDecision.REJECT


def test_unsafe_robot_state():
    suggestion = make_suggestion("retry_grasp")

    context = make_context(
        safety_state={"safe": False}
    )

    result = assess_suggestion(suggestion, context)

    assert result.decision == RiskDecision.REQUIRES_HUMAN


def test_maximum_attempts_reached():
    suggestion = make_suggestion("retry_grasp")

    context = make_context(
        attempt_count=3,
        max_attempts=3,
    )

    result = assess_suggestion(suggestion, context)

    assert result.decision == RiskDecision.REQUIRES_HUMAN


def test_forbidden_action():
    suggestion = make_suggestion("move_into_restricted_zone")

    result = assess_suggestion(
        suggestion,
        safety_constraints={
            "forbidden_actions": ["move_into_restricted_zone"]
        },
    )

    assert result.decision == RiskDecision.REJECT


def test_safety_overrides_low_risk():
    suggestion = make_suggestion("retry_grasp", risk=0.05)

    context = make_context(
        safety_state={"collision_detected": True}
    )

    result = assess_suggestion(suggestion, context)

    assert result.decision == RiskDecision.REQUIRES_HUMAN


if __name__ == "__main__":
    test_safe_suggestion()
    test_high_risk_suggestion()
    test_unsafe_robot_state()
    test_maximum_attempts_reached()
    test_forbidden_action()
    test_safety_overrides_low_risk()

    print("All risk assessor tests passed!")

def test_unknown_action_risk_requires_human():
    suggestion = make_suggestion(
        "move_to_waypoint",
        risk=None,
    )

    context = make_context()

    result = assess_suggestion(
        suggestion,
        context=context,
        max_allowed_risk=0.7,
    )

    assert result.decision == RiskDecision.REQUIRES_HUMAN