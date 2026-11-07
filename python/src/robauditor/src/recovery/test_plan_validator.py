from .models import (
    RecoveryAction,
    RecoveryPlan,
    RecoveryStrategy,
    RecoveryContext,
    PlanStatus,
)

from .plan_validator import validate_plan


def make_context(
    attempt_count=0,
    max_attempts=3,
    robot_state=None,
    object_state=None,
    safety_state=None,
):
    return RecoveryContext(
        failed_action_id="grasp_001",
        failed_action="grasp_canister",
        participants=[],
        object_state=object_state or {},
        robot_state=robot_state or {},
        world_state={},
        previous_actions=[],
        remaining_task=[],
        safety_state=safety_state or {},
        attempt_count=attempt_count,
        max_attempts=max_attempts,
    )


def make_plan(actions=None, status=PlanStatus.PROPOSED):
    return RecoveryPlan(
        recovery_id="recovery_test",
        failed_action_id="grasp_001",
        actions=(
            actions if actions is not None
            else [RecoveryAction(name="retry_grasp")]
        ),
        strategies=[RecoveryStrategy.GOB],
        status=status,
        explanation="Test recovery plan",
        accepted_suggestions=["test_verifier"],
        rejected_suggestions={},
        safety_constraints=[],
        max_attempts=3,
    )


def test_valid_plan():
    plan = make_plan()
    context = make_context()

    result = validate_plan(plan, context)

    assert result.valid is True


def test_empty_plan():
    plan = make_plan(actions=[])
    context = make_context()

    result = validate_plan(plan, context)

    assert result.valid is False
    assert any("no actions" in reason for reason in result.reasons)


def test_attempt_limit_reached():
    plan = make_plan()
    context = make_context(
        attempt_count=3,
        max_attempts=3,
    )

    result = validate_plan(plan, context)

    assert result.valid is False
    assert any("attempt limit" in reason for reason in result.reasons)


def test_unsafe_robot_state():
    plan = make_plan()
    context = make_context(
        robot_state={"collision_detected": True}
    )

    result = validate_plan(plan, context)

    assert result.valid is False
    assert any("collision_detected" in reason for reason in result.reasons)


def test_forbidden_action():
    plan = make_plan()
    context = make_context()

    result = validate_plan(
        plan,
        context,
        safety_constraints={
            "forbidden_actions": ["retry_grasp"]
        },
    )

    assert result.valid is False
    assert any("forbidden" in reason for reason in result.reasons)


def test_satisfied_prerequisite():
    action = RecoveryAction(
        name="place_canister",
        prerequisites=["robot.holding_object"],
    )

    plan = make_plan(actions=[action])

    context = make_context(
        robot_state={"holding_object": True}
    )

    result = validate_plan(plan, context)

    assert result.valid is True


def test_unsatisfied_prerequisite():
    action = RecoveryAction(
        name="place_canister",
        prerequisites=["robot.holding_object"],
    )

    plan = make_plan(actions=[action])
    context = make_context(
        robot_state={"holding_object": False}
    )

    result = validate_plan(plan, context)

    assert result.valid is False
    assert any(
        "robot.holding_object" in reason
        for reason in result.reasons
    )


if __name__ == "__main__":
    test_valid_plan()
    test_empty_plan()
    test_attempt_limit_reached()
    test_unsafe_robot_state()
    test_forbidden_action()
    test_satisfied_prerequisite()
    test_unsatisfied_prerequisite()

    print("All plan validator tests passed!")