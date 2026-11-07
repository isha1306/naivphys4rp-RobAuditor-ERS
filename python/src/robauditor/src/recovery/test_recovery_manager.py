from .models import (
    RecoveryAction,
    RecoveryContext,
    RecoverySuggestion,
    RecoveryStrategy,
    VerifierResult,
    PlanStatus,
)

from .recovery_manager import RecoveryManager


def make_context(robot_state=None, attempt_count=0):
    return RecoveryContext(
        failed_action_id="grasp_001",
        failed_action="grasp_canister",
        participants=[],
        object_state={},
        robot_state=robot_state or {},
        world_state={},
        previous_actions=[],
        remaining_task=[],
        safety_state={},
        attempt_count=attempt_count,
        max_attempts=3,
    )


def make_verifier_results():
    suggestion = RecoverySuggestion(
        verifier_id="grasp_verifier",
        strategy=RecoveryStrategy.GOB,
        actions=[
            RecoveryAction(name="retry_grasp")
        ],
        confidence=0.9,
        explanation="Retry the failed grasp.",
        modalities=[],
        failure_type="grasp_failure",
    )

    result = VerifierResult(
        verifier_id="grasp_verifier",
        decision=False,
        confidence=0.9,
        explanation="The grasp failed.",
        suggestion=suggestion,
        evidence=[],
    )

    return [result]


# ---------------------------------------------------------
# Standard recovery manager tests
# ---------------------------------------------------------

def test_successful_recovery():
    manager = RecoveryManager()
    context = make_context()
    results = make_verifier_results()

    outcome = manager.process(results, context)

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan is not None
    assert len(outcome.plan.actions) > 0


def test_no_suggestions():
    manager = RecoveryManager()
    context = make_context()

    outcome = manager.process([], context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None


def test_unsafe_robot_state():
    manager = RecoveryManager()

    context = make_context(
        robot_state={
            "safe": False,
            "emergency_stop": True,
        }
    )

    results = make_verifier_results()

    outcome = manager.process(results, context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is not None
    assert len(outcome.rejected_reasons) > 0


def test_attempt_limit_reached():
    manager = RecoveryManager()
    context = make_context(attempt_count=3)
    results = make_verifier_results()

    outcome = manager.process(results, context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION


# ---------------------------------------------------------
# Safety priority tests
# ---------------------------------------------------------

def test_unsafe_state_never_allows_ordinary_recovery():
    manager = RecoveryManager()

    context = make_context(
        robot_state={
            "safe": False,
            "emergency_stop": True,
        }
    )

    results = make_verifier_results()

    outcome = manager.process(results, context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION

    if outcome.plan is not None:
        assert RecoveryStrategy.ESS not in outcome.plan.strategies
        assert RecoveryStrategy.GOB not in outcome.plan.strategies


# ---------------------------------------------------------
# ESS (Enter Safe State) tests
# ---------------------------------------------------------

def test_configured_ess_action_is_validated():
    manager = RecoveryManager(
        ess_action=RecoveryAction(
            name="stop_robot",
            parameters={},
            description="Stop robot and enter a safe state",
            estimated_risk=0.1,
        )
    )

    context = make_context(
        robot_state={
            "safe": False,
            "emergency_stop": True,
        }
    )

    outcome = manager.process([], context)

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan is not None
    assert outcome.plan.status == PlanStatus.VALIDATED
    assert outcome.plan.strategies == [RecoveryStrategy.ESS]
    assert len(outcome.plan.actions) == 1
    assert outcome.plan.actions[0].name == "stop_robot"


def test_ess_action_above_risk_threshold_requires_human():
    manager = RecoveryManager(
        max_allowed_risk=0.7,
        ess_action=RecoveryAction(
            name="stop_robot",
            parameters={},
            estimated_risk=0.9,
        ),
    )

    context = make_context(
        robot_state={
            "safe": False,
            "emergency_stop": True,
        }
    )

    outcome = manager.process([], context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None
    assert "exceeds the allowed risk" in outcome.message


def test_ess_action_with_unknown_risk_requires_human():
    manager = RecoveryManager(
        ess_action=RecoveryAction(
            name="stop_robot",
            parameters={},
            estimated_risk=None,
        )
    )

    context = make_context(
        robot_state={
            "safe": False,
            "emergency_stop": True,
        }
    )

    outcome = manager.process([], context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None
    assert "unknown risk" in outcome.message


# ---------------------------------------------------------
# Optional direct execution
# ---------------------------------------------------------

if __name__ == "__main__":
    test_successful_recovery()
    test_no_suggestions()
    test_unsafe_robot_state()
    test_attempt_limit_reached()
    test_unsafe_state_never_allows_ordinary_recovery()
    test_configured_ess_action_is_validated()
    test_ess_action_above_risk_threshold_requires_human()
    test_ess_action_with_unknown_risk_requires_human()

    print("All recovery manager tests passed!")
