from recovery.models import (
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    VerifierResult,
    PlanStatus,
)
from recovery.recovery_manager import RecoveryManager


def make_context(**kwargs):
    defaults = {
        "failed_action_id": "action_05",
        "failed_action": "insert_canister",
        "previous_actions": [
            "approach_canister",
            "grasp_canister",
            "lift_canister",
            "transfer_canister",
        ],
        "remaining_task": ["insert_canister", "release_canister"],
        "robot_state": {"safe": True},
        "world_state": {},
        "safety_state": {},
    }
    defaults.update(kwargs)
    return RecoveryContext(**defaults)


def make_result(strategy, action_name, confidence=0.9, metadata=None):
    action = RecoveryAction(
        name=action_name,
        parameters=metadata or {},
        description=f"Test recovery action: {action_name}",
        estimated_risk=0.1,
    )

    from recovery.models import RecoverySuggestion

    suggestion = RecoverySuggestion(
        verifier_id=f"verifier_{strategy.value.lower()}",
        strategy=strategy,
        actions=[action],
        confidence=confidence,
        explanation=f"Test suggestion for {strategy.value}",
    )

    return VerifierResult(
        verifier_id=suggestion.verifier_id,
        decision=False,
        confidence=confidence,
        explanation="Injected test failure",
        suggestion=suggestion,
    )


def test_ess_strategy_is_available():
    assert RecoveryStrategy.ESS.value == "ESS"


def test_hic_strategy_is_available():
    assert RecoveryStrategy.HIC.value == "HIC"


def test_gbb_strategy_is_available():
    assert RecoveryStrategy.GBB.value == "GBB"


def test_gob_strategy_is_available():
    assert RecoveryStrategy.GOB.value == "GOB"


def test_ina_strategy_is_available():
    assert RecoveryStrategy.INA.value == "INA"


def test_ino_strategy_is_available():
    assert RecoveryStrategy.INO.value == "INO"


def test_ess_unsafe_state_requires_human_intervention():
    context = make_context(
        robot_state={"safe": False},
        safety_state="unsafe",
    )

    result = RecoveryManager().process(
        [
            make_result(
                RecoveryStrategy.ESS,
                "enter_safe_state",
            )
        ],
        context,
    )

    assert result.status == PlanStatus.HUMAN_INTERVENTION


def test_hic_requests_human_intervention_without_suggestion():
    context = make_context()

    result = RecoveryManager().process([], context)

    assert result.status == PlanStatus.HUMAN_INTERVENTION


def test_gbb_preserves_previous_stage_parameter():
    context = make_context()

    verifier_result = make_result(
        RecoveryStrategy.GBB,
        "return_to_previous_safe_stage",
        metadata={"target_stage": "pre_insertion"},
    )

    result = RecoveryManager().process(
        [verifier_result],
        context,
    )

    assert result.status == PlanStatus.VALIDATED
    assert RecoveryStrategy.GBB in result.plan.strategies

    action = result.plan.actions[0]
    assert action.parameters["target_stage"] == "pre_insertion"


def test_gob_preserves_specific_waypoint_parameter():
    context = make_context()

    verifier_result = make_result(
        RecoveryStrategy.GOB,
        "return_to_waypoint",
        metadata={"waypoint": "canister_pre_pick"},
    )

    result = RecoveryManager().process(
        [verifier_result],
        context,
    )

    assert result.status == PlanStatus.VALIDATED
    assert RecoveryStrategy.GOB in result.plan.strategies

    action = result.plan.actions[0]
    assert action.parameters["waypoint"] == "canister_pre_pick"


def test_gbb_and_gob_are_distinct_strategies():
    assert RecoveryStrategy.GBB != RecoveryStrategy.GOB


def test_ina_preserves_alternative_action():
    context = make_context()

    verifier_result = make_result(
        RecoveryStrategy.INA,
        "align_canister_using_alternative_approach",
    )

    result = RecoveryManager().process(
        [verifier_result],
        context,
    )

    assert result.status == PlanStatus.VALIDATED
    assert RecoveryStrategy.INA in result.plan.strategies
    assert (
        result.plan.actions[0].name
        == "align_canister_using_alternative_approach"
    )


def test_ino_preserves_alternative_object():
    context = make_context()

    verifier_result = make_result(
        RecoveryStrategy.INO,
        "select_alternative_canister",
        metadata={"object_id": "canister_02"},
    )

    result = RecoveryManager().process(
        [verifier_result],
        context,
    )

    assert result.status == PlanStatus.VALIDATED
    assert RecoveryStrategy.INO in result.plan.strategies
    assert (
        result.plan.actions[0].parameters["object_id"]
        == "canister_02"
    )