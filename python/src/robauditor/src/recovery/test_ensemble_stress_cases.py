"""Second-stage ensemble stress tests for RobAuditor-ERS.

Planning-only tests. These do not start MuJoCo or command robot hardware.
Run from the repository root:
    PYTHONPATH=python/src/robauditor/src python -m pytest \
      python/src/robauditor/src/recovery/test_ensemble_stress_cases.py -v
"""

import pytest

from recovery.models import (
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    RecoverySuggestion,
    VerifierResult,
)
from recovery.recovery_manager import RecoveryManager


def make_context(**overrides):
    values = dict(
        failed_action_id="ensemble_stress_failure",
        failed_action="move_to_pre_grasp",
        robot_state={"safe": True, "status": "ready"},
        safety_state={"safe": True},
        remaining_task=["approach", "grasp", "lift"],
        attempt_count=0,
        max_attempts=3,
    )
    values.update(overrides)
    return RecoveryContext(**values)


def make_result(
    verifier_id,
    *,
    decision=False,
    confidence=0.8,
    action="replan_pre_grasp",
    parameters=None,
    risk=0.2,
    strategy=RecoveryStrategy.INA,
    suggestion=True,
    metadata=None,
    evidence=None,
):
    recovery_suggestion = None
    if suggestion:
        recovery_suggestion = RecoverySuggestion(
            verifier_id=verifier_id,
            strategy=strategy,
            actions=[
                RecoveryAction(
                    name=action,
                    parameters=parameters or {},
                    estimated_risk=risk,
                )
            ],
            confidence=confidence,
            explanation=f"{verifier_id} recovery rationale",
            metadata=metadata or {},
            evidence={"source": verifier_id},
        )
    return VerifierResult(
        verifier_id=verifier_id,
        decision=decision,
        confidence=confidence,
        explanation=f"{verifier_id} verifier assessment",
        suggestion=recovery_suggestion,
        evidence=evidence or {"fixture": "ensemble_stress"},
    )


def test_three_distinct_strategies_are_retained_when_actions_are_compatible():
    results = [
        make_result("gob_verifier", action="return_to_checkpoint", strategy=RecoveryStrategy.GOB),
        make_result("ina_verifier", action="recompute_approach", strategy=RecoveryStrategy.INA),
        make_result("gbb_verifier", action="restore_previous_state", strategy=RecoveryStrategy.GBB),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert set(outcome.plan.strategies) == {
        RecoveryStrategy.GOB, RecoveryStrategy.INA, RecoveryStrategy.GBB
    }
    assert {action.name for action in outcome.plan.actions} == {
        "return_to_checkpoint", "recompute_approach", "restore_previous_state"
    }


def test_two_compatible_actions_survive_while_explicitly_incompatible_action_is_rejected():
    results = [
        make_result(
            "retreat", confidence=0.94, action="retreat_robot",
            metadata={"conflicts_with_actions": ["advance_robot"]},
        ),
        make_result("inspect", confidence=0.88, action="inspect_object"),
        make_result("advance", confidence=0.83, action="advance_robot"),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert {action.name for action in outcome.plan.actions} == {
        "retreat_robot", "inspect_object"
    }
    assert "advance" in outcome.plan.rejected_suggestions


def test_lower_risk_conflicting_recommendation_wins_under_consensus_policy():
    results = [
        make_result(
            "higher_confidence_risky", confidence=0.98,
            action="fast_recovery", parameters={"route": "direct"}, risk=0.55,
            metadata={"conflicts_with_actions": ["safe_recovery"]},
        ),
        make_result(
            "lower_risk_a", confidence=0.84,
            action="safe_recovery", parameters={"route": "detour"}, risk=0.12,
        ),
        make_result(
            "lower_risk_b", confidence=0.82,
            action="safe_recovery", parameters={"route": "detour"}, risk=0.10,
        ),
    ]
    outcome = RecoveryManager(selection_policy="consensus").process(
        results, make_context()
    )

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan.actions[0].parameters["route"] == "detour"
    assert set(outcome.plan.accepted_suggestions) == {"lower_risk_a", "lower_risk_b"}


def test_all_pass_results_without_recovery_suggestions_do_not_create_a_plan():
    results = [
        make_result("kinematics", decision=True, suggestion=False),
        make_result("task_logic", decision=True, suggestion=False),
        make_result("object_state", decision=True, suggestion=False),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None
    assert len(outcome.verifier_outputs) == 3
    assert all(item["Ds"]["label"] == "PASS" for item in outcome.verifier_outputs)


def test_contradictory_evidence_is_preserved_without_being_silently_reconciled():
    results = [
        make_result(
            "visual_verifier", confidence=0.86, action="reinspect_object",
            evidence={"object_visible": True, "source": "camera_frame_17"},
        ),
        make_result(
            "state_verifier", confidence=0.81, action="reinspect_object",
            evidence={"object_visible": False, "source": "scene_graph_17"},
        ),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    preserved = {item["verifier_id"]: item["Ed"]["evidence"] for item in outcome.verifier_outputs}
    assert preserved["visual_verifier"]["object_visible"] is True
    assert preserved["state_verifier"]["object_visible"] is False
    # The planner preserves the disagreement; it does not claim to adjudicate
    # the truth of the underlying sensor evidence.


def test_probabilistic_selection_returns_only_validated_candidates_and_normalized_probabilities():
    results = [
        make_result(
            "left_verifier", confidence=0.9, action="move_robot",
            parameters={"direction": "left"}, risk=0.2,
        ),
        make_result(
            "right_verifier", confidence=0.88, action="move_robot",
            parameters={"direction": "right"}, risk=0.18,
        ),
    ]
    manager = RecoveryManager(
        candidate_count=5,
        sampling_temperature=0.2,
        random_seed=19,
    )
    outcome = manager.process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan in outcome.candidate_plans
    assert outcome.candidate_plans
    assert all(plan.status == PlanStatus.VALIDATED for plan in outcome.candidate_plans)
    assert outcome.selection_probabilities
    assert sum(outcome.selection_probabilities.values()) == pytest.approx(1.0)
    assert outcome.plan.recovery_id in outcome.selection_probabilities


def test_forbidden_recovery_action_fails_validation_and_requires_human_intervention():
    results = [
        make_result("blocked_action_verifier", action="move_through_human_zone")
    ]
    manager = RecoveryManager(
        safety_constraints={"forbidden_actions": ["move_through_human_zone"]}
    )
    outcome = manager.process(results, make_context())

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None or outcome.plan.status != PlanStatus.VALIDATED


def test_missing_confidence_is_handled_without_crashing_or_inventing_confidence():
    result = make_result("confidence_missing", confidence=None, action="reinspect_scene")
    outcome = RecoveryManager().process([result], make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan.actions[0].name == "reinspect_scene"
    assert outcome.verifier_outputs[0]["Cf"]["confidence"] is None
