"""Edge-case tests for RobAuditor-ERS ensemble recovery synthesis.

Run from the repository root with:
    PYTHONPATH=python/src/robauditor/src python -m pytest \
      python/src/robauditor/src/recovery/test_ensemble_edge_cases.py -q

These tests exercise planning only; they do not start MuJoCo or move a robot.
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
        failed_action_id="edge_case_failure",
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
            explanation=f"{verifier_id} recommendation",
            metadata=metadata or {},
        )
    return VerifierResult(
        verifier_id=verifier_id,
        decision=decision,
        confidence=confidence,
        explanation=f"{verifier_id} assessment",
        suggestion=recovery_suggestion,
        evidence={"fixture": "ensemble_edge_case"},
    )


def test_pass_verifier_with_recovery_suggestion_is_not_used():
    results = [
        make_result("pass_verifier", decision=True, action="unsafe_recovery"),
        make_result("fail_verifier", decision=False, action="safe_recovery"),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert [a.name for a in outcome.plan.actions] == ["safe_recovery"]
    assert len(outcome.verifier_outputs) == 2
    assert outcome.verifier_outputs[0]["Ds"]["label"] == "PASS"


def test_uncertain_verifier_with_suggestion_can_contribute():
    uncertain = make_result(
        "uncertain_verifier", decision=None, confidence=0.7,
        action="retry_local_alignment",
    )
    outcome = RecoveryManager().process([uncertain], make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan.actions[0].name == "retry_local_alignment"
    assert outcome.verifier_outputs[0]["Ds"]["label"] == "UNCERTAIN"


def test_all_uncertain_without_suggestions_requires_human_intervention():
    results = [
        make_result("uncertain_a", decision=None, suggestion=False),
        make_result("uncertain_b", decision=None, suggestion=False),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None
    assert len(outcome.verifier_outputs) == 2


def test_conflicting_three_way_recommendations_select_one_and_record_rejections():
    results = [
        make_result("left_verifier", confidence=0.91, parameters={"direction": "left"}),
        make_result("right_verifier", confidence=0.88, parameters={"direction": "right"}),
        make_result("rear_verifier", confidence=0.82, parameters={"direction": "rear"}),
    ]
    outcome = RecoveryManager(selection_policy="confidence").process(
        results, make_context()
    )

    assert outcome.status == PlanStatus.VALIDATED
    assert len(outcome.plan.actions) == 1
    assert outcome.plan.actions[0].parameters["direction"] == "left"
    assert len(outcome.plan.rejected_suggestions) == 2


def test_high_confidence_but_high_risk_recommendation_is_rejected():
    results = [
        make_result(
            "overconfident_verifier", confidence=0.999,
            action="aggressive_recovery", risk=0.96,
        ),
        make_result(
            "conservative_verifier", confidence=0.72,
            action="conservative_recovery", risk=0.12,
        ),
    ]
    outcome = RecoveryManager(max_allowed_risk=0.7).process(
        results, make_context()
    )

    assert outcome.status == PlanStatus.VALIDATED
    assert [a.name for a in outcome.plan.actions] == ["conservative_recovery"]
    assert "overconfident_verifier" in outcome.plan.rejected_suggestions


def test_consensus_prefers_two_matching_verifiers_over_one_dissenter():
    results = [
        make_result(
            "single_high_confidence", confidence=0.99,
            parameters={"direction": "left"},
        ),
        make_result(
            "majority_a", confidence=0.86,
            parameters={"direction": "right"},
        ),
        make_result(
            "majority_b", confidence=0.83,
            parameters={"direction": "right"},
        ),
    ]
    outcome = RecoveryManager(selection_policy="consensus").process(
        results, make_context()
    )

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan.actions[0].parameters["direction"] == "right"
    assert set(outcome.plan.accepted_suggestions) == {"majority_a", "majority_b"}


def test_compatible_recommendations_are_merged_and_exact_duplicates_removed():
    results = [
        make_result("verifier_a", action="check_clearance"),
        make_result("verifier_b", action="confirm_object_pose"),
        make_result("verifier_c", action="check_clearance"),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert [a.name for a in outcome.plan.actions] == [
        "check_clearance", "confirm_object_pose"
    ]


def test_explicit_action_incompatibility_is_not_merged():
    results = [
        make_result(
            "retreat_verifier", confidence=0.92,
            action="retreat_robot",
            metadata={"conflicts_with_actions": ["advance_robot"]},
        ),
        make_result(
            "advance_verifier", confidence=0.85,
            action="advance_robot",
        ),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.VALIDATED
    assert [a.name for a in outcome.plan.actions] == ["retreat_robot"]
    assert "advance_verifier" in outcome.plan.rejected_suggestions


def test_malformed_or_missing_recommendations_do_not_create_empty_valid_plan():
    results = [
        make_result("no_er_a", decision=False, suggestion=False),
        make_result("no_er_b", decision=None, suggestion=False),
    ]
    outcome = RecoveryManager().process(results, make_context())

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert outcome.plan is None or not outcome.plan.actions


def test_explicit_unsafe_context_takes_ess_priority_over_non_ess_suggestion():
    results = [make_result("task_verifier", action="retry_task")]
    ess_action = RecoveryAction(
        name="stop_robot", parameters={}, estimated_risk=0.1
    )
    manager = RecoveryManager(ess_action=ess_action)
    context = make_context(
        robot_state={"safe": False, "status": "unsafe"},
        safety_state={"safe": False},
    )
    outcome = manager.process(results, context)

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan.strategies == [RecoveryStrategy.ESS]
    assert [a.name for a in outcome.plan.actions] == ["stop_robot"]
    assert outcome.plan.metadata.get("ess_priority") is True
