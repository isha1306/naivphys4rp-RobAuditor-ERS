"""Tests for multi-candidate probabilistic recovery plan synthesis."""

import pytest

from recovery.models import (
    PlanStatus, RecoveryAction, RecoveryContext, RecoveryStrategy,
    RecoverySuggestion, VerifierResult,
)
from recovery.recovery_manager import RecoveryManager


def _context():
    return RecoveryContext(
        failed_action_id="multi_candidate_failure",
        failed_action="approach_object",
        robot_state={"safe": True, "status": "ready"},
        safety_state={"safe": True},
    )


def _result(verifier_id, direction, confidence, risk):
    suggestion = RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=RecoveryStrategy.INA,
        actions=[RecoveryAction(
            name="replan_approach",
            parameters={"direction": direction},
            estimated_risk=risk,
        )],
        confidence=confidence,
        explanation=f"Recommend {direction}.",
    )
    return VerifierResult(
        verifier_id=verifier_id,
        decision=False,
        confidence=confidence,
        explanation=f"{verifier_id} recommends {direction}.",
        suggestion=suggestion,
    )


def _ensemble():
    return [
        _result("kinematics_verifier", "left", 0.96, 0.20),
        _result("task_logic_verifier", "right", 0.85, 0.15),
    ]


def test_multi_candidate_mode_generates_alternatives_and_probabilities():
    outcome = RecoveryManager(
        selection_policy="confidence",
        candidate_count=5,
        sampling_temperature=0.2,
        random_seed=7,
    ).process(_ensemble(), _context())

    assert outcome.status == PlanStatus.VALIDATED
    assert len(outcome.candidate_plans) == 2
    assert len(outcome.selection_probabilities) == 2
    assert abs(sum(outcome.selection_probabilities.values()) - 1.0) < 1e-9
    assert {
        plan.actions[0].parameters["direction"]
        for plan in outcome.candidate_plans
    } == {"left", "right"}
    assert outcome.plan in outcome.candidate_plans
    assert all(plan.status == PlanStatus.VALIDATED for plan in outcome.candidate_plans)


def test_sampling_is_reproducible_with_fixed_seed():
    result_a = RecoveryManager(candidate_count=5, random_seed=123).process(
        _ensemble(), _context()
    )
    result_b = RecoveryManager(candidate_count=5, random_seed=123).process(
        _ensemble(), _context()
    )
    assert result_a.plan.actions[0].parameters == result_b.plan.actions[0].parameters
    assert result_a.selection_probabilities == result_b.selection_probabilities


def test_legacy_single_candidate_mode_remains_available():
    outcome = RecoveryManager(candidate_count=1, random_seed=1).process(
        _ensemble(), _context()
    )
    assert outcome.status == PlanStatus.VALIDATED
    assert len(outcome.candidate_plans) == 1
    assert len(outcome.selection_probabilities) == 1


def test_invalid_candidate_configuration_is_rejected():
    with pytest.raises(ValueError):
        RecoveryManager(candidate_count=0)
    with pytest.raises(ValueError):
        RecoveryManager(sampling_temperature=0)
