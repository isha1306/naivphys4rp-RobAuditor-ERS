"""Tests for the Ds/Cf/Ed/Er verifier-output contract."""

import pytest

from recovery.models import (
    EvidenceItem,
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    VerifierDecision,
    VerifierResult,
    RecoverySuggestion,
)
from recovery.suggestion_normalizer import normalize_suggestion
from recovery.recovery_manager import RecoveryManager
from recovery.models import PlanStatus


def make_suggestion(verifier_id="kinematics_verifier"):
    return RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=RecoveryStrategy.INA,
        actions=[
            RecoveryAction(
                name="replan_pre_grasp",
                parameters={"offset": 0.02},
                estimated_risk=0.2,
            )
        ],
        confidence=0.91,
        explanation="Fallback explanation from suggestion.",
    )


def test_quadruplet_serializes_all_four_fields():
    result = VerifierResult(
        verifier_id="kinematics_verifier",
        decision=False,
        confidence=0.94,
        explanation="Target pose is outside the reachable region.",
        suggestion=make_suggestion(),
        evidence={"joint_limit": "joint_3", "measured_value": 2.1},
        visual_evidence=[
            EvidenceItem(
                evidence_type="image",
                uri="camera://frame/123",
                description="Camera frame at failure time.",
                metadata={"camera": "wrist", "timestamp": 12.5},
            )
        ],
        explanation_metadata={"format": "text+image"},
    )

    output = result.to_quadruplet()

    assert output["Ds"] == {"decision": False, "label": "FAIL"}
    assert output["Cf"]["confidence"] == pytest.approx(0.94)
    assert output["Ed"]["text"].startswith("Target pose")
    assert output["Ed"]["evidence"]["joint_limit"] == "joint_3"
    assert output["Ed"]["visual_evidence"][0]["uri"] == "camera://frame/123"
    assert output["Er"]["strategy"] == "INA"
    assert output["Er"]["actions"][0]["name"] == "replan_pre_grasp"


def test_decision_labels_cover_pass_fail_and_uncertain():
    assert VerifierResult("v1", True).decision_label == VerifierDecision.PASS
    assert VerifierResult("v2", False).decision_label == VerifierDecision.FAIL
    assert VerifierResult("v3", None).decision_label == VerifierDecision.UNCERTAIN


def test_pass_result_is_not_normalized_into_recovery_suggestion():
    result = VerifierResult(
        verifier_id="task_logic_verifier",
        decision=True,
        confidence=0.99,
        explanation="Task sequence is valid.",
        suggestion=make_suggestion("task_logic_verifier"),
    )
    assert normalize_suggestion(result) is None


def test_fail_normalization_preserves_explanation_and_evidence():
    result = VerifierResult(
        verifier_id="kinematics_verifier",
        decision=False,
        confidence=0.93,
        explanation="Reachability check failed.",
        suggestion=make_suggestion(),
        evidence={"constraint": "workspace"},
        visual_evidence=[
            EvidenceItem("image", "Workspace view.", "camera://frame/9")
        ],
    )

    suggestion = normalize_suggestion(result)

    assert suggestion is not None
    assert suggestion.verifier_id == "kinematics_verifier"
    assert suggestion.decision is False
    assert suggestion.explanation == "Reachability check failed."
    assert suggestion.evidence["constraint"] == "workspace"
    assert suggestion.visual_evidence[0].uri == "camera://frame/9"


def test_uncertain_result_with_suggestion_is_retained():
    result = VerifierResult(
        verifier_id="object_state_verifier",
        decision=None,
        confidence=0.55,
        explanation="Object state is ambiguous.",
        suggestion=make_suggestion("object_state_verifier"),
    )
    suggestion = normalize_suggestion(result)
    assert suggestion is not None
    assert suggestion.decision is None


def test_missing_recovery_suggestion_does_not_invent_er():
    result = VerifierResult(
        verifier_id="task_logic_verifier",
        decision=False,
        confidence=0.8,
        explanation="Task precondition failed.",
    )
    assert result.to_quadruplet()["Er"] is None
    assert normalize_suggestion(result) is None


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        VerifierResult("bad_verifier", False, confidence=1.2)


def test_manager_retains_pass_quadruplet_even_without_recovery_suggestion():
    result = VerifierResult(
        verifier_id="task_logic_verifier",
        decision=True,
        confidence=0.98,
        explanation="Task logic checks passed.",
        evidence={"checks": 12, "failed_checks": 0},
    )
    context = RecoveryContext(
        failed_action_id="test_failure",
        failed_action="test_action",
    )

    outcome = RecoveryManager().process([result], context)

    assert outcome.status == PlanStatus.HUMAN_INTERVENTION
    assert len(outcome.verifier_outputs) == 1
    assert outcome.verifier_outputs[0]["Ds"]["label"] == "PASS"
    assert outcome.verifier_outputs[0]["Ed"]["evidence"]["checks"] == 12
