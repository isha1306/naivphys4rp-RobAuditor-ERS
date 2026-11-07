"""End-to-end integration tests for verifier quadruplet ensemble synthesis."""

from recovery.models import (
    EvidenceItem,
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    RecoverySuggestion,
    VerifierResult,
)
from recovery.recovery_manager import RecoveryManager


def _context():
    return RecoveryContext(
        failed_action_id="approach_failure_01",
        failed_action="approach_canister",
        robot_state={"safe": True, "status": "ready"},
        safety_state={"safe": True},
        remaining_task=["replan", "approach", "grasp"],
    )


def _suggestion(verifier_id, direction, risk):
    return RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=RecoveryStrategy.INA,
        actions=[
            RecoveryAction(
                name="replan_pre_grasp",
                parameters={"direction": direction},
                estimated_risk=risk,
            )
        ],
        explanation=f"{verifier_id} proposes {direction}.",
    )


def test_full_ensemble_preserves_conflicts_and_all_quadruplets():
    """Two conflicting FAILs and one PASS traverse the manager end-to-end."""
    results = [
        VerifierResult(
            verifier_id="kinematics_verifier",
            decision=False,
            confidence=0.96,
            explanation="Left approach is reachable.",
            suggestion=_suggestion("kinematics_verifier", "left", 0.20),
            evidence={"reachable": True, "workspace_region": "left"},
            visual_evidence=[
                EvidenceItem(
                    evidence_type="image",
                    description="Wrist camera view supporting left approach.",
                    uri="evidence://camera/wrist/frame-101",
                    metadata={"frame": 101, "camera": "wrist"},
                )
            ],
            explanation_metadata={"source": "synthetic_integration_fixture"},
        ),
        VerifierResult(
            verifier_id="task_logic_verifier",
            decision=False,
            confidence=0.82,
            explanation="Right approach satisfies task-side clearance.",
            suggestion=_suggestion("task_logic_verifier", "right", 0.25),
            evidence={"task_clearance": True, "workspace_region": "right"},
            visual_evidence=[
                EvidenceItem(
                    evidence_type="image",
                    description="Overhead view supporting right approach.",
                    uri="evidence://camera/overhead/frame-202",
                    metadata={"frame": 202, "camera": "overhead"},
                )
            ],
            explanation_metadata={"source": "synthetic_integration_fixture"},
        ),
        VerifierResult(
            verifier_id="object_state_verifier",
            decision=True,
            confidence=0.99,
            explanation="Object identity and pose checks passed.",
            evidence={"object_id": "canister_01", "pose_valid": True},
            visual_evidence=[
                EvidenceItem(
                    evidence_type="image",
                    description="Object-state verification frame.",
                    uri="evidence://camera/object/frame-303",
                    metadata={"frame": 303, "camera": "object"},
                )
            ],
            explanation_metadata={"source": "synthetic_integration_fixture"},
        ),
    ]

    outcome = RecoveryManager(selection_policy="confidence").process(
        results, _context()
    )

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan is not None

    # The higher-confidence conflicting recommendation wins.
    assert len(outcome.plan.actions) == 1
    assert outcome.plan.actions[0].parameters["direction"] == "left"

    # Both the return object and the synthesized plan retain all original
    # verifier results, including the PASS verifier with no Er recommendation.
    assert len(outcome.verifier_outputs) == 3
    assert outcome.plan.metadata["verifier_outputs"] == outcome.verifier_outputs

    by_id = {item["verifier_id"]: item for item in outcome.verifier_outputs}
    assert set(by_id) == {
        "kinematics_verifier",
        "task_logic_verifier",
        "object_state_verifier",
    }

    assert by_id["kinematics_verifier"]["Ds"]["label"] == "FAIL"
    assert by_id["kinematics_verifier"]["Cf"]["confidence"] == 0.96
    assert by_id["kinematics_verifier"]["Ed"]["visual_evidence"][0]["uri"] == (
        "evidence://camera/wrist/frame-101"
    )
    assert by_id["kinematics_verifier"]["Er"]["actions"][0]["parameters"] == {
        "direction": "left"
    }

    assert by_id["task_logic_verifier"]["Ds"]["label"] == "FAIL"
    assert by_id["task_logic_verifier"]["Ed"]["visual_evidence"][0]["uri"] == (
        "evidence://camera/overhead/frame-202"
    )
    assert by_id["task_logic_verifier"]["Er"]["actions"][0]["parameters"] == {
        "direction": "right"
    }

    assert by_id["object_state_verifier"]["Ds"]["label"] == "PASS"
    assert by_id["object_state_verifier"]["Ed"]["evidence"]["pose_valid"] is True
    assert by_id["object_state_verifier"]["Ed"]["visual_evidence"][0]["uri"] == (
        "evidence://camera/object/frame-303"
    )
    assert by_id["object_state_verifier"]["Er"] is None


def test_consensus_policy_selects_majority_action_and_preserves_visuals():
    """Two agreeing verifiers outrank one higher-confidence dissenting verifier."""
    results = [
        VerifierResult(
            verifier_id="kinematics_verifier",
            decision=False,
            confidence=0.99,
            explanation="Left path is reachable.",
            suggestion=_suggestion("kinematics_verifier", "left", 0.20),
            visual_evidence=[
                EvidenceItem("image", "Left path view.", "evidence://left")
            ],
        ),
        VerifierResult(
            verifier_id="task_logic_verifier",
            decision=False,
            confidence=0.88,
            explanation="Right path meets task constraints.",
            suggestion=_suggestion("task_logic_verifier", "right", 0.15),
            visual_evidence=[
                EvidenceItem("image", "Right path view.", "evidence://right-1")
            ],
        ),
        VerifierResult(
            verifier_id="object_state_verifier",
            decision=False,
            confidence=0.84,
            explanation="Right path aligns with object pose.",
            suggestion=_suggestion("object_state_verifier", "right", 0.10),
            visual_evidence=[
                EvidenceItem("image", "Object alignment view.", "evidence://right-2")
            ],
        ),
    ]

    outcome = RecoveryManager(selection_policy="consensus").process(
        results, _context()
    )

    assert outcome.status == PlanStatus.VALIDATED
    assert outcome.plan.actions[0].parameters["direction"] == "right"
    assert len(outcome.verifier_outputs) == 3
    assert [
        item["Ed"]["visual_evidence"][0]["uri"]
        for item in outcome.verifier_outputs
    ] == ["evidence://left", "evidence://right-1", "evidence://right-2"]
