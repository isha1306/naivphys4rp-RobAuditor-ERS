"""Synthetic multi-role verifier ensemble for deterministic MuJoCo tests.

This adapter is a test fixture, not a connection to physical sensors or the
project's production verifiers. Each role emits an independently identified
result so the recovery pipeline can exercise ensemble normalization, risk
assessment, conflict resolution, and plan synthesis.
"""

from .models import (
    RecoveryAction,
    RecoveryContext,
    RecoverySuggestion,
    RecoveryStrategy,
    VerifierResult,
)
from .recovery_manager import RecoveryManager
from .failure_injector import InjectedFailure


STAGE_POLICIES = {
    "perception": {
        "action": "request_human_intervention",
        "strategy": RecoveryStrategy.HIC,
        "risk": 0.1,
    },
    "pre_grasp": {
        "action": "replan_pre_grasp",
        "strategy": RecoveryStrategy.INA,
        "risk": 0.2,
    },
    "grasp": {
        "action": "return_to_pre_grasp",
        "strategy": RecoveryStrategy.GOB,
        "risk": 0.2,
    },
    "lift": {
        "action": "stop_robot",
        "strategy": RecoveryStrategy.ESS,
        "risk": 0.1,
    },
    "transfer": {
        "action": "return_to_last_safe_waypoint",
        "strategy": RecoveryStrategy.GBB,
        "risk": 0.2,
    },
    "insertion": {
        "action": "replan_insertion",
        "strategy": RecoveryStrategy.INA,
        "risk": 0.2,
    },
    "placement": {
        "action": "return_to_placement_waypoint",
        "strategy": RecoveryStrategy.GOB,
        "risk": 0.2,
    },
    "object_unavailable": {
        "action": "introduce_replacement_canister",
        "strategy": RecoveryStrategy.INO,
        "risk": 0.2,
        "parameters": {
            "missing_object": "canister",
            "replacement_object": "replacement_canister",
            "replacement_type": "canister",
        },
    },
}

# Distinct synthetic roles and confidence values. These are deliberately
# deterministic test fixtures, not measured verifier performance.
VERIFIER_ROLES = (
    ("kinematics_verifier", 0.94, "motion feasibility"),
    ("task_logic_verifier", 0.91, "task-sequence consistency"),
    ("object_state_verifier", 0.88, "object and scene state"),
)


def build_failure_context(failure, attempt_count=0, max_attempts=3):
    """Build a minimal context for a deterministic simulation failure."""

    stage = failure.stage
    object_unavailable = stage == "object_unavailable"

    object_state = {
        "name": "canister",
        "available": not object_unavailable,
    }
    if object_unavailable:
        object_state.update(
            {
                "missing_object": "canister",
                "replacement_candidate": "replacement_canister",
            }
        )

    return RecoveryContext(
        failed_action_id=f"injected_{stage}",
        failed_action=stage,
        participants=["ur10"],
        object_state=object_state,
        robot_state={
            "safe": stage != "lift",
            "status": "unsafe" if stage == "lift" else "stopped",
        },
        world_state={
            "simulation": "mujoco",
            "replacement_object_available": object_unavailable,
        },
        previous_actions=[],
        remaining_task=[],
        safety_state={"safe": stage != "lift"},
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        metadata={
            "failure_source": "deterministic_simulation_injector",
            "failure_id": failure.failure_id,
            "failure_stage": stage,
            "object_unavailable": object_unavailable,
            "verifier_fixture": "synthetic_role_ensemble",
        },
    )


def build_test_verifier_results(failure):
    """Return three role-labelled synthetic verifier results for one failure."""

    if not isinstance(failure, InjectedFailure):
        raise TypeError("Expected an InjectedFailure.")

    policy = STAGE_POLICIES.get(failure.stage)
    if policy is None:
        raise ValueError(f"No test recovery policy for stage: {failure.stage}")

    results = []
    for verifier_id, confidence, focus in VERIFIER_ROLES:
        action = RecoveryAction(
            name=policy["action"],
            parameters=dict(policy.get("parameters", {})),
            description=(
                "Synthetic proposal to introduce a replacement canister."
                if failure.stage == "object_unavailable"
                else "Synthetic recovery action for ensemble adapter testing."
            ),
            estimated_risk=policy["risk"],
        )

        suggestion = RecoverySuggestion(
            verifier_id=verifier_id,
            strategy=policy["strategy"],
            actions=[action],
            confidence=confidence,
            explanation=(
                f"Synthetic {focus} assessment recommends "
                f"{policy['action']} for the injected {failure.stage} failure."
            ),
            modalities=[],
            failure_type=f"{failure.stage}_failure",
            metadata={
                "synthetic": True,
                "verifier_role": verifier_id,
                "focus": focus,
                "object_substitution": failure.stage == "object_unavailable",
            },
        )

        results.append(
            VerifierResult(
                verifier_id=verifier_id,
                decision=False,
                confidence=confidence,
                explanation=(
                    f"Synthetic {focus} verifier detected the injected "
                    f"{failure.stage} failure."
                ),
                suggestion=suggestion,
                evidence={
                    "failure_id": failure.failure_id,
                    "failure_stage": failure.stage,
                    "synthetic_fixture": True,
                    "focus": focus,
                },
            )
        )

    return results


def build_test_verifier_result(failure):
    """Backward-compatible helper returning the first synthetic role result."""

    return build_test_verifier_results(failure)[0]


def process_injected_failure(failure, manager=None):
    """Synthesize and validate a plan; deliberately do not execute robot actions."""

    if not isinstance(failure, InjectedFailure):
        raise TypeError("Expected an InjectedFailure.")

    policy = STAGE_POLICIES.get(failure.stage)
    if policy is None:
        raise ValueError(f"No test recovery policy for stage: {failure.stage}")

    context = build_failure_context(failure)
    verifier_results = build_test_verifier_results(failure)

    if manager is None:
        manager = RecoveryManager(
            ess_action=RecoveryAction(
                name="stop_robot",
                description="Stop the robot and request safe-state confirmation.",
                estimated_risk=0.1,
            )
        )

    outcome = manager.process(verifier_results, context)

    print("\n[recovery] injected failure:", failure.failure_id)
    print("[recovery] synthetic verifier results:", len(verifier_results))
    for result in verifier_results:
        print(
            "[recovery] verifier:",
            result.verifier_id,
            "confidence:",
            result.confidence,
        )
    print("[recovery] test strategy:", policy["strategy"].value)
    print("[recovery] manager status:", outcome.status.value)
    print("[recovery] message:", outcome.message)

    if outcome.plan is not None:
        print("[recovery] proposed actions:")
        for action in outcome.plan.actions:
            print("  -", action.name)
            if action.parameters:
                print("    parameters:", action.parameters)
        print("[recovery] plan status:", outcome.plan.status.value)
        print(
            "[recovery] accepted verifier suggestions:",
            len(outcome.plan.accepted_suggestions),
        )

    for reason in outcome.rejected_reasons:
        print("[recovery] rejected:", reason)

    print("[recovery] actions were NOT executed.")
    if failure.stage == "object_unavailable":
        print(
            "[recovery] INO note: replacement-object introduction is proposed "
            "but no object has been spawned by this adapter."
        )

    return outcome
