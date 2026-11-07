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
KINEMATIC_EMERGENCY_SPEED_LIMIT_RAD_S = 12.0  # Provisional MuJoCo simulation limit; calibrate against normal trajectories.

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
                    "source": "deterministic_simulation_injector",
                },
                visual_evidence=[],
                explanation_metadata={
                    "format": "text",
                    "evidence_origin": "synthetic_simulation_fixture",
                    "visual_observation_available": False,
                },
            )
        )

    return results



def _mujoco_state_snapshot(failure, model, data, mujoco_module):
    """Collect measurable MuJoCo state for the three verifier roles."""
    import numpy as np

    def object_id(obj_type, name):
        return int(mujoco_module.mj_name2id(model, obj_type, name))

    canister_id = object_id(mujoco_module.mjtObj.mjOBJ_BODY, "canister1")
    if canister_id < 0:
        return {"canister_available": False, "state_error": "canister1 body missing"}

    qpos = np.asarray(data.qpos, dtype=float)
    qvel = np.asarray(data.qvel, dtype=float)
    finite_qpos = bool(np.all(np.isfinite(qpos)))
    finite_qvel = bool(np.all(np.isfinite(qvel)))
    max_joint_speed = float(np.max(np.abs(qvel))) if finite_qvel and qvel.size else float("inf")
    canister_pos = np.asarray(data.xpos[canister_id], dtype=float).copy()
    canister_finite = bool(np.all(np.isfinite(canister_pos)))

    canister_geom_ids = set(np.flatnonzero(model.geom_bodyid == canister_id).tolist())
    pad_contact_count = 0
    environment_contact_count = 0
    contact_count = int(data.ncon)
    for index in range(contact_count):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        involves_canister = geom1 in canister_geom_ids or geom2 in canister_geom_ids
        if not involves_canister:
            continue
        other_geom = geom2 if geom1 in canister_geom_ids else geom1
        other_name = mujoco_module.mj_id2name(
            model, mujoco_module.mjtObj.mjOBJ_GEOM, other_geom
        ) or ""
        if "pad" in other_name.lower() or "finger" in other_name.lower():
            pad_contact_count += 1
        else:
            environment_contact_count += 1

    return {
        "canister_available": True,
        "canister_position_m": canister_pos.tolist(),
        "canister_position_finite": canister_finite,
        "joint_positions_finite": finite_qpos,
        "joint_velocities_finite": finite_qvel,
        "max_abs_joint_velocity_rad_s": max_joint_speed,
        "kinematic_emergency_speed_limit_rad_s": KINEMATIC_EMERGENCY_SPEED_LIMIT_RAD_S,
        "contact_count": contact_count,
        "canister_gripper_contact_count": pad_contact_count,
        "canister_environment_contact_count": environment_contact_count,
        "failure_stage": failure.stage,
        "measurement_source": "live_mujoco_state",
    }


def build_state_based_verifier_results(failure, model, data, mujoco_module):
    """Run role-specific live-state checks and derive role-specific proposals.

    Each verifier has its own predicate and recommendation policy. The task
    checkpoint is used as context, but action selection also depends on the
    measured state. Recommendations are proposals only; the manager validates
    them before any execution backend can act.
    """
    if not isinstance(failure, InjectedFailure):
        raise TypeError("Expected an InjectedFailure.")
    policy = STAGE_POLICIES.get(failure.stage)
    if policy is None:
        raise ValueError(f"No test recovery policy for stage: {failure.stage}")

    snapshot = _mujoco_state_snapshot(failure, model, data, mujoco_module)
    state_error = "state_error" in snapshot

    # Kinematics verifier: joint finiteness and measured joint speed.
    kinematics_ok = False if state_error else (
        snapshot["joint_positions_finite"]
        and snapshot["joint_velocities_finite"]
        and snapshot["max_abs_joint_velocity_rad_s"] < KINEMATIC_EMERGENCY_SPEED_LIMIT_RAD_S
    )
    # Task verifier: injected checkpoint was interrupted, so its transition
    # contract is incomplete even if the robot/object measurements look valid.
    task_ok = False
    # Object verifier: independently checks existence, pose, and stage-specific
    # gripper contact. At grasp, contact with both pads is required.
    object_ok = False if state_error else (
        snapshot["canister_available"] and snapshot["canister_position_finite"]
    )
    if failure.stage == "grasp" and object_ok:
        object_ok = snapshot["canister_gripper_contact_count"] >= 2

    def proposal(role, passed):
        if passed:
            return None
        if role == "kinematics_verifier":
            speed = snapshot.get("max_abs_joint_velocity_rad_s", float("inf"))
            if not snapshot.get("joint_positions_finite", False) or not snapshot.get("joint_velocities_finite", False) or speed >= KINEMATIC_EMERGENCY_SPEED_LIMIT_RAD_S:
                return ("stop_robot", RecoveryStrategy.ESS, 0.08,
                        "Measured joint state is non-finite or joint speed exceeds the provisional emergency limit.")
            return ("return_to_last_safe_waypoint", RecoveryStrategy.GBB, 0.18,
                    "Measured joint motion is valid but the kinematic checkpoint was interrupted.")
        if role == "task_logic_verifier":
            return (policy["action"], policy["strategy"], policy["risk"],
                    "The expected task transition was not completed; restore the stage checkpoint.")
        # Object-state-specific recovery follows the measured object condition.
        if not snapshot.get("canister_available", False):
            return ("introduce_replacement_canister", RecoveryStrategy.INO, 0.2,
                    "The required canister is unavailable; propose an object substitution.")
        if failure.stage == "grasp" and snapshot.get("canister_gripper_contact_count", 0) < 2:
            return ("return_to_pre_grasp", RecoveryStrategy.GOB, 0.2,
                    "The canister is not securely contacted by both gripper pads; retreat to a safe pre-grasp state.")
        return ("replan_pre_grasp", RecoveryStrategy.INA, 0.2,
                "The object is present but its measured pose/contact state does not satisfy the checkpoint.")

    roles = (
        ("kinematics_verifier", "joint_state_and_motion", kinematics_ok, 0.93),
        ("task_logic_verifier", "task_checkpoint_contract", task_ok, 0.90),
        ("object_state_verifier", "canister_pose_and_contacts", object_ok, 0.91),
    )
    results = []
    for verifier_id, focus, passed, base_confidence in roles:
        confidence = min(0.99, max(0.51, base_confidence - (0.08 if not passed else 0.0)))
        recommendation = proposal(verifier_id, passed)
        suggestion = None
        if recommendation is not None:
            action_name, strategy, estimated_risk, rationale = recommendation
            params = dict(STAGE_POLICIES.get(failure.stage, {}).get("parameters", {})) if action_name == "introduce_replacement_canister" else {}
            action = RecoveryAction(
                name=action_name, parameters=params,
                description=rationale, estimated_risk=estimated_risk,
            )
            suggestion = RecoverySuggestion(
                verifier_id=verifier_id, strategy=strategy, actions=[action],
                confidence=confidence, explanation=rationale, modalities=[],
                failure_type=f"{failure.stage}_failure",
                metadata={"synthetic": False, "evidence_origin": "live_mujoco_state",
                          "focus": focus, "recommendation_basis": "role_specific_state_predicate"},
            )
        results.append(VerifierResult(
            verifier_id=verifier_id, decision=bool(passed), confidence=confidence,
            explanation=(f"{focus} checks passed from live MuJoCo state." if passed
                         else f"{focus} checks failed from live MuJoCo state."),
            suggestion=suggestion, evidence=dict(snapshot, focus=focus),
            visual_evidence=[], explanation_metadata={
                "format": "structured_state", "evidence_origin": "live_mujoco_state",
                "visual_observation_available": False,
            },
        ))
    return results

def build_test_verifier_result(failure):
    """Backward-compatible helper returning the first synthetic role result."""

    return build_test_verifier_results(failure)[0]



def build_simulation_recovery_manager(
    selection_policy="confidence",
    candidate_count=1,
    sampling_temperature=0.15,
    random_seed=None,
):
    """Build the simulation manager with configurable ensemble selection.

    candidate_count=1 preserves the original deterministic single-plan path.
    Values greater than one enable bounded candidate generation and
    probabilistic selection among validated plans.
    """
    return RecoveryManager(
        ess_action=RecoveryAction(
            name="stop_robot",
            description="Stop the robot and request safe-state confirmation.",
            estimated_risk=0.1,
        ),
        selection_policy=selection_policy,
        candidate_count=candidate_count,
        sampling_temperature=sampling_temperature,
        random_seed=random_seed,
    )

def process_injected_failure(failure, manager=None, model=None, data=None, mujoco_module=None):
    """Synthesize and validate a plan; deliberately do not execute robot actions."""

    if not isinstance(failure, InjectedFailure):
        raise TypeError("Expected an InjectedFailure.")

    policy = STAGE_POLICIES.get(failure.stage)
    if policy is None:
        raise ValueError(f"No test recovery policy for stage: {failure.stage}")

    context = build_failure_context(failure)
    if model is not None and data is not None and mujoco_module is not None:
        verifier_results = build_state_based_verifier_results(failure, model, data, mujoco_module)
        context.metadata.update({"verifier_evidence_origin": "live_mujoco_state"})
        context.object_state.update({k: v for k, v in _mujoco_state_snapshot(failure, model, data, mujoco_module).items() if k in {"canister_available", "canister_position_m"}})
    else:
        verifier_results = build_test_verifier_results(failure)

    if manager is None:
        manager = build_simulation_recovery_manager()

    outcome = manager.process(verifier_results, context)

    print("\n[recovery] injected failure:", failure.failure_id)
    print("[recovery] verifier results:", len(verifier_results))
    print("[recovery] evidence origin:", "live_mujoco_state" if model is not None and data is not None else "synthetic_fixture")
    for result in verifier_results:
        print(
            "[recovery] verifier:",
            result.verifier_id,
            "Ds:",
            result.decision_label.value,
            "Cf:",
            result.confidence,
            "Ed:",
            result.explanation,
            "Er:",
            (
                result.suggestion.strategy.value
                if result.suggestion is not None
                else "not_proposed"
            ),
        )
        print("[recovery] quadruplet:", result.to_quadruplet())

    print("[recovery] test strategy:", policy["strategy"].value)
    print("[recovery] manager status:", outcome.status.value)
    print("[recovery] message:", outcome.message)
    if getattr(outcome, "candidate_plans", None):
        print(
            "[recovery] validated candidate plans:",
            len(outcome.candidate_plans),
        )
    if getattr(outcome, "selection_probabilities", None):
        print("[recovery] candidate selection probabilities:")
        for candidate_id, probability in outcome.selection_probabilities.items():
            print(f"  - {candidate_id}: {probability:.4f}")

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
