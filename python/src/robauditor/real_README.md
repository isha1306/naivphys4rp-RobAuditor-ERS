# RobAuditor: Real-Robot Integration Guide

## 1. Purpose

This document explains how to integrate the existing RobAuditor ensemble-based recovery system with a physical robot using the components already available in this repository.

The goal is to connect the recovery reasoning pipeline to the robot's existing control and sensing infrastructure. RobAuditor should be responsible for:

1. Receiving information about robot actions and execution state.
2. Collecting evidence from available sensors and robot interfaces.
3. Detecting execution failures through independent verifiers.
4. Combining verifier recommendations into a coherent recovery plan.
5. Validating the plan against the current robot and safety state.
6. Passing an approved plan to a robot-specific execution backend.
7. Checking whether the recovery actually succeeded.
8. Recording the result for traceability and audit.

**Important implementation status:** The repository contains the recovery planning architecture and a simulation execution backend. `RealRobotRecoveryBackend` in `recovery/execution_backend.py` is currently a fail-closed placeholder. It deliberately sends no hardware commands. The physical robot integration described below is therefore an implementation plan, not a claim that real-robot recovery is already operational.

---

## 2. Existing Components to Reuse

The implementation should extend the current recovery package instead of creating a separate recovery planner.

| Existing component | Responsibility in the real-robot integration |
|---|---|
| `recovery/models.py` | Common data structures for verifier results, evidence, recovery actions, context, plans and statuses. |
| `recovery/recovery_manager.py` | Combines verifier suggestions and produces a recovery plan. |
| `recovery/suggestion_normalizer.py` | Normalizes suggestions into a common representation. |
| `recovery/conflict_resolver.py` | Handles conflicting or duplicate recovery suggestions. |
| `recovery/risk_assessor.py` | Assesses the risk associated with proposed recovery actions. |
| `recovery/plan_generator.py` | Generates recovery plans from the accepted suggestions. |
| `recovery/plan_ranker.py` | Ranks candidate plans. |
| `recovery/plan_validator.py` | Checks plan structure, prerequisites, risk and configured constraints. |
| `recovery/execution_backend.py` | Defines the common execution interface and simulation/real backend classes. |
| `recovery/execution_coordinator.py` | Coordinates safety checks, execution, postcondition verification and conservative retry handling. |
| `recovery/mujoco_recovery_backend.py` | Provides a reference for how simulation recovery actions are dispatched. |
| `recovery/simulation_recovery_adapter.py` | Provides simulation-oriented verifier fixtures and state-based test logic. |
| `neem_interface/neem_interface.py` | Existing NEEM integration point for execution context and episodic-memory information. |
| `tracebot_audit_trail_generator/tracebot_audit_trail_generator.py` | Existing audit-trail generation component. |

The main design principle is to keep the recovery reasoning independent of the robot manufacturer and middleware.

The recovery manager should decide *what recovery is appropriate*. The hardware backend should decide *how that recovery is safely carried out on the particular robot*.

---

## 3. Proposed Real-Robot Architecture

The physical robot integration should follow this data flow:

```text
                 PHYSICAL ROBOT
                       |
          +------------+-------------+
          |                          |
      Robot sensors              Robot controller
          |                          |
          +------------+-------------+
                       |
              Robot state adapter
                       |
             Execution trace/context
                       |
                RobAuditor
                       |
          +------------+-------------+
          |            |             |
       Kinematics   Task logic   Object state
       verifier     verifier     verifier
          |            |             |
          +------------+-------------+
                       |
                Verifier results
                       |
                RecoveryManager
                       |
           Normalization and conflict
                  resolution
                       |
              Risk assessment
                       |
              Plan generation
                       |
                Plan validation
                       |
              VALIDATED plan
                       |
             ExecutionCoordinator
                       |
             Safety interlock check
                       |
           RealRobotRecoveryBackend
                       |
              Robot control API
                       |
              Physical robot
                       |
              Postcondition check
                       |
             Execution outcome
                       |
              NEEM / audit trail
```

The robot controller remains the authority for physical motion and its own safety functions. RobAuditor should not bypass the controller's collision checking, protective stops, speed limits or emergency-stop mechanisms.

---

## 4. Step 1: Connect the Robot's Existing Interfaces

Do not replace the existing robot controller or motion-planning system.

Instead, create an adapter that reads information from the robot's current interfaces and translates it into RobAuditor's data structures.

Depending on the robot installation, the adapter could use existing:

- ROS topics and services.
- ROS action servers.
- MoveIt planning and execution interfaces.
- Joint-state feedback.
- Gripper state interfaces.
- Camera and RGB-D streams.
- Force/torque measurements.
- Tactile sensors.
- Robot-controller status and safety-state interfaces.

These are integration possibilities, not dependencies already implemented by the recovery package. Select the interfaces actually available on the target robot.

### Information to collect

At minimum, the adapter should provide:

| Information | Example |
|---|---|
| Robot identity | UR10 or another configured robot |
| Active action | Grasping, transporting or inserting |
| Action identifier | Unique ID for the execution |
| Joint positions | Measured joint configuration |
| Joint velocities | Measured joint velocities |
| End-effector pose | Measured tool pose |
| Gripper state | Open, closed or fault |
| Object state | Object pose and availability |
| Task state | Completed and pending actions |
| Safety state | Normal, protective stop, emergency stop or fault |
| Sensor evidence | Camera, tactile and force/torque observations |

Use measured robot state rather than assuming that a requested motion was completed.

### Mapping into `RecoveryContext`

The existing `RecoveryContext` in `recovery/models.py` is the common input to the recovery manager.

A simplified example:

```python
from recovery.models import RecoveryContext

context = RecoveryContext(
    failed_action_id="execution_001_grasp",
    failed_action="grasp",

    participants={
        "robot": "ur10",
        "gripper": "robotiq",
        "object": "canister",
    },

    robot_state={
        "safe": True,
        "status": "stopped",
        "joint_positions": measured_joint_positions,
        "joint_velocities": measured_joint_velocities,
        "gripper_state": measured_gripper_state,
    },

    object_state={
        "name": "canister",
        "available": True,
        "pose": measured_object_pose,
    },

    world_state={
        "workspace": "table_1",
    },

    previous_actions=[
        "perceive_object",
        "move_to_pre_grasp",
    ],

    remaining_task=[
        "grasp",
        "lift",
        "transport",
        "place",
    ],

    safety_state={
        "safe": True,
        "emergency_stop": False,
        "collision_detected": False,
    },

    attempt_count=0,
    max_attempts=2,
)
```

This is an illustrative mapping. The actual sensor messages, robot state fields and participant identifiers must be adapted to the physical system.

Do not set `safe=True` merely because the robot controller has stopped moving. Obtain the actual safety state from the robot's authoritative interface.

---

## 5. Step 2: Implement Real Verifiers

The existing recovery package expects verifier results in the form of `VerifierResult` objects.

Each verifier should focus on a particular aspect of execution and independently report its decision, confidence, explanation, evidence and optional recovery suggestion.

The existing model serializes these as:

- `Ds`: verification decision.
- `Cf`: confidence.
- `Ed`: explanation and evidence.
- `Er`: recovery suggestion.

### Recommended initial verifier set

#### A. Kinematics verifier

Checks:

- Whether joint measurements are valid.
- Whether the robot is within configured joint limits.
- Whether measured velocities are within permitted limits.
- Whether the requested motion was feasible.
- Whether the robot controller reported a motion failure.

Possible recovery suggestions include:

- ESS: enter a safe state.
- GBB: return to a previous safe state.
- HIC: request human intervention.

#### B. Task-logic verifier

Checks:

- Whether the expected action transition occurred.
- Whether action prerequisites were satisfied.
- Whether the current task state is consistent with the execution history.
- Whether the next action is still valid.

Possible suggestions include GOB, GBB or INA.

#### C. Object-state verifier

Checks:

- Whether the target object is visible or otherwise detected.
- Whether its measured pose is plausible.
- Whether the gripper has established the expected contact.
- Whether the object moved as expected after grasping.
- Whether placement was successful.

Possible suggestions include GOB, INA or INO.

### Important distinction from the existing simulation adapter

`simulation_recovery_adapter.py` contains synthetic verifier fixtures and MuJoCo state-based checks. It is useful for testing the ensemble architecture, but it is not a physical sensor integration.

For a real robot, replace the simulation measurements with measurements from actual robot interfaces. Do not reuse the simulation confidence values as if they were calibrated real-world probabilities.

Each real verifier should identify the origin and timestamp of its evidence.

---

## 6. Step 3: Reuse the Existing Recovery Manager

Once the real verifiers have produced their results, pass them to the existing recovery manager.

Conceptually:

```python
from recovery.recovery_manager import RecoveryManager

recovery_manager = RecoveryManager(
    max_allowed_risk=0.3,
    selection_policy="consensus",
    candidate_count=1,
)

result = recovery_manager.process(
    verifier_results,
    context,
)

if result.plan is None:
    print("No executable recovery plan was generated.")
    print(result.message)
else:
    print("Recovery plan:", result.plan.recovery_id)
    print("Status:", result.plan.status)
    print("Actions:", result.plan.actions)
```

The example uses a conservative illustrative risk threshold. The threshold must be selected and justified for the particular robot, task and operating environment.

The manager already provides the central ensemble-recovery functionality:

1. Normalizing suggestions.
2. Resolving conflicts and duplicates.
3. Assessing risk.
4. Generating recovery candidates.
5. Ranking candidates.
6. Validating the selected plan.

The real-robot integration should preserve this pipeline.

### Recovery strategy mapping

The existing recovery models use these strategy codes:

| Strategy | Meaning | Possible physical-robot interpretation |
|---|---|---|
| ESS | Enter a safe state | Request a configured safe-stop or controlled-stop procedure. |
| HIC | Human intervention call | Stop autonomous recovery and request operator assistance. |
| GOB | Go back to a specific state | Return to a verified safe waypoint or repeat an action with updated parameters. |
| GBB | Go back before the failed action | Restore a previous state before attempting the failed task again. |
| INA | Introduce a new action | Add an action such as re-perception or re-planning. |
| INO | Introduce a new object | Request a replacement object or operator-provided object. |

A strategy code is not itself a robot command. It describes the recovery approach. The hardware backend must translate the resulting action into an approved operation.

---

## 7. Step 4: Implement the Physical Execution Backend

This is the main new implementation required for hardware integration.

The current `RealRobotRecoveryBackend` in `recovery/execution_backend.py` deliberately returns:

```text
Real-robot execution is not configured.
No hardware commands were sent.
```

It must remain fail-closed until a robot-specific implementation has been developed and tested.

The existing abstract interface is:

```python
class RecoveryExecutionBackend:

    def execute(
        self,
        plan: RecoveryPlan,
        context: RecoveryContext,
    ) -> RecoveryExecutionResult:
        ...
```

The new hardware implementation should preserve this interface.

### Recommended implementation structure

Create a robot-specific module, for example:

```text
recovery/
    execution_backend.py
    execution_coordinator.py
    real_robot_recovery_backend.py
    robot_action_adapter.py
    robot_state_adapter.py
```

Responsibilities:

**`robot_state_adapter.py`**

- Reads actual robot and sensor state.
- Provides a consistent state snapshot.
- Reports stale, unavailable or invalid measurements.
- Supplies the current safety state.

**`robot_action_adapter.py`**

- Connects to the existing robot controller.
- Maps approved recovery actions to controller operations.
- Handles command acceptance, completion and failure.
- Supports cancellation and timeouts.

**`real_robot_recovery_backend.py`**

- Implements `RecoveryExecutionBackend`.
- Checks that a plan is executable.
- Checks that every action is explicitly supported.
- Calls the robot action adapter.
- Records which actions completed.
- Returns a structured `RecoveryExecutionResult`.

### Action mapping

Use an explicit allowlist of supported action names.

For example, a robot integration might eventually support mappings such as:

| Recovery action | Robot-side operation |
|---|---|
| `stop_robot` | Request the configured safe-stop operation |
| `return_to_pre_grasp` | Execute a validated retreat to a saved pre-grasp state |
| `return_to_last_safe_waypoint` | Execute a validated return to a verified waypoint |
| `replan_pre_grasp` | Request a new motion plan from the robot's planning system |
| `request_human_intervention` | Stop autonomous recovery and notify an operator |

These are examples of mappings to implement. They are not registered hardware commands in the current repository.

Unknown actions must be rejected. Never translate an arbitrary action name or arbitrary verifier-provided parameter directly into a motion command.

### Backend execution requirements

Before executing any action, the backend should:

1. Confirm that the plan has `VALIDATED` status.
2. Confirm that the recovery ID and action list are valid.
3. Confirm that every action is in the configured allowlist.
4. Check that the robot is connected and responsive.
5. Read a fresh robot-state snapshot.
6. Check the actual safety state.
7. Check that the action prerequisites are satisfied.
8. Check that action parameters are within robot-specific limits.
9. Confirm that the robot controller accepts the command.
10. Monitor execution until success, failure or timeout.

If any required check fails, execution must be refused.

---

## 8. Step 5: Use the Existing Execution Coordinator

The repository already contains `RecoveryExecutionCoordinator` in `recovery/execution_coordinator.py`.

This is important because it separates planning from execution and provides conservative handling of failed execution attempts.

It supports:

- A safety checker.
- A postcondition checker.
- An optional retry authorizer.
- A maximum retry count.
- Structured execution results.

A simplified integration call looks like this:

```python
from recovery.execution_coordinator import (
    RecoveryExecutionCoordinator,
)

coordinator = RecoveryExecutionCoordinator()

outcome = coordinator.execute(
    plan=result.plan,
    context=context,
    backend=real_robot_backend,
    safety_checker=check_robot_safety,
    postcondition_checker=verify_recovery_postconditions,
    retry_authorizer=authorize_retry,
    max_retries=0,
)

print("Recovery status:", outcome.status)
print("Attempts:", outcome.attempts)
print("Message:", outcome.message)
```

The functions in this example must be implemented against the actual robot.

### Safety checker

The safety checker should obtain a fresh state from the robot's authoritative safety interface.

It should reject execution if:

- An emergency stop is active.
- A protective stop or controller fault is active.
- A collision or unsafe condition is detected.
- Required sensors are unavailable.
- State measurements are stale.
- The robot is not in an approved mode for the requested operation.

### Postcondition checker

The postcondition checker must determine whether the intended physical result actually occurred.

For example, after a recovery action intended to restore a pre-grasp position, it should check the measured robot pose against the configured tolerance.

After a grasp recovery, it should use appropriate evidence such as gripper state, tactile sensing, force measurements or visual confirmation.

A controller reporting that a command completed is not sufficient evidence that the task succeeded.

### Retry handling

The existing coordinator deliberately does not replay a partially executed plan.

This is important for physical robots. A failed motion may have changed the position of the arm or object even if the overall action did not complete.

Automatic retry should only be considered when:

- No action in the previous attempt completed.
- The current robot state has been re-read.
- The safety state has been checked again.
- A retry is explicitly authorized.

For initial hardware integration, keep `max_retries=0`.

---

## 9. Step 6: Connect Execution to NEEMs and Audit Trails

RobAuditor is intended to retain the relationship between robot experience, action narratives and the ontology.

The existing NEEM interface and audit-trail generator should be used as the integration points for recording real execution information.

For each recovery attempt, record:

- Original failed action ID.
- Recovery ID.
- Verifier identities.
- Verifier decisions and confidence values.
- Evidence references and timestamps.
- Accepted and rejected suggestions.
- Generated recovery plan.
- Plan validation result.
- Safety-check result.
- Commands requested from the robot controller.
- Actual execution results.
- Actions completed before a failure.
- Postcondition verification result.
- Human intervention, if required.

The recovery plan and its execution outcome must be stored separately.

**A generated recovery plan is not proof that the robot executed it.**

Likewise, a controller-reported command completion is not proof that the recovery objective was achieved. The physical postconditions must be checked and the result recorded.

The audit trail should make it possible to reconstruct what happened, why a recovery plan was selected, what the robot actually did and whether the task was ultimately recovered.

---

## 10. Step 7: Test in Stages Before Enabling Motion

Do not begin by allowing the recovery manager to send commands to a physical robot.

Use the following implementation stages.

### Stage 1 — Read-only integration

Connect the robot state adapter.

Verify that RobAuditor can receive:

- Joint states.
- Robot controller state.
- Gripper state.
- Object state.
- Sensor evidence.
- Current action information.

No recovery commands should be sent.

### Stage 2 — Real verifier integration

Replace synthetic simulation verifier inputs with actual sensor-based verifier results.

Test each verifier independently.

Check that the evidence and confidence values correspond to real measurements.

### Stage 3 — Recovery planning without execution

Connect the real verifier outputs to `RecoveryManager`.

Generate recovery plans and inspect:

- Accepted suggestions.
- Rejected suggestions.
- Conflicts.
- Risk scores.
- Validation reasons.
- Selected recovery actions.

Keep the physical robot execution disabled.

### Stage 4 — Hardware backend in dry-run mode

Implement the real backend so it can:

- Receive validated plans.
- Resolve action names.
- Validate parameters.
- Check robot state.
- Log the commands it would send.

Do not send motion commands.

Compare the proposed operations against what an experienced robot operator would approve.

### Stage 5 — Supervised, restricted execution

After the previous stages have passed, enable a small, explicitly approved set of recovery actions.

Use:

- A restricted workspace.
- Reduced, configured speed and force limits.
- An operator able to stop the robot.
- Controller-enforced safety functions.
- Independent state monitoring.
- Explicit operator authorization.

Start with low-risk recovery operations that have measurable outcomes.

### Stage 6 — Postcondition and audit validation

Verify that every execution attempt produces a reliable record.

Test:

- Successful recovery.
- Controller rejection.
- Communication loss.
- Stale sensor data.
- Partial execution.
- Postcondition failure.
- Emergency-stop activation.
- Human intervention.

Any ambiguous physical state should stop autonomous recovery and require operator assessment.

---

## 11. Suggested First Real-Robot Demonstration

A useful first demonstration would be a canister pick-and-place task, since the repository already contains a corresponding MuJoCo scenario.

The physical robot should perform the task using its existing controller and motion-planning system.

RobAuditor should initially observe and verify the following checkpoints:

1. Perceive the canister.
2. Move to the pre-grasp pose.
3. Grasp the canister.
4. Lift the canister.
5. Transport the canister.
6. Insert or place the canister.
7. Verify final placement.

### Example failure scenario: unsuccessful grasp

Suppose the robot closes its gripper but the object-state verifier cannot confirm that the canister was grasped.

The intended flow is:

```text
Robot attempts grasp
       |
       v
Object-state verifier detects failure
       |
       v
Kinematics verifier checks robot state
       |
       v
Task-logic verifier checks expected transition
       |
       v
RecoveryManager combines suggestions
       |
       v
Plan validation
       |
       v
Operator approval / safety checks
       |
       v
RealRobotRecoveryBackend
       |
       v
Robot executes approved recovery
       |
       v
Object-state verifier checks grasp again
       |
       v
Recovery outcome recorded
```

A possible recovery could be to retreat to a verified pre-grasp position and re-perceive the object before generating another grasp attempt.

The exact recovery must depend on the measured robot and object state. It should not be assumed that returning to the pre-grasp position is safe in every failed-grasp situation.

---

## 12. Implementation Checklist

### Existing components

- [x] Shared recovery data models.
- [x] Verifier result representation.
- [x] Ensemble recovery manager.
- [x] Suggestion normalization.
- [x] Conflict resolution.
- [x] Risk assessment.
- [x] Plan generation and ranking.
- [x] Plan validation.
- [x] Backend-neutral execution interface.
- [x] Execution coordinator.
- [x] MuJoCo simulation integration.
- [x] Fail-closed real-robot backend placeholder.

### New work required for physical deployment

- [ ] Identify the target robot and its existing control interfaces.
- [ ] Implement a real robot state adapter.
- [ ] Connect real sensor data to the verifier inputs.
- [ ] Implement and calibrate real kinematics verification.
- [ ] Implement real task-state verification.
- [ ] Implement real object-state verification.
- [ ] Define a robot-specific recovery action allowlist.
- [ ] Implement the hardware action adapter.
- [ ] Implement the physical execution backend.
- [ ] Add controller-specific safety and timeout handling.
- [ ] Implement measured postcondition checks.
- [ ] Add operator approval and intervention handling.
- [ ] Connect execution outcomes to NEEMs and audit trails.
- [ ] Complete dry-run and supervised hardware testing.

---

## 13. Important Safety and Design Principles

1. **Do not bypass the robot controller.** All physical motion must go through the robot's established control and safety interfaces.

2. **Do not treat simulation results as physical verification.** Synthetic verifier fixtures and MuJoCo measurements are useful for development but do not establish real-world sensor performance.

3. **Do not execute unvalidated plans.** The hardware backend must reject plans that are not explicitly validated.

4. **Do not trust stale state.** Safety and robot-state measurements must be current at the time of execution.

5. **Do not execute arbitrary verifier actions.** Use an explicit allowlist and robot-specific parameter validation.

6. **Do not assume that a command completed successfully just because it was accepted.** Verify the physical postconditions.

7. **Do not automatically replay partial plans.** A partially executed plan may have changed the physical environment.

8. **Fail closed.** When communication, sensing, controller state or safety status is uncertain, stop autonomous recovery and request human intervention.

---

## 14. Expected Result

After completing the integration, RobAuditor should operate as a recovery reasoning and monitoring layer connected to the robot's existing control system.

The physical robot remains responsible for low-level motion and hardware safety. RobAuditor provides the higher-level capability to combine independent verifier outputs, synthesize recovery plans, validate them, coordinate approved execution and determine whether recovery succeeded.

The central objective is not simply to make the robot repeat an action after failure. It is to make recovery decisions explainable, risk-aware, evidence-based and traceable.