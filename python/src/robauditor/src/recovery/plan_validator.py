"""Validate candidate recovery plans before execution.

Action-specific parameter requirements and action incompatibilities are supplied
through ``safety_constraints`` so the validator remains reusable across robots.
"""
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

from .models import RecoveryPlan, RecoveryContext, RecoveryStrategy, PlanStatus


@dataclass
class ValidationResult:
    """Result of validating a recovery plan."""
    valid: bool
    reasons: List[str] = field(default_factory=list)


def _get_state_value(path: str, context: RecoveryContext) -> Any:
    """Resolve a prerequisite path against the recovery context."""
    if not isinstance(path, str) or not path.strip():
        return None

    parts = path.split(".")
    state_sources = {
        "robot": context.robot_state or {},
        "object": context.object_state or {},
        "world": context.world_state or {},
        "safety": context.safety_state or {},
    }

    if parts[0] in state_sources:
        value = state_sources[parts[0]]
        for part in parts[1:]:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    for state in state_sources.values():
        if isinstance(state, dict) and path in state:
            return state[path]
    return None


def _check_prerequisites(action, context: RecoveryContext) -> List[str]:
    """Check whether all declared action prerequisites are satisfied."""
    failures = []
    for prerequisite in action.prerequisites or []:
        value = _get_state_value(prerequisite, context)
        # A prerequisite is satisfied by an existing, non-None value.
        # This permits meaningful values such as False and 0 to be represented.
        if value is None:
            failures.append(
                f"Action '{action.name}' requires '{prerequisite}', "
                "but the state value is missing."
            )
        elif isinstance(value, bool) and not value:
            failures.append(
                f"Action '{action.name}' requires '{prerequisite}', "
                "but it is false."
            )
    return failures


def _is_unsafe_state(state: Any) -> List[str]:
    failures = []
    if not isinstance(state, dict):
        failures.append("State snapshot is not a dictionary.")
        return failures

    unsafe_flags = {
        "safe": False,
        "collision_detected": True,
        "emergency_stop": True,
        "human_intervention_required": True,
    }
    for key, unsafe_value in unsafe_flags.items():
        if state.get(key) == unsafe_value:
            failures.append(f"{key}={unsafe_value}")

    status = str(state.get("status", "")).strip().lower()
    if status in {"unsafe", "collision", "emergency_stop", "fault"}:
        failures.append(f"status={status}")
    return failures


def _check_safety_state(context: RecoveryContext) -> List[str]:
    """Check for explicitly unsafe robot or safety states."""
    failures = []
    for state_name, state in (
        ("robot", context.robot_state or {}),
        ("safety", context.safety_state or {}),
    ):
        for issue in _is_unsafe_state(state):
            failures.append(f"Unsafe {state_name} state: {issue}.")
    return failures


def _valid_risk(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _check_action_parameters(action, constraints: Dict[str, Any]) -> List[str]:
    """Validate required keys and optional type schemas for one action.

    Configure required keys as:
      {"required_action_parameters": {"move_to_waypoint": ["waypoint"]}}

    Optional expected types can be configured as:
      {"action_parameter_types": {"move_to_waypoint":
          {"waypoint": str, "speed": (int, float)}}}
    """
    failures = []
    parameters = action.parameters
    if not isinstance(parameters, dict):
        return [f"Action '{action.name}' parameters must be a dictionary."]

    required_map = constraints.get("required_action_parameters", {}) or {}
    required = required_map.get(action.name, required_map.get(action.name.lower(), []))
    for key in required:
        if key not in parameters or parameters[key] is None:
            failures.append(
                f"Action '{action.name}' is missing required parameter '{key}'."
            )

    type_map = constraints.get("action_parameter_types", {}) or {}
    expected = type_map.get(action.name, type_map.get(action.name.lower(), {}))
    for key, expected_type in expected.items():
        if key in parameters and not isinstance(parameters[key], expected_type):
            failures.append(
                f"Action '{action.name}' parameter '{key}' has type "
                f"{type(parameters[key]).__name__}; expected "
                f"{getattr(expected_type, '__name__', expected_type)}."
            )
    return failures


def _check_action_pair_compatibility(actions, constraints: Dict[str, Any]) -> List[str]:
    """Reject configured mutually exclusive action pairs in one plan.

    Configure pairs as:
      {"incompatible_action_pairs": [
          ["hold_robot", "direct_transfer"],
          ["request_human_intervention", "replan_approach"]
      ]}
    """
    failures = []
    configured_pairs = constraints.get("incompatible_action_pairs", []) or []
    incompatible = {
        frozenset(str(name).strip().lower() for name in pair)
        for pair in configured_pairs
        if isinstance(pair, (list, tuple, set)) and len(pair) == 2
    }
    names = [str(action.name).strip().lower() for action in actions]
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if frozenset((first, second)) in incompatible:
                failures.append(
                    f"Actions '{first}' and '{second}' are configured as incompatible."
                )
    return failures


def _check_attempt_limit(plan: RecoveryPlan, context: RecoveryContext) -> List[str]:
    if context.attempt_count < 0 or plan.max_attempts < 0 or context.max_attempts < 0:
        return ["Recovery attempt counters and limits must be non-negative."]
    attempt_limit = min(context.max_attempts, plan.max_attempts)
    if context.attempt_count >= attempt_limit:
        return [
            f"Recovery attempt limit reached: "
            f"{context.attempt_count}/{attempt_limit}."
        ]
    return []


def _check_common_plan(
    plan: RecoveryPlan,
    context: RecoveryContext,
    safety_constraints: Dict[str, Any],
    *,
    check_safety_state: bool,
) -> List[str]:
    failures = []
    if plan.status not in {PlanStatus.PROPOSED, PlanStatus.VALIDATED}:
        status = plan.status.value if hasattr(plan.status, "value") else str(plan.status)
        failures.append(f"Plan status '{status}' is not eligible for execution.")

    if not plan.actions:
        failures.append("Recovery plan contains no actions.")

    failures.extend(_check_attempt_limit(plan, context))
    if check_safety_state:
        failures.extend(_check_safety_state(context))

    forbidden_actions = {
        str(name).strip().lower()
        for name in safety_constraints.get("forbidden_actions", [])
    }
    max_allowed_risk = safety_constraints.get("max_allowed_risk", 0.7)
    if not _valid_risk(max_allowed_risk):
        failures.append("Configured maximum risk must be a finite number from 0 to 1.")
        max_allowed_risk = -1.0

    for action in plan.actions:
        action_name = str(action.name).strip().lower()
        if action_name in forbidden_actions:
            failures.append(f"Action '{action.name}' is explicitly forbidden.")

        failures.extend(_check_action_parameters(action, safety_constraints))
        failures.extend(_check_prerequisites(action, context))

        risk = getattr(action, "estimated_risk", None)
        if not _valid_risk(risk):
            failures.append(
                f"Action '{action.name}' has unknown or invalid risk."
            )
        elif risk > max_allowed_risk:
            failures.append(
                f"Action '{action.name}' risk {risk:.3f} exceeds "
                f"the configured maximum {max_allowed_risk:.3f}."
            )

    failures.extend(_check_action_pair_compatibility(plan.actions, safety_constraints))
    return failures


def validate_plan(
    plan: RecoveryPlan,
    context: RecoveryContext,
    safety_constraints: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """Validate an ordinary recovery plan against the current context."""
    failures = _check_common_plan(
        plan, context, safety_constraints or {}, check_safety_state=True
    )
    return ValidationResult(
        valid=not failures,
        reasons=failures or ["Recovery plan passed all configured checks."],
    )


def validate_ess_plan(
    plan: RecoveryPlan,
    context: RecoveryContext,
    safety_constraints: Optional[Dict[str, Any]] = None,
    configured_action=None,
) -> ValidationResult:
    """Validate a dedicated ESS-only plan.

    An unsafe initial state is permitted only for the explicitly configured
    ESS action. The action itself must still pass risk and parameter checks.
    """
    failures = []
    constraints = safety_constraints or {}

    if plan.status not in {PlanStatus.PROPOSED, PlanStatus.VALIDATED}:
        status = plan.status.value if hasattr(plan.status, "value") else str(plan.status)
        failures.append(f"Plan status '{status}' is not eligible for execution.")

    if plan.strategies != [RecoveryStrategy.ESS]:
        failures.append(
            "ESS validation requires ESS to be the only recovery strategy."
        )

    if len(plan.actions) != 1:
        failures.append("ESS validation requires exactly one recovery action.")
    else:
        action = plan.actions[0]
        if configured_action is None:
            failures.append("No explicitly configured ESS action was provided.")
        elif action != configured_action:
            failures.append("ESS plan action does not match the configured ESS action.")

        forbidden_actions = {
            str(name).strip().lower()
            for name in constraints.get("forbidden_actions", [])
        }
        if str(action.name).strip().lower() in forbidden_actions:
            failures.append(f"ESS action '{action.name}' is explicitly forbidden.")

        failures.extend(_check_action_parameters(action, constraints))
        failures.extend(_check_prerequisites(action, context))

        risk = getattr(action, "estimated_risk", None)
        limit = constraints.get("max_allowed_risk", 0.7)
        if not _valid_risk(limit):
            failures.append("Configured maximum risk must be a finite number from 0 to 1.")
        elif not _valid_risk(risk):
            failures.append(f"ESS action '{action.name}' has unknown or invalid risk.")
        elif risk > limit:
            failures.append(
                f"ESS action risk {risk:.3f} exceeds the configured maximum {limit:.3f}."
            )

    failures.extend(_check_attempt_limit(plan, context))
    # Deliberately skip ordinary unsafe-state checks for the initial ESS state.
    return ValidationResult(
        valid=not failures,
        reasons=failures or ["Dedicated ESS plan passed all configured checks."],
    )
