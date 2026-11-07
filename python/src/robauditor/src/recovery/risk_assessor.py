"""Safety and risk assessment for recovery suggestions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .models import RecoverySuggestion, RecoveryContext


class RiskDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REQUIRES_HUMAN = "requires_human"


@dataclass
class RiskAssessment:
    """Result of assessing a recovery suggestion."""

    decision: RiskDecision
    risk_score: float
    reasons: List[str] = field(default_factory=list)


def _is_explicitly_unsafe(state: Any) -> List[str]:
    """Identify explicit unsafe indicators in a state dictionary."""

    if not isinstance(state, dict):
        return []

    reasons = []

    unsafe_flags = {
        "safe": False,
        "collision_detected": True,
        "emergency_stop": True,
        "human_intervention_required": True,
    }

    for key, unsafe_value in unsafe_flags.items():
        if key in state and state[key] == unsafe_value:
            reasons.append(f"Unsafe state indicator: {key}={unsafe_value}")

    status = str(state.get("status", "")).strip().lower()

    if status in {"unsafe", "collision", "emergency_stop", "fault"}:
        reasons.append(f"Unsafe state status: {status}")

    return reasons


def assess_suggestion(
    suggestion: RecoverySuggestion,
    context: Optional[RecoveryContext] = None,
    max_allowed_risk: float = 0.7,
    safety_constraints: Optional[Dict[str, Any]] = None,
) -> RiskAssessment:
    """Assess whether a recovery suggestion is safe to execute.

    Risk scores are interpreted on a scale from 0.0 to 1.0.

    Explicit safety violations always override the numerical risk score.
    """

    if not 0.0 <= max_allowed_risk <= 1.0:
        raise ValueError("max_allowed_risk must be between 0 and 1.")

    safety_constraints = safety_constraints or {}
    reasons = []

    # Check whether the recovery attempt limit has been reached.
    if context is not None:
        if context.attempt_count >= context.max_attempts:
            return RiskAssessment(
                decision=RiskDecision.REQUIRES_HUMAN,
                risk_score=1.0,
                reasons=["Maximum recovery attempts reached."],
            )

        # Check explicit robot safety indicators.
        reasons.extend(_is_explicitly_unsafe(context.safety_state))
        reasons.extend(_is_explicitly_unsafe(context.robot_state))

        if reasons:
            return RiskAssessment(
                decision=RiskDecision.REQUIRES_HUMAN,
                risk_score=1.0,
                reasons=reasons,
            )

    # Check whether the action is explicitly forbidden.
    forbidden_actions = {
        name.strip().lower()
        for name in safety_constraints.get("forbidden_actions", [])
    }

    action_names = [
        action.name.strip().lower()
        for action in (suggestion.actions or [])
    ]

    forbidden = forbidden_actions.intersection(action_names)

    if forbidden:
        return RiskAssessment(
            decision=RiskDecision.REJECT,
            risk_score=1.0,
            reasons=[
                f"Action '{name}' is explicitly forbidden."
                for name in sorted(forbidden)
            ],
        )

    # Unknown risk must not be interpreted as zero risk.
    unknown_risk_actions = [
        action.name
        for action in (suggestion.actions or [])
        if action.estimated_risk is None
    ]

    if unknown_risk_actions:
        return RiskAssessment(
            decision=RiskDecision.REQUIRES_HUMAN,
            risk_score=1.0,
            reasons=[
                f"Action '{name}' has no estimated risk."
                for name in unknown_risk_actions
            ],
        )

    action_risks = [
        action.estimated_risk
        for action in (suggestion.actions or [])
    ]

    risk_score = max(action_risks, default=0.0)

    # Apply an optional maximum risk defined by safety constraints.
    constraint_limit = safety_constraints.get("max_allowed_risk")

    if constraint_limit is not None:
        if not 0.0 <= constraint_limit <= 1.0:
            raise ValueError(
                "Safety constraint max_allowed_risk must be between 0 and 1."
            )

        max_allowed_risk = min(max_allowed_risk, constraint_limit)

    if risk_score > max_allowed_risk:
        return RiskAssessment(
            decision=RiskDecision.REJECT,
            risk_score=risk_score,
            reasons=[
                f"Risk score {risk_score:.2f} exceeds "
                f"the allowed threshold {max_allowed_risk:.2f}."
            ],
        )

    return RiskAssessment(
        decision=RiskDecision.ACCEPT,
        risk_score=risk_score,
        reasons=["No configured safety violations detected."],
    )