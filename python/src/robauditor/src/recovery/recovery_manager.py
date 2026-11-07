"""Orchestrate the recovery planning pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import random

from .models import (
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryPlan,
    RecoveryStrategy,
    VerifierResult,
)

from .suggestion_normalizer import normalize_suggestions
from .risk_assessor import assess_suggestion
from .plan_generator import generate_recovery_plan, generate_candidate_recovery_plans
from .plan_ranker import rank_plans
from .plan_validator import validate_plan


def _explicitly_unsafe(context: RecoveryContext) -> bool:
    """Return True if robot or safety state contains an unsafe indicator."""

    unsafe_flags = {
        "safe": False,
        "collision_detected": True,
        "emergency_stop": True,
        "human_intervention_required": True,
    }

    unsafe_statuses = {
        "unsafe",
        "collision",
        "emergency_stop",
        "fault",
    }

    for state in (context.robot_state, context.safety_state):
        if not isinstance(state, dict):
            continue

        for key, unsafe_value in unsafe_flags.items():
            if state.get(key) == unsafe_value:
                return True

        status = str(state.get("status", "")).strip().lower()

        if status in unsafe_statuses:
            return True

    return False


@dataclass
class RecoveryManagerResult:
    """Outcome returned by the recovery manager."""

    status: PlanStatus
    plan: Optional[RecoveryPlan] = None
    message: str = ""
    rejected_reasons: List[str] = field(default_factory=list)
    verifier_outputs: List[Dict[str, Any]] = field(default_factory=list)
    candidate_plans: List[RecoveryPlan] = field(default_factory=list)
    selection_probabilities: Dict[str, float] = field(default_factory=dict)


class RecoveryManager:
    """Coordinate verifier results into a validated recovery plan."""

    def __init__(
        self,
        max_allowed_risk: float = 0.7,
        safety_constraints: Optional[Dict[str, Any]] = None,
        ess_action: Optional[RecoveryAction] = None,
        selection_policy: str = "confidence",
        candidate_count: int = 1,
        sampling_temperature: float = 0.15,
        random_seed: Optional[int] = None,
    ):
        if selection_policy not in {"confidence", "consensus"}:
            raise ValueError(
                "selection_policy must be 'confidence' or 'consensus'."
            )
        self.max_allowed_risk = max_allowed_risk
        self.safety_constraints = safety_constraints or {}
        self.ess_action = ess_action
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least 1.")
        if sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be greater than zero.")
        self.selection_policy = selection_policy
        self.candidate_count = candidate_count
        self.sampling_temperature = sampling_temperature
        self.random_seed = random_seed

    def process(
        self,
        verifier_results: List[VerifierResult],
        context: RecoveryContext,
    ) -> RecoveryManagerResult:
        """Run the complete recovery planning pipeline."""

        # Preserve every original verifier quadruplet, including PASS results
        # and results that do not contain a recovery suggestion.
        verifier_outputs = [
            result.to_quadruplet() for result in verifier_results
        ]

        # 0. Explicitly unsafe states require the dedicated ESS path.
        if _explicitly_unsafe(context):
            ess_result = self._build_ess_plan(context)
            ess_result.verifier_outputs = verifier_outputs
            if ess_result.plan is not None:
                ess_result.plan.metadata["verifier_outputs"] = verifier_outputs
            return ess_result

        # 1. Normalize suggestions from unsuccessful verifiers.
        suggestions = normalize_suggestions(verifier_results)

        if not suggestions:
            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                message="No recovery suggestions were provided.",
                verifier_outputs=verifier_outputs,
            )

        # 1a. ESS is a safety-priority strategy, not an ordinary action to
        # merge with GOB/GBB/INA/INO proposals. If any verifier requests ESS,
        # build and validate a dedicated ESS-only plan. Preserve every
        # quadruplet, but defer all non-ESS suggestions until the robot has
        # been confirmed safe.
        ess_suggestions = [
            suggestion for suggestion in suggestions
            if (
                suggestion.strategy == RecoveryStrategy.ESS
                or any(
                    action.name.strip().lower() in {
                        "stop_robot", "emergency_stop", "enter_safe_state"
                    }
                    for action in (suggestion.actions or [])
                )
            )
        ]
        if ess_suggestions:
            ess_result = self._build_ess_plan(context)
            ess_result.verifier_outputs = verifier_outputs
            deferred = [
                suggestion.verifier_id for suggestion in suggestions
                if suggestion not in ess_suggestions
            ]
            ess_result.rejected_reasons.extend(
                f"Deferred while ESS safe-state recovery has priority: {verifier_id}."
                for verifier_id in deferred
            )
            if ess_result.plan is not None:
                ess_result.plan.metadata["verifier_outputs"] = verifier_outputs
                ess_result.plan.metadata["ess_trigger_verifiers"] = [
                    suggestion.verifier_id for suggestion in ess_suggestions
                ]
                ess_result.plan.metadata["deferred_verifiers"] = deferred
                ess_result.plan.metadata["ess_priority"] = True
            if ess_result.status != PlanStatus.VALIDATED:
                ess_result.status = PlanStatus.HUMAN_INTERVENTION
                if ess_result.plan is not None:
                    ess_result.plan.status = PlanStatus.HUMAN_INTERVENTION
                ess_result.message = (
                    "An ESS recommendation was received, but the dedicated "
                    "ESS plan could not be validated. Human intervention is required. "
                    + ess_result.message
                )
            return ess_result

        # 2. Assess the risk of every suggestion.
        risk_assessments = {}

        for suggestion in suggestions:
            assessment = assess_suggestion(
                suggestion,
                context=context,
                max_allowed_risk=self.max_allowed_risk,
                safety_constraints=self.safety_constraints,
            )

            risk_assessments[suggestion.verifier_id] = assessment

        # 3. Generate one legacy plan or a bounded set of alternatives.
        if self.candidate_count == 1:
            candidates = [
                generate_recovery_plan(
                    suggestions,
                    risk_assessments,
                    context,
                    safety_constraints=self.safety_constraints,
                    selection_policy=self.selection_policy,
                )
            ]
        else:
            candidates = generate_candidate_recovery_plans(
                suggestions,
                risk_assessments,
                context,
                safety_constraints=self.safety_constraints,
                selection_policy=self.selection_policy,
                max_candidates=self.candidate_count,
            )

        for candidate in candidates:
            candidate.metadata["verifier_outputs"] = verifier_outputs

        # Validate every candidate before allowing it to be sampled.
        ranked_candidates = rank_plans(candidates)
        validated_candidates = []
        validation_failures = []
        for candidate in ranked_candidates:
            validation = validate_plan(
                candidate,
                context,
                safety_constraints=self.safety_constraints,
            )
            if validation.valid:
                candidate.status = PlanStatus.VALIDATED
                validated_candidates.append(candidate)
            else:
                candidate.status = PlanStatus.HUMAN_INTERVENTION
                validation_failures.extend(validation.reasons)

        if not validated_candidates:
            fallback = candidates[0] if candidates else None
            message = (
                "No acceptable recovery actions could be generated."
                if fallback is not None and not fallback.actions
                else "No candidate recovery plan passed validation."
            )
            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                plan=fallback,
                message=message,
                rejected_reasons=(
                    [str(reason) for reason in fallback.rejected_suggestions.values()]
                    if fallback is not None else []
                ) + validation_failures,
                verifier_outputs=verifier_outputs,
                candidate_plans=candidates,
            )

        probabilities = self._candidate_probabilities(validated_candidates)
        selected = self._sample_candidate(validated_candidates, probabilities)
        probability_map = {
            candidate.recovery_id: probability
            for candidate, probability in zip(validated_candidates, probabilities)
        }
        for candidate in validated_candidates:
            candidate.metadata["selection_probability"] = probability_map[
                candidate.recovery_id
            ]
            candidate.metadata["sampling_temperature"] = self.sampling_temperature
            candidate.metadata["sampling_method"] = (
                "softmax(confidence, risk, verifier_support)"
                if len(validated_candidates) > 1
                else "single_validated_candidate"
            )

        return RecoveryManagerResult(
            status=PlanStatus.VALIDATED,
            plan=selected,
            message=(
                f"Selected 1 of {len(validated_candidates)} validated "
                "candidate recovery plan(s) using risk-aware probabilistic sampling."
                if len(validated_candidates) > 1
                else "A recovery plan passed validation."
            ),
            verifier_outputs=verifier_outputs,
            candidate_plans=validated_candidates,
            selection_probabilities=probability_map,
        )

    def _candidate_probabilities(self, candidates: List[RecoveryPlan]) -> List[float]:
        """Softmax weights reward confidence, verifier support and lower risk."""
        utilities = []
        for candidate in candidates:
            confidence = float(candidate.metadata.get("confidence_score", 0.0))
            risk = float(candidate.metadata.get("risk_score", 1.0))
            support_map = candidate.metadata.get("verifier_support", {})
            support_score = (
                sum(float(value) for value in support_map.values())
                / max(len(support_map), 1)
                / max(len(support_map), 1)
            )
            utility = (
                0.45 * min(max(confidence, 0.0), 1.0)
                + 0.35 * (1.0 - min(max(risk, 0.0), 1.0))
                + 0.20 * min(max(support_score, 0.0), 1.0)
            )
            utilities.append(utility)

        peak = max(utilities)
        weights = [
            math.exp((utility - peak) / self.sampling_temperature)
            for utility in utilities
        ]
        total = sum(weights)
        return [weight / total for weight in weights]

    def _sample_candidate(
        self,
        candidates: List[RecoveryPlan],
        probabilities: List[float],
    ) -> RecoveryPlan:
        """Sample one validated plan; fixed seeds make experiments repeatable."""
        rng = random.Random(self.random_seed)
        return rng.choices(candidates, weights=probabilities, k=1)[0]

    def _build_ess_plan(
        self,
        context: RecoveryContext,
    ) -> RecoveryManagerResult:
        """Build an ESS-only plan when the robot is explicitly unsafe."""

        if self.ess_action is None:
            reason = (
                "Robot state is unsafe and no ESS action is configured. "
                "Human intervention is required."
            )

            plan = RecoveryPlan(
                recovery_id=f"ess_{context.failed_action_id}",
                failed_action_id=context.failed_action_id,
                actions=[],
                strategies=[],
                status=PlanStatus.HUMAN_INTERVENTION,
                explanation=reason,
                max_attempts=context.max_attempts,
                metadata={
                    "requires_safe_state_confirmation": True,
                    "ess_action_configured": False,
                },
            )

            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                plan=plan,
                message=reason,
                rejected_reasons=[reason],
            )
    

        action = self.ess_action

        # Reject an ESS action explicitly forbidden by safety constraints.
        forbidden_actions = {
            name.strip().lower()
            for name in self.safety_constraints.get("forbidden_actions", [])
        }

        if action.name.strip().lower() in forbidden_actions:
            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                message=f"Configured ESS action '{action.name}' is forbidden.",
            )

        # Unknown action risk must not be treated as safe.
        if action.estimated_risk is None:
            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                message="Configured ESS action has unknown risk.",
            )

        # Validate the configured risk limit.
        constraint_limit = self.safety_constraints.get("max_allowed_risk")

        if constraint_limit is not None:
            if (
                not isinstance(constraint_limit, (int, float))
                or isinstance(constraint_limit, bool)
                or not 0.0 <= constraint_limit <= 1.0
            ):
                return RecoveryManagerResult(
                    status=PlanStatus.HUMAN_INTERVENTION,
                    message="Configured ESS risk limit is invalid.",
                )

        if (
            not isinstance(action.estimated_risk, (int, float))
            or isinstance(action.estimated_risk, bool)
            or not 0.0 <= action.estimated_risk <= 1.0
        ):
            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                message="Configured ESS action has an invalid risk value.",
            )

        allowed_risk = min(
            self.max_allowed_risk,
            constraint_limit
            if constraint_limit is not None
            else self.max_allowed_risk,
        )

        if action.estimated_risk > allowed_risk:
            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                message="Configured ESS action exceeds the allowed risk.",
            )

        # Construct a dedicated ESS-only plan.
        plan = RecoveryPlan(
            recovery_id=f"ess_{context.failed_action_id}",
            failed_action_id=context.failed_action_id,
            actions=[action],
            strategies=[RecoveryStrategy.ESS],
            status=PlanStatus.PROPOSED,
            explanation="Dedicated safe-state recovery plan.",
            max_attempts=context.max_attempts,
            metadata={
                "requires_safe_state_confirmation": True,
            },
        )

        # ESS validation must explicitly permit the unsafe starting state,
        # without weakening validation for ordinary recovery plans.
        from .plan_validator import validate_ess_plan

        validation = validate_ess_plan(
            plan,
            context,
            safety_constraints=self.safety_constraints,
            configured_action=self.ess_action,
        )

        if not validation.valid:
            plan.status = PlanStatus.HUMAN_INTERVENTION

            return RecoveryManagerResult(
                status=PlanStatus.HUMAN_INTERVENTION,
                plan=plan,
                message="ESS plan failed validation.",
                rejected_reasons=validation.reasons,
            )

        plan.status = PlanStatus.VALIDATED

        return RecoveryManagerResult(
            status=PlanStatus.VALIDATED,
            plan=plan,
            message=(
                "ESS plan validated. Execution and safe-state confirmation "
                "are still required."
            ),
        )