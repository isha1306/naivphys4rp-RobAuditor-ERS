"""Recovery-plan synthesis with selectable confidence and consensus policies.

The default remains confidence-first for backward compatibility. The optional
"consensus" policy first applies risk filtering, then ranks conflicting
suggestions using verifier support, mean confidence, and risk. It preserves
compatible actions and merges exact duplicate actions.

This module does not treat a verifier's confidence as proof of correctness.
The consensus policy is a deterministic synthesis heuristic, not a learned
probabilistic model.
"""

from collections import defaultdict
import hashlib
from typing import Dict, List, Optional

from uuid import uuid4

from .models import (
    PlanStatus,
    RecoveryContext,
    RecoveryPlan,
    RecoveryStrategy,
    RecoverySuggestion,
)
from .conflict_resolver import ConflictType, classify_pair
from .risk_assessor import RiskAssessment, RiskDecision


def _action_signature(action):
    """Create a comparable signature to identify duplicate actions."""
    return (
        action.name.strip().lower(),
        repr(sorted((action.parameters or {}).items())),
    )


def _confidence(suggestion):
    value = suggestion.confidence
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _suggestion_risk(suggestion, risk_assessments):
    assessment = risk_assessments.get(suggestion.verifier_id)
    if assessment is None:
        return 1.0
    try:
        return min(max(float(assessment.risk_score), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _pairwise_conflict_count(candidate, suggestions):
    count = 0
    for other in suggestions:
        if other is candidate:
            continue
        if classify_pair(candidate, other).classification == ConflictType.CONFLICTING:
            count += 1
    return count


def _consensus_priority(suggestion, accepted, risk_assessments):
    """Higher support, then confidence; lower risk and conflict count win."""

    # Support counts distinct verifier suggestions with the same action
    # signature. A verifier contributes at most once to a candidate signature.
    signatures = {
        _action_signature(action)
        for action in (suggestion.actions or [])
    }
    supporting_verifiers = set()
    for other in accepted:
        other_signatures = {
            _action_signature(action)
            for action in (other.actions or [])
        }
        if signatures.intersection(other_signatures):
            supporting_verifiers.add(other.verifier_id)

    support = len(supporting_verifiers)
    confidence = _confidence(suggestion)
    risk = _suggestion_risk(suggestion, risk_assessments)
    conflicts = _pairwise_conflict_count(suggestion, accepted)

    # Sorting is descending; negate values that should be minimized.
    return (support, confidence, -risk, -conflicts, suggestion.verifier_id)


def generate_recovery_plan(
    suggestions: List[RecoverySuggestion],
    risk_assessments: Dict[str, RiskAssessment],
    context: RecoveryContext,
    safety_constraints: Optional[List[str]] = None,
    selection_policy: str = "confidence",
    priority_verifier_ids: Optional[List[str]] = None,
) -> RecoveryPlan:
    """Combine safe, compatible suggestions into one recovery plan.

    selection_policy:
      - "confidence": original confidence-first conflict selection.
      - "consensus": rank proposals by matching-action verifier support,
        confidence, lower assessed risk, then lower conflict count.

    Safety filtering always precedes either selection policy.
    """

    if selection_policy not in {"confidence", "consensus"}:
        raise ValueError(
            "selection_policy must be 'confidence' or 'consensus'."
        )

    accepted = []
    rejected = {}

    # 1. Safety filtering is mandatory for every policy.
    for suggestion in suggestions:
        assessment = risk_assessments.get(suggestion.verifier_id)

        if assessment is None:
            rejected[suggestion.verifier_id] = (
                "No risk assessment was provided."
            )
        elif assessment.decision != RiskDecision.ACCEPT:
            rejected[suggestion.verifier_id] = (
                f"Risk assessment: {assessment.decision.value}. "
                + "; ".join(assessment.reasons)
            )
        else:
            accepted.append(suggestion)

    # 2. Choose deterministic priority policy. A candidate pivot can
    # override the first position while retaining the selected policy after it.
    priority_order = {
        verifier_id: index
        for index, verifier_id in enumerate(priority_verifier_ids or [])
    }

    if selection_policy == "confidence":
        accepted.sort(
            key=lambda item: (
                priority_order.get(item.verifier_id, len(priority_order)),
                -_confidence(item),
                item.verifier_id,
            )
        )
    else:
        accepted_for_scoring = list(accepted)
        accepted.sort(
            key=lambda item: (
                priority_order.get(item.verifier_id, len(priority_order)),
                tuple(
                    -value for value in _consensus_priority(
                        item, accepted_for_scoring, risk_assessments
                    )[:-1]
                ),
                item.verifier_id,
            )
        )

    # 3. Select suggestions while excluding incompatible proposals.
    selected = []
    for candidate in accepted:
        conflicting_with = [
            existing.verifier_id
            for existing in selected
            if classify_pair(candidate, existing).classification
            == ConflictType.CONFLICTING
        ]

        if conflicting_with:
            rejected[candidate.verifier_id] = (
                "Conflicts with higher-priority suggestion(s): "
                + ", ".join(conflicting_with)
            )
        else:
            selected.append(candidate)

    # 4. Merge selected actions, retaining compatible actions and
    # deduplicating exact action+parameter matches.
    combined_actions = []
    seen_actions = set()
    for suggestion in selected:
        for action in suggestion.actions or []:
            signature = _action_signature(action)
            if signature not in seen_actions:
                combined_actions.append(action)
                seen_actions.add(signature)

    strategies = list(dict.fromkeys(
        suggestion.strategy for suggestion in selected
    ))

    if not combined_actions:
        status = PlanStatus.HUMAN_INTERVENTION
        explanation = (
            "No safe recovery actions could be generated. "
            "Human intervention is required."
        )
    else:
        status = PlanStatus.PROPOSED
        explanation = (
            f"Generated a recovery plan using {len(selected)} "
            f"accepted verifier suggestion(s), containing "
            f"{len(combined_actions)} unique action(s), "
            f"with selection policy '{selection_policy}'."
        )

    selected_assessments = [
        risk_assessments[s.verifier_id] for s in selected
    ]

    return RecoveryPlan(
        recovery_id=f"recovery-{uuid4().hex[:8]}",
        failed_action_id=context.failed_action_id,
        actions=combined_actions,
        strategies=strategies,
        status=status,
        explanation=explanation,
        accepted_suggestions=[s.verifier_id for s in selected],
        rejected_suggestions=rejected,
        safety_constraints=safety_constraints or [],
        max_attempts=context.max_attempts,
        metadata={
            "selection_policy": selection_policy,
            "attempt_count": context.attempt_count,
            "selected_verifier_count": len(selected),
            "rejected_verifier_count": len(rejected),
            "confidence_score": (
                sum(_confidence(s) for s in selected) / len(selected)
                if selected else 0.0
            ),
            "risk_score": max(
                (assessment.risk_score for assessment in selected_assessments),
                default=1.0,
            ),
            "verifier_support": {
                s.verifier_id: len({
                    other.verifier_id
                    for other in accepted
                    if {
                        _action_signature(a) for a in (s.actions or [])
                    }.intersection({
                        _action_signature(a) for a in (other.actions or [])
                    })
                })
                for s in selected
            },
            "verifier_outputs": [
                {
                    "verifier_id": s.verifier_id,
                    "Ds": {
                        "decision": s.decision,
                        "label": (
                            "UNCERTAIN" if s.decision is None
                            else "FAIL" if s.decision is False
                            else "PASS"
                        ),
                    },
                    "Cf": {"confidence": s.confidence},
                    "Ed": {
                        "text": s.explanation,
                        "evidence": dict(s.evidence or {}),
                        "visual_evidence": [
                            {
                                "type": item.evidence_type,
                                "uri": item.uri,
                                "description": item.description,
                                "metadata": dict(item.metadata or {}),
                            }
                            for item in (s.visual_evidence or [])
                        ],
                    },
                    "Er": {
                        "strategy": (
                            s.strategy.value
                            if hasattr(s.strategy, "value")
                            else str(s.strategy)
                        ),
                        "actions": [
                            {
                                "name": action.name,
                                "parameters": dict(action.parameters or {}),
                                "estimated_risk": action.estimated_risk,
                            }
                            for action in (s.actions or [])
                        ],
                    },
                }
                for s in suggestions
            ],
            "selected_verifier_ids": [s.verifier_id for s in selected],
            "selection_policy_rationale": (
                "Risk filtering followed by "
                + ("consensus support, confidence and risk ranking."
                   if selection_policy == "consensus"
                   else "confidence-first ranking.")
            ),
        },
    )


def generate_candidate_recovery_plans(
    suggestions: List[RecoverySuggestion],
    risk_assessments: Dict[str, RiskAssessment],
    context: RecoveryContext,
    safety_constraints: Optional[List[str]] = None,
    selection_policy: str = "confidence",
    max_candidates: int = 5,
) -> List[RecoveryPlan]:
    """Generate bounded alternatives by giving each verifier a priority turn.

    This is a controlled alternative-generation heuristic, not exhaustive
    enumeration of all action subsets.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")

    candidates = []
    seen_signatures = set()
    pivot_orders = [None] + [
        [suggestion.verifier_id] for suggestion in suggestions
    ]

    for priority_ids in pivot_orders:
        candidate = generate_recovery_plan(
            suggestions,
            risk_assessments,
            context,
            safety_constraints=safety_constraints,
            selection_policy=selection_policy,
            priority_verifier_ids=priority_ids,
        )
        signature = tuple(
            (
                action.name.strip().lower(),
                repr(sorted((action.parameters or {}).items())),
            )
            for action in candidate.actions
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        # Stable IDs make experiment outputs and probability maps reproducible.
        signature_text = repr(signature).encode("utf-8")
        stable_id = hashlib.sha256(signature_text).hexdigest()[:10]
        candidate.recovery_id = f"candidate-{stable_id}"

        candidate.metadata["candidate_generation"] = {
            "method": "verifier_pivot_priority",
            "priority_verifier_ids": list(priority_ids or []),
        }
        candidate.metadata["candidate_index"] = len(candidates)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break

    return candidates
