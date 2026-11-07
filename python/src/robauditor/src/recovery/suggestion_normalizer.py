"""Normalize verifier quadruplets into recovery suggestions."""

from dataclasses import replace
from typing import Iterable, List

from .models import RecoverySuggestion, VerifierResult


def normalize_suggestion(result: VerifierResult) -> RecoverySuggestion | None:
    """Normalize Er while preserving Ds, Cf, Ed and evidence provenance.

    PASS results do not request recovery. FAIL and UNCERTAIN results may
    contribute a suggestion when one is present.
    """
    if not isinstance(result, VerifierResult):
        raise TypeError("Expected a VerifierResult.")

    if result.decision is True:
        return None

    suggestion = result.suggestion
    if suggestion is None:
        return None

    return replace(
        suggestion,
        verifier_id=result.verifier_id,
        confidence=(
            result.confidence
            if result.confidence is not None
            else suggestion.confidence
        ),
        explanation=result.explanation or suggestion.explanation,
        decision=result.decision,
        evidence=dict(result.evidence or {}),
        visual_evidence=list(result.visual_evidence or []),
        metadata={
            **dict(suggestion.metadata or {}),
            "verifier_explanation_metadata": dict(
                result.explanation_metadata or {}
            ),
            "decision_label": result.decision_label.value,
        },
    )


def normalize_suggestions(
    results: Iterable[VerifierResult],
) -> List[RecoverySuggestion]:
    """Normalize recovery suggestions from an ensemble of verifiers."""
    normalized = []
    for result in results:
        suggestion = normalize_suggestion(result)
        if suggestion is not None:
            normalized.append(suggestion)
    return normalized
