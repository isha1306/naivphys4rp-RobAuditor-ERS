from .models import (
    RecoveryAction,
    RecoveryStrategy,
    RecoverySuggestion,
    VerifierResult,
)

from .suggestion_normalizer import normalize_suggestions


def test_normalize_suggestions():

    suggestion = RecoverySuggestion(
        verifier_id="grasp_verifier",
        strategy=RecoveryStrategy.GOB,
        actions=[
            RecoveryAction(name="retry_grasp")
        ],
        confidence=0.85,
        explanation="Grasp failed.",
    )

    results = [
        VerifierResult(
            verifier_id="grasp_verifier",
            decision=False,
            confidence=0.90,
            suggestion=suggestion,
        ),
        VerifierResult(
            verifier_id="placement_verifier",
            decision=True,
        ),
    ]

    normalized = normalize_suggestions(results)

    assert len(normalized) == 1
    assert normalized[0].verifier_id == "grasp_verifier"
    assert normalized[0].confidence == 0.90
    assert normalized[0].strategy == RecoveryStrategy.GOB

    print("Normalizer test passed!")


if __name__ == "__main__":
    test_normalize_suggestions()