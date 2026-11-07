from .models import (
    RecoveryAction,
    RecoveryStrategy,
    RecoverySuggestion,
)

from .conflict_resolver import (
    ConflictType,
    classify_pair,
    resolve_conflicts,
)


def make_suggestion(verifier_id, action_name, parameters=None, metadata=None):
    return RecoverySuggestion(
        verifier_id=verifier_id,
        strategy=RecoveryStrategy.GOB,
        actions=[
            RecoveryAction(
                name=action_name,
                parameters=parameters or {},
            )
        ],
        confidence=0.85,
        explanation="Test recovery suggestion",
        metadata=metadata or {},
    )


def test_duplicate_suggestions():
    a = make_suggestion(
        "grasp_verifier",
        "retry_grasp",
        {"object": "canister"},
    )

    b = make_suggestion(
        "feasibility_verifier",
        "retry_grasp",
        {"object": "canister"},
    )

    result = classify_pair(a, b)

    assert result.classification == ConflictType.DUPLICATE


def test_compatible_suggestions():
    a = make_suggestion(
        "grasp_verifier",
        "retry_grasp",
    )

    b = make_suggestion(
        "safety_verifier",
        "move_to_safe_position",
    )

    result = classify_pair(a, b)

    assert result.classification == ConflictType.COMPATIBLE


def test_conflicting_parameters():
    a = make_suggestion(
        "grasp_verifier",
        "move_robot",
        {"direction": "left"},
    )

    b = make_suggestion(
        "safety_verifier",
        "move_robot",
        {"direction": "right"},
    )

    result = classify_pair(a, b)

    assert result.classification == ConflictType.CONFLICTING


def test_explicit_conflict():
    a = make_suggestion(
        "grasp_verifier",
        "retry_grasp",
        metadata={"conflicts_with": ["safety_verifier"]},
    )

    b = make_suggestion(
        "safety_verifier",
        "stop_robot",
    )

    result = classify_pair(a, b)

    assert result.classification == ConflictType.CONFLICTING


def test_multiple_suggestions():
    suggestions = [
        make_suggestion("verifier_a", "retry_grasp"),
        make_suggestion("verifier_b", "retry_grasp"),
        make_suggestion("verifier_c", "move_to_safe_position"),
    ]

    reports = resolve_conflicts(suggestions)

    assert len(reports) == 3


if __name__ == "__main__":
    test_duplicate_suggestions()
    test_compatible_suggestions()
    test_conflicting_parameters()
    test_explicit_conflict()
    test_multiple_suggestions()

    print("All conflict resolver tests passed!")