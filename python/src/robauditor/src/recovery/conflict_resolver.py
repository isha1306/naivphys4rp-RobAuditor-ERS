"""Detect duplicate, compatible, and conflicting recovery suggestions.

The resolver uses conservative structural rules and explicit conflict metadata.
Robot/domain-specific action incompatibilities can be supplied in suggestion
metadata rather than being guessed from arbitrary action names.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Sequence, Tuple

from .models import RecoverySuggestion


class ConflictType(str, Enum):
    COMPATIBLE = "compatible"
    DUPLICATE = "duplicate"
    CONFLICTING = "conflicting"


@dataclass
class ConflictReport:
    """Classification of a pair of recovery suggestions."""
    verifier_a: str
    verifier_b: str
    classification: ConflictType
    explanation: str


def _normalize_value(value: Any):
    """Convert nested values into deterministic, comparable structures."""
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((_normalize_value(key), _normalize_value(val))
                 for key, val in value.items()),
                key=repr,
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_normalize_value(item) for item in value), key=repr))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _action_signature(action) -> Tuple:
    """Build a signature used to identify duplicate actions."""
    return (
        str(action.name).strip().casefold(),
        _normalize_value(action.parameters or {}),
    )


def _get_declared_conflicts(suggestion) -> set:
    """Read explicitly declared verifier conflicts from suggestion metadata."""
    metadata = suggestion.metadata or {}
    conflicts = metadata.get("conflicts_with", [])
    if isinstance(conflicts, str):
        conflicts = [conflicts]
    if not isinstance(conflicts, (list, tuple, set)):
        return set()
    return {
        str(name).strip().casefold()
        for name in conflicts
        if str(name).strip()
    }


def _get_declared_action_conflicts(suggestion) -> set:
    """Read action names this suggestion declares incompatible."""
    metadata = suggestion.metadata or {}
    conflicts = metadata.get(
        "conflicts_with_actions",
        metadata.get("incompatible_actions", []),
    )
    if isinstance(conflicts, str):
        conflicts = [conflicts]
    if not isinstance(conflicts, (list, tuple, set)):
        return set()
    return {
        str(name).strip().casefold()
        for name in conflicts
        if str(name).strip()
    }


def _has_parameter_conflict(action_a, action_b) -> bool:
    """Detect contradictory values for shared parameters of the same action."""
    if str(action_a.name).strip().casefold() != str(action_b.name).strip().casefold():
        return False

    params_a = action_a.parameters or {}
    params_b = action_b.parameters or {}
    if not isinstance(params_a, dict) or not isinstance(params_b, dict):
        # Malformed parameters are a validation error; do not crash resolution.
        return False

    shared_keys = set(params_a) & set(params_b)
    return any(
        _normalize_value(params_a[key]) != _normalize_value(params_b[key])
        for key in shared_keys
    )


def classify_pair(
    suggestion_a: RecoverySuggestion,
    suggestion_b: RecoverySuggestion,
) -> ConflictReport:
    """Classify two suggestions as compatible, duplicate, or conflicting."""
    actions_a = suggestion_a.actions or []
    actions_b = suggestion_b.actions or []
    id_a = str(suggestion_a.verifier_id).strip()
    id_b = str(suggestion_b.verifier_id).strip()

    conflicts_a = _get_declared_conflicts(suggestion_a)
    conflicts_b = _get_declared_conflicts(suggestion_b)
    if id_b.casefold() in conflicts_a or id_a.casefold() in conflicts_b:
        return ConflictReport(
            id_a, id_b, ConflictType.CONFLICTING,
            "One verifier explicitly declared the other verifier's suggestion "
            "as conflicting.",
        )

    declared_action_conflicts_a = _get_declared_action_conflicts(suggestion_a)
    declared_action_conflicts_b = _get_declared_action_conflicts(suggestion_b)
    names_a = {str(action.name).strip().casefold() for action in actions_a}
    names_b = {str(action.name).strip().casefold() for action in actions_b}
    if names_b & declared_action_conflicts_a or names_a & declared_action_conflicts_b:
        return ConflictReport(
            id_a, id_b, ConflictType.CONFLICTING,
            "A suggestion explicitly declared an action in the other "
            "suggestion as incompatible.",
        )

    signatures_a = [_action_signature(action) for action in actions_a]
    signatures_b = [_action_signature(action) for action in actions_b]
    if signatures_a and signatures_a == signatures_b:
        return ConflictReport(
            id_a, id_b, ConflictType.DUPLICATE,
            "Both verifiers proposed the same recovery action sequence.",
        )

    for action_a in actions_a:
        for action_b in actions_b:
            if _has_parameter_conflict(action_a, action_b):
                return ConflictReport(
                    id_a, id_b, ConflictType.CONFLICTING,
                    f"Both suggestions contain action '{action_a.name}' "
                    "with different values for shared parameters.",
                )

    return ConflictReport(
        id_a, id_b, ConflictType.COMPATIBLE,
        "No explicitly declared, duplicate, or shared-parameter "
        "conflict was detected. Strategy labels alone do not imply conflict.",
    )


def resolve_conflicts(
    suggestions: Sequence[RecoverySuggestion],
) -> List[ConflictReport]:
    """Classify every unique pair of recovery suggestions."""
    reports = []
    for i in range(len(suggestions)):
        for j in range(i + 1, len(suggestions)):
            reports.append(classify_pair(suggestions[i], suggestions[j]))
    return reports
