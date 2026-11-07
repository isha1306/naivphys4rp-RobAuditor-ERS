"""Shared data models for ensemble-based recovery planning.

The recovery engine is independent of the robot simulator. Verifiers produce
these objects; the synthesizer turns their suggestions into a validated plan.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerifierDecision(str, Enum):
    """Normalized verifier decision labels."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class EvidenceItem:
    """A typed evidence reference; image payloads remain external."""

    evidence_type: str
    description: str = ""
    uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_type or not self.evidence_type.strip():
            raise ValueError("EvidenceItem.evidence_type is required.")
        if self.uri is not None and not isinstance(self.uri, str):
            raise TypeError("EvidenceItem.uri must be a string or None.")


class RecoveryStrategy(str, Enum):
    """Recovery strategy categories used by RobAuditor."""

    ESS = "ESS"  # Enter a safe state.
    HIC = "HIC"  # Request human intervention.
    GBB = "GBB"  # Go back before the failed action.
    GOB = "GOB"  # Go back to a specific waypoint or state.
    INA = "INA"  # Introduce a new action.
    INO = "INO"  # Introduce a new object.


class PlanStatus(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    HUMAN_INTERVENTION = "HUMAN_INTERVENTION"


@dataclass
class RecoveryAction:
    """One normalized action in a recovery sequence."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    prerequisites: List[str] = field(default_factory=list)
    estimated_risk: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("RecoveryAction.name must be a non-empty string.")
        if self.estimated_risk is not None and not 0.0 <= self.estimated_risk <= 1.0:
            raise ValueError("estimated_risk must be between 0.0 and 1.0.")


@dataclass
class RecoverySuggestion:
    """A verifier's proposed strategy and ordered recovery actions."""

    verifier_id: str
    strategy: RecoveryStrategy
    actions: List[RecoveryAction]
    confidence: Optional[float] = None
    explanation: str = ""
    modalities: List[str] = field(default_factory=list)
    failure_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    decision: Optional[bool] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    visual_evidence: List[EvidenceItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("RecoverySuggestion.verifier_id is required.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")


@dataclass
class VerifierResult:
    """Standard result emitted by one independent verifier."""

    verifier_id: str
    decision: Optional[bool]
    confidence: Optional[float] = None
    explanation: str = ""
    suggestion: Optional[RecoverySuggestion] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    visual_evidence: List[EvidenceItem] = field(default_factory=list)
    explanation_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def decision_label(self) -> VerifierDecision:
        """Map the legacy bool decision to PASS/FAIL/UNCERTAIN."""
        if self.decision is True:
            return VerifierDecision.PASS
        if self.decision is False:
            return VerifierDecision.FAIL
        return VerifierDecision.UNCERTAIN

    def to_quadruplet(self) -> Dict[str, Any]:
        """Serialize the verifier's Ds/Cf/Ed/Er output without losing evidence."""
        recovery = None
        if self.suggestion is not None:
            recovery = {
                "strategy": (
                    self.suggestion.strategy.value
                    if isinstance(self.suggestion.strategy, Enum)
                    else str(self.suggestion.strategy)
                ),
                "actions": [
                    {
                        "name": action.name,
                        "parameters": dict(action.parameters or {}),
                        "description": action.description,
                        "estimated_risk": action.estimated_risk,
                    }
                    for action in (self.suggestion.actions or [])
                ],
                "failure_type": self.suggestion.failure_type,
            }
        return {
            "verifier_id": self.verifier_id,
            "Ds": {"decision": self.decision, "label": self.decision_label.value},
            "Cf": {"confidence": self.confidence},
            "Ed": {
                "text": self.explanation,
                "evidence": dict(self.evidence or {}),
                "visual_evidence": [
                    {
                        "type": item.evidence_type,
                        "uri": item.uri,
                        "description": item.description,
                        "metadata": dict(item.metadata or {}),
                    }
                    for item in (self.visual_evidence or [])
                ],
                "metadata": dict(self.explanation_metadata or {}),
            },
            "Er": recovery,
        }

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("VerifierResult.verifier_id is required.")
        if self.decision not in (True, False, None):
            raise ValueError("decision must be True, False, or None.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")


@dataclass
class RecoveryContext:
    """Snapshot of task and robot state at the point of failure."""

    failed_action_id: str
    failed_action: str
    participants: Dict[str, str] = field(default_factory=dict)
    object_state: Optional[str] = None
    robot_state: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    previous_actions: List[str] = field(default_factory=list)
    remaining_task: List[str] = field(default_factory=list)
    safety_state: Dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 0
    max_attempts: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.failed_action_id or not self.failed_action:
            raise ValueError("A failed action ID and action name are required.")
        if self.attempt_count < 0 or self.max_attempts < 0:
            raise ValueError("attempt_count and max_attempts cannot be negative.")


@dataclass
class RecoveryPlan:
    """Synthesized plan plus provenance and validation outcome."""

    recovery_id: str
    failed_action_id: str
    actions: List[RecoveryAction]
    strategies: List[RecoveryStrategy] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PROPOSED
    score: Optional[float] = None
    explanation: str = ""
    accepted_suggestions: List[str] = field(default_factory=list)
    rejected_suggestions: Dict[str, str] = field(default_factory=dict)
    safety_constraints: List[str] = field(default_factory=list)
    max_attempts: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)
