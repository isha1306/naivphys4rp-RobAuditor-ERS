"""Deterministic, one-shot failure injection for simulation experiments."""

from dataclasses import dataclass
from typing import Optional


FAILURE_STAGES = (
    "perception",
    "pre_grasp",
    "grasp",
    "lift",
    "transfer",
    "insertion",
    "placement",
    "object_unavailable",
)


class InjectedFailure(RuntimeError):
    """Exception representing a deliberately injected simulation failure."""

    def __init__(self, stage: str):
        self.stage = stage
        self.failure_id = f"injected_{stage}"
        super().__init__(
            f"Injected failure at stage '{stage}' "
            f"(failure_id={self.failure_id})"
        )


@dataclass
class FailureInjector:
    """Inject one deterministic failure at a configured simulation stage."""

    enabled: bool = False
    stage: Optional[str] = None

    def __post_init__(self):
        if self.stage is not None and self.stage not in FAILURE_STAGES:
            raise ValueError(
                f"Unknown failure stage '{self.stage}'. "
                f"Valid stages: {', '.join(FAILURE_STAGES)}"
            )

        if self.enabled and self.stage is None:
            raise ValueError(
                "A failure stage must be specified when injection is enabled."
            )

        self.triggered = False
        self.triggered_stage = None

    def check(self, current_stage: str) -> bool:
        """
        Check whether the configured failure should occur.

        Returns False when no failure is injected.
        Raises InjectedFailure when the configured stage is reached.
        """
        if not self.enabled or self.triggered:
            return False

        if current_stage != self.stage:
            return False

        self.triggered = True
        self.triggered_stage = current_stage

        print(
            f"[failure-injection] Injecting failure at stage: "
            f"{current_stage}"
        )

        raise InjectedFailure(current_stage)

    def reset(self):
        """Reset the one-shot injector for another experiment."""
        self.triggered = False
        self.triggered_stage = None
