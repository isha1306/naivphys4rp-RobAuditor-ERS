"""Backend-neutral execution interface for validated recovery plans.

The recovery planner decides what should happen. An execution backend decides
how to carry out the resulting actions on a particular robot or simulator.

This module intentionally has no MuJoCo, ROS, or hardware dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional
from abc import ABC, abstractmethod

from .models import PlanStatus, RecoveryAction, RecoveryContext, RecoveryPlan


@dataclass
class RecoveryExecutionResult:
    """Structured outcome of asking an execution backend to run a plan."""

    success: bool
    backend: str
    recovery_id: str
    message: str
    executed_actions: List[str] = field(default_factory=list)
    failed_action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class RecoveryExecutionBackend(ABC):
    """Common interface implemented by simulation and hardware backends."""

    backend_name = "abstract"

    @abstractmethod
    def execute(
        self,
        plan: RecoveryPlan,
        context: RecoveryContext,
    ) -> RecoveryExecutionResult:
        """Execute a validated recovery plan or return a failure result."""
        raise NotImplementedError


ActionExecutor = Callable[
    [RecoveryPlan, RecoveryContext, RecoveryAction],
    Any,
]


class SimulationRecoveryBackend(RecoveryExecutionBackend):
    """Dispatch validated plan actions to explicitly registered sim callbacks.

    The callbacks are supplied by the MuJoCo integration layer. This keeps
    MuJoCo-specific state and function arguments outside the recovery core.
    """

    backend_name = "simulation"

    def __init__(self, action_executors: Mapping[str, ActionExecutor]):
        self._action_executors = dict(action_executors)

    def supports(self, plan: RecoveryPlan) -> bool:
        if not plan.actions:
            return False
        return all(action.name in self._action_executors for action in plan.actions)

    def execute(
        self,
        plan: RecoveryPlan,
        context: RecoveryContext,
    ) -> RecoveryExecutionResult:
        invalid = _preflight_plan(plan)
        if invalid:
            return RecoveryExecutionResult(
                success=False,
                backend=self.backend_name,
                recovery_id=getattr(plan, "recovery_id", ""),
                message=invalid,
            )

        unsupported = [
            action.name for action in plan.actions
            if action.name not in self._action_executors
        ]
        if unsupported:
            return RecoveryExecutionResult(
                success=False,
                backend=self.backend_name,
                recovery_id=plan.recovery_id,
                message="No simulation executor registered for: "
                        + ", ".join(unsupported),
                failed_action=unsupported[0],
            )

        executed: List[str] = []
        for action in plan.actions:
            try:
                callback_result = self._action_executors[action.name](
                    plan, context, action
                )
                if callback_result is False:
                    return RecoveryExecutionResult(
                        success=False,
                        backend=self.backend_name,
                        recovery_id=plan.recovery_id,
                        message=f"Simulation executor reported failure for "
                                f"'{action.name}'.",
                        executed_actions=executed,
                        failed_action=action.name,
                    )
                executed.append(action.name)
            except Exception as exc:
                return RecoveryExecutionResult(
                    success=False,
                    backend=self.backend_name,
                    recovery_id=plan.recovery_id,
                    message=f"Simulation execution failed for "
                            f"'{action.name}': {exc}",
                    executed_actions=executed,
                    failed_action=action.name,
                    details={"exception_type": type(exc).__name__},
                )

        return RecoveryExecutionResult(
            success=True,
            backend=self.backend_name,
            recovery_id=plan.recovery_id,
            message="All recovery actions completed in the simulation backend.",
            executed_actions=executed,
        )


class RealRobotRecoveryBackend(RecoveryExecutionBackend):
    """Fail-closed placeholder for a future physical robot integration.

    This class deliberately does not issue ROS commands. A real implementation
    must add robot-specific state checks, safety interlocks, command execution,
    timeout handling, and measured postcondition verification.
    """

    backend_name = "real"

    def execute(
        self,
        plan: RecoveryPlan,
        context: RecoveryContext,
    ) -> RecoveryExecutionResult:
        return RecoveryExecutionResult(
            success=False,
            backend=self.backend_name,
            recovery_id=getattr(plan, "recovery_id", ""),
            message=(
                "Real-robot execution is not configured. No hardware commands "
                "were sent."
            ),
        )


def _preflight_plan(plan: RecoveryPlan) -> Optional[str]:
    """Reject plans that are not explicitly validated and executable."""

    if plan is None:
        return "No recovery plan was supplied."
    if plan.status != PlanStatus.VALIDATED:
        return (
            "Execution refused: recovery plan status must be VALIDATED "
            f"(received {getattr(plan.status, 'value', plan.status)!r})."
        )
    if not plan.actions:
        return "Execution refused: recovery plan contains no actions."
    if not plan.recovery_id:
        return "Execution refused: recovery plan has no recovery_id."
    return None
