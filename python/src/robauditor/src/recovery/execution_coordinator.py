"""Guarded orchestration for executing and verifying recovery plans.

This module deliberately keeps planning separate from execution. It never
retries a partially executed plan. Retrying an action sequence is permitted
only when the backend reports that no action ran and the caller explicitly
authorizes retry after checking the current safety state.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .execution_backend import RecoveryExecutionBackend, RecoveryExecutionResult
from .models import PlanStatus, RecoveryContext, RecoveryPlan


PostconditionChecker = Callable[[RecoveryPlan, RecoveryContext], bool]
RetryAuthorizer = Callable[[RecoveryPlan, RecoveryContext, RecoveryExecutionResult], bool]
SafetyChecker = Callable[[RecoveryContext], bool]


@dataclass
class RecoveryOrchestrationResult:
    """Auditable outcome of execution, postcondition checking and retry policy."""

    status: PlanStatus
    recovery_id: str
    attempts: int
    execution_results: List[RecoveryExecutionResult] = field(default_factory=list)
    postconditions_verified: Optional[bool] = None
    message: str = ""


class RecoveryExecutionCoordinator:
    """Execute a validated plan with conservative failure and retry handling."""

    def execute(
        self,
        plan: RecoveryPlan,
        context: RecoveryContext,
        backend: RecoveryExecutionBackend,
        postcondition_checker: Optional[PostconditionChecker] = None,
        retry_authorizer: Optional[RetryAuthorizer] = None,
        safety_checker: Optional[SafetyChecker] = None,
        max_retries: int = 0,
    ) -> RecoveryOrchestrationResult:
        if plan is None or plan.status != PlanStatus.VALIDATED:
            return self._result(
                plan, PlanStatus.HUMAN_INTERVENTION, 0,
                "Execution refused: a VALIDATED recovery plan is required."
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        attempt_limit = min(
            max_retries + 1,
            max(1, context.max_attempts),
            max(1, plan.max_attempts),
        )
        attempts: List[RecoveryExecutionResult] = []

        while len(attempts) < attempt_limit:
            if safety_checker is not None:
                try:
                    safe = safety_checker(context) is True
                except Exception:
                    safe = False
                if not safe:
                    plan.status = PlanStatus.HUMAN_INTERVENTION
                    return RecoveryOrchestrationResult(
                        status=PlanStatus.HUMAN_INTERVENTION,
                        recovery_id=plan.recovery_id,
                        attempts=len(attempts),
                        execution_results=attempts,
                        message="Safety check failed; execution stopped and human intervention is required.",
                    )

            plan.status = PlanStatus.VALIDATED
            execution = backend.execute(plan, context)
            attempts.append(execution)

            if execution.success:
                if postcondition_checker is not None:
                    try:
                        verified = postcondition_checker(plan, context) is True
                    except Exception:
                        verified = False
                    if not verified:
                        plan.status = PlanStatus.HUMAN_INTERVENTION
                        return RecoveryOrchestrationResult(
                            status=PlanStatus.HUMAN_INTERVENTION,
                            recovery_id=plan.recovery_id,
                            attempts=len(attempts),
                            execution_results=attempts,
                            postconditions_verified=False,
                            message="Backend completed the plan, but postconditions were not verified; human intervention is required.",
                        )
                    plan.status = PlanStatus.SUCCEEDED
                    return RecoveryOrchestrationResult(
                        status=PlanStatus.SUCCEEDED,
                        recovery_id=plan.recovery_id,
                        attempts=len(attempts),
                        execution_results=attempts,
                        postconditions_verified=True,
                        message="Recovery execution and postconditions were verified.",
                    )

                plan.status = PlanStatus.SUCCEEDED
                return RecoveryOrchestrationResult(
                    status=PlanStatus.SUCCEEDED,
                    recovery_id=plan.recovery_id,
                    attempts=len(attempts),
                    execution_results=attempts,
                    postconditions_verified=None,
                    message="Backend reported success; no postcondition checker was configured.",
                )

            # Never replay a sequence if any action may already have changed state.
            if execution.executed_actions:
                plan.status = PlanStatus.HUMAN_INTERVENTION
                return RecoveryOrchestrationResult(
                    status=PlanStatus.HUMAN_INTERVENTION,
                    recovery_id=plan.recovery_id,
                    attempts=len(attempts),
                    execution_results=attempts,
                    message="Partial recovery execution detected; automatic retry is unsafe and human intervention is required.",
                )

            if len(attempts) >= attempt_limit or retry_authorizer is None:
                plan.status = PlanStatus.FAILED
                return RecoveryOrchestrationResult(
                    status=PlanStatus.FAILED,
                    recovery_id=plan.recovery_id,
                    attempts=len(attempts),
                    execution_results=attempts,
                    message="Recovery execution failed before any action completed; retry was not authorized or retry limit was reached.",
                )

            try:
                authorized = retry_authorizer(plan, context, execution) is True
            except Exception:
                authorized = False
            if not authorized:
                plan.status = PlanStatus.FAILED
                return RecoveryOrchestrationResult(
                    status=PlanStatus.FAILED,
                    recovery_id=plan.recovery_id,
                    attempts=len(attempts),
                    execution_results=attempts,
                    message="Retry was not explicitly authorized; recovery execution failed.",
                )

        plan.status = PlanStatus.FAILED
        return RecoveryOrchestrationResult(
            status=PlanStatus.FAILED,
            recovery_id=plan.recovery_id,
            attempts=len(attempts),
            execution_results=attempts,
            message="Recovery retry limit reached.",
        )

    @staticmethod
    def _result(plan: Optional[RecoveryPlan], status: PlanStatus,
                attempts: int, message: str) -> RecoveryOrchestrationResult:
        return RecoveryOrchestrationResult(
            status=status,
            recovery_id=getattr(plan, "recovery_id", ""),
            attempts=attempts,
            message=message,
        )
