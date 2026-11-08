"""Tests for guarded recovery execution orchestration."""
import unittest

from recovery.execution_backend import RecoveryExecutionBackend, RecoveryExecutionResult
from recovery.execution_coordinator import RecoveryExecutionCoordinator
from recovery.models import PlanStatus, RecoveryAction, RecoveryContext, RecoveryPlan, RecoveryStrategy


def make_context(max_attempts=3):
    return RecoveryContext(
        failed_action_id="failure_1",
        failed_action="grasp",
        max_attempts=max_attempts,
    )


def make_plan(max_attempts=3):
    return RecoveryPlan(
        recovery_id="recovery_1",
        failed_action_id="failure_1",
        actions=[RecoveryAction("return_to_pre_grasp", estimated_risk=0.2)],
        strategies=[RecoveryStrategy.GOB],
        status=PlanStatus.VALIDATED,
        max_attempts=max_attempts,
    )


class SequenceBackend(RecoveryExecutionBackend):
    backend_name = "test"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def execute(self, plan, context):
        self.calls += 1
        return self.results.pop(0)


def result(success, executed=None, failed_action=None):
    return RecoveryExecutionResult(
        success=success,
        backend="test",
        recovery_id="recovery_1",
        message="test result",
        executed_actions=list(executed or []),
        failed_action=failed_action,
    )


class TestRecoveryExecutionCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = RecoveryExecutionCoordinator()

    def test_success_requires_postcondition_confirmation_when_checker_supplied(self):
        backend = SequenceBackend([result(True, ["return_to_pre_grasp"])])
        outcome = self.coordinator.execute(
            make_plan(), make_context(), backend,
            postcondition_checker=lambda *_: True,
        )
        self.assertEqual(outcome.status, PlanStatus.SUCCEEDED)
        self.assertTrue(outcome.postconditions_verified)
        self.assertEqual(backend.calls, 1)

    def test_postcondition_failure_escalates(self):
        plan = make_plan()
        backend = SequenceBackend([result(True, ["return_to_pre_grasp"])])
        outcome = self.coordinator.execute(
            plan, make_context(), backend,
            postcondition_checker=lambda *_: False,
        )
        self.assertEqual(outcome.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertEqual(plan.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertFalse(outcome.postconditions_verified)

    def test_postcondition_exception_escalates(self):
        def broken(*_):
            raise RuntimeError("sensor unavailable")
        outcome = self.coordinator.execute(
            make_plan(), make_context(), SequenceBackend([result(True)]),
            postcondition_checker=broken,
        )
        self.assertEqual(outcome.status, PlanStatus.HUMAN_INTERVENTION)

    def test_partial_execution_is_never_retried(self):
        backend = SequenceBackend([result(False, ["first_action"], "second_action")])
        authorize = lambda *_: True
        outcome = self.coordinator.execute(
            make_plan(), make_context(), backend,
            retry_authorizer=authorize, max_retries=2,
        )
        self.assertEqual(outcome.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertEqual(backend.calls, 1)

    def test_zero_action_failure_retries_only_with_explicit_authorization(self):
        backend = SequenceBackend([result(False), result(True, ["return_to_pre_grasp"])])
        outcome = self.coordinator.execute(
            make_plan(), make_context(), backend,
            retry_authorizer=lambda *_: True, max_retries=1,
            postcondition_checker=lambda *_: True,
        )
        self.assertEqual(outcome.status, PlanStatus.SUCCEEDED)
        self.assertEqual(outcome.attempts, 2)

    def test_retry_denial_stops(self):
        backend = SequenceBackend([result(False), result(True)])
        outcome = self.coordinator.execute(
            make_plan(), make_context(), backend,
            retry_authorizer=lambda *_: False, max_retries=1,
        )
        self.assertEqual(outcome.status, PlanStatus.FAILED)
        self.assertEqual(backend.calls, 1)

    def test_retry_exhaustion_stops(self):
        backend = SequenceBackend([result(False), result(False)])
        outcome = self.coordinator.execute(
            make_plan(max_attempts=2), make_context(max_attempts=2), backend,
            retry_authorizer=lambda *_: True, max_retries=4,
        )
        self.assertEqual(outcome.status, PlanStatus.FAILED)
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(backend.calls, 2)

    def test_safety_interruption_prevents_execution(self):
        backend = SequenceBackend([result(True)])
        outcome = self.coordinator.execute(
            make_plan(), make_context(), backend,
            safety_checker=lambda *_: False,
        )
        self.assertEqual(outcome.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertEqual(backend.calls, 0)

    def test_safety_interruption_between_retry_attempts(self):
        backend = SequenceBackend([result(False), result(True)])
        checks = iter([True, False])
        outcome = self.coordinator.execute(
            make_plan(), make_context(), backend,
            retry_authorizer=lambda *_: True,
            safety_checker=lambda *_: next(checks),
            max_retries=1,
        )
        self.assertEqual(outcome.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertEqual(backend.calls, 1)

    def test_unvalidated_plan_is_refused(self):
        plan = make_plan()
        plan.status = PlanStatus.PROPOSED
        backend = SequenceBackend([result(True)])
        outcome = self.coordinator.execute(plan, make_context(), backend)
        self.assertEqual(outcome.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertEqual(backend.calls, 0)

    def test_negative_retry_limit_rejected(self):
        with self.assertRaises(ValueError):
            self.coordinator.execute(
                make_plan(), make_context(), SequenceBackend([]), max_retries=-1
            )


if __name__ == "__main__":
    unittest.main()
