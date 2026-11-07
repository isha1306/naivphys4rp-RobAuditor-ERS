"""Tests for the backend-neutral recovery execution interface."""

import unittest

from recovery.execution_backend import (
    RealRobotRecoveryBackend,
    SimulationRecoveryBackend,
)
from recovery.models import (
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryPlan,
    RecoveryStrategy,
)


def make_context():
    return RecoveryContext(
        failed_action_id="failure_1",
        failed_action="grasp",
        participants=["ur10"],
    )


def make_plan(status=PlanStatus.VALIDATED, actions=None):
    return RecoveryPlan(
        recovery_id="recovery_1",
        failed_action_id="failure_1",
        actions=actions or [
            RecoveryAction(name="return_to_pre_grasp", estimated_risk=0.2)
        ],
        strategies=[RecoveryStrategy.GOB],
        status=status,
        explanation="test plan",
    )


class TestSimulationRecoveryBackend(unittest.TestCase):
    def test_executes_registered_action(self):
        calls = []

        def executor(plan, context, action):
            calls.append((plan.recovery_id, context.failed_action_id, action.name))
            return True

        backend = SimulationRecoveryBackend(
            {"return_to_pre_grasp": executor}
        )
        result = backend.execute(make_plan(), make_context())

        self.assertTrue(result.success)
        self.assertEqual(result.backend, "simulation")
        self.assertEqual(result.executed_actions, ["return_to_pre_grasp"])
        self.assertEqual(len(calls), 1)

    def test_refuses_unvalidated_plan(self):
        backend = SimulationRecoveryBackend({"return_to_pre_grasp": lambda *_: True})
        result = backend.execute(
            make_plan(status=PlanStatus.PROPOSED), make_context()
        )

        self.assertFalse(result.success)
        self.assertIn("VALIDATED", result.message)

    def test_refuses_unregistered_action_before_execution(self):
        calls = []
        backend = SimulationRecoveryBackend(
            {"some_other_action": lambda *_: calls.append(True)}
        )
        result = backend.execute(make_plan(), make_context())

        self.assertFalse(result.success)
        self.assertEqual(calls, [])
        self.assertEqual(result.failed_action, "return_to_pre_grasp")

    def test_reports_callback_failure(self):
        backend = SimulationRecoveryBackend(
            {"return_to_pre_grasp": lambda *_: False}
        )
        result = backend.execute(make_plan(), make_context())

        self.assertFalse(result.success)
        self.assertEqual(result.failed_action, "return_to_pre_grasp")

    def test_reports_callback_exception(self):
        def fail(*_):
            raise RuntimeError("simulator fault")

        backend = SimulationRecoveryBackend({"return_to_pre_grasp": fail})
        result = backend.execute(make_plan(), make_context())

        self.assertFalse(result.success)
        self.assertIn("simulator fault", result.message)
        self.assertEqual(result.details["exception_type"], "RuntimeError")


class TestRealRobotRecoveryBackend(unittest.TestCase):
    def test_real_backend_fails_closed_without_sending_commands(self):
        backend = RealRobotRecoveryBackend()
        result = backend.execute(make_plan(), make_context())

        self.assertFalse(result.success)
        self.assertEqual(result.backend, "real")
        self.assertIn("No hardware commands were sent", result.message)


if __name__ == "__main__":
    unittest.main()
