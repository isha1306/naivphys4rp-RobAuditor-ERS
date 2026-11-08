"""Integration tests for the recovery coordinator with a live MuJoCo model.

These tests use a tiny real MuJoCo model and the production MuJoCo backend
builder. They do not launch the full canister demo or require its mesh assets.
"""

import unittest

try:
    import mujoco
except ImportError:  # pragma: no cover - depends on the test environment
    mujoco = None

from recovery.execution_coordinator import RecoveryExecutionCoordinator
from recovery.models import (
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryPlan,
    RecoveryStrategy,
)
from recovery.mujoco_recovery_backend import build_mujoco_recovery_backend


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class TestMuJoCoExecutionCoordinator(unittest.TestCase):
    def setUp(self):
        xml = """
        <mujoco model="coordinator_integration">
          <worldbody>
            <body name="arm_link" pos="0 0 0">
              <joint name="recovery_hinge" type="hinge" axis="0 0 1"/>
              <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.context = RecoveryContext(
            failed_action_id="integration-failure",
            failed_action="simulated_recovery",
            participants={"robot": "test_arm"},
            max_attempts=1,
        )
        self.plan = RecoveryPlan(
            recovery_id="mujoco-coordinator-integration",
            failed_action_id="integration-failure",
            actions=[RecoveryAction(name="stop_robot", estimated_risk=0.05)],
            strategies=[RecoveryStrategy.ESS],
            status=PlanStatus.VALIDATED,
            max_attempts=1,
        )
        self.coordinator = RecoveryExecutionCoordinator()

    def test_live_mujoco_state_change_and_postcondition_pass(self):
        def stop_callback(plan, context, action):
            self.data.qpos[0] = 0.25
            mujoco.mj_forward(self.model, self.data)
            return True

        backend = build_mujoco_recovery_backend({"stop_robot": stop_callback})

        def postconditions(plan, context):
            return (
                abs(float(self.data.qpos[0]) - 0.25) < 1e-9
                and bool((self.data.xpos == self.data.xpos).all())
            )

        result = self.coordinator.execute(
            self.plan,
            self.context,
            backend,
            postcondition_checker=postconditions,
        )

        self.assertEqual(result.status, PlanStatus.SUCCEEDED)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(result.postconditions_verified)
        self.assertEqual(
            result.execution_results[0].executed_actions, ["stop_robot"]
        )
        self.assertAlmostEqual(float(self.data.qpos[0]), 0.25)

    def test_failed_live_postcondition_escalates_to_human(self):
        def stop_callback(plan, context, action):
            self.data.qpos[0] = 0.25
            mujoco.mj_forward(self.model, self.data)
            return True

        backend = build_mujoco_recovery_backend({"stop_robot": stop_callback})
        result = self.coordinator.execute(
            self.plan,
            self.context,
            backend,
            postcondition_checker=lambda plan, context: False,
        )

        self.assertEqual(result.status, PlanStatus.HUMAN_INTERVENTION)
        self.assertFalse(result.postconditions_verified)
        self.assertEqual(result.attempts, 1)

    def test_callback_exception_is_reported_without_replay(self):
        def failing_callback(plan, context, action):
            raise RuntimeError("simulated MuJoCo callback failure")

        backend = build_mujoco_recovery_backend({"stop_robot": failing_callback})
        result = self.coordinator.execute(
            self.plan,
            self.context,
            backend,
            postcondition_checker=lambda plan, context: True,
            max_retries=2,
            retry_authorizer=lambda plan, context, execution: True,
            safety_checker=lambda context: True,
        )

        self.assertEqual(result.status, PlanStatus.FAILED)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(
            result.execution_results[0].details.get("exception_type"),
            "RuntimeError",
        )

    def test_unknown_or_unbound_action_fails_closed(self):
        backend = build_mujoco_recovery_backend({})
        result = self.coordinator.execute(
            self.plan,
            self.context,
            backend,
            postcondition_checker=lambda plan, context: True,
        )

        self.assertEqual(result.status, PlanStatus.FAILED)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.execution_results[0].success)
        self.assertEqual(result.execution_results[0].executed_actions, [])


if __name__ == "__main__":
    unittest.main()
