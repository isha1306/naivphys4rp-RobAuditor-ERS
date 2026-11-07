"""Unit tests for MuJoCo recovery callback binding."""

import unittest

from recovery.mujoco_recovery_backend import (
    MUJOCO_RECOVERY_ACTIONS,
    build_mujoco_recovery_backend,
    make_mujoco_recovery_callbacks,
)
from recovery.models import (
    PlanStatus,
    RecoveryAction,
    RecoveryContext,
    RecoveryPlan,
    RecoveryStrategy,
)


class TestMuJoCoRecoveryBackend(unittest.TestCase):
    def _functions(self, calls):
        def make(name):
            def function(*args, **kwargs):
                calls.append((name, args, kwargs))
                return True
            return function

        return {
            "gob_pre_grasp": make("gob_pre_grasp"),
            "ina_pregrasp": make("ina_pregrasp"),
            "gbb_transfer": make("gbb_transfer"),
            "gob_placement": make("gob_placement"),
            "ina_insertion": make("ina_insertion"),
            "ess_stop": make("ess_stop"),
            "ino_replacement": make("ino_replacement"),
        }

    def _plan(self, action_name, strategy):
        return RecoveryPlan(
            recovery_id="r1",
            failed_action_id="f1",
            actions=[RecoveryAction(name=action_name, estimated_risk=0.1)],
            strategies=[strategy],
            status=PlanStatus.VALIDATED,
        )

    def test_binds_all_seven_existing_recovery_actions(self):
        calls = []
        functions = self._functions(calls)
        callbacks = make_mujoco_recovery_callbacks(
            model="model",
            data="data",
            planning_data="planner",
            arm_name="left",
            pre_grasp_pos=[0, 0, 0],
            tool_rot=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            last_safe_waypoint=[0, 0, 0],
            carry_rot=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            safe_placement_waypoint=[0, 0, 0],
            target_center=[0, 0, 0],
            recovery_functions=functions,
        )
        self.assertEqual(set(callbacks), set(MUJOCO_RECOVERY_ACTIONS))

    def test_ino_can_be_bound_without_arm_waypoints(self):
        calls = []
        callbacks = make_mujoco_recovery_callbacks(
            model="model",
            data="data",
            recovery_functions=self._functions(calls),
        )
        self.assertIn("introduce_replacement_canister", callbacks)
        callbacks["introduce_replacement_canister"](
            self._plan(
                "introduce_replacement_canister", RecoveryStrategy.INO
            ),
            RecoveryContext(failed_action_id="f1", failed_action="object"),
            RecoveryAction(name="introduce_replacement_canister"),
        )
        self.assertEqual(calls[0][0], "ino_replacement")

    def test_missing_runtime_state_leaves_action_unregistered(self):
        callbacks = make_mujoco_recovery_callbacks(
            model="model",
            data="data",
            recovery_functions=self._functions([]),
        )
        self.assertNotIn("return_to_pre_grasp", callbacks)

    def test_unknown_callback_name_rejected(self):
        with self.assertRaises(ValueError):
            build_mujoco_recovery_backend({"not_a_recovery_action": lambda *_: True})


if __name__ == "__main__":
    unittest.main()
