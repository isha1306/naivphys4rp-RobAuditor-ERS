"""MuJoCo-specific adapter for the shared recovery execution backend.

This module does not import the main simulation script. The caller supplies
callbacks that close over the live MuJoCo model/data and task waypoints.
That avoids a circular import and keeps the recovery core simulator-neutral.
"""

from typing import Callable, Dict, Mapping, Optional

from .execution_backend import (
    RecoveryExecutionResult,
    SimulationRecoveryBackend,
)
from .models import RecoveryAction, RecoveryContext, RecoveryPlan


MUJOCO_RECOVERY_ACTIONS = (
    "return_to_pre_grasp",
    "replan_pre_grasp",
    "return_to_last_safe_waypoint",
    "return_to_placement_waypoint",
    "replan_insertion",
    "stop_robot",
    "introduce_replacement_canister",
)


def build_mujoco_recovery_backend(
    callbacks: Mapping[str, Callable[[RecoveryPlan, RecoveryContext, RecoveryAction], object]],
) -> SimulationRecoveryBackend:
    """Build a simulation backend from live-state MuJoCo callbacks.

    Required callback keys correspond to the action names implemented by the
    existing simulation recovery functions. Missing callbacks are allowed at
    construction time, but plans using them are refused during preflight.
    """

    unknown = set(callbacks) - set(MUJOCO_RECOVERY_ACTIONS)
    if unknown:
        raise ValueError(
            "Unknown MuJoCo recovery callback(s): "
            + ", ".join(sorted(unknown))
        )

    return SimulationRecoveryBackend(dict(callbacks))


def make_mujoco_recovery_callbacks(
    *,
    model,
    data,
    planning_data=None,
    arm_name: Optional[str] = None,
    pre_grasp_pos=None,
    tool_rot=None,
    last_safe_waypoint=None,
    carry_rot=None,
    safe_placement_waypoint=None,
    target_center=None,
    approach_height: float = 0.12,
    recovery_functions: Mapping[str, Callable],
) -> Dict[str, Callable]:
    """Bind existing MuJoCo recovery functions to their required arguments.

    `recovery_functions` must contain the existing functions keyed by:
      gob_pre_grasp, ina_pregrasp, gbb_transfer, gob_placement,
      ina_insertion, ess_stop, ino_replacement.

    The returned callbacks preserve those functions' internal validation,
    movement, and postcondition checks. All required live-state arguments are
    checked before callbacks are created.
    """

    required_functions = {
        "gob_pre_grasp",
        "ina_pregrasp",
        "gbb_transfer",
        "gob_placement",
        "ina_insertion",
        "ess_stop",
        "ino_replacement",
    }
    missing_functions = required_functions - set(recovery_functions)
    if missing_functions:
        raise ValueError(
            "Missing MuJoCo recovery function(s): "
            + ", ".join(sorted(missing_functions))
        )

    callbacks: Dict[str, Callable] = {}

    def bind(action_name, required_values, callback):
        missing = [
            name for name, value in required_values.items()
            if value is None
        ]
        if missing:
            return
        callbacks[action_name] = callback

    bind(
        "return_to_pre_grasp",
        {
            "planning_data": planning_data,
            "arm_name": arm_name,
            "pre_grasp_pos": pre_grasp_pos,
            "tool_rot": tool_rot,
        },
        lambda plan, context, action: recovery_functions["gob_pre_grasp"](
            model, data, planning_data, arm_name, pre_grasp_pos, tool_rot, plan
        ),
    )
    bind(
        "replan_pre_grasp",
        {
            "planning_data": planning_data,
            "arm_name": arm_name,
            "pre_grasp_pos": pre_grasp_pos,
            "tool_rot": tool_rot,
        },
        lambda plan, context, action: recovery_functions["ina_pregrasp"](
            model, data, planning_data, arm_name, pre_grasp_pos, tool_rot, plan
        ),
    )
    bind(
        "return_to_last_safe_waypoint",
        {
            "planning_data": planning_data,
            "arm_name": arm_name,
            "last_safe_waypoint": last_safe_waypoint,
            "carry_rot": carry_rot,
        },
        lambda plan, context, action: recovery_functions["gbb_transfer"](
            model, data, planning_data, arm_name,
            last_safe_waypoint, carry_rot, plan
        ),
    )
    bind(
        "return_to_placement_waypoint",
        {
            "planning_data": planning_data,
            "arm_name": arm_name,
            "safe_placement_waypoint": safe_placement_waypoint,
            "carry_rot": carry_rot,
        },
        lambda plan, context, action: recovery_functions["gob_placement"](
            model, data, planning_data, arm_name,
            safe_placement_waypoint, carry_rot, plan
        ),
    )
    bind(
        "replan_insertion",
        {
            "planning_data": planning_data,
            "arm_name": arm_name,
            "target_center": target_center,
            "carry_rot": carry_rot,
        },
        lambda plan, context, action: recovery_functions["ina_insertion"](
            model, data, planning_data, arm_name,
            target_center, carry_rot, plan,
            approach_height=approach_height,
        ),
    )
    bind(
        "stop_robot",
        {"arm_name": arm_name},
        lambda plan, context, action: recovery_functions["ess_stop"](
            model, data, arm_name, plan
        ),
    )
    bind(
        "introduce_replacement_canister",
        {},
        lambda plan, context, action: recovery_functions["ino_replacement"](
            model, data, plan
        ),
    )

    return callbacks
