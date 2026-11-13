#!/usr/bin/env python3
"""MuJoCo RGB-D perception and kinematic pick-and-place demo for a canister and tray.

The loader prunes unrelated task props from mjmodel.xml while retaining the
Tracy robot, table/floor, canister1, and both collision-defined tray rings.
The Robotiq fingers close around the canister. During kinematic arm movement,
a simulator-only pose-following attachment carries the free canister body.
This is not a physical grasp or dynamics validation.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from contextlib import nullcontext
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from recovery.failure_injector import (
    FAILURE_STAGES,
    FailureInjector,
    InjectedFailure,
)
from recovery.simulation_recovery_adapter import (
    build_failure_context,
    build_simulation_recovery_manager,
    process_injected_failure,
)
from recovery.execution_backend import RealRobotRecoveryBackend
from recovery.execution_coordinator import RecoveryExecutionCoordinator
from recovery.mujoco_recovery_backend import (
    build_mujoco_recovery_backend,
    make_mujoco_recovery_callbacks,
)

# MuJoCo is imported only after --mode sim is selected. This lets the
# fail-closed --mode real placeholder run without a MuJoCo installation.
mujoco = None


# Names in the supplied Tracy perception runtime model.
CAMERA_NAME = "d435_camera"
CANISTER_BODY = "canister1"
REPLACEMENT_CANISTER_BODY = "replacement_canister"
TRAY_BODIES = ("Pump1", "Pump2")
ARM_SPECS = {
    "left": {
        "joints": [
            "left_shoulder_pan_joint", "left_shoulder_lift_joint", "left_elbow_joint",
            "left_wrist_1_joint", "left_wrist_2_joint", "left_wrist_3_joint",
        ],
        "actuators": [
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow",
            "left_wrist_1", "left_wrist_2", "left_wrist_3",
        ],
        "site": "left_pinch",
        "gripper": "left_fingers_actuator",
        "mount_body": "left_gripper_mount",
    },
    "right": {
        "joints": [
            "right_shoulder_pan_joint", "right_shoulder_lift_joint", "right_elbow_joint",
            "right_wrist_1_joint", "right_wrist_2_joint", "right_wrist_3_joint",
        ],
        "actuators": [
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow",
            "right_wrist_1", "right_wrist_2", "right_wrist_3",
        ],
        "site": "right_pinch",
        "gripper": "right_fingers_actuator",
        "mount_body": "right_gripper_mount",
    },
}

# The supplied ring boxes are centred at z=.890 m and have a 10 mm vertical
# half-size, so their upper surface is z=.900 m.  The supplied reference
# insertion pose is z=.945 m: a model-specific 45 mm offset above that surface.
# The canister STL is about 119 mm long and 49.9 mm in diameter.  Recalibrate this
# offset if you change the tray or canister geometry.
DEFAULT_PLACE_Z_ABOVE_RING_TOP = 0.0450
ACTIVE_VIEWER = None
ARM_TO_USE = "left"  # Fixed in code: use the left Tracy arm.


def viewer_data_lock():
    """Return MuJoCo passive-viewer lock, or a no-op context without a viewer."""
    global ACTIVE_VIEWER
    if ACTIVE_VIEWER is None:
        return nullcontext()
    try:
        return ACTIVE_VIEWER.lock()
    except Exception as exc:
        print(f"[viewer] unable to acquire viewer data lock: {exc}")
        return nullcontext()


def sync_viewer():
    """Synchronize the passive viewer after releasing any data lock."""
    global ACTIVE_VIEWER
    viewer = ACTIVE_VIEWER
    if viewer is not None:
        try:
            viewer.sync()
        except Exception as exc:
            print(f"[viewer] synchronization failed; disabling viewer updates: {exc}")
            ACTIVE_VIEWER = None


def hold_viewer_after_recovery(viewer):
    """Keep the recovered MuJoCo state visible until the viewer is closed."""
    if viewer is None:
        return
    try:
        if not viewer.is_running():
            return
        print("[viewer] recovery finished; simulation is paused at the recovered state.")
        print("[viewer] inspect the robot/canister, then close the MuJoCo window to exit.")
        while viewer.is_running():
            try:
                viewer.sync()
            except Exception as exc:
                print(f"[viewer] synchronization stopped: {exc}")
                break
            time.sleep(0.02)
    except Exception as exc:
        print(f"[viewer] unable to hold recovery view open: {exc}")


@dataclass
class PerceivedObject:
    center: np.ndarray
    points: np.ndarray
    pixel_count: int


@dataclass
class PerceivedHole:
    name: str
    center_xy: np.ndarray
    ring_top_z: float
    radius: float
    points: np.ndarray
    pixel_count: int


@dataclass
class ArmPlan:
    name: str
    qpos_targets: list[np.ndarray]
    total_joint_travel: float
    max_position_error: float
    max_rotation_error: float


def named_id(model, object_type, name: str, required: bool = True) -> int:
    idx = mujoco.mj_name2id(model, object_type, name)
    if idx < 0 and required:
        raise RuntimeError(f"Required MuJoCo {object_type} named '{name}' was not found")
    return int(idx)


def resolve_model_and_meshdir(args) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[4]

    model_candidates = []
    if args.model:
        model_candidates.append(Path(args.model).expanduser())
    else:
        model_candidates.append(
            repo_root / "cpp/simulation_tools/Mujoco/robots/tracy/mjmodel.xml"
        )

    model_path = next((p.resolve() for p in model_candidates if p.is_file()), None)

    if model_path is None:
        attempted = "\n".join(f"  - {p}" for p in model_candidates)
        raise FileNotFoundError(
            "Could not find the MuJoCo model XML. Tried:\n" + attempted
        )

    mesh_candidates = []
    if args.meshdir:
        mesh_candidates.append(Path(args.meshdir).expanduser())

    mesh_candidates.append(
        repo_root / "cpp/simulation_tools/Mujoco/robots/tracy/flattened_meshes"
    )

    meshdir = next((p.resolve() for p in mesh_candidates if p.is_dir()), None)

    if meshdir is None:
        raise FileNotFoundError(
            f"Could not find flattened_meshes directory. Expected:\n"
            f"{repo_root / 'cpp/simulation_tools/Mujoco/robots/tracy/flattened_meshes'}"
        )

    return model_path, meshdir


def load_model_with_meshdir(xml_path: Path, meshdir: Path):
    """Load a task-minimal scene: robot/table, one canister, and both tray holes.

    Keep the complete `map` subtree because it contains the table, robot arms,
    grippers, and camera. Keep `yellow_floor`, `canister1`, Pump1, and Pump2.
    All other free props, the second canister, and expected-pose ghost meshes
    are removed before MuJoCo compiles the model.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(meshdir))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no <worldbody> element")

    keep_top_level_bodies = {
        "yellow_floor", "map", CANISTER_BODY, "Pump1", "Pump2"
    }
    removed = []
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") not in keep_top_level_bodies:
            removed.append(child.get("name", "<unnamed body>"))
            worldbody.remove(child)

    # Keep an inactive duplicate canister in the compiled model. MuJoCo model
    # topology cannot be extended safely after compilation, so INO activates
    # this precompiled free body by moving its free-joint pose at runtime.
    source_canister = next(
        (
            child for child in list(worldbody)
            if child.tag == "body" and child.get("name") == CANISTER_BODY
        ),
        None,
    )
    if source_canister is None:
        raise RuntimeError(f"Cannot clone missing body '{CANISTER_BODY}'.")

    replacement = copy.deepcopy(source_canister)
    replacement.set("name", REPLACEMENT_CANISTER_BODY)
    replacement.set("pos", "0 0 -10")
    for element in replacement.iter():
        element_name = element.get("name")
        if element is not replacement and element_name:
            element.set("name", f"{REPLACEMENT_CANISTER_BODY}_{element_name}")
    worldbody.append(replacement)

    missing = keep_top_level_bodies - {
        child.get("name") for child in list(worldbody) if child.tag == "body"
    }
    if missing:
        raise RuntimeError(f"Expected task bodies missing from MJCF: {sorted(missing)}")

    # Keyframes in the original full scene contain qpos values for every
    # removed free body. After pruning, their qpos vectors no longer match the
    # reduced model. This experiment sets the arm pose explicitly in Python,
    # so discard those stale keyframes from the temporary scene.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        stale_keyframes = list(keyframe.findall("key"))
        for key in stale_keyframes:
            keyframe.remove(key)
        if not list(keyframe):
            root.remove(keyframe)

    # Remove mesh assets not referenced by any remaining geom. This preserves
    # robot/gripper meshes but drops unrelated bottle/tool/ghost-object meshes.
    asset = root.find("asset")
    if asset is not None:
        used_meshes = {
            geom.get("mesh") for geom in root.iter("geom") if geom.get("mesh")
        }
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used_meshes:
                asset.remove(mesh)

    fd, patched_name = tempfile.mkstemp(prefix="tracy_canister_tray_", suffix=".xml")
    os.close(fd)
    patched_path = Path(patched_name)
    try:
        tree.write(patched_path, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(patched_path))
    finally:
        try:
            patched_path.unlink()
        except OSError:
            pass

    # Keep both tray bodies intact: their box geoms define the collision rings.
    print("[scene] retained: Tracy robot + table/floor + canister1 + replacement canister + Pump1/Pump2")
    print(f"[scene] removed unrelated bodies: {', '.join(removed) if removed else 'none'}")
    print("[scene] tray collision boxes preserved (Pump1/Pump2)")
    return model


def get_name_id(model, typ, name: str) -> int:
    return named_id(model, typ, name, required=True)


INITIAL_POSE = {
    "left_shoulder_pan_joint": 2.140,
    "left_shoulder_lift_joint": -1.317,
    "left_elbow_joint": 1.375,
    "left_wrist_1_joint": 5.603,
    "left_wrist_2_joint": 5.264,
    "left_wrist_3_joint": -1.053,
    "right_shoulder_pan_joint": 4.109,
    "right_shoulder_lift_joint": -1.867,
    "right_elbow_joint": -1.341,
    "right_wrist_1_joint": 3.973,
    "right_wrist_2_joint": 0.849,
    "right_wrist_3_joint": 1.257,
}

def reset_to_initial_pose(model, data):
    """Reset free objects/model, then apply the user's 12-joint start pose."""
    mujoco.mj_resetData(model, data)
    for joint_name, value in INITIAL_POSE.items():
        jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(value)
        # Match position actuator target to the specified starting pose.
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name.replace('_joint', ''))
        if aid >= 0:
            data.ctrl[aid] = float(value)
    mujoco.mj_forward(model, data)


def render_rgbd_segmentation(model, data, camera_id: int, width: int, height: int):
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_id)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_id)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_id)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()
    return rgb, depth, segmentation


def depth_mask_to_world_points(
    model, data, camera_id: int, depth: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Back-project masked metric depth pixels into MuJoCo world coordinates."""
    h, w = depth.shape[:2]
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.empty((0, 3), dtype=float)

    d = depth[v, u].astype(float)
    valid = np.isfinite(d) & (d > 1e-4) & (d < 5.0)
    u, v, d = u[valid], v[valid], d[valid]
    if len(d) == 0:
        return np.empty((0, 3), dtype=float)

    fovy = math.radians(float(model.cam_fovy[camera_id]))
    fy = 0.5 * h / math.tan(0.5 * fovy)
    fx = fy  # square pixels in MuJoCo's pinhole camera model
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)

    # MuJoCo camera convention: +X right, +Y up, looking along local -Z.
    x_cam = (u.astype(float) - cx) * d / fx
    y_cam = -(v.astype(float) - cy) * d / fy
    z_cam = -d
    points_cam = np.column_stack((x_cam, y_cam, z_cam))

    camera_pos = data.cam_xpos[camera_id].copy()
    camera_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()  # camera -> world
    return points_cam @ camera_rot.T + camera_pos


def body_segmentation_mask(model, segmentation: np.ndarray, body_id: int) -> np.ndarray:
    """Mask pixels whose segmentation label is a geom attached to body_id."""
    if segmentation.ndim != 3 or segmentation.shape[2] < 2:
        raise RuntimeError(
            "Expected MuJoCo segmentation image with (object id, object type) channels"
        )
    object_ids = segmentation[..., 0].astype(np.int64)
    object_types = segmentation[..., 1].astype(np.int64)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    # MuJoCo's documented layout is (object id, object type).  Restrict to geom
    # pixels first; the explicit geom-id membership also avoids body/geom ID confusion.
    mask = (object_types == geom_type) & np.isin(object_ids, geom_ids)
    return mask


def fit_circle_xy(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle fit in world XY; ring samples need not be ordered."""
    xy = np.asarray(points[:, :2], dtype=float)
    if len(xy) < 12:
        raise RuntimeError(f"Too few visible tray-ring depth samples ({len(xy)})")
    # x^2 + y^2 = 2*cx*x + 2*cy*y + k
    a = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    b = np.sum(xy * xy, axis=1)
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, k = params
    radius_sq = k + cx * cx + cy * cy
    if radius_sq <= 0:
        raise RuntimeError("Tray-ring circle fit produced a non-positive radius")
    return np.array([cx, cy]), float(math.sqrt(radius_sq))


def estimate_perception(
    model, data, width: int, height: int, hole_index: int, place_z_offset: float
):
    camera_id = get_name_id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    rgb, depth, segmentation = render_rgbd_segmentation(
        model, data, camera_id, width=width, height=height
    )

    canister_body_id = get_name_id(model, mujoco.mjtObj.mjOBJ_BODY, CANISTER_BODY)
    canister_mask = body_segmentation_mask(model, segmentation, canister_body_id)
    canister_points = depth_mask_to_world_points(
        model, data, camera_id, depth, canister_mask
    )
    if len(canister_points) < 20:
        raise RuntimeError(
            f"Canister perception failed: only {len(canister_points)} valid depth points."
        )

    # For the approximately cylindrical canister, the midpoint of the visible
    # point-cloud bounds is less sensitive to a few edge pixels than one point.
    lo = np.percentile(canister_points, 2.0, axis=0)
    hi = np.percentile(canister_points, 98.0, axis=0)
    canister_center = 0.5 * (lo + hi)
    perceived_canister = PerceivedObject(
        center=canister_center,
        points=canister_points,
        pixel_count=int(canister_mask.sum()),
    )

    holes: list[PerceivedHole] = []
    for tray_name in TRAY_BODIES:
        tray_body_id = get_name_id(model, mujoco.mjtObj.mjOBJ_BODY, tray_name)
        tray_mask = body_segmentation_mask(model, segmentation, tray_body_id)
        tray_points = depth_mask_to_world_points(model, data, camera_id, depth, tray_mask)
        if len(tray_points) < 30:
            raise RuntimeError(
                f"Tray perception failed for {tray_name}: only {len(tray_points)} valid points."
            )
        center_xy, radius = fit_circle_xy(tray_points)
        if not (0.015 <= radius <= 0.040):
            raise RuntimeError(
                f"Perceived {tray_name} ring radius {radius:.4f} m is outside expected range; "
                "rejecting uncertain tray perception."
            )
        ring_top_z = float(np.percentile(tray_points[:, 2], 90.0))
        holes.append(
            PerceivedHole(
                name=tray_name,
                center_xy=center_xy,
                ring_top_z=ring_top_z,
                radius=radius,
                points=tray_points,
                pixel_count=int(tray_mask.sum()),
            )
        )

    if hole_index not in (1, 2):
        raise ValueError("--hole must be 1 or 2")
    chosen = holes[hole_index - 1]
    target_center = np.array(
        [chosen.center_xy[0], chosen.center_xy[1], chosen.ring_top_z + place_z_offset],
        dtype=float,
    )
    return rgb, perceived_canister, holes, chosen, target_center


def rotation_error_world(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    """SO(3) logarithm of target * current.T (world-frame rotation error)."""
    r = target_rot @ current_rot.T
    cos_angle = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return 0.5 * vee
    s = math.sin(angle)
    if abs(s) < 1e-7:
        # Near pi: extract a stable axis from the diagonal.
        axis = np.sqrt(np.maximum((np.diag(r) + 1.0) * 0.5, 0.0))
        if r[0, 1] < 0:
            axis[1] *= -1
        if r[0, 2] < 0:
            axis[2] *= -1
        n = np.linalg.norm(axis)
        return angle * (axis / max(n, 1e-9))
    return (angle / (2.0 * s)) * vee


def arm_joint_indices(model, arm_name: str):
    spec = ARM_SPECS[arm_name]
    joint_ids = [get_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in spec["joints"]]
    actuator_ids = [get_name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in spec["actuators"]]
    qpos_ids = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
    dof_ids = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
    return qpos_ids, dof_ids, np.array(actuator_ids, dtype=int)


def solve_site_ik(
    model,
    planning_data,
    arm_name: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    seed_qpos: np.ndarray,
    max_iters: int = 350,
    damping: float = 0.035,
    pos_tol: float = 0.006,
    rot_tol: float = 0.06,
) -> tuple[Optional[np.ndarray], float, float]:
    spec = ARM_SPECS[arm_name]
    site_id = get_name_id(model, mujoco.mjtObj.mjOBJ_SITE, spec["site"])
    qpos_ids, dof_ids, _ = arm_joint_indices(model, arm_name)

    planning_data.qpos[:] = seed_qpos
    mujoco.mj_forward(model, planning_data)

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    best_q = planning_data.qpos.copy()
    best_cost = float("inf")
    best_pos_err = float("inf")
    best_rot_err = float("inf")

    for _ in range(max_iters):
        mujoco.mj_forward(model, planning_data)
        current_pos = planning_data.site_xpos[site_id].copy()
        current_rot = planning_data.site_xmat[site_id].reshape(3, 3).copy()
        pos_err = target_pos - current_pos
        rot_err = rotation_error_world(target_rot, current_rot)
        pnorm = float(np.linalg.norm(pos_err))
        rnorm = float(np.linalg.norm(rot_err))
        cost = pnorm + 0.20 * rnorm
        if cost < best_cost:
            best_cost = cost
            best_q = planning_data.qpos.copy()
            best_pos_err, best_rot_err = pnorm, rnorm
        if pnorm <= pos_tol and rnorm <= rot_tol:
            return planning_data.qpos.copy(), pnorm, rnorm

        mujoco.mj_jacSite(model, planning_data, jacp, jacr, site_id)
        jp = jacp[:, dof_ids]
        jr = jacr[:, dof_ids]
        # Weight orientation less than position, but still constrain the tool pose.
        j = np.vstack((jp, 0.20 * jr))
        e = np.concatenate((pos_err, 0.20 * rot_err))
        lhs = j @ j.T + (damping * damping) * np.eye(6)
        try:
            dq = j.T @ np.linalg.solve(lhs, e)
        except np.linalg.LinAlgError:
            break
        # Avoid large discontinuous joint updates near singular configurations.
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > 0.18:
            dq *= 0.18 / dq_norm
        planning_data.qpos[qpos_ids] += dq
        # Respect MuJoCo joint limits.
        for jid, qadr in zip(
            [get_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in spec["joints"]],
            qpos_ids,
        ):
            lo, hi = model.jnt_range[jid]
            planning_data.qpos[qadr] = np.clip(planning_data.qpos[qadr], lo + 1e-4, hi - 1e-4)

    # Accept a near-converged result only; do not plan a clearly unreachable pose.
    if best_pos_err <= 0.012 and best_rot_err <= 0.12:
        return best_q, best_pos_err, best_rot_err
    return None, best_pos_err, best_rot_err


def interpolate_qpos(q_start: np.ndarray, q_goal: np.ndarray, max_step: float = 0.018):
    distance = float(np.max(np.abs(q_goal - q_start)))
    n = max(2, int(math.ceil(distance / max_step)))
    for i in range(1, n + 1):
        t = i / n
        # Smoothstep gives zero endpoint velocity.
        s = t * t * (3.0 - 2.0 * t)
        yield q_start + s * (q_goal - q_start)


def move_to_joint_target(model, data, arm_name: str, q_target: np.ndarray, settle_ticks: int = 40, final_ticks: int = 600):
    """Kinematic demo motion: interpolate arm qpos and forward the MuJoCo model.

    The supplied actuator model is a stiff position servo whose wrist torque
    limits repeatedly saturate during the loaded cross-table transfer.  For a
    repeatable *simulation demo*, this routine drives joint coordinates
    directly; object contacts, the gripper, tray collisions and gravity remain
    MuJoCo-managed.  This is not a dynamics/controller validation and must not
    be used as a real-robot control method.
    """
    qpos_ids, _, actuator_ids = arm_joint_indices(model, arm_name)
    with viewer_data_lock():
        q_start = data.qpos[qpos_ids].copy()
        data.ctrl[actuator_ids] = q_target

    for point in interpolate_qpos(q_start, q_target, max_step=0.012):
        # The passive viewer renders on another thread. Protect shared MuJoCo
        # state while changing qpos or forwarding the model, then sync outside
        # the lock so the viewer can consume the updated state.
        with viewer_data_lock():
            data.qpos[qpos_ids] = point
            mujoco.mj_forward(model, data)
            update_kinematic_attachment(model, data)
            mujoco.mj_forward(model, data)
        sync_viewer()

    with viewer_data_lock():
        data.qpos[qpos_ids] = q_target
        data.ctrl[actuator_ids] = q_target
        mujoco.mj_forward(model, data)
        update_kinematic_attachment(model, data)
        mujoco.mj_forward(model, data)
    sync_viewer()


JOINT_FINAL_TICKS = 1000


def move_to_pose(model, data, planning_data, arm_name: str, target_pos, target_rot, label: str):
    """Reach a Cartesian tool pose using IK and small live-state corrections."""
    qpos_ids, _, _ = arm_joint_indices(model, arm_name)
    site_id = get_name_id(model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"])
    target_pos = np.asarray(target_pos, dtype=float)
    target_rot = np.asarray(target_rot, dtype=float)

    for attempt in range(4):
        error = float(np.linalg.norm(data.site_xpos[site_id] - target_pos))
        if error <= 0.012:
            print(f"[motion] {label}: reached ({error * 1000:.1f} mm)")
            return
        solution, _, _ = solve_site_ik(
            model, planning_data, arm_name, target_pos, target_rot, data.qpos.copy()
        )
        if solution is None:
            break
        move_to_joint_target(
            model, data, arm_name, solution[qpos_ids].copy(),
            settle_ticks=60 if attempt == 0 else 80,
            final_ticks=JOINT_FINAL_TICKS,
        )

    error = float(np.linalg.norm(data.site_xpos[site_id] - target_pos))
    raise RuntimeError(f"Could not reach '{label}': position error {error:.3f} m.")


def execute_gob_pre_grasp_recovery(
    model, data, planning_data, arm_name, pre_grasp_pos, tool_rot, recovery_plan
):
    """Execute and verify the narrowly scoped grasp-failure GOB recovery."""

    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("GOB recovery requires a validated recovery plan.")

    if RecoveryStrategy.GOB not in recovery_plan.strategies:
        raise RuntimeError("Recovery plan does not contain the GOB strategy.")

    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one GOB recovery action.")

    if recovery_plan.actions[0].name != "return_to_pre_grasp":
        raise RuntimeError("Unexpected GOB recovery action.")

    if KINEMATIC_ATTACHMENT is not None:
        raise RuntimeError("Cannot execute grasp-failure GOB while an attachment is active.")

    canister_before, _ = object_pose(model, data, CANISTER_BODY)

    print("[recovery-execution] opening gripper before retreat")
    # set_gripper(model, data, arm_name, 0.0, ticks=220)
    
    for opening_command in (180.0, 140.0, 100.0, 60.0, 20.0, 0.0):
        print(
            f"[recovery-execution] gradual gripper release: "
            f"command={opening_command:.0f}"
        )
        set_gripper(
            model, data, arm_name, opening_command, ticks=75,
            diagnostic_body=CANISTER_BODY,
        )
        diagnose_canister_gripper_contacts(model, data, arm_name)

        canister_position, _ = object_pose(model, data, CANISTER_BODY)

        # Read the free-joint linear velocity to determine whether the
        # canister is moving after pad contact has disappeared.
        canister_body_id = get_name_id(
            model, mujoco.mjtObj.mjOBJ_BODY, CANISTER_BODY
        )
        canister_joint_id = -1
        for joint_id in range(model.njnt):
            if (
                int(model.jnt_bodyid[joint_id]) == canister_body_id
                and int(model.jnt_type[joint_id])
                == int(mujoco.mjtJoint.mjJNT_FREE)
            ):
                canister_joint_id = joint_id
                break

        if canister_joint_id < 0:
            raise RuntimeError("Canister free joint was not found.")

        velocity_address = int(model.jnt_dofadr[canister_joint_id])
        linear_velocity = data.qvel[
            velocity_address:velocity_address + 3
        ].copy()
        linear_speed = float(np.linalg.norm(linear_velocity))

        print(
            "[recovery-diagnostic] canister linear velocity: "
            f"{linear_speed:.5f} m/s "
            f"(x={linear_velocity[0]:.5f}, "
            f"y={linear_velocity[1]:.5f}, "
            f"z={linear_velocity[2]:.5f})"
        )

        displacement = float(
            np.linalg.norm(canister_position - canister_before)
        )

        print(
            "[recovery-diagnostic] displacement during release: "
            f"{displacement * 1000:.1f} mm"
        )

        if displacement > 0.005:
            raise RuntimeError(
                "GOB release stopped: canister displacement exceeded 5 mm "
                f"({displacement * 1000:.1f} mm)."
            )

    canister_after_open, _ = object_pose(model, data, CANISTER_BODY)
    opening_displacement = float(
        np.linalg.norm(canister_after_open - canister_before)
    )
    print(
        "[recovery-diagnostic] displacement after gripper opening: "
        f"{opening_displacement * 1000:.1f} mm"
    )

    print("[recovery-execution] returning to pre-grasp waypoint")
    move_to_pose(
        model, data, planning_data, arm_name,
        pre_grasp_pos, tool_rot, "GOB-return-to-pre-grasp"
    )

    site_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"]
    )
    position_error = float(
        np.linalg.norm(data.site_xpos[site_id] - np.asarray(pre_grasp_pos))
    )

    canister_after, _ = object_pose(model, data, CANISTER_BODY)
    retreat_displacement = float(
        np.linalg.norm(canister_after - canister_after_open)
    )
    canister_displacement = float(np.linalg.norm(canister_after - canister_before))

    print(
        "[recovery-diagnostic] additional displacement during retreat: "
        f"{retreat_displacement * 1000:.1f} mm"
    )

    if KINEMATIC_ATTACHMENT is not None:
        raise RuntimeError("GOB postcondition failed: attachment is active.")

    if position_error > 0.012:
        raise RuntimeError(
            f"GOB postcondition failed: pre-grasp position error "
            f"{position_error:.4f} m."
        )

    if canister_displacement > 0.005:
        raise RuntimeError(
            f"GOB postcondition failed: canister moved "
            f"{canister_displacement:.4f} m."
        )

    print("[recovery-execution] postconditions: PASS")
    print(f"[recovery-execution] pre-grasp position error: {position_error * 1000:.1f} mm")
    print(f"[recovery-execution] canister displacement: {canister_displacement * 1000:.1f} mm")
    print("[recovery-execution] kinematic attachment: inactive")

    return True




def execute_ina_pregrasp_recovery(
    model, data, planning_data, arm_name, pre_grasp_pos, tool_rot,
    recovery_plan,
):
    """Execute a validated INA by adding a raised staging waypoint."""

    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("INA recovery requires a validated recovery plan.")

    if RecoveryStrategy.INA not in recovery_plan.strategies:
        raise RuntimeError("Recovery plan does not contain the INA strategy.")

    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one INA recovery action.")

    if recovery_plan.actions[0].name != "replan_pre_grasp":
        raise RuntimeError("Unexpected INA recovery action.")

    if KINEMATIC_ATTACHMENT is not None:
        raise RuntimeError(
            "Pre-grasp INA must not run while a canister attachment is active."
        )

    site_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"]
    )
    start_position = data.site_xpos[site_id].copy()
    canister_before, _ = object_pose(model, data, CANISTER_BODY)
    pre_grasp_pos = np.asarray(pre_grasp_pos, dtype=float)

    if start_position.shape != (3,) or pre_grasp_pos.shape != (3,):
        raise RuntimeError("INA approach positions must be 3D vectors.")
    if not np.all(np.isfinite(start_position)) or not np.all(np.isfinite(pre_grasp_pos)):
        raise RuntimeError("INA approach positions must be finite.")

    # Add a raised staging waypoint above the straight-line approach.
    # This is a deterministic task-level alternate approach, not a full
    # collision-free motion planner.
    staging_pos = 0.5 * (start_position + pre_grasp_pos)
    staging_pos[2] = max(start_position[2], pre_grasp_pos[2]) + 0.08

    print("[recovery-execution] INA adding raised pre-grasp staging waypoint")
    move_to_pose(
        model, data, planning_data, arm_name,
        staging_pos, tool_rot, "INA-raised-staging-waypoint"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        pre_grasp_pos, tool_rot, "INA-replanned-pre-grasp"
    )

    staging_error = float(
        np.linalg.norm(data.site_xpos[site_id] - pre_grasp_pos)
    )
    canister_after, _ = object_pose(model, data, CANISTER_BODY)
    canister_displacement = float(
        np.linalg.norm(canister_after - canister_before)
    )

    if staging_error > 0.012:
        raise RuntimeError(
            "INA postcondition failed: replanned pre-grasp position error "
            f"{staging_error:.4f} m."
        )
    if KINEMATIC_ATTACHMENT is not None:
        raise RuntimeError("INA postcondition failed: unexpected attachment is active.")
    if canister_displacement > 0.005:
        raise RuntimeError(
            "INA postcondition failed: canister moved "
            f"{canister_displacement:.4f} m."
        )

    print("[recovery-execution] INA postconditions: PASS")
    print(
        "[recovery-execution] replanned pre-grasp position error: "
        f"{staging_error * 1000:.1f} mm"
    )
    print(
        "[recovery-execution] canister displacement: "
        f"{canister_displacement * 1000:.1f} mm"
    )
    print("[recovery-execution] kinematic attachment: inactive")

    return True


def execute_gbb_transfer_recovery(
    model, data, planning_data, arm_name, last_safe_waypoint, carry_rot,
    recovery_plan,
):
    """Return an attached canister to the last safe transfer waypoint."""

    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("GBB recovery requires a validated recovery plan.")

    if RecoveryStrategy.GBB not in recovery_plan.strategies:
        raise RuntimeError("Recovery plan does not contain the GBB strategy.")

    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one GBB recovery action.")

    if recovery_plan.actions[0].name != "return_to_last_safe_waypoint":
        raise RuntimeError("Unexpected GBB recovery action.")

    if KINEMATIC_ATTACHMENT is None:
        raise RuntimeError(
            "Cannot execute transfer-stage GBB without an active canister attachment."
        )

    waypoint = np.asarray(last_safe_waypoint, dtype=float)
    if waypoint.shape != (3,) or not np.all(np.isfinite(waypoint)):
        raise RuntimeError("GBB last-safe waypoint must be a finite 3D position.")

    attachment = KINEMATIC_ATTACHMENT
    mount_id = int(attachment["mount_id"])
    object_id = int(attachment["object_id"])
    site_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"]
    )

    canister_before, _ = object_pose(model, data, CANISTER_BODY)

    print(
        "[recovery-execution] GBB returning attached canister "
        "to last safe transfer waypoint"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        waypoint, carry_rot, "GBB-last-safe-transfer-waypoint"
    )

    waypoint_error = float(
        np.linalg.norm(data.site_xpos[site_id] - waypoint)
    )
    canister_after, _ = object_pose(model, data, CANISTER_BODY)

    # Verify that the simulator-side attachment is still active and that
    # the canister pose agrees with the stored mount-relative attachment.
    if KINEMATIC_ATTACHMENT is None:
        raise RuntimeError("GBB postcondition failed: attachment was lost.")

    mount_pos = data.xpos[mount_id].copy()
    mount_rot = data.xmat[mount_id].reshape(3, 3).copy()
    expected_canister_pos = (
        mount_pos + mount_rot @ np.asarray(attachment["rel_pos"], dtype=float)
    )
    attachment_error = float(
        np.linalg.norm(data.xpos[object_id] - expected_canister_pos)
    )
    canister_motion = float(
        np.linalg.norm(canister_after - canister_before)
    )

    if waypoint_error > 0.012:
        raise RuntimeError(
            "GBB postcondition failed: last-safe waypoint error "
            f"{waypoint_error:.4f} m."
        )

    if attachment_error > 0.005:
        raise RuntimeError(
            "GBB postcondition failed: canister is not following the "
            f"kinematic attachment (error={attachment_error:.4f} m)."
        )

    print("[recovery-execution] GBB postconditions: PASS")
    print(
        "[recovery-execution] last-safe waypoint position error: "
        f"{waypoint_error * 1000:.1f} mm"
    )
    print(
        "[recovery-execution] canister movement during return: "
        f"{canister_motion * 1000:.1f} mm"
    )
    print(
        "[recovery-execution] mount-relative attachment error: "
        f"{attachment_error * 1000:.2f} mm"
    )
    print("[recovery-execution] kinematic attachment: active")

    return True


def choose_arm_and_plan(model, data, planning_data, object_center, target_center, approach_height):
    """Check that the fixed left arm can reach the pick-and-place waypoints."""
    plans = []
    errors = []
    arm_candidates = (ARM_TO_USE,)
    for arm_name in arm_candidates:
        spec = ARM_SPECS[arm_name]
        site_id = get_name_id(model, mujoco.mjtObj.mjOBJ_SITE, spec["site"])
        current_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        qpos_ids, _, _ = arm_joint_indices(model, arm_name)
        qseed = data.qpos.copy()
        qsolutions = []
        max_pe = 0.0
        max_re = 0.0
        feasible = True
        # Pre-grasp, grasp, lift, pre-insert and insert.  IK is chained from the
        # preceding waypoint so the resulting joint path stays on one branch.
        waypoints = [
            np.asarray(object_center) + np.array([0.0, 0.0, approach_height]),
            np.asarray(object_center),
            np.asarray(object_center) + np.array([0.0, 0.0, approach_height]),
            np.asarray(target_center) + np.array([0.0, 0.0, approach_height]),
            np.asarray(target_center),
        ]
        for i, pos in enumerate(waypoints):
            sol, pe, re = solve_site_ik(model, planning_data, arm_name, pos, current_rot, qseed)
            max_pe, max_re = max(max_pe, pe), max(max_re, re)
            if sol is None:
                feasible = False
                errors.append(f"{arm_name}: waypoint {i+1} residual {pe:.3f}m/{re:.3f}rad")
                break
            qsolutions.append(sol[qpos_ids].copy())
            qseed = sol
        if feasible:
            all_q = [data.qpos[qpos_ids].copy()] + qsolutions
            travel = sum(float(np.linalg.norm(b - a)) for a, b in zip(all_q[:-1], all_q[1:]))
            plans.append(ArmPlan(arm_name, qsolutions, travel, max_pe, max_re))
    if not plans:
        raise RuntimeError("The left arm cannot reach all pick-and-place waypoints: " + "; ".join(errors))
    return min(plans, key=lambda p: p.total_joint_travel)



def diagnose_canister_gripper_contacts(model, data, arm_name: str) -> int:
    """Report active contacts between this arm's pads and any canister geom.

    This is read-only instrumentation: it does not modify contact, force,
    collision, or simulation state.
    """
    canister_body_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_BODY, CANISTER_BODY
    )
    canister_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == canister_body_id
    }

    pad_names = (
        f"{arm_name}_right_pad1",
        f"{arm_name}_right_pad2",
        f"{arm_name}_left_pad1",
        f"{arm_name}_left_pad2",
    )
    pad_geom_ids = {
        get_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in pad_names
    }

    contact_count = 0
    environment_contact_count = 0

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        # Keep track of every contact involving the canister, not only
        # gripper-pad contacts. This helps identify tray, floor, pump, or
        # other scene contacts that may redirect the canister after release.
        if geom1 in canister_geom_ids:
            canister_geom_id = geom1
            other_geom_id = geom2
        elif geom2 in canister_geom_ids:
            canister_geom_id = geom2
            other_geom_id = geom1
        else:
            continue

        is_pad_contact = other_geom_id in pad_geom_ids

        normal_force = None
        if int(contact.efc_address) >= 0:
            contact_force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, contact_force)
            normal_force = float(contact_force[0])

        canister_geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, canister_geom_id
        )
        other_geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id
        )
        force_text = (
            f"{normal_force:.4f} N"
            if normal_force is not None
            else "not in constraint solve"
        )
        contact_type = "pad" if is_pad_contact else "environment"

        print(
            f"[contact-diagnostic] {contact_type}: "
            f"{other_geom_name} <-> {canister_geom_name} "
            f"distance={float(contact.dist) * 1000:.2f} mm "
            f"normal_force={force_text}"
        )

        if is_pad_contact:
            contact_count += 1
        else:
            environment_contact_count += 1

    print(
        "[contact-diagnostic] canister-pad contact count: "
        f"{contact_count}"
    )
    print(
        "[contact-diagnostic] canister-environment contact count: "
        f"{environment_contact_count}"
    )
    return contact_count


def set_gripper(
    model,
    data,
    arm_name: str,
    value: float,
    ticks: int = 350,
    diagnostic_body: str = None,
):
    actuator_id = get_name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_SPECS[arm_name]["gripper"])
    value = float(np.clip(value, 0.0, 255.0))
    qpos_ids, qvel_ids, arm_actuator_ids = arm_joint_indices(model, arm_name)
    held_arm_qpos = data.qpos[qpos_ids].copy()
    held_arm_qvel = np.zeros_like(data.qvel[qvel_ids])

    diagnostic_joint_address = None
    diagnostic_max_speed = 0.0
    diagnostic_max_speed_step = -1
    diagnostic_velocity_at_peak = None
    diagnostic_contacts_at_peak = []
    diagnostic_start_position = None
    if diagnostic_body is not None:
        diagnostic_body_id = get_name_id(
            model, mujoco.mjtObj.mjOBJ_BODY, diagnostic_body
        )
        diagnostic_joint_id = -1
        for joint_id in range(model.njnt):
            if (
                int(model.jnt_bodyid[joint_id]) == diagnostic_body_id
                and int(model.jnt_type[joint_id])
                == int(mujoco.mjtJoint.mjJNT_FREE)
            ):
                diagnostic_joint_id = joint_id
                break
        if diagnostic_joint_id < 0:
            raise RuntimeError(
                f"Free joint for diagnostic body '{diagnostic_body}' was not found."
            )
        diagnostic_joint_address = int(model.jnt_dofadr[diagnostic_joint_id])
        diagnostic_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == diagnostic_body_id
        }
        diagnostic_start_position, _ = object_pose(
            model, data, diagnostic_body
        )

    for i in range(ticks):
        data.ctrl[actuator_id] = value
        data.ctrl[arm_actuator_ids] = held_arm_qpos

        # Hold both arm position and velocity while operating the gripper.
        # Position-only resetting leaves residual joint velocity in MuJoCo;
        # that velocity can move the wrist/pads during the next physics step
        # and inject lateral motion into an object being released.
        data.qvel[qvel_ids] = held_arm_qvel
        mujoco.mj_step(model, data)
        data.qpos[qpos_ids] = held_arm_qpos
        data.qvel[qvel_ids] = held_arm_qvel
        mujoco.mj_forward(model, data)

        if diagnostic_joint_address is not None:
            linear_velocity = data.qvel[
                diagnostic_joint_address:diagnostic_joint_address + 3
            ].copy()
            speed = float(np.linalg.norm(linear_velocity))

            if speed > diagnostic_max_speed:
                diagnostic_max_speed = speed
                diagnostic_max_speed_step = i
                diagnostic_velocity_at_peak = linear_velocity.copy()
                peak_contacts = []

                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    geom1 = int(contact.geom1)
                    geom2 = int(contact.geom2)

                    if geom1 in diagnostic_geom_ids:
                        other_geom_id = geom2
                        canister_geom_id = geom1
                    elif geom2 in diagnostic_geom_ids:
                        other_geom_id = geom1
                        canister_geom_id = geom2
                    else:
                        continue

                    other_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id
                    ) or f"geom_{other_geom_id}"
                    canister_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, canister_geom_id
                    ) or f"geom_{canister_geom_id}"

                    normal_force = None
                    if int(contact.efc_address) >= 0:
                        force = np.zeros(6, dtype=float)
                        mujoco.mj_contactForce(
                            model, data, contact_id, force
                        )
                        normal_force = float(force[0])

                    peak_contacts.append(
                        (
                            other_name,
                            canister_name,
                            float(contact.dist),
                            normal_force,
                        )
                    )

                diagnostic_contacts_at_peak = peak_contacts

        if i % 40 == 0:
            sync_viewer()

    if diagnostic_joint_address is not None:
        final_position, _ = object_pose(model, data, diagnostic_body)
        final_velocity = data.qvel[
            diagnostic_joint_address:diagnostic_joint_address + 3
        ].copy()
        interval_displacement = float(
            np.linalg.norm(final_position - diagnostic_start_position)
        )
        print(
            "[recovery-diagnostic] per-step canister motion during "
            f"{ticks} release steps: max_speed={diagnostic_max_speed:.5f} m/s "
            f"at step={diagnostic_max_speed_step}, "
            f"final_speed={np.linalg.norm(final_velocity):.5f} m/s, "
            f"interval_displacement={interval_displacement * 1000:.2f} mm"
        )

        if diagnostic_velocity_at_peak is not None:
            print(
                "[recovery-diagnostic] velocity at peak: "
                f"x={diagnostic_velocity_at_peak[0]:.5f}, "
                f"y={diagnostic_velocity_at_peak[1]:.5f}, "
                f"z={diagnostic_velocity_at_peak[2]:.5f} m/s"
            )

        print(
            "[recovery-diagnostic] contacts at peak speed: "
            f"{len(diagnostic_contacts_at_peak)}"
        )
        for other_name, canister_name, distance, normal_force in diagnostic_contacts_at_peak:
            force_text = (
                f"{normal_force:.4f} N"
                if normal_force is not None
                else "not in constraint solve"
            )
            print(
                f"[recovery-diagnostic] peak-contact {other_name} <-> "
                f"{canister_name}: distance={distance * 1000:.2f} mm, "
                f"normal_force={force_text}"
            )



# In this kinematic demonstration the arm is moved by direct joint-coordinate
# updates. MuJoCo weld constraints are force constraints and do not automatically
# reposition a free body when qpos is edited followed only by mj_forward().
# Therefore the demo attachment explicitly carries the canister's free-joint
# pose with the gripper mount. This is a simulator-only kinematic grasp surrogate.
KINEMATIC_ATTACHMENT = None


def activate_kinematic_attachment(model, data, arm_name: str,
                                  object_body_name: Optional[str] = None) -> bool:
    global KINEMATIC_ATTACHMENT
    # Resolve the active object at call time. A default argument equal to
    # CANISTER_BODY would capture the original name ("canister1") when this
    # function is defined, before an INO substitution updates that global.
    if object_body_name is None:
        object_body_name = CANISTER_BODY
    mount_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_BODY, ARM_SPECS[arm_name]["mount_body"]
    )
    object_id = get_name_id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)

    # The canister must have a free joint so its world pose can be updated.
    free_joint_id = -1
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) == object_id and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_joint_id = jid
            break
    if free_joint_id < 0:
        raise RuntimeError(
            f"{object_body_name} has no free joint; cannot use the kinematic demo attachment"
        )

    mount_pos = data.xpos[mount_id].copy()
    mount_rot = data.xmat[mount_id].reshape(3, 3).copy()
    object_pos = data.xpos[object_id].copy()
    object_rot = data.xmat[object_id].reshape(3, 3).copy()

    rel_pos = mount_rot.T @ (object_pos - mount_pos)
    rel_rot = mount_rot.T @ object_rot
    rel_quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(-1))

    KINEMATIC_ATTACHMENT = {
        "mount_id": mount_id,
        "object_id": object_id,
        "qposadr": int(model.jnt_qposadr[free_joint_id]),
        "rel_pos": rel_pos,
        "rel_quat": rel_quat,
    }
    print(
        "[grasp] activated kinematic simulator attachment for "
        f"{object_body_name} from measured grasp pose"
    )
    return True


def update_kinematic_attachment(model, data):
    attachment = KINEMATIC_ATTACHMENT
    if attachment is None:
        return
    mount_id = attachment["mount_id"]
    mount_pos = data.xpos[mount_id].copy()
    mount_rot = data.xmat[mount_id].reshape(3, 3).copy()

    world_pos = mount_pos + mount_rot @ attachment["rel_pos"]
    mount_quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(mount_quat, mount_rot.reshape(-1))
    world_quat = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(world_quat, mount_quat, attachment["rel_quat"])

    adr = attachment["qposadr"]
    data.qpos[adr:adr + 3] = world_pos
    data.qpos[adr + 3:adr + 7] = world_quat


def deactivate_kinematic_attachment():
    global KINEMATIC_ATTACHMENT
    KINEMATIC_ATTACHMENT = None


def object_pose(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = get_name_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def free_joint_qpos_address(model, body_name: str) -> int:
    """Return the qpos address of a body's free joint."""
    body_id = get_name_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for joint_id in range(model.njnt):
        if (
            int(model.jnt_bodyid[joint_id]) == body_id
            and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            return int(model.jnt_qposadr[joint_id])
    raise RuntimeError(f"Body '{body_name}' has no free joint.")





def execute_gob_placement_recovery(
    model, data, planning_data, arm_name, safe_placement_waypoint,
    carry_rot, recovery_plan,
):
    """Execute a validated GOB return to a safe placement waypoint.

    The canister remains attached while the arm returns to the saved waypoint.
    This is a task-level waypoint recovery, not a collision-checked planner.
    """
    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("Placement GOB requires a validated recovery plan.")
    if recovery_plan.strategies != [RecoveryStrategy.GOB]:
        raise RuntimeError("Placement recovery must contain only the GOB strategy.")
    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one placement GOB action.")
    if recovery_plan.actions[0].name != "return_to_placement_waypoint":
        raise RuntimeError(
            "Unexpected placement GOB action "
            f"'{recovery_plan.actions[0].name}'."
        )
    if KINEMATIC_ATTACHMENT is None:
        raise RuntimeError(
            "Placement GOB requires the canister to remain attached."
        )

    waypoint = np.asarray(safe_placement_waypoint, dtype=float)
    carry_rot = np.asarray(carry_rot, dtype=float)
    if waypoint.shape != (3,) or not np.all(np.isfinite(waypoint)):
        raise RuntimeError("Safe placement waypoint must be a finite 3D position.")
    if carry_rot.shape != (3, 3) or not np.all(np.isfinite(carry_rot)):
        raise RuntimeError("Carry orientation must be a finite 3x3 rotation matrix.")

    attachment = KINEMATIC_ATTACHMENT
    object_id = int(attachment["object_id"])
    mount_id = int(attachment["mount_id"])
    site_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"]
    )

    object_start = data.xpos[object_id].copy()
    mount_pos = data.xpos[mount_id].copy()
    mount_rot = data.xmat[mount_id].reshape(3, 3).copy()
    rel_start = mount_rot.T @ (object_start - mount_pos)

    start_tool = data.site_xpos[site_id].copy()
    print(
        "[recovery-execution] GOB placement returning to saved safe waypoint "
        f"{np.round(waypoint, 4).tolist()}"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        waypoint, carry_rot, "GOB-safe-placement-waypoint"
    )

    final_tool = data.site_xpos[site_id].copy()
    waypoint_error = float(np.linalg.norm(final_tool - waypoint))
    object_end = data.xpos[object_id].copy()
    object_motion = float(np.linalg.norm(object_end - object_start))
    mount_pos_end = data.xpos[mount_id].copy()
    mount_rot_end = data.xmat[mount_id].reshape(3, 3).copy()
    rel_end = mount_rot_end.T @ (object_end - mount_pos_end)
    attachment_error = float(np.linalg.norm(rel_end - rel_start))

    if KINEMATIC_ATTACHMENT is None:
        raise RuntimeError("Placement GOB failed: attachment was lost.")
    if waypoint_error > 0.012:
        raise RuntimeError(
            f"Placement GOB waypoint error {waypoint_error*1000:.1f} mm exceeds 12 mm."
        )
    if attachment_error > 0.005:
        raise RuntimeError(
            "Placement GOB attachment error "
            f"{attachment_error*1000:.2f} mm exceeds 5 mm."
        )
    if object_motion < 0.010:
        raise RuntimeError(
            "Placement GOB failed: canister did not move with the robot."
        )

    print("[recovery-execution] GOB placement postconditions: PASS")
    print(
        f"[recovery-execution] safe-waypoint error: {waypoint_error*1000:.1f} mm"
    )
    print(
        f"[recovery-execution] canister movement during return: "
        f"{object_motion*1000:.1f} mm"
    )
    print(
        f"[recovery-execution] mount-relative attachment error: "
        f"{attachment_error*1000:.2f} mm"
    )
    print("[recovery-execution] canister attachment: active")
    print(
        "[recovery-execution] GOB placement recovery: SUCCEEDED "
        "(saved waypoint return; not collision-checked planning)"
    )
    return True


def execute_ina_insertion_recovery(
    model, data, planning_data, arm_name, target_center, carry_rot,
    recovery_plan, approach_height=0.12,
):
    """Execute a validated INA insertion re-plan while retaining the grasp.

    Adds a raised lateral-clearance waypoint, returns to a centered approach
    pose above the opening, then performs the final insertion. This is a
    deterministic task-level alternate approach, not collision-checked motion
    planning or a guarantee of clearance from every scene object.
    """
    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("Insertion INA requires a validated recovery plan.")
    if recovery_plan.strategies != [RecoveryStrategy.INA]:
        raise RuntimeError("Insertion recovery must contain only the INA strategy.")
    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one insertion INA action.")
    if recovery_plan.actions[0].name != "replan_insertion":
        raise RuntimeError(
            f"Unexpected INA action '{recovery_plan.actions[0].name}'."
        )
    if KINEMATIC_ATTACHMENT is None:
        raise RuntimeError(
            "Insertion INA requires the canister to remain attached to the gripper."
        )

    target_center = np.asarray(target_center, dtype=float)
    carry_rot = np.asarray(carry_rot, dtype=float)
    if target_center.shape != (3,) or not np.all(np.isfinite(target_center)):
        raise RuntimeError("Insertion target must be a finite 3D position.")
    if carry_rot.shape != (3, 3) or not np.all(np.isfinite(carry_rot)):
        raise RuntimeError("Carry orientation must be a finite 3x3 rotation matrix.")

    attachment = KINEMATIC_ATTACHMENT
    object_id = int(attachment["object_id"])
    mount_id = int(attachment["mount_id"])
    site_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"]
    )

    def attachment_error_now():
        mount_pos = data.xpos[mount_id].copy()
        mount_rot = data.xmat[mount_id].reshape(3, 3).copy()
        expected = (
            mount_pos
            + mount_rot @ np.asarray(attachment["rel_pos"], dtype=float)
        )
        return float(np.linalg.norm(data.xpos[object_id] - expected))

    object_start = data.xpos[object_id].copy()
    initial_attachment_error = attachment_error_now()
    if initial_attachment_error > 0.005:
        raise RuntimeError(
            "Insertion INA stopped: canister was already outside its attachment "
            f"tolerance ({initial_attachment_error*1000:.2f} mm)."
        )

    current_tool = data.site_xpos[site_id].copy()
    centered_approach = target_center + np.array(
        [0.0, 0.0, max(float(approach_height), 0.10)], dtype=float
    )

    # Choose a modest horizontal offset relative to the target. The raised
    # waypoint creates a distinct approach sequence without changing the
    # target opening or releasing the canister.
    lateral_offset = np.array([0.035, 0.0, 0.0], dtype=float)
    clearance_waypoint = current_tool.copy()
    clearance_waypoint[2] = max(
        current_tool[2], centered_approach[2]
    ) + 0.035
    offset_waypoint = clearance_waypoint + lateral_offset

    print(
        "[recovery-execution] INA replanning insertion with raised "
        "lateral-clearance approach"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        clearance_waypoint, carry_rot, "INA-insertion-raised-clearance"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        offset_waypoint, carry_rot, "INA-insertion-lateral-waypoint"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        centered_approach, carry_rot, "INA-insertion-centered-approach"
    )
    move_to_pose(
        model, data, planning_data, arm_name,
        target_center, carry_rot, "INA-replanned-insertion"
    )

    tool_error = float(
        np.linalg.norm(data.site_xpos[site_id] - target_center)
    )
    object_end = data.xpos[object_id].copy()
    object_motion = float(np.linalg.norm(object_end - object_start))
    final_attachment_error = attachment_error_now()

    if KINEMATIC_ATTACHMENT is None:
        raise RuntimeError("Insertion INA postcondition failed: attachment was lost.")
    if tool_error > 0.012:
        raise RuntimeError(
            "Insertion INA postcondition failed: insertion pose error "
            f"{tool_error*1000:.1f} mm."
        )
    if final_attachment_error > 0.005:
        raise RuntimeError(
            "Insertion INA postcondition failed: mount-relative attachment error "
            f"{final_attachment_error*1000:.2f} mm."
        )
    if object_motion < 0.020:
        raise RuntimeError(
            "Insertion INA postcondition failed: canister did not move with "
            "the insertion approach."
        )

    print("[recovery-execution] INA insertion postconditions: PASS")
    print(
        "[recovery-execution] insertion tool-position error: "
        f"{tool_error*1000:.1f} mm"
    )
    print(
        "[recovery-execution] canister motion during approach: "
        f"{object_motion*1000:.1f} mm"
    )
    print(
        "[recovery-execution] mount-relative attachment error: "
        f"{final_attachment_error*1000:.2f} mm"
    )
    print("[recovery-execution] canister attachment: active")
    print(
        "[recovery-execution] INA insertion recovery: SUCCEEDED "
        "(deterministic alternate approach; not collision-checked planning)"
    )
    return True


def execute_ess_stop_recovery(model, data, arm_name, recovery_plan,
                              hold_ticks: int = 100):
    """Execute a bounded ESS stop and verify the robot/object remain stationary.

    This is a simulator-level controlled hold, not a certified emergency stop
    and not a real-robot safety controller. The kinematic attachment is retained
    so a carried canister is not intentionally released during the stop.
    """
    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("ESS recovery requires a validated recovery plan.")
    if recovery_plan.strategies != [RecoveryStrategy.ESS]:
        raise RuntimeError("ESS recovery must contain only the ESS strategy.")
    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one ESS recovery action.")

    action = recovery_plan.actions[0]
    if action.name != "stop_robot":
        raise RuntimeError(
            f"Unexpected ESS action '{action.name}'; expected 'stop_robot'."
        )
    risk = action.estimated_risk
    if (risk is None or isinstance(risk, bool)
            or not isinstance(risk, (int, float))
            or not 0.0 <= float(risk) <= 0.7):
        raise RuntimeError("ESS stop action has missing or unacceptable risk.")

    qpos_ids, qvel_ids, actuator_ids = arm_joint_indices(model, arm_name)
    start_q = data.qpos[qpos_ids].copy()
    start_site_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"]
    )
    start_tool = data.site_xpos[start_site_id].copy()

    attachment = KINEMATIC_ATTACHMENT
    attached_object_start = None
    attached_rel_start = None
    if attachment is not None:
        object_id = attachment["object_id"]
        mount_id = attachment["mount_id"]
        attached_object_start = data.xpos[object_id].copy()
        mount_pos = data.xpos[mount_id].copy()
        mount_rot = data.xmat[mount_id].reshape(3, 3).copy()
        attached_rel_start = mount_rot.T @ (attached_object_start - mount_pos)

    # Hold the current joint configuration and clear arm joint velocities.
    data.ctrl[actuator_ids] = start_q
    data.qvel[qvel_ids] = 0.0
    mujoco.mj_forward(model, data)
    update_kinematic_attachment(model, data)
    mujoco.mj_forward(model, data)

    print(
        f"[recovery-execution] ESS controlled stop requested for {arm_name} arm; "
        f"holding current configuration for {hold_ticks} simulation steps"
    )

    for _ in range(hold_ticks):
        data.ctrl[actuator_ids] = start_q
        data.qvel[qvel_ids] = 0.0
        mujoco.mj_step(model, data)
        # The demo moves joints kinematically; retain the simulator-side
        # attachment while confirming the arm has not been commanded elsewhere.
        update_kinematic_attachment(model, data)
        mujoco.mj_forward(model, data)
        sync_viewer()

    final_q = data.qpos[qpos_ids].copy()
    joint_drift = float(np.max(np.abs(final_q - start_q)))
    final_tool = data.site_xpos[start_site_id].copy()
    tool_drift = float(np.linalg.norm(final_tool - start_tool))

    if joint_drift > 0.005 or tool_drift > 0.005:
        raise RuntimeError(
            "ESS hold verification failed: "
            f"joint drift={joint_drift:.6f} rad, tool drift={tool_drift*1000:.2f} mm"
        )

    object_drift = None
    attachment_error = None
    if attachment is not None:
        object_id = attachment["object_id"]
        mount_id = attachment["mount_id"]
        final_object = data.xpos[object_id].copy()
        object_drift = float(np.linalg.norm(final_object - attached_object_start))
        mount_pos = data.xpos[mount_id].copy()
        mount_rot = data.xmat[mount_id].reshape(3, 3).copy()
        final_rel = mount_rot.T @ (final_object - mount_pos)
        attachment_error = float(np.linalg.norm(final_rel - attached_rel_start))
        if object_drift > 0.005 or attachment_error > 0.005:
            raise RuntimeError(
                "ESS object-hold verification failed: "
                f"object drift={object_drift*1000:.2f} mm, "
                f"mount-relative error={attachment_error*1000:.2f} mm"
            )

    print(
        "[recovery-execution] ESS safe-hold checks PASS: "
        f"joint drift={joint_drift:.6f} rad, tool drift={tool_drift*1000:.2f} mm"
    )
    if object_drift is not None:
        print(
            "[recovery-execution] carried-object hold PASS: "
            f"object drift={object_drift*1000:.2f} mm, "
            f"mount-relative error={attachment_error*1000:.2f} mm"
        )
    print(
        "[recovery-execution] ESS recovery: SUCCEEDED "
        "(simulated controlled hold; not a certified emergency stop)"
    )
    return True


def execute_ino_replacement_recovery(model, data, recovery_plan):
    """Activate a precompiled replacement canister for a validated INO plan.

    The original canister is moved out of the task workspace to represent an
    unavailable object. The replacement's free-joint pose is set to the
    original pose. This is a deterministic MuJoCo scene substitution, not
    physical object acquisition or grasp verification.
    """
    global CANISTER_BODY

    from recovery.models import PlanStatus, RecoveryStrategy

    if recovery_plan is None or recovery_plan.status != PlanStatus.VALIDATED:
        raise RuntimeError("INO recovery requires a validated recovery plan.")
    if RecoveryStrategy.INO not in recovery_plan.strategies:
        raise RuntimeError("Recovery plan does not contain the INO strategy.")
    if len(recovery_plan.actions) != 1:
        raise RuntimeError("Expected exactly one INO recovery action.")

    action = recovery_plan.actions[0]
    if action.name != "introduce_replacement_canister":
        raise RuntimeError(f"Unexpected INO action: {action.name}")

    expected_parameters = {
        "missing_object": "canister",
        "replacement_object": REPLACEMENT_CANISTER_BODY,
        "replacement_type": "canister",
    }
    if action.parameters != expected_parameters:
        raise RuntimeError(
            "INO action parameters do not match the configured replacement: "
            f"{action.parameters}"
        )

    if CANISTER_BODY != "canister1":
        raise RuntimeError(
            "INO replacement can only be activated once from the original canister."
        )

    original_body_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_BODY, CANISTER_BODY
    )
    replacement_body_id = get_name_id(
        model, mujoco.mjtObj.mjOBJ_BODY, REPLACEMENT_CANISTER_BODY
    )
    original_joint_id = next(
        joint_id for joint_id in range(model.njnt)
        if int(model.jnt_bodyid[joint_id]) == original_body_id
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    replacement_joint_id = next(
        joint_id for joint_id in range(model.njnt)
        if int(model.jnt_bodyid[joint_id]) == replacement_body_id
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    original_qpos_address = int(model.jnt_qposadr[original_joint_id])
    replacement_qpos_address = int(model.jnt_qposadr[replacement_joint_id])
    original_dof_address = int(model.jnt_dofadr[original_joint_id])
    replacement_dof_address = int(model.jnt_dofadr[replacement_joint_id])

    original_pose = data.qpos[
        original_qpos_address:original_qpos_address + 7
    ].copy()
    if not np.all(np.isfinite(original_pose)):
        raise RuntimeError("Original canister pose is not finite.")

    print("[recovery-execution] INO substituting unavailable canister")
    # Move the unavailable original out of the workspace.
    data.qpos[original_qpos_address:original_qpos_address + 3] = (
        original_pose[:3] + np.array([0.0, 0.0, -10.0])
    )
    data.qvel[original_dof_address:original_dof_address + 6] = 0.0

    # Introduce the replacement at the original task-object pose.
    data.qpos[
        replacement_qpos_address:replacement_qpos_address + 7
    ] = original_pose

    data.qvel[replacement_dof_address:replacement_dof_address + 6] = 0.0

    mujoco.mj_forward(model, data)

    replacement_position, _ = object_pose(
        model, data, REPLACEMENT_CANISTER_BODY
    )
    position_error = float(np.linalg.norm(replacement_position - original_pose[:3]))
    if position_error > 1e-6:
        raise RuntimeError(
            "INO postcondition failed: replacement pose differs from the "
            f"original canister pose by {position_error:.6f} m."
        )

    original_position, _ = object_pose(model, data, "canister1")
    if original_position[2] > -5.0:
        raise RuntimeError(
            "INO postcondition failed: unavailable original canister "
            "was not removed from the task workspace."
        )

    # All downstream perception, planning, attachment, and verification
    # functions use this active object name.
    CANISTER_BODY = REPLACEMENT_CANISTER_BODY

    print("[recovery-execution] replacement body: present")
    print(
        "[recovery-execution] replacement pose error: "
        f"{position_error * 1000:.3f} mm"
    )
    print("[recovery-execution] original canister: moved out of workspace")
    print("[recovery-execution] active task object: replacement_canister")
    print("[recovery-execution] INO recovery: SUCCEEDED")
    return True


def save_observation(path: Path, rgb: np.ndarray):
    try:
        import imageio.v3 as iio
        iio.imwrite(path, rgb)
        return
    except Exception:
        pass
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except Exception as exc:
        print(f"[warning] Could not save perception image {path}: {exc}")


def maybe_open_viewer(model, data, enabled: bool):
    if not enabled:
        return None
    try:
        return mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        print(f"[warning] Could not open MuJoCo viewer: {exc}")
        return None


def execute_recovery_via_backend(
    failure,
    outcome,
    model,
    data,
    **runtime_state,
):
    """Dispatch a validated recovery plan through the MuJoCo backend.

    Runtime state is passed at the failure site so callbacks use live task
    waypoints and orientations rather than stale pre-failure snapshots.
    """
    recovery_functions = {
        "gob_pre_grasp": execute_gob_pre_grasp_recovery,
        "ina_pregrasp": execute_ina_pregrasp_recovery,
        "gbb_transfer": execute_gbb_transfer_recovery,
        "gob_placement": execute_gob_placement_recovery,
        "ina_insertion": execute_ina_insertion_recovery,
        "ess_stop": execute_ess_stop_recovery,
        "ino_replacement": execute_ino_replacement_recovery,
    }
    callbacks = make_mujoco_recovery_callbacks(
        model=model,
        data=data,
        recovery_functions=recovery_functions,
        **runtime_state,
    )
    backend = build_mujoco_recovery_backend(callbacks)
    context = build_failure_context(failure)

    def verify_simulation_postconditions(recovery_plan, recovery_context):
        # Each recovery callback performs its action-specific checks. This
        # shared check adds a final live-state sanity check before success is
        # reported by the orchestration layer.
        if not (
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.xpos))
        ):
            return False

        attachment = KINEMATIC_ATTACHMENT
        if attachment is not None:
            mount_id = int(attachment["mount_id"])
            object_id = int(attachment["object_id"])
            mount_pos = np.asarray(data.xpos[mount_id], dtype=float)
            mount_rot = np.asarray(data.xmat[mount_id], dtype=float).reshape(3, 3)
            expected_object_pos = (
                mount_pos
                + mount_rot @ np.asarray(attachment["rel_pos"], dtype=float)
            )
            actual_object_pos = np.asarray(data.xpos[object_id], dtype=float)
            if np.linalg.norm(actual_object_pos - expected_object_pos) > 0.005:
                return False
        return True

    coordinator = RecoveryExecutionCoordinator()
    result = coordinator.execute(
        outcome.plan,
        context,
        backend,
        postcondition_checker=verify_simulation_postconditions,
        max_retries=0,
    )

    print(f"[recovery-backend] backend=simulation")
    print(f"[recovery-backend] status={result.status.value}")
    print(f"[recovery-backend] attempts={result.attempts}")
    print(f"[recovery-backend] postconditions_verified={result.postconditions_verified}")
    print(f"[recovery-backend] message={result.message}")
    for execution in result.execution_results:
        if execution.executed_actions:
            print(
                "[recovery-backend] executed actions: "
                + ", ".join(execution.executed_actions)
            )
    if result.status.value != "SUCCEEDED":
        raise RuntimeError(
            "Recovery orchestration did not complete safely: " + result.message
        )
    return result


def apply_initial_state_perturbation(
    model,
    data,
    mujoco_module,
    *,
    seed: int,
    canister_xy_m: float = 0.0,
    canister_yaw_rad: float = 0.0,
    joint_position_rad: float = 0.0,
    joint_velocity_rad_s: float = 0.0,
):
    """Apply bounded, seeded perturbations before perception and task planning.

    Perturbations are deliberately small and applied to the selected left arm
    and canister free joint only. This is an experimental simulation fixture,
    not a model of calibrated sensor or actuator noise.
    """
    rng = np.random.default_rng(seed)
    canister_body_id = named_id(
        model, mujoco_module.mjtObj.mjOBJ_BODY, CANISTER_BODY
    )
    free_joint_id = next(
        (
            joint_id for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) == canister_body_id
            and int(model.jnt_type[joint_id]) == int(mujoco_module.mjtJoint.mjJNT_FREE)
        ),
        -1,
    )
    if free_joint_id < 0:
        raise RuntimeError("Canister free joint was not found; cannot perturb pose.")
    qpos_address = int(model.jnt_qposadr[free_joint_id])
    initial_canister = data.qpos[qpos_address:qpos_address + 7].copy()
    xy_delta = rng.uniform(-canister_xy_m, canister_xy_m, size=2)
    yaw_delta = float(rng.uniform(-canister_yaw_rad, canister_yaw_rad))
    data.qpos[qpos_address:qpos_address + 2] += xy_delta
    # MuJoCo free-joint quaternion is w,x,y,z. Compose a world-z yaw offset.
    half_yaw = yaw_delta / 2.0
    yaw_quat = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
    q = data.qpos[qpos_address + 3:qpos_address + 7].copy()
    w1, x1, y1, z1 = yaw_quat
    w2, x2, y2, z2 = q
    data.qpos[qpos_address + 3:qpos_address + 7] = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    arm_qpos_ids, arm_dof_ids, _ = arm_joint_indices(model, ARM_TO_USE)
    applied_joint_delta = np.zeros(len(arm_qpos_ids))
    if joint_position_rad > 0:
        applied_joint_delta = rng.uniform(
            -joint_position_rad, joint_position_rad, size=len(arm_qpos_ids)
        )
        data.qpos[arm_qpos_ids] += applied_joint_delta
    applied_joint_velocity = np.zeros(len(arm_dof_ids))
    if joint_velocity_rad_s > 0:
        applied_joint_velocity = rng.uniform(
            -joint_velocity_rad_s, joint_velocity_rad_s, size=len(arm_dof_ids)
        )
        data.qvel[arm_dof_ids] = applied_joint_velocity
    mujoco_module.mj_forward(model, data)
    return {
        "seed": int(seed),
        "canister_xy_delta_m": xy_delta.tolist(),
        "canister_yaw_delta_rad": yaw_delta,
        "canister_initial_pose_before": initial_canister.tolist(),
        "joint_position_delta_rad": applied_joint_delta.tolist(),
        "joint_velocity_delta_rad_s": applied_joint_velocity.tolist(),
        "bounds": {
            "canister_xy_m": canister_xy_m,
            "canister_yaw_rad": canister_yaw_rad,
            "joint_position_rad": joint_position_rad,
            "joint_velocity_rad_s": joint_velocity_rad_s,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sim", "real"),
        default="sim",
        help=(
            "'sim' runs MuJoCo; 'real' is a fail-closed placeholder and "
            "currently sends no hardware commands."
        ),
    )
    parser.add_argument("--model", default=None, help="Path to Tracy mjmodel.xml (default: repository model)")
    parser.add_argument("--meshdir", default=None, help="Path to Tracy flattened_meshes directory")
    parser.add_argument("--hole", type=int, choices=(1, 2), default=1, help="Tray opening to use")
    parser.add_argument("--width", type=int, default=1280, help="RGB-D render width")
    parser.add_argument("--height", type=int, default=720, help="RGB-D render height")
    parser.add_argument("--approach-height", type=float, default=0.15, help="Safe vertical approach/lift distance (m)")
    parser.add_argument("--place-z-offset", type=float, default=DEFAULT_PLACE_Z_ABOVE_RING_TOP,
                        help="Canister centre height above perceived ring top (m); default calibrated for supplied MJCF")
    parser.add_argument("--no-viewer", action="store_true", help="Run without interactive MuJoCo viewer")
    parser.add_argument("--save-perception", default="canister_tray_perception.png", help="Save RGB image used for perception")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="Physics settling time after release")
    parser.add_argument("--perturb-seed", type=int, default=None,
                        help="Seed for bounded initial-state perturbations")
    parser.add_argument("--canister-xy-perturb-m", type=float, default=0.0,
                        help="Maximum absolute x/y canister offset in metres")
    parser.add_argument("--canister-yaw-perturb-deg", type=float, default=0.0,
                        help="Maximum absolute canister yaw offset in degrees")
    parser.add_argument("--joint-position-perturb-deg", type=float, default=0.0,
                        help="Maximum absolute selected-arm joint offset in degrees")
    parser.add_argument("--joint-velocity-perturb-rad-s", type=float, default=0.0,
                        help="Maximum absolute selected-arm joint velocity in rad/s")
    parser.add_argument(
        "--inject-failure",
        choices=FAILURE_STAGES,
        default=None,
        help="Inject one deterministic failure at the selected simulation stage",
    )
    parser.add_argument(
        "--recovery-selection-policy",
        choices=("confidence", "consensus"),
        default="confidence",
        help="Policy used to prioritize verifier suggestions",
    )
    parser.add_argument(
        "--recovery-candidate-count",
        type=int,
        default=1,
        help="Number of recovery candidates to generate; 1 preserves deterministic legacy mode",
    )
    parser.add_argument(
        "--recovery-sampling-temperature",
        type=float,
        default=0.15,
        help="Softmax sampling temperature when multiple candidates are enabled",
    )
    parser.add_argument(
        "--recovery-random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible probabilistic candidate selection",
    )
    args = parser.parse_args()

    perturb_values = (
        args.canister_xy_perturb_m,
        args.canister_yaw_perturb_deg,
        args.joint_position_perturb_deg,
        args.joint_velocity_perturb_rad_s,
    )
    if any(value < 0 for value in perturb_values):
        parser.error("Perturbation magnitudes must be non-negative")
    if args.canister_xy_perturb_m > 0.015:
        parser.error("Canister XY perturbation is limited to 0.015 m")
    if args.canister_yaw_perturb_deg > 10.0:
        parser.error("Canister yaw perturbation is limited to 10 degrees")
    if args.joint_position_perturb_deg > 2.0:
        parser.error("Joint position perturbation is limited to 2 degrees")
    if args.joint_velocity_perturb_rad_s > 0.5:
        parser.error("Joint velocity perturbation is limited to 0.5 rad/s")
    if args.recovery_candidate_count < 1:
        parser.error("--recovery-candidate-count must be at least 1")
    if args.recovery_sampling_temperature <= 0:
        parser.error("--recovery-sampling-temperature must be greater than 0")

    recovery_manager = build_simulation_recovery_manager(
        selection_policy=args.recovery_selection_policy,
        candidate_count=args.recovery_candidate_count,
        sampling_temperature=args.recovery_sampling_temperature,
        random_seed=args.recovery_random_seed,
    )
    print(
        "[recovery-config] "
        f"policy={args.recovery_selection_policy}, "
        f"candidate_count={args.recovery_candidate_count}, "
        f"temperature={args.recovery_sampling_temperature}, "
        f"seed={args.recovery_random_seed}"
    )

    if args.mode == "real":
        result = RealRobotRecoveryBackend().execute(None, None)
        print(f"[recovery-backend] backend={result.backend}")
        print(f"[recovery-backend] success={result.success}")
        print(f"[recovery-backend] {result.message}")
        print(
            "[mode] Real-robot mode is not implemented. "
            "No hardware commands were sent."
        )
        raise SystemExit(2)

    global mujoco
    try:
        import mujoco as _mujoco
        mujoco = _mujoco
        try:
            import mujoco.viewer
        except Exception:
            pass
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo's Python package is not installed in this Python environment.\\n"
            "Activate the repository's MuJoCo venv first, as described in README.md.\\n"
            "Example: source python/src/robauditor/src/tracy_ws/.venv_mujoco/bin/activate"
        ) from exc

    failure_injector = FailureInjector(
        enabled=args.inject_failure is not None,
        stage=args.inject_failure,
    )
    model_path, meshdir = resolve_model_and_meshdir(args)
    model = load_model_with_meshdir(model_path, meshdir)
    data = mujoco.MjData(model)
    planning_data = mujoco.MjData(model)
    reset_to_initial_pose(model, data)
    if any((
        args.canister_xy_perturb_m,
        args.canister_yaw_perturb_deg,
        args.joint_position_perturb_deg,
        args.joint_velocity_perturb_rad_s,
    )):
        perturbation = apply_initial_state_perturbation(
            model, data, mujoco,
            seed=0 if args.perturb_seed is None else args.perturb_seed,
            canister_xy_m=args.canister_xy_perturb_m,
            canister_yaw_rad=math.radians(args.canister_yaw_perturb_deg),
            joint_position_rad=math.radians(args.joint_position_perturb_deg),
            joint_velocity_rad_s=args.joint_velocity_perturb_rad_s,
        )
        print("[initial-perturbation] " + str(perturbation))
    else:
        print("[initial-perturbation] disabled; baseline initial state")

    # Check that the requested model really is the runtime model this script
    # expects; fail early rather than silently using another Tracy scene.
    for name, typ in [
        (CAMERA_NAME, mujoco.mjtObj.mjOBJ_CAMERA),
        (CANISTER_BODY, mujoco.mjtObj.mjOBJ_BODY),
        ("Pump1", mujoco.mjtObj.mjOBJ_BODY),
        ("Pump2", mujoco.mjtObj.mjOBJ_BODY),
        ("left_pinch", mujoco.mjtObj.mjOBJ_SITE),
    ]:
        get_name_id(model, typ, name)

    viewer = maybe_open_viewer(model, data, not args.no_viewer)
    global ACTIVE_VIEWER
    ACTIVE_VIEWER = viewer
    try:
        # Give the initial scene a short settle before the first camera capture.
        for _ in range(300):
            mujoco.mj_step(model, data)
            if viewer is not None and _ % 50 == 0:
                viewer.sync()

        # INO is handled locally so the task can continue with the replacement
        # object after the recovery plan is validated and executed.
        try:
            failure_injector.check("object_unavailable")
        except InjectedFailure as failure:
            print(f"\\n[failure] Simulation interrupted: {failure}")
            ino_outcome = process_injected_failure(failure, manager=recovery_manager, model=model, data=data, mujoco_module=mujoco)
            execute_recovery_via_backend(
                failure,
                ino_outcome,
                model,
                data,
            )

        failure_injector.check("perception")
        rgb, perceived_canister, perceived_holes, chosen_hole, target_center = estimate_perception(
            model, data, args.width, args.height, args.hole, args.place_z_offset
        )
        save_observation(Path(args.save_perception), rgb)
        print(f"[perception] canister centre: {np.array2string(perceived_canister.center, precision=4)} m")
        print(f"[perception] selected opening: {chosen_hole.name}")

        # Preserve the home tool orientation.  In the supplied scene this is
        # the downward-facing gripper orientation used for tabletop grasping.
        candidate_plans = choose_arm_and_plan(
            model, data, planning_data,
            perceived_canister.center, target_center, args.approach_height,
        )
        arm_name = candidate_plans.name
        site_id = get_name_id(model, mujoco.mjtObj.mjOBJ_SITE, ARM_SPECS[arm_name]["site"])
        tool_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        print(f"\n[plan] selected {arm_name} arm")

        # Open gripper before approach.
        set_gripper(model, data, arm_name, 0.0, ticks=220)

        pre_pick = perceived_canister.center + np.array([0.0, 0.0, args.approach_height])
        pick = perceived_canister.center
        lift = perceived_canister.center + np.array([0.0, 0.0, args.approach_height])
        pre_place = target_center + np.array([0.0, 0.0, args.approach_height])

        failure_injector.check("pre_grasp")
        move_to_pose(model, data, planning_data, arm_name, pre_pick, tool_rot, "pre-pick")
        move_to_pose(model, data, planning_data, arm_name, pick, tool_rot, "grasp")

        # Close around the canister body and let the contacts settle.
        set_gripper(model, data, arm_name, 220.0, ticks=450)
        mujoco.mj_forward(model, data)
        before_grasp, _ = object_pose(model, data, CANISTER_BODY)
        failure_injector.check("grasp")
        # Attach the free canister pose to the gripper mount for this
        # kinematic-only demonstration. This is not a physical grasp test.
        attachment_active = activate_kinematic_attachment(
            model, data, arm_name, object_body_name=CANISTER_BODY
        )
        if not attachment_active:
            raise RuntimeError("Kinematic demo attachment could not be activated")

        failure_injector.check("lift")
        move_to_pose(model, data, planning_data, arm_name, lift, tool_rot, "lift")
        after_lift, _ = object_pose(model, data, CANISTER_BODY)
        if np.linalg.norm(after_lift - before_grasp) < 0.025:
            raise RuntimeError(
                "Grasp verification failed: canister did not rise with the gripper. "
                "Inspect gripper contact, canister pose, and the simulated weld setting."
            )
        print(f"[verification] canister lifted by {np.linalg.norm(after_lift-before_grasp):.3f} m")

        # Preserve the grasp orientation for the entire transfer.  The arm is
        # moved kinematically by move_to_joint_target in this demo mode, so no
        # artificial actuator-force escalation is required.
        carry_rot = tool_rot.copy()

        # Transfer the loaded canister in short Cartesian increments instead
        # of asking the wrist to track one large cross-table joint-space move.
        # Each segment is solved from the measured live state, allowing the
        # controller to settle before proceeding to the next segment.
        transfer_start = data.site_xpos[site_id].copy()
        transfer_end = np.asarray(pre_place, dtype=float)
        transfer_distance = float(np.linalg.norm(transfer_end - transfer_start))
        transfer_step = 0.045  # maximum Cartesian segment length: 45 mm
        transfer_segments = max(2, int(math.ceil(transfer_distance / transfer_step)))
        print(
            f"[transfer] staged loaded path: {transfer_segments} segments, "
            f"distance={transfer_distance:.3f} m, step<={transfer_step:.3f} m"
        )
        for segment in range(1, transfer_segments + 1):
            fraction = segment / transfer_segments
            transfer_target = (
                transfer_start + fraction * (transfer_end - transfer_start)
            )
            move_to_pose(
                model, data, planning_data, arm_name,
                transfer_target, carry_rot,
                f"loaded-transfer-{segment}/{transfer_segments}"
            )
            # Inject after one completed loaded-transfer segment so GBB has
            # a genuine previous safe waypoint to return to.
            if segment == 1:
                failure_injector.check("transfer")

        failure_injector.check("insertion")
        move_to_pose(model, data, planning_data, arm_name, target_center, carry_rot, "insert")

        # Placement-stage recovery is injected after reaching the insertion pose
        # but BEFORE releasing the canister. The pre-place pose is the saved
        # safe waypoint to which GOB can return while retaining the attachment.
        placement_safe_waypoint = pre_place.copy()
        failure_injector.check("placement")

        # Release the attachment first, open the physical gripper, and allow the
        # canister to settle into the actual ring-defined opening.
        deactivate_kinematic_attachment()
        set_gripper(model, data, arm_name, 0.0, ticks=450)
        # Retract vertically after release so the fingers do not drag the
        # canister or obstruct its settling into the ring-defined opening.
        move_to_pose(
            model, data, planning_data, arm_name,
            target_center + np.array([0.0, 0.0, args.approach_height]),
            carry_rot, "post-release retreat"
        )
        steps = max(1, int(args.settle_seconds / model.opt.timestep))
        for i in range(steps):
            mujoco.mj_step(model, data)
            if viewer is not None and i % 100 == 0:
                viewer.sync()

        final_pos, _ = object_pose(model, data, CANISTER_BODY)
        xy_error = float(np.linalg.norm(final_pos[:2] - target_center[:2]))
        z_error = abs(float(final_pos[2] - target_center[2]))
        print("\n[result]")
        print(f"  canister final centre : {np.array2string(final_pos, precision=4)} m")
        print(f"  target centre         : {np.array2string(target_center, precision=4)} m")
        print(f"  XY centre error       : {xy_error*1000:.1f} mm")
        print(f"  Z centre error        : {z_error*1000:.1f} mm")

        # Placement verification tolerances intentionally allow small contact
        # settling, but reject a clear drop outside the chosen hole.
        if xy_error > 0.018 or z_error > 0.030:
            raise RuntimeError(
                "Placement verification failed: canister did not settle near the perceived hole. "
                f"XY error={xy_error:.3f} m, Z error={z_error:.3f} m."
            )

        print("[success] canister is placed within the selected tray opening tolerance")

        # Return the selected (left) arm to the exact joint configuration used
        # at startup. The canister has already been released and verified.
        home_joint_names = ARM_SPECS[ARM_TO_USE]["joints"]
        home_q = np.array([INITIAL_POSE[name] for name in home_joint_names], dtype=float)
        print(f"[home] returning {ARM_TO_USE} arm to its original joint position")
        move_to_joint_target(
            model, data, ARM_TO_USE, home_q,
            settle_ticks=60, final_ticks=JOINT_FINAL_TICKS
        )
        home_error = float(np.max(np.abs(
            data.qpos[arm_joint_indices(model, ARM_TO_USE)[0]] - home_q
        )))
        print(f"[home] maximum joint position error: {home_error:.6f} rad")
        if home_error > 1e-6:
            raise RuntimeError(
                f"{ARM_TO_USE} arm did not reach its original joint position "
                f"(max error {home_error:.6f} rad)"
            )


        # Keep the passive MuJoCo viewer alive after the task so the user can
        # inspect the final placement. The simulation is intentionally left
        # settled (no further mj_step calls); closing the viewer ends the hold.
        if viewer is not None:
            print("[viewer] task complete; left arm is home and the simulation is paused.")
            print("[viewer] close the MuJoCo window or press Ctrl+C here to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except InjectedFailure as failure:
        print(f"\\n[failure] Simulation interrupted: {failure}")
        outcome = process_injected_failure(failure, manager=recovery_manager, model=model, data=data, mujoco_module=mujoco)

        if failure.stage == "pre_grasp":
            print("[recovery-execution] attempting validated INA recovery")
            execute_recovery_via_backend(
                failure, outcome, model, data,
                planning_data=planning_data,
                arm_name=arm_name,
                pre_grasp_pos=pre_pick,
                tool_rot=tool_rot,
            )
        elif failure.stage == "grasp":
            print("[recovery-execution] attempting validated GOB recovery")
            execute_recovery_via_backend(
                failure, outcome, model, data,
                planning_data=planning_data,
                arm_name=arm_name,
                pre_grasp_pos=pre_pick,
                tool_rot=tool_rot,
            )
        elif failure.stage == "transfer":
            print("[recovery-execution] attempting validated GBB recovery")
            execute_recovery_via_backend(
                failure, outcome, model, data,
                planning_data=planning_data,
                arm_name=arm_name,
                last_safe_waypoint=transfer_start,
                carry_rot=carry_rot,
            )
        elif failure.stage == "lift":
            print("[recovery-execution] attempting validated ESS controlled stop")
            execute_recovery_via_backend(
                failure, outcome, model, data,
                arm_name=arm_name,
            )
        elif failure.stage == "insertion":
            print("[recovery-execution] attempting validated INA insertion recovery")
            execute_recovery_via_backend(
                failure, outcome, model, data,
                planning_data=planning_data,
                arm_name=arm_name,
                target_center=target_center,
                carry_rot=carry_rot,
                approach_height=args.approach_height,
            )
        elif failure.stage == "placement":
            print("[recovery-execution] attempting validated GOB placement recovery")
            execute_recovery_via_backend(
                failure, outcome, model, data,
                planning_data=planning_data,
                arm_name=arm_name,
                safe_placement_waypoint=placement_safe_waypoint,
                carry_rot=carry_rot,
            )
        else:
            # Perception/HIC is an explicit handoff, not an autonomous motion
            # command. Unknown stages also remain non-executable.
            print(
                "[recovery-execution] autonomous execution disabled for "
                f"failure stage '{failure.stage}'."
            )

        # Unlike the normal-success path, the exception path previously
        # closed the passive viewer immediately after recovery. Keep the
        # recovered pose on screen until the user closes the window.
        hold_viewer_after_recovery(viewer)
    finally:
        ACTIVE_VIEWER = None
        if viewer is not None:
            try:
                if viewer.is_running():
                    viewer.close()
            except Exception as exc:
                print(f"[viewer] shutdown warning: {exc}")


if __name__ == "__main__":
    main()
