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
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mujoco
    try:
        import mujoco.viewer  # viewer is an optional submodule in some installs
    except Exception:
        pass
except ImportError as exc:
    raise SystemExit(
        "MuJoCo's Python package is not installed in this Python environment.\n"
        "Activate the repository's MuJoCo venv first, as described in README.md.\n"
        "Example: source python/src/robauditor/src/tracy_ws/.venv_mujoco/bin/activate"
    ) from exc


# Names in the supplied Tracy perception runtime model.
CAMERA_NAME = "d435_camera"
CANISTER_BODY = "canister1"
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


def sync_viewer():
    """Refresh the passive viewer during control without affecting physics."""
    global ACTIVE_VIEWER
    if ACTIVE_VIEWER is not None:
        try:
            ACTIVE_VIEWER.sync()
        except Exception:
            ACTIVE_VIEWER = None


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

    keep_top_level_bodies = {"yellow_floor", "map", CANISTER_BODY, "Pump1", "Pump2"}
    removed = []
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") not in keep_top_level_bodies:
            removed.append(child.get("name", "<unnamed body>"))
            worldbody.remove(child)

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
    print("[scene] retained: Tracy robot + table/floor + canister1 + Pump1/Pump2")
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
    q_start = data.qpos[qpos_ids].copy()
    data.ctrl[actuator_ids] = q_target
    for point in interpolate_qpos(q_start, q_target, max_step=0.012):
        data.qpos[qpos_ids] = point
        mujoco.mj_forward(model, data)
        update_kinematic_attachment(model, data)
        mujoco.mj_forward(model, data)
        sync_viewer()
    data.qpos[qpos_ids] = q_target
    data.ctrl[actuator_ids] = q_target
    mujoco.mj_forward(model, data)
    update_kinematic_attachment(model, data)
    mujoco.mj_forward(model, data)


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


def set_gripper(model, data, arm_name: str, value: float, ticks: int = 350):
    actuator_id = get_name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_SPECS[arm_name]["gripper"])
    value = float(np.clip(value, 0.0, 255.0))
    qpos_ids, _, arm_actuator_ids = arm_joint_indices(model, arm_name)
    held_arm_qpos = data.qpos[qpos_ids].copy()
    for i in range(ticks):
        data.ctrl[actuator_id] = value
        data.ctrl[arm_actuator_ids] = held_arm_qpos
        mujoco.mj_step(model, data)
        # Keep the arm stationary while the gripper actuator closes/opens.
        data.qpos[qpos_ids] = held_arm_qpos
        mujoco.mj_forward(model, data)
        if i % 40 == 0:
            sync_viewer()



# In this kinematic demonstration the arm is moved by direct joint-coordinate
# updates. MuJoCo weld constraints are force constraints and do not automatically
# reposition a free body when qpos is edited followed only by mj_forward().
# Therefore the demo attachment explicitly carries the canister's free-joint
# pose with the gripper mount. This is a simulator-only kinematic grasp surrogate.
KINEMATIC_ATTACHMENT = None


def activate_kinematic_attachment(model, data, arm_name: str,
                                  object_body_name: str = CANISTER_BODY) -> bool:
    global KINEMATIC_ATTACHMENT
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
    print("[grasp] activated kinematic simulator attachment from measured grasp pose")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    model_path, meshdir = resolve_model_and_meshdir(args)
    model = load_model_with_meshdir(model_path, meshdir)
    data = mujoco.MjData(model)
    planning_data = mujoco.MjData(model)
    reset_to_initial_pose(model, data)

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

        move_to_pose(model, data, planning_data, arm_name, pre_pick, tool_rot, "pre-pick")
        move_to_pose(model, data, planning_data, arm_name, pick, tool_rot, "grasp")

        # Close around the canister body and let the contacts settle.
        set_gripper(model, data, arm_name, 220.0, ticks=450)
        mujoco.mj_forward(model, data)
        before_grasp, _ = object_pose(model, data, CANISTER_BODY)
        # Attach the free canister pose to the gripper mount for this
        # kinematic-only demonstration. This is not a physical grasp test.
        attachment_active = activate_kinematic_attachment(model, data, arm_name)
        if not attachment_active:
            raise RuntimeError("Kinematic demo attachment could not be activated")

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

        move_to_pose(model, data, planning_data, arm_name, target_center, carry_rot, "insert")

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
    finally:
        if viewer is not None:
            viewer.close()
        ACTIVE_VIEWER = None


if __name__ == "__main__":
    main()
