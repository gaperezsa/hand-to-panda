"""
From the recording, we are able to extract calibrated end-effector poses and gripper states
at 30 Hz.

This module turns an end-effector path into LIBERO actions.

robosuite's OSC is an impedance controller, so handing it absolute targets makes the arm
trail them by 20-110 mm (measured, 0/5 successes).

Therefore, each step instead commands the delta from the measured pose to the next waypoint,
in relative mode, saturated to the controller's range.
"""
from __future__ import annotations

import cv2
import numpy as np

POS_SCALE = 0.05   # robosuite OSC_POSE output_max for position, metres per unit action
ROT_SCALE = 0.5    # ditto for orientation, radians per unit action


def axisangle_to_matrix(v: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))[0]


def matrix_to_axisangle(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


# robosuite has two end-effector orientation frames 90 degrees apart about the body z
# axis: robot.controller.ee_ori_mat and quat2mat(obs["robot0_eef_quat"]). Targets are
# expressed in the second (it is what observation.state carries), so `cur_rotmat` must be
# the second as well, or the gripper tracks an orientation 90 degrees off and closes on air.


def track_step(cur_pos, cur_rotmat, tgt_pos, tgt_rotmat, gripper,
               kp: float = 6.0, kr: float = 3.0) -> np.ndarray:
    """One relative 7-D action toward a target pose.

    The OSC realises only ~15% of a commanded delta per step (6.8 mm achieved for up to
    47 mm commanded), hence gains well above one. `cur_rotmat` is the observation-frame
    orientation, see above. `gripper` passes through: it is binary in LIBERO.
    """
    dpos = kp * (np.asarray(tgt_pos, float) - np.asarray(cur_pos, float)) / POS_SCALE

    # robosuite premultiplies the current orientation by the delta, so the error is
    # R_target @ R_cur^T.
    drot = kr * matrix_to_axisangle(np.asarray(tgt_rotmat) @ np.asarray(cur_rotmat).T) / ROT_SCALE

    return np.clip(np.concatenate([dpos, drot, [gripper]]), -1.0, 1.0)


def place_correction(path, grasp_idx, release_idx, object_xy, target_xy):
    """Shift the carry so the object, not the hand, lands on the target.

    While held, the object sits at a fixed offset from the gripper, so the release point
    must be target + (grasp_point - object_start). Replayed verbatim the bowl lands
    ~55 mm from the plate. The shift ramps in between grasp and release so the grasp
    itself is untouched.
    """
    p = np.asarray(path, dtype=float).copy()
    grasp_xy = p[grasp_idx, :2]
    held_offset = grasp_xy - np.asarray(object_xy, float)
    want_release_xy = np.asarray(target_xy, float) + held_offset
    delta = want_release_xy - p[release_idx, :2]

    n = len(p)
    ramp = np.zeros(n)
    if release_idx > grasp_idx:
        ramp[grasp_idx:release_idx + 1] = np.linspace(0.0, 1.0, release_idx - grasp_idx + 1)
    ramp[release_idx + 1:] = 1.0
    p[:, 0] += ramp * delta[0]
    p[:, 1] += ramp * delta[1]
    return p, float(np.linalg.norm(delta))


def anchor_grasp(path, grasp_idx, anchor_xyz, clearance=0.03, descent=5):
    """Move the grasp waypoint onto the robot's own grasp pose on the rim.

    The one thing a capture establishes for certain is that the hand was on the rim
    when the pinch closed; where the path says that happened carries the session's frame
    error and the ~22 mm fingertip-landmark offset.

    In one of the experiments, we simply snap to the position used by LIBERO demonstrations
    (47 mm from the bowl centre along the jaw axis, 38 mm up). This cancels both the frame
    error and the offset.

    The shift ramps in over the approach and is held afterwards, and the
    approach is kept `clearance` above the grasp height until the final `descent`
    waypoints, as the demonstrations do.
    """
    p = np.asarray(path, dtype=float).copy()
    delta = np.asarray(anchor_xyz, float) - p[grasp_idx, :3]
    ramp = np.ones(len(p))
    if grasp_idx > 0:
        ramp[:grasp_idx + 1] = np.linspace(0.0, 1.0, grasp_idx + 1)
    p[:, :3] += ramp[:, None] * delta
    z_grasp = float(anchor_xyz[2])
    k0 = max(0, grasp_idx - descent)
    p[:k0, 2] = np.maximum(p[:k0, 2], z_grasp + clearance)
    for j in range(k0, grasp_idx):
        f = (j - k0) / max(1, grasp_idx - k0)
        p[j, 2] = max(p[j, 2], z_grasp + clearance * (1.0 - f))
    return p, delta
