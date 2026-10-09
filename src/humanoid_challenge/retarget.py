"""Capture -> end-effector path in the table frame.

Per frame, we use hand landmarks from RGB, thumb and index tips lifted through the LiDAR depth,
and then taken into the table frame.

The important position is the thumb/index midpoint (the Panda's end-effector
site sits between its fingers; the wrist is ~10 cm behind it). Orientation is top-down
with the hand's yaw. The gripper command is the thumb-index distance thresholded with
hysteresis, since LIBERO's gripper channel is binary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import medfilt, savgol_filter

from .geometry import apply, sample_depth, unproject
from .hands import INDEX_TIP, THUMB_TIP, WRIST

# A top-down gripper in the table frame: approach along -z, fingers in the table plane.
TOPDOWN = np.array([[0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0]])


@dataclass
class RetargetResult:
    t: np.ndarray
    eef_pos: np.ndarray                 # (T, 3) table frame, metres
    eef_rotmat: np.ndarray              # (T, 3, 3)
    gripper: np.ndarray                 # (T,) int8, -1 open / +1 closed
    pinch_mm: np.ndarray                # (T,) raw thumb-index distance
    hand_conf: np.ndarray               # (T,)
    valid: np.ndarray                   # (T,) bool: hand seen AND depth available
    notes: list[str] = field(default_factory=list)


def _yaw_rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ TOPDOWN


def lift_point(uv, depth_map, K_rgb, K_depth, T_table_cam):
    d = sample_depth(depth_map, uv, K_rgb, K_depth)
    if not np.isfinite(d):
        return None
    return apply(T_table_cam, unproject(uv, d, K_rgb))


def retarget(cap, traj, hand_frames, *, pinch_closed_mm: float, pinch_open_mm: float,
             yaw_only: bool = True, stride: int = 1,
             smooth_median: int = 5, smooth_window: int = 31, smooth_poly: int = 2,
             min_state_frames: int = 8) -> RetargetResult:
    """`hand_frames` is a list of HandFrame, one per capture frame (stride applied)."""
    K_rgb, K_depth = cap.K, cap.K_depth()
    idx = list(range(0, cap.n_frames, stride))
    n = len(idx)
    pos = np.full((n, 3), np.nan)
    rot = np.tile(TOPDOWN, (n, 1, 1)).astype(float)
    pinch = np.full(n, np.nan)
    conf = np.zeros(n)
    valid = np.zeros(n, bool)
    notes: list[str] = []

    for k, i in enumerate(idx):
        hf = hand_frames[k]
        if not hf.ok:
            continue
        conf[k] = hf.score
        depth = cap.depth(i)
        T = traj.T_table_cam[i]
        p_thumb = lift_point(hf.landmarks_px[THUMB_TIP], depth, K_rgb, K_depth, T)
        p_index = lift_point(hf.landmarks_px[INDEX_TIP], depth, K_rgb, K_depth, T)
        if p_thumb is None or p_index is None:
            continue
        pos[k] = (p_thumb + p_index) / 2.0
        pinch[k] = np.linalg.norm(p_thumb - p_index) * 1000.0
        valid[k] = True

        if yaw_only:
            p_wrist = lift_point(hf.landmarks_px[WRIST], depth, K_rgb, K_depth, T)
            if p_wrist is not None:
                v = pos[k] - p_wrist
                if np.hypot(v[0], v[1]) > 1e-3:
                    rot[k] = _yaw_rotation(np.arctan2(v[1], v[0]))
        else:
            notes.append("full 6-DoF hand orientation is not implemented; used top-down")

    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f"only {n_valid} frames have both a hand and valid depth; "
                         "nothing can be retargeted from this clip")
    if n_valid < 0.5 * n:
        notes.append(f"only {n_valid}/{n} frames usable -- check hand visibility and "
                     "that the LiDAR sees the fingers")

    # Interpolate across short gaps; the controller needs a continuous path.
    good = np.flatnonzero(valid)
    for c in range(3):
        pos[:, c] = np.interp(np.arange(n), good, pos[good, c])
    pinch = np.interp(np.arange(n), good, pinch[good])
    for k in range(n):
        if not valid[k]:
            rot[k] = rot[good[np.argmin(np.abs(good - k))]]

    # The raw lifted path is very jittery (4.41 m of path for 0.24 m of net motion on a real
    # capture). A median filter removes the isolated depth spikes, then Savitzky-Golay
    # smooths the rest while keeping the lift and touch-down peaks.
    if n >= smooth_window:
        for c in range(3):
            pos[:, c] = medfilt(pos[:, c], smooth_median)
        pos = savgol_filter(pos, smooth_window, smooth_poly, axis=0)
        pinch = savgol_filter(medfilt(pinch, smooth_median), smooth_window, smooth_poly)

    # Hysteresis, not a bare threshold: a single noisy frame near the boundary must not
    # flip the gripper, because a spurious open mid-transport drops the object.
    grip = np.full(n, -1, np.int8)
    state, held = -1, 0
    for k in range(n):
        want = state
        if state < 0 and pinch[k] < pinch_closed_mm:
            want = 1
        elif state > 0 and pinch[k] > pinch_open_mm:
            want = -1

        # A new state must hold for a few frames before it counts: fingers close partway
        # while reaching, and that should not register as a grasp.
        if want != state:
            held += 1
            if held >= min_state_frames:
                state, held = want, 0
        else:
            held = 0
        grip[k] = state

    return RetargetResult(cap.timestamps[idx], pos, rot, grip, pinch, conf, valid, notes)
