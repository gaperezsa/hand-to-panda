"""One drift-corrected camera trajectory per episode, in the table frame.

ARKit sets its world origin wherever the phone happened to be when recording started, so
raw ARKit poses are in a different arbitrary frame in every episode. The markers supply
the fixed table frame.

ARKit supplies continuity through the long stretches where the hands occlude everything,
especially the markers.

We solve the table frame from the markers wherever they are visible,
estimate the single rigid transform from ARKit's world to the table, and apply it
everywhere.

Because that transform is over-determined by every marker sighting, the spread across
sightings *is* the drift measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import rigid_fit


@dataclass
class Trajectory:
    T_table_cam: np.ndarray          # (T, 4, 4) camera pose in the table frame
    anchored: np.ndarray             # (T,) bool: was a marker actually solved here
    T_table_world: np.ndarray        # the fitted ARKit-world -> table transform
    drift_m: float                   # disagreement between first and last sighting
    drift_deg: float
    n_sightings: int
    fit_resid_m: float


def fuse(T_world_cam: np.ndarray, marker_poses: dict[int, np.ndarray]) -> Trajectory:
    """`marker_poses` maps frame index -> 4x4 camera-in-table from solve_board_pose."""
    if not marker_poses:
        raise ValueError("no marker sightings; the episode cannot be placed in the table "
                         "frame. Start each recording with a marker in view.")
    idx = sorted(marker_poses)

    # Each sighting proposes a complete transform T_table_world = T_table_cam @
    # inv(T_world_cam). Average those SE(3) estimates directly.
    # Each sighting already carries a full orientation, so we use it.
    props = [marker_poses[i] @ np.linalg.inv(T_world_cam[i]) for i in idx]
    t_all = np.stack([P[:3, 3] for P in props])
    R_all = np.stack([P[:3, :3] for P in props])
    t_med = np.median(t_all, axis=0)
    U, _, Vt = np.linalg.svd(R_all.mean(axis=0))          # chordal mean rotation
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    T_tw = np.eye(4)
    T_tw[:3, :3], T_tw[:3, 3] = R_mean, t_med
    # spread of the individual proposals about the average, as a position residual
    resid_max = float(np.linalg.norm(t_all - t_med, axis=1).max())

    T_table_cam = np.einsum("ij,njk->nik", T_tw, T_world_cam)

    # Drift: how far the ARKit-propagated pose has slipped from the marker solution by
    # the last sighting. This is the number that goes in the README.
    d = np.linalg.inv(marker_poses[idx[-1]]) @ T_table_cam[idx[-1]]
    drift_m = float(np.linalg.norm(d[:3, 3]))
    drift_deg = float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))

    anchored = np.zeros(len(T_world_cam), bool)
    anchored[idx] = True
    return Trajectory(T_table_cam, anchored, T_tw, drift_m, drift_deg, len(idx), resid_max)
