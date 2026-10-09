
"""
Here we aim to detect a board of ArUco markers on the table, and to estimate the camera pose
relative to that board.

The board is defined in a fixed "table frame" coordinate system, where the x-axis points toward
the operator (depth of the table), the y-axis points to the operator's right (the robot's left),
and the z-axis points up, out of the table surface. Markers lie flat in the z=0 plane.

For the processed dataset, I put the markers in a 2x2 grid. Each marker's corners are ordered
the way cv2.aruco returns them -- top-left, top-right, bottom-right, bottom-left as seen from
+z looking down.

Since OpenCV 5.0 removed aruco.estimatePoseSingleMarkers and aruco.estimatePoseBoard, the pose
is extracted using Board.matchImagePoints + cv2.solvePnP. This is the correct route for a
multi-marker board anyway: every visible marker contributes correspondences to one solve,
rather than averaging independent per-marker poses.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import cv2.aruco as aruco
import numpy as np

Layout = dict[int, tuple[float, float, float]]  # id -> (x, y, yaw) metres, radians


def marker_object_points(x: float, y: float, yaw: float, marker_length: float) -> np.ndarray:
    """The four corners of one flat marker, in table-frame metres, in cv2.aruco order."""
    h = marker_length / 2.0
    local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    xy = local @ rot.T + np.array([x, y])
    return np.column_stack([xy, np.zeros(4)]).astype(np.float32)


def make_board(layout: Layout, marker_length: float,
               dictionary_name: str = "DICT_4X4_50") -> aruco.Board:
    ids = sorted(layout)
    obj = [marker_object_points(*layout[i], marker_length) for i in ids]
    d = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    return aruco.Board(obj, d, np.array(ids, dtype=np.int32).reshape(-1, 1))


def make_detector(dictionary_name: str = "DICT_4X4_50") -> aruco.ArucoDetector:
    d = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    params = aruco.DetectorParameters()
    # Head-mounted capture is motion-blurred and often oblique. Subpixel refinement is
    # what buys the sub-millimetre corner accuracy the whole metric pipeline rests on.
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(d, params)


@dataclass
class BoardPose:
    """Result of one solve. ``ok`` false means too few markers were seen."""
    ok: bool
    n_markers: int
    ids: list[int]
    rvec: np.ndarray | None = None          # table -> camera, Rodrigues
    tvec: np.ndarray | None = None          # table -> camera, metres
    reproj_px: float = float("nan")         # RMS reprojection error

    @property
    def T_cam_table(self) -> np.ndarray:
        """4x4 taking table-frame points into the camera frame."""
        R, _ = cv2.Rodrigues(self.rvec)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, self.tvec.ravel()
        return T

    @property
    def T_table_cam(self) -> np.ndarray:
        """4x4 taking camera-frame points into the table frame (i.e. the camera pose)."""
        return np.linalg.inv(self.T_cam_table)


def solve_board_pose(board: aruco.Board, corners, ids,
                     K: np.ndarray, dist: np.ndarray | None = None) -> BoardPose:
    """Camera pose from whatever subset of the board is visible."""
    dist = np.zeros(5) if dist is None else dist
    if ids is None or len(ids) == 0:
        return BoardPose(ok=False, n_markers=0, ids=[])

    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return BoardPose(ok=False, n_markers=0, ids=[int(i) for i in np.ravel(ids)])

    # IPPE is the right solver for coplanar points and is far more stable than the
    # iterative default when only one marker is visible.
    flags = cv2.SOLVEPNP_IPPE if len(obj_pts) >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags)
    if not ok:
        return BoardPose(ok=False, n_markers=0, ids=[int(i) for i in np.ravel(ids)])

    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2).sum(axis=1).mean()))
    return BoardPose(True, len(np.ravel(ids)), [int(i) for i in np.ravel(ids)], rvec, tvec, rms)
