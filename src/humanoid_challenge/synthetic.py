"""
Render synthetic views of the marker board, so the geometry can be tested before any real
video actually exists.

The idea is to draw each marker as a white tile (marker + quiet zone) warped onto the image
based on the known relationship between its table-frame corners and their projection.

That is enough to exercise detection and pose solving end to end.

Here I don't model lighting, blur or lens distortion, which is a separate problem.
"""

from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np

from .markers import Layout, marker_object_points


def camera_matrix(width: int, height: int, focal_px: float) -> np.ndarray:
    return np.array([[focal_px, 0, width / 2.0],
                     [0, focal_px, height / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Camera at ``eye`` looking at ``target``, in OpenCV convention (+z forward,
    +x right, +y down). Returns (rvec, tvec) taking table points into the camera."""
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:
        # Straight-down view: the default up is parallel to the view axis and the
        # cross product degenerates. Any perpendicular up gives a valid camera.
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)                      # +y points "down" in image space
    R = np.stack([r, d, f])                 # world -> camera
    t = -R @ eye
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


def render_board(layout: Layout, marker_length: float, K: np.ndarray,
                 rvec: np.ndarray, tvec: np.ndarray, width: int, height: int,
                 quiet_ratio: float = 0.25, dictionary_name: str = "DICT_4X4_50",
                 background: int = 235) -> np.ndarray:
    d = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    canvas = np.full((height, width), background, np.uint8)
    tile = marker_length * (1.0 + 2.0 * quiet_ratio)
    px = 480
    pad = int(round(px * quiet_ratio))

    for mid, (x, y, yaw) in sorted(layout.items()):
        img = aruco.generateImageMarker(d, mid, px, borderBits=1)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        n = img.shape[0]

        obj = marker_object_points(x, y, yaw, tile)       # corners of the FULL tile
        proj, _ = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, np.zeros(5))
        dst = proj.reshape(-1, 2).astype(np.float32)
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, H, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), H, (width, height),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        canvas[mask > 0] = warped[mask > 0]
    return canvas
