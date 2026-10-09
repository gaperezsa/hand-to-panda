"""
In order to do the retargeting, unprojection and frame fitting are needed.

Kept separate from the capture and marker code because these are the pieces that are
pure math.
"""
from __future__ import annotations

import numpy as np


def unproject(uv: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    """Pixel + depth-along-camera-z -> 3D point in the camera frame (metres)."""
    u, v = float(uv[0]), float(uv[1])
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.array([x, y, depth], dtype=np.float64)


def sample_depth(depth_map: np.ndarray, uv: np.ndarray, K_rgb: np.ndarray,
                 K_depth: np.ndarray, radius: int = 3) -> float:
    """Depth at an RGB pixel, read from the lower-resolution depth map from the iPhone LiDAR.

    Takes the near quartile of a small patch rather than the single nearest pixel. The LiDAR map
    is 256x192 and a fingertip is a couple of pixels across, so single-pixel reads land on
    holes or on the edge of the hand where depth jumps to the table behind it.

    Returns NaN if the patch has no valid depth at all.
    """
    h, w = depth_map.shape
    su = K_depth[0, 0] / K_rgb[0, 0]
    sv = K_depth[1, 1] / K_rgb[1, 1]
    du = (float(uv[0]) - K_rgb[0, 2]) * su + K_depth[0, 2]
    dv = (float(uv[1]) - K_rgb[1, 2]) * sv + K_depth[1, 2]
    u0, v0 = int(round(du)), int(round(dv))
    if not (0 <= u0 < w and 0 <= v0 < h):
        return float("nan")
    patch = depth_map[max(0, v0 - radius):v0 + radius + 1,
                      max(0, u0 - radius):u0 + radius + 1]
    vals = patch[np.isfinite(patch)]
    if vals.size == 0:
        return float("nan")
    # Nearest surface, not the median: a patch straddling the finger edge contains both
    # the finger and the table behind it, and the finger is what we want.
    return float(np.percentile(vals, 25))


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to a point or an (N,3) array of points."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


def rigid_fit(src: np.ndarray, dst: np.ndarray, allow_scale: bool = False):
    """
    Through a least-squares rigid (or similarity) transform we take `src` points onto `dst`.

    Used to express the measured marker board, which the calibration anchors at ID 0, in
    LIBERO's table frame, which is defined by where the markers are supposed to be.

    Four marker centres give an over-determined fit, so taping error shows up as a residual
    instead of silently shifting every coordinate.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    cs, cd = src.mean(0), dst.mean(0)
    S, D = src - cs, dst - cd
    U, sv, Vt = np.linalg.svd(S.T @ D)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0] * (src.shape[1] - 1) + [d]) @ U.T
    s = (sv[:-1].sum() + d * sv[-1]) / (S ** 2).sum() if allow_scale else 1.0

    # Coplanar points (the marker board) leave the out-of-plane direction unconstrained,
    # and the fit can come out mirrored, with everything above the table mapped below it.
    # In that case build the rotation from the two plane normals instead.
    if src.shape[1] == 3 and sv[-1] <= 1e-8 * max(sv[0], 1e-12):
        def plane_normal(X):
            n = np.linalg.svd(X)[2][-1]
            # SVD returns the normal up to sign. For points lying exactly in a plane
            # that sign is arbitrary, so orient it deterministically -- otherwise the
            # "is the normal preserved" test below compares against a coin flip.
            for k in (2, 1, 0):
                if abs(n[k]) > 1e-9:
                    return n if n[k] > 0 else -n
            return n
        n_s, n_d = plane_normal(S), plane_normal(D)
        if (R @ n_s) @ n_d < 0:
            def basis(n):
                a = np.array([1.0, 0.0, 0.0])
                if abs(a @ n) > 0.9:
                    a = np.array([0.0, 1.0, 0.0])
                e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
                return np.stack([e1, np.cross(n, e1), n], axis=1)
            Bs, Bd = basis(n_s), basis(n_d)
            # in-plane rotation from the 2-D Procrustes of the projected points
            P, Q = S @ Bs[:, :2], D @ Bd[:, :2]
            u2, _, vt2 = np.linalg.svd(P.T @ Q)
            r2 = vt2.T @ np.diag([1.0, np.sign(np.linalg.det(vt2.T @ u2.T))]) @ u2.T
            R3 = np.eye(3)
            R3[:2, :2] = r2
            R = Bd @ R3 @ Bs.T
    t = cd - s * R @ cs
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    resid = np.linalg.norm(apply(T, src) - dst, axis=1)
    return T, float(resid.max()), float(resid.mean())
